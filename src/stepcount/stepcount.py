import warnings
import os
import pathlib
import urllib
import shutil
import time
import argparse
import json
import re
import numpy as np
import pandas as pd
import joblib

from stepcount import utils
from stepcount import __version__
from stepcount import __model_version__
from stepcount import __model_md5__

warnings.filterwarnings('ignore', message='Mean of empty slice')  # shut .median() warning when all-NaN



def main():

    parser = argparse.ArgumentParser(
        description="A tool to estimate step counts from accelerometer data",
        add_help=True
    )
    parser.add_argument("filepath", help="Enter file to be processed")
    parser.add_argument("--outdir", "-o", help="Enter folder location to save output files", default="outputs/")
    parser.add_argument("--model-path", "-m", help="Enter custom model file to use", default=None)
    parser.add_argument("--force-download", action="store_true", help="Force download of model file")
    parser.add_argument('--model-type', '-t',
                        help='Enter model type to run (Self-Supervised Learning model or Random Forest)',
                        choices=['ssl', 'rf'], default='ssl')
    parser.add_argument("--pytorch-device", "-d", help="Pytorch device to use, e.g.: 'cpu' or 'cuda:0' (for SSL only)",
                        type=str, default='cpu')
    parser.add_argument("--sample-rate", "-r", help="Sample rate for measurement, otherwise inferred.",
                        type=int, default=None)
    parser.add_argument("--txyz",
                        help=("Use this option to specify the column names for time, x, y, z "
                              "in the input file, in that order. Use a comma-separated string. "
                              "Default: 'time,x,y,z'"),
                        type=str, default="time,x,y,z")
    parser.add_argument("--exclude-wear-below", "-w",
                        help=("Minimum wear time for a day to be considered valid, otherwise exclude it. "
                              "Pass values as strings, e.g.: '12H', '30min'. Default: None (no exclusion)"),
                        type=str, default=None)
    parser.add_argument("--exclude-first-last", "-e",
                        help="Exclude first, last or both days of data. Default: None (no exclusion)",
                        type=str, choices=['first', 'last', 'both'], default=None)
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    args = parser.parse_args()

    before = time.time()

    verbose = not args.quiet

    # Output paths
    basename = utils.resolve_path(args.filepath)[1]
    outdir = os.path.join(args.outdir, basename)
    os.makedirs(outdir, exist_ok=True)

    # Info.json contains high-level summary of the data and results
    info = {}
    info['StepCountVersion'] = __version__
    info['StepCountArgs'] = vars(args)

    # Load file
    data, info_read = utils.read(
        args.filepath, 
        usecols=args.txyz, 
        resample_hz=30 if args.model_type == 'ssl' else None,
        sample_rate=args.sample_rate, 
        verbose=verbose
    )
    info.update(info_read)

    # Exclusion: first/last days
    if args.exclude_first_last is not None:
        data = utils.exclude_first_last_days(data, args.exclude_first_last)

    # Exclusion: days with wear time below threshold
    if args.exclude_wear_below is not None:
        data = utils.exclude_wear_below_days(data, args.exclude_wear_below)

    # If no data left, save info and exit early
    if len(data) == 0 or data.isna().all().all():
        print("No data left after exclusion. Exiting early...")
        # Reset start, end, wear time, etc.
        info['StartTime'] = None
        info['EndTime'] = None
        info['WearTime(days)'] = None
        info['NonwearTime(days)'] = None
        with open(f"{outdir}/{basename}-Info.json", 'w') as f:
            json.dump(info, f, indent=4, cls=utils.NpEncoder)
        print("\nSummary\n-------")
        print(json.dumps(
            {k: v for k, v in info.items() if not re.search(r'_Weekend|_Weekday|_Hour\d{2}', k)},
            indent=4, cls=utils.NpEncoder
        ))
        print("\nOutput files saved in:", outdir)
        print(f"Done! ({round(time.time() - before,2)}s)")
        return  # exit early

    # Update start, end, wear time, etc. if exclusions applied
    if args.exclude_first_last is not None or args.exclude_wear_below is not None:
        dt = utils.infer_freq(data.index).total_seconds()
        nonwear_time = data.isna().any(axis=1).sum() * dt
        wear_time = len(data) * dt - nonwear_time
        info['StartTime'] = data.index[0].strftime("%Y-%m-%d %H:%M:%S")
        info['EndTime'] = data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        info['WearTime(days)'] = wear_time / (60 * 60 * 24)
        info['NonwearTime(days)'] = nonwear_time / (60 * 60 * 24)

    # Run model
    if verbose:
        print("Loading model...")
    model_path = pathlib.Path(__file__).parent / f"{__model_version__[args.model_type]}.joblib.lzma"
    check_md5 = args.model_path is None
    model = load_model(args.model_path or model_path, args.model_type, check_md5, args.force_download)
    # TODO: implement reset_sample_rate()
    model.sample_rate = info['ResampleRate']
    model.window_len = int(np.ceil(info['ResampleRate'] * model.window_sec))
    model.wd.sample_rate = info['ResampleRate']
    model.verbose = verbose
    model.wd.verbose = verbose

    model.wd.device = args.pytorch_device

    if verbose:
        print("Running step counter...")
    Y, W, T_steps = model.predict_from_frame(data)

    # Quality control: Wear time coverage
    coverage = Y.groupby(Y.index.hour).agg(lambda x: x.notna().mean())
    if len(coverage) < 24 or coverage.min() < 0.01:
        info['Covers24hOK'] = 0
    else:
        info['Covers24hOK'] = 1
    del coverage  # free memory

    # Save step counts
    Y.to_csv(f"{outdir}/{basename}-Steps.csv.gz")
    # Save timestamps of each step
    T_steps.to_csv(f"{outdir}/{basename}-StepTimes.csv.gz", index=False)

    # ENMO summary
    enmo_summary = summarize_enmo(data)
    info['ENMO(mg)'] = enmo_summary['avg']
    info['ENMO(mg)_Weekend'] = enmo_summary['weekend_avg']
    info['ENMO(mg)_Weekday'] = enmo_summary['weekday_avg']
    info.update({f'ENMO(mg)_Hour{h:02}': enmo_summary['hour_avgs'].loc[h] for h in range(24)})
    info.update({f'ENMO(mg)_Hour{h:02}_Weekend': enmo_summary['weekend_hour_avgs'].loc[h] for h in range(24)})
    info.update({f'ENMO(mg)_Hour{h:02}_Weekday': enmo_summary['weekday_hour_avgs'].loc[h] for h in range(24)})

    # ENMO summary, adjusted
    enmo_summary_adj = summarize_enmo(data, adjust_estimates=True)
    info['ENMOAdjusted(mg)'] = enmo_summary_adj['avg']
    info['ENMOAdjusted(mg)_Weekend'] = enmo_summary_adj['weekend_avg']
    info['ENMOAdjusted(mg)_Weekday'] = enmo_summary_adj['weekday_avg']
    info.update({f'ENMOAdjusted(mg)_Hour{h:02}': enmo_summary_adj['hour_avgs'].loc[h] for h in range(24)})
    info.update({f'ENMOAdjusted(mg)_Hour{h:02}_Weekend': enmo_summary_adj['weekend_hour_avgs'].loc[h] for h in range(24)})
    info.update({f'ENMOAdjusted(mg)_Hour{h:02}_Weekday': enmo_summary_adj['weekday_hour_avgs'].loc[h] for h in range(24)})

    # Steps summary
    steps_summary = summarize_steps(Y, model.steptol)
    # steps, overall stats
    info['TotalSteps'] = steps_summary['total_steps']
    info['StepsDayAvg'] = steps_summary['avg_steps']
    info['StepsDayMed'] = steps_summary['med_steps']
    info['StepsDayMin'] = steps_summary['min_steps']
    info['StepsDayMax'] = steps_summary['max_steps']
    # steps, weekend stats
    info['TotalSteps_Weekend'] = steps_summary['weekend_total_steps']
    info['StepsDayAvg_Weekend'] = steps_summary['weekend_avg_steps']
    info['StepsDayMed_Weekend'] = steps_summary['weekend_med_steps']
    info['StepsDayMin_Weekend'] = steps_summary['weekend_min_steps']
    info['StepsDayMax_Weekend'] = steps_summary['weekend_max_steps']
    # steps, weekday stats
    info['TotalSteps_Weekday'] = steps_summary['weekday_total_steps']
    info['StepsDayAvg_Weekday'] = steps_summary['weekday_avg_steps']
    info['StepsDayMed_Weekday'] = steps_summary['weekday_med_steps']
    info['StepsDayMin_Weekday'] = steps_summary['weekday_min_steps']
    info['StepsDayMax_Weekday'] = steps_summary['weekday_max_steps']
    # walking, overall stats
    info['TotalWalking(mins)'] = steps_summary['total_walk']
    info['WalkingDayAvg(mins)'] = steps_summary['avg_walk']
    info['WalkingDayMed(mins)'] = steps_summary['med_walk']
    info['WalkingDayMin(mins)'] = steps_summary['min_walk']
    info['WalkingDayMax(mins)'] = steps_summary['max_walk']
    # walking, weekend stats
    info['TotalWalking(mins)_Weekend'] = steps_summary['weekend_total_walk']
    info['WalkingDayAvg(mins)_Weekend'] = steps_summary['weekend_avg_walk']
    info['WalkingDayMed(mins)_Weekend'] = steps_summary['weekend_med_walk']
    info['WalkingDayMin(mins)_Weekend'] = steps_summary['weekend_min_walk']
    info['WalkingDayMax(mins)_Weekend'] = steps_summary['weekend_max_walk']
    # walking, weekday stats
    info['TotalWalking(mins)_Weekday'] = steps_summary['weekday_total_walk']
    info['WalkingDayAvg(mins)_Weekday'] = steps_summary['weekday_avg_walk']
    info['WalkingDayMed(mins)_Weekday'] = steps_summary['weekday_med_walk']
    info['WalkingDayMin(mins)_Weekday'] = steps_summary['weekday_min_walk']
    info['WalkingDayMax(mins)_Weekday'] = steps_summary['weekday_max_walk']
    # time of accumulated steps
    info['Steps5thAt'] = steps_summary['ptile_at_avgs']['p05_at']
    info['Steps25thAt'] = steps_summary['ptile_at_avgs']['p25_at']
    info['Steps50thAt'] = steps_summary['ptile_at_avgs']['p50_at']
    info['Steps75thAt'] = steps_summary['ptile_at_avgs']['p75_at']
    info['Steps95thAt'] = steps_summary['ptile_at_avgs']['p95_at']
    # hour-of-day averages
    info.update({f'Steps_Hour{h:02}': steps_summary['hour_steps'].loc[h] for h in range(24)})
    info.update({f'Steps_Hour{h:02}_Weekend': steps_summary['weekend_hour_steps'].loc[h] for h in range(24)})
    info.update({f'Steps_Hour{h:02}_Weekday': steps_summary['weekday_hour_steps'].loc[h] for h in range(24)})
    info.update({f'Walking(mins)_Hour{h:02}': steps_summary['hour_walks'].loc[h] for h in range(24)})
    info.update({f'Walking(mins)_Hour{h:02}_Weekend': steps_summary['weekend_hour_walks'].loc[h] for h in range(24)})
    info.update({f'Walking(mins)_Hour{h:02}_Weekday': steps_summary['weekday_hour_walks'].loc[h] for h in range(24)})

    # Steps summary, adjusted
    steps_summary_adj = summarize_steps(Y, model.steptol, adjust_estimates=True)
    # steps, overall stats
    info['TotalStepsAdjusted'] = steps_summary_adj['total_steps']
    info['StepsDayAvgAdjusted'] = steps_summary_adj['avg_steps']
    info['StepsDayMedAdjusted'] = steps_summary_adj['med_steps']
    info['StepsDayMinAdjusted'] = steps_summary_adj['min_steps']
    info['StepsDayMaxAdjusted'] = steps_summary_adj['max_steps']
    # steps, weekend stats
    info['TotalStepsAdjusted_Weekend'] = steps_summary_adj['weekend_total_steps']
    info['StepsDayAvgAdjusted_Weekend'] = steps_summary_adj['weekend_avg_steps']
    info['StepsDayMedAdjusted_Weekend'] = steps_summary_adj['weekend_med_steps']
    info['StepsDayMinAdjusted_Weekend'] = steps_summary_adj['weekend_min_steps']
    info['StepsDayMaxAdjusted_Weekend'] = steps_summary_adj['weekend_max_steps']
    # steps, weekday stats
    info['TotalStepsAdjusted_Weekday'] = steps_summary_adj['weekday_total_steps']
    info['StepsDayAvgAdjusted_Weekday'] = steps_summary_adj['weekday_avg_steps']
    info['StepsDayMedAdjusted_Weekday'] = steps_summary_adj['weekday_med_steps']
    info['StepsDayMinAdjusted_Weekday'] = steps_summary_adj['weekday_min_steps']
    info['StepsDayMaxAdjusted_Weekday'] = steps_summary_adj['weekday_max_steps']
    # walking, overall stats
    info['TotalWalkingAdjusted(mins)'] = steps_summary_adj['total_walk']
    info['WalkingDayAvgAdjusted(mins)'] = steps_summary_adj['avg_walk']
    info['WalkingDayMedAdjusted(mins)'] = steps_summary_adj['med_walk']
    info['WalkingDayMinAdjusted(mins)'] = steps_summary_adj['min_walk']
    info['WalkingDayMaxAdjusted(mins)'] = steps_summary_adj['max_walk']
    # walking, weekend stats
    info['TotalWalkingAdjusted(mins)_Weekend'] = steps_summary_adj['weekend_total_walk']
    info['WalkingDayAvgAdjusted(mins)_Weekend'] = steps_summary_adj['weekend_avg_walk']
    info['WalkingDayMedAdjusted(mins)_Weekend'] = steps_summary_adj['weekend_med_walk']
    info['WalkingDayMinAdjusted(mins)_Weekend'] = steps_summary_adj['weekend_min_walk']
    info['WalkingDayMaxAdjusted(mins)_Weekend'] = steps_summary_adj['weekend_max_walk']
    # walking, weekday stats
    info['TotalWalkingAdjusted(mins)_Weekday'] = steps_summary_adj['weekday_total_walk']
    info['WalkingDayAvgAdjusted(mins)_Weekday'] = steps_summary_adj['weekday_avg_walk']
    info['WalkingDayMedAdjusted(mins)_Weekday'] = steps_summary_adj['weekday_med_walk']
    info['WalkingDayMinAdjusted(mins)_Weekday'] = steps_summary_adj['weekday_min_walk']
    info['WalkingDayMaxAdjusted(mins)_Weekday'] = steps_summary_adj['weekday_max_walk']
    # steps, time of accumulated steps
    info['Steps5thAtAdjusted'] = steps_summary_adj['ptile_at_avgs']['p05_at']
    info['Steps25thAtAdjusted'] = steps_summary_adj['ptile_at_avgs']['p25_at']
    info['Steps50thAtAdjusted'] = steps_summary_adj['ptile_at_avgs']['p50_at']
    info['Steps75thAtAdjusted'] = steps_summary_adj['ptile_at_avgs']['p75_at']
    info['Steps95thAtAdjusted'] = steps_summary_adj['ptile_at_avgs']['p95_at']
    # hour-of-day averages
    info.update({f'StepsAdjusted_Hour{h:02}': steps_summary_adj['hour_steps'].loc[h] for h in range(24)})
    info.update({f'StepsAdjusted_Hour{h:02}_Weekend': steps_summary_adj['weekend_hour_steps'].loc[h] for h in range(24)})
    info.update({f'StepsAdjusted_Hour{h:02}_Weekday': steps_summary_adj['weekday_hour_steps'].loc[h] for h in range(24)})
    info.update({f'WalkingAdjusted(mins)_Hour{h:02}': steps_summary_adj['hour_walks'].loc[h] for h in range(24)})
    info.update({f'WalkingAdjusted(mins)_Hour{h:02}_Weekend': steps_summary_adj['weekend_hour_walks'].loc[h] for h in range(24)})
    info.update({f'WalkingAdjusted(mins)_Hour{h:02}_Weekday': steps_summary_adj['weekday_hour_walks'].loc[h] for h in range(24)})

    # Cadence summary
    cadence_summary = summarize_cadence(Y, model.steptol)
    # overall stats
    info['CadencePeak1(steps/min)'] = cadence_summary['cadence_peak1']
    info['CadencePeak30(steps/min)'] = cadence_summary['cadence_peak30']
    info['Cadence95th(steps/min)'] = cadence_summary['cadence_p95']
    # weekend stats
    info['CadencePeak1(steps/min)_Weekend'] = cadence_summary['weekend_cadence_peak1']
    info['CadencePeak30(steps/min)_Weekend'] = cadence_summary['weekend_cadence_peak30']
    info['Cadence95th(steps/min)_Weekend'] = cadence_summary['weekend_cadence_p95']
    # weekday stats
    info['CadencePeak1(steps/min)_Weekday'] = cadence_summary['weekday_cadence_peak1']
    info['CadencePeak30(steps/min)_Weekday'] = cadence_summary['weekday_cadence_peak30']
    info['Cadence95th(steps/min)_Weekday'] = cadence_summary['weekday_cadence_p95']

    # Cadence summary, adjusted
    cadence_summary_adj = summarize_cadence(Y, model.steptol, adjust_estimates=True)
    info['CadencePeak1Adjusted(steps/min)'] = cadence_summary_adj['cadence_peak1']
    info['CadencePeak30Adjusted(steps/min)'] = cadence_summary_adj['cadence_peak30']
    info['Cadence95thAdjusted(steps/min)'] = cadence_summary_adj['cadence_p95']
    # weekend stats
    info['CadencePeak1Adjusted(steps/min)_Weekend'] = cadence_summary_adj['weekend_cadence_peak1']
    info['CadencePeak30Adjusted(steps/min)_Weekend'] = cadence_summary_adj['weekend_cadence_peak30']
    info['Cadence95thAdjusted(steps/min)_Weekend'] = cadence_summary_adj['weekend_cadence_p95']
    # weekday stats
    info['CadencePeak1Adjusted(steps/min)_Weekday'] = cadence_summary_adj['weekday_cadence_peak1']
    info['CadencePeak30Adjusted(steps/min)_Weekday'] = cadence_summary_adj['weekday_cadence_peak30']
    info['Cadence95thAdjusted(steps/min)_Weekday'] = cadence_summary_adj['weekday_cadence_p95']

    # Save Info.json
    with open(f"{outdir}/{basename}-Info.json", 'w') as f:
        json.dump(info, f, indent=4, cls=utils.NpEncoder)

    # Save hourly data
    hourly = pd.concat([
        steps_summary['hourly_steps'],
        enmo_summary['hourly'],
    ], axis=1)
    hourly.index.name = 'Time'
    hourly.reset_index(inplace=True)
    hourly.insert(0, 'Filename', info['Filename'])  # add filename for reference
    hourly.to_csv(f"{outdir}/{basename}-Hourly.csv.gz", index=False)
    del hourly  # free memory

    # Save hourly data, adjusted
    hourly_adj = pd.concat([
        steps_summary_adj['hourly_steps'],
        enmo_summary_adj['hourly'],
    ], axis=1)
    hourly_adj.index.name = 'Time'
    hourly_adj.reset_index(inplace=True)
    hourly_adj.insert(0, 'Filename', info['Filename'])  # add filename for reference
    hourly_adj.to_csv(f"{outdir}/{basename}-HourlyAdjusted.csv.gz", index=False)
    del hourly_adj  # free memory

    # Save minutely data
    minutely = pd.concat([
        steps_summary['minutely_steps'],
        enmo_summary['minutely'],
    ], axis=1)
    minutely.index.name = 'Time'
    minutely.reset_index(inplace=True)
    minutely.insert(0, 'Filename', info['Filename'])  # add filename for reference
    minutely.to_csv(f"{outdir}/{basename}-Minutely.csv.gz", index=False)
    del minutely  # free memory

    # Save minutely data, adjusted
    minutely_adj = pd.concat([
        steps_summary_adj['minutely_steps'],
        enmo_summary_adj['minutely'],
    ], axis=1)
    minutely_adj.index.name = 'Time'
    minutely_adj.reset_index(inplace=True)
    minutely_adj.insert(0, 'Filename', info['Filename'])  # add filename for reference
    minutely_adj.to_csv(f"{outdir}/{basename}-MinutelyAdjusted.csv.gz", index=False)
    del minutely_adj  # free memory

    # Save daily data
    daily = pd.concat([
        steps_summary['daily_steps'],
        cadence_summary['daily'],
        enmo_summary['daily'],
    ], axis=1)
    daily.index.name = 'Date'
    daily.reset_index(inplace=True)
    daily.insert(0, 'Filename', info['Filename'])  # add filename for reference
    daily.to_csv(f"{outdir}/{basename}-Daily.csv.gz", index=False)
    # del daily  # still needed for printing

    # Save daily data, adjusted
    daily_adj = pd.concat([
        steps_summary_adj['daily_steps'],
        cadence_summary_adj['daily'],
        enmo_summary_adj['daily'],
    ], axis=1)
    daily_adj.index.name = 'Date'
    daily_adj.reset_index(inplace=True)
    daily_adj.insert(0, 'Filename', info['Filename'])  # add filename for reference
    daily_adj.to_csv(f"{outdir}/{basename}-DailyAdjusted.csv.gz", index=False)
    # del daily_adj  # still needed for printing

    # Print
    print("\nSummary\n-------")
    print(json.dumps(
        {k: v for k, v in info.items() if not re.search(r'_Weekend|_Weekday|_Hour\d{2}', k)},
        indent=4, cls=utils.NpEncoder
    ))
    print("\nEstimated Daily Stats\n---------------------")
    print(daily.set_index('Date').drop(columns='Filename'))
    print("\nEstimated Daily Stats (Adjusted)\n---------------------")
    print(daily_adj.set_index('Date').drop(columns='Filename'))
    print("\nOutput files saved in:", outdir)

    after = time.time()
    print(f"Done! ({round(after - before,2)}s)")


def load_model(
    model_path: str,
    model_type: str,
    check_md5: bool = True,
    force_download: bool = False
):
    """
    Load a trained model from the specified path. Download the model if it does not exist.

    This function attempts to load a trained model from the given path. If the model file does not 
    exist or if `force_download` is set to True, it downloads the model from a predefined URL. 
    Optionally, it can check the MD5 checksum of the downloaded file to ensure its integrity.

    Parameters:
    - model_path (str or pathlib.Path): The path to the model file.
    - model_type (str): The type of model: "rf" for random forest model, or "ssl" for self-supervised learning model.
    - check_md5 (bool, optional): Whether to check the MD5 checksum of the model file. Defaults to True.
    - force_download (bool, optional): Whether to force download the model file even if it exists. Defaults to False.

    Returns:
    - The loaded model object.

    Raises:
    - AssertionError: If the MD5 checksum of the model file does not match the expected value.

    Example:
        model = load_model("path/to/model.joblib", "ssl")
    """

    pth = pathlib.Path(model_path)

    if force_download or not pth.exists():

        url = f"https://wearables-files.ndph.ox.ac.uk/files/models/stepcount/{__model_version__[model_type]}.joblib.lzma"

        print(f"Downloading {url}...")

        with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)

    if check_md5:
        assert utils.md5(pth) == __model_md5__[model_type], (
            "Model file is corrupted. Please run with --force-download "
            "to download the model file again."
        )

    return joblib.load(pth)


def summarize_enmo(
    data: pd.DataFrame,
    adjust_estimates: bool = False
):
    """
    Summarize ENMO information from raw accelerometer data, e.g. daily and hourly averages, percentiles, etc.

    Parameters:
    - data (pd.DataFrame): A pandas DataFrame of raw accelerometer data with columns 'x', 'y', 'z'.
    - adjust_estimates (bool, optional): Whether to adjust estimates to account for missing data. Defaults to False.

    Returns:
    - dict: A dictionary containing various summary ENMO statistics.

    Example:
        summary = summarize_enmo(data, adjust_estimates=True)
    """

    def _is_enough(x, min_wear=None, dt=None):
        if min_wear is None:
            return True  # no minimum wear time, then default to True
        if dt is None:
            dt = utils.infer_freq(x.index).total_seconds()
        return x.notna().sum() * dt / 60 > min_wear

    def _mean(x, min_wear=None, dt=None):
        if not _is_enough(x, min_wear, dt):
            return np.nan
        return x.mean()

    # Truncated ENMO: Euclidean norm minus one and clipped at zero
    v = np.sqrt(data['x'] ** 2 + data['y'] ** 2 + data['z'] ** 2)
    v = np.clip(v - 1, a_min=0, a_max=None)
    v *= 1000  # convert to mg
    # promptly downsample to minutely to reduce future computation and memory at minimal loss to accuracy
    v = v.resample('T').mean()

    dt = utils.infer_freq(v.index).total_seconds()

    if adjust_estimates:
        v = utils.impute_missing(v)

    if adjust_estimates:
        # adjusted estimates account for NAs
        minutely = v.resample('T').agg(_mean, min_wear=0.5, dt=dt).rename('ENMO(mg)')  # up to 30s/min missingness
        hourly = v.resample('H').agg(_mean, min_wear=50, dt=dt).rename('ENMO(mg)')  # up to 10min/h missingness
        daily = v.resample('D').agg(_mean, min_wear=21 * 60, dt=dt).rename('ENMO(mg)')  # up to 3h/d missingness
        # adjusted estimates first form a 7-day representative week before final aggregation
        # TODO: 7-day padding for shorter recordings
        day_of_week = utils.impute_days(daily).groupby(daily.index.weekday).mean()
        avg = day_of_week.mean()
        weekend_avg = day_of_week[day_of_week.index >= 5].mean()
        weekday_avg = day_of_week[day_of_week.index < 5].mean()
    else:
        # crude (unadjusted) estimates ignore NAs
        minutely = v.resample('T').mean().rename('ENMO(mg)')
        hourly = v.resample('H').mean().rename('ENMO(mg)')
        daily = v.resample('D').mean().rename('ENMO(mg)')
        avg = daily.mean()
        weekend_avg = daily[daily.index.weekday >= 5].mean()
        weekday_avg = daily[daily.index.weekday < 5].mean()

    # hour of day averages, 24-hour profile
    hour_avgs = hourly.groupby(hourly.index.hour).mean().reindex(range(24))
    weekend_hour_avgs = hourly[hourly.index.weekday >= 5].pipe(lambda x: x.groupby(x.index.hour).mean()).reindex(range(24))
    weekday_hour_avgs = hourly[hourly.index.weekday < 5].pipe(lambda x: x.groupby(x.index.hour).mean()).reindex(range(24))

    return {
        'avg': avg,
        'weekend_avg': weekend_avg,
        'weekday_avg': weekday_avg,
        'hourly': hourly,
        'daily': daily,
        'minutely': minutely,
        'hour_avgs': hour_avgs,
        'weekend_hour_avgs': weekend_hour_avgs,
        'weekday_hour_avgs': weekday_hour_avgs,
    }


def summarize_steps(
    Y: pd.Series, 
    steptol: int = 3, 
    adjust_estimates: bool = False
):
    """
    Summarize a series of step counts, e.g. daily and hourly averages, percentiles, etc.

    Parameters:
    - Y (pd.Series): A pandas Series of step counts.
    - steptol (int, optional): The minimum number of steps per window for the window to be considered valid for calculation. Defaults to 3 steps per window.
    - adjust_estimates (bool, optional): Whether to adjust estimates to account for missing data. Defaults to False.

    Returns:
    - dict: A dictionary containing various summary step count statistics.

    Example:
        summary = summarize_steps(Y, steptol=3, adjust_estimates=True)
    """

    # there's a bug with .resample().sum(skipna)
    # https://github.com/pandas-dev/pandas/issues/29382

    def _is_enough(x, min_wear=None, dt=None):
        if min_wear is None:
            return True  # no minimum wear time, then default to True
        if dt is None:
            dt = utils.infer_freq(x.index).total_seconds()
        return x.notna().sum() * dt / 60 > min_wear

    def _sum(x, min_wear=None, dt=None):
        if x.isna().all():  # have to explicitly check this because pandas' .sum() returns 0 if all-NaN
            return np.nan
        if not _is_enough(x, min_wear, dt):
            return np.nan
        return x.sum()

    def _mean(x, min_wear=None, dt=None):
        if not _is_enough(x, min_wear, dt):
            return np.nan
        return x.mean()

    def _min(x, min_wear=None, dt=None):
        if not _is_enough(x, min_wear, dt):
            return np.nan
        return x.min()

    def _max(x, min_wear=None, dt=None):
        if not _is_enough(x, min_wear, dt):
            return np.nan
        return x.max()

    def _median(x, min_wear=None, dt=None):
        if not _is_enough(x, min_wear, dt):
            return np.nan
        return x.median()

    def _percentile_at(x, ps=(5, 25, 50, 75, 95), min_wear=None, dt=None):
        percentiles = {f'p{p:02}_at': np.nan for p in ps}
        if not _is_enough(x, min_wear, dt):
            return percentiles
        z = x.cumsum() / x.sum()
        for p in ps:
            try:
                p_at = z[z >= p / 100].index[0]
                p_at = p_at - p_at.floor('D')
                percentiles[f'p{p:02}_at'] = p_at
            except IndexError:
                pass
        return percentiles

    def _tdelta_to_str(tdelta):
        if pd.isna(tdelta):
            return np.nan
        hours, rem = divmod(tdelta.seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    dt = utils.infer_freq(Y.index).total_seconds()
    W = Y.mask(~Y.isna(), Y >= steptol).astype('float')

    if adjust_estimates:
        Y = utils.impute_missing(Y)
        W = utils.impute_missing(W)

    # steps
    if adjust_estimates:
        # adjusted estimates account for NAs
        minutely_steps = Y.resample('T').agg(_sum, min_wear=0.5, dt=dt).rename('Steps')  # up to 30s/min missingness
        hourly_steps = Y.resample('H').agg(_sum, min_wear=50, dt=dt).rename('Steps')  # up to 10min/h missingness
        daily_steps = Y.resample('D').agg(_sum, min_wear=21 * 60, dt=dt).rename('Steps')  # up to 3h/d missingness
        # adjusted estimates first form a 7-day representative week before final aggregation
        # TODO: 7-day padding for shorter recordings
        day_of_week = utils.impute_days(daily_steps).groupby(daily_steps.index.weekday).mean()
        avg_steps = day_of_week.mean()
        med_steps = day_of_week.median()
        min_steps = day_of_week.min()
        max_steps = day_of_week.max()
        # weekend stats
        weekend_avg_steps = day_of_week[day_of_week.index >= 5].mean()
        weekend_med_steps = day_of_week[day_of_week.index >= 5].median()
        weekend_min_steps = day_of_week[day_of_week.index >= 5].min()
        weekend_max_steps = day_of_week[day_of_week.index >= 5].max()
        # weekday stats
        weekday_avg_steps = day_of_week[day_of_week.index < 5].mean()
        weekday_med_steps = day_of_week[day_of_week.index < 5].median()
        weekday_min_steps = day_of_week[day_of_week.index < 5].min()
        weekday_max_steps = day_of_week[day_of_week.index < 5].max()
    else:
        # crude (unadjusted) estimates ignore NAs
        minutely_steps = Y.resample('T').agg(_sum).rename('Steps')
        hourly_steps = Y.resample('H').agg(_sum).rename('Steps')
        daily_steps = Y.resample('D').agg(_sum).rename('Steps')
        avg_steps = daily_steps.mean()
        med_steps = daily_steps.median()
        min_steps = daily_steps.min()
        max_steps = daily_steps.max()
        # weekend stats
        weekend_avg_steps = daily_steps[daily_steps.index.weekday >= 5].mean()
        weekend_med_steps = daily_steps[daily_steps.index.weekday >= 5].median()
        weekend_min_steps = daily_steps[daily_steps.index.weekday >= 5].min()
        weekend_max_steps = daily_steps[daily_steps.index.weekday >= 5].max()
        # weekday stats
        weekday_avg_steps = daily_steps[daily_steps.index.weekday < 5].mean()
        weekday_med_steps = daily_steps[daily_steps.index.weekday < 5].median()
        weekday_min_steps = daily_steps[daily_steps.index.weekday < 5].min()
        weekday_max_steps = daily_steps[daily_steps.index.weekday < 5].max()

    total_steps = daily_steps.sum() if not daily_steps.isna().all() else np.nan  # note that .sum() returns 0 if all-NaN
    # weekend/weekday totals
    weekend_total_steps = daily_steps[daily_steps.index.weekday >= 5].pipe(lambda x: x.sum() if not x.isna().all() else np.nan)
    weekday_total_steps = daily_steps[daily_steps.index.weekday < 5].pipe(lambda x: x.sum() if not x.isna().all() else np.nan)

    # walking
    if adjust_estimates:
        # adjusted estimates account for NAs
        # minutely_walk = (W.resample('T').agg(_sum, min_wear=0.5, dt=dt) * dt / 60).rename('Walk(mins)')  # up to 30s/min missingness
        hourly_walk = (W.resample('H').agg(_sum, min_wear=50, dt=dt) * dt / 60).rename('Walk(mins)')  # up to 10min/h missingness
        daily_walk = (W.resample('D').agg(_sum, min_wear=21 * 60, dt=dt) * dt / 60).rename('Walk(mins)')  # up to 3h/d missingness
        # adjusted estimates first form a 7-day representative week before final aggregation
        # TODO: 7-day padding for shorter recordings
        day_of_week_walk = utils.impute_days(daily_walk).groupby(daily_walk.index.weekday).mean()
        avg_walk = day_of_week_walk.mean()
        med_walk = day_of_week_walk.median()
        min_walk = day_of_week_walk.min()
        max_walk = day_of_week_walk.max()
        # weekend stats
        weekend_avg_walk = day_of_week_walk[day_of_week_walk.index >= 5].mean()
        weekend_med_walk = day_of_week_walk[day_of_week_walk.index >= 5].median()
        weekend_min_walk = day_of_week_walk[day_of_week_walk.index >= 5].min()
        weekend_max_walk = day_of_week_walk[day_of_week_walk.index >= 5].max()
        # weekday stats
        weekday_avg_walk = day_of_week_walk[day_of_week_walk.index < 5].mean()
        weekday_med_walk = day_of_week_walk[day_of_week_walk.index < 5].median()
        weekday_min_walk = day_of_week_walk[day_of_week_walk.index < 5].min()
        weekday_max_walk = day_of_week_walk[day_of_week_walk.index < 5].max()
    else:
        # crude (unadjusted) estimates ignore NAs
        # minutely_walk = (W.resample('T').agg(_sum) * dt / 60).rename('Walk(mins)')
        hourly_walk = (W.resample('H').agg(_sum) * dt / 60).rename('Walk(mins)')
        daily_walk = (W.resample('D').agg(_sum) * dt / 60).rename('Walk(mins)')
        avg_walk = daily_walk.mean()
        med_walk = daily_walk.median()
        min_walk = daily_walk.min()
        max_walk = daily_walk.max()
        # weekend stats
        weekend_avg_walk = daily_walk[daily_walk.index.weekday >= 5].mean()
        weekend_med_walk = daily_walk[daily_walk.index.weekday >= 5].median()
        weekend_min_walk = daily_walk[daily_walk.index.weekday >= 5].min()
        weekend_max_walk = daily_walk[daily_walk.index.weekday >= 5].max()
        # weekday stats
        weekday_avg_walk = daily_walk[daily_walk.index.weekday < 5].mean()
        weekday_med_walk = daily_walk[daily_walk.index.weekday < 5].median()
        weekday_min_walk = daily_walk[daily_walk.index.weekday < 5].min()
        weekday_max_walk = daily_walk[daily_walk.index.weekday < 5].max()

    total_walk = daily_walk.sum() if not daily_walk.isna().all() else np.nan  # note that .sum() returns 0 if all-NaN
    # weekend/weekday walking totals
    weekend_total_walk = daily_walk[daily_walk.index.weekday >= 5].pipe(lambda x: x.sum() if not x.isna().all() else np.nan)
    weekday_total_walk = daily_walk[daily_walk.index.weekday < 5].pipe(lambda x: x.sum() if not x.isna().all() else np.nan)

    # time of accumulated steps
    if adjust_estimates:
        # adjusted estimates account for NAs
        daily_ptile_at = Y.groupby(pd.Grouper(freq='D')).apply(_percentile_at, min_wear=21 * 60, dt=dt).unstack(1)  # up to 3h/d missingness
    else:
        # crude (unadjusted) estimates ignore NAs
        daily_ptile_at = Y.groupby(pd.Grouper(freq='D')).apply(_percentile_at).unstack(1)
    ptile_at_avgs = daily_ptile_at.mean()

    # hour of day averages, 24-hour profile
    hour_steps = hourly_steps.groupby(hourly_steps.index.hour).mean().reindex(range(24))
    weekend_hour_steps = hourly_steps[hourly_steps.index.weekday >= 5].pipe(lambda x: x.groupby(x.index.hour).mean()).reindex(range(24))
    weekday_hour_steps = hourly_steps[hourly_steps.index.weekday < 5].pipe(lambda x: x.groupby(x.index.hour).mean()).reindex(range(24))
    hour_walks = hourly_walk.groupby(hourly_walk.index.hour).mean().reindex(range(24))
    weekend_hour_walks = hourly_walk[hourly_walk.index.weekday >= 5].pipe(lambda x: x.groupby(x.index.hour).mean()).reindex(range(24))
    weekday_hour_walks = hourly_walk[hourly_walk.index.weekday < 5].pipe(lambda x: x.groupby(x.index.hour).mean()).reindex(range(24))

    # daily stats
    daily_steps = pd.concat([
        daily_walk,
        daily_steps.round().astype(pd.Int64Dtype()),
        # convert timedelta to human-friendly format
        daily_ptile_at.rename(columns={
            'p05_at': 'Steps5thAt',
            'p25_at': 'Steps25thAt',
            'p50_at': 'Steps50thAt',
            'p75_at': 'Steps75thAt',
            'p95_at': 'Steps95thAt'
        }).applymap(_tdelta_to_str).astype(pd.StringDtype()),
    ], axis=1)

    # round steps
    minutely_steps = minutely_steps.round().astype(pd.Int64Dtype())
    hourly_steps = hourly_steps.round().astype(pd.Int64Dtype())
    total_steps = utils.nanint(np.round(total_steps))
    avg_steps = utils.nanint(np.round(avg_steps))
    med_steps = utils.nanint(np.round(med_steps))
    min_steps = utils.nanint(np.round(min_steps))
    max_steps = utils.nanint(np.round(max_steps))
    weekend_total_steps = utils.nanint(np.round(weekend_total_steps))
    weekend_avg_steps = utils.nanint(np.round(weekend_avg_steps))
    weekend_med_steps = utils.nanint(np.round(weekend_med_steps))
    weekend_min_steps = utils.nanint(np.round(weekend_min_steps))
    weekend_max_steps = utils.nanint(np.round(weekend_max_steps))
    weekday_total_steps = utils.nanint(np.round(weekday_total_steps))
    weekday_avg_steps = utils.nanint(np.round(weekday_avg_steps))
    weekday_med_steps = utils.nanint(np.round(weekday_med_steps))
    weekday_min_steps = utils.nanint(np.round(weekday_min_steps))
    weekday_max_steps = utils.nanint(np.round(weekday_max_steps))
    hour_steps = hour_steps.round().astype(pd.Int64Dtype())
    weekend_hour_steps = weekend_hour_steps.round().astype(pd.Int64Dtype())
    weekday_hour_steps = weekday_hour_steps.round().astype(pd.Int64Dtype())
    # convert timedelta to human-friendly format
    ptile_at_avgs = ptile_at_avgs.map(_tdelta_to_str)

    return {
        'minutely_steps': minutely_steps,
        'hourly_steps': hourly_steps,
        'daily_steps': daily_steps,
        # steps, overall stats
        'total_steps': total_steps,
        'avg_steps': avg_steps,
        'med_steps': med_steps,
        'min_steps': min_steps,
        'max_steps': max_steps,
        # steps, weekend stats
        'weekend_total_steps': weekend_total_steps,
        'weekend_avg_steps': weekend_avg_steps,
        'weekend_med_steps': weekend_med_steps,
        'weekend_min_steps': weekend_min_steps,
        'weekend_max_steps': weekend_max_steps,
        # steps, weekday stats
        'weekday_total_steps': weekday_total_steps,
        'weekday_avg_steps': weekday_avg_steps,
        'weekday_med_steps': weekday_med_steps,
        'weekday_min_steps': weekday_min_steps,
        'weekday_max_steps': weekday_max_steps,
        # walking, overall stats
        'total_walk': total_walk,
        'avg_walk': avg_walk,
        'med_walk': med_walk,
        'min_walk': min_walk,
        'max_walk': max_walk,
        # walking, weekend stats
        'weekend_total_walk': weekend_total_walk,
        'weekend_avg_walk': weekend_avg_walk,
        'weekend_med_walk': weekend_med_walk,
        'weekend_min_walk': weekend_min_walk,
        'weekend_max_walk': weekend_max_walk,
        # walking, weekday stats
        'weekday_total_walk': weekday_total_walk,
        'weekday_avg_walk': weekday_avg_walk,
        'weekday_med_walk': weekday_med_walk,
        'weekday_min_walk': weekday_min_walk,
        'weekday_max_walk': weekday_max_walk,
        # hour of day averages
        'hour_steps': hour_steps,
        'weekend_hour_steps': weekend_hour_steps,
        'weekday_hour_steps': weekday_hour_steps,
        'hour_walks': hour_walks,
        'weekend_hour_walks': weekend_hour_walks,
        'weekday_hour_walks': weekday_hour_walks,
        # time of accumulated steps
        'ptile_at_avgs': ptile_at_avgs,
    }


def summarize_cadence(
    Y: pd.Series,
    steptol: int = 3,
    adjust_estimates: bool = False
):
    """
    Summarize cadence information from a series of step counts.

    Parameters:
    - Y (pd.Series): A pandas Series of step counts.
    - steptol (int, optional): The minimum number of steps per window for the window to be considered valid for calculation. Defaults to 3 steps per window.
    - adjust_estimates (bool, optional): Whether to adjust estimates to account for missing data. Defaults to False.

    Returns:
    - dict: A dictionary containing various summary cadence statistics.

    Example:
        summary = summarize_cadence(Y, steptol=3, adjust_estimates=True)
    """

    # TODO: split walking and running cadence?

    def _cadence_max(x, steptol, walktol=30, n=1):
        y = x[x >= steptol]
        # if not enough walking time, return NA.
        # note: walktol in minutes, x must be minutely
        if len(y) < walktol:
            return np.nan
        return y.nlargest(n, keep='all').mean()

    def _cadence_p95(x, steptol, walktol=30):
        y = x[x >= steptol]
        # if not enough walking time, return NA.
        # note: walktol in minutes, x must be minutely
        if len(y) < walktol:
            return np.nan
        return y.quantile(.95)

    dt = utils.infer_freq(Y.index).total_seconds()
    steptol_in_minutes = steptol * 60 / dt  # rescale steptol to steps/min
    minutely = Y.resample('T').sum().rename('Steps')  # steps/min

    # cadence https://jamanetwork.com/journals/jama/fullarticle/2763292

    daily_cadence_peak1 = minutely.resample('D').agg(_cadence_max, steptol=steptol_in_minutes, walktol=10, n=1).rename('CadencePeak1(steps/min)')
    daily_cadence_peak30 = minutely.resample('D').agg(_cadence_max, steptol=steptol_in_minutes, walktol=30, n=30).rename('CadencePeak30(steps/min)')
    daily_cadence_p95 = minutely.resample('D').agg(_cadence_p95, walktol=10, steptol=steptol_in_minutes).rename('Cadence95th(steps/min)')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')

        if adjust_estimates:
            # adjusted estimates first form a 7-day representative week before final aggregation
            # TODO: 7-day padding for shorter recordings
            # TODO: maybe impute output daily_cadence? but skip user-excluded days
            day_of_week_cadence_peak1 = utils.impute_days(daily_cadence_peak1, method='median').groupby(daily_cadence_peak1.index.weekday).median()
            day_of_week_cadence_peak30 = utils.impute_days(daily_cadence_peak30, method='median').groupby(daily_cadence_peak30.index.weekday).median()
            day_of_week_cadence_p95 = utils.impute_days(daily_cadence_p95, method='median').groupby(daily_cadence_p95.index.weekday).median()

            cadence_peak1 = day_of_week_cadence_peak1.median()
            cadence_peak30 = day_of_week_cadence_peak30.median()
            cadence_p95 = day_of_week_cadence_p95.median()
            # weekend stats
            weekend_cadence_peak1 = day_of_week_cadence_peak1[day_of_week_cadence_peak1.index >= 5].median()
            weekend_cadence_peak30 = day_of_week_cadence_peak30[day_of_week_cadence_peak30.index >= 5].median()
            weekend_cadence_p95 = day_of_week_cadence_p95[day_of_week_cadence_p95.index >= 5].median()
            # weekday stats
            weekday_cadence_peak1 = day_of_week_cadence_peak1[day_of_week_cadence_peak1.index < 5].median()
            weekday_cadence_peak30 = day_of_week_cadence_peak30[day_of_week_cadence_peak30.index < 5].median()
            weekday_cadence_p95 = day_of_week_cadence_p95[day_of_week_cadence_p95.index < 5].median()

        else:
            cadence_peak1 = daily_cadence_peak1.median()
            cadence_peak30 = daily_cadence_peak30.median()
            cadence_p95 = daily_cadence_p95.median()
            # weekend stats
            weekend_cadence_peak1 = daily_cadence_peak1[daily_cadence_peak1.index.weekday >= 5].median()
            weekend_cadence_peak30 = daily_cadence_peak30[daily_cadence_peak30.index.weekday >= 5].median()
            weekend_cadence_p95 = daily_cadence_p95[daily_cadence_p95.index.weekday >= 5].median()
            # weekday stats
            weekday_cadence_peak1 = daily_cadence_peak1[daily_cadence_peak1.index.weekday < 5].median()
            weekday_cadence_peak30 = daily_cadence_peak30[daily_cadence_peak30.index.weekday < 5].median()
            weekday_cadence_p95 = daily_cadence_p95[daily_cadence_p95.index.weekday < 5].median()

    daily = pd.concat([
        daily_cadence_peak1.round().astype(pd.Int64Dtype()),
        daily_cadence_peak30.round().astype(pd.Int64Dtype()),
        daily_cadence_p95.round().astype(pd.Int64Dtype()),
    ], axis=1)

    return {
        'daily': daily,
        'cadence_peak1': utils.nanint(np.round(cadence_peak1)),
        'cadence_peak30': utils.nanint(np.round(cadence_peak30)),
        'cadence_p95': utils.nanint(np.round(cadence_p95)),
        # weekend stats
        'weekend_cadence_peak1': utils.nanint(np.round(weekend_cadence_peak1)),
        'weekend_cadence_peak30': utils.nanint(np.round(weekend_cadence_peak30)),
        'weekend_cadence_p95': utils.nanint(np.round(weekend_cadence_p95)),
        # weekday stats
        'weekday_cadence_peak1': utils.nanint(np.round(weekday_cadence_peak1)),
        'weekday_cadence_peak30': utils.nanint(np.round(weekday_cadence_peak30)),
        'weekday_cadence_p95': utils.nanint(np.round(weekday_cadence_p95)),
    }



if __name__ == '__main__':
    main()
