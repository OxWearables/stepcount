import os
import pathlib
import urllib
import shutil
import time
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
import scipy.stats as stats
from pandas.tseries.frequencies import to_offset

import actipy

from stepcount import __model_version__
from stepcount import __model_md5__


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
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    args = parser.parse_args()

    before = time.time()

    verbose = not args.quiet

    # Load file
    if args.model_type == 'ssl':
        resample_hz = 30
    else:
        resample_hz = None
    data, info = read(args.filepath, resample_hz, sample_rate=args.sample_rate, verbose=verbose)

    # Output paths
    basename = resolve_path(args.filepath)[1]
    outdir = os.path.join(args.outdir, basename)
    os.makedirs(outdir, exist_ok=True)

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

    # Save step counts
    Y.to_csv(f"{outdir}/{basename}-Steps.csv")
    # Save timestamps of each step
    T_steps.to_csv(f"{outdir}/{basename}-StepTimes.csv", index=False)

    # Summary
    summary = summarize(Y, model.steptol)
    summary['hourly'].to_csv(f"{outdir}/{basename}-HourlySteps.csv")
    summary['daily_stats'].to_csv(f"{outdir}/{basename}-DailySteps.csv")
    info['TotalSteps'] = summary['total']
    info['StepsDayAvg'] = summary['daily_avg']
    info['StepsDayMed'] = summary['daily_med']
    info['StepsDayMin'] = summary['daily_min']
    info['StepsDayMax'] = summary['daily_max']
    info['TotalWalking(mins)'] = summary['total_walk']
    info['WalkingDayAvg(mins)'] = summary['daily_walk_avg']
    info['WalkingDayMed(mins)'] = summary['daily_walk_med']
    info['WalkingDayMin(mins)'] = summary['daily_walk_min']
    info['WalkingDayMax(mins)'] = summary['daily_walk_max']
    info['CadencePeak1(steps/min)'] = summary['cadence_peak1']
    info['CadencePeak30(steps/min)'] = summary['cadence_peak30']
    info['StepsQ1DayAvgAt'] = summary['daily_QAt_avg']['StepsQ1At']
    info['StepsQ2DayAvgAt'] = summary['daily_QAt_avg']['StepsQ2At']
    info['StepsQ3DayAvgAt'] = summary['daily_QAt_avg']['StepsQ3At']
    info['StepsQ1DayMedAt'] = summary['daily_QAt_med']['StepsQ1At']
    info['StepsQ2DayMedAt'] = summary['daily_QAt_med']['StepsQ2At']
    info['StepsQ3DayMedAt'] = summary['daily_QAt_med']['StepsQ3At']

    # Impute missing periods & recalculate summary
    summary_adj = summarize(Y, model.steptol, adjust_estimates=True)
    summary_adj['hourly'].to_csv(f"{outdir}/{basename}-HourlyStepsAdjusted.csv")
    summary_adj['daily_stats'].to_csv(f"{outdir}/{basename}-DailyStepsAdjusted.csv")
    info['TotalStepsAdjusted'] = summary_adj['total']
    info['StepsDayAvgAdjusted'] = summary_adj['daily_avg']
    info['StepsDayMedAdjusted'] = summary_adj['daily_med']
    info['StepsDayMinAdjusted'] = summary_adj['daily_min']
    info['StepsDayMaxAdjusted'] = summary_adj['daily_max']
    info['TotalWalkingAdjusted(mins)'] = summary_adj['total_walk']
    info['WalkingDayAvgAdjusted(mins)'] = summary_adj['daily_walk_avg']
    info['WalkingDayMedAdjusted(mins)'] = summary_adj['daily_walk_med']
    info['WalkingDayMinAdjusted(mins)'] = summary_adj['daily_walk_min']
    info['WalkingDayMaxAdjusted(mins)'] = summary_adj['daily_walk_max']
    info['CadencePeak1Adjusted(steps/min)'] = summary_adj['cadence_peak1']
    info['CadencePeak30Adjusted(steps/min)'] = summary_adj['cadence_peak30']
    info['StepsQ1DayAvgAdjustedAt'] = summary_adj['daily_QAt_avg']['StepsQ1At']
    info['StepsQ2DayAvgAdjustedAt'] = summary_adj['daily_QAt_avg']['StepsQ2At']
    info['StepsQ3DayAvgAdjustedAt'] = summary_adj['daily_QAt_avg']['StepsQ3At']
    info['StepsQ1DayMedAdjustedAt'] = summary_adj['daily_QAt_med']['StepsQ1At']
    info['StepsQ2DayMedAdjustedAt'] = summary_adj['daily_QAt_med']['StepsQ2At']
    info['StepsQ3DayMedAdjustedAt'] = summary_adj['daily_QAt_med']['StepsQ3At']

    # Save info
    with open(f"{outdir}/{basename}-Info.json", 'w') as f:
        json.dump(info, f, indent=4, cls=NpEncoder)

    # Print
    print("\nSummary\n-------")
    print(json.dumps(info, indent=4, cls=NpEncoder))
    print("\nEstimated Daily Stats\n---------------------")
    print(summary['daily_stats'])
    print("\nEstimated Daily Stats (Adjusted)\n---------------------")
    print(summary_adj['daily_stats'])

    after = time.time()
    print(f"Done! ({round(after - before,2)}s)")


def summarize(Y, steptol=3, adjust_estimates=False):

    if adjust_estimates:
        Y = impute_missing(Y)
        skipna = False
    else:
        # crude summary ignores missing data
        skipna = True

    def _sum(x):
        x = x.to_numpy()
        if skipna:
            return np.nansum(x)
        return np.sum(x)

    # there's a bug with .resample().sum(skipna)
    # https://github.com/pandas-dev/pandas/issues/29382

    # steps
    total = np.round(Y.agg(_sum))  # total steps
    hourly = Y.resample('H').agg(_sum).round().rename('Steps')  # steps, hourly
    daily = Y.resample('D').agg(_sum).round().rename('Steps')  # steps, daily
    if not adjust_estimates:
        daily_avg = np.round(daily.mean())
        daily_med = np.round(daily.median())
        daily_min = np.round(daily.min())
        daily_max = np.round(daily.max())
    else:
        weekdaily = daily.groupby(daily.index.weekday).mean()
        daily_avg = np.round(weekdaily.mean())
        daily_med = np.round(weekdaily.median())
        daily_min = np.round(weekdaily.min())
        daily_max = np.round(weekdaily.max())

    # walking
    dt = pd.Timedelta(infer_freq(Y.index)).seconds
    W = Y.mask(~Y.isna(), Y >= steptol)
    total_walk = np.round(W.agg(_sum) * dt / 60)
    daily_walk = (W.resample('D').agg(_sum) * dt / 60).round().rename('Walk(mins)')
    if not adjust_estimates:
        daily_walk_avg = np.round(daily_walk.mean())
        daily_walk_med = np.round(daily_walk.median())
        daily_walk_min = np.round(daily_walk.min())
        daily_walk_max = np.round(daily_walk.max())
    else:
        weekdaily_walk = daily_walk.groupby(daily_walk.index.weekday).mean()
        daily_walk_avg = np.round(weekdaily_walk.mean())
        daily_walk_med = np.round(weekdaily_walk.median())
        daily_walk_min = np.round(weekdaily_walk.min())
        daily_walk_max = np.round(weekdaily_walk.max())

    def _max(x, n=1):
        return x.nlargest(n, keep='all').mean()

    # cadence https://jamanetwork.com/journals/jama/fullarticle/2763292
    cadence = Y.resample('min').sum()
    daily_cadence_peak1 = cadence.resample('D').agg(_max, n=1)
    daily_cadence_peak30 = cadence.resample('D').agg(_max, n=30)
    if not adjust_estimates:
        cadence_peak1 = np.round(daily_cadence_peak1.mean())
        cadence_peak30 = np.round(daily_cadence_peak30.mean())
    else:
        weekdaily_cadence_peak1 = daily_cadence_peak1.groupby(daily_cadence_peak1.index.weekday).mean()
        weekdaily_cadence_peak30 = daily_cadence_peak30.groupby(daily_cadence_peak30.index.weekday).mean()
        cadence_peak1 = np.round(weekdaily_cadence_peak1.mean())
        cadence_peak30 = np.round(weekdaily_cadence_peak30.mean())

    # distributional features - quantiles
    def _QAt(x):
        na_return = {'StepsQ1At': np.nan, 'StepsQ2At': np.nan, 'StepsQ3At': np.nan}
        if adjust_estimates and x.isna().any():
            return na_return
        z = x.cumsum() / x.sum()
        try:
            q1_at = z[z >= 0.25].index[0]
            q1_at = q1_at - q1_at.floor('D')
            q2_at = z[z >= 0.5].index[0]
            q2_at = q2_at - q2_at.floor('D')
            q3_at = z[z >= 0.75].index[0]
            q3_at = q3_at - q3_at.floor('D')
            return {'StepsQ1At': q1_at, 'StepsQ2At': q2_at, 'StepsQ3At': q3_at}
        except IndexError:
            return na_return

    def _QAt_to_str(tdelta):
        if pd.isna(tdelta):
            return np.nan
        hours, rem = divmod(tdelta.seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    daily_QAt = Y.groupby(pd.Grouper(freq='D')).apply(_QAt).unstack(1)
    daily_QAt_avg = daily_QAt.mean()
    daily_QAt_med = daily_QAt.median()

    # daily stats
    daily_stats = pd.concat([
        daily_walk,
        daily,
        daily_QAt.applymap(_QAt_to_str),
    ], axis=1)

    # convert units
    total = nanint(total)
    hourly = pd.to_numeric(hourly, downcast='integer')
    daily = pd.to_numeric(daily, downcast='integer')
    daily_avg = nanint(daily_avg)
    daily_med = nanint(daily_med)
    daily_min = nanint(daily_min)
    daily_max = nanint(daily_max)
    total_walk = nanint(total_walk)
    daily_walk = pd.to_numeric(daily_walk, downcast='integer')
    daily_walk_avg = nanint(daily_walk_avg)
    daily_walk_med = nanint(daily_walk_med)
    daily_walk_min = nanint(daily_walk_min)
    daily_walk_max = nanint(daily_walk_max)
    cadence_peak1 = nanint(cadence_peak1)
    cadence_peak30 = nanint(cadence_peak30)
    daily_QAt_avg = daily_QAt_avg.map(_QAt_to_str)
    daily_QAt_med = daily_QAt_med.map(_QAt_to_str)

    return {
        'total': total,
        'hourly': hourly,
        'daily_stats': daily_stats,
        'daily_avg': daily_avg,
        'daily_med': daily_med,
        'daily_min': daily_min,
        'daily_max': daily_max,
        'total_walk': total_walk,
        'daily_walk_avg': daily_walk_avg,
        'daily_walk_med': daily_walk_med,
        'daily_walk_min': daily_walk_min,
        'daily_walk_max': daily_walk_max,
        'cadence_peak1': cadence_peak1,
        'cadence_peak30': cadence_peak30,
        'daily_QAt_avg': daily_QAt_avg,
        'daily_QAt_med': daily_QAt_med,
    }


def impute_missing(data: pd.DataFrame, extrapolate=True):

    if extrapolate:
        # padding at the boundaries to have full 24h
        data = data.reindex(
            pd.date_range(
                data.index[0].floor('D'),
                data.index[-1].ceil('D'),
                freq=to_offset(infer_freq(data.index)),
                inclusive='left',
                name='time',
            ),
            method='nearest',
            tolerance=pd.Timedelta('1m'),
            limit=1)

    def fillna(subframe):
        if isinstance(subframe, pd.Series):
            x = subframe.to_numpy()
            nan = np.isnan(x)
            nanlen = len(x[nan])
            if 0 < nanlen < len(x):  # check x contains a NaN and is not all NaN
                x[nan] = np.nanmean(x)
                return x  # will be cast back to a Series automatically
            else:
                return subframe

    data = (
        data
        # first attempt imputation using same day of week
        .groupby([data.index.weekday, data.index.hour, data.index.minute])
        .transform(fillna)
        # then try within weekday/weekend
        .groupby([data.index.weekday >= 5, data.index.hour, data.index.minute])
        .transform(fillna)
        # finally, use all other days
        .groupby([data.index.hour, data.index.minute])
        .transform(fillna)
    )

    return data


def nanint(x):
    if np.isnan(x):
        return x
    return int(x)


def read(filepath, resample_hz='uniform', sample_rate=None, verbose=True):

    p = pathlib.Path(filepath)
    ftype = p.suffixes[0].lower()
    fsize = round(p.stat().st_size / (1024 * 1024), 1)

    if ftype in (".csv", ".pkl"):

        if ftype == ".csv":
            data = pd.read_csv(
                filepath,
                usecols=['time', 'x', 'y', 'z'],
                parse_dates=['time'],
                index_col='time',
                dtype={'x': 'f4', 'y': 'f4', 'z': 'f4'},
            )
        elif ftype == ".pkl":
            data = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unknown file format: {ftype}")

        if sample_rate in (None, False):
            freq = infer_freq(data.index)
            sample_rate = int(np.round(pd.Timedelta('1s') / freq))

        # Quick fix: Drop duplicate indices. TODO: Maybe should be handled by actipy.
        data = data[~data.index.duplicated(keep='first')]

        data, info = actipy.process(
            data, sample_rate,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=resample_hz,
            verbose=verbose,
        )

        info = {
            **{"Filename": filepath,
                "Device": ftype,
                "Filesize(MB)": fsize,
                "SampleRate": sample_rate},
            **info
        }

    elif ftype in (".cwa", ".gt3x", ".bin"):

        data, info = actipy.read_device(
            filepath,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=resample_hz,
            verbose=verbose,
        )

    if 'ResampleRate' not in info:
        info['ResampleRate'] = info['SampleRate']

    return data, info


def infer_freq(x):
    """ Like pd.infer_freq but more forgiving """
    freq, _ = stats.mode(np.diff(x), keepdims=False)
    freq = pd.Timedelta(freq)
    return freq


def resolve_path(path):
    """ Return parent folder, file name and file extension """
    p = pathlib.Path(path)
    extension = p.suffixes[0]
    filename = p.name.rsplit(extension)[0]
    dirname = p.parent
    return dirname, filename, extension


def load_model(model_path, model_type, check_md5=True, force_download=False):
    """ Load trained model. Download if not exists. """

    pth = pathlib.Path(model_path)

    if force_download or not pth.exists():

        url = f"https://wearables-files.ndph.ox.ac.uk/files/models/stepcount/{__model_version__[model_type]}.joblib.lzma"

        print(f"Downloading {url}...")

        with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)

    if check_md5:
        assert md5(pth) == __model_md5__[model_type], (
            "Model file is corrupted. Please run with --force-download "
            "to download the model file again."
        )

    return joblib.load(pth)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



if __name__ == '__main__':
    main()
