import warnings
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
    parser.add_argument("--txyz",
                        help=("Use this option to specify the column names for time, x, y, z "
                              "in the input file, in that order. Use a comma-separated string. "
                              "Default: 'time,x,y,z'"),
                        type=str, default="time,x,y,z")
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    args = parser.parse_args()

    before = time.time()

    verbose = not args.quiet

    # Load file
    data, info = read(
        args.filepath, 
        usecols=args.txyz, 
        resample_hz=30 if args.model_type == 'ssl' else None,
        sample_rate=args.sample_rate, 
        verbose=verbose
    )

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
    Y.to_csv(f"{outdir}/{basename}-Steps.csv.gz")
    # Save timestamps of each step
    T_steps.to_csv(f"{outdir}/{basename}-StepTimes.csv.gz", index=False)

    # ENMO summary
    enmo_summary = summarize_enmo(data)
    info['ENMO(mg)'] = enmo_summary['avg']
    enmo_summary['minutely'].to_csv(f"{outdir}/{basename}-MinutelyENMO.csv.gz")
    enmo_summary['hourly'].to_csv(f"{outdir}/{basename}-HourlyENMO.csv.gz")
    enmo_summary['daily'].to_csv(f"{outdir}/{basename}-DailyENMO.csv.gz")

    # ENMO summary, adjusted
    enmo_summary_adj = summarize_enmo(data, adjust_estimates=True)
    info['ENMOAdjusted(mg)'] = enmo_summary_adj['avg']
    enmo_summary_adj['minutely'].to_csv(f"{outdir}/{basename}-MinutelyENMOAdjusted.csv.gz")
    enmo_summary_adj['hourly'].to_csv(f"{outdir}/{basename}-HourlyENMOAdjusted.csv.gz")
    enmo_summary_adj['daily'].to_csv(f"{outdir}/{basename}-DailyENMOAdjusted.csv.gz")

    # Steps summary
    summary = summarize_steps(Y, model.steptol)
    summary['minutely'].to_csv(f"{outdir}/{basename}-MinutelySteps.csv.gz")
    summary['hourly'].to_csv(f"{outdir}/{basename}-HourlySteps.csv.gz")
    summary['daily'].to_csv(f"{outdir}/{basename}-DailySteps.csv.gz")
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
    info['Cadence95th(steps/min)'] = summary['cadence_p95']
    info['Steps5thDayAvgAt'] = summary['daily_ptile_at_avg']['p05_at']
    info['Steps25thDayAvgAt'] = summary['daily_ptile_at_avg']['p25_at']
    info['Steps50thDayAvgAt'] = summary['daily_ptile_at_avg']['p50_at']
    info['Steps75thDayAvgAt'] = summary['daily_ptile_at_avg']['p75_at']
    info['Steps95thDayAvgAt'] = summary['daily_ptile_at_avg']['p95_at']
    info['Steps5thDayMedAt'] = summary['daily_ptile_at_med']['p05_at']
    info['Steps25thDayMedAt'] = summary['daily_ptile_at_med']['p25_at']
    info['Steps50thDayMedAt'] = summary['daily_ptile_at_med']['p50_at']
    info['Steps75thDayMedAt'] = summary['daily_ptile_at_med']['p75_at']
    info['Steps95thDayMedAt'] = summary['daily_ptile_at_med']['p95_at']

    # Steps summary, adjusted
    summary_adj = summarize_steps(Y, model.steptol, adjust_estimates=True)
    summary_adj['minutely'].to_csv(f"{outdir}/{basename}-MinutelyStepsAdjusted.csv.gz")
    summary_adj['hourly'].to_csv(f"{outdir}/{basename}-HourlyStepsAdjusted.csv.gz")
    summary_adj['daily'].to_csv(f"{outdir}/{basename}-DailyStepsAdjusted.csv.gz")
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
    info['Cadence95thAdjusted(steps/min)'] = summary_adj['cadence_p95']
    info['Steps5thDayAvgAdjustedAt'] = summary_adj['daily_ptile_at_avg']['p05_at']
    info['Steps25thDayAvgAdjustedAt'] = summary_adj['daily_ptile_at_avg']['p25_at']
    info['Steps50thDayAvgAdjustedAt'] = summary_adj['daily_ptile_at_avg']['p50_at']
    info['Steps75thDayAvgAdjustedAt'] = summary_adj['daily_ptile_at_avg']['p75_at']
    info['Steps95thDayAvgAdjustedAt'] = summary_adj['daily_ptile_at_avg']['p95_at']
    info['Steps5thDayMedAdjustedAt'] = summary_adj['daily_ptile_at_med']['p05_at']
    info['Steps25thDayMedAdjustedAt'] = summary_adj['daily_ptile_at_med']['p25_at']
    info['Steps50thDayMedAdjustedAt'] = summary_adj['daily_ptile_at_med']['p50_at']
    info['Steps75thDayMedAdjustedAt'] = summary_adj['daily_ptile_at_med']['p75_at']
    info['Steps95thDayMedAdjustedAt'] = summary_adj['daily_ptile_at_med']['p95_at']

    # Save info
    with open(f"{outdir}/{basename}-Info.json", 'w') as f:
        json.dump(info, f, indent=4, cls=NpEncoder)

    # Print
    print("\nSummary\n-------")
    print(json.dumps(info, indent=4, cls=NpEncoder))
    print("\nEstimated Daily Stats\n---------------------")
    print(summary['daily'])
    print("\nEstimated Daily Stats (Adjusted)\n---------------------")
    print(summary_adj['daily'])

    after = time.time()
    print(f"Done! ({round(after - before,2)}s)")


def summarize_enmo(data: pd.DataFrame, adjust_estimates=False):
    """ Summarize ENMO data """

    # Truncated ENMO: Euclidean norm minus one and clipped at zero
    v = np.sqrt(data['x'] ** 2 + data['y'] ** 2 + data['z'] ** 2)
    v = np.clip(v - 1, a_min=0, a_max=None)
    v *= 1000  # convert to mg

    if adjust_estimates:
        v = impute_missing(v)
        skipna = False
    else:
        # crude summary ignores missing data
        skipna = True

    def _mean(x):
        if not skipna and x.isna().any():
            return np.nan
        return x.mean()

    # steps
    hourly = v.resample('H').agg(_mean).rename('ENMO(mg)')  # ENMO, hourly
    daily = v.resample('D').agg(_mean).rename('ENMO(mg)')  # ENMO, daily
    minutely = v.resample('T').agg(_mean).rename('ENMO(mg)')  # ENMO, minutely

    # steps, daily stats
    if not adjust_estimates:
        avg = daily.mean()
    else:
        day_of_week = daily.groupby(daily.index.weekday).mean()
        avg = day_of_week.mean()

    return {
        'avg': avg,
        'hourly': hourly,
        'daily': daily,
        'minutely': minutely,
    }


def summarize_steps(Y, steptol=3, adjust_estimates=False):
    """ Summarize step count data """

    if adjust_estimates:
        Y = impute_missing(Y)
        skipna = False
    else:
        # crude summary ignores missing data
        skipna = True

    def _sum(x):
        if not skipna and x.isna().any():
            return np.nan
        return x.sum()

    def _max(x, n=1):
        if not skipna and x.isna().any():
            return np.nan
        return x.nlargest(n, keep='all').mean()

    def _p95(x, steptol):
        if not skipna and x.isna().any():
            return np.nan
        return x[x >= steptol].quantile(.95)

    def _percentile_at(x, ps=(5, 25, 50, 75, 95)):
        percentiles = {f'p{p:02}_at': np.nan for p in ps}
        if not skipna and x.isna().any():
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

    # there's a bug with .resample().sum(skipna)
    # https://github.com/pandas-dev/pandas/issues/29382

    # steps
    total = np.round(Y.agg(_sum))  # total steps
    hourly = Y.resample('H').agg(_sum).rename('Steps')  # steps, hourly
    daily = Y.resample('D').agg(_sum).rename('Steps')  # steps, daily
    minutely = Y.resample('T').agg(_sum).rename('Steps')  # steps, minutely

    # steps, daily stats
    if not adjust_estimates:
        daily_avg = np.round(daily.mean())
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            daily_med = np.round(daily.median())
        daily_min = np.round(daily.min())
        daily_max = np.round(daily.max())
    else:
        day_of_week = daily.groupby(daily.index.weekday).mean()
        daily_avg = np.round(day_of_week.mean())
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            daily_med = np.round(day_of_week.median())
        daily_min = np.round(day_of_week.min())
        daily_max = np.round(day_of_week.max())

    # walking
    dt = pd.Timedelta(infer_freq(Y.index)).seconds
    W = Y.mask(~Y.isna(), Y >= steptol)
    total_walk = np.round(W.agg(_sum) * dt / 60)
    daily_walk = (W.resample('D').agg(_sum) * dt / 60).rename('Walk(mins)')

    # walking, daily stats
    if not adjust_estimates:
        daily_walk_avg = np.round(daily_walk.mean())
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            daily_walk_med = np.round(daily_walk.median())
        daily_walk_min = np.round(daily_walk.min())
        daily_walk_max = np.round(daily_walk.max())
    else:
        day_of_week_walk = daily_walk.groupby(daily_walk.index.weekday).mean()
        daily_walk_avg = np.round(day_of_week_walk.mean())
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            daily_walk_med = np.round(day_of_week_walk.median())
        daily_walk_min = np.round(day_of_week_walk.min())
        daily_walk_max = np.round(day_of_week_walk.max())

    # cadence https://jamanetwork.com/journals/jama/fullarticle/2763292
    daily_cadence_peak1 = minutely.resample('D').agg(_max, n=1).rename('CadencePeak1')
    daily_cadence_peak30 = minutely.resample('D').agg(_max, n=30).rename('CadencePeak30')
    daily_cadence_p95 = minutely.resample('D').agg(_p95, steptol=steptol * 60 / dt).rename('Cadence95th')  # scale steptol to steps/min
    if not adjust_estimates:
        cadence_peak1 = np.round(daily_cadence_peak1.mean())
        cadence_peak30 = np.round(daily_cadence_peak30.mean())
        cadence_p95 = np.round(daily_cadence_p95.mean())
    else:
        day_of_week_cadence_peak1 = daily_cadence_peak1.groupby(daily_cadence_peak1.index.weekday).mean()
        day_of_week_cadence_peak30 = daily_cadence_peak30.groupby(daily_cadence_peak30.index.weekday).mean()
        day_of_week_cadence_p95 = daily_cadence_p95.groupby(daily_cadence_p95.index.weekday).mean()
        cadence_peak1 = np.round(day_of_week_cadence_peak1.mean())
        cadence_peak30 = np.round(day_of_week_cadence_peak30.mean())
        cadence_p95 = np.round(day_of_week_cadence_p95.mean())

    daily_ptile_at = Y.groupby(pd.Grouper(freq='D')).apply(_percentile_at).unstack(1)
    daily_ptile_at_avg = daily_ptile_at.mean()
    daily_ptile_at_med = daily_ptile_at.median()

    # daily stats
    daily = pd.concat([
        pd.to_numeric(daily_walk.round(), downcast='integer'),
        pd.to_numeric(daily.round(), downcast='integer'),
        pd.to_numeric(daily_cadence_peak1.round(), downcast='integer'),
        pd.to_numeric(daily_cadence_peak30.round(), downcast='integer'),
        pd.to_numeric(daily_cadence_p95.round(), downcast='integer'),
        daily_ptile_at.rename(columns={
            'p05_at': 'Steps5thAt',
            'p25_at': 'Steps25thAt',
            'p50_at': 'Steps50thAt',
            'p75_at': 'Steps75thAt',
            'p95_at': 'Steps95thAt'
        }).applymap(_tdelta_to_str),
    ], axis=1)

    # convert units
    total = nanint(total)
    minutely = pd.to_numeric(minutely.round(), downcast='integer')
    hourly = pd.to_numeric(hourly.round(), downcast='integer')
    daily_avg = nanint(daily_avg)
    daily_med = nanint(daily_med)
    daily_min = nanint(daily_min)
    daily_max = nanint(daily_max)
    total_walk = nanint(total_walk)
    daily_walk_avg = nanint(daily_walk_avg)
    daily_walk_med = nanint(daily_walk_med)
    daily_walk_min = nanint(daily_walk_min)
    daily_walk_max = nanint(daily_walk_max)
    cadence_peak1 = nanint(cadence_peak1)
    cadence_peak30 = nanint(cadence_peak30)
    cadence_p95 = nanint(cadence_p95)
    daily_ptile_at_avg = daily_ptile_at_avg.map(_tdelta_to_str)
    daily_ptile_at_med = daily_ptile_at_med.map(_tdelta_to_str)

    return {
        'total': total,
        'minutely': minutely,
        'hourly': hourly,
        'daily': daily,
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
        'cadence_p95': cadence_p95,
        'daily_ptile_at_avg': daily_ptile_at_avg,
        'daily_ptile_at_med': daily_ptile_at_med,
    }


def impute_missing(data: pd.DataFrame, extrapolate=True):

    if extrapolate:  # extrapolate beyond start/end times to have full 24h
        freq = infer_freq(data.index)
        if pd.isna(freq):
            warnings.warn("Cannot infer frequency, using 1s")
            freq = pd.Timedelta('1s')
        freq = to_offset(freq)
        data = data.reindex(
            pd.date_range(
                # TODO: This fails if data.index[0] happens to be midnight
                data.index[0].floor('D'),
                data.index[-1].ceil('D'),
                freq=freq,
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


def read(
    filepath: str,
    usecols: str = 'time,x,y,z',
    resample_hz: str = 'uniform',
    sample_rate: float = None,
    verbose: bool = True
):

    p = pathlib.Path(filepath)
    fsize = round(p.stat().st_size / (1024 * 1024), 1)
    ftype = p.suffix.lower()
    if ftype in (".gz", ".xz", ".lzma", ".bz2", ".zip"):  # if file is compressed, check the next extension
        ftype = pathlib.Path(p.stem).suffix.lower()

    if ftype in (".csv", ".pkl"):

        if ftype == ".csv":
            tcol, xcol, ycol, zcol = usecols.split(',')
            data = pd.read_csv(
                filepath,
                usecols=[tcol, xcol, ycol, zcol],
                parse_dates=[tcol],
                index_col=tcol,
                dtype={xcol: 'f4', ycol: 'f4', zcol: 'f4'},
            )
            # rename to standard names
            data = data.rename(columns={xcol: 'x', ycol: 'y', zcol: 'z'})
            data.index.name = 'time'

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

    else:
        raise ValueError(f"Unknown file format: {ftype}")

    if 'ResampleRate' not in info:
        info['ResampleRate'] = info['SampleRate']

    return data, info


def infer_freq(t):
    """ Like pd.infer_freq but more forgiving """
    tdiff = t.to_series().diff()
    q1, q3 = tdiff.quantile([0.25, 0.75])
    tdiff = tdiff[(q1 <= tdiff) & (tdiff <= q3)]
    freq = tdiff.mean()
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
