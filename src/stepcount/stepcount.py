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
    parser.add_argument("--exclude-wear-below", "-w",
                        help=("Minimum wear time for a day to be considered valid, otherwise exclude it. "
                              "Pass values as strings, e.g.: '12H', '30min'. Default: None (no exclusion)"),
                        type=str, default=None)
    parser.add_argument("--txyz",
                        help=("Use this option to specify the column names for time, x, y, z "
                              "in the input file, in that order. Use a comma-separated string. "
                              "Default: 'time,x,y,z'"),
                        type=str, default="time,x,y,z")
    parser.add_argument("--exclude-first-last", "-e",
                        help="Exclude first, last or both days of data. Default: None (no exclusion)",
                        type=str, choices=['first', 'last', 'both'], default=None)
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
    enmo_summary = summarize_enmo(data, exclude_wear_below=args.exclude_wear_below, exclude_first_last=args.exclude_first_last)
    info['ENMO(mg)'] = enmo_summary['avg']

    # ENMO summary, adjusted
    enmo_summary_adj = summarize_enmo(data, exclude_wear_below=args.exclude_wear_below, exclude_first_last=args.exclude_first_last, adjust_estimates=True)
    info['ENMOAdjusted(mg)'] = enmo_summary_adj['avg']

    # Steps summary
    steps_summary = summarize_steps(Y, model.steptol, exclude_wear_below=args.exclude_wear_below, exclude_first_last=args.exclude_first_last)
    info['TotalSteps'] = steps_summary['total']
    info['StepsDayAvg'] = steps_summary['daily_avg']
    info['StepsDayMed'] = steps_summary['daily_med']
    info['StepsDayMin'] = steps_summary['daily_min']
    info['StepsDayMax'] = steps_summary['daily_max']
    info['TotalWalking(mins)'] = steps_summary['total_walk']
    info['WalkingDayAvg(mins)'] = steps_summary['daily_walk_avg']
    info['WalkingDayMed(mins)'] = steps_summary['daily_walk_med']
    info['WalkingDayMin(mins)'] = steps_summary['daily_walk_min']
    info['WalkingDayMax(mins)'] = steps_summary['daily_walk_max']
    info['Steps5thAt'] = steps_summary['daily_ptile_at_avg']['p05_at']
    info['Steps25thAt'] = steps_summary['daily_ptile_at_avg']['p25_at']
    info['Steps50thAt'] = steps_summary['daily_ptile_at_avg']['p50_at']
    info['Steps75thAt'] = steps_summary['daily_ptile_at_avg']['p75_at']
    info['Steps95thAt'] = steps_summary['daily_ptile_at_avg']['p95_at']

    # Steps summary, adjusted
    steps_summary_adj = summarize_steps(Y, model.steptol, exclude_wear_below=args.exclude_wear_below, exclude_first_last=args.exclude_first_last, adjust_estimates=True)
    info['TotalStepsAdjusted'] = steps_summary_adj['total']
    info['StepsDayAvgAdjusted'] = steps_summary_adj['daily_avg']
    info['StepsDayMedAdjusted'] = steps_summary_adj['daily_med']
    info['StepsDayMinAdjusted'] = steps_summary_adj['daily_min']
    info['StepsDayMaxAdjusted'] = steps_summary_adj['daily_max']
    info['TotalWalkingAdjusted(mins)'] = steps_summary_adj['total_walk']
    info['WalkingDayAvgAdjusted(mins)'] = steps_summary_adj['daily_walk_avg']
    info['WalkingDayMedAdjusted(mins)'] = steps_summary_adj['daily_walk_med']
    info['WalkingDayMinAdjusted(mins)'] = steps_summary_adj['daily_walk_min']
    info['WalkingDayMaxAdjusted(mins)'] = steps_summary_adj['daily_walk_max']
    info['Steps5thAtAdjusted'] = steps_summary_adj['daily_ptile_at_avg']['p05_at']
    info['Steps25thAtAdjusted'] = steps_summary_adj['daily_ptile_at_avg']['p25_at']
    info['Steps50thAtAdjusted'] = steps_summary_adj['daily_ptile_at_avg']['p50_at']
    info['Steps75thAtAdjusted'] = steps_summary_adj['daily_ptile_at_avg']['p75_at']
    info['Steps95thAtAdjusted'] = steps_summary_adj['daily_ptile_at_avg']['p95_at']

    # Cadence summary
    cadence_summary = summarize_cadence(Y, model.steptol, exclude_wear_below=args.exclude_wear_below, exclude_first_last=args.exclude_first_last)
    info['CadencePeak1(steps/min)'] = cadence_summary['cadence_peak1']
    info['CadencePeak30(steps/min)'] = cadence_summary['cadence_peak30']
    info['Cadence95th(steps/min)'] = cadence_summary['cadence_p95']

    # Cadence summary, adjusted
    cadence_summary_adj = summarize_cadence(Y, model.steptol, exclude_wear_below=args.exclude_wear_below, exclude_first_last=args.exclude_first_last, adjust_estimates=True)
    info['CadencePeak1Adjusted(steps/min)'] = cadence_summary_adj['cadence_peak1']
    info['CadencePeak30Adjusted(steps/min)'] = cadence_summary_adj['cadence_peak30']
    info['Cadence95thAdjusted(steps/min)'] = cadence_summary_adj['cadence_p95']

    # Save Info.json
    with open(f"{outdir}/{basename}-Info.json", 'w') as f:
        json.dump(info, f, indent=4, cls=NpEncoder)

    # Save hourly data
    hourly = pd.concat([
        steps_summary['hourly'],
        enmo_summary['hourly'],
    ], axis=1)
    hourly.index.name = 'Time'
    hourly.reset_index(inplace=True)
    hourly.insert(0, 'Filename', info['Filename'])  # add filename for reference
    hourly.to_csv(f"{outdir}/{basename}-Hourly.csv.gz", index=False)
    del hourly  # free memory

    # Save hourly data, adjusted
    hourly_adj = pd.concat([
        steps_summary_adj['hourly'],
        enmo_summary_adj['hourly'],
    ], axis=1)
    hourly_adj.index.name = 'Time'
    hourly_adj.reset_index(inplace=True)
    hourly_adj.insert(0, 'Filename', info['Filename'])  # add filename for reference
    hourly_adj.to_csv(f"{outdir}/{basename}-HourlyAdjusted.csv.gz", index=False)
    del hourly_adj  # free memory

    # Save minutely data
    minutely = pd.concat([
        steps_summary['minutely'],
        enmo_summary['minutely'],
    ], axis=1)
    minutely.index.name = 'Time'
    minutely.reset_index(inplace=True)
    minutely.insert(0, 'Filename', info['Filename'])  # add filename for reference
    minutely.to_csv(f"{outdir}/{basename}-Minutely.csv.gz", index=False)
    del minutely  # free memory

    # Save minutely data, adjusted
    minutely_adj = pd.concat([
        steps_summary_adj['minutely'],
        enmo_summary_adj['minutely'],
    ], axis=1)
    minutely_adj.index.name = 'Time'
    minutely_adj.reset_index(inplace=True)
    minutely_adj.insert(0, 'Filename', info['Filename'])  # add filename for reference
    minutely_adj.to_csv(f"{outdir}/{basename}-MinutelyAdjusted.csv.gz", index=False)
    del minutely_adj  # free memory

    # Save daily data
    daily = pd.concat([
        steps_summary['daily'],
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
        steps_summary_adj['daily'],
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
    print(json.dumps(info, indent=4, cls=NpEncoder))
    print("\nEstimated Daily Stats\n---------------------")
    print(daily.set_index('Date').drop(columns='Filename'))
    print("\nEstimated Daily Stats (Adjusted)\n---------------------")
    print(daily_adj.set_index('Date').drop(columns='Filename'))
    print("\nOutput files saved in:", outdir)

    after = time.time()
    print(f"Done! ({round(after - before,2)}s)")


def summarize_enmo(data: pd.DataFrame, exclude_wear_below=None, exclude_first_last=None, adjust_estimates=False):
    """ Summarize ENMO data """

    def _is_enough(x, min_wear=None, dt=None):
        if min_wear is None:
            return True  # no minimum wear time, then default to True
        if dt is None:
            dt = infer_freq(x.index).total_seconds()
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
    v = v.resample('T').agg(_mean)

    dt = infer_freq(v.index).total_seconds()

    if exclude_first_last is not None:
        v = exclude_first_last_days(v, exclude_first_last)

    if exclude_wear_below is not None:
        v = exclude_wear_below_days(v, exclude_wear_below)

    if adjust_estimates:
        v = impute_missing(v)

    if adjust_estimates:
        # adjusted estimates account for NAs
        minutely = v.resample('T').agg(_mean, min_wear=0.5, dt=dt).rename('ENMO(mg)')  # up to 30s/min missingness
        hourly = v.resample('H').agg(_mean, min_wear=50, dt=dt).rename('ENMO(mg)')  # up to 10min/h missingness
        daily = v.resample('D').agg(_mean, min_wear=21*60, dt=dt).rename('ENMO(mg)')  # up to 3h/d missingness
        # adjusted estimates first form a 7-day representative week before final aggregation
        # TODO: 7-day padding for shorter recordings
        day_of_week = impute_days(daily).groupby(daily.index.weekday).agg(_mean)
        avg = day_of_week.agg(_mean)
    else:
        # crude (unadjusted) estimates ignore NAs
        minutely = v.resample('T').agg(_mean).rename('ENMO(mg)')
        hourly = v.resample('H').agg(_mean).rename('ENMO(mg)')
        daily = v.resample('D').agg(_mean).rename('ENMO(mg)')
        avg = daily.agg(_mean)

    return {
        'avg': avg,
        'hourly': hourly,
        'daily': daily,
        'minutely': minutely,
    }


def summarize_steps(Y, steptol=3, exclude_wear_below=None, exclude_first_last=None, adjust_estimates=False):
    """ Summarize step count data """

    # there's a bug with .resample().sum(skipna)
    # https://github.com/pandas-dev/pandas/issues/29382

    def _is_enough(x, min_wear=None, dt=None):
        if min_wear is None:
            return True  # no minimum wear time, then default to True
        if dt is None:
            dt = infer_freq(x.index).total_seconds()
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
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
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

    if exclude_first_last is not None:
        Y = exclude_first_last_days(Y, exclude_first_last)

    if exclude_wear_below is not None:
        Y = exclude_wear_below_days(Y, exclude_wear_below)

    dt = infer_freq(Y.index).total_seconds()
    W = Y.mask(~Y.isna(), Y >= steptol).astype('float')

    if adjust_estimates:
        Y = impute_missing(Y)
        W = impute_missing(W)

    # steps
    if adjust_estimates:
        # adjusted estimates account for NAs
        minutely = Y.resample('T').agg(_sum, min_wear=0.5, dt=dt).rename('Steps')  # up to 30s/min missingness
        hourly = Y.resample('H').agg(_sum, min_wear=50, dt=dt).rename('Steps')  # up to 10min/h missingness
        daily = Y.resample('D').agg(_sum, min_wear=21*60, dt=dt).rename('Steps')  # up to 3h/d missingness
        # adjusted estimates first form a 7-day representative week before final aggregation
        # TODO: 7-day padding for shorter recordings
        day_of_week = impute_days(daily).groupby(daily.index.weekday).agg(_mean)
        daily_avg = day_of_week.agg(_mean)
        daily_med = day_of_week.agg(_median)
        daily_min = day_of_week.agg(_min)
        daily_max = day_of_week.agg(_max)
    else:
        # crude (unadjusted) estimates ignore NAs
        minutely = Y.resample('T').agg(_sum).rename('Steps')
        hourly = Y.resample('H').agg(_sum).rename('Steps')
        daily = Y.resample('D').agg(_sum).rename('Steps')
        daily_avg = daily.agg(_mean)
        daily_med = daily.agg(_median)
        daily_min = daily.agg(_min)
        daily_max = daily.agg(_max)

    total = daily.sum() if not daily.isna().all() else np.nan  # note that .sum() returns 0 if all-NaN

    # walking
    if adjust_estimates:
        # adjusted estimates account for NAs
        daily_walk = (W.resample('D').agg(_sum, min_wear=21*60, dt=dt) * dt / 60).rename('Walk(mins)')  # up to 3h/d missingness
        # adjusted estimates first form a 7-day representative week before final aggregation
        # TODO: 7-day padding for shorter recordings
        day_of_week_walk = impute_days(daily_walk).groupby(daily_walk.index.weekday).agg(_mean)
        daily_walk_avg = day_of_week_walk.agg(_mean)
        daily_walk_med = day_of_week_walk.agg(_median)
        daily_walk_min = day_of_week_walk.agg(_min)
        daily_walk_max = day_of_week_walk.agg(_max)
    else:
        # crude (unadjusted) estimates ignore NAs
        daily_walk = (W.resample('D').agg(_sum) * dt / 60).rename('Walk(mins)')
        daily_walk_avg = daily_walk.agg(_mean)
        daily_walk_med = daily_walk.agg(_median)
        daily_walk_min = daily_walk.agg(_min)
        daily_walk_max = daily_walk.agg(_max)

    total_walk = daily_walk.sum() if not daily_walk.isna().all() else np.nan  # note that .sum() returns 0 if all-NaN

    # time of accumulated steps
    if adjust_estimates:
        # adjusted estimates account for NAs
        daily_ptile_at = Y.groupby(pd.Grouper(freq='D')).apply(_percentile_at, min_wear=21*60, dt=dt).unstack(1)  # up to 3h/d missingness
    else:
        # crude (unadjusted) estimates ignore NAs
        daily_ptile_at = Y.groupby(pd.Grouper(freq='D')).apply(_percentile_at).unstack(1)
    daily_ptile_at_avg = daily_ptile_at.mean()

    # daily stats
    daily = pd.concat([
        daily_walk.round().astype(pd.Int64Dtype()),
        daily.round().astype(pd.Int64Dtype()),
        daily_ptile_at.rename(columns={
            'p05_at': 'Steps5thAt',
            'p25_at': 'Steps25thAt',
            'p50_at': 'Steps50thAt',
            'p75_at': 'Steps75thAt',
            'p95_at': 'Steps95thAt'
        }).applymap(_tdelta_to_str).astype(pd.StringDtype()),
    ], axis=1)

    # convert units
    total = nanint(np.round(total))
    minutely = minutely.round().astype(pd.Int64Dtype())
    hourly = hourly.round().astype(pd.Int64Dtype())
    daily_avg = nanint(np.round(daily_avg))
    daily_med = nanint(np.round(daily_med))
    daily_min = nanint(np.round(daily_min))
    daily_max = nanint(np.round(daily_max))
    total_walk = nanint(np.round(total_walk))
    daily_walk_avg = nanint(np.round(daily_walk_avg))
    daily_walk_med = nanint(np.round(daily_walk_med))
    daily_walk_min = nanint(np.round(daily_walk_min))
    daily_walk_max = nanint(np.round(daily_walk_max))
    daily_ptile_at_avg = daily_ptile_at_avg.map(_tdelta_to_str)

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
        'daily_ptile_at_avg': daily_ptile_at_avg,
    }


def summarize_cadence(Y, steptol=3, exclude_wear_below=None, exclude_first_last=None, adjust_estimates=False):
    """ Summarize cadence data """

    # TODO: split walking and running cadence?

    if exclude_first_last is not None:
        Y = exclude_first_last_days(Y, exclude_first_last)

    if exclude_wear_below is not None:
        Y = exclude_wear_below_days(Y, exclude_wear_below)

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

    dt = infer_freq(Y.index).total_seconds()
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
            day_of_week_cadence_peak1 = impute_days(daily_cadence_peak1, method='median').groupby(daily_cadence_peak1.index.weekday).median()
            day_of_week_cadence_peak30 = impute_days(daily_cadence_peak30, method='median').groupby(daily_cadence_peak30.index.weekday).median()
            day_of_week_cadence_p95 = impute_days(daily_cadence_p95, method='median').groupby(daily_cadence_p95.index.weekday).median()

            cadence_peak1 = day_of_week_cadence_peak1.median()
            cadence_peak30 = day_of_week_cadence_peak30.median()
            cadence_p95 = day_of_week_cadence_p95.median()

        else:
            cadence_peak1 = daily_cadence_peak1.median()
            cadence_peak30 = daily_cadence_peak30.median()
            cadence_p95 = daily_cadence_p95.median()

    daily = pd.concat([
        daily_cadence_peak1.round().astype(pd.Int64Dtype()),
        daily_cadence_peak30.round().astype(pd.Int64Dtype()),
        daily_cadence_p95.round().astype(pd.Int64Dtype()),
    ], axis=1)

    return {
        'daily': daily,
        'cadence_peak1': nanint(np.round(cadence_peak1)),
        'cadence_peak30': nanint(np.round(cadence_peak30)),
        'cadence_p95': nanint(np.round(cadence_p95)),
    }


def exclude_wear_below_days(x: pd.Series, min_wear: str):
    """ Exclude days with less than `min_wear` of valid data """

    min_wear = pd.Timedelta(min_wear)
    dt = infer_freq(x.index)
    ok = x.notna()
    if isinstance(ok, pd.DataFrame):
        ok = ok.all(axis=1)
    ok = (
        ok
        .groupby(x.index.date)
        .sum() * dt
        >= min_wear
    )
    # keep ok days, rest is set to NaN
    x = x.copy()  # make a copy to avoid modifying the original data
    x[np.isin(x.index.date, ok[~ok].index)] = np.nan
    return x


def exclude_first_last_days(x: pd.Series, first_or_last='both'):
    """ Exclude first day, last day, or both """

    x = x.copy()  # make a copy to avoid modifying the original data
    if first_or_last == 'first':
        x[x.index.date == x.index.date[0]] = np.nan
    elif first_or_last == 'last':
        x[x.index.date == x.index.date[-1]] = np.nan
    elif first_or_last == 'both':
        x[(x.index.date == x.index.date[0]) | (x.index.date == x.index.date[-1])] = np.nan
    return x


def impute_missing(data: pd.DataFrame, extrapolate=True, skip_full_missing_days=True):

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

    def impute(data):
        return (
            data
            # first attempt imputation using same day of week
            .groupby([data.index.weekday, data.index.hour, data.index.minute // 5])
            .transform(fillna)
            # then try within weekday/weekend
            .groupby([data.index.weekday >= 5, data.index.hour, data.index.minute // 5])
            .transform(fillna)
            # finally, use all other days
            .groupby([data.index.hour, data.index.minute // 5])
            .transform(fillna)
        )

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

    if skip_full_missing_days:
        # find ok days
        ok = data.notna().groupby(data.index.date).any()
        ok = np.isin(data.index.date, ok[ok].index)
        # impute only on ok days
        data.loc[ok] = impute(data.loc[ok])
    else:
        data = impute(data)

    return data


def impute_days(x, method='mean'):

    if x.isna().all():
        return x

    def fillna(x):
        if method == 'mean':
            return x.fillna(x.mean())
        elif method == 'median':
            return x.fillna(x.median())
        else:
            raise ValueError(f"Unknown method: {method}")

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        return (
            x
            .groupby(x.index.weekday).transform(fillna)
            .groupby(x.index.weekday >= 5).transform(fillna)
            .transform(fillna)
        )


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
