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

MODEL_PATH = pathlib.Path(__file__).parent / f"{__model_version__}.joblib.lzma"


def main():

    parser = argparse.ArgumentParser(
        description="A tool to estimate step counts from accelerometer data",
        add_help=True
    )
    parser.add_argument("filepath", help="Enter file to be processed")
    parser.add_argument("--outdir", "-o", help="Enter folder location to save output files", default="outputs/")
    parser.add_argument("--model_path", "-m", help="Enter custom model file to use", default=None)
    parser.add_argument("--force-download", action="store_true", help="Force download of model file")
    args = parser.parse_args()

    # Timing
    start = time.time()

    # Load file
    data, info = read(args.filepath)

    # Output paths
    basename = resolve_path(args.filepath)[1]
    outdir = os.path.join(args.outdir, basename)
    os.makedirs(outdir, exist_ok=True)

    # Run model
    print("Running step counter...")
    model = load_model(args.model_path or MODEL_PATH, args.force_download)
    # TODO: implement reset_sample_rate()
    model.sample_rate = info['SampleRate']
    model.window_len = int(np.ceil(info['SampleRate'] * model.window_sec))
    model.wd.sample_rate = info['SampleRate']
    Y = model.predict_from_frame(data)

    # Save raw output timeseries
    Y.to_csv(f"{outdir}/{basename}_Steps.csv")

    # Summary
    summary = summarize(Y)
    summary['hourly'].to_csv(f"{outdir}/{basename}_HourlySteps.csv")
    summary['daily'].to_csv(f"{outdir}/{basename}_DailySteps.csv")
    info['TotalSteps'] = summary['total']
    info['TotalWalking(min)'] = summary['total_walk']

    # Impute missing periods & recalculate summary
    summary_adj = summarize(Y, adjust_estimates=True)
    summary_adj['hourly'].to_csv(f"{outdir}/{basename}_HourlyStepsAdjusted.csv")
    summary_adj['daily'].to_csv(f"{outdir}/{basename}_DailyStepsAdjusted.csv")
    info['TotalStepsAdjusted'] = summary_adj['total']
    info['TotalWalkingAdjusted(min)'] = summary_adj['total_walk']

    # Save info
    with open(f"{outdir}/{basename}_Info.json", 'w') as f:
        json.dump(info, f, indent=4, cls=NpEncoder)

    # Print
    print("\nSummary\n-------")
    print(json.dumps(info, indent=4, cls=NpEncoder))
    print("\nEstimated Daily Steps\n---------------------")
    print(pd.concat([
        summary['daily'].rename('Crude'), 
        summary_adj['daily'].rename('Adjusted')
    ], axis=1))

    # Timing
    end = time.time()
    print(f"Done! ({round(end - start,2)}s)")


def summarize(Y, adjust_estimates=False):

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

    total = np.round(Y.agg(_sum))  # total steps
    hourly = Y.resample('H').agg(_sum).round()  # steps, hourly
    daily = Y.resample('D').agg(_sum).round()  # steps, daily
    total_walk = (  # total walk (mins)
        (pd.Timedelta(infer_freq(Y.index)) * Y.mask(~Y.isna(), Y > 0).agg(_sum))
        .total_seconds() / 60
    )

    total = nanint(total)
    hourly = pd.to_numeric(hourly, downcast='integer')
    daily = pd.to_numeric(daily, downcast='integer')
    total_walk = float(total_walk)

    return {
        'total': total,
        'hourly': hourly,
        'daily': daily,
        'total_walk': total_walk,
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


def read(filepath):

    p = pathlib.Path(filepath)
    ftype = p.suffixes[0].upper()
    fsize = round(p.stat().st_size / (1024 * 1024), 1)

    if ftype in (".CSV", ".PKL"):

        if ftype == ".CSV":
            data = pd.read_csv(
                filepath,
                usecols=['time', 'x', 'y', 'z'],
                parse_dates=['time'],
                index_col='time'
            )
        elif ftype == ".PKL":
            data = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unknown file format: {ftype}")

        freq = infer_freq(data.index)
        sample_rate = int(np.round(pd.Timedelta('1s') / freq))

        data, info = actipy.process(
            data, sample_rate,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz='uniform',
        )

        info = {
            **{"Filename": filepath,
                "Device": ftype,
                "Filesize(MB)": fsize,
                "SampleRate": sample_rate},
            **info
        }

    elif ftype in (".CWA", ".GT3X", ".BIN"):

        data, info = actipy.read_device(
            filepath,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz='uniform',
        )

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


def load_model(model_path=MODEL_PATH, force_download=False):
    """ Load trained model. Download if not exists. """

    pth = pathlib.Path(model_path)

    if force_download or not pth.exists():

        # url = f"https://wearables-files.ndph.ox.ac.uk/files/models/stepcounter/{__model_version__}.joblib.lzma"
        url = "https://tinyurl.com/34dswhss"  # 20220921.joblib.lzma

        print(f"Downloading {url}...")

        with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)

    assert md5(pth) == __model_md5__, (
        "Model file is corrupted. Please run again with --force-download "
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
