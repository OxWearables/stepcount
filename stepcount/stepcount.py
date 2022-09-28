import os
import pathlib
import urllib
import shutil
import time
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from pandas.tseries.frequencies import to_offset

import actipy

from stepcount import __model_version__

MODEL_PATH = pathlib.Path(__file__).parent / f"{__model_version__}.joblib.lzma"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Enter file location to be processed")
    parser.add_argument("--outdir", "-o", help="Enter folder location to save output files", default="outputs/")
    parser.add_argument("--model_path", "-m", help="Enter file location to custom model to use", default=None)
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
    model = load_model(args.model_path or MODEL_PATH)
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
    Y = impute_missing(Y)
    summary = summarize(Y)
    summary['hourly'].to_csv(f"{outdir}/{basename}_HourlyStepsAdjusted.csv")
    summary['daily'].to_csv(f"{outdir}/{basename}_DailyStepsAdjusted.csv")
    info['TotalStepsAdjusted'] = summary['total']
    info['TotalWalkingAdjusted(min)'] = summary['total_walk']

    # Save info
    with open(f"{outdir}/{basename}_Info.json", 'w') as f:
        json.dump(info, f, indent=4, cls=NpEncoder)

    # Print
    print("\nSummary\n-------")
    print(json.dumps(info, indent=4, cls=NpEncoder))
    print("\nEstimated daily step count\n--------------------------")
    print(summary['daily'])  # adjusted

    # Timing
    end = time.time()
    print(f"Done! ({round(end - start,2)}s)")


def summarize(Y):

    total = int(np.round(Y.sum()))  # total steps
    hourly = Y.resample('H').sum().round().astype('int')  # steps, hourly
    daily = Y.resample('D').sum().round().astype('int')  # steps, daily
    total_walk = float(  # total walk (mins)
        (pd.Timedelta(pd.infer_freq(Y.index)) * (Y > 0).sum())
        .total_seconds() / 60
    )

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
                freq=to_offset(pd.infer_freq(data.index)),
                closed='left',
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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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

        # TODO: assert columns and index OK
        # TODO: process with actipy

        info = {
            "Filename": filepath,
            "Device": ftype,
            "Filesize(MB)": fsize,
            "NumTicks": len(data),
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


def resolve_path(path):
    """ Return parent folder, file name and file extension """
    p = pathlib.Path(path)
    extension = p.suffixes[0]
    filename = p.name.rsplit(extension)[0]
    dirname = p.parent
    return dirname, filename, extension


def load_model(model_path=MODEL_PATH):
    """ Load trained model. Download if not exists. """

    pth = pathlib.Path(model_path)

    if not pth.exists():

        # url = f"https://wearables-files.ndph.ox.ac.uk/files/models/stepcounter/{__model_version__}.joblib.lzma"
        url = "https://tinyurl.com/34dswhss"  # 20220921.joblib.lzma

        print(f"Downloading {url}...")

        with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)

    return joblib.load(pth)



if __name__ == '__main__':
    main()
