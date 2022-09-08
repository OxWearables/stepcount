import pathlib
import urllib
import shutil
import time
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import actipy

from stepcount import __model_version__

MODEL_PATH = pathlib.Path(__file__).parent / f"{__model_version__}.joblib"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Enter location the file to be processed")
    args = parser.parse_args()

    # Computational timing
    start = time.time()

    # Load file
    data, info = read(args.filepath)

    # Output paths
    outdir, basename, _ = resolve_path(args.filepath)

    # Run model
    model = load_model()
    window_sec = model.window_sec
    print("Splitting data into windows...")
    X, T = make_windows(data, window_sec=window_sec)
    print("Running step counter...")
    Y = model.predict(X)
    Y = pd.DataFrame({'steps': Y}, index=T)
    Y.index.name = 'time'

    # Total walking mins
    total_walking = (Y > 0).sum().item() * window_sec / 60  # minutes
    info['TotalWalking(min)'] = total_walking

    # Total steps
    total_steps = int(Y.sum().item())
    info['TotalSteps'] = total_steps

    # All steps
    steps_outpath = f"{outdir}/{basename}_Steps.csv"
    Y.to_csv(steps_outpath)

    # Sum steps by hour
    hourly = Y.resample('H').sum()
    hourly_outpath = f"{outdir}/{basename}_HourlySteps.csv"
    hourly.to_csv(hourly_outpath)

    # Sum steps by day
    daily = Y.resample('D').sum()
    daily_outpath = f"{outdir}/{basename}_DailySteps.csv"
    daily.to_csv(daily_outpath)

    # Output info
    info_outpath = f"{outdir}/{basename}_Info.json"

    with open(info_outpath, 'w') as f:
        json.dump(info, f, indent=4, cls=NpEncoder)

    # Print
    print("\nSummary\n-------")
    print(json.dumps(info, indent=4, cls=NpEncoder))
    print("\nDaily step count\n----------------")
    print(daily)

    # Computational Timing
    end = time.time()
    print(f"Done! ({round(end - start,2)}s)")


def make_windows(data, window_sec, n_jobs=1, verbose=True):  # Note: n_jobs>1 not working
    """ Split data into windows """

    X = np.asarray(
        Parallel(n_jobs=n_jobs)(
            delayed(extract_xyz)(w)
            for i, w in tqdm(data.resample(f"{window_sec}s"), disable=not verbose)
        ),
        dtype='object'
    )

    T = (
        data.index
        .to_series()
        .resample(f"{window_sec}s")
        .nearest()
    )

    return X, T


def extract_xyz(w):
    return w[['x', 'y', 'z']].to_numpy()


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
    parent = p.parent
    return parent, filename, extension


def load_model():
    """ Load trained model. Download if not exists. """

    pth = pathlib.Path(MODEL_PATH)

    if not pth.exists():

        url = f"https://wearables-files.ndph.ox.ac.uk/files/models/stepcounter/{__model_version__}.joblib"

        print(f"Downloading {url}...")

        with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)

    return joblib.load(pth)



if __name__ == '__main__':
    main()
