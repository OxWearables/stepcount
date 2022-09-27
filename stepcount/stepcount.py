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

import actipy

from stepcount import __model_version__

MODEL_PATH = pathlib.Path(__file__).parent / f"{__model_version__}.joblib.lzma"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Enter file location to be processed")
    parser.add_argument("--outdir", "-o", help="Enter folder location to save output files", default="outputs/")
    parser.add_argument("--model_path", "-m", help="Enter file location to custom model to use", default=None)
    args = parser.parse_args()

    # Computational timing
    start = time.time()

    # Load file
    data, info = read(args.filepath)

    # Output paths
    basename = resolve_path(args.filepath)[1]
    outdir = os.path.join(args.outdir, basename)
    os.makedirs(outdir, exist_ok=True)

    # Run model
    model = load_model(args.model_path or MODEL_PATH)
    window_sec = model.window_sec
    print("Running step counter...")
    Y = model.predict_from_frame(data)

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
