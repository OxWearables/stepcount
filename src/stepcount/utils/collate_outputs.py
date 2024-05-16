import argparse
import json
import os
from collections import OrderedDict

import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm



def collate_outputs(results, include_hourly=False, include_minutely=False, outdir="collated_outputs/"):
    """Read all *-Info.json files under <outputs> and merge into one CSV file.
    :param str outputs: Directory containing JSON files.
    :param str outfile: Output CSV filename.
    :return: New file written to <outfile>
    :rtype: void
    """

    os.makedirs(outdir, exist_ok=True)
    
    # Find all relevant files under <outputs>/
    # - *-Info.json files contain the summary information
    # - *-Daily.json files contain daily summaries
    # - *-Hourly.json files contain hourly summaries
    # - *-Minutely.json files contain minute-by-minute summaries
    info_files = []
    daily_files = []
    hourly_files = []
    minutes_files = []
    dailyadj_files = []
    hourlyadj_files = []
    minutesadj_files = []

    results = Path(results)

    print("Searching files...")

    for file in results.rglob('*'):
        if file.is_file():
            if file.name.endswith("-Info.json"):
                info_files.append(file)
            if file.name.endswith("-Daily.csv.gz"):
                daily_files.append(file)
            if file.name.endswith("-Hourly.csv.gz"):
                hourly_files.append(file)
            if file.name.endswith("-Minutely.csv.gz"):
                minutes_files.append(file)
            if file.name.endswith("-DailyAdjusted.csv.gz"):
                dailyadj_files.append(file)
            if file.name.endswith("-HourlyAdjusted.csv.gz"):
                hourlyadj_files.append(file)
            if file.name.endswith("-MinutelyAdjusted.csv.gz"):
                minutesadj_files.append(file)

    outdir = Path(outdir) 

    print(f"Found {len(info_files)} summary files...")
    info = []
    for file in tqdm(info_files):
        with open(file, 'r') as f:
            info.append(json.load(f, object_pairs_hook=OrderedDict))
    info = pd.DataFrame.from_dict(info)  # merge to a dataframe
    info = info.applymap(convert_ordereddict)  # convert any OrderedDict cell values to regular dict
    info_file = outdir / "Info.csv.gz" 
    info.to_csv(info_file, index=False)
    print('Collated info CSV written to', info_file)

    print(f"Found {len(daily_files)} daily files...")
    daily_file = outdir / "Daily.csv.gz"
    header_written = False
    for file in tqdm(daily_files):
        df = pd.read_csv(file)
        df.to_csv(daily_file, mode='a', index=False, header=not header_written)
        header_written = True
    print('Collated daily CSV written to', daily_file)

    print(f"Found {len(dailyadj_files)} adjusted daily files...")
    dailyadj_file = outdir / "DailyAdjusted.csv.gz"
    header_written = False
    for file in tqdm(dailyadj_files):
        df = pd.read_csv(file)
        df.to_csv(dailyadj_file, mode='a', index=False, header=not header_written)
        header_written = True
    print('Collated adjusted daily CSV written to', dailyadj_file)

    if include_hourly:
        print(f"Found {len(hourly_files)} hourly files...")
        hourly_file = outdir / "Hourly.csv.gz"
        header_written = False
        for file in tqdm(hourly_files):
            df = pd.read_csv(file)
            df.to_csv(hourly_file, mode='a', index=False, header=not header_written)
            header_written = True
        print('Collated hourly CSV written to', hourly_file)

        print(f"Found {len(hourlyadj_files)} adjusted hourly files...")
        hourlyadj_file = outdir / "HourlyAdjusted.csv.gz"
        header_written = False
        for file in tqdm(hourlyadj_files):
            df = pd.read_csv(file)
            df.to_csv(hourlyadj_file, mode='a', index=False, header=not header_written)
            header_written = True
        print('Collated adjusted hourly CSV written to', hourlyadj_file)

    if include_minutely:

        print(f"Found {len(minutes_files)} minutes files...")
        minutes_file = outdir / "Minutely.csv.gz"
        header_written = False
        for file in tqdm(minutes_files):
            df = pd.read_csv(file)
            df.to_csv(minutes_file, mode='a', index=False, header=not header_written)
            header_written = True
        print('Collated minutes CSV written to', minutes_file)
        
        print(f"Found {len(minutesadj_files)} adjusted minutes files...")
        minutesadj_file = outdir / "MinutelyAdjusted.csv.gz"
        header_written = False
        for file in tqdm(minutesadj_files):
            df = pd.read_csv(file)
            df.to_csv(minutesadj_file, mode='a', index=False, header=not header_written)
            header_written = True
        print('Collated adjusted minutes CSV written to', minutesadj_file)

    return


def convert_ordereddict(value):
    """ Convert OrderedDict to regular dict """
    if isinstance(value, OrderedDict):
        return dict(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', help="Directory containing the result files")
    parser.add_argument('--include-hourly', action='store_true', help="Collate hourly files")
    parser.add_argument('--include-minutely', action='store_true', help="Collate minutely files")
    parser.add_argument('--outdir', '-o', default="collated-outputs/", help="Output directory")
    args = parser.parse_args()

    return collate_outputs(
        results=args.results,
        include_hourly=args.include_hourly,
        include_minutely=args.include_minutely,
        outdir=args.outdir
    )



if __name__ == '__main__':
    main()
