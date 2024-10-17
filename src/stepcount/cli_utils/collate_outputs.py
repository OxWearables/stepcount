import argparse
import json
import os
from collections import OrderedDict

import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


def collate_outputs(
    results,
    include_hourly=False,
    include_minutely=False,
    include_bouts=False,
    outdir="collated_outputs/"
):
    """Collate all output files under <outdir> into one CSV file.
    :param str outdir: Root directory from which to search for output files.
    :param str outfile: Output CSV filename.
    :return: New file written to <outfile>
    :rtype: void
    """

    os.makedirs(outdir, exist_ok=True)

    # Find all relevant files under <outputs>/
    # - *-Info.json files contain the summary information
    # - *-Daily.json files contain daily summaries
    # - *-Hourly.json files contain hourly summaries
    # - *-Minutely.json files contain minute-level summaries
    # - *-Bouts.json files contain bout information
    info_files = []
    daily_files = []
    hourly_files = []
    minutes_files = []
    dailyadj_files = []
    hourlyadj_files = []
    minutesadj_files = []
    bouts_files = []

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
            if file.name.endswith("-Bouts.csv.gz"):
                bouts_files.append(file)

    outdir = Path(outdir) 

    print(f"Collating {len(info_files)} summary files...")
    info = []
    for file in tqdm(info_files):
        with open(file, 'r') as f:
            info.append(json.load(f, object_pairs_hook=OrderedDict))
    info = pd.DataFrame.from_dict(info)  # merge to a dataframe
    info = info.applymap(convert_ordereddict)  # convert any OrderedDict cell values to regular dict
    info_file = outdir / "Info.csv.gz" 
    info.to_csv(info_file, index=False)
    print('Collated info CSV written to', info_file)

    print(f"Collating {len(daily_files)} daily files...")
    daily_csv = outdir / "Daily.csv.gz"
    collate_to_csv(daily_files, daily_csv)
    print('Collated daily CSV written to', daily_csv)

    print(f"Collating {len(dailyadj_files)} adjusted daily files...")
    dailyadj_csv = outdir / "DailyAdjusted.csv.gz"
    collate_to_csv(dailyadj_files, dailyadj_csv)
    print('Collated adjusted daily CSV written to', dailyadj_csv)

    if include_hourly:

        print(f"Collating {len(hourly_files)} hourly files...")
        hourly_csv = outdir / "Hourly.csv.gz"
        collate_to_csv(hourly_files, hourly_csv)
        print('Collated hourly CSV written to', hourly_csv)

        print(f"Collating {len(hourlyadj_files)} adjusted hourly files...")
        hourlyadj_csv = outdir / "HourlyAdjusted.csv.gz"
        collate_to_csv(hourlyadj_files, hourlyadj_csv)
        print('Collated adjusted hourly CSV written to', hourlyadj_csv)

    if include_minutely:

        print(f"Collating {len(minutes_files)} minutes files...")
        minutes_csv = outdir / "Minutely.csv.gz"
        collate_to_csv(minutes_files, minutes_csv)
        print('Collated minutes CSV written to', minutes_csv)

        print(f"Collating {len(minutesadj_files)} adjusted minutes files...")
        minutesadj_csv = outdir / "MinutelyAdjusted.csv.gz"
        collate_to_csv(minutesadj_files, minutesadj_csv)
        print('Collated adjusted minutes CSV written to', minutesadj_csv)

    if include_bouts:

        print(f"Collating {len(bouts_files)} bouts files...")
        bouts_csv = outdir / "Bouts.csv.gz"
        collate_to_csv(bouts_files, bouts_csv)
        print('Collated bouts CSV written to', bouts_csv)

    return


def collate_to_csv(file_list, outfile, overwrite=True):
    """ Collate a list of files into a single CSV file."""

    if overwrite and outfile.exists():
        outfile.unlink()  # remove existing file

    header_written = False
    for file in tqdm(file_list):
        df = pd.read_csv(file)
        df.to_csv(outfile, mode='a', index=False, header=not header_written)
        header_written = True

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
    parser.add_argument('--include-bouts', action='store_true', help="Collate bouts files")
    parser.add_argument('--outdir', '-o', default="collated-outputs/", help="Output directory")
    args = parser.parse_args()

    return collate_outputs(
        results=args.results,
        include_hourly=args.include_hourly,
        include_minutely=args.include_minutely,
        include_bouts=args.include_bouts,
        outdir=args.outdir
    )



if __name__ == '__main__':
    main()
