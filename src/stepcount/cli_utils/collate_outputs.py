import argparse
import json
from collections import OrderedDict

import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


def collate_outputs(
    results_dir,
    collated_results_dir="collated_outputs/",
    included=["daily", "hourly", "minutely", "bouts"],
):
    """Collate all results files in <results_dir>.
    :param str results_dir: Root directory in which to search for result files.
    :param str collated_results_dir: Directory to write the collated files to.
    :param list included: Type of result files to collate ('daily', 'hourly', 'minutely', 'bouts').
    :return: Collated files written to <collated_results_dir>
    :rtype: void
    """

    print("Searching files...")

    # Find all relevant files under <results_dir>/
    # - *-Info.json files contain the summary information
    # - *-Daily.json files contain daily summaries
    # - *-Hourly.json files contain hourly summaries
    # - *-Minutely.json files contain minute-level summaries
    # - *-Bouts.json files contain bout information

    info_files = []
    csv_files = {}

    # lowercase the include list
    included  = [x.lower() for x in included]
    if "daily" in included:
        csv_files["Daily"] = []
        csv_files["DailyAdjusted"] = []
    if "hourly" in included:
        csv_files["Hourly"] = []
        csv_files["HourlyAdjusted"] = []
    if "minutely" in included:
        csv_files["Minutely"] = []
        csv_files["MinutelyAdjusted"] = []
    if "bouts" in included:
        csv_files["Bouts"] = []

    # Iterate through the files and append to the appropriate list based on the suffix
    for file in Path(results_dir).rglob('*'):
        if file.is_file():
            if file.name.endswith("-Info.json"):
                info_files.append(file)
            for key, file_list in csv_files.items():
                if file.name.endswith(f"-{key}.csv.gz"):
                    file_list.append(file)
                    break

    collated_results_dir = Path(collated_results_dir) 
    collated_results_dir.mkdir(parents=True, exist_ok=True)

    # Collate Info.json files
    print(f"Collating {len(info_files)} Info files...")
    outfile = collated_results_dir / "Info.csv.gz"
    collate_jsons(info_files, outfile)
    print('Collated info CSV written to', outfile)

    # Collate the remaining files (Daily, Hourly, Minutely, Bouts, etc.)
    for key, file_list in csv_files.items():
        print(f"Collating {len(file_list)} {key} files...")
        outfile = collated_results_dir / f"{key}.csv.gz"
        collate_csvs(file_list, outfile)
        print(f'Collated {key} CSV written to', outfile)

    return


def collate_jsons(file_list, outfile, overwrite=True):
    """ Collate a list of JSON files into a single CSV file."""

    if overwrite and outfile.exists():
        print(f"Overwriting existing file: {outfile}")
        outfile.unlink()  # remove existing file

    df = []
    for file in tqdm(file_list):
        with open(file, 'r') as f:
            df.append(json.load(f, object_pairs_hook=OrderedDict))
    df = pd.DataFrame.from_dict(df)  # merge to a dataframe
    df = df.applymap(convert_ordereddict)  # convert any OrderedDict cell values to regular dict
    df.to_csv(outfile, index=False)

    return


def collate_csvs(file_list, outfile, overwrite=True):
    """ Collate a list of CSV files into a single CSV file."""

    if overwrite and outfile.exists():
        print(f"Overwriting existing file: {outfile}")
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
    parser.add_argument('results_dir', help="Root directory in which to search for result files")
    parser.add_argument('--output', '-o', default="collated-outputs/", help="Directory to write the collated files to")
    parser.add_argument('--include', '-i', nargs='+', default=["daily", "hourly", "minutely", "bouts"], help="Type of result files to collate ('daily', 'hourly', 'minutely', 'bouts')")
    args = parser.parse_args()

    return collate_outputs(
        results_dir=args.results_dir,
        collated_results_dir=args.output,
        included=args.include,
    )



if __name__ == '__main__':
    main()
