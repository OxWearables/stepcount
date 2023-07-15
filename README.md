# stepcount

A Python package to estimate step counts from accelerometer data.

The algorithm is tuned for wrist-worn AX3 data collected at 100 Hz, using data from the open-source [OxWalk Dataset](https://ora.ox.ac.uk/objects/uuid:19d3cb34-e2b3-4177-91b6-1bad0e0163e7), making it compatible with the [UK Biobank Accelerometer Dataset](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169649). 

## Install

We recommend using **Anaconda**:

1. Download & install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (light-weight version of Anaconda).
1. (Windows) Once installed, launch the **Anaconda Prompt**.
1. Create a virtual environment:
    ```console
    $ conda create -n stepcount python=3.9 openjdk pip
    ```
    This creates a virtual environment called `stepcount` with Python version 3.9, OpenJDK, and Pip.
1. Activate the environment:
    ```console
    $ conda activate stepcount
    ```
    You should now see `(stepcount)` written in front of your prompt.
1. Install `stepcount`:
    ```console
    $ pip install stepcount
    ```

You are all set! The next time that you want to use `stepcount`, open the Anaconda Prompt and activate the environment (step 4). If you see `(stepcount)` in front of your prompt, you are ready to go!

Check out the 5-minute video tutorial to get started: [https://www.youtube.com/watch?v=FPb7H-jyRVQ](https://www.youtube.com/watch?v=FPb7H-jyRVQ).

## Usage

```bash
# Process an AX3 file
$ stepcount sample.cwa

# Or an ActiGraph file
$ stepcount sample.gt3x

# Or a GENEActiv file
$ stepcount sample.bin

# Or a CSV file (see data format below)
$ stepcount sample.csv
```

Output:
```console
Summary
-------
{
    "Filename": "sample.cwa",
    "Filesize(MB)": 65.1,
    "Device": "Axivity",
    "DeviceID": 2278,
    "ReadErrors": 0,
    "SampleRate": 100.0,
    "ReadOK": 1,
    "StartTime": "2013-10-21 10:00:07",
    "EndTime": "2013-10-28 10:00:01",
    "TotalWalking(min)": 655.75,
    "TotalSteps": 43132,
    ...
}

Estimated Daily Steps
---------------------
              steps
time
2013-10-21     5368
2013-10-22     7634
2013-10-23    10009
...

Output: outputs/sample/
```

### Troubleshooting 
Some systems may face issues with Java when running the script. If this is your case, try fixing OpenJDK to version 8:
```console
$ conda install -n stepcount openjdk=8
```

### Output files
By default, output files will be stored in a folder named after the input file, `outputs/{filename}/`, created in the current working directory. You can change the output path with the `-o` flag:

```console
$ stepcount sample.cwa -o /path/to/some/folder/
```

The following output files are created:

- *Info.json* Summary info, as shown above.
- *Steps.csv* Raw time-series of step counts
- *HourlySteps.csv* Hourly step counts
- *DailySteps.csv* Daily step counts
- *HourlyStepsAdjusted.csv* Like HourlySteps but accounting for missing data (see section below).
- *DailyStepsAdjusted.csv* Like DailySteps but accounting for missing data (see section below).


### Machine learning model type
By default, the `stepcount` tool employs a self-supervised Resnet18 model to detect walking periods.
However, it is possible to switch to a random forest model, by using the `-t` flag:

```console
$ stepcount sample.cwa -t rf
```

When using the random forest model, a set of signal features is extracted from the accelerometer data. 
These features are subsequently used as inputs for the model's classification process. 
For a comprehensive list of the extracted features, please refer to [rf-feature-list.md](rf-feature-list.md).


### Crude vs. Adjusted Estimates
Adjusted estimates are provided that account for missing data.
Missing values in the time-series are imputed with the mean of the same timepoint of other available days.
For adjusted totals and daily statistics, 24h multiples are needed and will be imputed if necessary.
Estimates will be NaN where data is still missing after imputation.


### Processing CSV files
If a CSV file is provided, it must have the following header: `time`, `x`, `y`, `z`. 

Example:
```console
time,x,y,z
2013-10-21 10:00:08.000,-0.078923,0.396706,0.917759
2013-10-21 10:00:08.010,-0.094370,0.381479,0.933580
2013-10-21 10:00:08.020,-0.094370,0.366252,0.901938
2013-10-21 10:00:08.030,-0.078923,0.411933,0.901938
...
```

### Processing multiple files
#### Windows
To process multiple files you can create a text file in Notepad which includes one line for each file you wish to process, as shown below for *file1.cwa*, *file2.cwa*, and *file2.cwa*.

Example text file *commands.txt*: 
```console
stepcount file1.cwa &
stepcount file2.cwa &
stepcount file3.cwa 
:END
````
Once this file is created, run `cmd < commands.txt` from the terminal.

#### Linux
Create a file *command.sh* with:
```console
stepcount file1.cwa
stepcount file2.cwa
stepcount file3.cwa
```
Then, run `bash command.sh` from the terminal.

#### Collating outputs

A utility script is provided to collate outputs from multiple runs:

```console
$ stepcount-collate-outputs outputs/
```
This will collate all *-Info.json files found in outputs/ and generate a CSV file.

## Validation

Validation for this algorithm is presented in a preprint on medRxiv at: [https://www.medrxiv.org/content/10.1101/2023.02.20.23285750v1](https://www.medrxiv.org/content/10.1101/2023.02.20.23285750v1). 


## Citing our work

When using this tool, please consider citing the works listed in [CITATION.md](CITATION.md).


## Licence
See [LICENSE.md](LICENSE.md).


## Acknowledgements
We would like to thank all our code contributors, manuscript co-authors, and research participants for their help in making this work possible.
