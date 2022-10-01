# stepcount

A Python package to estimate step counts from accelerometer data.

The algorithm is tuned for wrist-worn AX3 data collected at 100 Hz, making it compatible with the [UK Biobank Accelerometer Dataset](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169649).


## Getting started

### Prerequisite

- Python 3.8 or greater
    ```console
    $ python --version  # or python3 --version
    ```

- Java 8 (1.8.0) or greater
    ```console
    $ java -version
    ```


### Install

<!-- ```console
$ pip install git+https://github.com/OxWearables/stepcount.git@master#egg=stepcount
``` -->

<!-- ```console
$ pip install git+ssh://git@github.com/OxWearables/stepcount.git@master#egg=stepcount
``` -->

```console
$ pip install stepcount
```

### Usage

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

#### Output files
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

#### Crude vs. Adjusted Estimates
Adjusted estimates are provided that account for missing data.
Missing values in the time-series are imputed with the mean of the same timepoint of other available days.
For adjusted totals and daily statistics, 24h multiples are needed and will be imputed if necessary.
Estimates will be NaN where data is still missing after imputation.

#### Processing CSV files
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
