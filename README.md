# stepcount

A Python package to estimate step counts from accelerometer data.

The algorithm is tuned for wrist-worn AX3 data collected at 100 Hz, matching the [UK Biobank Accelerometer Dataset](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169649).


## Getting started

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

# Or a CSV file
$ stepcount sample.csv

# Or an ActiGraph file
$ stepcount sample.gt3x

# Or a GENEActiv file
$ stepcount sample.bin
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

Daily step count
----------------
              steps
time
2013-10-21   5368.0
2013-10-22   7634.0
2013-10-23  10009.0
...

Output: outputs/sample/
```

#### Output files
By default, output files will be stored in a folder `outputs/{filename}/` created in the current working directory. You can change the output path with the `-o` flag:

```console
$ stepcount sample.cwa -o /path/to/some/folder/
```

#### Processing CSV files
If a CSV file is provided, it must have the following header: `time`, `x`, `y`, `z`. For example:

```console
time,x,y,z
2013-10-21 10:00:08.000,-0.078923,0.396706,0.917759
2013-10-21 10:00:08.010,-0.094370,0.381479,0.933580
2013-10-21 10:00:08.020,-0.094370,0.366252,0.901938
2013-10-21 10:00:08.030,-0.078923,0.411933,0.901938
...
```
