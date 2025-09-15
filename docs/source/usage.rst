Getting started
=====

Prerequisite 
------------

- Python 3.8–3.10
    .. code-block:: console

        $ python --version  # or python3 --version

- Java 8 (1.8.0) or greater
    .. code-block:: console

        $ java -version


Installation
------------

.. code-block:: console

    $ pip install stepcount


Usage
-------------

.. code-block:: console

    # Process an AX3 file
    $ stepcount sample.cwa

    # Or an ActiGraph file
    $ stepcount sample.gt3x

    # Or a GENEActiv file
    $ stepcount sample.bin

    # Or a CSV file (see data format below)
    $ stepcount sample.csv


Output:

.. code-block:: console

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

See also :doc:`api` for more options.


Output files
..................
By default, output files will be stored in a folder named after the input file,
:code:`outputs/{filename}/`, created in the current working directory. You can
change the output path with the :code:`-o` flag:

.. code-block:: console

    $ stepcount sample.cwa -o /path/to/some/folder/

The following files are written (CSV files are gzipped):

- *Info.json* High-level summary and metrics.
- *Steps.csv.gz* Per-window step counts (10 s windows for SSL).
- *StepTimes.csv.gz* Timestamps of each detected step.
- *Minutely.csv.gz* Minute-level steps and ENMO (mg).
- *MinutelyAdjusted.csv.gz* Minute-level after time-of-day imputation.
- *Hourly.csv.gz* Hourly steps and ENMO (mg).
- *HourlyAdjusted.csv.gz* Hourly after time-of-day imputation.
- *Daily.csv.gz* Daily metrics (steps, walking mins, step-percentile times, cadence peaks, ENMO).
- *DailyAdjusted.csv.gz* Daily metrics after time-of-day imputation.
- *Bouts.csv.gz* Walking bouts with duration, steps, cadence stats, ENMO.
- *Steps.png* Per-day plot of steps/min; missing shaded.

Notes
-----
- All CSV files are compressed as ``.csv.gz``.
- Window length for SSL is 10 s.
- Adjusted outputs apply time-of-day imputation and wear-time thresholds (defaults: day 21 h, hour 50 min, minute 30 s). Short recordings may contain many NaNs.

Crude vs. Adjusted Estimates
..................
Adjusted estimates are provided that account for missing data.
Missing values in the time-series are imputed with the mean of the same timepoint of other available days.
For adjusted totals and daily statistics, 24h multiples are needed and will be imputed if necessary.
Estimates will be NaN where data is still missing after imputation.

Collating outputs
..................
You can collate outputs from multiple runs into a single directory of summary CSVs:

.. code-block:: console

    $ stepcount-collate-outputs outputs/

This writes ``collated-outputs/`` containing:

- ``Info.csv.gz`` from all ``*-Info.json`` files
- ``Daily.csv.gz``, ``Hourly.csv.gz``, ``Minutely.csv.gz``, and ``Bouts.csv.gz`` collated from matching files

Processing CSV files
..................
If a CSV file is provided, it must have the following header: :code:`time`, :code:`x`, :code:`y`, :code:`z`. 

Example:

.. code-block:: console

    time,x,y,z
    2013-10-21 10:00:08.000,-0.078923,0.396706,0.917759
    2013-10-21 10:00:08.010,-0.094370,0.381479,0.933580
    2013-10-21 10:00:08.020,-0.094370,0.366252,0.901938
    2013-10-21 10:00:08.030,-0.078923,0.411933,0.901938
    ...
