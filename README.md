# stepcount
Development and implementation of a step counting algorithm using Biobank Accelerometer Analysis Tool
5 second epoch classifications, and step counting associated raw accelerometer data. This model configuration is 
specifically tuned for wrist-worn AX3 data collected at 100 Hz. 

To run:

1. Follow usage instructions for processing accelerometer data via the Biobank Accelerometer Analysis Tool:
https://biobankaccanalysis.readthedocs.io/en/latest/usage.html#processing-multiple-files

using the following processing options:
cmdOptions="--activityModel = "/path_to_model/small_steps-aug2021.tar" --epochPeriod 5"

2. Process cwa file in conjunction with time series output from Biobank Accelerometer Analysis Tool

python stepcount.py ".cwa_file_to_process" "timeSeries_csv_from_BBAT_classification" --sampleRate
