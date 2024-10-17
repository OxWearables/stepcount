# Glossary

## Info.json

1. General Information
    - **StepCountVersion**: Version number of the `stepcount` program used to run the analysis. This information is useful to ensure reproducibility.
    - **StepCountArgs**: Arguments passed to the `stepcount` program to run the analysis. This information is useful to ensure reproducibility.
    - **Filename**: The name of the data file.
    - **Filesize(MB)**: The size of the file in megabytes.
    - **Device**: The brand or model of the device that recorded the data, or the file extension.
    - **DeviceID**: A unique identifier for the specific device used.
    - **ReadErrors**: Number of errors encountered while reading data from the device.
    - **SampleRate**: The frequency at which data points are sampled per second, measured in Hz.
    - **ReadOK**: A binary indicator (1 for success, 0 for failure) showing whether the data was read successfully.
    - **NumTicks**: Total number of data points recorded.
    - **StartTime**: Timestamp marking the beginning of the device recording.
    - **EndTime**: Timestamp marking the end of the device recording.
    - **WearStartTime**: Timestamp marking the beginning of valid wear data.
    - **WearEndTime**: Timestamp marking the end of valid wear data.
    - **WearTime(days)**: Total time the device was worn, expressed in days.
    - **NumInterrupts**: Number of interruptions in the data recording.
    - **CalibErrorBefore(mg)**: Calibration error before any correction was applied, measured in milli-g.
    - **CalibOK**: A binary indicator (1 for successful calibration, 0 for unsuccessful) showing the calibration status.
    - **CalibErrorAfter(mg)**: Calibration error after attempting correction, measured in milli-g.
    - **NonwearTime(days)**: Total time the device was not worn, expressed in days.
    - **NumNonwearEpisodes**: Number of separate episodes when the device was not worn.
    - **ResampleRate**: The new sampling rate after data has been resampled.
    - **NumTicksAfterResample**: Number of data points after resampling.
    - **Covers24hOK**: Indicates whether data is available for each hour bin of the 24h across any day. For example, if data is absent from 14:00-15:00 on Day 1 (and all other hours are covered), and then it is available during 14:00-15:00 on Day 2, then `Covers24hOK=1`. If data is fully absent from 14:00-15:00 across all days, then `Covers24hOK=0`.
1. ENMO
    - **ENMO(mg)**: Euclidean norm minus one and zero-truncated, measured in milli-g.
    - **ENMO(mg)_Hour{XX}**: ENMO for each hour bin, where `{XX}` is a placeholder for the hour in 24-hour format. For example, `ENMO(mg)_Hour12` is the ENMO during 12:00 - 13:00, averaged across all days.
    - **ENMO(mg)_Weekend**: Euclidean norm minus one and zero-truncated, measured in milli-g. Weekends only.
    - **ENMO(mg)_Hour{XX}_Weekend**: ENMO for each hour bin, where `{XX}` is a placeholder for the hour in 24-hour format. For example, `ENMO(mg)_Hour12_Weekend` is the ENMO during 12:00 - 13:00, averaged across all days. Weekends only.
    - **ENMO(mg)_Weekday**: Euclidean norm minus one and zero-truncated, measured in milli-g. Weekdays only.
    - **ENMO(mg)_Hour{XX}_Weekday**: ENMO for each hour bin, where `{XX}` is a placeholder for the hour in 24-hour format. For example, `ENMO(mg)_Hour12_Weekend` is the ENMO during 12:00 - 13:00, averaged across all days. Weekdays only.
1. Steps
    - **TotalSteps**: Total number of steps.
    - **StepsDayAvg**: Average number of steps per day.
    - **StepsDayMed**: Median number of steps per day.
    - **StepsDayMin**: Minimum number of steps per day.
    - **StepsDayMax**: Maximum number of steps per day.
    - **Steps_Hour{XX}**: Average number of steps taken during each hour bin, where `{XX}` is a placeholder for the hour in 24-hour format. For example, `Steps_Hour12` is the average number of steps taken during 12:00 - 13:00, averaged across all days.
    <!-- weekends -->
    - **TotalSteps_Weekend**: Total number of steps. Weekends only.
    - **StepsDayAvg_Weekend**: Average number of steps per day. Weekends only.
    - **StepsDayMed_Weekend**: Median number of steps per day. Weekends only.
    - **StepsDayMin_Weekend**: Minimum number of steps per day. Weekends only.
    - **StepsDayMax_Weekend**: Maximum number of steps per day. Weekends only.
    - **Steps_Hour{XX}_Weekend**: Average number of steps taken during each hour bin, where `{XX}` is a placeholder for the hour in 24-hour format. For example, `Steps_Hour12_Weekend` is the average number of steps taken during 12:00 - 13:00, averaged across all days. Weekends only.
    <!-- weekdays -->
    - **TotalSteps_Weekday**: Total number of steps. Weekdays only.
    - **StepsDayAvg_Weekday**: Average number of steps per day. Weekdays only.
    - **StepsDayMed_Weekday**: Median number of steps per day. Weekdays only.
    - **StepsDayMin_Weekday**: Minimum number of steps per day. Weekdays only.
    - **StepsDayMax_Weekday**: Maximum number of steps per day. Weekdays only.
    - **Steps_Hour{XX}_Weekday**: Average number of steps taken during each hour bin, where `{XX}` is a placeholder for the hour in 24-hour format. For example, `Steps_Hour12_Weekday` is the average number of steps taken during 12:00 - 13:00, averaged across all days. Weekdays only.
1. Walking
    - **TotalWalking(mins)**: Total minutes spent walking, in minutes.
    - **WalkingDayAvg(mins)**: Average time spent walking per day, in minutes.
    - **WalkingDayMed(mins)**: Median time spent walking per day, in minutes.
    - **WalkingDayMin(mins)**: Minimum time spent walking per day, in minutes.
    - **WalkingDayMax(mins)**: Maximum time spent walking per day, in minutes.
    - **Walking(mins)_Hour{XX}**: Average time spent walking during each hour bin, where `{XX}` is a placeholder for the hour in 24-hour format. For example, `Walking(mins)_Hour12` is the average time spent walking during 12:00 - 13:00, averaged across all days.
    <!-- weekends -->
    - **TotalWalking(mins)_Weekend**: Total time spent walking, in minutes. Weekends only.
    - **WalkingDayAvg(mins)_Weekend**: Average time spent walking per day, in minutes. Weekends only.
    - **WalkingDayMed(mins)_Weekend**: Median time spent walking per day, in minutes. Weekends only.
    - **WalkingDayMin(mins)_Weekend**: Minimum time spent walking per day, in minutes. Weekends only.
    - **WalkingDayMax(mins)_Weekend**: Maximum time spent walking per day, in minutes. Weekends only.
    - **Walking(mins)_Hour{XX}_Weekend**: Average time spent walking during each hour bin, where `{XX}` is a placeholder for the hour in 24-hour format. For example, `Walking(mins)_Hour12_Weekend` is the average time spent walking during 12:00 - 13:00, averaged across all days. Weekends only.
    <!-- weekdays -->
    - **TotalWalking(mins)_Weekday**: Total time spent walking, in minutes. Weekdays only.
    - **WalkingDayAvg(mins)_Weekday**: Average time spent walking per day, in minutes. Weekdays only.
    - **WalkingDayMed(mins)_Weekday**: Median time spent walking per day, in minutes. Weekdays only.
    - **WalkingDayMin(mins)_Weekday**: Minimum time spent walking per day, in minutes. Weekdays only.
    - **WalkingDayMax(mins)_Weekday**: Maximum time spent walking per day, in minutes. Weekdays only.
    - **Walking(mins)_Hour{XX}_Weekday**: Average time spent walking during each hour bin, where `{XX}` is a placeholder for the hour in 24-hour format. For example, `Walking(mins)_Hour12_Weekday` is the average time spent walking during 12:00 - 13:00, averaged across all days. Weekdays only.
1. Cadence
    - **CadencePeak1(steps/min)**: Highest cadence recorded in one minute of the day, averaged across all days.
    - **CadencePeak30(steps/min)**: Mean of the 30 highest minutes of the day (not necessarily continuous), averaged across all days.
    - **Cadence95th(steps/min)**: 95th percentile of cadence of the day, averaged across all days.
    <!-- weekends -->
    - **CadencePeak1(steps/min)_Weekend**: Highest cadence recorded in one minute of the day, averaged across all days. Weekends only.
    - **CadencePeak30(steps/min)_Weekend**: Mean of the 30 highest minutes of the day (not necessarily continuous), averaged across all days. Weekends only.
    - **Cadence95th(steps/min)_Weekend**: 95th percentile of cadence of the day, averaged across all days. Weekends only.
    <!-- weekdays -->
    - **CadencePeak1(steps/min)_Weekday**: Highest cadence recorded in one minute of the day, averaged across all days. Weekdays only.
    - **CadencePeak30(steps/min)_Weekday**: Mean of the 30 highest minutes of the day (not necessarily continuous), averaged across all days. Weekdays only.
    - **Cadence95th(steps/min)_Weekday**: 95th percentile of cadence of the day, averaged across all days. Weekdays only.
1. Steps Distribution
    - **Steps5thAt**: Average time of day when 5% of steps was accumulated.
    - **Steps25thAt**: Average time of day when 25% of steps was accumulated.
    - **Steps50thAt**: Average time of day when 50% of steps was accumulated.
    - **Steps75thAt**: Average time of day when 75% of steps was accumulated.
    - **Steps95thAt**: Average time of day when 95% of steps was accumulated.


## Random Forest Feature List
The table below describes the handcrafted features used as inputs to the Random Forest model. 
These features are extracted from the Euclidean norm of the triaxial accelerometer data.

| Feature Name                    | Description                                                        | Units          |
|---------------------------------|--------------------------------------------------------------------|----------------|
| <b>Moment features</b>                                                                                                 |
| avg                             | Mean                                                               | g               |
| std                             | Standard deviation                                                 | g               |
| skew                            | Skewness                                                           |                 |
| kurt                            | Kurtosis                                                           |                 |
| <b>Quantile Features</b>                                                                                               |
| min                             | Minimum                                                            | g               |
| q25                             | Lower quartile                                                     | g               |
| med                             | Median                                                             | g               |
| q75                             | Upper quartile                                                     | g               |
| max                             | Maximum                                                            | g               |
| <b>Autocorrelation features</b>                                                                                        |
| acf_1st_max                     | Maximum autocorrelation                                            |                 |
| acf_1st_max_loc                 | Location of 1st autocorrelation maximum                            | s               |
| acf_1st_min                     | Minimum autocorrelation                                            |                 |
| acf_1st_min_loc                 | Location of 1st autocorrelation minimum                            | s               |
| acf_zeros                       | Number of autocorrelation zero-crossings                           |                 |
| <b>Spectral features</b>                                                                                               |
| pentropy                        | Signal's spectral entropy                                          | nats            |
| power                           | Signal's total power                                               | g<sup>2</sup>/s |
| f1, f2, f3                      | 1st, 2nd and 3rd dominant frequencies                              | Hz              |
| p1, p2, p3                      | Power spectral densities of respective dominant frequencies        | g<sup>2</sup>/s |
| fft0, fft1, fft2, ...           | Power spectral density for frequencies 0Hz, 1Hz, 2Hz, ...          | g<sup>2</sup>/s |
| <b>Peak features</b>                                                                                                   |
| npeaks                          | Number of peaks in the signal per second                           | 1/s             |
| peaks_avg_promin                | Average prominence of peaks                                        | g               |
| peaks_min_promin                | Minimum prominence of peaks                                        | g               |
| peaks_max_promin                | Maximum prominence of peaks                                        | g               |
