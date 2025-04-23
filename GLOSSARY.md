# Glossary

## General Information
- **StepCountVersion**: Version number of the `stepcount` program used to run the analysis. This information is useful to ensure reproducibility.
- **StepCountArgs**: Full list of command-line arguments supplied to `stepcount`. This information is useful to ensure reproducibility.
- **Filename**: The name of the input file.
- **Filesize(MB)**: The size of the file in megabytes.
- **Device**: Brand, model, or file extension of the recording device (e.g., .cwa, .csv).
- **DeviceID**: A unique identifier for the specific device used.
- **ReadErrors**: Count of read failures or corrupt blocks encountered during file parsing.
- **SampleRate(Hz)**: Number of sensor readings captured per second.
- **ReadOK**: Boolean flag (1 = successful import, 0 = failure).
- **NumTicks**: Total number of raw data samples (ticks) recorded.
- **StartTime**: Time when recording began.
- **EndTime**: Time when recording ended.
- **WearStartTime**: Time marking when valid wear data begins (after non-wear filtering).
- **WearEndTime**: Time marking when valid wear data ends.
- **WearTime(days)**: Total duration of valid wear time, expressed in days.
- **NumInterrupts**: Number of interruptions in the data recording.
- **CalibErrorBefore(mg)**: Calibration error before any correction was applied, measured in milli-g.
- **CalibOK**: Boolean flag (1 = calibration succeeded, 0 = failed).
- **CalibErrorAfter(mg)**: Calibration error after attempting correction, measured in milli-g.
- **NonwearTime(days)**: Total duration classified as non-wear.
- **NumNonwearEpisodes**: Number of distinct non-wear intervals detected.
- **ResampleRate(Hz)**: Sampling frequency after resampling.
- **NumTicksAfterResample**: Number of samples after resampling.
- **Covers24hOK**: Flag indicating whether every hour of the 24‑hour cycle has at least one wear epoch on any day (1 = fully covered, 0 = at least one hour never covered across all days).

## ENMO
- **ENMO(mg)**: Euclidean norm minus one and zero-truncated, measured in milli-g.
- **ENMO(mg)_Hour{XX}**: ENMO in the hour bin `{XX}` (00–23).
- **ENMO(mg)_Weekend**: Overall average ENMO in weekend days.
- **ENMO(mg)_Hour{XX}_Weekend**: ENMO in the hour bin `{XX}` (00-23) on weekend days.
- **ENMO(mg)_Weekday**: Overall average ENMO in weekdays (Monday–Friday).
- **ENMO(mg)_Hour{XX}_Weekday**: ENMO in the hour bin `{XX}` (00-23) on weekdays.

## Steps

- **TotalSteps**: Total step count.
- **StepsDayAvg**: Mean daily steps.
- **StepsDayMed**: Median daily steps.
- **StepsDayMin**: Minimum steps in any single day.
- **StepsDayMax**: Maximum steps in any single day.
- **Steps_Hour{XX}**: Steps in the hour bin `{XX}` (00–23).

### Steps -- Weekends Only

- **TotalSteps_Weekend**: Total step count on weekends.
- **StepsDayAvg_Weekend**: Mean daily steps on weekends.
- **StepsDayMed_Weekend**: Median daily steps on weekends.
- **StepsDayMin_Weekend**: Minimum daily steps on a weekend day.
- **StepsDayMax_Weekend**: Maximum daily steps on a weekend day.
- **Steps_Hour{XX}_Weekend**: Steps in the hour bin `{XX}` (00-23) on weekends.

### Steps -- Weekdays Only

- **TotalSteps_Weekday**: Total step count on weekdays.
- **StepsDayAvg_Weekday**: Mean daily steps on weekdays.
- **StepsDayMed_Weekday**: Median daily steps on weekdays.
- **StepsDayMin_Weekday**: Minimum daily steps on a weekday.
- **StepsDayMax_Weekday**: Maximum daily steps on a weekday.
- **Steps_Hour{XX}_Weekday**: Steps in the hour bin `{XX}` (00-23) on weekdays.

## Walking

- **TotalWalking(mins)**: Total walking duration.
- **WalkingDayAvg(mins)**: Average daily walking duration.
- **WalkingDayMed(mins)**: Median daily walking duration.
- **WalkingDayMin(mins)**: Minimum walking duration in any single day.
- **WalkingDayMax(mins)**: Maximum walking duration in any single day.
- **Walking(mins)_Hour{XX}**: Walking time in the hour bin `{XX}` (00-23).

*(Weekend and Weekday subsections analogous to Steps above.)*

## Cadence

- **CadenceTop1(steps/min)**: Highest cadence per day.
- **CadenceTop30(steps/min)**: Mean cadence of the thirty most active one-minute epochs per day.
- **Cadence95th(steps/min)**: 95th percentile of cadence per day.

*(Weekend and Weekday subsections analogous to Steps above.)*

## Steps Distribution

- **Steps5thAt**: Average clock time when 5% of total daily steps are reached.
- **Steps25thAt**: Average clock time when 25% of total daily steps are are reached.
- **Steps50thAt**: Average clock time when half (50%) of total daily steps are reached.
- **Steps75thAt**: Average clock time when 75% of total daily steps are reached.
- **Steps95thAt**: Average clock time when 95% of total daily steps are reached.

## Crude vs. Adjusted Estimates
Crude estimates represent raw metrics calculated directly from observed data.
Adjusted estimates compensate for missing time-series values by imputing each
absent timepoint with the average value at that same clock time across all other
recorded days. To derive adjusted totals and daily summaries, any gaps in the
required 24‑hour span are similarly imputed; if data remain missing after this
process, the estimate is reported as NaN. Adjusted metrics are labeled with an
"Adjusted" suffix&mdash;for example, `StepsDayAvgAdjusted_Weekend`.

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
