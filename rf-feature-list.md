# Random Forest Feature List
The table below presents an overview of the handcrafted features that have been curated as inputs for the Random Forest model. 
These features are extracted from the Euclidean norm of the triaxial accelerometer data, minus 1 to remove gravity.

| Feature Name                    | Description                                                        | Units          |
|---------------------------------|--------------------------------------------------------------------|----------------|
| <b>Moment features</b>          |                                                                    |                |
| avg                             | Mean                                                               | g              |
| std                             | Standard deviation                                                 | g              |
| skew                            | Skewness                                                           |                |
| kurt                            | Kurtosis                                                           |                |
| <b>Quantile Features</b>        |                                                                    |                |
| min                             | Minimum                                                            | g              |
| q25                             | Lower quartile                                                     | g              |
| med                             | Median                                                             | g              |
| q75                             | Upper quartile                                                     | g              |
| max                             | Maximum                                                            | g              |
| <b>Autocorrelation features</b> |                                                                    |                |
| acf_1st_max                     | Maximum autocorrelation                                            |                |
| acf_1st_max_loc                 | Location of 1st autocorrelation maximum                            | s              |
| acf_1st_min                     | Minimum autocorrelation                                            |                |
| acf_1st_min_loc                 | Location of 1st autocorrelation minimum                            | s              |
| acf_zeros                       | Number of autocorrelation zero-crossings                           |                |
| <b>Spectral features</b>        |                                                                    |                |
| pentropy                        | Spectral entropy calculated from the power spectrum                |                |
| power                           | Sum of the power spectrum                                          |                |
| f1, f2, f3                      | Dominant frequencies in the power spectrum                         | Hz             |
| p1, p2, p3                      | Power associated with respective dominant frequencies              |                |
| <b>FFT features</b>             |                                                                    |                |
| fft0, fft1, fft2, ...           | Power values at specific frequencies obtained using Welch's method |                |
| <b>Peak features</b>            |                                                                    |                |
| npeaks                          | Number of peaks in the signal per second                           | s<sup>-1</sup> |
| peaks_avg_promin                | Average prominence of peaks                                        | g              |
| peaks_min_promin                | Minimum prominence of peaks                                        | g              |
| peaks_max_promin                | Maximum prominence of peaks                                        | g              |
