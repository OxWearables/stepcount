import catch22
import numpy as np
import scipy.stats as stats
import scipy.signal as signal


def extract_features(xyz, sample_rate):
    ''' Signal features. xyz is a window of shape (N,3) '''

    feats = {}

    xyz = np.clip(xyz, -3, 3)

    feats['xAbsMed'], feats['yAbsMed'], feats['zAbsMed'] = np.abs(np.median(xyz, axis=0))
    feats['xRange'], feats['yRange'], feats['zRange'] = np.ptp(xyz, axis=0)
    feats['xIQR'], feats['yIQR'], feats['zIQR'] = stats.iqr(xyz, axis=0)

    v = np.linalg.norm(xyz, axis=1)

    feats['median'] = np.median(v)
    feats['min'] = np.min(v)
    feats['max'] = np.max(v)
    feats['q25'] = np.quantile(v, .25)
    feats['q75'] = np.quantile(v, .75)

    # Spectral features
    feats.update(spectral_features(v, sample_rate))

    # Catch22 features
    vtup = tuple(v)  # catch22 takes tuple or list
    # Successive differences
    # Shannon entropy of two successive letters in equiprobable 3-letter symbolization
    feats['SB_MotifThree_quantile_hh'] = catch22.SB_MotifThree_quantile_hh(vtup)
    # Change in correlation length after iterative differencing
    feats['FC_LocalSimple_mean1_tauresrat'] = catch22.FC_LocalSimple_mean1_tauresrat(vtup)
    # Proportion of successive differences exceeding 0.04 sigma
    feats['MD_hrv_classic_pnn40'] = catch22.MD_hrv_classic_pnn40(vtup)
    # Simple temporal statistics
    # Longest period of consecutive values above the mean
    feats['SB_BinaryStats_mean_longstretch1'] = catch22.SB_BinaryStats_mean_longstretch1(vtup)
    # Nonlinear autocorrelation
    # First minimum of the automutual information function
    feats['IN_AutoMutualInfoStats_40_gaussian_fmmi'] = catch22.IN_AutoMutualInfoStats_40_gaussian_fmmi(vtup)
    # Linear autocorrelation
    # First 1 / e crossing of autocorrelation function
    feats['CO_f1ecac'] = catch22.CO_f1ecac(vtup)

    # Signal peaks
    u = butterfilt(v, (.6, 5), fs=sample_rate, order=8)
    # Prominence 0.25 seems to be best for 0.6-5Hz, based on the RMSE map plot
    # Also, the smallest largest prominence in Rowlands walks is ~0.3,
    # meaning some walking windows would have no steps for any higher value
    peaks, peak_props = signal.find_peaks(u, distance=0.2 * sample_rate, prominence=0.25)
    feats['numPeaks'] = len(peaks)
    if len(peak_props['prominences']) > 0:
        feats['peakPromin'] = np.median(peak_props['prominences'])
        feats['peakProminIQR'] = stats.iqr(peak_props['prominences'])
    else:
        feats['peakPromin'] = 0
        feats['peakProminIQR'] = 0

    return feats


def spectral_features(v, sample_rate):
    """ Spectral entropy, 1st & 2nd dominant frequencies """

    feats = {}

    # Spectrum using Welch's method with 3s segment length
    # First run without detrending to get the true spectrum
    freqs, powers = signal.welch(v, fs=sample_rate,
                                 nperseg=3 * sample_rate,
                                 noverlap=2 * sample_rate,
                                 detrend=False,
                                 average='median')

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        feats['pentropy'] = np.nan_to_num(stats.entropy(powers + 1e-16))

    # Spectrum using Welch's method with 3s segment length
    # Now do detrend to focus on the relevant freqs
    freqs, powers = signal.welch(v, fs=sample_rate,
                                 nperseg=3 * sample_rate,
                                 noverlap=2 * sample_rate,
                                 detrend='constant',
                                 average='median')

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]
    if len(peaks) >= 2:
        feats['f1'] = peak_freqs[peak_ranks[0]]
        feats['f2'] = peak_freqs[peak_ranks[1]]
    elif len(peaks) == 1:
        feats['f1'] = feats['f2'] = peak_freqs[peak_ranks[0]]
    else:
        feats['f1'] = feats['f2'] = 0

    return feats


def butterfilt(x, cutoffs, fs, order=8, axis=0):
    nyq = 0.5 * fs
    if isinstance(cutoffs, tuple):
        hicut, lowcut = cutoffs
        if hicut > 0:
            btype = 'bandpass'
            Wn = (hicut / nyq, lowcut / nyq)
        else:
            btype = 'low'
            Wn = lowcut / nyq
    else:
        btype = 'low'
        Wn = cutoffs / nyq
    sos = signal.butter(order, Wn, btype=btype, analog=False, output='sos')
    y = signal.sosfiltfilt(sos, x, axis=axis)
    return y
