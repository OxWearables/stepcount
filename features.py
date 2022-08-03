import numpy as np
import scipy.stats as stats
import scipy.signal as signal


def extract_features(xyz, sample_rate=100):
    ''' Extract commonly used HAR time-series features. xyz is a window of shape (N,3) '''

    feats = {}

    v = np.linalg.norm(xyz, axis=1)
    v = v - 1  # detrend: "remove gravity"
    v = np.clip(v, -2, 2)  # clip abnormaly high values

    # Quantile features
    feats.update(quantile_features(v, sample_rate))

    # Autocorrelation features
    feats.update(autocorr_features(v, sample_rate))

    # Spectral features
    feats.update(spectral_features(v, sample_rate))

    # Peak features
    feats.update(peaks_features(v, sample_rate))

    return feats


def quantile_features(v, sample_rate=None):
    """ Quantiles (min, 25th, med, 75th, max) """
    feats = {}
    feats['min'], feats['q25'], feats['med'], feats['q75'], feats['max'] = np.quantile(v, (0, .25, .5, .75, 1))
    return feats


def autocorr_features(v, sample_rate):
    """ Autocorrelation (0.5, 1 and 2 seconds lag) """
    feats = {}
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        onesec = int(sample_rate)
        halfsec = int(sample_rate // 2)
        twosec = int(2 * sample_rate)
        feats['autocorr_halfsec'] = np.nan_to_num(np.corrcoef(v[:-halfsec], v[halfsec:]))[0, 1]
        feats['autocorr_onesec'] = np.nan_to_num(np.corrcoef(v[:-onesec], v[onesec:]))[0, 1]
        feats['autocorr_twosec'] = np.nan_to_num(np.corrcoef(v[:-twosec], v[twosec:]))[0, 1]
    return feats


def spectral_features(v, sample_rate):
    """ Spectral entropy, 1st & 2nd dominant frequencies """

    feats = {}

    # Spectrum using Welch's method with 3s chunks and 2s overlaps
    freqs, powers = signal.welch(
        v, fs=sample_rate,
        nperseg=3 * sample_rate,
        noverlap=2 * sample_rate,
        detrend='constant',
        average='median'
    )

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        feats['pentropy'] = np.nan_to_num(stats.entropy(powers + 1e-16))

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]
    if len(peaks) >= 2:
        feats['f1'] = peak_freqs[peak_ranks[0]]
        feats['f2'] = peak_freqs[peak_ranks[1]]
        feats['p1'] = peak_powers[peak_ranks[0]]
        feats['p2'] = peak_powers[peak_ranks[1]]
    elif len(peaks) == 1:
        feats['f1'] = feats['f2'] = peak_freqs[peak_ranks[0]]
        feats['p1'] = feats['p2'] = peak_powers[peak_ranks[0]]
    else:
        feats['f1'] = feats['f2'] = 0
        feats['p1'] = feats['p2'] = 0

    return feats


def peaks_features(v, sample_rate):
    """ Features of the signal peaks """

    feats = {}
    u = butterfilt(v, 5, fs=sample_rate)  # lowpass 5Hz
    peaks, peak_props = signal.find_peaks(u, distance=0.2 * sample_rate, prominence=0.25)
    feats['npeaks'] = len(peaks)
    if len(peak_props['prominences']) > 0:
        feats['peaks_med_promin'], feats['peaks_min_promin'], feats['peaks_max_promin'] = \
            np.quantile(peak_props['prominences'], (0, .5, 1))
    else:
        feats['peaks_med_promin'] = feats['peaks_min_promin'] = feats['peaks_max_promin'] = 0

    return feats


def butterfilt(x, cutoffs, fs, order=10, axis=0):
    """ Butterworth filter """
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


def get_feature_names():
    """ Hacky way to get the list of feature names """

    feats = extract_features(np.zeros((500, 3)), 100)
    return list(feats.keys())
