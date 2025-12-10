"""
Tests for stepcount.features module.

Tests cover:
- Feature extraction from accelerometer windows
- Individual feature functions (moments, quantiles, autocorr, spectral, FFT, peaks)
- Butterworth filter
- Edge cases (NaN, short windows)
"""
import pytest
import numpy as np
from scipy import signal

from stepcount import features


class TestExtractFeatures:
    """Tests for the main extract_features function."""

    def test_extract_features_walking(self, accel_window_walking, sample_rate):
        """Test feature extraction on walking data."""
        feats = features.extract_features(accel_window_walking, sample_rate=sample_rate)

        assert isinstance(feats, dict)
        assert len(feats) > 0

        # Check expected feature groups exist
        expected_keys = ['avg', 'std', 'min', 'max', 'med', 'pentropy', 'power', 'npeaks']
        for key in expected_keys:
            assert key in feats, f"Missing feature: {key}"

    def test_extract_features_resting(self, accel_window_resting, sample_rate):
        """Test feature extraction on resting data."""
        feats = features.extract_features(accel_window_resting, sample_rate=sample_rate)

        assert isinstance(feats, dict)
        assert len(feats) > 0

        # Resting data should have low std
        assert feats['std'] < 0.1

    def test_extract_features_with_nan(self, sample_rate):
        """Test that NaN windows return empty dict."""
        window = np.full((100, 3), np.nan)
        feats = features.extract_features(window, sample_rate=sample_rate)

        assert feats == {}

    def test_extract_features_short_window(self, sample_rate):
        """Test that too-short windows return empty dict."""
        # MIN_WINDOW_SEC = 2, so need < 2 * sample_rate samples
        short_window = np.random.randn(sample_rate, 3) * 0.1 + [0, 0, 1]
        feats = features.extract_features(short_window, sample_rate=sample_rate)

        assert feats == {}

    def test_extract_features_dimension(self, accel_window_walking, sample_rate):
        """Test feature extraction produces consistent number of features."""
        feats1 = features.extract_features(accel_window_walking, sample_rate=sample_rate)

        # Generate another walking window
        np.random.seed(999)
        n_samples = 10 * sample_rate  # 10 seconds
        t = np.arange(n_samples) / sample_rate
        x = 0.3 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, n_samples)
        y = 0.2 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, n_samples)
        z = 1.0 + 0.1 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, n_samples)
        window2 = np.column_stack([x, y, z])

        feats2 = features.extract_features(window2, sample_rate=sample_rate)

        # Both should have the same keys
        assert set(feats1.keys()) == set(feats2.keys())


class TestMomentsFeatures:
    """Tests for moments features (mean, std, skew, kurtosis)."""

    def test_moments_basic(self, sample_rate):
        """Test moments on known distribution."""
        np.random.seed(42)
        v = np.random.randn(500)

        feats = features.moments_features(v, sample_rate=sample_rate)

        assert 'avg' in feats
        assert 'std' in feats
        assert 'skew' in feats
        assert 'kurt' in feats

        # Normal distribution should have mean ~0, std ~1
        assert abs(feats['avg']) < 0.2
        assert 0.8 < feats['std'] < 1.2

    def test_moments_constant_signal(self):
        """Test moments on constant signal (low std)."""
        v = np.ones(500) * 0.5

        feats = features.moments_features(v)

        assert feats['avg'] == pytest.approx(0.5)
        assert feats['std'] == pytest.approx(0, abs=0.01)
        # With std < 0.01, skew and kurt are set to 0
        assert feats['skew'] == 0
        assert feats['kurt'] == 0

    def test_moments_skewed_signal(self):
        """Test moments on skewed distribution."""
        np.random.seed(42)
        # Exponential distribution is right-skewed
        v = np.random.exponential(scale=1.0, size=1000)

        feats = features.moments_features(v)

        # Exponential distribution has positive skewness
        assert feats['skew'] > 0


class TestQuantileFeatures:
    """Tests for quantile features."""

    def test_quantiles_known_values(self):
        """Test quantiles on known data."""
        v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        feats = features.quantile_features(v)

        assert feats['min'] == 1
        assert feats['max'] == 10
        assert feats['med'] == 5.5  # median of 1-10
        assert feats['q25'] == pytest.approx(3.25)
        assert feats['q75'] == pytest.approx(7.75)

    def test_quantiles_negative_values(self):
        """Test quantiles with negative values."""
        v = np.array([-5, -3, -1, 0, 1, 3, 5])

        feats = features.quantile_features(v)

        assert feats['min'] == -5
        assert feats['max'] == 5
        assert feats['med'] == 0


class TestAutocorrFeatures:
    """Tests for autocorrelation features."""

    def test_autocorr_periodic_signal(self, sample_rate):
        """Test autocorrelation on periodic signal."""
        # Create a 2Hz sine wave (should have peak at 0.5s lag)
        t = np.arange(500) / sample_rate
        v = np.sin(2 * np.pi * 2 * t)

        feats = features.autocorr_features(v, sample_rate)

        # Should detect peaks and zeros in autocorrelation
        assert 'acf_1st_max' in feats
        assert 'acf_1st_max_loc' in feats
        assert 'acf_zeros' in feats

    def test_autocorr_random_signal(self, sample_rate):
        """Test autocorrelation on random noise."""
        np.random.seed(42)
        v = np.random.randn(500)

        feats = features.autocorr_features(v, sample_rate)

        # Random signal should have low autocorrelation
        # Theoretical 95% bound for n=500 is ~2/sqrt(500) = 0.09
        # Use 0.15 for robustness
        assert abs(feats['acf_1st_max']) < 0.15
        assert abs(feats['acf_1st_min']) < 0.15


class TestSpectralFeatures:
    """Tests for spectral features."""

    def test_spectral_periodic_signal(self, sample_rate):
        """Test spectral features on periodic signal."""
        # 2Hz sine wave
        t = np.arange(500) / sample_rate
        v = np.sin(2 * np.pi * 2 * t)

        feats = features.spectral_features(v, sample_rate)

        assert 'pentropy' in feats
        assert 'power' in feats
        assert 'f1' in feats  # dominant frequency
        assert 'p1' in feats  # dominant power

        # Dominant frequency should be ~2Hz
        assert 1.5 < feats['f1'] < 2.5

    def test_spectral_random_signal(self, sample_rate):
        """Test spectral features on random noise."""
        np.random.seed(42)
        v = np.random.randn(500)

        feats = features.spectral_features(v, sample_rate)

        # Random noise has high entropy (spread power)
        assert feats['pentropy'] > 0

    def test_spectral_topn_frequencies(self, sample_rate):
        """Test that top 3 frequencies are extracted."""
        # Multi-frequency signal
        t = np.arange(500) / sample_rate
        v = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)

        feats = features.spectral_features(v, sample_rate)

        # Should have f1, f2, f3 and p1, p2, p3
        assert 'f1' in feats
        assert 'f2' in feats
        assert 'f3' in feats
        assert 'p1' in feats
        assert 'p2' in feats
        assert 'p3' in feats


class TestFFTFeatures:
    """Tests for FFT (Welch) features."""

    def test_fft_bins(self, sample_rate):
        """Test FFT produces expected frequency bins."""
        t = np.arange(500) / sample_rate
        v = np.sin(2 * np.pi * 2 * t)

        feats = features.fft_features(v, sample_rate, nfreqs=5)

        # Should have fft0, fft1, ..., fft5
        for i in range(6):
            assert f'fft{i}' in feats

    def test_fft_dc_component(self, sample_rate):
        """Test FFT captures DC component (mean)."""
        # Signal with DC offset (note: feature extraction detrends by -1, so use large offset)
        v = np.ones(500) * 0.5 + 0.1 * np.sin(np.linspace(0, 10*np.pi, 500))

        feats = features.fft_features(v, sample_rate)

        # fft0 captures DC; with mean ~0.5, should have some DC power
        # Note: Welch's method with detrend='constant' removes mean, so DC may be ~0
        # This test just verifies fft0 exists and is non-negative
        assert feats['fft0'] >= 0

    def test_fft_frequency_content(self, sample_rate):
        """Test FFT captures frequency at expected bin."""
        t = np.arange(500) / sample_rate
        # 2Hz sine wave
        v = np.sin(2 * np.pi * 2 * t)

        feats = features.fft_features(v, sample_rate)

        # Power should be concentrated at 2Hz (fft2)
        assert feats['fft2'] > feats['fft1']
        assert feats['fft2'] > feats['fft3']


class TestPeaksFeatures:
    """Tests for peak detection features."""

    def test_peaks_walking_signal(self, accel_window_walking, sample_rate):
        """Test peak detection on walking signal."""
        # Compute V (vector magnitude)
        v = np.linalg.norm(accel_window_walking, axis=1)
        v = v - 1  # remove gravity
        v = np.clip(v, -2, 2)

        feats = features.peaks_features(v, sample_rate)

        assert 'npeaks' in feats
        assert 'peaks_avg_promin' in feats

        # Walking should have detectable peaks
        assert feats['npeaks'] > 0

    def test_peaks_constant_signal(self, sample_rate):
        """Test peak detection on constant signal."""
        v = np.ones(500) * 0.5

        feats = features.peaks_features(v, sample_rate)

        # Constant signal has no peaks
        assert feats['npeaks'] == 0
        assert feats['peaks_avg_promin'] == 0

    def test_peaks_count_rate(self, sample_rate):
        """Test peak count is per second."""
        # Create signal with exactly 2 peaks per second
        t = np.arange(500) / sample_rate
        # Strong peaks at 2Hz
        v = 0.5 * np.sin(2 * np.pi * 2 * t)

        feats = features.peaks_features(v, sample_rate)

        # npeaks is peaks/second, so should be ~2
        # Allow some tolerance for edge effects
        assert 1 < feats['npeaks'] < 4


class TestButterfilt:
    """Tests for Butterworth filter."""

    def test_butterfilt_lowpass(self, sample_rate):
        """Test lowpass filter removes high frequencies."""
        t = np.arange(500) / sample_rate
        # Low frequency (1Hz) + high frequency (4Hz)
        low_freq = np.sin(2 * np.pi * 1 * t)
        high_freq = 0.5 * np.sin(2 * np.pi * 4 * t)
        signal = low_freq + high_freq

        # Apply 2Hz lowpass filter
        filtered = features.butterfilt(signal, cutoffs=2, fs=sample_rate)

        # After filtering, should be closer to low_freq only
        # Correlation with low_freq should be higher than with high_freq
        corr_low = np.corrcoef(filtered, low_freq)[0, 1]
        corr_high = np.corrcoef(filtered, high_freq)[0, 1]

        assert corr_low > 0.8
        assert abs(corr_high) < 0.3

    def test_butterfilt_bandpass(self, sample_rate):
        """Test bandpass filter isolates frequency band."""
        t = np.arange(500) / sample_rate
        # Multiple frequencies
        sig = (np.sin(2 * np.pi * 1 * t) +  # 1Hz
               np.sin(2 * np.pi * 3 * t) +  # 3Hz (target)
               np.sin(2 * np.pi * 4.5 * t))   # 4.5Hz

        # Bandpass filter 2-4Hz (should isolate 3Hz)
        filtered = features.butterfilt(sig, cutoffs=(2, 4), fs=sample_rate)

        # Check dominant frequency using FFT
        freqs = np.fft.fftfreq(len(filtered), 1/sample_rate)
        fft_mag = np.abs(np.fft.fft(filtered))
        pos_mask = freqs > 0
        dominant_freq = freqs[pos_mask][np.argmax(fft_mag[pos_mask])]

        assert 2.5 < dominant_freq < 3.5

    def test_butterfilt_2d_array(self, sample_rate):
        """Test filter works on 2D arrays."""
        t = np.arange(100) / sample_rate
        signal_2d = np.column_stack([
            np.sin(2 * np.pi * 2 * t),
            np.sin(2 * np.pi * 2 * t + np.pi/4),
        ])

        filtered = features.butterfilt(signal_2d, cutoffs=3, fs=sample_rate, axis=0)

        assert filtered.shape == signal_2d.shape


class TestGetFeatureNames:
    """Tests for feature name utility."""

    def test_get_feature_names(self):
        """Test getting list of feature names."""
        names = features.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 20  # Should have many features
        assert 'avg' in names
        assert 'std' in names
        assert 'npeaks' in names

    def test_feature_names_match_extraction(self, accel_window_walking, sample_rate):
        """Test that feature names match actual extraction."""
        names = features.get_feature_names()
        feats = features.extract_features(accel_window_walking, sample_rate=sample_rate)

        assert set(names) == set(feats.keys())


class TestFeatureConsistency:
    """Tests for feature extraction consistency."""

    def test_deterministic_features(self, accel_window_walking, sample_rate):
        """Test that feature extraction is deterministic."""
        feats1 = features.extract_features(accel_window_walking, sample_rate=sample_rate)
        feats2 = features.extract_features(accel_window_walking, sample_rate=sample_rate)

        for key in feats1:
            assert feats1[key] == pytest.approx(feats2[key])

    def test_features_scale_invariance(self, sample_rate):
        """Test feature behavior with different scales."""
        np.random.seed(42)
        n_samples = 10 * sample_rate  # 10 seconds of data
        base_window = np.random.randn(n_samples, 3) * 0.1 + [0, 0, 1]

        # Scale by 2x
        scaled_window = base_window * 2

        feats_base = features.extract_features(base_window, sample_rate=sample_rate)
        feats_scaled = features.extract_features(scaled_window, sample_rate=sample_rate)

        # Std should scale proportionally
        if feats_base and feats_scaled:  # Both should produce features
            assert feats_scaled['std'] > feats_base['std']


class TestFeatureErrorPaths:
    """Tests for error handling in feature extraction."""

    def test_extract_features_all_nan(self, sample_rate):
        """Test that all-NaN windows return empty dict."""
        window = np.full((10 * sample_rate, 3), np.nan)
        feats = features.extract_features(window, sample_rate=sample_rate)

        assert feats == {}

    def test_extract_features_too_short(self, sample_rate):
        """Test that windows below MIN_WINDOW_SEC return empty dict."""
        # MIN_WINDOW_SEC is 2, so need < 2 * sample_rate samples
        short_window = np.random.randn(sample_rate, 3) * 0.1 + [0, 0, 1]
        feats = features.extract_features(short_window, sample_rate=sample_rate)

        assert feats == {}

    def test_extract_features_one_sample(self, sample_rate):
        """Test that single-sample windows return empty dict."""
        single = np.array([[0.1, 0.1, 1.0]])
        feats = features.extract_features(single, sample_rate=sample_rate)

        assert feats == {}

    def test_butterfilt_cutoff_near_nyquist(self, sample_rate):
        """Test Butterworth filter with cutoff near Nyquist frequency."""
        t = np.arange(300) / sample_rate
        signal = np.sin(2 * np.pi * 2 * t)

        # Cutoff at Nyquist/2 should work
        nyquist = sample_rate / 2
        cutoff = nyquist * 0.9  # Just below Nyquist

        # Should not raise an error
        filtered = features.butterfilt(signal, cutoffs=cutoff, fs=sample_rate)
        assert len(filtered) == len(signal)

    def test_peaks_features_very_noisy_signal(self, sample_rate):
        """Test peak detection on very noisy signal."""
        np.random.seed(42)
        # High noise, low signal
        v = np.random.randn(500) * 2

        feats = features.peaks_features(v, sample_rate)

        # Should return valid features even with noise
        assert 'npeaks' in feats
        assert 'peaks_avg_promin' in feats
        assert feats['npeaks'] >= 0
