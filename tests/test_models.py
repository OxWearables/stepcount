"""
Tests for stepcount.models module.

Tests cover:
- make_windows function
- Peak counting functions
- toV (vector magnitude) computation
- Sample weight calculation
- Calibration functions
- Cross-validation utilities
- StepCounter class (basic functionality)
- WalkDetectorRF class (basic functionality)
"""
import pytest
import numpy as np
import pandas as pd

from stepcount import models


class TestMakeWindows:
    """Tests for make_windows function."""

    def test_make_windows_basic(self, sample_rate):
        """Test basic window creation with evenly divisible data."""
        window_sec = 10
        n_windows = 100
        n_samples = n_windows * sample_rate * window_sec

        # Create evenly divisible test data
        times = pd.date_range('2024-01-15', periods=n_samples, freq=f'{1000000//sample_rate}us')
        data = pd.DataFrame(
            np.random.randn(n_samples, 3) * 0.1,
            columns=['x', 'y', 'z'],
            index=times
        )
        data.index.name = 'time'

        def fn(chunk):
            return chunk[['x', 'y', 'z']].to_numpy()

        X = models.make_windows(data, window_sec, fn=fn, verbose=False)

        # Check output - may be 3D (homogeneous) or object array (if windows vary)
        assert len(X) == n_windows
        # Verify each window has the expected shape
        for window in X:
            assert window.ndim == 2  # (samples_per_window, 3)
            assert window.shape[1] == 3  # 3 columns (x, y, z)

    def test_make_windows_with_index(self, sample_rate):
        """Test window creation returns timestamps."""
        window_sec = 10
        n_windows = 50
        n_samples = n_windows * sample_rate * window_sec

        times = pd.date_range('2024-01-15', periods=n_samples, freq=f'{1000000//sample_rate}us')
        data = pd.DataFrame(
            np.random.randn(n_samples, 3) * 0.1,
            columns=['x', 'y', 'z'],
            index=times
        )
        data.index.name = 'time'

        def fn(chunk):
            return chunk[['x', 'y', 'z']].to_numpy()

        X, T = models.make_windows(data, window_sec, fn=fn, return_index=True, verbose=False)

        assert len(X) == len(T)
        assert isinstance(T, pd.DatetimeIndex)
        assert len(X) == n_windows

    def test_make_windows_custom_fn(self, sample_rate):
        """Test window creation with custom function that returns fixed-size output."""
        window_sec = 10
        n_windows = 20
        n_samples = n_windows * sample_rate * window_sec

        times = pd.date_range('2024-01-15', periods=n_samples, freq=f'{1000000//sample_rate}us')
        data = pd.DataFrame(
            np.random.randn(n_samples, 3) * 0.1,
            columns=['x', 'y', 'z'],
            index=times
        )
        data.index.name = 'time'

        def extract_mean(chunk):
            # Returns fixed-size output (3 values) regardless of input size
            return chunk[['x', 'y', 'z']].mean().to_numpy()

        X = models.make_windows(data, window_sec, fn=extract_mean, verbose=False)

        # Each window result should have 3 values
        assert X.shape == (n_windows, 3)

    def test_make_windows_different_sizes(self, sample_rate):
        """Test window creation with different window sizes."""
        for window_sec in [5, 10, 30]:
            n_windows = 10
            n_samples = n_windows * sample_rate * window_sec

            times = pd.date_range('2024-01-15', periods=n_samples, freq=f'{1000000//sample_rate}us')
            data = pd.DataFrame(
                np.random.randn(n_samples, 3) * 0.1,
                columns=['x', 'y', 'z'],
                index=times
            )
            data.index.name = 'time'

            def fn(chunk):
                return chunk[['x', 'y', 'z']].to_numpy()

            X = models.make_windows(data, window_sec, fn=fn, verbose=False)
            assert len(X) == n_windows


class TestToV:
    """Tests for toV (vector magnitude) computation."""

    def test_toV_shape(self, accel_window_walking, sample_rate):
        """Test toV output shape."""
        X = np.expand_dims(accel_window_walking, 0)  # Add batch dimension

        V = models.toV(X, sample_rate, lowpass_hz=5)

        assert V.shape[0] == 1  # batch size
        assert V.shape[1] == accel_window_walking.shape[0]  # samples

    def test_toV_removes_gravity(self, sample_rate):
        """Test that toV removes gravity (subtracts 1)."""
        # Stationary signal at 1g on z-axis
        xyz = np.tile([[0, 0, 1]], (100, 1))
        xyz = np.expand_dims(xyz, 0)

        V = models.toV(xyz, sample_rate, lowpass_hz=5)

        # After removing gravity, should be close to 0
        assert np.abs(V.mean()) < 0.1

    def test_toV_clips_extreme(self, sample_rate):
        """Test that toV clips extreme values."""
        # Need at least 16 samples for sosfiltfilt (padlen=15)
        # Create extreme values repeated enough times
        xyz = np.tile([[10, 10, 10]], (100, 1))
        xyz = np.expand_dims(xyz, 0)

        V = models.toV(xyz, sample_rate, lowpass_hz=5)

        # Should be clipped to [-2, 2] (with small floating point tolerance)
        assert V.max() <= 2 + 1e-10
        assert V.min() >= -2 - 1e-10

    def test_toV_lowpass_filter(self, sample_rate):
        """Test that toV applies lowpass filter."""
        # High frequency noise on top of signal
        t = np.arange(100) / sample_rate
        noise = 0.5 * np.sin(2 * np.pi * 4.5 * t)  # 4.5Hz (just below Nyquist for 10Hz)
        xyz = np.column_stack([
            noise,
            np.zeros(100),
            np.ones(100)
        ])
        xyz = np.expand_dims(xyz, 0)

        V_filtered = models.toV(xyz, sample_rate, lowpass_hz=3)

        # High frequency should be attenuated
        # V should have lower variance than original noise
        assert V_filtered.std() < 0.4


class TestBatchCountPeaks:
    """Tests for peak counting functions."""

    def test_batch_count_peaks_walking(self, accel_window_walking, sample_rate):
        """Test peak counting on walking signal."""
        X = np.expand_dims(accel_window_walking, 0)

        params = {'distance': 0.4, 'prominence': 0.2}
        counts = models.batch_count_peaks(X, sample_rate, lowpass_hz=5, params=params)

        assert len(counts) == 1
        # Walking at ~1.8Hz for 10s should have ~18 steps
        assert 5 < counts[0] < 30

    def test_batch_count_peaks_resting(self, accel_window_resting, sample_rate):
        """Test peak counting on resting signal."""
        X = np.expand_dims(accel_window_resting, 0)

        params = {'distance': 0.4, 'prominence': 0.2}
        counts = models.batch_count_peaks(X, sample_rate, lowpass_hz=5, params=params)

        # Resting should have few/no detectable steps
        assert counts[0] < 5

    def test_batch_count_peaks_multiple_windows(self, accel_windows_mixed, sample_rate):
        """Test peak counting on multiple windows."""
        X, labels = accel_windows_mixed

        params = {'distance': 0.4, 'prominence': 0.2}
        counts = models.batch_count_peaks(X, sample_rate, lowpass_hz=5, params=params)

        assert len(counts) == len(X)

        # Walking windows should generally have more peaks
        walking_counts = counts[labels == 1]
        resting_counts = counts[labels == 0]

        assert walking_counts.mean() > resting_counts.mean()

    def test_batch_count_peaks_with_timestamps(self, accel_window_walking, sample_rate):
        """Test peak counting returns peak timestamps."""
        X = np.expand_dims(accel_window_walking, 0)

        params = {'distance': 0.4, 'prominence': 0.2}
        counts, peaks = models.batch_count_peaks(
            X, sample_rate, lowpass_hz=5, params=params, return_peaks=True
        )

        assert len(peaks) == 1
        # Peaks should be timestamps (in seconds) within window
        if len(peaks[0]) > 0:
            assert peaks[0].min() >= 0
            assert peaks[0].max() <= 10  # 10-second window


class TestCalcSampleWeight:
    """Tests for sample weight calculation."""

    def test_calc_sample_weight_no_pnr(self):
        """Test sample weights without PNR."""
        yt = np.array([0, 0, 1, 1, 0])

        weights = models.calc_sample_weight(yt, pnr=None)

        assert np.allclose(weights, np.ones(5))

    def test_calc_sample_weight_with_pnr(self):
        """Test sample weights with PNR (positive to negative ratio)."""
        yt = np.array([0, 0, 0, 1, 1])  # 3 negatives, 2 positives

        weights = models.calc_sample_weight(yt, pnr=1.0)

        # Negatives should be downweighted to balance
        # Weight for 0 = (2) / (1.0 * 3) = 2/3
        assert np.allclose(weights[yt == 0], 2/3)
        assert np.allclose(weights[yt == 1], 1.0)

    def test_calc_sample_weight_shape(self):
        """Test sample weight shape matches input."""
        yt = np.random.randint(0, 2, size=100)

        weights = models.calc_sample_weight(yt, pnr=1.0)

        assert weights.shape == yt.shape


class TestCalibrate:
    """Tests for calibration function."""

    def test_calibrate_returns_thresholds(self):
        """Test calibration returns threshold information."""
        np.random.seed(42)
        yt = np.random.randint(0, 2, size=100)
        yp = np.random.rand(100)  # Random probability predictions

        results = models.calibrate(yp, yt)

        # Should have various calibration results
        assert 'best_f1' in results
        assert 'best_precision' in results
        assert 'best_recall' in results
        assert 'best_balanced_accuracy' in results

        # Each should have a threshold
        assert 'thresh' in results['best_f1']
        assert 0 <= results['best_f1']['thresh'] <= 1

    def test_calibrate_f1_threshold(self):
        """Test F1-optimal threshold is reasonable."""
        # Good predictions
        yt = np.array([0]*50 + [1]*50)
        yp = np.concatenate([
            np.random.uniform(0, 0.4, 50),  # Low for negatives
            np.random.uniform(0.6, 1.0, 50)  # High for positives
        ])

        results = models.calibrate(yp, yt)

        # Threshold should be around 0.5
        assert 0.3 < results['best_f1']['thresh'] < 0.7

    def test_calibrate_with_pnr(self):
        """Test calibration with positive-to-negative ratio."""
        np.random.seed(42)
        yt = np.random.randint(0, 2, size=100)
        yp = np.random.rand(100)

        results = models.calibrate(yp, yt, pnr=2.0)

        assert 'best_f1' in results


class TestGroupKFold:
    """Tests for group k-fold utility."""

    def test_groupkfold_splits(self):
        """Test groupkfold creates correct splits."""
        groups = np.array(['A', 'A', 'B', 'B', 'C', 'C'])

        splits = list(models.groupkfold(groups, n_splits=3))

        assert len(splits) == 3

        # Each split should have train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx[0]) > 0
            assert len(test_idx[0]) > 0

    def test_groupkfold_no_leak(self):
        """Test that groups don't leak between train/test."""
        groups = np.array(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'])

        for train_idx, test_idx in models.groupkfold(groups, n_splits=2):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            # No overlap between train and test groups
            assert train_groups.isdisjoint(test_groups)


class TestGetCVScores:
    """Tests for cross-validation scoring."""

    def test_get_cv_scores_classification(self):
        """Test CV scores for classification."""
        yt = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        yp = np.array([0, 0, 1, 1, 0, 1, 1, 1])  # One mistake
        cv_test_idxs = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]
        sample_weight = np.ones_like(yt, dtype=float)

        raw_scores, summary = models.get_cv_scores(
            yt, yp, cv_test_idxs,
            sample_weight=sample_weight,
            scorer_type='classif'
        )

        assert 'accuracy' in raw_scores
        assert 'f1' in raw_scores
        assert 'precision' in raw_scores
        assert 'recall' in raw_scores

        # Summary should have statistics
        assert 'mean' in summary['accuracy']
        assert 'std' in summary['accuracy']

    def test_get_cv_scores_regression(self):
        """Test CV scores for regression."""
        yt = np.array([10, 20, 30, 40, 15, 25, 35, 45])
        yp = np.array([12, 18, 32, 38, 16, 24, 34, 46])  # Close predictions
        cv_test_idxs = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]
        sample_weight = np.ones_like(yt, dtype=float)

        raw_scores, summary = models.get_cv_scores(
            yt, yp, cv_test_idxs,
            sample_weight=sample_weight,
            scorer_type='regress'
        )

        assert 'mae' in raw_scores
        assert 'rmse' in raw_scores
        assert 'mape' in raw_scores


class TestBatchExtractFeatures:
    """Tests for batch feature extraction."""

    def test_batch_extract_features(self, accel_windows_mixed, sample_rate):
        """Test batch feature extraction."""
        X, _ = accel_windows_mixed

        feats = models.batch_extract_features(
            X, sample_rate, to_numpy=True, n_jobs=1, verbose=False
        )

        assert feats.shape[0] == len(X)
        assert feats.shape[1] > 20  # Should have many features

    def test_batch_extract_features_handles_nan(self, sample_rate):
        """Test batch feature extraction handles NaN windows."""
        # Mix of valid and NaN windows
        valid_window = np.random.randn(100, 3) * 0.1 + [0, 0, 1]
        nan_window = np.full((100, 3), np.nan)
        X = np.array([valid_window, nan_window, valid_window])

        feats = models.batch_extract_features(
            X, sample_rate, to_numpy=True, n_jobs=1, verbose=False
        )

        assert feats.shape[0] == 3
        # NaN window should produce NaN features
        assert np.isnan(feats[1]).all()


class TestStepCounterBasic:
    """Basic tests for StepCounter class."""

    def test_stepcounter_init_ssl(self):
        """Test StepCounter initialization with SSL type."""
        model = models.StepCounter(wd_type='ssl', verbose=False)

        # SSL should set window_sec=10 and sample_rate=30
        assert model.window_sec == 10
        assert model.sample_rate == 30

    def test_stepcounter_init_rf(self):
        """Test StepCounter initialization with RF type."""
        model = models.StepCounter(
            wd_type='rf',
            window_sec=5,
            sample_rate=100,
            verbose=False
        )

        assert model.window_sec == 5
        assert model.sample_rate == 100

    def test_stepcounter_steptol(self):
        """Test StepCounter step tolerance setting."""
        model = models.StepCounter(wd_type='ssl', steptol=5, verbose=False)

        assert model.steptol == 5

    def test_stepcounter_predict_untrained(self, accel_windows_mixed):
        """Test predict on untrained model gives warning."""
        model = models.StepCounter(wd_type='ssl', verbose=False)

        X, _ = accel_windows_mixed

        # Should print warning but not crash
        result = model.predict(X[:5])

        assert result is None  # Returns None when untrained


class TestWalkDetectorRFBasic:
    """Basic tests for WalkDetectorRF class."""

    def test_walk_detector_rf_init(self, sample_rate):
        """Test WalkDetectorRF initialization."""
        detector = models.WalkDetectorRF(sample_rate=sample_rate, verbose=False)

        assert detector.sample_rate == sample_rate
        assert detector.thresh == 0.5  # Default threshold

    def test_walk_detector_rf_predict_empty(self, sample_rate):
        """Test WalkDetectorRF handles empty input."""
        detector = models.WalkDetectorRF(sample_rate=sample_rate, verbose=False)

        # Manually set up minimal HMM parameters for prediction
        detector.hmms.labels = np.array([0, 1])
        detector.hmms.startprob = np.array([0.5, 0.5])
        detector.hmms.emissionprob = np.eye(2)
        detector.hmms.transmat = np.array([[0.9, 0.1], [0.1, 0.9]])

        result = detector.predict(np.array([]))

        assert len(result) == 0


class TestWalkDetectorSSLBasic:
    """Basic tests for WalkDetectorSSL class."""

    def test_walk_detector_ssl_init(self):
        """Test WalkDetectorSSL initialization."""
        detector = models.WalkDetectorSSL(
            device='cpu',
            batch_size=100,
            verbose=False
        )

        assert detector.device == 'cpu'
        assert detector.batch_size == 100

    @pytest.mark.skip(reason="SSL model requires valid state_dict to be loaded before predict (needs model weights)")
    def test_walk_detector_ssl_predict_empty(self):
        """Test WalkDetectorSSL handles empty input.

        Note: The SSL model tries to load state_dict on first predict call.
        Without valid weights, this fails. This test documents the expected
        behavior but cannot run without model weights.
        """
        detector = models.WalkDetectorSSL(device='cpu', verbose=False)

        # Manually set up minimal HMM parameters
        detector.hmms.labels = np.array([0, 1])
        detector.hmms.startprob = np.array([0.5, 0.5])
        detector.hmms.emissionprob = np.eye(2)
        detector.hmms.transmat = np.array([[0.9, 0.1], [0.1, 0.9]])

        result = detector.predict(np.array([]))

        assert len(result) == 0


class TestBatchFindPeaks:
    """Tests for batch_find_peaks_from_V function."""

    def test_batch_find_peaks_timing(self, sample_rate):
        """Test that peak times are in seconds."""
        # Create a simple signal with known peaks
        t = np.arange(100) / sample_rate
        v = np.sin(2 * np.pi * 1 * t)  # 1Hz sine = peaks at 0.25, 1.25, etc.
        V = v.reshape(1, -1)

        params = {'distance': 0.4, 'prominence': 0.5}
        peaks = models.batch_find_peaks_from_V(V, sample_rate, params)

        assert len(peaks) == 1
        # Peak times should be in seconds
        if len(peaks[0]) > 0:
            assert peaks[0].max() <= 10  # Within 10-second window


class TestClassificationReport:
    """Tests for classification report utility."""

    def test_classification_report(self):
        """Test classification report generation."""
        yt = np.array([0, 0, 1, 1, 0, 1])
        yp = np.array([0, 0, 1, 0, 0, 1])

        report = models.classification_report(yt, yp, pnr=1.0)

        assert isinstance(report, str)
        assert 'precision' in report.lower()
        assert 'recall' in report.lower()


class TestWalkDetectorRFFit:
    """Tests for WalkDetectorRF.fit() method."""

    @pytest.fixture
    def walk_training_data(self, sample_rate):
        """Generate minimal training data for walk detector."""
        np.random.seed(42)
        n_windows = 40
        window_len = 10 * sample_rate

        X = []
        Y = []  # 0 or 1 (walking label)
        groups = []

        for i in range(n_windows):
            is_walking = i % 2 == 0
            if is_walking:
                t = np.arange(window_len) / sample_rate
                x = 0.3 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, window_len)
                y = 0.2 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, window_len)
                z = 1.0 + 0.1 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, window_len)
            else:
                x = np.random.normal(0, 0.02, window_len)
                y = np.random.normal(0, 0.02, window_len)
                z = np.random.normal(1.0, 0.02, window_len)

            X.append(np.column_stack([x, y, z]))
            Y.append(1 if is_walking else 0)
            groups.append(f"subj_{i // 10}")

        return np.array(X), np.array(Y), np.array(groups)

    def test_walk_detector_rf_fit(self, walk_training_data, sample_rate):
        """Test WalkDetectorRF.fit() completes successfully."""
        X, Y, groups = walk_training_data

        detector = models.WalkDetectorRF(
            sample_rate=sample_rate,
            cv=2,
            n_jobs=1,
            verbose=False
        )

        detector.fit(X, Y, groups=groups)

        # Verify training completed
        assert detector.thresh is not None
        assert 0 <= detector.thresh <= 1
        assert detector.hmms.startprob is not None
        assert detector.hmms.emissionprob is not None
        assert detector.hmms.transmat is not None

    def test_walk_detector_rf_fit_predict(self, walk_training_data, sample_rate):
        """Test WalkDetectorRF can predict after fitting."""
        X, Y, groups = walk_training_data

        detector = models.WalkDetectorRF(
            sample_rate=sample_rate,
            cv=2,
            n_jobs=1,
            verbose=False
        )

        detector.fit(X, Y, groups=groups)
        predictions = detector.predict(X[:5])

        assert len(predictions) == 5
        assert set(predictions).issubset({0, 1})


class TestCVP:
    """Tests for cvp() cross-validation predict utility."""

    def test_cvp_basic(self):
        """Test cvp returns predictions for all samples."""
        np.random.seed(42)

        # Simple feature array
        X = np.random.randn(50, 10)
        Y = np.array([0] * 25 + [1] * 25)
        groups = np.array(['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 + ['E'] * 10)

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=42, max_iter=200)

        Yp = models.cvp(clf, X, Y, groups, n_splits=5, n_jobs=1)

        assert len(Yp) == len(Y)
        assert set(Yp).issubset({0, 1})

    def test_cvp_with_indices(self):
        """Test cvp returns fold indices when requested."""
        np.random.seed(42)

        X = np.random.randn(50, 10)
        Y = np.array([0] * 25 + [1] * 25)
        groups = np.array(['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 + ['E'] * 10)

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=42, max_iter=200)

        Yp, cv_idxs = models.cvp(clf, X, Y, groups, n_splits=5, n_jobs=1, return_indices=True)

        assert len(cv_idxs) == 5
        # All indices should be covered
        all_idxs = np.concatenate([idx[0] for idx in cv_idxs])
        assert len(all_idxs) == len(Y)

    def test_cvp_predict_proba(self):
        """Test cvp with predict_proba method."""
        np.random.seed(42)

        X = np.random.randn(50, 10)
        Y = np.array([0] * 25 + [1] * 25)
        groups = np.array(['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 + ['E'] * 10)

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=42, max_iter=200)

        Yp = models.cvp(clf, X, Y, groups, method='predict_proba', n_splits=5, n_jobs=1)

        assert len(Yp) == len(Y)
        assert Yp.shape[1] == 2  # Two classes
        # Probabilities should sum to 1
        assert np.allclose(Yp.sum(axis=1), 1.0)


class TestPredictFromFrame:
    """Tests for StepCounter.predict_from_frame() method."""

    @pytest.fixture
    def trained_rf_model(self, sample_rate):
        """Create a trained RF model for predict_from_frame tests."""
        np.random.seed(42)
        n_windows = 40
        window_len = 10 * sample_rate

        # Generate training data
        X = []
        Y = []  # step counts
        groups = []

        for i in range(n_windows):
            if i % 2 == 0:  # Walking
                t = np.arange(window_len) / sample_rate
                x = 0.3 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, window_len)
                y = 0.2 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, window_len)
                z = 1.0 + 0.1 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, window_len)
                steps = np.random.randint(10, 20)
            else:  # Resting
                x = np.random.normal(0, 0.02, window_len)
                y = np.random.normal(0, 0.02, window_len)
                z = np.random.normal(1.0, 0.02, window_len)
                steps = 0

            X.append(np.column_stack([x, y, z]))
            Y.append(steps)
            groups.append(f"subj_{i // 10}")

        X = np.array(X)
        Y = np.array(Y)
        groups = np.array(groups)

        # Create and fit the model
        model = models.StepCounter(
            wd_type='rf',
            sample_rate=sample_rate,
            window_sec=10,
            cv=2,
            verbose=False
        )
        model.fit(X, Y, groups=groups)
        return model

    def test_predict_from_frame_shape(self, trained_rf_model, accel_data_1_5_days, sample_rate):
        """Test predict_from_frame returns correct types."""
        model = trained_rf_model

        # Use first 30 minutes of data for faster test
        data_subset = accel_data_1_5_days.iloc[:sample_rate * 1800]

        Y, W, T_steps = model.predict_from_frame(data_subset)

        assert isinstance(Y, pd.Series)
        assert isinstance(W, pd.Series)
        assert isinstance(T_steps, pd.Series)
        assert Y.name == 'Steps'
        assert W.name == 'Walk'
        # Should have some windows
        assert len(Y) > 0
        assert len(W) == len(Y)


class TestStepCounterFit:
    """Tests for StepCounter.fit() method."""

    @pytest.fixture
    def training_data(self, sample_rate):
        """Generate minimal training data for fit tests."""
        np.random.seed(42)
        n_windows = 50
        window_len = 10 * sample_rate  # 10 second windows

        # Generate windows with mixed walking/non-walking
        X = []
        Y = []  # step counts
        groups = []

        for i in range(n_windows):
            if i % 3 == 0:  # Walking windows
                t = np.arange(window_len) / sample_rate
                x = 0.3 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, window_len)
                y = 0.2 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, window_len)
                z = 1.0 + 0.1 * np.sin(2 * np.pi * 1.8 * t) + np.random.normal(0, 0.05, window_len)
                steps = np.random.randint(10, 20)
            else:  # Non-walking windows
                x = np.random.normal(0, 0.02, window_len)
                y = np.random.normal(0, 0.02, window_len)
                z = np.random.normal(1.0, 0.02, window_len)
                steps = 0

            X.append(np.column_stack([x, y, z]))
            Y.append(steps)
            groups.append(f"subj_{i // 10}")

        return np.array(X), np.array(Y), np.array(groups)

    def test_fit_rf_basic(self, training_data, sample_rate):
        """Test basic RF model fitting."""
        X, Y, groups = training_data

        model = models.StepCounter(
            wd_type='rf',
            sample_rate=sample_rate,
            window_sec=10,
            cv=2,
            verbose=False
        )

        model.fit(X, Y, groups=groups)

        # Verify model was trained
        assert model.find_peaks_params is not None
        assert 'distance' in model.find_peaks_params
        assert 'prominence' in model.find_peaks_params
        assert model.cv_results is not None
        assert 'walk_detector' in model.cv_results
        assert 'step_counter' in model.cv_results

    def test_fit_sets_cv_results(self, training_data, sample_rate):
        """Test that fit populates cross-validation results."""
        X, Y, groups = training_data

        model = models.StepCounter(
            wd_type='rf',
            sample_rate=sample_rate,
            window_sec=10,
            cv=2,
            verbose=False
        )

        model.fit(X, Y, groups=groups)

        # Check CV results structure
        cv_results = model.cv_results
        assert 'test_indices' in cv_results
        assert len(cv_results['test_indices']) == 2  # cv=2

        # Walk detector scores
        wd_scores = cv_results['walk_detector']['scores']
        assert 'accuracy' in wd_scores
        assert 'f1' in wd_scores

        # Step counter scores
        sc_scores = cv_results['step_counter']['scores']
        assert 'mae' in sc_scores
        assert 'rmse' in sc_scores


class TestBoundaryValues:
    """Tests for boundary value conditions in model utilities."""

    def test_calc_sample_weight_pnr_zero(self):
        """Test calc_sample_weight with pnr=0 (edge case)."""
        yt = np.array([0, 0, 1, 1])

        # pnr=0 causes division by zero in the formula:
        # sample_weight[yt == 0] = (yt == 1).sum() / (pnr * (yt == 0).sum())
        # This is a known edge case - the function will raise ZeroDivisionError
        # or produce inf values
        with pytest.warns(RuntimeWarning, match="divide by zero|invalid value"):
            result = models.calc_sample_weight(yt, pnr=0)
            # Result will have inf or nan values
            assert np.isinf(result).any() or np.isnan(result).any()

    def test_calc_sample_weight_no_negatives(self):
        """Test calc_sample_weight when all samples are positive."""
        yt = np.array([1, 1, 1, 1])  # No negative samples

        # (yt == 0).sum() = 0, causes division by zero
        # But since yt == 0 selects no elements, the assignment doesn't happen
        # for any elements, so no actual error occurs
        result = models.calc_sample_weight(yt, pnr=1.0)

        # All weights should be 1.0 (no negatives to reweight)
        assert np.allclose(result, 1.0)

    def test_calc_sample_weight_no_positives(self):
        """Test calc_sample_weight when all samples are negative."""
        yt = np.array([0, 0, 0, 0])  # No positive samples

        # (yt == 1).sum() = 0, so numerator is 0
        # Result: 0 / (pnr * (yt == 0).sum()) = 0
        result = models.calc_sample_weight(yt, pnr=1.0)

        # All weights should be 0 (no positives)
        assert np.allclose(result, 0.0)

    def test_calc_sample_weight_pnr_none(self):
        """Test calc_sample_weight with pnr=None (default)."""
        yt = np.array([0, 0, 1, 1])

        result = models.calc_sample_weight(yt, pnr=None)

        # Should return all ones
        assert np.allclose(result, 1.0)

    def test_calibrate_all_positive(self):
        """Test calibrate when all true labels are positive."""
        yp = np.array([0.6, 0.7, 0.8, 0.9])
        yt = np.array([1, 1, 1, 1])  # All positive

        # precision_recall_curve needs both classes
        # This should either raise an error or handle gracefully
        try:
            result = models.calibrate(yp, yt)
            # If it doesn't raise, check the result is valid
            assert 'best_f1' in result
        except (ValueError, IndexError):
            # Expected - sklearn can't compute precision-recall with one class
            pass

    def test_calibrate_all_negative(self):
        """Test calibrate when all true labels are negative."""
        yp = np.array([0.1, 0.2, 0.3, 0.4])
        yt = np.array([0, 0, 0, 0])  # All negative

        # Same edge case as all positive
        try:
            result = models.calibrate(yp, yt)
            assert 'best_f1' in result
        except (ValueError, IndexError):
            # Expected
            pass
