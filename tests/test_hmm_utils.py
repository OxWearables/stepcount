"""
Tests for stepcount.hmm_utils module.

Tests cover:
- HMM smoother class
- Viterbi algorithm
- Transition/emission/prior computation
- Group handling
"""
import pytest
import numpy as np

from stepcount import hmm_utils


class TestComputeTransition:
    """Tests for transition matrix computation."""

    def test_transition_basic(self):
        """Test transition matrix computation on simple sequence."""
        # Sequence: 0, 0, 1, 1, 0
        Y = np.array([0, 0, 1, 1, 0])

        trans = hmm_utils.compute_transition(Y)

        assert trans.shape == (2, 2)
        # Rows should sum to 1
        assert np.allclose(trans.sum(axis=1), [1.0, 1.0])

    def test_transition_with_labels(self):
        """Test transition matrix with explicit labels."""
        Y = np.array([0, 0, 1, 1, 0, 0, 1])
        labels = np.array([0, 1])

        trans = hmm_utils.compute_transition(Y, labels=labels)

        assert trans.shape == (2, 2)
        # Verify transitions: 0->0 (2), 0->1 (2), 1->1 (1), 1->0 (1)
        # From 0: [2/4, 2/4] = [0.5, 0.5]
        # From 1: [1/2, 1/2] = [0.5, 0.5]
        assert np.allclose(trans, [[0.5, 0.5], [0.5, 0.5]])

    def test_transition_with_groups(self):
        """Test transition matrix respects group boundaries."""
        # Two separate sequences
        Y = np.array([0, 1, 0, 1, 1, 0])
        groups = np.array(['A', 'A', 'A', 'B', 'B', 'B'])

        trans = hmm_utils.compute_transition(Y, groups=groups)

        # Transitions within groups only
        # Group A: 0->1, 1->0
        # Group B: 1->1, 1->0
        assert trans.shape == (2, 2)
        assert np.allclose(trans.sum(axis=1), [1.0, 1.0])

    def test_transition_single_state(self):
        """Test transition when sequence is all one state."""
        Y = np.array([1, 1, 1, 1])

        trans = hmm_utils.compute_transition(Y)

        # With only one unique state, the transition matrix is 1x1
        # The single state always transitions to itself
        assert trans.shape == (1, 1)
        assert trans[0, 0] == pytest.approx(1.0)


class TestComputeEmission:
    """Tests for emission matrix computation."""

    def test_emission_perfect_predictions(self):
        """Test emission matrix with perfect predictions."""
        Y_true = np.array([0, 0, 1, 1, 0])
        Y_pred = np.array([0, 0, 1, 1, 0])

        emission = hmm_utils.compute_emission(Y_pred, Y_true)

        # Perfect predictions: emission[i, i] = 1
        assert np.allclose(emission, np.eye(2))

    def test_emission_probabilistic(self):
        """Test emission matrix with probability predictions."""
        Y_true = np.array([0, 0, 1, 1])
        # Probability predictions: [P(0), P(1)]
        Y_pred = np.array([
            [0.9, 0.1],  # true 0
            [0.8, 0.2],  # true 0
            [0.1, 0.9],  # true 1
            [0.2, 0.8],  # true 1
        ])

        emission = hmm_utils.compute_emission(Y_pred, Y_true)

        # emission[0] = mean of predictions where true = 0 = [0.85, 0.15]
        # emission[1] = mean of predictions where true = 1 = [0.15, 0.85]
        assert np.allclose(emission[0], [0.85, 0.15])
        assert np.allclose(emission[1], [0.15, 0.85])

    def test_emission_shape(self):
        """Test emission matrix has correct shape."""
        Y_true = np.array([0, 0, 1, 1, 0, 1])
        Y_pred = np.array([0, 1, 1, 1, 0, 0])

        emission = hmm_utils.compute_emission(Y_pred, Y_true)

        assert emission.shape == (2, 2)


class TestComputePrior:
    """Tests for prior probability computation."""

    def test_prior_uniform(self):
        """Test uniform prior probabilities."""
        Y_true = np.array([0, 0, 1, 1, 1])

        prior = hmm_utils.compute_prior(Y_true, uniform=True)

        assert np.allclose(prior, [0.5, 0.5])

    def test_prior_observed(self):
        """Test prior from observed rates."""
        Y_true = np.array([0, 0, 0, 1, 1])  # 60% 0, 40% 1

        prior = hmm_utils.compute_prior(Y_true, uniform=False)

        assert np.allclose(prior, [0.6, 0.4])

    def test_prior_with_labels(self):
        """Test prior computation with explicit labels."""
        Y_true = np.array([0, 1, 1, 1])
        labels = np.array([0, 1])

        prior = hmm_utils.compute_prior(Y_true, labels=labels, uniform=True)

        assert np.allclose(prior, [0.5, 0.5])


class TestViterbi:
    """Tests for Viterbi algorithm."""

    def test_viterbi_basic(self):
        """Test Viterbi on simple sequence."""
        Y = np.array([0, 0, 1, 0, 1, 1])

        hmm_params = {
            'prior': np.array([0.5, 0.5]),
            'emission': np.array([[0.9, 0.1], [0.1, 0.9]]),
            'transition': np.array([[0.8, 0.2], [0.2, 0.8]]),
            'labels': np.array([0, 1]),
        }

        result = hmm_utils.viterbi(Y, hmm_params)

        assert len(result) == len(Y)
        assert set(result).issubset({0, 1})

    def test_viterbi_smoothing(self):
        """Test that Viterbi smooths noisy predictions."""
        # Noisy sequence with isolated flip
        Y = np.array([0, 0, 0, 1, 0, 0, 0])

        hmm_params = {
            'prior': np.array([0.5, 0.5]),
            'emission': np.array([[0.8, 0.2], [0.2, 0.8]]),
            'transition': np.array([[0.9, 0.1], [0.1, 0.9]]),  # Strong self-transition
            'labels': np.array([0, 1]),
        }

        result = hmm_utils.viterbi(Y, hmm_params)

        # With strong self-transition, isolated 1 might be smoothed to 0
        # (depends on emission probabilities)
        assert len(result) == len(Y)

    def test_viterbi_empty_sequence(self):
        """Test Viterbi handles empty sequence."""
        Y = np.array([])

        hmm_params = {
            'prior': np.array([0.5, 0.5]),
            'emission': np.array([[0.9, 0.1], [0.1, 0.9]]),
            'transition': np.array([[0.8, 0.2], [0.2, 0.8]]),
            'labels': np.array([0, 1]),
        }

        result = hmm_utils.viterbi(Y, hmm_params)

        assert len(result) == 0

    def test_viterbi_deterministic(self):
        """Test Viterbi produces deterministic results."""
        Y = np.array([0, 1, 0, 1, 1, 0])

        hmm_params = {
            'prior': np.array([0.5, 0.5]),
            'emission': np.array([[0.9, 0.1], [0.1, 0.9]]),
            'transition': np.array([[0.7, 0.3], [0.3, 0.7]]),
            'labels': np.array([0, 1]),
        }

        result1 = hmm_utils.viterbi(Y, hmm_params)
        result2 = hmm_utils.viterbi(Y, hmm_params)

        assert np.array_equal(result1, result2)


class TestHMMSmoother:
    """Tests for HMMSmoother class."""

    def test_hmm_smoother_fit_predict(self):
        """Test HMMSmoother fit and predict."""
        np.random.seed(42)

        # Training data
        Y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        Y_pred = Y_true.copy()
        # Add some noise
        Y_pred[3] = 0  # Wrong prediction

        smoother = hmm_utils.HMMSmoother()
        smoother.fit(Y_pred, Y_true)

        # Predict on new data
        Y_test = np.array([0, 1, 0, 1, 1])
        result = smoother.predict(Y_test)

        assert len(result) == len(Y_test)
        assert set(result).issubset({0, 1})

    def test_hmm_smoother_with_groups(self):
        """Test HMMSmoother respects groups during prediction."""
        Y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        Y_pred = Y_true.copy()
        groups_train = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        smoother = hmm_utils.HMMSmoother()
        smoother.fit(Y_pred, Y_true, groups=groups_train)

        # Predict with groups
        Y_test = np.array([0, 1, 0, 1])
        groups_test = np.array(['C', 'C', 'D', 'D'])
        result = smoother.predict(Y_test, groups=groups_test)

        assert len(result) == len(Y_test)

    def test_hmm_smoother_custom_params(self):
        """Test HMMSmoother with custom parameters."""
        smoother = hmm_utils.HMMSmoother(
            startprob=np.array([0.8, 0.2]),
            transmat=np.array([[0.9, 0.1], [0.1, 0.9]]),
        )

        Y_true = np.array([0, 0, 1, 1])
        Y_pred = Y_true.copy()

        smoother.fit(Y_pred, Y_true)

        # Should use provided parameters
        assert np.allclose(smoother.startprob, [0.8, 0.2])
        assert np.allclose(smoother.transmat, [[0.9, 0.1], [0.1, 0.9]])

    def test_hmm_smoother_learns_params(self):
        """Test HMMSmoother learns parameters from data."""
        Y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
        Y_pred = Y_true.copy()

        smoother = hmm_utils.HMMSmoother()
        smoother.fit(Y_pred, Y_true)

        # Should have learned parameters
        assert smoother.startprob is not None
        assert smoother.emissionprob is not None
        assert smoother.transmat is not None

        # Transition matrix should reflect the data
        # High self-transition (mostly 0->0 and 1->1)
        assert smoother.transmat[0, 0] > 0.5
        assert smoother.transmat[1, 1] > 0.5


class TestOrderedUnique:
    """Tests for ordered_unique utility."""

    def test_ordered_unique_preserves_order(self):
        """Test that unique values are returned in order of first appearance."""
        x = np.array(['B', 'A', 'C', 'A', 'B', 'D'])

        result = hmm_utils.ordered_unique(x)

        assert list(result) == ['B', 'A', 'C', 'D']

    def test_ordered_unique_numeric(self):
        """Test ordered_unique with numeric values."""
        x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 3])

        result = hmm_utils.ordered_unique(x)

        assert list(result) == [3, 1, 4, 5, 9, 2, 6]

    def test_ordered_unique_already_unique(self):
        """Test ordered_unique on already unique array."""
        x = np.array(['A', 'B', 'C'])

        result = hmm_utils.ordered_unique(x)

        assert list(result) == ['A', 'B', 'C']


class TestLengthsFromGroups:
    """Tests for lengths_from_groups utility."""

    def test_lengths_from_groups_basic(self):
        """Test lengths computation from groups."""
        groups = np.array(['A', 'A', 'A', 'B', 'B', 'C'])

        lengths = hmm_utils.lengths_from_groups(groups)

        assert list(lengths) == [3, 2, 1]

    def test_lengths_from_groups_single_group(self):
        """Test lengths with single group returns None."""
        groups = np.array(['A', 'A', 'A'])

        lengths = hmm_utils.lengths_from_groups(groups)

        assert lengths is None

    def test_lengths_from_groups_none(self):
        """Test lengths with None input returns None."""
        lengths = hmm_utils.lengths_from_groups(None)

        assert lengths is None

    def test_lengths_from_groups_order_preserved(self):
        """Test that lengths are in order of appearance."""
        groups = np.array(['C', 'C', 'A', 'A', 'A', 'B'])

        lengths = hmm_utils.lengths_from_groups(groups)

        assert list(lengths) == [2, 3, 1]


class TestHMMSmootherIntegration:
    """Integration tests for HMM smoothing."""

    def test_full_pipeline_noisy_data(self):
        """Test HMM smoothing on noisy prediction data."""
        np.random.seed(42)

        # Generate true labels with clear pattern
        true_labels = np.array([0]*20 + [1]*20 + [0]*20)

        # Add 20% noise to predictions
        pred_labels = true_labels.copy()
        noise_idx = np.random.choice(len(pred_labels), size=12, replace=False)
        pred_labels[noise_idx] = 1 - pred_labels[noise_idx]

        # Fit smoother
        smoother = hmm_utils.HMMSmoother()
        smoother.fit(pred_labels, true_labels)

        # Smooth predictions
        smoothed = smoother.predict(pred_labels)

        # Smoothed should be closer to true than noisy predictions
        noisy_accuracy = (pred_labels == true_labels).mean()
        smooth_accuracy = (smoothed == true_labels).mean()

        # Smoothing should improve or maintain accuracy
        assert smooth_accuracy >= noisy_accuracy

    def test_hmm_smoother_with_probabilities(self):
        """Test HMM smoothing with probability predictions."""
        # Probability predictions
        Y_true = np.array([0, 0, 1, 1, 0, 0])
        Y_pred_proba = np.array([
            [0.9, 0.1],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.6, 0.4],
        ])

        smoother = hmm_utils.HMMSmoother()
        smoother.fit(Y_pred_proba, Y_true)

        # Convert to class predictions for testing
        Y_test = np.array([0, 1, 0])
        result = smoother.predict(Y_test)

        assert len(result) == 3
        assert set(result).issubset({0, 1})


# Skip all hmmlearn tests if hmmlearn is not installed
pytest.importorskip("hmmlearn", reason="hmmlearn not installed")


class TestHMMSmootherHMMLearn:
    """Tests for HMMSmoother with use_hmmlearn=True path."""

    def test_hmmlearn_fit_predict_basic(self):
        """Test HMMSmoother with hmmlearn backend."""
        np.random.seed(42)

        # Training data with clear pattern
        Y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        Y_pred = Y_true.copy()

        smoother = hmm_utils.HMMSmoother(
            use_hmmlearn=True,
            n_components=2,
            n_trials=5,  # Fewer trials for speed
            n_iter=10,
        )
        smoother.fit(Y_pred, Y_true)

        # Predict on test data
        Y_test = np.array([0, 0, 1, 1, 0, 1])
        result = smoother.predict(Y_test)

        assert len(result) == len(Y_test)
        assert set(result).issubset({0, 1})
        # Check that hmmlearn-specific attributes were set
        assert hasattr(smoother, 'hmm')
        assert hasattr(smoother, 'score')

    def test_hmmlearn_fit_predict_with_groups(self):
        """Test HMMSmoother hmmlearn with groups (stratify_groups=False)."""
        np.random.seed(42)

        # Larger dataset with more variation per group
        Y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
        Y_pred = Y_true.copy()
        groups_train = np.array(['A']*8 + ['B']*8)

        smoother = hmm_utils.HMMSmoother(
            use_hmmlearn=True,
            n_components=2,
            stratify_groups=False,  # Process all groups together
            n_trials=3,
            n_iter=10,
        )
        smoother.fit(Y_pred, Y_true, groups=groups_train)

        # Predict with groups
        Y_test = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        groups_test = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
        result = smoother.predict(Y_test, groups=groups_test)

        assert len(result) == len(Y_test)
        # With stratify_groups=False, hmm is a single object
        assert smoother.hmm is not None

    def test_hmmlearn_predict_proba(self):
        """Test HMMSmoother hmmlearn predict_proba method."""
        np.random.seed(42)

        Y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        Y_pred = Y_true.copy()

        smoother = hmm_utils.HMMSmoother(
            use_hmmlearn=True,
            n_components=2,
            n_trials=3,
            n_iter=10,
            stratify_groups=False,  # Need this for predict_proba without groups
        )
        smoother.fit(Y_pred, Y_true)

        # predict_proba should return probability matrix
        Y_test = np.array([0, 0, 1, 1])
        result = smoother.predict_proba(Y_test)

        assert result.shape == (len(Y_test), 2)
        # Each row should sum to 1 (probabilities)
        assert np.allclose(result.sum(axis=1), 1.0)

    def test_hmmlearn_with_custom_priors(self):
        """Test HMMSmoother hmmlearn with custom start/transition probabilities."""
        Y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        Y_pred = Y_true.copy()

        smoother = hmm_utils.HMMSmoother(
            use_hmmlearn=True,
            n_components=2,
            startprob=np.array([0.8, 0.2]),
            transmat=np.array([[0.9, 0.1], [0.1, 0.9]]),
            n_trials=3,
            n_iter=10,
        )
        smoother.fit(Y_pred, Y_true)

        Y_test = np.array([0, 1, 0, 1])
        result = smoother.predict(Y_test)

        assert len(result) == len(Y_test)

    def test_hmmlearn_train_test_split(self):
        """Test HMMSmoother hmmlearn with train_test_split=True."""
        np.random.seed(42)

        # Need enough data for splitting
        Y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
        Y_pred = Y_true.copy()

        smoother = hmm_utils.HMMSmoother(
            use_hmmlearn=True,
            n_components=2,
            train_test_split=True,
            n_trials=3,
            n_iter=10,
            stratify_groups=False,
        )
        smoother.fit(Y_pred, Y_true)

        Y_test = np.array([0, 1, 0, 1, 0, 0])
        result = smoother.predict(Y_test)

        assert len(result) == len(Y_test)


class TestHMMLearnFitPredict:
    """Tests for the hmmlearn_fit_predict standalone function."""

    def test_hmmlearn_fit_predict_basic(self):
        """Test basic hmmlearn_fit_predict call with provided parameters."""
        np.random.seed(42)

        Y = np.array([[0], [0], [1], [1], [0], [0], [1], [1]])

        # Provide normalized parameters to avoid random initialization issues
        hmm, score, Y_pred = hmm_utils.hmmlearn_fit_predict(
            Y,
            n_components=2,
            n_trials=0,  # Skip random trials
            n_iter=10,
            startprob=np.array([0.5, 0.5]),
            transmat=np.array([[0.8, 0.2], [0.2, 0.8]]),
            emissionprob=np.array([[0.9, 0.1], [0.1, 0.9]]),
        )

        assert Y_pred.shape[0] == len(Y)
        assert score is not None
        assert hmm is not None

    def test_hmmlearn_fit_predict_with_groups(self):
        """Test hmmlearn_fit_predict with group lengths."""
        np.random.seed(42)

        Y = np.array([[0], [0], [1], [1], [0], [0], [1], [1]])
        groups = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

        # Provide normalized parameters
        hmm, score, Y_pred = hmm_utils.hmmlearn_fit_predict(
            Y,
            groups=groups,
            n_components=2,
            n_trials=0,
            n_iter=10,
            startprob=np.array([0.5, 0.5]),
            transmat=np.array([[0.8, 0.2], [0.2, 0.8]]),
            emissionprob=np.array([[0.9, 0.1], [0.1, 0.9]]),
        )

        assert Y_pred.shape[0] == len(Y)

    def test_hmmlearn_fit_predict_method_predict_proba(self):
        """Test hmmlearn_fit_predict with method='predict_proba'."""
        np.random.seed(42)

        Y = np.array([[0], [0], [1], [1], [0], [0], [1], [1], [0], [1]])

        # Provide normalized parameters
        hmm, score, Y_pred = hmm_utils.hmmlearn_fit_predict(
            Y,
            n_components=2,
            method='predict_proba',
            n_trials=0,
            n_iter=10,
            startprob=np.array([0.5, 0.5]),
            transmat=np.array([[0.8, 0.2], [0.2, 0.8]]),
            emissionprob=np.array([[0.9, 0.1], [0.1, 0.9]]),
        )

        # predict_proba returns probability matrix
        assert Y_pred.shape == (len(Y), 2)
        assert np.allclose(Y_pred.sum(axis=1), 1.0)


class TestHMMErrorPaths:
    """Tests for error handling in HMM utilities."""

    def test_viterbi_single_element(self):
        """Test Viterbi handles single-element sequence."""
        Y = np.array([1])

        hmm_params = {
            'prior': np.array([0.5, 0.5]),
            'emission': np.array([[0.9, 0.1], [0.1, 0.9]]),
            'transition': np.array([[0.8, 0.2], [0.2, 0.8]]),
            'labels': np.array([0, 1]),
        }

        result = hmm_utils.viterbi(Y, hmm_params)

        assert len(result) == 1
        assert result[0] in [0, 1]

    def test_hmmlearn_requires_n_components(self):
        """Test HMMSmoother with hmmlearn requires n_components."""
        with pytest.raises(AssertionError, match="n_components"):
            hmm_utils.HMMSmoother(use_hmmlearn=True)

    def test_hmmlearn_n_components_mismatch(self):
        """Test HMMSmoother raises when n_components doesn't match labels."""
        smoother = hmm_utils.HMMSmoother(
            use_hmmlearn=True,
            n_components=3,  # But data only has 2 labels
        )

        Y_true = np.array([0, 0, 1, 1])
        Y_pred = Y_true.copy()

        with pytest.raises(AssertionError, match="n_components"):
            smoother.fit(Y_pred, Y_true)

    def test_compute_transition_all_same_label(self):
        """Test compute_transition with only one unique label."""
        Y = np.array([1, 1, 1, 1])

        trans = hmm_utils.compute_transition(Y)

        # Single state should have 100% self-transition
        assert trans.shape == (1, 1)
        assert trans[0, 0] == pytest.approx(1.0)
