"""Tests for sslmodel.py - SSL model utilities."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import DataLoader

from stepcount import sslmodel


class TestRandomSwitchAxis:
    """Tests for axis permutation transform."""

    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        transform = sslmodel.RandomSwitchAxis()
        # Input: (3, time) - 3 axes, 100 time samples
        sample = torch.randn(3, 100)
        result = transform(sample)
        assert result.shape == sample.shape

    def test_output_is_permutation(self):
        """Output should be a valid permutation of input axes."""
        transform = sslmodel.RandomSwitchAxis()
        sample = torch.tensor([
            [1.0, 2.0, 3.0],  # x
            [4.0, 5.0, 6.0],  # y
            [7.0, 8.0, 9.0],  # z
        ])
        result = transform(sample)

        # Each row in output should be one of the original rows
        original_rows = {tuple(sample[i].tolist()) for i in range(3)}
        result_rows = {tuple(result[i].tolist()) for i in range(3)}
        assert original_rows == result_rows


class TestRotationAxis:
    """Tests for rotation transform."""

    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        transform = sslmodel.RotationAxis()
        # Input: (3, time)
        sample = np.random.randn(3, 100).astype('f4')
        result = transform(sample)
        assert result.shape == sample.shape

    def test_rotation_preserves_magnitude(self):
        """Rotation should approximately preserve vector magnitudes."""
        np.random.seed(42)
        transform = sslmodel.RotationAxis()
        sample = np.random.randn(3, 50).astype('f4')

        # Compute magnitude per time step
        orig_mag = np.linalg.norm(sample, axis=0)

        result = transform(sample)
        result_mag = np.linalg.norm(result, axis=0)

        # Magnitudes should be approximately equal
        assert np.allclose(orig_mag, result_mag, rtol=1e-5)


class TestRandomDecimation:
    """Tests for decimation transform."""

    def test_output_shape_preserved(self):
        """Output shape should match input shape after interpolation."""
        transform = sslmodel.RandomDecimation()
        sample = torch.randn(3, 100)
        result = transform(sample)
        assert result.shape == sample.shape

    def test_multiple_calls_vary(self):
        """Multiple calls should produce different results (due to random decimation)."""
        transform = sslmodel.RandomDecimation()
        sample = torch.randn(3, 100)

        results = [transform(sample.clone()) for _ in range(10)]

        # At least some results should differ
        differs = sum(1 for i in range(1, 10) if not torch.allclose(results[0], results[i]))
        assert differs > 0


class TestNormalDataset:
    """Tests for NormalDataset class."""

    def test_len(self):
        """Dataset length should match input array length."""
        X = np.random.randn(10, 100, 3).astype('f4')  # 10 samples, 100 time, 3 axes
        dataset = sslmodel.NormalDataset(X)
        assert len(dataset) == 10

    def test_getitem_returns_tuple(self):
        """__getitem__ should return (sample, label, pid) tuple."""
        X = np.random.randn(5, 100, 3).astype('f4')
        dataset = sslmodel.NormalDataset(X)
        result = dataset[0]
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_getitem_with_labels(self):
        """Dataset should return correct labels when provided."""
        X = np.random.randn(5, 100, 3).astype('f4')
        y = np.array([0, 1, 0, 1, 0])
        dataset = sslmodel.NormalDataset(X, y=y)

        sample, label, pid = dataset[2]
        assert label.item() == 0

    def test_getitem_without_labels(self):
        """Dataset should return NaN label when not provided."""
        X = np.random.randn(5, 100, 3).astype('f4')
        dataset = sslmodel.NormalDataset(X)

        sample, label, pid = dataset[0]
        assert np.isnan(label)

    def test_tensor_conversion(self):
        """Samples should be converted to torch tensors."""
        X = np.random.randn(5, 100, 3).astype('f4')
        dataset = sslmodel.NormalDataset(X)

        sample, _, _ = dataset[0]
        assert isinstance(sample, torch.Tensor)

    def test_transpose_channels_first(self):
        """With transpose_channels_first=True, output should be (channels, time)."""
        X = np.random.randn(5, 100, 3).astype('f4')  # (batch, time, channels)
        dataset = sslmodel.NormalDataset(X, transpose_channels_first=True)

        sample, _, _ = dataset[0]
        # Should be (3, 100) - channels first
        assert sample.shape == (3, 100)

    def test_no_transpose(self):
        """With transpose_channels_first=False, output keeps original shape."""
        X = np.random.randn(5, 100, 3).astype('f4')  # (batch, time, channels)
        dataset = sslmodel.NormalDataset(X, transpose_channels_first=False)

        sample, _, _ = dataset[0]
        # Should be (100, 3) - original shape
        assert sample.shape == (100, 3)


class TestEarlyStopping:
    """Tests for EarlyStopping utility."""

    def test_initial_state(self):
        """Initial state should have no early stop triggered."""
        es = sslmodel.EarlyStopping(patience=5)
        assert es.early_stop is False
        assert es.counter == 0
        assert es.best_score is None

    def test_improvement_resets_counter(self, tmp_path):
        """Improvement in validation loss should reset counter."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        es = sslmodel.EarlyStopping(patience=3, verbose=False, path=str(checkpoint_path))

        # Use real simple model instead of mock (torch.save needs picklable object)
        model = nn.Linear(10, 2)

        # First call sets baseline
        es(val_loss=1.0, model=model)
        assert es.counter == 0

        # Improvement (lower loss)
        es(val_loss=0.8, model=model)
        assert es.counter == 0
        assert es.early_stop is False

    def test_no_improvement_increments_counter(self, tmp_path):
        """No improvement should increment counter."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        es = sslmodel.EarlyStopping(patience=3, verbose=False, path=str(checkpoint_path))

        model = nn.Linear(10, 2)

        es(val_loss=1.0, model=model)
        es(val_loss=1.1, model=model)  # Worse
        assert es.counter == 1

        es(val_loss=1.2, model=model)  # Worse
        assert es.counter == 2

    def test_patience_exceeded_stops(self, tmp_path):
        """Exceeding patience should trigger early stop."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        es = sslmodel.EarlyStopping(patience=2, verbose=False, path=str(checkpoint_path))

        model = nn.Linear(10, 2)

        es(val_loss=1.0, model=model)
        es(val_loss=1.1, model=model)  # counter=1
        es(val_loss=1.2, model=model)  # counter=2, triggers stop

        assert es.early_stop is True

    def test_checkpoint_saved_on_improvement(self, tmp_path):
        """Model checkpoint should be saved when loss improves."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        es = sslmodel.EarlyStopping(patience=3, path=str(checkpoint_path), verbose=False)

        model = nn.Linear(10, 2)

        es(val_loss=1.0, model=model)
        assert checkpoint_path.exists()


class TestGetSSLNet:
    """Tests for get_sslnet function with mocked torch.hub."""

    @patch('stepcount.sslmodel.torch.hub.load')
    @patch('stepcount.sslmodel.torch_cache_path')
    def test_returns_module(self, mock_cache_path, mock_hub_load):
        """get_sslnet should return an nn.Module."""
        mock_cache_path.exists.return_value = True
        mock_cache_path.iterdir.return_value = []

        mock_model = Mock(spec=nn.Module)
        mock_hub_load.return_value = mock_model

        result = sslmodel.get_sslnet()

        assert result == mock_model
        mock_hub_load.assert_called_once()

    @patch('stepcount.sslmodel.torch.hub.load')
    @patch('stepcount.sslmodel.torch_cache_path')
    def test_pretrained_flag_passed(self, mock_cache_path, mock_hub_load):
        """pretrained flag should be passed to torch.hub.load."""
        mock_cache_path.exists.return_value = True
        mock_cache_path.iterdir.return_value = []
        mock_hub_load.return_value = Mock(spec=nn.Module)

        sslmodel.get_sslnet(pretrained=True)

        call_kwargs = mock_hub_load.call_args[1]
        assert call_kwargs['pretrained'] is True


class TestPredict:
    """Tests for predict function."""

    def test_empty_dataloader(self):
        """Empty dataloader should return empty arrays."""
        mock_model = Mock()
        empty_loader = []

        y_true, y_pred, pids = sslmodel.predict(mock_model, empty_loader, 'cpu')

        assert len(y_true) == 0
        assert len(y_pred) == 0
        assert len(pids) == 0

    def test_single_batch(self):
        """Single batch prediction should work correctly."""
        # Create simple model that handles (batch, channels, time) input
        # Dataset with transpose_channels_first=True returns (3, 10) per sample
        # Batched: (4, 3, 10) -> flatten -> (4, 30) -> Linear(30, 2) -> (4, 2)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30, 2)
        )

        # Create dataset and dataloader
        X = np.random.randn(4, 10, 3).astype('f4')  # 4 samples, 10 time, 3 axes
        y = np.array([0, 1, 0, 1])
        pids = np.array([1, 2, 3, 4])
        dataset = sslmodel.NormalDataset(X, y=y, pid=pids, transpose_channels_first=True)
        loader = DataLoader(dataset, batch_size=4)

        y_true, y_pred, pid_out = sslmodel.predict(model, loader, 'cpu')

        assert len(y_true) == 4
        assert len(y_pred) == 4
        assert len(pid_out) == 4

    def test_output_logits_flag(self):
        """output_logits=True should return raw logits instead of class predictions."""
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30, 2)
        )

        X = np.random.randn(4, 10, 3).astype('f4')
        y = np.array([0, 1, 0, 1])
        dataset = sslmodel.NormalDataset(X, y=y, transpose_channels_first=True)
        loader = DataLoader(dataset, batch_size=4)

        _, y_pred_classes, _ = sslmodel.predict(model, loader, 'cpu', output_logits=False)
        _, y_pred_logits, _ = sslmodel.predict(model, loader, 'cpu', output_logits=True)

        # Classes should be 1D (class indices)
        assert y_pred_classes.ndim == 1

        # Logits should be 2D (batch, num_classes)
        assert y_pred_logits.ndim == 2
        assert y_pred_logits.shape[1] == 2


class TestValidateModel:
    """Tests for _validate_model function."""

    def test_returns_loss_and_accuracy(self):
        """_validate_model should return mean loss and accuracy."""
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30, 2)
        )
        loss_fn = nn.CrossEntropyLoss()

        X = np.random.randn(8, 10, 3).astype('f4')
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        dataset = sslmodel.NormalDataset(X, y=y, transpose_channels_first=True)
        loader = DataLoader(dataset, batch_size=4)

        loss, acc = sslmodel._validate_model(model, loader, 'cpu', loss_fn)

        assert isinstance(loss, (float, np.floating))
        assert isinstance(acc, (float, np.floating))
        assert 0 <= acc <= 1


class TestTrain:
    """Tests for train function."""

    @patch('stepcount.sslmodel._validate_model')
    @patch('stepcount.sslmodel.verbose', False)
    def test_single_epoch(self, mock_validate, tmp_path):
        """Train should complete at least one epoch."""
        mock_validate.return_value = (0.5, 0.8)

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30, 2)
        )
        X = np.random.randn(8, 10, 3).astype('f4')
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        dataset = sslmodel.NormalDataset(X, y=y, transpose_channels_first=True)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        weights_path = tmp_path / "weights.pt"

        result = sslmodel.train(
            model, train_loader, val_loader, 'cpu',
            weights_path=str(weights_path),
            num_epoch=1,
            patience=10
        )

        assert result is not None
        mock_validate.assert_called()

    @patch('stepcount.sslmodel._validate_model')
    @patch('stepcount.sslmodel.verbose', False)
    def test_early_stopping_triggered(self, mock_validate, tmp_path):
        """Training should stop early when patience is exceeded."""
        # Return worsening loss to trigger early stopping
        # EarlyStopping requires score < best_score to increment counter
        losses = iter([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
        mock_validate.side_effect = lambda *args, **kwargs: (next(losses), 0.5)

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30, 2)
        )
        X = np.random.randn(8, 10, 3).astype('f4')
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        dataset = sslmodel.NormalDataset(X, y=y, transpose_channels_first=True)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        weights_path = tmp_path / "weights.pt"

        sslmodel.train(
            model, train_loader, val_loader, 'cpu',
            weights_path=str(weights_path),
            num_epoch=100,  # Would take long without early stopping
            patience=2
        )

        # Should stop after patience+1 epochs:
        # Epoch 1: loss=0.5, best=0.5, counter=0
        # Epoch 2: loss=0.6 > best, counter=1
        # Epoch 3: loss=0.7 > best, counter=2 → early_stop=True
        assert mock_validate.call_count == 3


class TestTrainIntegration:
    """Integration tests for training without mocking _validate_model."""

    def test_single_epoch_actual_forward_pass(self, tmp_path):
        """Test actual forward pass through model during training."""
        # Simple model
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30, 2)
        )

        # Small dataset
        np.random.seed(42)
        X = np.random.randn(16, 10, 3).astype('f4')
        y = np.array([0, 1] * 8)  # Balanced classes
        dataset = sslmodel.NormalDataset(X, y=y, transpose_channels_first=True)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        weights_path = tmp_path / "weights.pt"

        # Run actual training for 1 epoch without mocking
        result = sslmodel.train(
            model, train_loader, val_loader, 'cpu',
            weights_path=str(weights_path),
            num_epoch=1,
            patience=10,
            learning_rate=0.01
        )

        # Should have completed
        assert result is not None
        # Weights should be saved
        assert weights_path.exists()
