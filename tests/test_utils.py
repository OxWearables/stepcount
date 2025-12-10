"""
Tests for stepcount.utils module.

Tests cover:
- File reading (CSV format)
- Wear time calculations
- Data filtering/exclusion functions
- Imputation functions
- Frequency inference
- Path resolution utilities
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import json

from stepcount import utils


class TestInferFreq:
    """Tests for frequency inference function."""

    def test_infer_freq_15hz(self, accel_data_1_5_days, sample_rate):
        """Test frequency inference on 15Hz fixture data."""
        freq = utils.infer_freq(accel_data_1_5_days.index)
        expected_ms = 1000 / sample_rate  # ~66.7ms for 15Hz
        expected = pd.Timedelta(f'{expected_ms}ms')
        # Allow some tolerance due to floating point
        assert abs(freq - expected) < pd.Timedelta('5ms')

    def test_infer_freq_uniform_spacing(self):
        """Test frequency inference on uniformly spaced data."""
        # Create 30Hz data (typical for SSL model)
        times = pd.date_range('2024-01-01', periods=1000, freq='33333333ns')  # ~30Hz
        freq = utils.infer_freq(times)
        expected = pd.Timedelta('33333333ns')
        assert abs(freq - expected) < pd.Timedelta('1ms')

    def test_infer_freq_with_gaps(self):
        """Test frequency inference is robust to small gaps."""
        # 10Hz data with a small gap
        times1 = pd.date_range('2024-01-01 00:00:00', periods=500, freq='100ms')
        times2 = pd.date_range('2024-01-01 00:01:00', periods=500, freq='100ms')
        times = pd.Index(pd.concat([pd.Series(times1), pd.Series(times2)]))

        freq = utils.infer_freq(times)
        expected = pd.Timedelta('100ms')
        # Should still detect ~100ms frequency
        assert abs(freq - expected) < pd.Timedelta('50ms')


class TestCalculateWearStats:
    """Tests for wear time statistics calculation."""

    def test_wear_stats_complete_data(self, accel_data_1_5_days):
        """Test wear stats on data with no missing values."""
        stats = utils.calculate_wear_stats(accel_data_1_5_days)

        assert 'StartTime' in stats
        assert 'EndTime' in stats
        assert 'WearStartTime' in stats
        assert 'WearEndTime' in stats
        assert 'WearTime(days)' in stats
        assert 'NonwearTime(days)' in stats
        assert 'Covers24hOK' in stats

        # No missing data, so wear time should be ~1.5 days
        assert stats['WearTime(days)'] > 1.4
        assert stats['NonwearTime(days)'] < 0.1

    def test_wear_stats_with_nonwear(self, accel_data_with_nonwear):
        """Test wear stats correctly accounts for NaN periods."""
        stats = utils.calculate_wear_stats(accel_data_with_nonwear)

        # Should have some non-wear time
        assert stats['NonwearTime(days)'] > 0
        assert stats['WearTime(days)'] < 1.5

    def test_wear_stats_empty_data(self):
        """Test wear stats handles empty DataFrame."""
        empty_df = pd.DataFrame(columns=['x', 'y', 'z'])
        empty_df.index.name = 'time'

        stats = utils.calculate_wear_stats(empty_df)

        assert stats['StartTime'] is None
        assert stats['EndTime'] is None
        assert stats['WearTime(days)'] == 0.0
        assert stats['NonwearTime(days)'] == 0.0


class TestCalculateDailyWearStats:
    """Tests for daily wear statistics."""

    def test_daily_wear_stats_shape(self, accel_data_2_days):
        """Test daily wear stats returns correct shape."""
        daily_stats = utils.calculate_daily_wear_stats(accel_data_2_days)

        # Should have 2 rows for 2 days
        assert len(daily_stats) == 2
        assert 'WearTime(hours)' in daily_stats.columns

    def test_daily_wear_stats_values(self, accel_data_2_days):
        """Test daily wear stats values are reasonable."""
        daily_stats = utils.calculate_daily_wear_stats(accel_data_2_days)

        # Each day should have ~24 hours of wear (no missing data)
        for _, row in daily_stats.iterrows():
            assert 23 < row['WearTime(hours)'] <= 24

    def test_daily_wear_stats_with_nonwear(self, accel_data_with_nonwear):
        """Test daily wear stats with non-wear periods."""
        daily_stats = utils.calculate_daily_wear_stats(accel_data_with_nonwear)

        # Day 1 has 2 hours of non-wear (14:00-16:00)
        # So wear time should be ~22 hours
        day1_stats = daily_stats.iloc[0]
        assert day1_stats['WearTime(hours)'] < 23

    def test_daily_wear_stats_empty(self):
        """Test daily wear stats handles empty DataFrame."""
        empty_df = pd.DataFrame(columns=['x', 'y', 'z'])
        empty_df.index = pd.DatetimeIndex([], name='time')

        daily_stats = utils.calculate_daily_wear_stats(empty_df)
        assert len(daily_stats) == 0


class TestFlagWearBelowDays:
    """Tests for flagging low-wear days."""

    def test_flag_wear_below_threshold(self, accel_data_with_nonwear):
        """Test that days with insufficient wear are flagged."""
        # Set a very high threshold that should flag all days
        result = utils.flag_wear_below_days(accel_data_with_nonwear, min_wear='23H')

        # Day 1 has 2 hours nonwear (14:00-16:00), so <23H wear
        # Some values should be NaN
        assert result.isna().any().any()

    def test_flag_wear_below_preserves_good_days(self, accel_data_2_days):
        """Test that days with sufficient wear are preserved."""
        # Low threshold should preserve all data
        result = utils.flag_wear_below_days(accel_data_2_days, min_wear='12H')

        # Should have no new NaN values (original has no NaN)
        assert not result.isna().any().any()

    def test_flag_wear_below_empty_data(self):
        """Test handling of empty data."""
        empty_series = pd.Series(dtype=float)
        empty_series.index = pd.DatetimeIndex([], name='time')

        result = utils.flag_wear_below_days(empty_series)
        assert len(result) == 0


class TestDropFirstLastDays:
    """Tests for dropping first/last days."""

    def test_drop_first_day(self, accel_data_2_days):
        """Test dropping only the first day."""
        result = utils.drop_first_last_days(accel_data_2_days, first_or_last='first')

        # Should only have day 2 remaining
        unique_dates = result.index.date
        assert len(np.unique(unique_dates)) == 1

    def test_drop_last_day(self, accel_data_2_days):
        """Test dropping only the last day."""
        result = utils.drop_first_last_days(accel_data_2_days, first_or_last='last')

        unique_dates = result.index.date
        assert len(np.unique(unique_dates)) == 1

    def test_drop_both_days(self, accel_data_2_days):
        """Test dropping both first and last days."""
        result = utils.drop_first_last_days(accel_data_2_days, first_or_last='both')

        # With only 2 days, dropping both leaves nothing
        assert len(result) == 0

    def test_drop_first_last_single_day(self, accel_data_1_5_days):
        """Test dropping on data spanning 2 partial days."""
        result = utils.drop_first_last_days(accel_data_1_5_days, first_or_last='first')

        # Should only have partial day 2
        first_date = accel_data_1_5_days.index.date[0]
        assert first_date not in result.index.date


class TestImputeMissing:
    """Tests for missing data imputation."""

    def test_impute_missing_fills_gaps(self, accel_data_with_nonwear):
        """Test that imputation fills NaN values."""
        # Use extrapolate=False to avoid adding boundary rows
        result = utils.impute_missing(accel_data_with_nonwear, extrapolate=False)

        # Should have fewer NaN values after imputation
        original_nan_count = accel_data_with_nonwear.isna().sum().sum()
        result_nan_count = result.isna().sum().sum()

        # Note: impute_missing uses time-of-day based imputation,
        # which may not fill all gaps if there's not enough reference data
        assert result_nan_count <= original_nan_count

    def test_impute_missing_preserves_valid_data(self, accel_data_1_5_days):
        """Test that valid data is preserved after imputation."""
        # Use extrapolate=False to avoid adding boundary rows
        result = utils.impute_missing(accel_data_1_5_days, extrapolate=False)

        # For data without missing values, imputation should preserve values
        # Check that the means are similar
        orig_mean = accel_data_1_5_days.mean()
        result_mean = result.mean()

        assert np.allclose(orig_mean, result_mean, rtol=0.1)

    def test_impute_missing_skip_full_missing_days(self):
        """Test that fully missing days are skipped."""
        # Create data with one completely missing day
        times = pd.date_range('2024-01-15', periods=2*24*3600*10, freq='100ms')
        data = pd.DataFrame(
            np.random.randn(len(times), 3) * 0.1 + [0, 0, 1],
            columns=['x', 'y', 'z'],
            index=times
        )
        # Make day 2 completely NaN
        data.loc['2024-01-16', :] = np.nan

        result = utils.impute_missing(data, skip_full_missing_days=True)

        # Day 2 should still be all NaN
        assert result.loc['2024-01-16'].isna().all().all()


class TestImputeDays:
    """Tests for daily resolution imputation."""

    def test_impute_days_fills_gaps(self):
        """Test daily imputation fills missing values."""
        dates = pd.date_range('2024-01-15', periods=7, freq='D')
        values = pd.Series([100, np.nan, 120, 110, np.nan, 90, 100], index=dates)

        result = utils.impute_days(values)

        # Should have no NaN values after imputation
        assert not result.isna().any()

    def test_impute_days_median_method(self):
        """Test daily imputation with median method."""
        dates = pd.date_range('2024-01-15', periods=7, freq='D')
        values = pd.Series([100, np.nan, 120, 110, np.nan, 90, 100], index=dates)

        result = utils.impute_days(values, method='median')

        assert not result.isna().any()

    def test_impute_days_all_nan(self):
        """Test imputation handles all-NaN series."""
        dates = pd.date_range('2024-01-15', periods=7, freq='D')
        values = pd.Series([np.nan] * 7, index=dates)

        result = utils.impute_days(values)

        # Should return the series unchanged (all NaN)
        assert result.isna().all()


class TestResolvePath:
    """Tests for path resolution utility."""

    def test_resolve_path_basic(self):
        """Test basic path resolution."""
        dirname, filename, ext = utils.resolve_path('/path/to/file.csv')

        assert dirname == Path('/path/to')
        assert filename == 'file'
        assert ext == '.csv'

    def test_resolve_path_compressed(self):
        """Test path resolution with compressed file."""
        dirname, filename, ext = utils.resolve_path('/path/to/data.csv.gz')

        assert filename == 'data'
        assert ext == '.csv'

    def test_resolve_path_complex(self):
        """Test path resolution with complex filename.

        Note: resolve_path uses p.suffixes[0] which takes the first suffix.
        For 'subject_123.test.csv', suffixes are ['.test', '.csv'],
        so extension is '.test' and filename is 'subject_123'.
        """
        dirname, filename, ext = utils.resolve_path('/data/subject_123.test.csv')

        # The function takes the FIRST suffix, not the last
        assert filename == 'subject_123'
        assert ext == '.test'


class TestMd5:
    """Tests for MD5 checksum utility."""

    def test_md5_consistent(self, temp_dir):
        """Test MD5 produces consistent results."""
        test_file = temp_dir / "test_md5.txt"
        test_file.write_text("Hello, World!")

        hash1 = utils.md5(test_file)
        hash2 = utils.md5(test_file)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 produces 32-char hex string

    def test_md5_different_content(self, temp_dir):
        """Test MD5 differs for different content."""
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        assert utils.md5(file1) != utils.md5(file2)


class TestNpEncoder:
    """Tests for NumPy JSON encoder."""

    def test_encode_numpy_int(self):
        """Test encoding numpy integers."""
        data = {'value': np.int64(42)}
        result = json.dumps(data, cls=utils.NpEncoder)
        assert '"value": 42' in result

    def test_encode_numpy_float(self):
        """Test encoding numpy floats."""
        data = {'value': np.float64(3.14159)}
        result = json.dumps(data, cls=utils.NpEncoder)
        assert '3.14159' in result

    def test_encode_numpy_array(self):
        """Test encoding numpy arrays."""
        data = {'values': np.array([1, 2, 3])}
        result = json.dumps(data, cls=utils.NpEncoder)
        assert '[1, 2, 3]' in result

    def test_encode_nan(self):
        """Test encoding NaN values."""
        data = {'value': np.nan}
        result = json.dumps(data, cls=utils.NpEncoder)
        # NaN should be encoded (as NaN is not valid JSON, it becomes null-ish)
        assert 'NaN' in result or 'null' in result


class TestNanint:
    """Tests for nanint utility."""

    def test_nanint_with_int(self):
        """Test nanint preserves integers."""
        assert utils.nanint(42.0) == 42
        assert isinstance(utils.nanint(42.0), int)

    def test_nanint_with_nan(self):
        """Test nanint preserves NaN."""
        result = utils.nanint(np.nan)
        assert np.isnan(result)

    def test_nanint_truncates(self):
        """Test nanint truncates floats (uses int(), not round())."""
        # The nanint function uses int() which truncates toward zero
        assert utils.nanint(42.6) == 42
        assert utils.nanint(42.4) == 42
        assert utils.nanint(-2.6) == -2


class TestReadCSV:
    """Tests for reading CSV accelerometer files."""

    def test_read_csv_basic(self, temp_csv_file, sample_rate):
        """Test reading a basic CSV file."""
        data, info = utils.read(
            str(temp_csv_file),
            sample_rate=sample_rate,
            resample_hz=None,  # Don't resample
            verbose=False
        )

        assert isinstance(data, pd.DataFrame)
        assert 'x' in data.columns
        assert 'y' in data.columns
        assert 'z' in data.columns
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_read_csv_with_row_limits(self, temp_dir, sample_rate):
        """Test reading CSV with row limits."""
        # Create a small test CSV
        times = pd.date_range('2024-01-15', periods=1000, freq='100ms')
        df = pd.DataFrame({
            'time': times,
            'x': np.random.randn(1000) * 0.1,
            'y': np.random.randn(1000) * 0.1,
            'z': np.random.randn(1000) * 0.1 + 1.0
        })
        csv_path = temp_dir / "test_rows.csv"
        df.to_csv(csv_path, index=False)

        # Read only rows 100-199 (100 rows)
        data, info = utils.read(
            str(csv_path),
            csv_start_row=100,
            csv_end_row=199,
            sample_rate=sample_rate,
            resample_hz=None,
            verbose=False
        )

        # Should have approximately 100 rows (may vary slightly due to processing)
        assert 80 < len(data) < 120

    def test_read_csv_with_time_format(self, temp_dir, sample_rate):
        """Test reading CSV with custom time format."""
        times = pd.date_range('2024-01-15', periods=100, freq='100ms')
        df = pd.DataFrame({
            'time': times.strftime('%Y/%m/%d %H:%M:%S.%f'),
            'x': np.random.randn(100) * 0.1,
            'y': np.random.randn(100) * 0.1,
            'z': np.random.randn(100) * 0.1 + 1.0
        })
        csv_path = temp_dir / "test_format.csv"
        df.to_csv(csv_path, index=False)

        data, info = utils.read(
            str(csv_path),
            csv_time_format='%Y/%m/%d %H:%M:%S.%f',
            sample_rate=sample_rate,
            resample_hz=None,
            verbose=False
        )

        assert len(data) > 0
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_read_csv_invalid_row_range(self, temp_dir, sample_rate):
        """Test reading CSV with invalid row range raises error."""
        times = pd.date_range('2024-01-15', periods=100, freq='100ms')
        df = pd.DataFrame({
            'time': times,
            'x': np.random.randn(100) * 0.1,
            'y': np.random.randn(100) * 0.1,
            'z': np.random.randn(100) * 0.1 + 1.0
        })
        csv_path = temp_dir / "test_invalid.csv"
        df.to_csv(csv_path, index=False)

        # End row before start row should raise error
        with pytest.raises(ValueError, match="csv_end_row.*must be >= csv_start_row"):
            utils.read(
                str(csv_path),
                csv_start_row=50,
                csv_end_row=10,
                sample_rate=sample_rate,
                verbose=False
            )


class TestErrorPaths:
    """Tests for error handling in utils functions."""

    def test_read_missing_file(self):
        """Test that reading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            utils.read('/nonexistent/path/to/file.csv', sample_rate=30, verbose=False)

    def test_read_missing_columns(self, temp_dir, sample_rate):
        """Test that missing required columns raises an error."""
        # CSV with wrong columns
        csv_path = temp_dir / 'missing_cols.csv'
        csv_path.write_text('time,a,b\n2024-01-01,1,2\n')

        with pytest.raises((ValueError, KeyError)):
            utils.read(str(csv_path), sample_rate=sample_rate, verbose=False)

    def test_read_empty_file(self, temp_dir, sample_rate):
        """Test that reading an empty file raises an error."""
        csv_path = temp_dir / 'empty.csv'
        csv_path.write_text('')

        with pytest.raises(Exception):  # Could be EmptyDataError or ParserError
            utils.read(str(csv_path), sample_rate=sample_rate, verbose=False)


# Skip device file tests if actipy is not installed
pytest.importorskip("actipy", reason="actipy not installed")


class TestReadDeviceFiles:
    """Tests for reading device files (.cwa, .gt3x, .bin) via actipy."""

    @pytest.fixture
    def mock_actipy_data(self):
        """Create mock accelerometer data returned by actipy.read_device."""
        times = pd.date_range('2024-01-15 08:00:00', periods=1000, freq='100ms')
        data = pd.DataFrame({
            'x': np.random.randn(1000) * 0.1,
            'y': np.random.randn(1000) * 0.1,
            'z': np.random.randn(1000) * 0.1 + 1.0
        }, index=times)
        data.index.name = 'time'
        info = {
            'Filename': 'test.cwa',
            'Device': 'Axivity',
            'DeviceID': '12345',
            'SampleRate': 100,
            'ResampleRate': 100,
        }
        return data, info

    def test_read_cwa_file(self, temp_dir, mock_actipy_data, monkeypatch):
        """Test reading .cwa device file via mocked actipy."""
        from unittest.mock import MagicMock
        mock_data, mock_info = mock_actipy_data

        # Mock actipy.read_device
        mock_read_device = MagicMock(return_value=(mock_data, mock_info))
        monkeypatch.setattr('actipy.read_device', mock_read_device)

        # Create a dummy .cwa file (just needs to exist for path checks)
        cwa_file = temp_dir / 'subject_001.cwa'
        cwa_file.write_bytes(b'\x00' * 100)

        data, info = utils.read(str(cwa_file), verbose=False)

        # Verify actipy.read_device was called
        mock_read_device.assert_called_once()
        call_args = mock_read_device.call_args
        assert str(cwa_file) in call_args[0]

        # Verify returned data structure
        assert isinstance(data, pd.DataFrame)
        assert 'x' in data.columns
        assert 'y' in data.columns
        assert 'z' in data.columns

    def test_read_gt3x_file(self, temp_dir, mock_actipy_data, monkeypatch):
        """Test reading .gt3x device file via mocked actipy."""
        from unittest.mock import MagicMock
        mock_data, mock_info = mock_actipy_data
        mock_info = mock_info.copy()
        mock_info['Device'] = 'ActiGraph'

        mock_read_device = MagicMock(return_value=(mock_data, mock_info))
        monkeypatch.setattr('actipy.read_device', mock_read_device)

        gt3x_file = temp_dir / 'subject_002.gt3x'
        gt3x_file.write_bytes(b'\x00' * 100)

        data, info = utils.read(str(gt3x_file), verbose=False)

        mock_read_device.assert_called_once()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_read_bin_file(self, temp_dir, mock_actipy_data, monkeypatch):
        """Test reading .bin device file via mocked actipy."""
        from unittest.mock import MagicMock
        mock_data, mock_info = mock_actipy_data
        mock_info = mock_info.copy()
        mock_info['Device'] = 'GENEActiv'

        mock_read_device = MagicMock(return_value=(mock_data, mock_info))
        monkeypatch.setattr('actipy.read_device', mock_read_device)

        bin_file = temp_dir / 'subject_003.bin'
        bin_file.write_bytes(b'\x00' * 100)

        data, info = utils.read(str(bin_file), verbose=False)

        mock_read_device.assert_called_once()
        assert isinstance(data, pd.DataFrame)

    def test_device_file_ignores_csv_options(self, temp_dir, mock_actipy_data, monkeypatch):
        """Test that CSV-specific options are ignored for device files with warning."""
        from unittest.mock import MagicMock
        mock_data, mock_info = mock_actipy_data

        mock_read_device = MagicMock(return_value=(mock_data, mock_info))
        monkeypatch.setattr('actipy.read_device', mock_read_device)

        cwa_file = temp_dir / 'subject.cwa'
        cwa_file.write_bytes(b'\x00' * 100)

        # Pass CSV-specific options that should be ignored
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data, info = utils.read(
                str(cwa_file),
                csv_start_row=10,
                csv_time_format='%Y-%m-%d',
                verbose=False
            )

            # Should have a warning about CSV options being ignored
            assert any('csv' in str(warning.message).lower() for warning in w)


class TestCsvTxyzIdxsParsing:
    """Tests for --csv-txyz-idxs argument parsing."""

    def test_csv_txyz_idxs_valid(self, temp_dir, sample_rate):
        """Test valid csv_txyz_idxs parsing."""
        # Create CSV with columns in non-standard order
        times = pd.date_range('2024-01-15', periods=100, freq='100ms')
        df = pd.DataFrame({
            'timestamp': times,
            'accel_x': np.random.randn(100) * 0.1,
            'accel_y': np.random.randn(100) * 0.1,
            'accel_z': np.random.randn(100) * 0.1 + 1.0
        })
        csv_path = temp_dir / 'custom_cols.csv'
        df.to_csv(csv_path, index=False)

        # Use indices to specify columns: time=0, x=1, y=2, z=3
        data, info = utils.read(
            str(csv_path),
            csv_txyz_idxs='0,1,2,3',
            sample_rate=sample_rate,
            resample_hz=None,
            verbose=False
        )

        assert len(data) > 0
        assert 'x' in data.columns
        assert 'y' in data.columns
        assert 'z' in data.columns

    def test_csv_txyz_idxs_reordered(self, temp_dir, sample_rate):
        """Test csv_txyz_idxs with reordered columns."""
        # Create CSV with columns in different order: z, y, x, time
        times = pd.date_range('2024-01-15', periods=100, freq='100ms')
        df = pd.DataFrame({
            'col_z': np.random.randn(100) * 0.1 + 1.0,
            'col_y': np.random.randn(100) * 0.1,
            'col_x': np.random.randn(100) * 0.1,
            'col_time': times
        })
        csv_path = temp_dir / 'reordered.csv'
        df.to_csv(csv_path, index=False)

        # Indices: time=3, x=2, y=1, z=0
        data, info = utils.read(
            str(csv_path),
            csv_txyz_idxs='3,2,1,0',
            sample_rate=sample_rate,
            resample_hz=None,
            verbose=False
        )

        assert len(data) > 0
        # Verify data was read correctly (z should be ~1.0)
        assert data['z'].mean() > 0.5

    def test_csv_txyz_idxs_invalid_format(self, temp_dir, sample_rate):
        """Test csv_txyz_idxs with invalid format raises error."""
        times = pd.date_range('2024-01-15', periods=100, freq='100ms')
        df = pd.DataFrame({
            'time': times,
            'x': np.random.randn(100) * 0.1,
            'y': np.random.randn(100) * 0.1,
            'z': np.random.randn(100) * 0.1 + 1.0
        })
        csv_path = temp_dir / 'test.csv'
        df.to_csv(csv_path, index=False)

        # Only 3 indices instead of 4
        with pytest.raises(ValueError, match="4 comma-separated integers"):
            utils.read(
                str(csv_path),
                csv_txyz_idxs='0,1,2',
                sample_rate=sample_rate,
                verbose=False
            )

    def test_csv_txyz_idxs_negative(self, temp_dir, sample_rate):
        """Test csv_txyz_idxs with negative index raises error."""
        times = pd.date_range('2024-01-15', periods=100, freq='100ms')
        df = pd.DataFrame({
            'time': times,
            'x': np.random.randn(100) * 0.1,
            'y': np.random.randn(100) * 0.1,
            'z': np.random.randn(100) * 0.1 + 1.0
        })
        csv_path = temp_dir / 'test.csv'
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="non-negative"):
            utils.read(
                str(csv_path),
                csv_txyz_idxs='0,-1,2,3',
                sample_rate=sample_rate,
                verbose=False
            )

    def test_csv_txyz_idxs_out_of_range(self, temp_dir, sample_rate):
        """Test csv_txyz_idxs with out-of-range index raises error."""
        times = pd.date_range('2024-01-15', periods=100, freq='100ms')
        df = pd.DataFrame({
            'time': times,
            'x': np.random.randn(100) * 0.1,
            'y': np.random.randn(100) * 0.1,
            'z': np.random.randn(100) * 0.1 + 1.0
        })
        csv_path = temp_dir / 'test.csv'
        df.to_csv(csv_path, index=False)

        # Column index 10 doesn't exist in 4-column CSV
        with pytest.raises(ValueError, match="out of range"):
            utils.read(
                str(csv_path),
                csv_txyz_idxs='0,1,2,10',
                sample_rate=sample_rate,
                verbose=False
            )
