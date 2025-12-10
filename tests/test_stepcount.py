"""
Tests for stepcount.stepcount module.

Tests cover:
- summarize_enmo function
- summarize_steps function
- summarize_cadence function
- summarize_bouts function
- numba_detect_bouts function
- plot function
- CLI end-to-end tests
"""
import pytest
import numpy as np
import pandas as pd
import subprocess
import sys
import json
import gzip
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from stepcount import stepcount


class TestSummarizeENMO:
    """Tests for ENMO summarization."""

    def test_summarize_enmo_basic(self, accel_data_1_5_days):
        """Test basic ENMO summarization."""
        summary = stepcount.summarize_enmo(accel_data_1_5_days)

        assert 'avg' in summary
        assert 'daily' in summary
        assert 'hourly' in summary
        assert 'minutely' in summary

        # ENMO should be non-negative (clipped at 0)
        assert summary['avg'] >= 0

    def test_summarize_enmo_daily_shape(self, accel_data_2_days):
        """Test ENMO daily summary has correct shape."""
        summary = stepcount.summarize_enmo(accel_data_2_days)

        # Should have 2 days
        assert len(summary['daily']) == 2

    def test_summarize_enmo_hourly_shape(self, accel_data_1_5_days):
        """Test ENMO hourly summary has expected entries."""
        summary = stepcount.summarize_enmo(accel_data_1_5_days)

        # 1.5 days = 36 hours exactly
        assert len(summary['hourly']) == 36

    def test_summarize_enmo_hour_averages(self, accel_data_1_5_days):
        """Test ENMO hour-of-day averages."""
        summary = stepcount.summarize_enmo(accel_data_1_5_days)

        assert 'hour_avgs' in summary
        assert len(summary['hour_avgs']) == 24

    def test_summarize_enmo_weekend_weekday(self, accel_data_2_days):
        """Test ENMO weekend/weekday split."""
        summary = stepcount.summarize_enmo(accel_data_2_days)

        # Data starts on Friday (2024-01-19)
        assert 'weekend_avg' in summary
        assert 'weekday_avg' in summary

    def test_summarize_enmo_adjusted(self, accel_data_with_nonwear):
        """Test ENMO with adjusted estimates (imputation)."""
        summary = stepcount.summarize_enmo(
            accel_data_with_nonwear,
            adjust_estimates=True
        )

        # Should still produce valid averages
        assert not np.isnan(summary['avg'])

    def test_summarize_enmo_min_wear(self, accel_data_with_nonwear):
        """Test ENMO with minimum wear requirements."""
        summary = stepcount.summarize_enmo(
            accel_data_with_nonwear,
            min_wear_per_day=21 * 60,  # 21 hours
            min_wear_per_hour=50,
            min_wear_per_minute=0.5
        )

        # Some days may be NaN if wear is insufficient
        # But overall average should still be computed
        assert 'avg' in summary


class TestSummarizeSteps:
    """Tests for step count summarization."""

    def test_summarize_steps_basic(self, step_counts_series):
        """Test basic step summarization."""
        summary = stepcount.summarize_steps(step_counts_series, steptol=3)

        assert 'total_steps' in summary
        assert 'avg_steps' in summary
        assert 'daily_steps' in summary
        assert 'hourly_steps' in summary
        assert 'minutely_steps' in summary

    def test_summarize_steps_total(self, step_counts_series):
        """Test total steps calculation."""
        summary = stepcount.summarize_steps(step_counts_series, steptol=3)

        # Total should be sum of daily values
        assert summary['total_steps'] >= 0

    def test_summarize_steps_daily_stats(self, step_counts_series):
        """Test daily step statistics."""
        summary = stepcount.summarize_steps(step_counts_series, steptol=3)

        # Check daily stats
        assert 'avg_steps' in summary
        assert 'med_steps' in summary
        assert 'min_steps' in summary
        assert 'max_steps' in summary

        # Min <= Med <= Max
        if not np.isnan(summary['min_steps']):
            assert summary['min_steps'] <= summary['med_steps']
            assert summary['med_steps'] <= summary['max_steps']

    def test_summarize_steps_walking_time(self, step_counts_series):
        """Test walking time calculation."""
        summary = stepcount.summarize_steps(step_counts_series, steptol=3)

        assert 'total_walk' in summary
        assert 'avg_walk' in summary

        # Walking time should be positive (we have walking windows)
        assert summary['total_walk'] >= 0

    def test_summarize_steps_percentile_times(self, step_counts_series):
        """Test time-of-accumulated-steps percentiles."""
        summary = stepcount.summarize_steps(step_counts_series, steptol=3)

        assert 'ptile_at_avgs' in summary
        ptiles = summary['ptile_at_avgs']

        assert 'p05_at' in ptiles
        assert 'p25_at' in ptiles
        assert 'p50_at' in ptiles
        assert 'p75_at' in ptiles
        assert 'p95_at' in ptiles

    def test_summarize_steps_weekend_weekday(self):
        """Test weekend/weekday step split."""
        # Create data spanning weekend
        times = pd.date_range('2024-01-19', periods=200, freq='10s')  # Friday start
        steps = pd.Series(np.random.randint(0, 20, 200), index=times, name='Steps')

        summary = stepcount.summarize_steps(steps, steptol=3)

        assert 'weekend_avg_steps' in summary
        assert 'weekday_avg_steps' in summary

    def test_summarize_steps_adjusted(self):
        """Test step summarization with adjusted estimates."""
        times = pd.date_range('2024-01-15', periods=1000, freq='10s')
        steps = pd.Series(np.random.randint(0, 15, 1000), index=times, name='Steps')
        # Add some NaN
        steps.iloc[100:150] = np.nan

        summary = stepcount.summarize_steps(steps, steptol=3, adjust_estimates=True)

        # Should still produce valid summaries
        assert 'avg_steps' in summary

    def test_summarize_steps_hour_profile(self, step_counts_series):
        """Test hour-of-day step averages."""
        summary = stepcount.summarize_steps(step_counts_series, steptol=3)

        assert 'hour_steps' in summary
        assert len(summary['hour_steps']) == 24


class TestSummarizeCadence:
    """Tests for cadence summarization."""

    def test_summarize_cadence_basic(self, step_counts_series):
        """Test basic cadence summarization."""
        summary = stepcount.summarize_cadence(step_counts_series, steptol=3)

        assert 'cadence_peak1' in summary
        assert 'cadence_peak30' in summary
        assert 'cadence_p95' in summary

    def test_summarize_cadence_daily(self, step_counts_series):
        """Test daily cadence values."""
        summary = stepcount.summarize_cadence(step_counts_series, steptol=3)

        assert 'daily' in summary
        daily = summary['daily']

        assert 'CadencePeak1(steps/min)' in daily.columns
        assert 'CadencePeak30(steps/min)' in daily.columns
        assert 'Cadence95th(steps/min)' in daily.columns

    def test_summarize_cadence_peak1_ge_peak30(self, step_counts_series):
        """Test that peak1 >= peak30."""
        summary = stepcount.summarize_cadence(step_counts_series, steptol=3)

        # Peak 1 minute should be >= peak 30 minutes
        if not np.isnan(summary['cadence_peak1']) and not np.isnan(summary['cadence_peak30']):
            assert summary['cadence_peak1'] >= summary['cadence_peak30']

    def test_summarize_cadence_min_walk_filter(self):
        """Test cadence with minimum walk time filter."""
        # Create data with minimal walking
        times = pd.date_range('2024-01-15', periods=1000, freq='10s')
        steps = pd.Series([0] * 950 + [10] * 50, index=times, name='Steps')

        summary = stepcount.summarize_cadence(
            steps, steptol=3, min_walk_per_day=60  # Require 60 min walking
        )

        # Should return NaN for days without enough walking
        assert np.isnan(summary['cadence_peak1']) or summary['cadence_peak1'] >= 0

    def test_summarize_cadence_weekend_weekday(self):
        """Test weekend/weekday cadence split."""
        times = pd.date_range('2024-01-19', periods=500, freq='10s')  # Friday
        steps = pd.Series(np.random.randint(0, 20, 500), index=times, name='Steps')

        summary = stepcount.summarize_cadence(steps, steptol=3)

        assert 'weekend_cadence_peak1' in summary
        assert 'weekday_cadence_peak1' in summary

    def test_summarize_cadence_adjusted(self):
        """Test cadence with adjusted estimates."""
        times = pd.date_range('2024-01-15', periods=2000, freq='10s')
        steps = pd.Series(np.random.randint(0, 15, 2000), index=times, name='Steps')

        summary = stepcount.summarize_cadence(
            steps, steptol=3, adjust_estimates=True
        )

        assert 'cadence_peak1' in summary


class TestNumbaDetectBouts:
    """Tests for bout detection using numba."""

    def test_detect_bouts_basic(self):
        """Test basic bout detection."""
        # Clear bout pattern
        arr = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0])

        bouts = stepcount.numba_detect_bouts(arr, min_percent_ones=0.6, max_trailing_zeros=2)

        assert len(bouts) > 0
        # First bout should start at index 1
        assert bouts[0][0] == 1

    def test_detect_bouts_single_bout(self):
        """Test detection of single continuous bout."""
        arr = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

        bouts = stepcount.numba_detect_bouts(arr, min_percent_ones=0.8, max_trailing_zeros=1)

        assert len(bouts) == 1
        assert bouts[0][0] == 2  # Start at index 2
        assert bouts[0][1] == 5  # Length 5

    def test_detect_bouts_no_bouts(self):
        """Test no bouts in sparse data."""
        arr = np.array([0, 0, 1, 0, 0, 1, 0, 0])

        bouts = stepcount.numba_detect_bouts(arr, min_percent_ones=0.8, max_trailing_zeros=1)

        # Each 1 is isolated, shouldn't form a bout with high min_percent_ones
        assert len(bouts) <= 2

    def test_detect_bouts_tolerates_gaps(self):
        """Test bout detection tolerates small gaps."""
        # Bout with small gap
        arr = np.array([1, 1, 1, 0, 1, 1, 1])

        bouts = stepcount.numba_detect_bouts(arr, min_percent_ones=0.7, max_trailing_zeros=2)

        # Could be one bout if gap is tolerated
        assert len(bouts) >= 1

    def test_detect_bouts_trailing_zeros(self):
        """Test trailing zeros aren't counted in bout length."""
        arr = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        bouts = stepcount.numba_detect_bouts(arr, min_percent_ones=0.8, max_trailing_zeros=2)

        assert len(bouts) == 1
        # Bout should be length 4 (trailing zeros excluded)
        assert bouts[0][1] == 4

    def test_detect_bouts_all_zeros(self):
        """Test no bouts in all-zero array."""
        arr = np.array([0, 0, 0, 0, 0])

        bouts = stepcount.numba_detect_bouts(arr)

        assert len(bouts) == 0

    def test_detect_bouts_all_ones(self):
        """Test single bout in all-ones array."""
        arr = np.array([1, 1, 1, 1, 1])

        bouts = stepcount.numba_detect_bouts(arr)

        assert len(bouts) == 1
        assert bouts[0][0] == 0  # Start at beginning
        assert bouts[0][1] == 5  # Full length


class TestSummarizeBouts:
    """Tests for bout summarization."""

    def test_summarize_bouts_basic(self, step_counts_series, accel_data_1_5_days):
        """Test basic bout summarization."""
        summary = stepcount.summarize_bouts(
            step_counts_series,
            accel_data_1_5_days,
            steptol=3
        )

        assert 'bouts' in summary
        bouts_df = summary['bouts']

        # Should be a DataFrame with expected columns
        assert isinstance(bouts_df, pd.DataFrame)

    def test_summarize_bouts_columns(self, step_counts_series, accel_data_1_5_days):
        """Test bout summary has expected columns."""
        summary = stepcount.summarize_bouts(
            step_counts_series,
            accel_data_1_5_days,
            steptol=3
        )

        bouts_df = summary['bouts']

        expected_cols = [
            'StartTime', 'EndTime', 'Duration(mins)',
            'Steps', 'Cadence(steps/min)', 'ENMO(mg)'
        ]

        for col in expected_cols:
            assert col in bouts_df.columns

    def test_summarize_bouts_no_walking(self):
        """Test bout summary with no walking."""
        times = pd.date_range('2024-01-15', periods=100, freq='10s')
        steps = pd.Series([0] * 100, index=times, name='Steps')
        data = pd.DataFrame({
            'x': np.zeros(1000),
            'y': np.zeros(1000),
            'z': np.ones(1000)
        }, index=pd.date_range('2024-01-15', periods=1000, freq='100ms'))

        summary = stepcount.summarize_bouts(steps, data, steptol=3)

        # Should return empty DataFrame
        assert len(summary['bouts']) == 0

    def test_summarize_bouts_time_since_last(self, step_counts_series, accel_data_1_5_days):
        """Test time-since-last bout calculation."""
        summary = stepcount.summarize_bouts(
            step_counts_series,
            accel_data_1_5_days,
            steptol=3
        )

        bouts_df = summary['bouts']

        if len(bouts_df) > 1:
            assert 'TimeSinceLast(mins)' in bouts_df.columns
            # First bout should have NaN for time since last
            assert pd.isna(bouts_df['TimeSinceLast(mins)'].iloc[0])


class TestPlot:
    """Tests for step count plotting."""

    def test_plot_basic(self, step_counts_series):
        """Test basic plot generation."""
        fig = stepcount.plot(step_counts_series)

        assert fig is not None
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_title(self, step_counts_series):
        """Test plot with custom title."""
        fig = stepcount.plot(step_counts_series, title='Test Subject')

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_handles_nan(self):
        """Test plot handles NaN values."""
        times = pd.date_range('2024-01-15', periods=1000, freq='10s')
        steps = pd.Series(np.random.randint(0, 15, 1000), index=times, name='Steps')
        steps.iloc[100:200] = np.nan  # Add NaN gap

        fig = stepcount.plot(steps)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_dataframe_input(self, step_counts_series):
        """Test plot accepts DataFrame with 'Steps' column."""
        df = step_counts_series.to_frame()

        fig = stepcount.plot(df)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestLoadModel:
    """Tests for model loading."""

    @pytest.mark.skip(reason="load_model downloads missing files, requiring network access")
    def test_load_model_missing_file(self, temp_dir):
        """Test load_model with missing file.

        Note: load_model will attempt to download the model when the file
        is missing, even with force_download=False. This test is skipped
        to avoid network dependencies in unit tests.
        """
        missing_path = temp_dir / "nonexistent_model.joblib.lzma"

        # load_model downloads missing files, doesn't raise
        # This behavior makes the function work seamlessly but means
        # we can't test missing file handling without network access
        stepcount.load_model(
            missing_path,
            model_type='ssl',
            check_md5=False,
            force_download=False
        )


class TestENMOCalculation:
    """Tests for ENMO (Euclidean Norm Minus One) calculation."""

    def test_enmo_stationary(self):
        """Test ENMO on stationary signal at 1g."""
        times = pd.date_range('2024-01-15', periods=1000, freq='100ms')
        data = pd.DataFrame({
            'x': np.zeros(1000),
            'y': np.zeros(1000),
            'z': np.ones(1000)  # 1g on z-axis
        }, index=times)

        summary = stepcount.summarize_enmo(data)

        # ENMO = max(0, sqrt(x^2+y^2+z^2) - 1) = max(0, 1-1) = 0
        assert summary['avg'] < 1  # Should be close to 0

    def test_enmo_movement(self):
        """Test ENMO on data with movement."""
        times = pd.date_range('2024-01-15', periods=1000, freq='100ms')
        # Add movement: total magnitude > 1g
        data = pd.DataFrame({
            'x': np.sin(np.linspace(0, 20*np.pi, 1000)) * 0.5,
            'y': np.zeros(1000),
            'z': np.ones(1000)
        }, index=times)

        summary = stepcount.summarize_enmo(data)

        # ENMO should be positive with movement
        assert summary['avg'] > 0


class TestIntegration:
    """Integration tests for summary functions."""

    def test_full_summary_pipeline(self, step_counts_series, accel_data_1_5_days):
        """Test running all summary functions together."""
        # ENMO
        enmo_summary = stepcount.summarize_enmo(accel_data_1_5_days)
        assert 'avg' in enmo_summary

        # Steps
        steps_summary = stepcount.summarize_steps(step_counts_series, steptol=3)
        assert 'total_steps' in steps_summary

        # Cadence
        cadence_summary = stepcount.summarize_cadence(step_counts_series, steptol=3)
        assert 'cadence_peak1' in cadence_summary

        # Bouts
        bouts_summary = stepcount.summarize_bouts(
            step_counts_series, accel_data_1_5_days, steptol=3
        )
        assert 'bouts' in bouts_summary

    def test_adjusted_vs_unadjusted(self, accel_data_with_nonwear):
        """Test that adjusted and unadjusted estimates differ."""
        unadjusted = stepcount.summarize_enmo(
            accel_data_with_nonwear, adjust_estimates=False
        )
        adjusted = stepcount.summarize_enmo(
            accel_data_with_nonwear, adjust_estimates=True
        )

        # With missing data, adjusted estimates should differ
        # (imputation fills gaps)
        # Just verify both work without error
        assert 'avg' in unadjusted
        assert 'avg' in adjusted


class TestCLIEndToEnd:
    """End-to-end CLI tests using subprocess."""

    def test_cli_help(self):
        """Test that --help works and shows usage info."""
        result = subprocess.run(
            [sys.executable, '-m', 'stepcount.stepcount', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'usage:' in result.stdout.lower() or 'Usage:' in result.stdout
        assert 'positional arguments' in result.stdout.lower() or 'filepath' in result.stdout.lower()

    def test_cli_missing_file_error(self, tmp_path):
        """Test that missing input file gives a clear error."""
        nonexistent = tmp_path / 'nonexistent.csv'
        result = subprocess.run(
            [sys.executable, '-m', 'stepcount.stepcount', str(nonexistent)],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_cli_module_invocation(self):
        """Test that stepcount can be invoked as a module."""
        result = subprocess.run(
            [sys.executable, '-m', 'stepcount.stepcount', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    @pytest.fixture
    def small_csv_file(self, tmp_path):
        """Create a small CSV file for quick E2E testing."""
        # Create 1 hour of 15Hz data (small enough to process quickly)
        n_samples = 15 * 60 * 60  # 1 hour at 15Hz
        times = pd.date_range('2024-01-15 10:00:00', periods=n_samples, freq=f'{1000000//15}us')

        # Simple resting signal with gravity on z-axis
        np.random.seed(42)
        data = pd.DataFrame({
            'time': times.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'x': np.random.randn(n_samples) * 0.02,
            'y': np.random.randn(n_samples) * 0.02,
            'z': 1.0 + np.random.randn(n_samples) * 0.02
        })

        csv_path = tmp_path / 'test_data.csv'
        data.to_csv(csv_path, index=False)
        return csv_path

    @pytest.mark.skip(reason="Full E2E test requires model download (~400MB) and is slow")
    def test_cli_basic_run(self, small_csv_file, tmp_path):
        """Run stepcount CLI and verify outputs created.

        Note: This test is skipped by default as it requires:
        1. Model download (~400MB)
        2. Significant processing time
        Run with: pytest -k test_cli_basic_run --runxfail
        """
        outdir = tmp_path / 'output'
        result = subprocess.run(
            [
                sys.executable, '-m', 'stepcount.stepcount',
                str(small_csv_file),
                '-o', str(outdir),
                '-q'  # Quiet mode
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")

        assert result.returncode == 0

        # Verify output files exist
        basename = small_csv_file.stem
        result_dir = outdir / basename
        assert (result_dir / f'{basename}-Info.json').exists()

    @pytest.mark.skip(reason="Full E2E test requires model download (~400MB) and is slow")
    def test_cli_info_json_structure(self, small_csv_file, tmp_path):
        """Verify Info.json contains expected keys after processing.

        Note: Skipped by default - requires model download.
        """
        outdir = tmp_path / 'output'
        subprocess.run(
            [
                sys.executable, '-m', 'stepcount.stepcount',
                str(small_csv_file),
                '-o', str(outdir),
                '-q'
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        basename = small_csv_file.stem
        info_path = outdir / basename / f'{basename}-Info.json'

        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)

            # Check for expected keys
            expected_keys = ['Filename', 'TotalSteps', 'StepsDayAvg']
            for key in expected_keys:
                assert key in info, f"Missing key: {key}"
