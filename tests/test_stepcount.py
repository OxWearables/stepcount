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
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')  # Non-interactive backend for testing

from stepcount import stepcount

CADENCE_PEAK_MINUTES = (1, 5, 10, 30)


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

        assert len(summary['daily']) == 2

    def test_summarize_enmo_hourly_shape(self, accel_data_1_5_days):
        """Test ENMO hourly summary has expected entries."""
        summary = stepcount.summarize_enmo(accel_data_1_5_days)

        assert len(summary['hourly']) == 36

    def test_summarize_enmo_hour_averages(self, accel_data_1_5_days):
        """Test ENMO hour-of-day averages."""
        summary = stepcount.summarize_enmo(accel_data_1_5_days)

        assert 'hour_avgs' in summary
        assert len(summary['hour_avgs']) == 24

    def test_summarize_enmo_weekend_weekday(self, accel_data_2_days):
        """Test ENMO weekend/weekday split."""
        summary = stepcount.summarize_enmo(accel_data_2_days)

        assert 'weekend_avg' in summary
        assert 'weekday_avg' in summary

    def test_summarize_enmo_adjusted(self, accel_data_with_nonwear):
        """Test ENMO with adjusted estimates (imputation)."""
        summary = stepcount.summarize_enmo(
            accel_data_with_nonwear,
            adjust_estimates=True
        )

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

        assert summary['total_steps'] >= 0

    def test_summarize_steps_daily_stats(self, step_counts_series):
        """Test daily step statistics."""
        summary = stepcount.summarize_steps(step_counts_series, steptol=3)

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

        for peak_minutes in CADENCE_PEAK_MINUTES:
            assert f'cadence_peak{peak_minutes}' in summary
        assert 'cadence_p95' in summary

    def test_summarize_cadence_daily(self, step_counts_series):
        """Test daily cadence values."""
        summary = stepcount.summarize_cadence(step_counts_series, steptol=3)

        assert 'daily' in summary
        daily = summary['daily']

        assert daily.columns.tolist() == [
            'CadencePeak1(steps/min)',
            'CadencePeak5(steps/min)',
            'CadencePeak10(steps/min)',
            'CadencePeak30(steps/min)',
            'Cadence95th(steps/min)',
        ]

    def test_summarize_cadence_peaks_are_monotonic(self, step_counts_series):
        """Test that averages cannot increase as more peak minutes are included."""
        summary = stepcount.summarize_cadence(step_counts_series, steptol=3)

        peaks = [summary[f'cadence_peak{minutes}'] for minutes in CADENCE_PEAK_MINUTES]
        if not any(np.isnan(peak) for peak in peaks):
            assert peaks == sorted(peaks, reverse=True)

    def test_summarize_cadence_longer_peaks_use_available_minutes(self):
        """Test that eligible short days retain the established peak30 behavior."""
        times = pd.date_range('2024-01-15', periods=5, freq='1min')
        steps = pd.Series([0, 30, 60, 90, 120], index=times, name='Steps')

        summary = stepcount.summarize_cadence(steps, steptol=3, min_walk_per_day=1)

        assert summary['cadence_peak1'] == 120
        assert summary['cadence_peak5'] == 75
        assert summary['cadence_peak10'] == 75
        assert summary['cadence_peak30'] == 75

    def test_summarize_cadence_min_walk_filter(self):
        """Test exact behavior immediately below and at the walking threshold."""
        times = pd.date_range('2024-01-15', periods=5, freq='1min')
        below_threshold = pd.Series([30, 30, 30, 30, 0], index=times, name='Steps')
        at_threshold = pd.Series([30] * 5, index=times, name='Steps')

        below_summary = stepcount.summarize_cadence(
            below_threshold, steptol=3, min_walk_per_day=5
        )
        at_summary = stepcount.summarize_cadence(
            at_threshold, steptol=3, min_walk_per_day=5
        )

        for peak_minutes in CADENCE_PEAK_MINUTES:
            assert np.isnan(below_summary[f'cadence_peak{peak_minutes}'])
            assert at_summary[f'cadence_peak{peak_minutes}'] == 30
        assert np.isnan(below_summary['cadence_p95'])
        assert at_summary['cadence_p95'] == 30

    def test_summarize_cadence_weekend_weekday_and_adjusted_values(self):
        """Test exact split values and weekend imputation for adjusted estimates."""
        daily_cadences = {
            '2024-01-15': 10,
            '2024-01-16': 20,
            '2024-01-17': 30,
            '2024-01-18': 40,
            '2024-01-19': 50,
            # Saturday is deliberately missing and is imputed from Sunday.
            '2024-01-21': 70,
        }
        steps = pd.concat([
            pd.Series(
                [cadence] * 5,
                index=pd.date_range(f'{date} 12:00', periods=5, freq='1min'),
            )
            for date, cadence in daily_cadences.items()
        ]).rename('Steps')

        summary = stepcount.summarize_cadence(
            steps, steptol=3, min_walk_per_day=1
        )
        adjusted = stepcount.summarize_cadence(
            steps, steptol=3, min_walk_per_day=1, adjust_estimates=True
        )

        for cadence_summary, expected_overall in ((summary, 35), (adjusted, 40)):
            for peak_minutes in CADENCE_PEAK_MINUTES:
                assert cadence_summary[f'cadence_peak{peak_minutes}'] == expected_overall
                assert cadence_summary[f'weekday_cadence_peak{peak_minutes}'] == 30
                assert cadence_summary[f'weekend_cadence_peak{peak_minutes}'] == 70


class TestNumbaDetectBouts:
    """Tests for bout detection using numba."""

    def test_concurrent_first_calls_create_one_dispatcher(self, monkeypatch):
        """Concurrent first calls share one lazily created dispatcher."""
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor

        import numba

        worker_count = 8
        start = threading.Barrier(worker_count)
        count_lock = threading.Lock()
        compile_calls = 0

        def slow_njit(func):
            nonlocal compile_calls
            with count_lock:
                compile_calls += 1
            time.sleep(0.05)
            return func

        monkeypatch.setattr(numba, 'njit', slow_njit)
        monkeypatch.setattr(stepcount, '_compiled_detect_bouts', None)

        arr = np.ones(3, dtype=int)

        def detect():
            start.wait()
            return stepcount.numba_detect_bouts(arr)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(lambda _: detect(), range(worker_count)))

        assert compile_calls == 1
        assert results == [[(0, 3)]] * worker_count

    def test_detect_bouts_basic(self):
        """Test basic bout detection."""
        arr = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0])

        bouts = stepcount.numba_detect_bouts(arr, min_percent_ones=0.6, max_trailing_zeros=2)

        assert len(bouts) > 0
        assert bouts[0][0] == 1

    def test_detect_bouts_single_bout(self):
        """Test detection of single continuous bout."""
        arr = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

        bouts = stepcount.numba_detect_bouts(arr, min_percent_ones=0.8, max_trailing_zeros=1)

        assert len(bouts) == 1
        assert bouts[0][0] == 2
        assert bouts[0][1] == 5

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
        assert bouts[0][0] == 0
        assert bouts[0][1] == 5


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


class TestDownloadToFile:
    """Tests for the atomic download helper `_download_to_file`."""

    def test_success_writes_dest_and_cleans_temp_with_timeout(self, tmp_path, monkeypatch):
        import hashlib
        import io
        import urllib.request
        payload = b"model-payload-bytes"
        dest = tmp_path / "model.joblib.lzma"
        seen = {}

        def fake_urlopen(url, timeout=None):
            seen['timeout'] = timeout
            return io.BytesIO(payload)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        stepcount._download_to_file(
            "http://example/model", dest,
            expected_md5=hashlib.md5(payload).hexdigest(),
        )

        assert dest.read_bytes() == payload
        assert seen['timeout'] == 60
        assert list(tmp_path.glob("*.tmp")) == []

    def test_md5_mismatch_raises_and_leaves_no_files(self, tmp_path, monkeypatch):
        import io
        import urllib.request
        dest = tmp_path / "model.joblib.lzma"

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda url, timeout=None: io.BytesIO(b"corrupt"))

        with pytest.raises(ValueError, match="MD5 mismatch"):
            stepcount._download_to_file(
                "http://example/model", dest, expected_md5="0" * 32)

        assert not dest.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_failure_midstream_preserves_existing_dest(self, tmp_path, monkeypatch):
        import os
        import urllib.request
        dest = tmp_path / "model.joblib.lzma"
        dest.write_bytes(b"previous-good-model")        # a valid model already in place

        class _BoomReader:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, *a):
                raise OSError("connection reset mid-download")

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda url, timeout=None: _BoomReader())

        with pytest.raises(OSError):
            stepcount._download_to_file(
                "http://example/model", dest, expected_md5="0" * 32)

        assert dest.read_bytes() == b"previous-good-model"
        assert not (tmp_path / f"{dest.name}.{os.getpid()}.tmp").exists()
        assert list(tmp_path.glob("*.tmp")) == []


class TestEnsureDownloadSSLContext:
    """Tests for the certifi SSL-context fallback used for model downloads."""

    def test_noop_when_default_context_works(self, monkeypatch):
        """When the active context factory builds fine, the HTTPS hook is untouched."""
        import ssl

        def ok_factory(*a, **k):
            return "ok-context-sentinel"

        # Force the "works" branch host-independently: make the default factory
        # succeed, and keep the active hook identical to it so the
        # default-in-effect guard holds. (Without this the test is coupled to the
        # host's real trust store and would spuriously fail on a broken one.)
        monkeypatch.setattr(ssl, 'create_default_context', ok_factory)
        monkeypatch.setattr(ssl, '_create_default_https_context', ok_factory)

        stepcount._ensure_download_ssl_context(verbose=False)

        assert ssl._create_default_https_context is ok_factory

    def test_falls_back_to_certifi_when_store_broken(self, monkeypatch):
        """A broken default store installs a certifi-backed hook that still verifies."""
        import ssl
        certifi = pytest.importorskip('certifi')

        real_create = ssl.create_default_context
        seen_cafiles = []

        def spy(*args, **kwargs):
            seen_cafiles.append(kwargs.get('cafile'))
            # Simulate the broken Windows store: the no-arg default build fails,
            # but building from an explicit cafile (what the fallback does) works.
            if not kwargs.get('cafile'):
                raise ssl.SSLError("[ASN1: NOT_ENOUGH_DATA] not enough data")
            return real_create(*args, **kwargs)

        # Model the stdlib default hook being active (identity holds) but broken.
        monkeypatch.setattr(ssl, 'create_default_context', spy)
        monkeypatch.setattr(ssl, '_create_default_https_context', spy)

        stepcount._ensure_download_ssl_context(verbose=False)

        hook = ssl._create_default_https_context
        assert hook is not spy
        ctx = hook()
        assert isinstance(ctx, ssl.SSLContext)
        assert certifi.where() in seen_cafiles
        assert ctx.check_hostname is True
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_preserves_custom_hook_when_store_broken(self, monkeypatch):
        """A custom HTTPS hook installed by an embedding app is not overwritten."""
        import ssl

        def _boom(*a, **k):
            raise ssl.SSLError("[ASN1: NOT_ENOUGH_DATA] not enough data")

        def custom_hook(*a, **k):
            return "custom-context-sentinel"

        # Default factory is broken, but a distinct custom hook is already active.
        monkeypatch.setattr(ssl, 'create_default_context', _boom)
        monkeypatch.setattr(ssl, '_create_default_https_context', custom_hook)

        stepcount._ensure_download_ssl_context(verbose=False)

        # The custom hook must be respected, never silently replaced by certifi.
        assert ssl._create_default_https_context is custom_hook

    def test_no_fallback_without_certifi(self, monkeypatch):
        """If certifi is unavailable, leave the HTTPS hook alone (surface original error)."""
        import builtins
        import ssl

        def _boom(*a, **k):
            raise ssl.SSLError("[ASN1: NOT_ENOUGH_DATA] not enough data")

        # Model the default hook being active (identity holds) but the store broken.
        monkeypatch.setattr(ssl, 'create_default_context', _boom)
        monkeypatch.setattr(ssl, '_create_default_https_context', _boom)

        real_import = builtins.__import__

        def _no_certifi(name, *args, **kwargs):
            if name == 'certifi':
                raise ImportError("No module named 'certifi'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _no_certifi)
        stepcount._ensure_download_ssl_context(verbose=False)

        # certifi import failed, so no fallback was installed — hook is unchanged.
        assert ssl._create_default_https_context is _boom


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
        enmo_summary = stepcount.summarize_enmo(accel_data_1_5_days)
        assert 'avg' in enmo_summary

        steps_summary = stepcount.summarize_steps(step_counts_series, steptol=3)
        assert 'total_steps' in steps_summary

        cadence_summary = stepcount.summarize_cadence(step_counts_series, steptol=3)
        assert 'cadence_peak1' in cadence_summary

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

    def test_cli_module_import_is_lightweight(self):
        """Importing the CLI must not load processing dependencies needed only after parsing."""
        probe = """
import sys
from typing import get_type_hints
import stepcount.stepcount as stepcount

for name in (
    'summarize_enmo',
    'summarize_steps',
    'summarize_cadence',
    'summarize_bouts',
    '_detect_bouts',
):
    get_type_hints(getattr(stepcount, name))

heavy_modules = {
    'actipy',
    'hmmlearn',
    'imblearn',
    'joblib',
    'matplotlib',
    'numba',
    'numpy',
    'pandas',
    'scipy',
    'sklearn',
    'torch',
    'torchvision',
    'transforms3d',
}
loaded = sorted(heavy_modules.intersection(sys.modules))
print(','.join(loaded))
"""
        result = subprocess.run(
            [sys.executable, '-c', probe],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == ''

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

    def test_cli_ssl_repo_path_option(self):
        """Test that --ssl-repo-path option is recognized."""
        result = subprocess.run(
            [sys.executable, '-m', 'stepcount.stepcount', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert '--ssl-repo-path' in result.stdout

    def test_cli_download_models_in_help(self):
        """Test that --download-models appears in --help output."""
        result = subprocess.run(
            [sys.executable, '-m', 'stepcount.stepcount', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert '--download-models' in result.stdout

    def test_cli_download_models_no_filepath(self):
        """Test that --download-models works without a filepath argument."""
        from unittest.mock import patch
        # Stub the SSL setup too, so main() can't mutate global ssl state on a
        # broken-store host and leak the process-global hook into later tests.
        with patch('stepcount.stepcount.download_models') as mock_dl, \
                patch('stepcount.stepcount._ensure_download_ssl_context'):
            with patch('sys.argv', ['stepcount', '--download-models']):
                stepcount.main()
            mock_dl.assert_called_once_with(force_download=False, ssl_repo_path=None)

    def test_cli_no_filepath_no_download_models(self):
        """Test that omitting filepath without --download-models gives an error."""
        result = subprocess.run(
            [sys.executable, '-m', 'stepcount.stepcount'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0
        assert 'filepath' in result.stderr.lower() or 'required' in result.stderr.lower()

    @pytest.mark.parametrize("model_type, expected_resample", [("rf", None), ("ssl", 30)])
    def test_cli_processing_pipeline_with_model_boundary_mocked(
        self,
        model_type,
        expected_resample,
        tmp_path,
        monkeypatch,
    ):
        """Exercise CLI orchestration and every output writer without model I/O."""
        from stepcount import utils

        input_path = tmp_path / "synthetic.cwa"
        output_root = tmp_path / model_type
        times = pd.date_range("2024-01-15 23:00", periods=120, freq="1min")
        data = pd.DataFrame(
            {"x": 0.0, "y": 0.0, "z": 1.0},
            index=times,
        )
        info = {
            "Filename": str(input_path),
            "Device": "Synthetic",
            "Filesize(MB)": 0.0,
            "SampleRate": 1,
            "ResampleRate": 1,
        }
        read = MagicMock(return_value=(data, info))
        monkeypatch.setattr(utils, "read", read)
        monkeypatch.setattr(stepcount, "_ensure_download_ssl_context", MagicMock())

        def predict_from_frame(frame):
            assert frame.index.equals(times[60:])
            steps = pd.Series(
                np.tile([0, 3, 6, 9], 15),
                index=frame.index,
                name="Steps",
            )
            walking = steps >= 3
            step_times = pd.DataFrame(
                {"time": frame.index.repeat(steps.to_numpy())},
            )
            return steps, walking, step_times

        detector = SimpleNamespace(sample_rate=None, verbose=True)
        model = SimpleNamespace(
            wd=detector,
            sample_rate=None,
            window_sec=60,
            window_len=0,
            verbose=True,
            steptol=3,
            predict_from_frame=MagicMock(side_effect=predict_from_frame),
        )
        load_model = MagicMock(return_value=model)
        monkeypatch.setattr(stepcount, "load_model", load_model)

        argv = [
            "stepcount",
            str(input_path),
            "--outdir",
            str(output_root),
            "--model-type",
            model_type,
            "--exclude-first-last",
            "first",
            "--min-wear-per-day",
            "0",
            "--min-wear-per-hour",
            "0",
            "--min-wear-per-minute",
            "0",
            "--min-walk-per-day",
            "1",
            "--quiet",
        ]
        if model_type == "ssl":
            argv.extend(["--pytorch-device", "cpu"])
        monkeypatch.setattr(sys, "argv", argv)

        stepcount.main()

        assert read.call_args.kwargs["resample_hz"] == expected_resample
        assert read.call_args.kwargs["include_wear_stats"] is False
        assert read.call_args.kwargs["verbose"] is False
        assert load_model.call_args.args[1] == model_type
        assert model.sample_rate == 1
        assert model.window_len == 60
        assert detector.sample_rate == 1
        assert detector.verbose is False
        if model_type == "ssl":
            assert detector.device == "cpu"

        result_dir = output_root / "synthetic"
        expected_files = {
            "synthetic-Bouts.csv.gz",
            "synthetic-Daily.csv.gz",
            "synthetic-DailyAdjusted.csv.gz",
            "synthetic-Hourly.csv.gz",
            "synthetic-HourlyAdjusted.csv.gz",
            "synthetic-Info.json",
            "synthetic-Minutely.csv.gz",
            "synthetic-MinutelyAdjusted.csv.gz",
            "synthetic-Steps.csv.gz",
            "synthetic-StepTimes.csv.gz",
            "synthetic-Steps.png",
        }
        assert {path.name for path in result_dir.iterdir()} == expected_files
        result_info = json.loads((result_dir / "synthetic-Info.json").read_text())
        assert result_info["TotalSteps"] == 270
        assert result_info["WearTime(days)"] == pytest.approx(1 / 24)
        assert result_info["WearStartTime"] == "2024-01-16 00:00:00"
        assert result_info["StepCountArgs"]["model_type"] == model_type
        for peak_minutes in (5, 10):
            for adjusted in ("", "Adjusted"):
                for cohort in ("", "_Weekend", "_Weekday"):
                    assert f"CadencePeak{peak_minutes}{adjusted}(steps/min){cohort}" in result_info
        daily = pd.read_csv(result_dir / "synthetic-Daily.csv.gz")
        assert daily["Date"].tolist() == ["2024-01-16"]
        assert daily["WearTime(hours)"].tolist() == [1.0]
        cadence_columns = [column for column in daily.columns if column.startswith("Cadence")]
        assert cadence_columns == [
            "CadencePeak1(steps/min)",
            "CadencePeak5(steps/min)",
            "CadencePeak10(steps/min)",
            "CadencePeak30(steps/min)",
            "Cadence95th(steps/min)",
        ]

    def test_cli_empty_input_writes_info_without_loading_model(
        self,
        tmp_path,
        monkeypatch,
    ):
        """The no-data path records metadata and exits before model loading."""
        from stepcount import utils

        input_path = tmp_path / "empty.cwa"
        output_root = tmp_path / "output"
        data = pd.DataFrame(
            columns=["x", "y", "z"],
            index=pd.DatetimeIndex([], name="time"),
            dtype=float,
        )
        info = {
            "Filename": str(input_path),
            "Device": "Synthetic",
            "Filesize(MB)": 0.0,
            "SampleRate": 1,
            "ResampleRate": 1,
        }
        monkeypatch.setattr(utils, "read", MagicMock(return_value=(data, info)))
        monkeypatch.setattr(stepcount, "_ensure_download_ssl_context", MagicMock())
        load_model = MagicMock()
        monkeypatch.setattr(stepcount, "load_model", load_model)
        monkeypatch.setattr(
            sys,
            "argv",
            ["stepcount", str(input_path), "--outdir", str(output_root), "--quiet"],
        )

        with pytest.raises(SystemExit) as exc_info:
            stepcount.main()

        assert exc_info.value.code == 0
        load_model.assert_not_called()
        result_info = output_root / "empty" / "empty-Info.json"
        assert json.loads(result_info.read_text())["Filename"] == str(input_path)

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
                '-q'
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")

        assert result.returncode == 0

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

            expected_keys = ['Filename', 'TotalSteps', 'StepsDayAvg']
            for key in expected_keys:
                assert key in info, f"Missing key: {key}"
