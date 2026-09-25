"""Numerical regression tests for the public summary calculations.

The inputs here are deliberately small and hand-calculable.  These tests pin
business results, rather than only checking output shapes or broad invariants.
"""

import numpy as np
import pandas as pd
import pytest

from stepcount import stepcount, utils


def test_wear_summaries_match_hand_calculated_values():
    times = pd.date_range("2024-01-15", periods=6, freq="1h")
    data = pd.DataFrame(
        {
            "x": [0.0, 0.0, np.nan, 0.0, np.nan, 0.0],
            "y": [0.0, 0.0, np.nan, 0.0, np.nan, 0.0],
            "z": [1.0, 1.0, np.nan, 1.0, np.nan, 1.0],
        },
        index=times,
    )

    result = utils.calculate_wear_stats(data)
    daily = utils.calculate_daily_wear_stats(data)

    assert result == {
        "StartTime": "2024-01-15 00:00:00",
        "EndTime": "2024-01-15 05:00:00",
        "WearStartTime": "2024-01-15 00:00:00",
        "WearEndTime": "2024-01-15 05:00:00",
        "WearTime(days)": pytest.approx(4 / 24),
        "NonwearTime(days)": pytest.approx(2 / 24),
        "Covers24hOK": 0,
    }
    assert daily.index.tolist() == [pd.Timestamp("2024-01-15")]
    assert daily["WearTime(hours)"].tolist() == [4.0]


def test_enmo_summary_matches_hand_calculated_values():
    times = pd.date_range("2024-01-19 23:56", periods=8, freq="1min")
    enmo_mg = np.arange(8, dtype=float) * 100
    data = pd.DataFrame(
        {"x": 0.0, "y": 0.0, "z": 1.0 + enmo_mg / 1000},
        index=times,
    )

    result = stepcount.summarize_enmo(data)

    pd.testing.assert_series_equal(
        result["minutely"],
        pd.Series(enmo_mg, index=times, name="ENMO(mg)"),
    )
    pd.testing.assert_series_equal(
        result["daily"],
        pd.Series(
            [150.0, 550.0],
            index=pd.to_datetime(["2024-01-19", "2024-01-20"]),
            name="ENMO(mg)",
        ),
        check_freq=False,
    )
    assert result["avg"] == pytest.approx(350.0)
    assert result["weekday_avg"] == pytest.approx(150.0)
    assert result["weekend_avg"] == pytest.approx(550.0)
    assert result["hour_avgs"].loc[23] == pytest.approx(150.0)
    assert result["hour_avgs"].loc[0] == pytest.approx(550.0)


def test_step_summary_matches_hand_calculated_values():
    times = pd.date_range("2024-01-19 23:56", periods=8, freq="1min")
    steps = pd.Series([0, 3, 6, 9, 12, 15, 0, 3], index=times, name="Steps")

    result = stepcount.summarize_steps(steps, steptol=3)

    assert result["total_steps"] == 48
    assert result["avg_steps"] == 24
    assert result["med_steps"] == 24
    assert result["min_steps"] == 18
    assert result["max_steps"] == 30
    assert result["weekday_total_steps"] == 18
    assert result["weekend_total_steps"] == 30
    assert result["total_walk"] == pytest.approx(6.0)
    assert result["weekday_total_walk"] == pytest.approx(3.0)
    assert result["weekend_total_walk"] == pytest.approx(3.0)
    assert result["daily_steps"]["Steps"].tolist() == [18, 30]
    assert result["daily_steps"]["Walk(mins)"].tolist() == [3.0, 3.0]
    assert result["daily_steps"]["Steps50thAt"].tolist() == ["23:58:00", "00:01:00"]
    assert result["ptile_at_avgs"].to_dict() == {
        "p05_at": "23:58:30",
        "p25_at": "23:59:00",
        "p50_at": "23:59:30",
        "p75_at": "00:00:00",
        "p95_at": "00:01:00",
    }
    assert result["hour_steps"].loc[23] == 18
    assert result["hour_steps"].loc[0] == 30


def test_step_summary_clock_time_average_is_unchanged_away_from_midnight():
    times = pd.date_range("2024-01-15", periods=2 * 24 * 60, freq="1min")
    steps = pd.Series(0, index=times, name="Steps")
    steps.loc["2024-01-15 10:00"] = 3
    steps.loc["2024-01-16 10:02"] = 3

    result = stepcount.summarize_steps(steps, steptol=3)

    assert result["ptile_at_avgs"].tolist() == ["10:01:00"] * 5


def test_adjusted_step_summary_imputes_a_known_gap():
    times = pd.date_range("2024-01-15", periods=24 * 60, freq="1min")
    steps = pd.Series(1.0, index=times, name="Steps")
    steps.iloc[12 * 60 + 2] = np.nan

    result = stepcount.summarize_steps(
        steps,
        steptol=3,
        adjust_estimates=True,
        min_wear_per_day=0,
        min_wear_per_hour=0,
        min_wear_per_minute=0,
    )

    assert result["total_steps"] == 1440
    assert result["daily_steps"]["Steps"].tolist() == [1440]
    assert result["total_walk"] == pytest.approx(0.0)
    assert result["minutely_steps"].isna().sum() == 0


def test_cadence_summary_matches_hand_calculated_values():
    times = pd.date_range("2024-01-15", periods=5, freq="1min")
    steps = pd.Series([0, 30, 60, 90, 120], index=times, name="Steps")

    result = stepcount.summarize_cadence(
        steps,
        steptol=3,
        min_walk_per_day=1,
    )

    assert result["cadence_peak1"] == 120
    assert result["cadence_peak30"] == 75
    assert result["cadence_p95"] == 115
    assert result["daily"].iloc[0].to_dict() == {
        "CadencePeak1(steps/min)": 120,
        "CadencePeak30(steps/min)": 75,
        "Cadence95th(steps/min)": 115,
    }


def test_bout_summary_matches_hand_calculated_values():
    times = pd.date_range("2024-01-15", periods=8, freq="1min")
    steps = pd.Series([0, 3, 6, 0, 0, 9, 12, 0], index=times, name="Steps")
    data = pd.DataFrame(
        {"x": 0.0, "y": 0.0, "z": 1.0 + np.arange(8) / 10},
        index=times,
    )

    bouts = stepcount.summarize_bouts(
        steps,
        data,
        steptol=3,
        bouts_min_walk=1.0,
        bouts_max_idle=0,
    )["bouts"]

    assert bouts["StartTime"].tolist() == [
        "2024-01-15 00:01:00",
        "2024-01-15 00:05:00",
    ]
    assert bouts["EndTime"].tolist() == [
        "2024-01-15 00:02:00",
        "2024-01-15 00:06:00",
    ]
    np.testing.assert_allclose(bouts["Duration(mins)"], [2.0, 2.0])
    np.testing.assert_allclose(bouts["Steps"], [9.0, 21.0])
    np.testing.assert_allclose(bouts["Cadence(steps/min)"], [4.5, 10.5])
    np.testing.assert_allclose(bouts["Cadence50th(steps/min)"], [4.5, 10.5])
    np.testing.assert_allclose(bouts["ENMO(mg)"], [150.0, 550.0])
    assert np.isnan(bouts["TimeSinceLast(mins)"].iloc[0])
    assert bouts["TimeSinceLast(mins)"].iloc[1] == pytest.approx(3.0)
