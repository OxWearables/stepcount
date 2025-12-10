"""
Pytest fixtures for stepcount test suite.

Generates realistic accelerometer data simulating 1.5+ days of wrist-worn
sensor data at 15Hz sampling rate.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil


# Constants for data generation
# Using 15Hz for test fixtures - minimum needed for filtering.
# Nyquist = 7.5Hz > 5Hz lowpass cutoff in features.py
SAMPLE_RATE = 15  # Hz
GRAVITY = 1.0  # g units


def generate_resting_signal(n_samples, noise_std=0.02):
    """
    Generate resting/sedentary accelerometer signal.
    Simulates arm at rest with gravity primarily on one axis.
    """
    x = np.random.normal(0.0, noise_std, n_samples)
    y = np.random.normal(0.0, noise_std, n_samples)
    z = np.random.normal(GRAVITY, noise_std, n_samples)  # gravity on z-axis
    return np.column_stack([x, y, z])


def generate_walking_signal(n_samples, cadence_hz=1.8, amplitude=0.3, noise_std=0.05):
    """
    Generate walking accelerometer signal.
    Simulates periodic wrist movement during walking at ~108 steps/min.

    Args:
        n_samples: Number of samples to generate
        cadence_hz: Walking cadence in Hz (steps per second)
        amplitude: Peak acceleration amplitude in g
        noise_std: Noise standard deviation
    """
    t = np.arange(n_samples) / SAMPLE_RATE

    # Primary walking frequency with harmonics
    phase = np.random.uniform(0, 2 * np.pi)
    x = amplitude * np.sin(2 * np.pi * cadence_hz * t + phase)
    x += 0.3 * amplitude * np.sin(2 * np.pi * 2 * cadence_hz * t + phase)  # 2nd harmonic

    y = 0.5 * amplitude * np.sin(2 * np.pi * cadence_hz * t + phase + np.pi/4)

    # Z maintains gravity with vertical oscillation
    z = GRAVITY + 0.4 * amplitude * np.sin(2 * np.pi * cadence_hz * t + phase + np.pi/2)

    # Add noise
    x += np.random.normal(0, noise_std, n_samples)
    y += np.random.normal(0, noise_std, n_samples)
    z += np.random.normal(0, noise_std, n_samples)

    return np.column_stack([x, y, z])


def generate_activity_signal(n_samples, intensity=0.15, noise_std=0.03):
    """
    Generate light activity signal (e.g., fidgeting, typing).
    """
    t = np.arange(n_samples) / SAMPLE_RATE

    # Random low-frequency movements
    x = intensity * np.sin(2 * np.pi * 0.5 * t + np.random.uniform(0, 2*np.pi))
    y = intensity * np.sin(2 * np.pi * 0.3 * t + np.random.uniform(0, 2*np.pi))
    z = GRAVITY + 0.5 * intensity * np.sin(2 * np.pi * 0.4 * t)

    x += np.random.normal(0, noise_std, n_samples)
    y += np.random.normal(0, noise_std, n_samples)
    z += np.random.normal(0, noise_std, n_samples)

    return np.column_stack([x, y, z])


def generate_realistic_day(sample_rate=SAMPLE_RATE, seed=None):
    """
    Generate one day of realistic accelerometer data with mixed activities.

    Pattern:
    - Night (00:00-07:00): mostly resting with occasional movement
    - Morning (07:00-09:00): activity + walking bouts
    - Day (09:00-17:00): mixed activity, walking, sedentary
    - Evening (17:00-22:00): activity + walking bouts
    - Night (22:00-24:00): resting

    Returns:
        np.ndarray: (n_samples, 3) array of x, y, z acceleration
    """
    if seed is not None:
        np.random.seed(seed)

    samples_per_hour = sample_rate * 3600
    day_data = []

    # Night: 00:00 - 07:00 (7 hours of rest with occasional movement)
    for hour in range(7):
        hour_data = generate_resting_signal(samples_per_hour)
        # Add occasional movement (10% of the time)
        if np.random.random() < 0.1:
            start = np.random.randint(0, samples_per_hour - 60 * sample_rate)
            duration = np.random.randint(30, 120) * sample_rate
            hour_data[start:start+duration] = generate_activity_signal(duration)
        day_data.append(hour_data)

    # Morning: 07:00 - 09:00 (wake up, activity, morning walk)
    for hour in range(2):
        hour_data = generate_activity_signal(samples_per_hour)
        # Walking bout (20-40 minutes)
        walk_start = np.random.randint(0, samples_per_hour // 2)
        walk_duration = np.random.randint(20, 40) * 60 * sample_rate
        walk_duration = min(walk_duration, samples_per_hour - walk_start)
        hour_data[walk_start:walk_start+walk_duration] = generate_walking_signal(walk_duration)
        day_data.append(hour_data)

    # Day: 09:00 - 17:00 (8 hours work - mixed sedentary, activity, brief walks)
    for hour in range(8):
        if np.random.random() < 0.4:  # 40% sedentary
            hour_data = generate_resting_signal(samples_per_hour)
        else:  # 60% light activity
            hour_data = generate_activity_signal(samples_per_hour)

        # Brief walking bouts (5-15 min, 30% probability)
        if np.random.random() < 0.3:
            walk_start = np.random.randint(0, samples_per_hour - 15 * 60 * sample_rate)
            walk_duration = np.random.randint(5, 15) * 60 * sample_rate
            hour_data[walk_start:walk_start+walk_duration] = generate_walking_signal(walk_duration)
        day_data.append(hour_data)

    # Evening: 17:00 - 22:00 (5 hours - activity, walking, rest)
    for hour in range(5):
        hour_data = generate_activity_signal(samples_per_hour)
        # Evening walk (15-30 min, 40% probability)
        if np.random.random() < 0.4:
            walk_start = np.random.randint(0, samples_per_hour - 30 * 60 * sample_rate)
            walk_duration = np.random.randint(15, 30) * 60 * sample_rate
            hour_data[walk_start:walk_start+walk_duration] = generate_walking_signal(walk_duration)
        day_data.append(hour_data)

    # Late night: 22:00 - 24:00 (2 hours of rest)
    for hour in range(2):
        day_data.append(generate_resting_signal(samples_per_hour))

    return np.vstack(day_data)


@pytest.fixture(scope="session")
def sample_rate():
    """Return the sample rate used for test data."""
    return SAMPLE_RATE


@pytest.fixture(scope="session")
def accel_data_1_5_days():
    """
    Generate 1.5 days of realistic accelerometer data as a DataFrame.

    Returns DataFrame with:
    - DatetimeIndex starting from 2024-01-15 00:00:00
    - Columns: x, y, z (acceleration in g)
    - 15Hz sampling rate
    - ~1.5 days = 36 hours of data
    """
    np.random.seed(42)

    # Generate 1.5 days
    day1 = generate_realistic_day(seed=42)
    half_day = generate_realistic_day(seed=43)[:SAMPLE_RATE * 3600 * 12]  # 12 hours

    data = np.vstack([day1, half_day])

    # Create datetime index
    start_time = pd.Timestamp('2024-01-15 00:00:00')
    n_samples = len(data)
    time_index = pd.date_range(
        start=start_time,
        periods=n_samples,
        freq=f'{1000000//SAMPLE_RATE}us'  # ~33.3ms for 30Hz
    )

    df = pd.DataFrame(data, columns=['x', 'y', 'z'], index=time_index)
    df.index.name = 'time'

    return df


@pytest.fixture(scope="session")
def accel_data_2_days():
    """
    Generate 2 full days of realistic accelerometer data.
    Useful for testing weekend/weekday splits.

    Days are set to a Friday and Saturday for testing.
    """
    np.random.seed(123)

    day1 = generate_realistic_day(seed=123)  # Friday
    day2 = generate_realistic_day(seed=456)  # Saturday

    data = np.vstack([day1, day2])

    # Start on a Friday (2024-01-19)
    start_time = pd.Timestamp('2024-01-19 00:00:00')  # This is a Friday
    n_samples = len(data)
    time_index = pd.date_range(
        start=start_time,
        periods=n_samples,
        freq=f'{1000000//SAMPLE_RATE}us'
    )

    df = pd.DataFrame(data, columns=['x', 'y', 'z'], index=time_index)
    df.index.name = 'time'

    return df


@pytest.fixture(scope="session")
def accel_data_with_nonwear(accel_data_1_5_days):
    """
    Accelerometer data with simulated non-wear periods (NaN values).

    Non-wear periods:
    - 2 hours in the middle of day 1 (simulating device removed)
    - 30 minutes in the morning of day 2
    """
    df = accel_data_1_5_days.copy()

    # Non-wear period 1: Day 1, 14:00-16:00
    nonwear_start1 = pd.Timestamp('2024-01-15 14:00:00')
    nonwear_end1 = pd.Timestamp('2024-01-15 16:00:00')
    df.loc[nonwear_start1:nonwear_end1, :] = np.nan

    # Non-wear period 2: Day 2, 08:30-09:00
    nonwear_start2 = pd.Timestamp('2024-01-16 08:30:00')
    nonwear_end2 = pd.Timestamp('2024-01-16 09:00:00')
    df.loc[nonwear_start2:nonwear_end2, :] = np.nan

    return df


@pytest.fixture(scope="session")
def accel_window_walking():
    """Single 10-second window of walking data at 15Hz."""
    np.random.seed(100)
    window_samples = 10 * SAMPLE_RATE  # 10 seconds
    return generate_walking_signal(window_samples, cadence_hz=1.8)


@pytest.fixture(scope="session")
def accel_window_resting():
    """Single 10-second window of resting data at 15Hz."""
    np.random.seed(101)
    window_samples = 10 * SAMPLE_RATE  # 10 seconds
    return generate_resting_signal(window_samples)


@pytest.fixture(scope="session")
def accel_windows_mixed():
    """
    Array of multiple windows with mixed activities.
    Returns (n_windows, n_samples, 3) array.

    Mix: 30% walking, 30% resting, 40% light activity
    """
    np.random.seed(200)
    n_windows = 100
    window_samples = 10 * SAMPLE_RATE

    windows = []
    labels = []  # 1 = walking, 0 = not walking

    for i in range(n_windows):
        r = np.random.random()
        if r < 0.3:  # walking
            windows.append(generate_walking_signal(window_samples, cadence_hz=np.random.uniform(1.5, 2.2)))
            labels.append(1)
        elif r < 0.6:  # resting
            windows.append(generate_resting_signal(window_samples))
            labels.append(0)
        else:  # light activity
            windows.append(generate_activity_signal(window_samples))
            labels.append(0)

    return np.array(windows), np.array(labels)


@pytest.fixture(scope="session")
def step_counts_series(accel_data_1_5_days):
    """
    Generate a step count series that aligns with accel_data_1_5_days.
    Returns a Series indexed by 10-second windows with realistic step counts.
    """
    np.random.seed(42)

    # Create window-level timestamps
    start_time = accel_data_1_5_days.index[0]
    window_sec = 10
    n_windows = len(accel_data_1_5_days) // (SAMPLE_RATE * window_sec)

    window_times = pd.date_range(
        start=start_time,
        periods=n_windows,
        freq=f'{window_sec}s'
    )

    # Generate step counts based on time of day
    step_counts = []
    for t in window_times:
        hour = t.hour

        # Night hours: mostly 0 steps
        if hour < 7 or hour >= 22:
            steps = np.random.choice([0, 0, 0, 0, 0, 3, 4, 5], p=[0.7, 0.1, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025])
        # Active hours
        elif 7 <= hour < 9 or 17 <= hour < 20:
            steps = np.random.choice([0, 3, 5, 8, 12, 15, 18], p=[0.3, 0.1, 0.15, 0.15, 0.15, 0.1, 0.05])
        # Work hours: mostly sedentary with occasional walking
        else:
            steps = np.random.choice([0, 0, 3, 5, 8, 12], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])

        step_counts.append(steps)

    return pd.Series(step_counts, index=window_times, name='Steps')


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test outputs, cleaned up after test."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="session")
def temp_csv_file(accel_data_1_5_days):
    """
    Create a temporary CSV file with accelerometer data.
    Returns path to the file (cleaned up after session).
    """
    tmpdir = tempfile.mkdtemp()
    csv_path = Path(tmpdir) / "test_accel_data.csv.gz"

    # Reset index to have time as column
    df = accel_data_1_5_days.reset_index()
    df.to_csv(csv_path, index=False)

    yield csv_path

    shutil.rmtree(tmpdir)


@pytest.fixture(scope="session")
def mock_info_json():
    """Sample Info.json content for testing collate_outputs."""
    return {
        "Filename": "test_subject.csv",
        "Device": ".csv",
        "Filesize(MB)": 10.5,
        "SampleRate": 10,
        "ResampleRate": 10,
        "StartTime": "2024-01-15 00:00:00",
        "EndTime": "2024-01-16 12:00:00",
        "WearTime(days)": 1.5,
        "NonwearTime(days)": 0.0,
        "TotalSteps": 8500,
        "StepsDayAvg": 8500,
        "ENMO(mg)": 25.5,
    }


# Utility functions available for tests
@pytest.fixture(scope="session")
def generate_walking():
    """Return the walking signal generator function."""
    return generate_walking_signal


@pytest.fixture(scope="session")
def generate_resting():
    """Return the resting signal generator function."""
    return generate_resting_signal
