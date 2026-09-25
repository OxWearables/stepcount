"""End-to-end numerical regression tests for the real CWA fixture."""

import hashlib
import json
import math
import os
import subprocess
import sys
from numbers import Integral, Real
from pathlib import Path

import pandas as pd
import pytest

from stepcount import __model_md5__, __model_version__

RUN_MODEL_REGRESSION = os.environ.get("STEPCOUNT_RUN_MODEL_REGRESSION") == "1"
DATA_DIR = Path(__file__).parent / "data"
FIXTURE = DATA_DIR / "tiny-sample.cwa.gz"
BASELINE = DATA_DIR / "tiny-sample-baseline.json"
FLOAT_COMPARISON_DECIMALS = 6


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _steps_signature(frame):
    values = [
        [str(timestamp), None if pd.isna(value) else int(value)]
        for timestamp, value in zip(frame["time"], frame["Steps"])
    ]
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_value(value):
    if pd.isna(value):
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, Integral):
        return ["int", int(value)]
    if isinstance(value, Real):
        rounded = round(float(value), FLOAT_COMPARISON_DECIMALS)
        return ["float", 0.0 if rounded == 0 else rounded]
    return ["string", str(value)]


def _frame_signature(frame):
    canonical = frame.copy()
    if "Filename" in canonical:
        canonical["Filename"] = canonical["Filename"].map(
            lambda value: Path(str(value)).name if not pd.isna(value) else value
        )
    payload = {
        "columns": canonical.columns.tolist(),
        "records": [
            [_canonical_value(value) for value in row]
            for row in canonical.itertuples(index=False, name=None)
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.model_regression
@pytest.mark.skipif(
    not RUN_MODEL_REGRESSION,
    reason="set STEPCOUNT_RUN_MODEL_REGRESSION=1 to run model-backed tests",
)
@pytest.mark.parametrize("model_type", ["rf", "ssl"])
def test_tiny_cwa_matches_versioned_numerical_baseline(model_type, tmp_path):
    baseline = json.loads(BASELINE.read_text())
    expected = baseline["models"][model_type]

    assert baseline["schema_version"] == 1
    assert baseline["float_comparison_decimals"] == FLOAT_COMPARISON_DECIMALS
    assert _sha256(FIXTURE) == baseline["fixture"]["sha256"]

    configured_output = os.environ.get("STEPCOUNT_REGRESSION_OUTPUT_DIR")
    if configured_output is None:
        output_root = tmp_path / model_type
    else:
        output_root = Path(configured_output) / model_type
    command = [
        sys.executable,
        "-m",
        "stepcount.stepcount",
        str(FIXTURE),
        "--outdir",
        str(output_root),
        "--model-type",
        model_type,
        "--quiet",
    ]
    if model_type == "ssl":
        command.extend(["--pytorch-device", "cpu"])

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    output_dir = output_root / "tiny-sample"
    assert output_dir.is_dir(), f"regression output directory not created: {output_dir}"
    assert (
        __model_version__[model_type] == expected["model_version"]
    ), "model version mismatch"
    assert __model_md5__[model_type] == expected["model_md5"], "model MD5 mismatch"
    expected_outputs = {
        "tiny-sample-Bouts.csv.gz",
        "tiny-sample-Daily.csv.gz",
        "tiny-sample-DailyAdjusted.csv.gz",
        "tiny-sample-Hourly.csv.gz",
        "tiny-sample-HourlyAdjusted.csv.gz",
        "tiny-sample-Info.json",
        "tiny-sample-Minutely.csv.gz",
        "tiny-sample-MinutelyAdjusted.csv.gz",
        "tiny-sample-Steps.csv.gz",
        "tiny-sample-StepTimes.csv.gz",
        "tiny-sample-Steps.png",
    }
    outputs = {path.name: path for path in output_dir.iterdir()}
    assert outputs.keys() == expected_outputs

    info = json.loads(outputs["tiny-sample-Info.json"].read_text())
    for expectations in (baseline["fixture"], expected):
        for key, expected_value in expectations["info_exact"].items():
            assert info[key] == expected_value, key
        for key, (expected_value, tolerance) in expectations["info_approx"].items():
            assert info[key] == pytest.approx(expected_value, abs=tolerance, rel=0), key
    assert math.isnan(info["TotalStepsAdjusted"])
    assert math.isnan(info["ENMOAdjusted(mg)"])

    steps = pd.read_csv(outputs["tiny-sample-Steps.csv.gz"])
    step_expected = expected["steps"]
    assert list(steps.columns) == ["time", "Steps"]
    assert len(steps) == step_expected["rows"]
    assert int(steps["Steps"].notna().sum()) == step_expected["non_null"]
    assert int(steps["Steps"].sum()) == step_expected["sum"]
    assert steps["time"].iloc[0] == step_expected["first_time"]
    assert steps["time"].iloc[-1] == step_expected["last_time"]
    assert _steps_signature(steps) == step_expected["sha256"]

    step_times = pd.read_csv(outputs["tiny-sample-StepTimes.csv.gz"])
    step_times_expected = expected["step_times"]
    assert list(step_times.columns) == ["time"]
    assert len(step_times) == step_times_expected["rows"]
    assert step_times["time"].iloc[0] == step_times_expected["first_time"]
    assert step_times["time"].iloc[-1] == step_times_expected["last_time"]
    assert _frame_signature(step_times) == step_times_expected["sha256"]
    assert len(step_times) == int(steps["Steps"].sum()) == info["TotalSteps"]

    export_files = {
        "bouts": "tiny-sample-Bouts.csv.gz",
        "daily": "tiny-sample-Daily.csv.gz",
        "daily_adjusted": "tiny-sample-DailyAdjusted.csv.gz",
        "hourly": "tiny-sample-Hourly.csv.gz",
        "hourly_adjusted": "tiny-sample-HourlyAdjusted.csv.gz",
        "minutely": "tiny-sample-Minutely.csv.gz",
        "minutely_adjusted": "tiny-sample-MinutelyAdjusted.csv.gz",
    }
    for export_name, filename in export_files.items():
        frame = pd.read_csv(outputs[filename])
        export_expected = expected["exports"][export_name]
        assert frame.columns.tolist() == export_expected["columns"], export_name
        assert len(frame) == export_expected["rows"], export_name
        assert _frame_signature(frame) == export_expected["sha256"], export_name


def test_model_identity_mismatch_keeps_generated_outputs(tmp_path, monkeypatch):
    baseline = json.loads(BASELINE.read_text())
    baseline["models"]["rf"]["model_md5"] = "outdated"
    mismatched_baseline = tmp_path / "baseline.json"
    mismatched_baseline.write_text(json.dumps(baseline))
    monkeypatch.setattr(sys.modules[__name__], "BASELINE", mismatched_baseline)

    diagnostic_root = tmp_path / "diagnostics"
    monkeypatch.setenv("STEPCOUNT_REGRESSION_OUTPUT_DIR", str(diagnostic_root))

    def fake_run(command, **kwargs):
        output_root = Path(command[command.index("--outdir") + 1])
        output_dir = output_root / "tiny-sample"
        output_dir.mkdir(parents=True)
        (output_dir / "candidate-output.txt").write_text("generated")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AssertionError, match="model MD5 mismatch"):
        test_tiny_cwa_matches_versioned_numerical_baseline("rf", tmp_path)

    assert (
        diagnostic_root / "rf" / "tiny-sample" / "candidate-output.txt"
    ).read_text() == "generated"
