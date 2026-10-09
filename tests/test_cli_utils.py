"""
Tests for stepcount.cli_utils module.

Tests cover:
- collate_outputs functions
- generate_commands functions
"""
import csv
import gzip
import importlib
import json
import os
import stat
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

# Import the module files directly using importlib
# (The __init__.py exports functions with same names as the module files,
# which shadows normal `import stepcount.cli_utils.collate_outputs` approach)
collate_mod = importlib.import_module('stepcount.cli_utils.collate_outputs')
gencmd_mod = importlib.import_module('stepcount.cli_utils.generate_commands')


class TestCollateJsons:
    """Tests for JSON collation."""

    def test_collate_jsons_basic(self, temp_dir, mock_info_json):
        """Test basic JSON collation."""
        # Create test JSON files
        json_dir = temp_dir / "results"
        json_dir.mkdir()

        for i in range(3):
            info = mock_info_json.copy()
            info['Filename'] = f"subject_{i}.csv"
            info['TotalSteps'] = 8000 + i * 100

            json_file = json_dir / f"subject_{i}" / f"subject_{i}-Info.json"
            json_file.parent.mkdir(parents=True)
            with open(json_file, 'w') as f:
                json.dump(info, f)

        # Collate
        outfile = temp_dir / "Info.csv.gz"
        json_files = list(json_dir.rglob("*-Info.json"))

        collate_mod.collate_jsons(json_files, outfile)

        # Verify output
        assert outfile.exists()
        df = pd.read_csv(outfile)
        assert len(df) == 3
        assert 'Filename' in df.columns
        assert 'TotalSteps' in df.columns

    def test_collate_jsons_overwrite(self, temp_dir, mock_info_json):
        """Test JSON collation overwrites existing file."""
        json_dir = temp_dir / "results"
        json_dir.mkdir()

        # Create one JSON file
        info = mock_info_json.copy()
        json_file = json_dir / "subject_0" / "subject_0-Info.json"
        json_file.parent.mkdir(parents=True)
        with open(json_file, 'w') as f:
            json.dump(info, f)

        outfile = temp_dir / "Info.csv.gz"

        # First collation
        collate_mod.collate_jsons([json_file], outfile)
        assert outfile.exists()

        # Second collation (should overwrite)
        collate_mod.collate_jsons([json_file], outfile)
        assert outfile.exists()

        df = pd.read_csv(outfile)
        assert len(df) == 1  # Not duplicated

    def test_collate_jsons_empty(self, temp_dir):
        """Test collation with empty file list."""
        outfile = temp_dir / "Info.csv.gz"

        collate_mod.collate_jsons([], outfile)

        assert outfile.exists()
        # Empty JSON list produces empty DataFrame which writes as empty CSV
        # pandas can't read empty CSV, so check file size is minimal
        file_size = os.path.getsize(outfile)
        # Gzipped empty DataFrame is very small (< 50 bytes)
        assert file_size < 50

    def test_load_json_frame_preserves_one_level_conversion(self, temp_dir):
        source = temp_dir / "subject-Info.json"
        source.write_text(
            '{"direct": {"second": 2, "first": 1}, '
            '"listed": [{"fourth": 4, "third": 3}]}',
            encoding="utf-8",
        )

        frame = collate_mod._load_json_frame([source])

        assert type(frame.at[0, "direct"]) is dict
        assert frame.at[0, "direct"] == {"second": 2, "first": 1}
        listed_value = frame.at[0, "listed"]
        assert isinstance(listed_value[0], OrderedDict)
        assert listed_value[0] == {"fourth": 4, "third": 3}


class TestCollateCsvs:
    """Tests for CSV collation."""

    def test_collate_csvs_basic(self, temp_dir):
        """Test basic CSV collation."""
        csv_dir = temp_dir / "results"
        csv_dir.mkdir()

        # Create test CSV files
        for i in range(3):
            csv_path = csv_dir / f"subject_{i}" / f"subject_{i}-Daily.csv.gz"
            csv_path.parent.mkdir(parents=True)

            df = pd.DataFrame({
                'Filename': [f'subject_{i}.csv'],
                'Date': ['2024-01-15'],
                'Steps': [8000 + i * 100]
            })
            df.to_csv(csv_path, index=False)

        # Collate
        outfile = temp_dir / "Daily.csv.gz"
        csv_files = list(csv_dir.rglob("*-Daily.csv.gz"))

        collate_mod.collate_csvs(csv_files, outfile)

        # Verify output
        assert outfile.exists()
        df = pd.read_csv(outfile)
        assert len(df) == 3
        assert 'Filename' in df.columns

    def test_collate_csvs_preserves_headers(self, temp_dir):
        """Test that CSV collation preserves headers correctly."""
        csv_dir = temp_dir / "results"
        csv_dir.mkdir()

        # Create CSV files with same structure
        for i in range(2):
            csv_path = csv_dir / f"subject_{i}-Daily.csv.gz"
            df = pd.DataFrame({
                'Filename': [f'subject_{i}.csv'],
                'Date': ['2024-01-15'],
                'Steps': [8000],
                'ENMO': [25.5]
            })
            df.to_csv(csv_path, index=False)

        outfile = temp_dir / "Daily.csv.gz"
        csv_files = list(csv_dir.glob("*-Daily.csv.gz"))

        collate_mod.collate_csvs(csv_files, outfile)

        df = pd.read_csv(outfile)
        # Should have header only once
        assert 'Filename' in df.columns
        assert len(df) == 2

    def test_csv_plan_interns_repeated_schemas(self, temp_dir):
        files = []
        for index in range(3):
            source = temp_dir / f"subject-{index}-Daily.csv.gz"
            pd.DataFrame({"Filename": [f"subject-{index}.csv"], "Steps": [index]}).to_csv(
                source,
                index=False,
            )
            files.append(source)

        plan = collate_mod._plan_csv_collation(files, "union")

        assert plan.schemas == (("Filename", "Steps"),)
        assert plan.files == tuple((file, 0) for file in files)

    @pytest.mark.parametrize("new_schema_first", [False, True])
    def test_collate_csvs_aligns_mixed_schemas_by_name(self, temp_dir, new_schema_first):
        old_file = temp_dir / "old-Daily.csv.gz"
        new_file = temp_dir / "new-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"

        pd.DataFrame({
            "Filename": ["old.csv"],
            "CadencePeak1(steps/min)": [101],
            "CadencePeak30(steps/min)": [71],
            "Cadence95th(steps/min)": [91],
            "ENMO(mg)": [12.5],
            "WearTime(hours)": [23.0],
        }).to_csv(old_file, index=False)
        pd.DataFrame({
            "Filename": ["new.csv"],
            "CadencePeak1(steps/min)": [102],
            "CadencePeak5(steps/min)": [92],
            "CadencePeak10(steps/min)": [82],
            "CadencePeak30(steps/min)": [72],
            "Cadence95th(steps/min)": [93],
            "ENMO(mg)": [13.5],
            "WearTime(hours)": [22.0],
        }).to_csv(new_file, index=False)

        files = [new_file, old_file] if new_schema_first else [old_file, new_file]
        collate_mod.collate_csvs(files, outfile)

        result = pd.read_csv(outfile)
        old_columns = [
            "Filename",
            "CadencePeak1(steps/min)",
            "CadencePeak30(steps/min)",
            "Cadence95th(steps/min)",
            "ENMO(mg)",
            "WearTime(hours)",
        ]
        new_columns = [
            "Filename",
            "CadencePeak1(steps/min)",
            "CadencePeak5(steps/min)",
            "CadencePeak10(steps/min)",
            "CadencePeak30(steps/min)",
            "Cadence95th(steps/min)",
            "ENMO(mg)",
            "WearTime(hours)",
        ]
        expected_columns = new_columns if new_schema_first else [
            *old_columns,
            "CadencePeak5(steps/min)",
            "CadencePeak10(steps/min)",
        ]
        assert result.columns.tolist() == expected_columns

        result = result.set_index("Filename")
        assert result.loc["old.csv", "CadencePeak30(steps/min)"] == 71
        assert result.loc["old.csv", "Cadence95th(steps/min)"] == 91
        assert result.loc["old.csv", "ENMO(mg)"] == 12.5
        assert result.loc["old.csv", "WearTime(hours)"] == 23.0
        assert pd.isna(result.loc["old.csv", "CadencePeak5(steps/min)"])
        assert pd.isna(result.loc["old.csv", "CadencePeak10(steps/min)"])
        assert result.loc["new.csv", "CadencePeak5(steps/min)"] == 92
        assert result.loc["new.csv", "CadencePeak10(steps/min)"] == 82

    def test_collate_csvs_strict_accepts_reordered_columns(self, temp_dir):
        first = temp_dir / "first-Daily.csv.gz"
        second = temp_dir / "second-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        pd.DataFrame({"Filename": ["first.csv"], "Steps": [1]}).to_csv(first, index=False)
        pd.DataFrame({"Steps": [2], "Filename": ["second.csv"]}).to_csv(second, index=False)

        collate_mod.collate_csvs([first, second], outfile, schema_policy="strict")

        result = pd.read_csv(outfile)
        assert result.columns.tolist() == ["Filename", "Steps"]
        assert result.to_dict("records") == [
            {"Filename": "first.csv", "Steps": 1},
            {"Filename": "second.csv", "Steps": 2},
        ]

    def test_collate_csvs_strict_accepts_mixed_bom_inputs(self, temp_dir):
        bom_file = temp_dir / "bom-Daily.csv.gz"
        plain_file = temp_dir / "plain-Daily.csv"
        outfile = temp_dir / "Daily.csv.gz"
        with gzip.open(
            bom_file,
            "wt",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            stream.write("Filename,Steps\nbom.csv,1\n")
        plain_file.write_text("Filename,Steps\nplain.csv,2\n", encoding="utf-8")

        collate_mod.collate_csvs(
            [bom_file, plain_file],
            outfile,
            schema_policy="strict",
        )

        result = pd.read_csv(outfile)
        assert result.columns.tolist() == ["Filename", "Steps"]
        assert result.to_dict("records") == [
            {"Filename": "bom.csv", "Steps": 1},
            {"Filename": "plain.csv", "Steps": 2},
        ]

    def test_collate_csvs_strict_rejects_mixed_schemas_without_replacing_output(self, temp_dir):
        first = temp_dir / "first-Daily.csv.gz"
        second = temp_dir / "second-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        pd.DataFrame({"Filename": ["first.csv"], "Steps": [1]}).to_csv(first, index=False)
        pd.DataFrame({"Filename": ["second.csv"], "Steps": [2], "ENMO": [3]}).to_csv(second, index=False)
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)

        with pytest.raises(ValueError, match="strict policy"):
            collate_mod.collate_csvs([first, second], outfile, schema_policy="strict")

        assert pd.read_csv(outfile).to_dict("records") == [{"sentinel": 42}]

    @pytest.mark.parametrize(
        ("header", "message"),
        [
            ("Filename,Steps,Steps\n", "duplicate column names"),
            ("Filename, ,Steps\n", "blank column names"),
        ],
    )
    def test_collate_csvs_rejects_invalid_headers(self, temp_dir, header, message):
        source = temp_dir / "source-Daily.csv.gz"
        with gzip.open(source, "wt", newline="") as stream:
            stream.write(header)
            stream.write("subject.csv,1,2\n")

        with pytest.raises(ValueError, match=message):
            collate_mod.collate_csvs([source], temp_dir / "Daily.csv.gz")

    @pytest.mark.parametrize("contents", ["", "\n"], ids=["empty", "blank-record"])
    def test_collate_csvs_rejects_missing_header_without_replacing_output(
        self,
        temp_dir,
        contents,
    ):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        with gzip.open(source, "wt", newline="") as stream:
            stream.write(contents)
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)

        with pytest.raises(ValueError, match="no header"):
            collate_mod.collate_csvs([source], outfile)

        assert pd.read_csv(outfile).to_dict("records") == [{"sentinel": 42}]

    def test_collate_csvs_skips_blank_data_records(self, temp_dir):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        with gzip.open(source, "wt", newline="") as stream:
            stream.write("Filename,Steps\nfirst.csv,1\n\nsecond.csv,2\n")

        collate_mod.collate_csvs([source], outfile)

        assert pd.read_csv(outfile).to_dict("records") == [
            {"Filename": "first.csv", "Steps": 1},
            {"Filename": "second.csv", "Steps": 2},
        ]

    def test_collate_csvs_preserves_output_write_error(self, temp_dir, monkeypatch):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        pd.DataFrame({"Filename": ["subject.csv"], "Steps": [1]}).to_csv(
            source,
            index=False,
        )
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)

        class FailDataRowWriter:
            def __init__(self):
                self.write_count = 0

            def writerow(self, row):
                self.write_count += 1
                if self.write_count > 1:
                    raise OSError("simulated output failure")

        monkeypatch.setattr(
            collate_mod.csv,
            "writer",
            lambda *args, **kwargs: FailDataRowWriter(),
        )

        with pytest.raises(OSError, match="simulated output failure"):
            collate_mod.collate_csvs([source], outfile)

        assert pd.read_csv(outfile).to_dict("records") == [{"sentinel": 42}]

    def test_collate_csvs_rejects_malformed_rows_without_replacing_output(self, temp_dir):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        with gzip.open(source, "wt", newline="") as stream:
            stream.write("Filename,Steps\nsubject.csv\n")
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)

        with pytest.raises(ValueError, match="row has 1 fields"):
            collate_mod.collate_csvs([source], outfile)

        assert pd.read_csv(outfile).to_dict("records") == [{"sentinel": 42}]
        assert not list(temp_dir.glob(".Daily.csv.gz.*.tmp"))

    def test_collate_csvs_rejects_header_change_after_planning(
        self,
        temp_dir,
        monkeypatch,
    ):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        pd.DataFrame({"Filename": ["source.csv"], "Steps": [1]}).to_csv(
            source,
            index=False,
        )
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)
        original_open_csv = collate_mod._open_csv
        read_count = 0

        def mutate_before_second_read(file, mode, compressed=None):
            nonlocal read_count
            if Path(file) == source and mode == "rt":
                read_count += 1
                if read_count == 2:
                    pd.DataFrame({"Filename": ["source.csv"], "ENMO": [1]}).to_csv(
                        source,
                        index=False,
                    )
            return original_open_csv(file, mode, compressed)

        monkeypatch.setattr(collate_mod, "_open_csv", mutate_before_second_read)

        with pytest.raises(ValueError, match="header changed during collation"):
            collate_mod.collate_csvs([source], outfile)

        assert pd.read_csv(outfile).to_dict("records") == [{"sentinel": 42}]

    def test_collate_csvs_preserves_csv_text_and_quoting(self, temp_dir):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        with gzip.open(source, "wt", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["Filename", "Participant", "Status"])
            writer.writerow(["subject,one.csv", "00123", "NA"])

        collate_mod.collate_csvs([source], outfile)

        with gzip.open(outfile, "rt", newline="") as stream:
            assert list(csv.reader(stream)) == [
                ["Filename", "Participant", "Status"],
                ["subject,one.csv", "00123", "NA"],
            ]

    def test_collate_csvs_rejects_unknown_schema_policy(self, temp_dir):
        with pytest.raises(ValueError, match="Unknown schema policy"):
            collate_mod.collate_csvs(
                [],
                temp_dir / "Daily.csv.gz",
                schema_policy="intersection",
            )

    def test_collate_csvs_truncated_input_does_not_replace_output(self, temp_dir):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        pd.DataFrame({
            "Filename": ["subject.csv"] * 100,
            "Steps": list(range(100)),
        }).to_csv(source, index=False)
        source.write_bytes(source.read_bytes()[:-8])
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)

        with pytest.raises(ValueError, match="Could not read CSV file") as error:
            collate_mod.collate_csvs([source], outfile)

        assert str(source) in str(error.value)
        assert isinstance(error.value.__cause__, (EOFError, gzip.BadGzipFile))
        assert pd.read_csv(outfile).to_dict("records") == [{"sentinel": 42}]
        assert not list(temp_dir.glob(".Daily.csv.gz.*.tmp"))

    def test_collate_csvs_reports_source_for_csv_parse_errors(self, temp_dir):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        with gzip.open(source, "wt", newline="") as stream:
            stream.write('Filename,Steps\n"subject.csv,1\n')
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)

        with pytest.raises(ValueError, match="Could not read CSV file") as error:
            collate_mod.collate_csvs([source], outfile)

        assert str(source) in str(error.value)
        assert isinstance(error.value.__cause__, csv.Error)
        assert pd.read_csv(outfile).to_dict("records") == [{"sentinel": 42}]

    def test_collate_csvs_uses_umask_permissions_for_new_output(self, temp_dir):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        control = temp_dir / "control.csv.gz"
        pd.DataFrame({"Filename": ["subject.csv"]}).to_csv(source, index=False)
        control.touch()

        collate_mod.collate_csvs([source], outfile)

        assert stat.S_IMODE(outfile.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)

    def test_collate_csvs_preserves_existing_output_permissions(self, temp_dir):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        pd.DataFrame({"Filename": ["subject.csv"]}).to_csv(source, index=False)
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)
        outfile.chmod(0o640)

        collate_mod.collate_csvs([source], outfile)

        assert stat.S_IMODE(outfile.stat().st_mode) == 0o640

    def test_collate_csvs_replaces_read_only_output(self, temp_dir):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        pd.DataFrame({"Filename": ["source.csv"], "Steps": [1]}).to_csv(
            source,
            index=False,
        )
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)
        outfile.chmod(0o444)

        collate_mod.collate_csvs([source], outfile)

        assert pd.read_csv(outfile).to_dict("records") == [
            {"Filename": "source.csv", "Steps": 1},
        ]
        assert stat.S_IMODE(outfile.stat().st_mode) == 0o444

    def test_collate_jsons_replaces_read_only_output(
        self,
        temp_dir,
        mock_info_json,
    ):
        source = temp_dir / "source-Info.json"
        outfile = temp_dir / "Info.csv.gz"
        source.write_text(json.dumps(mock_info_json))
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)
        outfile.chmod(0o444)

        collate_mod.collate_jsons([source], outfile)

        assert pd.read_csv(outfile).to_dict("records") == [mock_info_json]
        assert stat.S_IMODE(outfile.stat().st_mode) == 0o444

    def test_collate_csvs_empty_input_removes_stale_output(self, temp_dir):
        outfile = temp_dir / "Daily.csv.gz"
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)

        collate_mod.collate_csvs([], outfile)

        assert not outfile.exists()

    def test_collate_csvs_empty_input_waits_for_output_lock(self, temp_dir):
        outfile = temp_dir / "Daily.csv.gz"
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)
        finished = threading.Event()

        def remove_output():
            collate_mod.collate_csvs([], outfile)
            finished.set()

        with collate_mod._output_lock(temp_dir):
            remover = threading.Thread(target=remove_output)
            remover.start()
            assert not finished.wait(timeout=0.1)
            assert outfile.exists()

        assert finished.wait(timeout=1)
        remover.join(timeout=1)
        assert not remover.is_alive()
        assert not outfile.exists()


class TestCollateOutputs:
    """Tests for main collate_outputs function."""

    def test_collate_outputs_full(self, temp_dir, mock_info_json):
        """Test full output collation."""
        # Create directory structure with all file types
        results_dir = temp_dir / "results"

        for i in range(2):
            subject_dir = results_dir / f"subject_{i}"
            subject_dir.mkdir(parents=True)

            # Info.json
            info = mock_info_json.copy()
            info['Filename'] = f"subject_{i}.csv"
            with open(subject_dir / f"subject_{i}-Info.json", 'w') as f:
                json.dump(info, f)

            # Daily.csv.gz
            pd.DataFrame({
                'Filename': [f'subject_{i}.csv'],
                'Date': ['2024-01-15'],
                'Steps': [8000]
            }).to_csv(subject_dir / f"subject_{i}-Daily.csv.gz", index=False)

            # Hourly.csv.gz
            pd.DataFrame({
                'Filename': [f'subject_{i}.csv'],
                'Time': ['2024-01-15 00:00:00'],
                'Steps': [100]
            }).to_csv(subject_dir / f"subject_{i}-Hourly.csv.gz", index=False)

        # Run collation
        output_dir = temp_dir / "collated"
        collate_mod.collate_outputs(
            str(results_dir),
            str(output_dir),
            included=['daily', 'hourly']
        )

        # Verify outputs
        assert (output_dir / "Info.csv.gz").exists()
        assert (output_dir / "Daily.csv.gz").exists()
        assert (output_dir / "Hourly.csv.gz").exists()

    def test_collate_outputs_subset(self, temp_dir, mock_info_json):
        """Test collation with subset of file types."""
        results_dir = temp_dir / "results"
        subject_dir = results_dir / "subject_0"
        subject_dir.mkdir(parents=True)

        # Create files
        with open(subject_dir / "subject_0-Info.json", 'w') as f:
            json.dump(mock_info_json, f)

        pd.DataFrame({
            'Date': ['2024-01-15'],
            'Steps': [8000]
        }).to_csv(subject_dir / "subject_0-Daily.csv.gz", index=False)

        pd.DataFrame({
            'StartTime': ['2024-01-15 08:00:00'],
            'Duration(mins)': [30]
        }).to_csv(subject_dir / "subject_0-Bouts.csv.gz", index=False)

        # Collate only daily
        output_dir = temp_dir / "collated"
        collate_mod.collate_outputs(
            str(results_dir),
            str(output_dir),
            included=['daily']  # Only daily, not bouts
        )

        assert (output_dir / "Daily.csv.gz").exists()

    def test_collate_outputs_reports_written_removed_and_absent_outputs(
        self,
        temp_dir,
        capsys,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        pd.DataFrame({"Filename": ["subject.csv"], "Steps": [1]}).to_csv(
            results_dir / "subject-Daily.csv.gz",
            index=False,
        )
        stale_adjusted = output_dir / "DailyAdjusted.csv.gz"
        pd.DataFrame({"sentinel": [42]}).to_csv(stale_adjusted, index=False)

        collate_mod.collate_outputs(
            results_dir,
            output_dir,
            included=["daily", "hourly"],
        )

        output = capsys.readouterr().out
        daily = output_dir / "Daily.csv.gz"
        hourly = output_dir / "Hourly.csv.gz"
        assert daily.exists()
        assert not stale_adjusted.exists()
        assert not hourly.exists()
        assert f"Collated Daily CSV written to {daily}" in output
        assert (
            f"No DailyAdjusted inputs found; removed existing CSV at {stale_adjusted}"
            in output
        )
        assert f"No Hourly inputs found; no CSV written at {hourly}" in output
        assert "Collated DailyAdjusted CSV written" not in output
        assert "Collated Hourly CSV written" not in output

    def test_collate_outputs_removes_info_when_inputs_are_missing(
        self,
        temp_dir,
        capsys,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        pd.DataFrame({"Filename": ["subject.csv"], "Steps": [1]}).to_csv(
            results_dir / "subject-Daily.csv.gz",
            index=False,
        )
        info_outfile = output_dir / "Info.csv.gz"
        pd.DataFrame({"sentinel": [42]}).to_csv(info_outfile, index=False)

        collate_mod.collate_outputs(results_dir, output_dir, included=["daily"])

        assert not info_outfile.exists()
        assert (
            f"No Info inputs found; removed existing CSV at {info_outfile}"
            in capsys.readouterr().out
        )

    def test_collate_outputs_sorts_sources_and_excludes_output_tree(self, temp_dir):
        results_dir = temp_dir / "results"
        output_dir = results_dir / "collated"
        output_dir.mkdir(parents=True)

        for subject in ["subject_b", "subject_a"]:
            subject_dir = results_dir / subject
            subject_dir.mkdir()
            pd.DataFrame({
                "Filename": [f"{subject}.csv"],
                "Steps": [1],
            }).to_csv(subject_dir / f"{subject}-Daily.csv.gz", index=False)

        stale_dir = output_dir / "stale"
        stale_dir.mkdir()
        pd.DataFrame({
            "Filename": ["stale.csv"],
            "Steps": [999],
        }).to_csv(stale_dir / "stale-Daily.csv.gz", index=False)

        collate_mod.collate_outputs(
            results_dir,
            output_dir,
            included=["daily"],
        )

        result = pd.read_csv(output_dir / "Daily.csv.gz")
        assert result["Filename"].tolist() == ["subject_a.csv", "subject_b.csv"]

    def test_discovery_stats_only_matching_candidates(self, temp_dir, monkeypatch):
        results_dir = temp_dir / "results"
        results_dir.mkdir()
        for index in range(20):
            (results_dir / f"unrelated-{index}.txt").write_text("ignored")
        matching_file = results_dir / "subject-Daily.csv.gz"
        matching_file.write_bytes(b"candidate")

        checked_paths = []
        original_is_regular_output_file = collate_mod._is_regular_output_file

        def reject_unrelated_metadata(path):
            if path.suffix == ".txt":
                raise AssertionError(f"unrelated file was inspected: {path}")
            checked_paths.append(path)
            return original_is_regular_output_file(path)

        monkeypatch.setattr(
            collate_mod,
            "_is_regular_output_file",
            reject_unrelated_metadata,
        )

        _, csv_files = collate_mod._discover_output_files(
            results_dir,
            temp_dir / "collated",
            ["Daily"],
        )

        assert matching_file.resolve() in checked_paths
        assert csv_files == {"Daily": [matching_file.resolve()]}

    def test_discovery_propagates_matching_file_stat_error(self, temp_dir, monkeypatch):
        results_dir = temp_dir / "results"
        results_dir.mkdir()
        matching_file = results_dir / "subject-Daily.csv.gz"
        matching_file.write_bytes(b"candidate")
        original_stat = Path.stat

        def fail_matching_stat(path, *args, **kwargs):
            if path.name == matching_file.name:
                raise PermissionError("candidate stat denied")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", fail_matching_stat)

        with pytest.raises(PermissionError, match="candidate stat denied"):
            collate_mod._discover_output_files(
                results_dir,
                temp_dir / "collated",
                ["Daily"],
            )

    def test_discovery_skips_dangling_matching_symlink(self, temp_dir):
        results_dir = temp_dir / "results"
        results_dir.mkdir()
        (results_dir / "missing-Daily.csv.gz").symlink_to(
            results_dir / "missing-target.csv.gz"
        )

        _, csv_files = collate_mod._discover_output_files(
            results_dir,
            temp_dir / "collated",
            ["Daily"],
        )

        assert csv_files == {"Daily": []}

    def test_discovery_propagates_walk_error(self, temp_dir, monkeypatch):
        results_dir = temp_dir / "results"
        results_dir.mkdir()

        def fail_walk(root, onerror=None):
            assert onerror is not None
            onerror(PermissionError("directory scan denied"))
            return []

        monkeypatch.setattr(collate_mod.os, "walk", fail_walk)

        with pytest.raises(PermissionError, match="directory scan denied"):
            collate_mod._discover_output_files(
                results_dir,
                temp_dir / "collated",
                ["Daily"],
            )

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires FIFO support")
    def test_discovery_skips_matching_special_files(self, temp_dir):
        results_dir = temp_dir / "results"
        results_dir.mkdir()
        os.mkfifo(results_dir / "blocked-Daily.csv.gz")

        _, csv_files = collate_mod._discover_output_files(
            results_dir,
            temp_dir / "collated",
            ["Daily"],
        )

        assert csv_files == {"Daily": []}

    def test_collate_outputs_supports_output_equal_to_results(self, temp_dir):
        results_dir = temp_dir / "results"
        results_dir.mkdir()
        pd.DataFrame({
            "Filename": ["subject.csv"],
            "Steps": [1],
        }).to_csv(results_dir / "subject-Daily.csv.gz", index=False)

        collate_mod.collate_outputs(
            results_dir,
            results_dir,
            included=["daily"],
        )

        result = pd.read_csv(results_dir / "Daily.csv.gz")
        assert result.to_dict("records") == [{"Filename": "subject.csv", "Steps": 1}]

    def test_collate_outputs_supports_output_ancestor_of_results(self, temp_dir):
        output_dir = temp_dir / "workspace"
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True)
        pd.DataFrame({
            "Filename": ["subject.csv"],
            "Steps": [1],
        }).to_csv(results_dir / "subject-Daily.csv.gz", index=False)
        pd.DataFrame({"sentinel": [42]}).to_csv(output_dir / "Daily.csv.gz", index=False)

        collate_mod.collate_outputs(
            results_dir,
            output_dir,
            included=["daily"],
        )

        result = pd.read_csv(output_dir / "Daily.csv.gz")
        assert result.to_dict("records") == [{"Filename": "subject.csv", "Steps": 1}]

    def test_collate_outputs_strict_policy_preserves_all_outputs_on_mismatch(
        self,
        temp_dir,
        mock_info_json,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        output_dir.mkdir()
        for subject, columns in [
            ("subject_a", {"Filename": ["a.csv"], "Steps": [1]}),
            ("subject_b", {"Filename": ["b.csv"], "Steps": [2], "ENMO": [3]}),
        ]:
            subject_dir = results_dir / subject
            subject_dir.mkdir(parents=True)
            with open(subject_dir / f"{subject}-Info.json", "w") as stream:
                json.dump({**mock_info_json, "Filename": f"{subject}.csv"}, stream)
            pd.DataFrame(columns).to_csv(
                subject_dir / f"{subject}-Daily.csv.gz",
                index=False,
            )

        pd.DataFrame({"sentinel": ["info"]}).to_csv(output_dir / "Info.csv.gz", index=False)
        pd.DataFrame({"sentinel": ["daily"]}).to_csv(output_dir / "Daily.csv.gz", index=False)

        with pytest.raises(ValueError, match="strict policy"):
            collate_mod.collate_outputs(
                results_dir,
                output_dir,
                included=["daily"],
                schema_policy="strict",
            )

        assert pd.read_csv(output_dir / "Info.csv.gz").to_dict("records") == [
            {"sentinel": "info"},
        ]
        assert pd.read_csv(output_dir / "Daily.csv.gz").to_dict("records") == [
            {"sentinel": "daily"},
        ]

    def test_collate_outputs_stages_all_rows_before_publication(self, temp_dir):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        pd.DataFrame({"Filename": ["subject.csv"], "Steps": [1]}).to_csv(
            results_dir / "subject-Daily.csv.gz",
            index=False,
        )
        with gzip.open(results_dir / "subject-Hourly.csv.gz", "wt", newline="") as stream:
            stream.write("Filename,Steps\nsubject.csv\n")
        pd.DataFrame({"sentinel": ["info"]}).to_csv(output_dir / "Info.csv.gz", index=False)
        pd.DataFrame({"sentinel": ["daily"]}).to_csv(output_dir / "Daily.csv.gz", index=False)
        pd.DataFrame({"sentinel": ["hourly"]}).to_csv(output_dir / "Hourly.csv.gz", index=False)

        with pytest.raises(ValueError, match="row has 1 fields"):
            collate_mod.collate_outputs(
                results_dir,
                output_dir,
                included=["daily", "hourly"],
            )

        for key in ["Info", "Daily", "Hourly"]:
            assert pd.read_csv(output_dir / f"{key}.csv.gz").to_dict("records") == [
                {"sentinel": key.lower()},
            ]

    def test_collate_outputs_rolls_back_publication_failure(
        self,
        temp_dir,
        mock_info_json,
        monkeypatch,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        with open(results_dir / "subject-Info.json", "w") as stream:
            json.dump(mock_info_json, stream)
        pd.DataFrame({"Filename": ["subject.csv"], "Steps": [1]}).to_csv(
            results_dir / "subject-Daily.csv.gz",
            index=False,
        )
        pd.DataFrame({"sentinel": ["info"]}).to_csv(output_dir / "Info.csv.gz", index=False)
        pd.DataFrame({"sentinel": ["daily"]}).to_csv(output_dir / "Daily.csv.gz", index=False)

        original_replace = os.replace
        failed = False

        def fail_daily_once(source, destination):
            nonlocal failed
            if Path(destination).name == "Daily.csv.gz" and not failed:
                failed = True
                raise OSError("simulated publication failure")
            original_replace(source, destination)

        monkeypatch.setattr(collate_mod.os, "replace", fail_daily_once)

        with pytest.raises(OSError, match="simulated publication failure"):
            collate_mod.collate_outputs(results_dir, output_dir, included=["daily"])

        assert pd.read_csv(output_dir / "Info.csv.gz").to_dict("records") == [
            {"sentinel": "info"},
        ]
        assert pd.read_csv(output_dir / "Daily.csv.gz").to_dict("records") == [
            {"sentinel": "daily"},
        ]
        assert not list(output_dir.glob(".*.bak"))
        assert not list(output_dir.glob(".*.tmp"))

    def test_collate_outputs_rolls_back_mutation_that_raises_afterward(
        self,
        temp_dir,
        mock_info_json,
        monkeypatch,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        (results_dir / "subject-Info.json").write_text(json.dumps(mock_info_json))
        pd.DataFrame({"Filename": ["subject.csv"], "Steps": [1]}).to_csv(
            results_dir / "subject-Daily.csv.gz",
            index=False,
        )
        pd.DataFrame({"sentinel": ["info"]}).to_csv(
            output_dir / "Info.csv.gz",
            index=False,
        )
        pd.DataFrame({"sentinel": ["daily"]}).to_csv(
            output_dir / "Daily.csv.gz",
            index=False,
        )
        original_replace = os.replace
        interrupted = False

        def replace_daily_then_interrupt(source, destination):
            nonlocal interrupted
            original_replace(source, destination)
            if Path(destination).name == "Daily.csv.gz" and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("simulated post-mutation interrupt")

        monkeypatch.setattr(collate_mod.os, "replace", replace_daily_then_interrupt)

        with pytest.raises(KeyboardInterrupt, match="post-mutation interrupt"):
            collate_mod.collate_outputs(results_dir, output_dir, included=["daily"])

        assert pd.read_csv(output_dir / "Info.csv.gz").to_dict("records") == [
            {"sentinel": "info"},
        ]
        assert pd.read_csv(output_dir / "Daily.csv.gz").to_dict("records") == [
            {"sentinel": "daily"},
        ]

    def test_output_lock_serializes_publishers(self, temp_dir):
        started = threading.Event()
        acquired = threading.Event()

        def contend_for_lock():
            started.set()
            with collate_mod._output_lock(temp_dir):
                acquired.set()

        with collate_mod._output_lock(temp_dir):
            contender = threading.Thread(target=contend_for_lock)
            contender.start()
            assert started.wait(timeout=1)
            assert not acquired.wait(timeout=0.1)

        assert acquired.wait(timeout=1)
        contender.join(timeout=1)
        assert not contender.is_alive()

    def test_output_lock_rejects_symlink_without_modifying_target(self, temp_dir):
        target = temp_dir / "target.txt"
        target.write_bytes(b"")
        (temp_dir / ".stepcount-collate.lock").symlink_to(target.name)

        with pytest.raises((OSError, RuntimeError)):
            with collate_mod._output_lock(temp_dir):
                pass

        assert target.read_bytes() == b""

    def test_snapshot_recovers_interrupted_publication(self, temp_dir):
        info_outfile = temp_dir / "Info.csv.gz"
        daily_outfile = temp_dir / "Daily.csv.gz"
        pd.DataFrame({"generation": ["old-info"]}).to_csv(info_outfile, index=False)
        pd.DataFrame({"generation": ["old-daily"]}).to_csv(daily_outfile, index=False)
        transaction_dir = temp_dir / ".stepcount-publish-crash.tmp"
        transaction_dir.mkdir()
        records = collate_mod._prepare_publication_records(
            [collate_mod._deletion(info_outfile), collate_mod._deletion(daily_outfile)],
            transaction_dir,
        )
        collate_mod._write_publication_journal(records, transaction_dir)
        collate_mod._backup_publication(records)
        pd.DataFrame({"generation": ["new-info"]}).to_csv(info_outfile, index=False)

        with collate_mod.collated_outputs_snapshot(temp_dir) as snapshot:
            assert pd.read_csv(snapshot / "Info.csv.gz").iloc[0, 0] == "old-info"
            assert pd.read_csv(snapshot / "Daily.csv.gz").iloc[0, 0] == "old-daily"

        assert not transaction_dir.exists()

    def test_snapshot_preserves_committed_publication_during_cleanup(self, temp_dir):
        info_outfile = temp_dir / "Info.csv.gz"
        pd.DataFrame({"generation": ["old"]}).to_csv(info_outfile, index=False)
        transaction_dir = temp_dir / ".stepcount-publish-crash.tmp"
        transaction_dir.mkdir()
        records = collate_mod._prepare_publication_records(
            [collate_mod._deletion(info_outfile)],
            transaction_dir,
        )
        collate_mod._write_publication_journal(records, transaction_dir)
        collate_mod._backup_publication(records)
        pd.DataFrame({"generation": ["new"]}).to_csv(info_outfile, index=False)
        collate_mod._create_publication_marker(
            transaction_dir,
            collate_mod._PUBLICATION_COMMITTED,
        )

        with collate_mod.collated_outputs_snapshot(temp_dir) as snapshot:
            assert pd.read_csv(snapshot / "Info.csv.gz").iloc[0, 0] == "new"

        assert not transaction_dir.exists()

    def test_committed_marker_survives_cleanup_failure(self, temp_dir, monkeypatch):
        info_outfile = temp_dir / "Info.csv.gz"
        pd.DataFrame({"generation": ["old"]}).to_csv(info_outfile, index=False)
        transaction_dir = temp_dir / ".stepcount-publish-crash.tmp"
        transaction_dir.mkdir()
        records = collate_mod._prepare_publication_records(
            [collate_mod._deletion(info_outfile)],
            transaction_dir,
        )
        collate_mod._write_publication_journal(records, transaction_dir)
        collate_mod._backup_publication(records)
        pd.DataFrame({"generation": ["new"]}).to_csv(info_outfile, index=False)
        collate_mod._create_publication_marker(
            transaction_dir,
            collate_mod._PUBLICATION_COMMITTED,
        )
        backup = records[0].backup
        assert backup is not None
        original_unlink = Path.unlink

        def fail_backup_cleanup(path, *args, **kwargs):
            if path == backup:
                raise OSError("simulated cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)
        collate_mod._cleanup_publication(records, transaction_dir)

        assert (transaction_dir / collate_mod._PUBLICATION_COMMITTED).exists()
        monkeypatch.setattr(Path, "unlink", original_unlink)
        with collate_mod.collated_outputs_snapshot(temp_dir) as snapshot:
            assert pd.read_csv(snapshot / "Info.csv.gz").iloc[0, 0] == "new"
        assert not transaction_dir.exists()

    def test_snapshot_blocks_multi_file_publication(
        self,
        temp_dir,
        mock_info_json,
        monkeypatch,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        (results_dir / "subject-Info.json").write_text(json.dumps(mock_info_json))
        pd.DataFrame({"Filename": ["new.csv"], "Steps": [2]}).to_csv(
            results_dir / "subject-Daily.csv.gz",
            index=False,
        )
        pd.DataFrame({"Filename": ["old.csv"]}).to_csv(
            output_dir / "Info.csv.gz",
            index=False,
        )
        pd.DataFrame({"Filename": ["old.csv"], "Steps": [1]}).to_csv(
            output_dir / "Daily.csv.gz",
            index=False,
        )
        discovery_started = threading.Event()
        errors = []
        original_discover = collate_mod._discover_output_files

        def track_discovery(*args, **kwargs):
            discovery_started.set()
            return original_discover(*args, **kwargs)

        def publish():
            try:
                collate_mod.collate_outputs(results_dir, output_dir, included=["daily"])
            except BaseException as error:
                errors.append(error)

        monkeypatch.setattr(collate_mod, "_discover_output_files", track_discovery)
        with collate_mod.collated_outputs_snapshot(output_dir) as snapshot:
            publisher = threading.Thread(target=publish)
            publisher.start()
            assert not discovery_started.wait(timeout=0.1)
            assert pd.read_csv(snapshot / "Info.csv.gz")["Filename"].tolist() == [
                "old.csv",
            ]
            assert pd.read_csv(snapshot / "Daily.csv.gz")["Filename"].tolist() == [
                "old.csv",
            ]

        publisher.join(timeout=2)
        assert not publisher.is_alive()
        assert discovery_started.is_set()
        assert errors == []
        with collate_mod.collated_outputs_snapshot(output_dir) as snapshot:
            assert pd.read_csv(snapshot / "Info.csv.gz")["Filename"].tolist() == [
                mock_info_json["Filename"],
            ]
            assert pd.read_csv(snapshot / "Daily.csv.gz")["Filename"].tolist() == [
                "new.csv",
            ]

    def test_rollback_cleanup_preserves_replaced_destination(self, temp_dir):
        outfile = temp_dir / "Info.csv.gz"
        outfile.write_bytes(b"replacement")
        current_stat = outfile.stat()
        different_identity = (current_stat.st_dev, current_stat.st_ino + 1)

        with pytest.raises(RuntimeError, match="Refusing to remove"):
            collate_mod._remove_if_identity_matches(outfile, different_identity)

        assert outfile.read_bytes() == b"replacement"

    def test_publication_rejects_directory_destination(self, temp_dir):
        outfile = temp_dir / "Info.csv.gz"
        outfile.mkdir()

        with pytest.raises(OSError, match="not a regular file or symlink"):
            collate_mod._publish_outputs_locked([collate_mod._deletion(outfile)])

        assert outfile.is_dir()
        assert not list(temp_dir.glob(".stepcount-publish-*.tmp"))

    def test_collate_outputs_locks_before_discovery(
        self,
        temp_dir,
        mock_info_json,
        monkeypatch,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        (results_dir / "subject-Info.json").write_text(json.dumps(mock_info_json))
        first_discovery_started = threading.Event()
        release_first_discovery = threading.Event()
        second_discovery_started = threading.Event()
        call_count = 0
        call_count_lock = threading.Lock()
        errors = []
        original_discover = collate_mod._discover_output_files

        def pause_first_discovery(*args, **kwargs):
            nonlocal call_count
            with call_count_lock:
                call_count += 1
                current_call = call_count
            if current_call == 1:
                first_discovery_started.set()
                assert release_first_discovery.wait(timeout=2)
            else:
                second_discovery_started.set()
            return original_discover(*args, **kwargs)

        def run_collation():
            try:
                collate_mod.collate_outputs(results_dir, output_dir, included=[])
            except BaseException as error:
                errors.append(error)

        monkeypatch.setattr(
            collate_mod,
            "_discover_output_files",
            pause_first_discovery,
        )
        first = threading.Thread(target=run_collation)
        second = threading.Thread(target=run_collation)
        first.start()
        assert first_discovery_started.wait(timeout=1)
        second.start()
        assert not second_discovery_started.wait(timeout=0.1)
        release_first_discovery.set()

        first.join(timeout=2)
        second.join(timeout=2)
        assert not first.is_alive()
        assert not second.is_alive()
        assert second_discovery_started.is_set()
        assert errors == []

    def test_cleanup_failure_does_not_mask_staging_error(
        self,
        temp_dir,
        monkeypatch,
        caplog,
    ):
        source = temp_dir / "source-Daily.csv.gz"
        outfile = temp_dir / "Daily.csv.gz"
        with gzip.open(source, "wt", newline="") as stream:
            stream.write("Filename,Steps\nsubject.csv\n")
        original_rmdir = Path.rmdir

        def fail_staging_rmdir(path):
            if path.name.endswith(".tmp"):
                raise OSError("simulated cleanup failure")
            original_rmdir(path)

        monkeypatch.setattr(Path, "rmdir", fail_staging_rmdir)

        with pytest.raises(ValueError, match="row has 1 fields"):
            collate_mod.collate_csvs([source], outfile)

        assert "Could not remove staging directory" in caplog.text

    def test_collate_outputs_missing_root_preserves_existing_output(self, temp_dir):
        output_dir = temp_dir / "collated"
        output_dir.mkdir()
        outfile = output_dir / "Info.csv.gz"
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            collate_mod.collate_outputs(temp_dir / "missing", output_dir)

        assert pd.read_csv(outfile).to_dict("records") == [{"sentinel": 42}]

    def test_collate_outputs_rejects_non_directory_results_path(self, temp_dir):
        results_file = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_file.write_text("not a directory")

        with pytest.raises(NotADirectoryError, match="not a directory"):
            collate_mod.collate_outputs(results_file, output_dir)

        assert not output_dir.exists()

    def test_collate_outputs_empty_root_preserves_existing_output(self, temp_dir):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        outfile = output_dir / "Info.csv.gz"
        pd.DataFrame({"sentinel": [42]}).to_csv(outfile, index=False)

        with pytest.raises(FileNotFoundError, match="No result files found"):
            collate_mod.collate_outputs(results_dir, output_dir)

        assert pd.read_csv(outfile).to_dict("records") == [{"sentinel": 42}]

    def test_collate_outputs_does_not_use_hardlinks(
        self,
        temp_dir,
        mock_info_json,
        monkeypatch,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        with open(results_dir / "subject-Info.json", "w") as stream:
            json.dump(mock_info_json, stream)
        pd.DataFrame({"Filename": ["subject.csv"], "Steps": [1]}).to_csv(
            results_dir / "subject-Daily.csv.gz",
            index=False,
        )
        pd.DataFrame({"sentinel": ["info"]}).to_csv(output_dir / "Info.csv.gz", index=False)
        pd.DataFrame({"sentinel": ["daily"]}).to_csv(output_dir / "Daily.csv.gz", index=False)

        def reject_hardlink(source, destination):
            raise AssertionError("publication should not use hard links")

        monkeypatch.setattr(collate_mod.os, "link", reject_hardlink)

        collate_mod.collate_outputs(results_dir, output_dir, included=["daily"])

        assert pd.read_csv(output_dir / "Daily.csv.gz").to_dict("records") == [
            {"Filename": "subject.csv", "Steps": 1},
        ]
        assert not list(output_dir.glob(".*.bak"))

    def test_symlink_output_is_rejected_without_modifying_target(
        self,
        temp_dir,
        mock_info_json,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        (results_dir / "subject-Info.json").write_text(json.dumps(mock_info_json))
        pd.DataFrame({"Filename": ["subject.csv"], "Steps": [1]}).to_csv(
            results_dir / "subject-Daily.csv.gz",
            index=False,
        )
        target = output_dir / "prior-info.csv.gz"
        pd.DataFrame({"sentinel": [42]}).to_csv(target, index=False)
        info_outfile = output_dir / "Info.csv.gz"
        info_outfile.symlink_to(target.name)
        with pytest.raises(OSError, match="must not be a symlink"):
            collate_mod.collate_outputs(results_dir, output_dir, included=["daily"])

        assert info_outfile.is_symlink()
        assert os.readlink(info_outfile) == target.name
        assert pd.read_csv(info_outfile).to_dict("records") == [{"sentinel": 42}]

    def test_rollback_failure_preserves_only_unrestored_backup(
        self,
        temp_dir,
        mock_info_json,
        monkeypatch,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        with open(results_dir / "subject-Info.json", "w") as stream:
            json.dump(mock_info_json, stream)
        pd.DataFrame({"Filename": ["subject.csv"], "Steps": [1]}).to_csv(
            results_dir / "subject-Daily.csv.gz",
            index=False,
        )
        pd.DataFrame({"sentinel": ["info"]}).to_csv(output_dir / "Info.csv.gz", index=False)
        pd.DataFrame({"sentinel": ["daily"]}).to_csv(output_dir / "Daily.csv.gz", index=False)

        original_replace = os.replace
        publication_failed = False

        def fail_publication_and_rollback(source, destination):
            nonlocal publication_failed
            source = Path(source)
            destination = Path(destination)
            if destination.name == "Daily.csv.gz" and not publication_failed:
                publication_failed = True
                raise OSError("simulated publication failure")
            if (
                destination.name == "Info.csv.gz"
                and source.parent.name.startswith(".stepcount-publish-")
            ):
                raise OSError("simulated rollback failure")
            original_replace(source, destination)

        monkeypatch.setattr(collate_mod.os, "replace", fail_publication_and_rollback)

        with pytest.raises(RuntimeError, match="backups preserved") as error:
            collate_mod.collate_outputs(results_dir, output_dir, included=["daily"])

        backups = list(output_dir.glob(".stepcount-publish-*.tmp/*-Info.csv.gz"))
        assert len(backups) == 1
        assert backups[0].name.endswith("Info.csv.gz")
        assert str(backups[0]) in str(error.value)
        assert pd.read_csv(output_dir / "Daily.csv.gz").to_dict("records") == [
            {"sentinel": "daily"},
        ]

        monkeypatch.setattr(collate_mod.os, "replace", original_replace)
        with collate_mod.collated_outputs_snapshot(output_dir):
            assert pd.read_csv(output_dir / "Info.csv.gz").to_dict("records") == [
                {"sentinel": "info"},
            ]
            assert pd.read_csv(output_dir / "Daily.csv.gz").to_dict("records") == [
                {"sentinel": "daily"},
            ]
        assert not list(output_dir.glob(".stepcount-publish-*.tmp"))

    def test_rollback_interruption_preserves_unrestored_backup(
        self,
        temp_dir,
        mock_info_json,
        monkeypatch,
    ):
        results_dir = temp_dir / "results"
        output_dir = temp_dir / "collated"
        results_dir.mkdir()
        output_dir.mkdir()
        (results_dir / "subject-Info.json").write_text(json.dumps(mock_info_json))
        pd.DataFrame({"Filename": ["subject.csv"], "Steps": [1]}).to_csv(
            results_dir / "subject-Daily.csv.gz",
            index=False,
        )
        pd.DataFrame({"sentinel": ["info"]}).to_csv(
            output_dir / "Info.csv.gz",
            index=False,
        )
        pd.DataFrame({"sentinel": ["daily"]}).to_csv(
            output_dir / "Daily.csv.gz",
            index=False,
        )
        original_replace = os.replace

        def fail_publication_and_interrupt_rollback(source, destination):
            source = Path(source)
            destination = Path(destination)
            if (
                destination.name == "Daily.csv.gz"
                and not source.parent.name.startswith(".stepcount-publish-")
            ):
                raise OSError("simulated publication failure")
            if (
                destination.name == "Info.csv.gz"
                and source.parent.name.startswith(".stepcount-publish-")
            ):
                raise KeyboardInterrupt("simulated rollback interruption")
            original_replace(source, destination)

        monkeypatch.setattr(
            collate_mod.os,
            "replace",
            fail_publication_and_interrupt_rollback,
        )

        with pytest.raises(RuntimeError, match="backups preserved") as error:
            collate_mod.collate_outputs(results_dir, output_dir, included=["daily"])

        backups = list(output_dir.glob(".stepcount-publish-*.tmp/*-Info.csv.gz"))
        assert len(backups) == 1
        assert str(backups[0]) in str(error.value)
        assert pd.read_csv(backups[0]).to_dict("records") == [
            {"sentinel": "info"},
        ]
        assert not (output_dir / "Info.csv.gz").exists()
        assert pd.read_csv(output_dir / "Daily.csv.gz").to_dict("records") == [
            {"sentinel": "daily"},
        ]

        monkeypatch.setattr(collate_mod.os, "replace", original_replace)
        with collate_mod.collated_outputs_snapshot(output_dir):
            assert pd.read_csv(output_dir / "Info.csv.gz").to_dict("records") == [
                {"sentinel": "info"},
            ]
            assert pd.read_csv(output_dir / "Daily.csv.gz").to_dict("records") == [
                {"sentinel": "daily"},
            ]
        assert not list(output_dir.glob(".stepcount-publish-*.tmp"))

    def test_collate_outputs_rejects_unknown_policy_before_mutation(self, temp_dir):
        output_dir = temp_dir / "collated"
        output_dir.mkdir()
        pd.DataFrame({"sentinel": [42]}).to_csv(output_dir / "Info.csv.gz", index=False)

        with pytest.raises(ValueError, match="Unknown schema policy"):
            collate_mod.collate_outputs(
                temp_dir / "missing-results",
                output_dir,
                included=[],
                schema_policy="intersection",
            )

        assert pd.read_csv(output_dir / "Info.csv.gz").to_dict("records") == [
            {"sentinel": 42},
        ]

    def test_main_forwards_schema_policy(self, monkeypatch):
        arguments = {}

        def capture_arguments(**kwargs):
            arguments.update(kwargs)

        monkeypatch.setattr(collate_mod, "collate_outputs", capture_arguments)
        monkeypatch.setattr(sys, "argv", [
            "stepcount-collate-outputs",
            "results",
            "--output",
            "collated",
            "--include",
            "daily",
            "--schema-policy",
            "strict",
        ])

        collate_mod.main()

        assert arguments == {
            "results_dir": "results",
            "collated_results_dir": "collated",
            "included": ["daily"],
            "schema_policy": "strict",
        }


class TestConvertOrdereddict:
    """Tests for OrderedDict conversion utility."""

    def test_convert_ordereddict(self):
        """Test OrderedDict to dict conversion."""
        od = OrderedDict([('a', 1), ('b', 2)])

        result = collate_mod.convert_ordereddict(od)

        assert isinstance(result, dict)
        assert result == {'a': 1, 'b': 2}

    def test_convert_regular_value(self):
        """Test non-OrderedDict values pass through."""
        assert collate_mod.convert_ordereddict(42) == 42
        assert collate_mod.convert_ordereddict("text") == "text"
        assert collate_mod.convert_ordereddict([1, 2, 3]) == [1, 2, 3]


class TestGenerateCommands:
    """Tests for command generation."""

    def test_generate_commands_basic(self, temp_dir):
        """Test basic command generation."""
        # Create input directory with accelerometer files
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        for i in range(3):
            (input_dir / f"subject_{i}.cwa").touch()

        output_dir = temp_dir / "output"
        cmds_file = temp_dir / "commands.txt"

        gencmd_mod.generate_commands(
            str(input_dir),
            str(output_dir),
            cmdsfile=str(cmds_file),
            fext="cwa"
        )

        assert cmds_file.exists()
        with open(cmds_file) as f:
            lines = f.readlines()

        assert len(lines) == 3
        for line in lines:
            assert "stepcount" in line
            assert "--outdir" in line

    def test_generate_commands_nested_dirs(self, temp_dir):
        """Test command generation with nested directories."""
        input_dir = temp_dir / "input"
        (input_dir / "group_a").mkdir(parents=True)
        (input_dir / "group_b").mkdir(parents=True)

        (input_dir / "group_a" / "subject_1.cwa").touch()
        (input_dir / "group_a" / "subject_2.cwa").touch()
        (input_dir / "group_b" / "subject_3.cwa").touch()

        output_dir = temp_dir / "output"
        cmds_file = temp_dir / "commands.txt"

        gencmd_mod.generate_commands(
            str(input_dir),
            str(output_dir),
            cmdsfile=str(cmds_file)
        )

        with open(cmds_file) as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify directory structure is preserved
        for line in lines:
            if "subject_1" in line:
                assert "group_a" in line

    def test_generate_commands_with_options(self, temp_dir):
        """Test command generation with additional options."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        (input_dir / "subject.cwa").touch()

        cmds_file = temp_dir / "commands.txt"

        gencmd_mod.generate_commands(
            str(input_dir),
            str(temp_dir / "output"),
            cmdsfile=str(cmds_file),
            cmdopts="--model-type rf --quiet"
        )

        with open(cmds_file) as f:
            content = f.read()

        assert "--model-type rf" in content
        assert "--quiet" in content

    def test_generate_commands_different_extensions(self, temp_dir):
        """Test command generation with different file extensions."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        (input_dir / "subject_1.cwa").touch()
        (input_dir / "subject_2.gt3x").touch()
        (input_dir / "subject_3.csv").touch()

        # Test CWA only
        cmds_file = temp_dir / "commands_cwa.txt"
        gencmd_mod.generate_commands(
            str(input_dir),
            str(temp_dir / "output"),
            cmdsfile=str(cmds_file),
            fext="cwa"
        )

        with open(cmds_file) as f:
            lines = f.readlines()

        assert len(lines) == 1
        assert "subject_1.cwa" in lines[0]

    def test_generate_commands_compressed_files(self, temp_dir):
        """Test command generation finds compressed files."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        (input_dir / "subject_1.csv").touch()
        (input_dir / "subject_2.csv.gz").touch()

        cmds_file = temp_dir / "commands.txt"

        gencmd_mod.generate_commands(
            str(input_dir),
            str(temp_dir / "output"),
            cmdsfile=str(cmds_file),
            fext="csv"
        )

        with open(cmds_file) as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_generate_commands_empty_dir(self, temp_dir):
        """Test command generation with empty input directory."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        cmds_file = temp_dir / "commands.txt"

        gencmd_mod.generate_commands(
            str(input_dir),
            str(temp_dir / "output"),
            cmdsfile=str(cmds_file)
        )

        with open(cmds_file) as f:
            content = f.read()

        assert content == ""


class TestCLIIntegration:
    """Integration tests for CLI utilities."""

    def test_roundtrip_workflow(self, temp_dir, mock_info_json):
        """Test generate commands -> (mock run) -> collate outputs."""
        # Setup input files
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        (input_dir / "subject_1.cwa").touch()
        (input_dir / "subject_2.cwa").touch()

        # Generate commands
        output_dir = temp_dir / "output"
        cmds_file = temp_dir / "commands.txt"

        gencmd_mod.generate_commands(
            str(input_dir),
            str(output_dir),
            cmdsfile=str(cmds_file)
        )

        # Simulate processing (create output files)
        for i in [1, 2]:
            subject_dir = output_dir / f"subject_{i}"
            subject_dir.mkdir(parents=True)

            info = mock_info_json.copy()
            info['Filename'] = f"subject_{i}.cwa"
            with open(subject_dir / f"subject_{i}-Info.json", 'w') as f:
                json.dump(info, f)

            pd.DataFrame({
                'Filename': [f'subject_{i}.cwa'],
                'Date': ['2024-01-15'],
                'Steps': [8000 + i * 100]
            }).to_csv(subject_dir / f"subject_{i}-Daily.csv.gz", index=False)

        # Collate outputs
        collated_dir = temp_dir / "collated"
        collate_mod.collate_outputs(
            str(output_dir),
            str(collated_dir),
            included=['daily']
        )

        # Verify final output
        info_df = pd.read_csv(collated_dir / "Info.csv.gz")
        daily_df = pd.read_csv(collated_dir / "Daily.csv.gz")

        assert len(info_df) == 2
        assert len(daily_df) == 2


class TestErrorPaths:
    """Tests for error handling in CLI utilities."""

    def test_generate_commands_empty_dir(self, temp_dir):
        """Test generate_commands with empty input directory handles gracefully."""
        empty_dir = temp_dir / 'empty_inputs'
        empty_dir.mkdir()
        output_dir = temp_dir / 'output'
        cmds_file = temp_dir / 'commands.txt'

        # Should complete without error for empty input
        gencmd_mod.generate_commands(str(empty_dir), str(output_dir), cmdsfile=str(cmds_file))

        # Verify command list file was created but is empty
        assert cmds_file.exists()
        content = cmds_file.read_text()
        # Should have no actual commands (0 files found) - file should be empty
        assert content.strip() == '', f"Expected empty commands file for empty input dir, got: {content[:100]}"

    def test_collate_empty_results_dir(self, temp_dir):
        """Reject an empty results directory before publishing outputs."""
        empty_results = temp_dir / 'empty_results'
        empty_results.mkdir()
        output_dir = temp_dir / 'collated'

        for _ in range(2):
            with pytest.raises(FileNotFoundError, match="No result files found"):
                collate_mod.collate_outputs(str(empty_results), str(output_dir))

        assert not list(output_dir.glob("*.csv.gz"))

    def test_collate_malformed_json(self, temp_dir):
        """Test collate_outputs handles malformed JSON gracefully."""
        results_dir = temp_dir / 'results'
        subject_dir = results_dir / 'subject1'
        subject_dir.mkdir(parents=True)

        # Write malformed JSON
        bad_json = subject_dir / 'subject1-Info.json'
        bad_json.write_text('{invalid json content')

        output_dir = temp_dir / 'collated'

        with pytest.raises(ValueError, match=str(bad_json)) as error:
            collate_mod.collate_outputs(str(results_dir), str(output_dir))

        assert isinstance(error.value.__cause__, json.JSONDecodeError)


class TestCollationEdgeCases:
    """Edge case tests for collation utilities."""

    def test_collate_csvs_same_columns(self, temp_dir):
        """Test CSV collation with files having same columns."""
        csv_dir = temp_dir / "results"
        csv_dir.mkdir()

        # CSV 1
        csv1 = csv_dir / "subject1-Daily.csv.gz"
        pd.DataFrame({
            'Filename': ['subject1.csv'],
            'Date': ['2024-01-15'],
            'Steps': [8000]
        }).to_csv(csv1, index=False)

        # CSV 2 with same columns
        csv2 = csv_dir / "subject2-Daily.csv.gz"
        pd.DataFrame({
            'Filename': ['subject2.csv'],
            'Date': ['2024-01-16'],
            'Steps': [9000]
        }).to_csv(csv2, index=False)

        outfile = temp_dir / "Daily.csv.gz"
        csv_files = list(csv_dir.glob("*-Daily.csv.gz"))

        collate_mod.collate_csvs(csv_files, outfile)

        df = pd.read_csv(outfile)
        assert len(df) == 2
        assert 'Filename' in df.columns
        assert 'Steps' in df.columns

    def test_collate_jsons_with_nested_values(self, temp_dir, mock_info_json):
        """Test JSON collation with nested dictionary values."""
        json_dir = temp_dir / "results"
        json_dir.mkdir()

        # Create JSON with nested value
        info = mock_info_json.copy()
        info['NestedData'] = {'inner_key': 'inner_value', 'count': 42}

        json_file = json_dir / "subject" / "subject-Info.json"
        json_file.parent.mkdir(parents=True)
        with open(json_file, 'w') as f:
            json.dump(info, f)

        outfile = temp_dir / "Info.csv.gz"

        collate_mod.collate_jsons([json_file], outfile)

        # Should handle nested values (convert to string or flatten)
        df = pd.read_csv(outfile)
        assert len(df) == 1

    def test_collate_outputs_missing_file_types(self, temp_dir, mock_info_json):
        """Test collation when some expected file types are missing."""
        results_dir = temp_dir / "results"
        subject_dir = results_dir / "subject"
        subject_dir.mkdir(parents=True)

        # Only create Info.json, no Daily or Hourly
        with open(subject_dir / "subject-Info.json", 'w') as f:
            json.dump(mock_info_json, f)

        output_dir = temp_dir / "collated"

        # Should handle missing file types gracefully
        collate_mod.collate_outputs(
            str(results_dir),
            str(output_dir),
            included=['daily', 'hourly']  # These don't exist
        )

        # Info.csv.gz should still be created
        assert (output_dir / "Info.csv.gz").exists()

    def test_generate_commands_with_slurm_array(self, temp_dir):
        """Test generate_commands with SLURM array job format."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        for i in range(5):
            (input_dir / f"subject_{i}.cwa").touch()

        cmds_file = temp_dir / "commands.txt"

        gencmd_mod.generate_commands(
            str(input_dir),
            str(temp_dir / "output"),
            cmdsfile=str(cmds_file),
            fext="cwa"
        )

        with open(cmds_file) as f:
            lines = f.readlines()

        # Should have 5 commands, one per file
        assert len(lines) == 5
        # Each should be a valid command
        for line in lines:
            assert line.strip().startswith("stepcount")


class TestBoundaryValues:
    """Tests for boundary value conditions."""

    def test_collate_single_file(self, temp_dir, mock_info_json):
        """Test collation with single file."""
        json_dir = temp_dir / "results"
        json_dir.mkdir()

        json_file = json_dir / "single-Info.json"
        with open(json_file, 'w') as f:
            json.dump(mock_info_json, f)

        outfile = temp_dir / "Info.csv.gz"
        collate_mod.collate_jsons([json_file], outfile)

        df = pd.read_csv(outfile)
        assert len(df) == 1

    def test_collate_many_files(self, temp_dir, mock_info_json):
        """Test collation with many files."""
        json_dir = temp_dir / "results"
        json_dir.mkdir()

        json_files = []
        for i in range(100):
            info = mock_info_json.copy()
            info['Filename'] = f"subject_{i}.csv"
            info['TotalSteps'] = 8000 + i

            json_file = json_dir / f"subject_{i}-Info.json"
            with open(json_file, 'w') as f:
                json.dump(info, f)
            json_files.append(json_file)

        outfile = temp_dir / "Info.csv.gz"
        collate_mod.collate_jsons(json_files, outfile)

        df = pd.read_csv(outfile)
        assert len(df) == 100
