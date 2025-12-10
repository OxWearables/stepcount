"""
Tests for stepcount.cli_utils module.

Tests cover:
- collate_outputs functions
- generate_commands functions
"""
import pytest
import json
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import importlib

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
        collate_mod.collate_jsons([json_file], outfile, overwrite=True)
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
        import os
        file_size = os.path.getsize(outfile)
        # Gzipped empty DataFrame is very small (< 50 bytes)
        assert file_size < 50


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
        """Test collate_outputs with empty results directory."""
        import gzip
        empty_results = temp_dir / 'empty_results'
        empty_results.mkdir()
        output_dir = temp_dir / 'collated'

        # Should handle gracefully (empty output)
        collate_mod.collate_outputs(str(empty_results), str(output_dir))

        # Verify output directory was created
        assert output_dir.exists(), "Output directory should be created"
        # Verify output files exist but contain no data (just gzip headers or empty CSV)
        output_files = list(output_dir.glob('*.csv.gz'))
        assert len(output_files) > 0, "Collate should create output files"
        # Each file should be empty or contain only whitespace when decompressed
        for f in output_files:
            with gzip.open(f, 'rt') as gz:
                content = gz.read().strip()
                assert content == '', f"Expected empty content in {f.name}, got: {content[:50]}"

    def test_collate_malformed_json(self, temp_dir):
        """Test collate_outputs handles malformed JSON gracefully."""
        results_dir = temp_dir / 'results'
        subject_dir = results_dir / 'subject1'
        subject_dir.mkdir(parents=True)

        # Write malformed JSON
        bad_json = subject_dir / 'subject1-Info.json'
        bad_json.write_text('{invalid json content')

        output_dir = temp_dir / 'collated'

        # Should raise JSONDecodeError or skip the file
        with pytest.raises(json.JSONDecodeError):
            collate_mod.collate_outputs(str(results_dir), str(output_dir))


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
