from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import stat
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence, TextIO, Union

import pandas as pd
from tqdm.auto import tqdm

_LOGGER = logging.getLogger(__name__)
_PUBLICATION_JOURNAL = "journal.json"
_PUBLICATION_PREPARED = "PREPARED"
_PUBLICATION_COMMITTED = "COMMITTED"


def collate_outputs(
    results_dir: Union[str, PathLike[str]],
    collated_results_dir: Union[str, PathLike[str]] = "collated_outputs/",
    included: Sequence[str] = ("daily", "hourly", "minutely", "bouts"),
    schema_policy: str = "union",
) -> None:
    """Collate all results files in <results_dir>.
    :param str results_dir: Root directory in which to search for result files.
    :param str collated_results_dir: Directory to write the collated files to.
    :param list included: Type of result files to collate ('daily', 'hourly', 'minutely', 'bouts').
    :param str schema_policy: Reconcile CSV columns by union or require matching column sets.
    :return: Collated files written to <collated_results_dir>; requested types without inputs are removed.
    :rtype: void
    """

    _validate_schema_policy(schema_policy)

    print("Searching files...")

    results_dir = Path(results_dir)
    collated_results_dir = Path(collated_results_dir)
    csv_keys_by_include = {
        "daily": ("Daily", "DailyAdjusted"),
        "hourly": ("Hourly", "HourlyAdjusted"),
        "minutely": ("Minutely", "MinutelyAdjusted"),
        "bouts": ("Bouts",),
    }
    csv_keys = [
        key
        for included_value in included
        for key in csv_keys_by_include.get(included_value.lower(), ())
    ]
    _resolve_results_directory(results_dir)
    collated_results_dir.mkdir(parents=True, exist_ok=True)
    with _locked_output_directory(collated_results_dir.resolve()):
        info_files, csv_files = _discover_output_files(
            results_dir,
            collated_results_dir,
            csv_keys,
        )
        if not info_files and not any(csv_files.values()):
            raise FileNotFoundError(f"No result files found under {results_dir}")

        info_frame = _load_json_frame(info_files) if info_files else None
        csv_plans = {
            key: _plan_csv_collation(file_list, schema_policy)
            for key, file_list in csv_files.items()
        }
        info_outfile = collated_results_dir / "Info.csv.gz"
        csv_outfiles = {
            key: collated_results_dir / f"{key}.csv.gz"
            for key in csv_plans
        }
        existing_outputs = {
            outfile: _path_exists(outfile)
            for outfile in [info_outfile, *csv_outfiles.values()]
        }

        staged_outputs: list[_StagedOutput] = []
        try:
            print(f"Collating {len(info_files)} Info files...")
            if info_frame is None:
                staged_outputs.append(_deletion(info_outfile))
            else:
                staged_outputs.append(_stage_json_frame(info_frame, info_outfile))

            for key, plan in csv_plans.items():
                print(f"Collating {len(plan.files)} {key} files...")
                staged_outputs.append(
                    _stage_csv_collation(plan, csv_outfiles[key])
                )

            _publish_outputs_locked(staged_outputs)
        finally:
            for staged_output in staged_outputs:
                _cleanup_staged_output(staged_output)

    _report_output(bool(info_files), info_outfile, existing_outputs[info_outfile], "Info")
    for key, plan in csv_plans.items():
        _report_output(
            bool(plan.files),
            csv_outfiles[key],
            existing_outputs[csv_outfiles[key]],
            key,
        )

    return


def collate_jsons(
    file_list: Sequence[Path],
    outfile: Path,
) -> None:
    """ Collate a list of JSON files into a single CSV file."""

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with _locked_output_directory(outfile.parent.resolve()):
        frame = _load_json_frame(file_list)
        staged_output = _stage_json_frame(frame, outfile)
        try:
            if _path_exists(outfile):
                print(f"Overwriting existing file: {outfile}")
            _publish_outputs_locked([staged_output])
        finally:
            _cleanup_staged_output(staged_output)

    return


def collate_csvs(
    file_list: Sequence[Path],
    outfile: Path,
    schema_policy: str = "union",
) -> None:
    """Collate CSV files by column name using bounded memory.

    The union policy retains columns in first-observed order and fills fields
    absent from an input with an empty value. The strict policy accepts reordered
    columns but rejects differing column sets.
    """

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with _locked_output_directory(outfile.parent.resolve()):
        plan = _plan_csv_collation(file_list, schema_policy)
        existed = _path_exists(outfile)
        staged_output = _stage_csv_collation(plan, outfile)
        try:
            if plan.files and existed:
                print(f"Overwriting existing file: {outfile}")
            elif not plan.files and existed:
                print(f"Removing existing file with no replacement inputs: {outfile}")
            _publish_outputs_locked([staged_output])
        finally:
            _cleanup_staged_output(staged_output)

    return


@dataclass(frozen=True)
class _CsvCollationPlan:
    columns: tuple[str, ...]
    schemas: tuple[tuple[str, ...], ...]
    files: tuple[tuple[Path, int], ...]


@dataclass(frozen=True)
class _StagedOutput:
    outfile: Path
    temporary_path: Union[Path, None]
    final_mode: Union[int, None]


@dataclass
class _PublicationRecord:
    staged_output: _StagedOutput
    backup: Union[Path, None] = None
    had_output: bool = False
    attempted: bool = False
    published_identity: Union[tuple[int, int], None] = None


def _discover_output_files(
    results_dir: Path,
    collated_results_dir: Path,
    csv_keys: Sequence[str],
) -> tuple[list[Path], dict[str, list[Path]]]:
    resolved_results_dir = _resolve_results_directory(results_dir)
    resolved_collated_results_dir = collated_results_dir.resolve()

    exclude_collated_tree = (
        resolved_results_dir != resolved_collated_results_dir
        and _is_relative_to(resolved_collated_results_dir, resolved_results_dir)
    )
    info_files: list[Path] = []
    csv_files: dict[str, list[Path]] = {key: [] for key in csv_keys}
    csv_suffix_to_key = {
        f"-{key}.csv.gz": key
        for key in csv_files
    }
    csv_suffixes = tuple(csv_suffix_to_key)

    for root_value, directory_names, file_names in os.walk(
        resolved_results_dir,
        onerror=_raise_walk_error,
    ):
        root = Path(root_value)
        if exclude_collated_tree:
            directory_names[:] = [
                name
                for name in directory_names
                if root / name != resolved_collated_results_dir
            ]

        for file_name in file_names:
            is_info_file = file_name.endswith("-Info.json")
            if not is_info_file and not file_name.endswith(csv_suffixes):
                continue

            file = root / file_name
            if not (
                _is_regular_output_file(file)
                and _is_contained_output_file(file, resolved_results_dir)
            ):
                continue
            if is_info_file:
                info_files.append(file)
                continue
            for suffix, key in csv_suffix_to_key.items():
                if file_name.endswith(suffix):
                    csv_files[key].append(file)
                    break

    info_files.sort(key=lambda path: path.as_posix())
    for files_for_key in csv_files.values():
        files_for_key.sort(key=lambda path: path.as_posix())

    return info_files, csv_files


def _resolve_results_directory(results_dir: Path) -> Path:
    resolved_results_dir = results_dir.resolve()
    try:
        results_stat = resolved_results_dir.stat()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Results directory does not exist: {resolved_results_dir}"
        ) from error
    if not stat.S_ISDIR(results_stat.st_mode):
        raise NotADirectoryError(f"Results path is not a directory: {resolved_results_dir}")
    return resolved_results_dir


def _raise_walk_error(error: OSError) -> None:
    raise error


def _is_regular_output_file(file: Path) -> bool:
    try:
        file.stat()
        return stat.S_ISREG(file.lstat().st_mode)
    except FileNotFoundError:
        return False


def _is_contained_output_file(file: Path, results_dir: Path) -> bool:
    try:
        resolved_file = file.resolve(strict=True)
    except FileNotFoundError:
        return False
    return _is_relative_to(resolved_file, results_dir)


def _write_csv_collation(
    plan: _CsvCollationPlan,
    outfile: Path,
    compressed: bool,
) -> None:
    previous_schema_id: Union[int, None] = None
    previous_mapping: Union[tuple[Union[int, None], ...], None] = None
    with _open_csv(outfile, "wt", compressed=compressed) as output_stream:
        writer = csv.writer(output_stream, lineterminator="\n")
        writer.writerow(plan.columns)

        for file, schema_id in tqdm(plan.files):
            expected_header = plan.schemas[schema_id]
            if schema_id == previous_schema_id and previous_mapping is not None:
                mapping = previous_mapping
            else:
                source_positions = {column: index for index, column in enumerate(expected_header)}
                mapping = tuple(source_positions.get(column) for column in plan.columns)
                previous_schema_id = schema_id
                previous_mapping = mapping
            for row in _iter_mapped_csv_rows(file, expected_header, mapping):
                writer.writerow(row)


def _iter_mapped_csv_rows(
    file: Path,
    expected_header: tuple[str, ...],
    mapping: tuple[Union[int, None], ...],
) -> Iterator[list[str]]:
    identity_mapping = mapping == tuple(range(len(expected_header)))
    try:
        with _open_csv(file, "rt") as input_stream:
            reader = csv.reader(input_stream, strict=True)
            current_header = _next_csv_header(reader, file)
            _validate_csv_header(current_header, file)
            if current_header != expected_header:
                raise ValueError(f"CSV header changed during collation: {file}")

            for row in reader:
                if row == []:
                    continue
                if len(row) != len(expected_header):
                    raise ValueError(
                        f"CSV row has {len(row)} fields but header has "
                        f"{len(expected_header)} at {file}:{reader.line_num}"
                    )
                if identity_mapping:
                    yield row
                else:
                    yield [
                        row[position] if position is not None else ""
                        for position in mapping
                    ]
    except (OSError, EOFError, csv.Error, UnicodeError) as error:
        raise ValueError(f"Could not read CSV file {file}: {error}") from error


def _load_json_frame(file_list: Sequence[Path]) -> pd.DataFrame:
    records: list[Any] = []
    for file in tqdm(file_list):
        try:
            with open(file, 'r') as stream:
                record = json.load(stream, object_pairs_hook=OrderedDict)
        except (OSError, json.JSONDecodeError, UnicodeError) as error:
            raise ValueError(f"Could not read JSON file {file}: {error}") from error
        if isinstance(record, dict):
            record = {
                key: convert_ordereddict(value)
                for key, value in record.items()
            }
        records.append(record)
    return pd.DataFrame(records)


def _stage_json_frame(
    frame: pd.DataFrame,
    outfile: Path,
) -> _StagedOutput:
    staged_output = _create_staged_output(outfile)
    temporary_path = staged_output.temporary_path
    assert temporary_path is not None
    try:
        frame.to_csv(temporary_path, index=False)
        _finalize_staged_output(staged_output)
    except BaseException:
        _cleanup_staged_output(staged_output)
        raise
    return staged_output


def _stage_csv_collation(
    plan: _CsvCollationPlan,
    outfile: Path,
) -> _StagedOutput:
    if not plan.files:
        return _deletion(outfile)

    staged_output = _create_staged_output(outfile)
    temporary_path = staged_output.temporary_path
    assert temporary_path is not None
    try:
        _write_csv_collation(
            plan,
            temporary_path,
            compressed=outfile.suffix == ".gz",
        )
        _finalize_staged_output(staged_output)
    except BaseException:
        _cleanup_staged_output(staged_output)
        raise
    return staged_output


def _create_staged_output(outfile: Path) -> _StagedOutput:
    temporary_path = Path(tempfile.mkdtemp(
        dir=outfile.parent,
        prefix=f".{outfile.name}.",
        suffix=".tmp",
    )) / outfile.name
    try:
        temporary_path.touch(mode=0o666, exist_ok=False)
        if outfile.exists():
            final_mode = stat.S_IMODE(outfile.stat().st_mode)
        else:
            final_mode = stat.S_IMODE(temporary_path.stat().st_mode)
        os.chmod(temporary_path, final_mode | stat.S_IWUSR)
    except BaseException:
        _cleanup_staged_output(
            _StagedOutput(outfile, temporary_path, None)
        )
        raise
    return _StagedOutput(outfile, temporary_path, final_mode)


def _deletion(outfile: Path) -> _StagedOutput:
    return _StagedOutput(outfile, None, None)


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _report_output(
    has_inputs: bool,
    outfile: Path,
    existed: bool,
    label: str,
) -> None:
    if has_inputs:
        print(f"Collated {label} CSV written to", outfile)
    elif existed:
        print(f"No {label} inputs found; removed existing CSV at", outfile)
    else:
        print(f"No {label} inputs found; no CSV written at", outfile)


def _finalize_staged_output(staged_output: _StagedOutput) -> None:
    if staged_output.temporary_path is None or staged_output.final_mode is None:
        return
    os.chmod(staged_output.temporary_path, staged_output.final_mode)


def _cleanup_staged_output(staged_output: _StagedOutput) -> None:
    temporary_path = staged_output.temporary_path
    if temporary_path is None:
        return
    try:
        temporary_path.unlink(missing_ok=True)
    except OSError as error:
        _LOGGER.warning(
            "Could not remove staged output %s: %s",
            temporary_path,
            error,
        )
    try:
        temporary_path.parent.rmdir()
    except OSError as error:
        _LOGGER.warning(
            "Could not remove staging directory %s: %s",
            temporary_path.parent,
            error,
        )


def _publish_outputs_locked(staged_outputs: Sequence[_StagedOutput]) -> None:
    if not staged_outputs:
        return
    output_directory = _validate_publication(staged_outputs)
    transaction_dir = Path(tempfile.mkdtemp(
        dir=output_directory,
        prefix=".stepcount-publish-",
        suffix=".tmp",
    ))
    records = _prepare_publication_records(staged_outputs, transaction_dir)
    cleanup_transaction = False
    try:
        _write_publication_journal(records, transaction_dir)
        _backup_publication(records)
        _fsync_directory(transaction_dir)
        _fsync_directory(output_directory)
        _apply_publication(records)
        _sync_published_outputs(records)
        _fsync_directory(output_directory)
        _create_publication_marker(transaction_dir, _PUBLICATION_COMMITTED)
    except BaseException as error:
        if (transaction_dir / _PUBLICATION_COMMITTED).exists():
            raise
        rollback_errors = _restore_publication(records)
        if rollback_errors:
            backup_paths = _remaining_backup_paths(rollback_errors)
            raise RuntimeError(
                f"Could not publish collated outputs and rollback failed for "
                f"{[record.staged_output.outfile for record, _ in rollback_errors]}; "
                f"backups preserved at {backup_paths}"
            ) from error
        cleanup_transaction = True
        raise
    else:
        cleanup_transaction = True
    finally:
        if cleanup_transaction:
            _cleanup_publication(records, transaction_dir)


def _validate_publication(staged_outputs: Sequence[_StagedOutput]) -> Path:
    output_directories = {output.outfile.parent.resolve() for output in staged_outputs}
    if len(output_directories) != 1:
        raise ValueError("All staged outputs must share one output directory")
    outfiles = [output.outfile for output in staged_outputs]
    if len(set(outfiles)) != len(outfiles):
        raise ValueError("Each staged output must have a unique destination")
    for outfile in outfiles:
        try:
            output_mode = outfile.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(output_mode):
            raise OSError(f"Output path must not be a symlink: {outfile}")
        if not stat.S_ISREG(output_mode):
            raise OSError(f"Output path is not a regular file or symlink: {outfile}")
    return next(iter(output_directories))


def _prepare_publication_records(
    staged_outputs: Sequence[_StagedOutput],
    transaction_dir: Path,
) -> list[_PublicationRecord]:
    records: list[_PublicationRecord] = []
    for index, staged_output in enumerate(staged_outputs):
        had_output = _path_exists(staged_output.outfile)
        backup = (
            transaction_dir / f"{index}-{staged_output.outfile.name}"
            if had_output
            else None
        )
        records.append(_PublicationRecord(staged_output, backup, had_output))
    return records


def _write_publication_journal(
    records: Sequence[_PublicationRecord],
    transaction_dir: Path,
) -> None:
    journal = {
        "version": 1,
        "outputs": [
            {
                "name": record.staged_output.outfile.name,
                "had_output": record.had_output,
                "backup": record.backup.name if record.backup is not None else None,
            }
            for record in records
        ],
    }
    journal_path = transaction_dir / _PUBLICATION_JOURNAL
    with open(journal_path, "x", encoding="utf-8") as stream:
        json.dump(journal, stream, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    _create_publication_marker(transaction_dir, _PUBLICATION_PREPARED)


def _create_publication_marker(transaction_dir: Path, marker_name: str) -> None:
    marker = transaction_dir / marker_name
    with open(marker, "xb") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(transaction_dir)


def _backup_publication(
    records: Sequence[_PublicationRecord],
) -> None:
    for record in records:
        if record.backup is not None:
            os.replace(record.staged_output.outfile, record.backup)


def _apply_publication(records: Sequence[_PublicationRecord]) -> None:
    for record in records:
        staged_output = record.staged_output
        record.attempted = True
        if staged_output.temporary_path is None:
            continue
        source_stat = staged_output.temporary_path.stat()
        record.published_identity = source_stat.st_dev, source_stat.st_ino
        os.replace(staged_output.temporary_path, staged_output.outfile)


def _sync_published_outputs(records: Sequence[_PublicationRecord]) -> None:
    for record in records:
        outfile = record.staged_output.outfile
        if record.staged_output.temporary_path is None or not _path_exists(outfile):
            continue
        descriptor = os.open(outfile, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _remaining_backup_paths(
    rollback_errors: Sequence[tuple[_PublicationRecord, BaseException]],
) -> list[Path]:
    return [
        record.backup
        for record, _ in rollback_errors
        if record.backup is not None and _path_exists(record.backup)
    ]


def _restore_publication(
    records: Sequence[_PublicationRecord],
) -> list[tuple[_PublicationRecord, BaseException]]:
    failures: list[tuple[_PublicationRecord, BaseException]] = []
    for record in reversed(records):
        try:
            _restore_publication_record(record)
        except BaseException as error:
            failures.append((record, error))
    return failures


def _restore_publication_record(record: _PublicationRecord) -> None:
    outfile = record.staged_output.outfile
    has_backup = record.backup is not None and _path_exists(record.backup)
    if not record.attempted and not has_backup:
        return
    if _path_exists(outfile):
        if record.published_identity is None:
            raise RuntimeError(f"Unexpected output appeared during rollback: {outfile}")
        _remove_if_identity_matches(outfile, record.published_identity)
    if has_backup:
        assert record.backup is not None
        os.replace(record.backup, outfile)


def _remove_if_identity_matches(path: Path, identity: tuple[int, int]) -> None:
    try:
        current_stat = path.lstat()
    except FileNotFoundError:
        return
    if (current_stat.st_dev, current_stat.st_ino) != identity:
        raise RuntimeError(f"Refusing to remove replaced output during rollback: {path}")
    path.unlink()


def _cleanup_publication(
    records: Sequence[_PublicationRecord],
    transaction_dir: Path,
) -> None:
    cleanup_succeeded = _cleanup_publication_backups(records)
    if (transaction_dir / _PUBLICATION_COMMITTED).exists():
        cleanup_succeeded = _cleanup_committed_publication(
            transaction_dir,
            cleanup_succeeded,
        )
    elif cleanup_succeeded:
        cleanup_succeeded = _cleanup_rolled_back_publication(transaction_dir)
    if not cleanup_succeeded:
        return
    try:
        transaction_dir.rmdir()
        _fsync_directory(transaction_dir.parent)
    except OSError as error:
        _LOGGER.warning(
            "Could not remove publication directory %s: %s",
            transaction_dir,
            error,
        )


def _cleanup_publication_backups(
    records: Sequence[_PublicationRecord],
) -> bool:
    cleanup_succeeded = True
    for record in records:
        backup = record.backup
        if backup is None:
            continue
        cleanup_succeeded = _unlink_publication_path(backup) and cleanup_succeeded
    return cleanup_succeeded


def _cleanup_committed_publication(
    transaction_dir: Path,
    cleanup_succeeded: bool,
) -> bool:
    for metadata_name in (_PUBLICATION_JOURNAL, _PUBLICATION_PREPARED):
        cleanup_succeeded = (
            _unlink_publication_path(transaction_dir / metadata_name)
            and cleanup_succeeded
        )
    if not cleanup_succeeded:
        return False
    if not _try_fsync_directory(transaction_dir):
        return False
    return _unlink_publication_path(transaction_dir / _PUBLICATION_COMMITTED)


def _cleanup_rolled_back_publication(transaction_dir: Path) -> bool:
    if not _unlink_publication_path(transaction_dir / _PUBLICATION_PREPARED):
        return False
    if not _try_fsync_directory(transaction_dir):
        return False
    return _unlink_publication_path(transaction_dir / _PUBLICATION_JOURNAL)


def _unlink_publication_path(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        _LOGGER.warning("Could not remove publication artifact %s: %s", path, error)
        return False
    return True


def _try_fsync_directory(directory: Path) -> bool:
    try:
        _fsync_directory(directory)
    except OSError as error:
        _LOGGER.warning("Could not sync publication directory %s: %s", directory, error)
        return False
    return True


@contextmanager
def collated_outputs_snapshot(
    collated_results_dir: Union[str, PathLike[str]],
) -> Iterator[Path]:
    """Yield a consistent collated-output snapshot while blocking publishers."""

    output_directory = Path(collated_results_dir).resolve()
    if not output_directory.is_dir():
        raise NotADirectoryError(
            f"Collated output path is not a directory: {output_directory}"
        )
    with _locked_output_directory(output_directory):
        yield output_directory


@contextmanager
def _locked_output_directory(output_directory: Path) -> Iterator[None]:
    with _output_lock(output_directory):
        _recover_publications(output_directory)
        yield


def _recover_publications(output_directory: Path) -> None:
    for transaction_dir in sorted(
        output_directory.glob(".stepcount-publish-*.tmp")
    ):
        transaction_stat = transaction_dir.lstat()
        if not stat.S_ISDIR(transaction_stat.st_mode):
            raise RuntimeError(
                f"Publication recovery path is not a directory: {transaction_dir}"
            )
        if (transaction_dir / _PUBLICATION_COMMITTED).exists():
            _remove_publication_directory(transaction_dir)
            _fsync_directory(output_directory)
            continue
        prepared = transaction_dir / _PUBLICATION_PREPARED
        if not prepared.exists():
            _remove_publication_directory(transaction_dir)
            continue

        records = _load_publication_journal(output_directory, transaction_dir)
        rollback_errors = _restore_recovered_publication(records)
        if rollback_errors:
            backup_paths = _remaining_backup_paths(rollback_errors)
            raise RuntimeError(
                f"Could not recover interrupted publication for "
                f"{[record.staged_output.outfile for record, _ in rollback_errors]}; "
                f"backups preserved at {backup_paths}"
            ) from rollback_errors[0][1]
        _fsync_directory(output_directory)
        _cleanup_publication(records, transaction_dir)


def _load_publication_journal(
    output_directory: Path,
    transaction_dir: Path,
) -> list[_PublicationRecord]:
    journal_path = transaction_dir / _PUBLICATION_JOURNAL
    try:
        with open(journal_path, "r", encoding="utf-8") as stream:
            journal = json.load(stream)
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise RuntimeError(
            f"Could not read publication journal {journal_path}: {error}"
        ) from error
    if not isinstance(journal, dict) or journal.get("version") != 1:
        raise RuntimeError(f"Invalid publication journal: {journal_path}")
    outputs = journal.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError(f"Invalid publication journal: {journal_path}")

    records: list[_PublicationRecord] = []
    names: set[str] = set()
    backup_names: set[str] = set()
    for output in outputs:
        if not isinstance(output, dict):
            raise RuntimeError(f"Invalid publication journal: {journal_path}")
        name = output.get("name")
        backup_name = output.get("backup")
        had_output = output.get("had_output")
        if (
            not _is_path_component(name)
            or not isinstance(had_output, bool)
            or (
                backup_name is not None
                and not _is_path_component(backup_name)
            )
            or (had_output != (backup_name is not None))
        ):
            raise RuntimeError(f"Invalid publication journal: {journal_path}")
        assert isinstance(name, str)
        if name in names or (
            isinstance(backup_name, str) and backup_name in backup_names
        ):
            raise RuntimeError(f"Invalid publication journal: {journal_path}")
        names.add(name)
        if isinstance(backup_name, str):
            backup_names.add(backup_name)
        backup = (
            transaction_dir / backup_name
            if isinstance(backup_name, str)
            else None
        )
        records.append(
            _PublicationRecord(
                _deletion(output_directory / name),
                backup,
                had_output,
            )
        )
    return records


def _is_path_component(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value not in {"", ".", ".."}
        and Path(value).name == value
    )


def _restore_recovered_publication(
    records: Sequence[_PublicationRecord],
) -> list[tuple[_PublicationRecord, BaseException]]:
    failures: list[tuple[_PublicationRecord, BaseException]] = []
    for record in reversed(records):
        try:
            outfile = record.staged_output.outfile
            if record.had_output:
                if record.backup is not None and _path_exists(record.backup):
                    _validate_recovery_output(outfile)
                    os.replace(record.backup, outfile)
                elif not _path_exists(outfile):
                    raise RuntimeError(f"Missing output and backup during recovery: {outfile}")
            elif _path_exists(outfile):
                _validate_recovery_output(outfile)
                outfile.unlink()
        except BaseException as error:
            failures.append((record, error))
    return failures


def _validate_recovery_output(outfile: Path) -> None:
    try:
        output_mode = outfile.lstat().st_mode
    except FileNotFoundError:
        return
    if not (stat.S_ISREG(output_mode) or stat.S_ISLNK(output_mode)):
        raise RuntimeError(f"Recovery output is not a regular file or symlink: {outfile}")


def _remove_publication_directory(transaction_dir: Path) -> None:
    for child in transaction_dir.iterdir():
        child_mode = child.lstat().st_mode
        if not (stat.S_ISREG(child_mode) or stat.S_ISLNK(child_mode)):
            raise RuntimeError(
                f"Unprepared publication contains unsupported path: {child}"
            )
        child.unlink()
    transaction_dir.rmdir()
    _fsync_directory(transaction_dir.parent)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _output_lock(output_directory: Path) -> Iterator[None]:
    lock_path = output_directory / ".stepcount-collate.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o666)
    try:
        _validate_lock_file(lock_path, descriptor)
        _lock_descriptor(descriptor)
        try:
            yield
        finally:
            _unlock_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _validate_lock_file(lock_path: Path, descriptor: int) -> None:
    descriptor_stat = os.fstat(descriptor)
    path_stat = lock_path.lstat()
    if not stat.S_ISREG(descriptor_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise RuntimeError(f"Collation lock is not a regular file: {lock_path}")
    if (
        descriptor_stat.st_dev != path_stat.st_dev
        or descriptor_stat.st_ino != path_stat.st_ino
    ):
        raise RuntimeError(f"Collation lock changed while opening: {lock_path}")


def _lock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        vars(msvcrt)["locking"](descriptor, vars(msvcrt)["LK_LOCK"], 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        vars(msvcrt)["locking"](descriptor, vars(msvcrt)["LK_UNLCK"], 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _plan_csv_collation(
    file_list: Sequence[Path],
    schema_policy: str,
) -> _CsvCollationPlan:
    _validate_schema_policy(schema_policy)

    planned_files: list[tuple[Path, int]] = []
    schemas: list[tuple[str, ...]] = []
    schema_ids: dict[tuple[str, ...], int] = {}
    columns: list[str] = []
    seen_columns: set[str] = set()
    reference_columns: Union[set[str], None] = None

    for file_value in file_list:
        file = Path(file_value)
        try:
            with _open_csv(file, "rt") as input_stream:
                header = _next_csv_header(csv.reader(input_stream, strict=True), file)
        except (OSError, EOFError, csv.Error, UnicodeError) as error:
            raise ValueError(f"Could not read CSV file {file}: {error}") from error
        _validate_csv_header(header, file)
        schema_id = schema_ids.get(header)
        if schema_id is None:
            schema_id = len(schemas)
            schemas.append(header)
            schema_ids[header] = schema_id
            if schema_policy == "strict":
                header_columns = set(header)
                if reference_columns is None:
                    reference_columns = header_columns
                elif header_columns != reference_columns:
                    missing = sorted(reference_columns - header_columns)
                    unexpected = sorted(header_columns - reference_columns)
                    raise ValueError(
                        f"CSV schema differs under strict policy for {file}: "
                        f"missing={missing}, unexpected={unexpected}"
                    )
        planned_files.append((file, schema_id))

        for column in header:
            if column not in seen_columns:
                seen_columns.add(column)
                columns.append(column)

    return _CsvCollationPlan(tuple(columns), tuple(schemas), tuple(planned_files))


def _validate_schema_policy(schema_policy: str) -> None:
    if schema_policy not in {"union", "strict"}:
        raise ValueError(
            f"Unknown schema policy {schema_policy!r}; expected 'union' or 'strict'"
        )


def _open_csv(
    file: Path,
    mode: Literal["rt", "wt"],
    compressed: Union[bool, None] = None,
) -> TextIO:
    if compressed is None:
        compressed = file.suffix == ".gz"
    encoding = "utf-8-sig" if mode == "rt" else "utf-8"
    if compressed:
        return gzip.open(file, mode, encoding=encoding, newline="")
    return open(file, mode, encoding=encoding, newline="")


def _next_csv_header(reader: Any, file: Path) -> tuple[str, ...]:
    try:
        return tuple(next(reader))
    except StopIteration as error:
        raise ValueError(f"CSV file has no header: {file}") from error


def _validate_csv_header(header: tuple[str, ...], file: Path) -> None:
    if not header:
        raise ValueError(f"CSV file has no header: {file}")

    blank_columns = [column for column in header if not column.strip()]
    if blank_columns:
        raise ValueError(f"CSV file has blank column names: {file}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for column in header:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    if duplicates:
        raise ValueError(f"CSV file has duplicate column names {duplicates}: {file}")


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def convert_ordereddict(value: Any) -> Any:
    """ Convert OrderedDict to regular dict """
    if isinstance(value, OrderedDict):
        return dict(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help="Root directory in which to search for result files")
    parser.add_argument('--output', '-o', default="collated-outputs/", help="Directory to write the collated files to")
    parser.add_argument('--include', '-i', nargs='+', default=["daily", "hourly", "minutely", "bouts"], help="Type of result files to collate ('daily', 'hourly', 'minutely', 'bouts')")
    parser.add_argument('--schema-policy', choices=["union", "strict"], default="union", help="How to handle differing CSV columns")
    args = parser.parse_args()

    collate_outputs(
        results_dir=args.results_dir,
        collated_results_dir=args.output,
        included=args.include,
        schema_policy=args.schema_policy,
    )



if __name__ == '__main__':
    main()
