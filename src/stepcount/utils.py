from __future__ import annotations

import pathlib
import json
import hashlib
import warnings
from os import PathLike
from typing import Any, Dict, Literal, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import actipy

from stepcount import _status
from stepcount._types import NDArray


PandasObject = TypeVar("PandasObject", "pd.Series[Any]", "pd.DataFrame")


def read(
    filepath: str,
    usecols: str = 'time,x,y,z',
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    calibration_stdtol_min: Optional[float] = None,
    sample_rate: Optional[float] = None,
    resample_hz: Optional[Union[Literal['uniform'], int, float, bool]] = 'uniform',
    start_first_complete_minute: bool = False,
    csv_start_row: Optional[int] = None,
    csv_end_row: Optional[int] = None,
    csv_time_format: Optional[str] = None,
    csv_txyz_idxs: Optional[str] = None,
    verbose: bool = True,
    include_wear_stats: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read and preprocess activity data from a file.

    This function reads activity data from various file formats, processes it using the `actipy` library,
    and returns the processed data along with metadata information.

    Parameters:
    - filepath (str): The path to the file containing activity data.
    - usecols (str, optional): A comma-separated string of column names to use when reading CSV files.
      Default is 'time,x,y,z'.
    - start_first_complete_minute (bool, optional): Whether to start data from the first complete minute.
      Uses 1 second tolerance. Default is False.
    - calibration_stdtol_min (float, optional): The minimum standard deviation tolerance for detecting stationary periods for calibration. If None,
      no minimum is applied. Default is None.
    - resample_hz (str, optional): The resampling frequency for the data. If 'uniform', it will use `sample_rate`
      and resample to ensure it is evenly spaced. Default is 'uniform'.
    - sample_rate (float, optional): The sample rate of the data. If None, it will be inferred. Default is None.
    - csv_start_row (int, optional): The file row number where the header is located (0-indexed).
      Rows before this are skipped. Only applies to CSV files. Default is None (header at row 0).
    - csv_end_row (int, optional): The file row number to stop reading at, inclusive (0-indexed).
      Only applies to CSV files. Default is None (read to the end).
    - csv_time_format (str, optional): Format string for parsing the time column (e.g., '%Y-%m-%d %H:%M:%S.%f').
      Only applies to CSV files. Default is None (auto-detect).
    - csv_txyz_idxs (str, optional): Column indices for time,x,y,z as comma-separated string (0-indexed, e.g., '0,1,2,3').
      Overrides usecols for CSV files. Default is None (use usecols/csv_txyz).
    - verbose (bool, optional): If True, enables verbose output during processing. Default is True.
    - include_wear_stats (bool, optional): Whether to add whole-record wear statistics to the returned metadata.
      Default is True.

    Returns:
    - tuple: A tuple containing:
        - data (pd.DataFrame): The processed activity data.
        - info (dict): A dictionary containing metadata information about the data.

    Raises:
    - ValueError: If the file format is unknown or unsupported.

    Example:
        data, info = read('activity_data.csv')
    """

    p = pathlib.Path(filepath)
    fsize = round(p.stat().st_size / (1024 * 1024), 1)
    ftype = p.suffix.lower()
    if ftype in (".gz", ".xz", ".lzma", ".bz2", ".zip"):  # if file is compressed, check the next extension
        ftype = pathlib.Path(p.stem).suffix.lower()

    if ftype in (".csv", ".pkl"):

        if ftype == ".csv":
            # Determine column names: either from indices or from usecols
            if csv_txyz_idxs is not None:
                # Parse and validate indices
                try:
                    tidx, xidx, yidx, zidx = map(int, csv_txyz_idxs.split(','))
                except ValueError:
                    raise ValueError(f"csv_txyz_idxs must be 4 comma-separated integers, got: '{csv_txyz_idxs}'")
                if any(i < 0 for i in [tidx, xidx, yidx, zidx]):
                    raise ValueError(f"csv_txyz_idxs must be non-negative integers, got: '{csv_txyz_idxs}'")
                # Read header to get column names at those indices
                # Skip csv_start_row rows to reach the actual header row
                header_kwargs: Dict[str, Any] = {"nrows": 0}
                if csv_start_row is not None:
                    header_kwargs["skiprows"] = csv_start_row
                header = pd.read_csv(filepath, **header_kwargs).columns.tolist()
                max_idx = max(tidx, xidx, yidx, zidx)
                if max_idx >= len(header):
                    raise ValueError(f"Column index {max_idx} out of range. CSV has {len(header)} columns.")
                tcol, xcol, ycol, zcol = header[tidx], header[xidx], header[yidx], header[zidx]
            else:
                tcol, xcol, ycol, zcol = usecols.split(',')

            # Validate csv_start_row and csv_end_row
            if csv_start_row is not None and csv_end_row is not None:
                if csv_end_row < csv_start_row:
                    raise ValueError(f"csv_end_row ({csv_end_row}) must be >= csv_start_row ({csv_start_row})")

            # skiprows: skip rows before the header if csv_start_row is specified
            # csv_start_row is 0-indexed file row where the header is located
            # skiprows=N skips rows 0 to N-1, making row N the header
            skiprows = csv_start_row

            # nrows: number of data rows to read
            # csv_end_row is file row to stop at (inclusive, 0-indexed)
            # Data rows are from (csv_start_row + 1) to csv_end_row
            if csv_end_row is None:
                nrows = None
            elif csv_start_row is None:
                nrows = csv_end_row  # rows 1 to csv_end_row = csv_end_row rows
            else:
                nrows = csv_end_row - csv_start_row  # rows (csv_start_row+1) to csv_end_row

            # Common read_csv kwargs
            read_kwargs: Dict[str, Any] = dict(
                usecols=[tcol, xcol, ycol, zcol],
                dtype={xcol: 'f4', ycol: 'f4', zcol: 'f4'},
                skiprows=skiprows,
                nrows=nrows,
            )

            if csv_time_format is None:
                # Auto-detect datetime format
                read_kwargs['parse_dates'] = [tcol]
                read_kwargs['index_col'] = tcol
                data = pd.read_csv(filepath, **read_kwargs)
            else:
                # Use specified datetime format
                data = pd.read_csv(filepath, **read_kwargs)
                data[tcol] = pd.to_datetime(data[tcol], format=csv_time_format)
                data = data.set_index(tcol)

            # rename to standard names
            data = data.rename(columns={xcol: 'x', ycol: 'y', zcol: 'z'})
            data.index.name = 'time'

        elif ftype == ".pkl":
            data = pd.read_pickle(filepath)

        else:
            raise ValueError(f"Unknown file format: {ftype}")

        if sample_rate is None or sample_rate == 0:
            freq = infer_freq(data.index)
            sample_rate = int(np.round(pd.Timedelta('1s') / freq))

        # Quick fix: Drop duplicate indices. TODO: Maybe should be handled by actipy.
        data = data[~data.index.duplicated(keep='first')]

        data, info = actipy.process(
            data, sample_rate,
            lowpass_hz=None,
            calibrate_gravity=True,
            calibrate_gravity_kwargs={'stdtol_min': calibration_stdtol_min},
            detect_nonwear=True,
            resample_hz=resample_hz,
            start_first_complete_minute=start_first_complete_minute,
            verbose=verbose,
        )

        info.update({
            "Filename": filepath,
            "Device": ftype,
            "Filesize(MB)": fsize,
            "SampleRate": sample_rate,
        })

    elif ftype in (".cwa", ".gt3x", ".bin"):

        if csv_start_row is not None or csv_end_row is not None or csv_time_format is not None or csv_txyz_idxs is not None:
            warnings.warn("--csv-* options are only supported for CSV files. Ignoring.")

        data, info = actipy.read_device(
            filepath,
            lowpass_hz=None,
            calibrate_gravity=True,
            calibrate_gravity_kwargs={'stdtol_min': calibration_stdtol_min},
            detect_nonwear=True,
            resample_hz=resample_hz,
            start_first_complete_minute=start_first_complete_minute,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown file format: {ftype}")

    if 'ResampleRate' not in info:
        info['ResampleRate'] = info['SampleRate']

    # Trim the data if start/end times are specified
    if start_time is not None:
        data = cast(Any, data).loc[cast(Any, start_time):]
    if end_time is not None:
        data = cast(Any, data).loc[:cast(Any, end_time)]

    if include_wear_stats:
        with _status.timed_status("Calculating wear statistics", verbose):
            info.update(calculate_wear_stats(data))

    return data, info


def calculate_wear_stats(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate wear time and related information from raw accelerometer data.

    Parameters:
    - data (pd.DataFrame): A pandas DataFrame of raw accelerometer data with columns 'x', 'y', 'z' and a DatetimeIndex.

    Returns:
    - dict: A dictionary containing various wear time stats.

    Example:
        info = calculate_wear_stats(data)
    """

    if len(data) == 0:
        return _empty_wear_stats()

    wear, dt = _wear_inputs(data)
    return _calculate_wear_stats(data, wear, dt)


def calculate_daily_wear_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily wear time statistics from raw accelerometer data.

    Parameters:
    - data (pd.DataFrame): A pandas DataFrame of raw accelerometer data with columns 'x', 'y', 'z' and a DatetimeIndex.

    Returns:
    - pd.DataFrame: A DataFrame with dates as index and daily wear statistics as columns:
        - 'WearTime(hours)': Total wear time in hours

    Example:
        daily_wear_stats = calculate_daily_wear_stats(data)
    """

    if len(data) == 0:
        return pd.DataFrame()

    wear, dt = _wear_inputs(data)
    return _calculate_daily_wear_stats(wear, dt)


def _summarize_wear(
    data: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame, bool]:
    """Calculate wear statistics and the CLI's xyz-only no-data status."""
    if len(data) == 0:
        return _empty_wear_stats(), pd.DataFrame(), True

    missing = data.isna()
    wear = ~missing.any(axis=1)
    no_data = bool((missing['x'] | missing['y'] | missing['z']).all())
    dt = infer_freq(data.index).total_seconds()
    return (
        _calculate_wear_stats(data, wear, dt),
        _calculate_daily_wear_stats(wear, dt),
        no_data,
    )


def _wear_inputs(data: pd.DataFrame) -> Tuple[pd.Series[Any], float]:
    wear = ~data.isna().any(axis=1)
    dt = infer_freq(data.index).total_seconds()
    return wear, dt


def _empty_wear_stats() -> Dict[str, Any]:
    return {
        'StartTime': None,
        'EndTime': None,
        'WearStartTime': None,
        'WearEndTime': None,
        'WearTime(days)': 0.0,
        'NonwearTime(days)': 0.0,
        'Covers24hOK': 0,
    }


def _calculate_wear_stats(
    data: pd.DataFrame,
    wear: pd.Series[Any],
    dt: float,
) -> Dict[str, Any]:
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    datetime_index = cast(pd.DatetimeIndex, data.index)

    wear_start: Any
    if data.iloc[0].notna().any():
        wear_start = datetime_index[0]
    else:
        wear_start = data.first_valid_index()
    wear_end: Any
    if data.iloc[-1].notna().any():
        wear_end = datetime_index[-1]
    else:
        wear_end = data.last_valid_index()

    wear_start_time = None
    if wear_start is not None:
        wear_start_time = pd.Timestamp(cast(Any, wear_start)).strftime(TIME_FORMAT)
    wear_end_time = None
    if wear_end is not None:
        wear_end_time = pd.Timestamp(cast(Any, wear_end)).strftime(TIME_FORMAT)

    nonwear_duration = (len(wear) - wear.sum()) * dt / (60 * 60 * 24)
    wear_duration = len(data) * dt / (60 * 60 * 24) - nonwear_duration
    coverage = wear.groupby(datetime_index.hour).mean()

    return {
        'StartTime': datetime_index[0].strftime(TIME_FORMAT),
        'EndTime': datetime_index[-1].strftime(TIME_FORMAT),
        'WearStartTime': wear_start_time,
        'WearEndTime': wear_end_time,
        'WearTime(days)': wear_duration,
        'NonwearTime(days)': nonwear_duration,
        'Covers24hOK': int(len(coverage) == 24 and coverage.min() >= 0.01),
    }


def _calculate_daily_wear_stats(
    wear: pd.Series[Any],
    dt: float,
) -> pd.DataFrame:
    datetime_index = cast(pd.DatetimeIndex, wear.index)
    dates = datetime_index.tz_localize(None).normalize()
    wear_samples = wear.groupby(dates).sum()
    wear_hours = (wear_samples * dt / 3600).round(2)
    daily_stats = wear_hours.to_frame('WearTime(hours)')
    daily_stats.index.name = 'Date'
    return daily_stats


def flag_wear_below_days(
    x: PandasObject,
    min_wear: str = '12H'
) -> PandasObject:
    """
    Set days containing less than the specified minimum wear time (`min_wear`) to NaN.

    Parameters:
    - x (pd.Series or pd.DataFrame): A pandas Series or DataFrame with a DatetimeIndex representing time series data.
    - min_wear (str): A string representing the minimum wear time required per day (e.g., '8H' for 8 hours).

    Returns:
    - pd.Series or pd.DataFrame: A pandas Series or DataFrame with days having less than `min_wear` of valid data set to NaN.

    Example:
        # Exclude days with less than 12 hours of valid data
        series = exclude_wear_below_days(series, min_wear='12H')
    """
    if len(x) == 0:
        print("No data to exclude")
        return x

    min_wear_delta = pd.Timedelta(min_wear)
    dt = infer_freq(x.index)
    not_na = x.notna()
    if isinstance(not_na, pd.DataFrame):
        ok = not_na.all(axis=1)
    else:
        ok = not_na
    datetime_index = cast(pd.DatetimeIndex, x.index)
    ok = (
        ok.groupby(datetime_index.date)
        .sum() * dt
        >= min_wear_delta
    )
    # keep ok days, rest is set to NaN
    x = x.copy()  # make a copy to avoid modifying the original data
    x[np.isin(datetime_index.date, ok[~ok].index)] = np.nan
    return x


def drop_first_last_days(
    x: PandasObject,
    first_or_last: Literal['first', 'last', 'both'] = 'both'
) -> PandasObject:
    """
    Drop the first day, last day, or both from a time series.

    Parameters:
    - x (pd.Series or pd.DataFrame): A pandas Series or DataFrame with a DatetimeIndex representing time series data.
    - first_or_last (str, optional): A string indicating which days to drop. Options are 'first', 'last', or 'both'. Default is 'both'.

    Returns:
    - pd.Series or pd.DataFrame: A pandas Series or DataFrame with the values of the specified days dropped.

    Example:
        # Drop the first day from the series
        series = drop_first_last_days(series, first_or_last='first')
    """
    if len(x) == 0:
        print("No data to drop")
        return x

    datetime_index = cast(pd.DatetimeIndex, x.index)
    if first_or_last == 'first':
        x = x[datetime_index.date != datetime_index.date[0]]
    elif first_or_last == 'last':
        x = x[datetime_index.date != datetime_index.date[-1]]
    elif first_or_last == 'both':
        x = x[(datetime_index.date != datetime_index.date[0]) & (datetime_index.date != datetime_index.date[-1])]
    return x


def impute_missing(
    data: PandasObject,
    extrapolate: bool = True,
    skip_full_missing_days: bool = True
) -> PandasObject:
    """
    Impute missing values in the given DataFrame using a multi-step approach.

    This function fills in missing values in a time series DataFrame by applying a series of 
    imputation strategies. It can also extrapolate data to ensure full 24-hour coverage and 
    optionally skip days that are entirely missing.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the time series data to be imputed. 
      The index should be a datetime index.
    - extrapolate (bool, optional): Whether to extrapolate data beyond the start and end times 
      to ensure full 24-hour coverage. Defaults to True.
    - skip_full_missing_days (bool, optional): Whether to skip days that have all missing values. 
      Defaults to True.

    Returns:
    - pd.DataFrame: The DataFrame with missing values imputed.

    Notes:
    - The imputation process involves three steps in the following order:
        1. Imputation using the same day of the week.
        2. Imputation within weekdays or weekends.
        3. Imputation using all other days.
    - The granularity of the imputation is 5 minutes. 
    - If `extrapolate` is True, the function will attempt to fill in data beyond the start and end times, so that 
      the first and last day have full 24-hour coverage.
    - If `skip_full_missing_days` is True, days with all missing values will be excluded from the imputation process.
    """
    def impute(frame: PandasObject) -> PandasObject:
        datetime_index = cast(pd.DatetimeIndex, frame.index)
        weekday = datetime_index.weekday
        hour = datetime_index.hour
        slot = datetime_index.minute // 5
        groupings = (
            [weekday, hour, slot],
            [weekday >= 5, hour, slot],
            [hour, slot],
        )
        result = frame.copy()
        if isinstance(result, pd.DataFrame):
            numeric = result.select_dtypes(include='number').copy()
        else:
            numeric = result

        for keys in groupings:
            missing = numeric.isna()
            if isinstance(numeric, pd.DataFrame):
                active = missing.any(axis=0) & ~missing.all(axis=0)
                if not active.any():
                    break
                columns = active.index[active].tolist()
                values = numeric.loc[:, columns]
                means = values.groupby(keys).transform('mean')
                numeric.loc[:, columns] = values.fillna(means)
            else:
                if not missing.any() or missing.all():
                    break
                means = numeric.groupby(keys).transform('mean')
                numeric = numeric.fillna(means)

        if isinstance(result, pd.DataFrame):
            result.loc[:, numeric.columns] = numeric
        else:
            result = numeric
        return result

    if skip_full_missing_days:
        # Compute dates where ALL values are NaN (across all columns if DataFrame)
        # Handle both Series (1D) and DataFrame (2D) cases
        if isinstance(data, pd.DataFrame):
            # For DataFrame: check if all columns are NaN per row, then group by date
            row_all_na = data.isna().all(axis=1)
        else:
            # For Series: just check if each value is NaN
            row_all_na = data.isna()
        datetime_index = cast(pd.DatetimeIndex, data.index)
        full_na_flags = row_all_na.groupby(datetime_index.date).all()
        full_na_dates = full_na_flags[full_na_flags].index

    if extrapolate:  # extrapolate beyond start/end times to have full 24h
        freq = infer_freq(data.index)
        if pd.isna(freq):
            warnings.warn("Cannot infer frequency, using 1s")
            freq = pd.Timedelta('1s')
        offset = to_offset(freq)
        if offset is None:
            raise ValueError(f"Cannot convert frequency to an offset: {freq}")
        reindex_kwargs: Dict[str, Any] = {
            "method": "nearest",
            "tolerance": pd.Timedelta('1m'),
            "limit": 1,
        }
        datetime_index = cast(pd.DatetimeIndex, data.index)
        data = data.reindex(
            pd.date_range(
                # Note that at exactly 00:00:00, the floor('D') and ceil('D') will be the same
                datetime_index[0].floor('D'),
                datetime_index[-1].ceil('D'),
                freq=offset,
                inclusive='left',
                name='time',
            ),
            **reindex_kwargs,
        )

    data = impute(data)

    if skip_full_missing_days:
        # Restore dates that were intentionally excluded from imputation.
        mask = np.isin(cast(pd.DatetimeIndex, data.index).date, full_na_dates)
        data.loc[mask] = np.nan

    return data


def impute_days(
    x: pd.Series[Any],
    method: Literal['mean', 'median'] = 'mean'
) -> pd.Series[Any]:
    """
    Impute missing values for data with a daily resolution.

    The imputation is performed in three steps: first by the same day of the
    week, then by weekdays or weekends, and finally by the entire series.

    Parameters:
    - x (pd.Series): A pandas Series at a daily resolution level.
    - method (str, optional): The imputation method to use. Options are 'mean' or 'median'. 
      Defaults to 'mean'.

    Returns:
    - pd.Series: A pandas Series with missing days imputed.

    Raises:
    - ValueError: If an unknown imputation method is specified.

    Notes:
    - The imputation process involves three steps in the following order:
        1. Imputation using the same day of the week.
        2. Imputation within weekdays or weekends.
        3. Imputation using the entire series.
    - If the entire Series is missing, it will be returned as is.
    """
    if x.isna().all():
        return x

    def fillna(values: pd.Series[Any]) -> pd.Series[Any]:
        if method == 'mean':
            return values.fillna(values.mean())
        elif method == 'median':
            return values.fillna(values.median())
        else:
            raise ValueError(f"Unknown method: {method}")

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        datetime_index = cast(pd.DatetimeIndex, x.index)
        return (
            x
            .groupby(datetime_index.weekday).transform(fillna)
            .groupby(datetime_index.weekday >= 5).transform(fillna)
            .transform(fillna)
        )


def infer_freq(t: pd.Index) -> pd.Timedelta:
    """ Like pd.infer_freq but more forgiving """
    tdiff = t.to_series().diff()
    q1, q3 = tdiff.quantile([0.25, 0.75])
    tdiff = tdiff[(q1 <= tdiff) & (tdiff <= q3)]
    return pd.Timedelta(cast(Any, tdiff.mean()))


def resolve_path(path: Union[str, PathLike[str]]) -> Tuple[pathlib.Path, str, str]:
    """ Return parent folder, file name and file extension """
    p = pathlib.Path(path)
    extension = p.suffixes[0]
    filename = p.name.rsplit(extension)[0]
    dirname = p.parent
    return dirname, filename, extension


def md5(fname: Union[str, PathLike[str]]) -> str:
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isnull(obj):  # handles pandas NAType
            return np.nan
        return json.JSONEncoder.default(self, obj)


def nanint(x: Union[float, np.floating[Any]]) -> Union[int, float]:
    if np.isnan(x):
        return float(x)
    return int(x)
