import pathlib
import json
import hashlib
import warnings
from typing import Union
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import actipy


def read(
    filepath: str,
    usecols: str = 'time,x,y,z',
    resample_hz: str = 'uniform',
    sample_rate: float = None,
    verbose: bool = True
):
    """
    Read and preprocess activity data from a file.

    This function reads activity data from various file formats, processes it using the `actipy` library, 
    and returns the processed data along with metadata information.

    Parameters:
    - filepath (str): The path to the file containing activity data.
    - usecols (str, optional): A comma-separated string of column names to use when reading CSV files. 
      Default is 'time,x,y,z'.
    - resample_hz (str, optional): The resampling frequency for the data. If 'uniform', it will use `sample_rate`
      and resample to ensure it is evenly spaced. Default is 'uniform'.
    - sample_rate (float, optional): The sample rate of the data. If None, it will be inferred. Default is None.
    - verbose (bool, optional): If True, enables verbose output during processing. Default is True.

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
            tcol, xcol, ycol, zcol = usecols.split(',')
            data = pd.read_csv(
                filepath,
                usecols=[tcol, xcol, ycol, zcol],
                parse_dates=[tcol],
                index_col=tcol,
                dtype={xcol: 'f4', ycol: 'f4', zcol: 'f4'},
            )
            # rename to standard names
            data = data.rename(columns={xcol: 'x', ycol: 'y', zcol: 'z'})
            data.index.name = 'time'

        elif ftype == ".pkl":
            data = pd.read_pickle(filepath)

        else:
            raise ValueError(f"Unknown file format: {ftype}")

        if sample_rate in (None, False):
            freq = infer_freq(data.index)
            sample_rate = int(np.round(pd.Timedelta('1s') / freq))

        # Quick fix: Drop duplicate indices. TODO: Maybe should be handled by actipy.
        data = data[~data.index.duplicated(keep='first')]

        data, info = actipy.process(
            data, sample_rate,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=resample_hz,
            verbose=verbose,
        )

        info.update({
            "Filename": filepath,
            "Device": ftype,
            "Filesize(MB)": fsize,
            "SampleRate": sample_rate,
            "StartTime": data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
            "EndTime": data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        })

    elif ftype in (".cwa", ".gt3x", ".bin"):

        data, info = actipy.read_device(
            filepath,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=resample_hz,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown file format: {ftype}")

    if 'ResampleRate' not in info:
        info['ResampleRate'] = info['SampleRate']

    return data, info


def exclude_wear_below_days(
    x: Union[pd.Series, pd.DataFrame],
    min_wear: str = '12H'
):
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

    min_wear = pd.Timedelta(min_wear)
    dt = infer_freq(x.index)
    ok = x.notna()
    if isinstance(ok, pd.DataFrame):
        ok = ok.all(axis=1)
    ok = (
        ok
        .groupby(x.index.date)
        .sum() * dt
        >= min_wear
    )
    # keep ok days, rest is set to NaN
    x = x.copy()  # make a copy to avoid modifying the original data
    x[np.isin(x.index.date, ok[~ok].index)] = np.nan
    return x


def exclude_first_last_days(
    x: Union[pd.Series, pd.DataFrame],
    first_or_last='both'
):
    """
    Set the values of the first day, last day, or both to NaN in a time series.

    Parameters:
    - x (pd.Series or pd.DataFrame): A pandas Series or DataFrame with a DatetimeIndex representing time series data.
    - first_or_last (str, optional): A string indicating which days to exclude. Options are 'first', 'last', or 'both'. Default is 'both'.

    Returns:
    - pd.Series or pd.DataFrame: A pandas Series or DataFrame with the values of the specified days set to NaN.

    Example:
        # Exclude the first day from the series
        series = exclude_first_last_days(series, first_or_last='first')
    """
    if len(x) == 0:
        print("No data to exclude")
        return x

    if first_or_last == 'first':
        x[x.index.date == x.index.date[0]] = np.nan
    elif first_or_last == 'last':
        x[x.index.date == x.index.date[-1]] = np.nan
    elif first_or_last == 'both':
        x[(x.index.date == x.index.date[0]) | (x.index.date == x.index.date[-1])] = np.nan
    return x


def impute_missing(
    data: pd.DataFrame,
    extrapolate=True,
    skip_full_missing_days=True
):
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
    def fillna(subframe):
        if isinstance(subframe, pd.Series):
            x = subframe.to_numpy()
            nan = np.isnan(x)
            nanlen = len(x[nan])
            if 0 < nanlen < len(x):  # check x contains a NaN and is not all NaN
                x[nan] = np.nanmean(x)
                return x  # will be cast back to a Series automatically
            else:
                return subframe

    def impute(data):
        return (
            data
            # first attempt imputation using same day of week
            .groupby([data.index.weekday, data.index.hour, data.index.minute // 5])
            .transform(fillna)
            # then try within weekday/weekend
            .groupby([data.index.weekday >= 5, data.index.hour, data.index.minute // 5])
            .transform(fillna)
            # finally, use all other days
            .groupby([data.index.hour, data.index.minute // 5])
            .transform(fillna)
        )

    if skip_full_missing_days:
        na_dates = data.isna().groupby(data.index.date).all()

    if extrapolate:  # extrapolate beyond start/end times to have full 24h
        freq = infer_freq(data.index)
        if pd.isna(freq):
            warnings.warn("Cannot infer frequency, using 1s")
            freq = pd.Timedelta('1s')
        freq = to_offset(freq)
        data = data.reindex(
            pd.date_range(
                # Note that at exactly 00:00:00, the floor('D') and ceil('D') will be the same
                data.index[0].floor('D'),
                data.index[-1].ceil('D'),
                freq=freq,
                inclusive='left',
                name='time',
            ),
            method='nearest',
            tolerance=pd.Timedelta('1m'),
            limit=1)

    data = impute(data)

    if skip_full_missing_days:
        data.mask(np.isin(data.index.date, na_dates[na_dates].index), inplace=True)

    return data


def impute_days(
    x: pd.Series,
    method='mean'
):
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

    def fillna(x):
        if method == 'mean':
            return x.fillna(x.mean())
        elif method == 'median':
            return x.fillna(x.median())
        else:
            raise ValueError(f"Unknown method: {method}")

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        return (
            x
            .groupby(x.index.weekday).transform(fillna)
            .groupby(x.index.weekday >= 5).transform(fillna)
            .transform(fillna)
        )


def infer_freq(t):
    """ Like pd.infer_freq but more forgiving """
    tdiff = t.to_series().diff()
    q1, q3 = tdiff.quantile([0.25, 0.75])
    tdiff = tdiff[(q1 <= tdiff) & (tdiff <= q3)]
    freq = tdiff.mean()
    freq = pd.Timedelta(freq)
    return freq


def resolve_path(path):
    """ Return parent folder, file name and file extension """
    p = pathlib.Path(path)
    extension = p.suffixes[0]
    filename = p.name.rsplit(extension)[0]
    dirname = p.parent
    return dirname, filename, extension


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isnull(obj):  # handles pandas NAType
            return np.nan
        return json.JSONEncoder.default(self, obj)


def nanint(x):
    if np.isnan(x):
        return x
    return int(x)
