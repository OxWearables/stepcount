from __future__ import annotations

import warnings
from copy import deepcopy
from collections import defaultdict, Counter
from os import PathLike
from typing import Any, Callable, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple, Union, cast, overload

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.special import softmax
from scipy import stats
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit
from imblearn.ensemble import BalancedRandomForestClassifier
from stepcount import hmm_utils
from stepcount import features
from stepcount import sslmodel
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from stepcount._types import FeatureDict, NDArray, Numeric, PeakParams


CVIndices = Sequence[Any]
ScoreSummary = Dict[str, Dict[str, Numeric]]


class StepCounter:

    def __init__(
        self,
        window_sec: float = 5,
        sample_rate: float = 100,
        steptol: int = 3,
        pnr: float = 1.0,
        lowpass_hz: float = 5,
        cv: int = 5,
        wd_type: Literal['ssl', 'rf'] = 'ssl',
        wd_params: Optional[Dict[str, Any]] = None,
        n_jobs: int = -1,
        verbose: bool = False
    ) -> None:
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.steptol = steptol
        self.pnr = pnr
        self.lowpass_hz = lowpass_hz
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

        wd_params = wd_params or dict()

        if wd_type == 'ssl':
            wd_defaults: Dict[str, Any] = {
                'device': 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu',
                'batch_size': 100,
                'verbose': verbose
            }
            wd: Any = WalkDetectorSSL
            # Match the SSL model's training configuration.
            self.window_sec = 10
            self.sample_rate = 30
            # PyTorch manages parallelism, and GPU inference must remain single-process.
            self.n_jobs = 1
        else:
            wd_defaults = {
                'sample_rate': sample_rate,
                'n_jobs': n_jobs,
                'verbose': verbose,
                'pnr': pnr,
                'cv': cv,
            }
            wd = WalkDetectorRF

        for key, value in wd_defaults.items():
            if key not in wd_params:
                wd_params[key] = value
        self.wd_params = wd_params
        self.wd: Union[WalkDetectorRF, WalkDetectorSSL] = wd(**self.wd_params)

        self.window_len = int(np.ceil(self.window_sec * self.sample_rate))
        self.find_peaks_params: Optional[PeakParams] = None
        self.cv_scores: Optional[Dict[str, Any]] = None
        self.cv_results: Dict[str, Any] = {}

    def fit(
        self,
        X: NDArray,
        Y: NDArray,
        groups: Optional[NDArray] = None,
    ) -> StepCounter:

        W = np.zeros_like(Y)
        W[Y >= self.steptol] = 1
        Y = Y.copy()
        Y[Y < self.steptol] = 0

        if self.verbose:
            print("Running cross_val_predict...")
        self.wd.n_jobs = 1
        Wp, cv_test_idxs = cvp(
            self.wd, X, W, groups=groups,
            fit_predict_groups=True,
            n_splits=self.cv,
            n_jobs=self.n_jobs,
            return_indices=True,
        )
        self.wd.n_jobs = self.n_jobs

        whr_walk_pred = Wp == 1

        if self.verbose:
            print("Fitting walk detector...")
        self.wd.fit(X, W, groups=groups)

        Xw, Yw = X[whr_walk_pred], Y[whr_walk_pred]
        Vw = toV(Xw, self.sample_rate, self.lowpass_hz)

        sample_weight = calc_sample_weight(W, self.pnr)
        sample_weight_w = sample_weight[whr_walk_pred]

        def mae(x: NDArray) -> float:
            Ywp = batch_count_peaks_from_V(Vw, self.sample_rate, to_params(x))
            err = metrics.mean_absolute_error(Yw, Ywp, sample_weight=sample_weight_w)
            return float(err)

        def to_params(x: NDArray) -> PeakParams:
            params: PeakParams = {
                "distance": float(x[0]),
                "prominence": float(x[1]),
            }
            return params

        if self.verbose:
            print("Tuning step counter...")
        res = minimize(
            mae,
            x0=[.5, .5],
            bounds=[
                (.2, 2),  # 0.2s to 2s (4Hz - 0.5Hz)
                (.15, 1),  # 0.15g to 1g
            ],
            method='Nelder-Mead'
        )

        self.find_peaks_params = to_params(res.x)

        _, wd_scores = get_cv_scores(
            W, Wp, cv_test_idxs,
            sample_weight=sample_weight,
            scorer_type='classif'
        )

        Yp = np.zeros_like(Y)
        Yp[whr_walk_pred] = batch_count_peaks_from_V(Vw, self.sample_rate, self.find_peaks_params)
        _, sc_scores = get_cv_scores(
            Y, Yp, cv_test_idxs,
            sample_weight=sample_weight,
            scorer_type='regress'
        )

        whr_walk_true = W == 1
        walk_true_idxs = np.flatnonzero(whr_walk_true)
        _, sc_scores_walk = get_cv_scores(
            Y[whr_walk_true], Yp[whr_walk_true],
            [np.flatnonzero(np.isin(walk_true_idxs, idxs)) for idxs in cv_test_idxs],
            sample_weight=sample_weight[whr_walk_true],
            scorer_type='regress'
        )

        self.cv_results = {
            'test_indices': cv_test_idxs,
            'groups': groups,
            'walk_detector': {
                'scores': wd_scores,
                'y_true': W,
                'y_pred': Wp,
            },
            'step_counter': {
                'scores': sc_scores,
                'scores_walk': sc_scores_walk,
                'y_true': Y,
                'y_pred': Yp,
            },
        }

        return self

    def predict(
        self,
        X: NDArray,
        groups: Optional[NDArray] = None,
        return_walk: bool = False,
        return_step_times: bool = False,
    ) -> Optional[Tuple[NDArray, Optional[NDArray], Optional[NDArray]]]:

        if self.find_peaks_params is None:
            print("Model not yet trained. Call .fit() first.")
            return None

        invalid = np.isnan(X).any(axis=tuple(range(1, X.ndim)))
        ok = np.flatnonzero(~invalid)
        if groups is not None and len(groups) != len(X):
            raise ValueError("groups must have the same length as X")

        W_ = self.wd.predict(X, groups, indices=ok).astype('bool')
        Y_ = np.zeros_like(W_, dtype='float')
        Z_ = np.full_like(W_, fill_value=None, dtype=np.ndarray)

        w_ = np.flatnonzero(W_)
        (Y_[w_], Z_[w_]) = batch_count_peaks(
            X[ok[w_]],
            self.sample_rate,
            self.lowpass_hz,
            self.find_peaks_params,
            return_peaks=True
        )
        below_threshold = Y_ < self.steptol
        Y_[below_threshold] = 0
        Z_[below_threshold] = None

        Y = np.full(len(X), fill_value=np.nan)
        Y[ok] = Y_

        W = None
        if return_walk:
            W = np.full(len(X), fill_value=np.nan)
            W[ok] = W_

        Z = None
        if return_step_times:
            Z = np.full(len(X), fill_value=None, dtype=np.ndarray)
            Z[ok] = Z_

        return Y, W, Z

    def predict_from_frame(
        self,
        data: pd.DataFrame,
    ) -> Tuple[pd.Series[Any], pd.Series[Any], pd.Series[Any]]:

        def fn(chunk: pd.DataFrame) -> NDArray:
            """ Process the chunk. Apply padding if length is not enough. """
            n = len(chunk)
            x = chunk[['x', 'y', 'z']].to_numpy()
            if n == self.window_len:
                x = x
            elif n > self.window_len:
                x = x[:self.window_len]
            elif n < self.window_len and n > self.window_len / 2:
                m = self.window_len - n
                x = np.pad(x, ((0, m), (0, 0)), mode='wrap')
            else:
                x = np.full((self.window_len, 3), fill_value=np.nan)
            return x

        X, T = make_windows(data, self.window_sec, fn=fn, return_index=True, verbose=self.verbose)

        prediction = self.predict(
            X,
            return_walk=True,
            return_step_times=True,
        )
        if prediction is None:
            raise RuntimeError("Model not yet trained. Call .fit() first.")
        Y_values, W_values, Z_values = prediction
        if W_values is None or Z_values is None:
            raise RuntimeError("Walk and step-time outputs were not produced")

        Y = pd.Series(Y_values, index=T, name='Steps')
        W = pd.Series(W_values, index=T, name='Walk')

        step_times: list[pd.Timestamp] = []
        for t, z in zip(T, Z_values):
            if z is not None:
                # convert the local window timestamps to global timestamps
                step_times.extend([t + pd.Timedelta(seconds=dt) for dt in z])
        T_steps = pd.Series(step_times, name='time')

        return Y, W, T_steps


def _select_prediction_groups(
    groups: Optional[NDArray],
    source_count: int,
    indices: Optional[NDArray],
) -> Optional[NDArray]:
    """Resolve group labels for selected windows, preferring source alignment."""
    if groups is None:
        return None

    groups = np.asarray(groups)
    selected_count = source_count if indices is None else len(indices)
    if indices is not None and len(groups) == source_count:
        return cast(NDArray, groups[np.asarray(indices)])
    if len(groups) == selected_count:
        return groups
    raise ValueError("groups must align with X or with the selected indices")


class WalkDetectorRF:
    def __init__(
        self,
        sample_rate: float = 100,
        pnr: float = 1.0,
        calib_method: Optional[Literal['balanced_accuracy', 'f1', 'precision', 'recall']] = 'balanced_accuracy',
        precision_tol: float = .9,
        recall_tol: float = .9,
        cv: int = 5,
        clf_params: Optional[Dict[str, Any]] = None,
        hmm_params: Optional[Dict[str, Any]] = None,
        n_jobs: int = -1,
        verbose: bool = False,
    ) -> None:

        self.sample_rate = sample_rate

        self.pnr = pnr
        self.calib_method = calib_method
        self.precision_tol = precision_tol
        self.recall_tol = recall_tol

        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

        clf_params = clf_params or dict()
        hmm_params = hmm_params or dict()

        self.clf: Any = BalancedRandomForestClassifier(
            n_estimators=clf_params.get('n_estimators', 1000),
            replacement=clf_params.get('replacement', True),
            sampling_strategy=clf_params.get('sampling_strategy', 'not minority'),
            random_state=clf_params.get('random_state', 42),
            verbose=0, n_jobs=1,
        )

        self.hmms = hmm_utils.HMMSmoother(**hmm_params)

        self.thresh = 0.5

    def fit(
        self,
        X: NDArray,
        Y: NDArray,
        groups: Optional[NDArray] = None,
    ) -> WalkDetectorRF:

        X_feats = batch_extract_features(X, self.sample_rate, n_jobs=self.n_jobs, verbose=self.verbose)

        whr_ok = ~(np.isnan(X_feats).any(1))
        X_feats = X_feats[whr_ok]
        Y = Y[whr_ok]
        filtered_groups = groups[whr_ok] if groups is not None else None

        Yp = cvp(
            self.clf, X_feats, Y, filtered_groups,
            method='predict_proba',
            fit_predict_groups=False,
            n_splits=self.cv,
            n_jobs=self.n_jobs,
        )

        self.clf.n_jobs = self.n_jobs
        self.clf.fit(X_feats, Y)
        self.clf.n_jobs = 1

        if self.calib_method is not None:
            calib_ops = calibrate(Yp[:, 1], Y, self.pnr, self.precision_tol, self.recall_tol)

            if self.calib_method == 'balanced_accuracy':
                self.thresh = calib_ops['best_balanced_accuracy']['thresh']
                Ypp = calib_ops['best_balanced_accuracy']['predicted']

            elif self.calib_method in ('f1', 'precision', 'recall'):
                # first optimize f1, then adjust for precision or recall if needed
                self.thresh = calib_ops['best_f1']['thresh']
                Ypp = calib_ops['best_f1']['predicted']

                if self.calib_method == 'precision':
                    if calib_ops['best_f1']['precision'] < self.precision_tol:
                        self.thresh = calib_ops['best_precision']['thresh']
                        Ypp = calib_ops['best_precision']['predicted']

                if self.calib_method == 'recall':
                    if calib_ops['best_f1']['recall'] < self.recall_tol:
                        self.thresh = calib_ops['best_recall']['thresh']
                        Ypp = calib_ops['best_recall']['predicted']

            else:
                raise ValueError(f"Unrecognized {self.calib_method=}")

        else:
            Ypp = Yp

        self.hmms.fit(Ypp, Y, groups=filtered_groups)

        return self

    def predict(
        self,
        X: NDArray,
        groups: Optional[NDArray] = None,
        indices: Optional[NDArray] = None,
    ) -> NDArray:

        sample_count = len(X) if indices is None else len(indices)
        if sample_count == 0:
            warnings.warn("No data to predict", stacklevel=2)
            return np.array([], dtype='int')

        selected_groups = _select_prediction_groups(groups, len(X), indices)
        selected_X = X if indices is None else X[np.asarray(indices)]
        W = np.zeros(sample_count, dtype='int')  # nan defaults to non-walk
        X_feats = batch_extract_features(selected_X, self.sample_rate, n_jobs=self.n_jobs, verbose=self.verbose)
        ok = ~(np.isnan(X_feats).any(1))
        if ok.any():
            W[ok] = (self.clf.predict_proba(X_feats[ok])[:, 1] > self.thresh).astype('int')
        W = self.hmms.predict(W, groups=selected_groups)

        return W


class WalkDetectorSSL:
    def __init__(
        self,
        device: sslmodel.Device = 'cpu',
        batch_size: int = 100,
        weights_path: Union[str, PathLike[str]] = 'state_dict.pt',
        repo_tag: str = 'v1.0.0',
        ssl_repo_path: Optional[Union[str, PathLike[str]]] = None,
        hmm_params: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.weights_path = weights_path
        self.repo_tag = repo_tag
        self.ssl_repo_path = ssl_repo_path
        self.batch_size = batch_size
        self.state_dict: Optional[Mapping[str, Any]] = None

        self.model: Optional[nn.Module] = None
        self.n_jobs = 1

        self.verbose = verbose

        hmm_params = hmm_params or dict()
        self.hmms = hmm_utils.HMMSmoother(**hmm_params)

    def fit(
        self,
        X: NDArray,
        Y: NDArray,
        groups: Optional[NDArray] = None,
    ) -> WalkDetectorSSL:
        sslmodel.verbose = self.verbose

        split_groups = groups if groups is not None else np.arange(len(Y))

        if self.verbose:
            print('Training SSL')

        folds = GroupShuffleSplit(
            1, test_size=0.2, random_state=41
        ).split(X, Y, groups=split_groups)
        train_idx, val_idx = next(folds)

        x_train = X[train_idx]
        x_val = X[val_idx]

        y_train = Y[train_idx]
        y_val = Y[val_idx]

        group_train = groups[train_idx] if groups is not None else None
        group_val = groups[val_idx] if groups is not None else None

        train_dataset = sslmodel.NormalDataset(x_train, y_train, pid=group_train, name="training", augmentation=True)
        val_dataset = sslmodel.NormalDataset(x_val, y_val, pid=group_val, name="validation")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )

        # Weight classes toward the expected 90% non-walking / 10% walking mix.
        c = Counter(y_train)
        notwalk = c[0]
        walk = c[1]
        class_weights = [(walk * 9.0) / notwalk, 1.0]

        model = sslmodel.get_sslnet(tag=self.repo_tag, pretrained=True,
                                    repo_path=getattr(self, 'ssl_repo_path', None))
        model.to(self.device)

        sslmodel.train(model, train_loader, val_loader, self.device, class_weights, weights_path=self.weights_path)
        model.load_state_dict(torch.load(self.weights_path, map_location=self.device))

        if self.verbose:
            print('Training HMM')

        # train HMM with predictions of the validation set
        y_val, y_val_pred, _ = sslmodel.predict(model, val_loader, self.device, output_logits=True)
        y_val_pred_sf = softmax(y_val_pred, axis=1)

        self.hmms.fit(y_val_pred_sf, y_val, groups=group_val)

        # move model to cpu to get a device-less state dict (prevents device conflicts when loading on cpu/gpu later)
        model.to('cpu')
        self.state_dict = model.state_dict()

        return self

    def predict(
        self,
        X: NDArray,
        groups: Optional[NDArray] = None,
        indices: Optional[NDArray] = None,
    ) -> NDArray:
        """Predict windows, optionally selecting them by source-array index.

        With ``indices``, groups may align with either the full source array or
        the selected windows. Full-source alignment takes precedence when both
        lengths are equal.
        """

        sslmodel.verbose = self.verbose

        sample_count = len(X) if indices is None else len(indices)
        if sample_count == 0:
            warnings.warn("No data to predict", stacklevel=2)
            return np.array([], dtype='int')

        selected_groups = _select_prediction_groups(groups, len(X), indices)

        if not hasattr(self, 'model') or self.model is None:
            self.model = self.load_model()

        dataset = sslmodel.InferenceDataset(X, indices=indices)
        dataloader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
            num_workers=0,
        )

        _, y_pred, _ = sslmodel.predict(
            self.model,
            dataloader,
            self.device,
            output_logits=False,
            collect_metadata=False,
        )
        y_pred = self.hmms.predict(y_pred, groups=selected_groups)

        return y_pred

    def load_model(self) -> nn.Module:

        if self.state_dict is None:
            raise RuntimeError("No fitted SSL model state is available")

        model = sslmodel.get_sslnet(tag=self.repo_tag, pretrained=False,
                                    repo_path=getattr(self, 'ssl_repo_path', None))
        model.load_state_dict(self.state_dict)
        model.to(self.device)

        if self.verbose:
            print(f"Using pytorch device: {self.device}")

        return model


@overload
def make_windows(
    data: pd.DataFrame,
    window_sec: float,
    fn: Optional[Callable[[pd.DataFrame], Any]] = None,
    return_index: Literal[False] = False,
    verbose: bool = True,
) -> NDArray:
    ...


@overload
def make_windows(
    data: pd.DataFrame,
    window_sec: float,
    fn: Optional[Callable[[pd.DataFrame], Any]] = None,
    return_index: Literal[True] = True,
    verbose: bool = True,
) -> Tuple[NDArray, pd.DatetimeIndex]:
    ...


def make_windows(
    data: pd.DataFrame,
    window_sec: float,
    fn: Optional[Callable[[pd.DataFrame], Any]] = None,
    return_index: bool = False,
    verbose: bool = True,
) -> Union[NDArray, Tuple[NDArray, pd.DatetimeIndex]]:
    """ Split data into windows """

    if verbose:
        print("Defining segments...")

    transform = fn
    if transform is None:
        def transform(x: pd.DataFrame) -> pd.DataFrame:
            return x

    windows: list[Any] = []
    times: list[Any] = []
    for t, x in data.resample(f"{window_sec}s", origin="start"):
        transformed = transform(x)
        windows.append(transformed)
        times.append(t)

    # Handle potentially inhomogeneous window sizes gracefully
    # (pandas resample can produce windows with different sample counts)
    try:
        X = cast(NDArray, np.stack(windows, axis=0))
    except ValueError:
        # Windows have different shapes - use object array for compatibility
        X = cast(NDArray, np.array(windows, dtype=object))

    if return_index:
        T = pd.DatetimeIndex(times, name=data.index.name)
        return X, T

    return X


@overload
def cvp(
    model: Any, X: Union[NDArray, pd.DataFrame], Y: NDArray,
    groups: Optional[NDArray],
    method: str = 'predict', fit_predict_groups: bool = False,
    return_indices: Literal[False] = False, n_splits: int = 5, n_jobs: int = -1,
) -> NDArray:
    ...


@overload
def cvp(
    model: Any, X: Union[NDArray, pd.DataFrame], Y: NDArray,
    groups: Optional[NDArray],
    method: str = 'predict', fit_predict_groups: bool = False,
    return_indices: Literal[True] = True, n_splits: int = 5, n_jobs: int = -1,
) -> Tuple[NDArray, List[Any]]:
    ...


def cvp(
    model: Any, X: Union[NDArray, pd.DataFrame], Y: NDArray,
    groups: Optional[NDArray],
    method: str = 'predict', fit_predict_groups: bool = False,
    return_indices: bool = False, n_splits: int = 5, n_jobs: int = -1,
) -> Union[NDArray, Tuple[NDArray, List[Any]]]:
    """ Like cross_val_predict with custom tweaks """

    split_groups = groups if groups is not None else np.arange(len(Y))

    if n_splits == -1:
        n_splits = len(np.unique(split_groups))

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    def worker(train_idxs: Any, test_idxs: Any) -> Tuple[NDArray, Any]:
        X_train, Y_train = X[train_idxs], Y[train_idxs]
        X_test = X[test_idxs]
        groups_train = groups[train_idxs] if groups is not None else None
        groups_test = groups[test_idxs] if groups is not None else None

        m = deepcopy(model)
        m.n_jobs = 1

        if fit_predict_groups:
            m.fit(X_train, Y_train, groups=groups_train)
            Y_test_pred = cast(NDArray, getattr(m, method)(X_test, groups=groups_test))
        else:
            m.fit(X_train, Y_train)
            Y_test_pred = cast(NDArray, getattr(m, method)(X_test))

        return Y_test_pred, test_idxs

    results: list[Tuple[NDArray, Any]] = Parallel(n_jobs=n_jobs)(
        delayed(worker)(train_idxs, test_idxs)
        for train_idxs, test_idxs in groupkfold(split_groups, n_splits)
    )

    Y_pred = cast(NDArray, np.concatenate([r[0] for r in results]))
    cv_test_idxs = [r[1] for r in results]

    if return_indices:
        return Y_pred, cv_test_idxs

    return Y_pred


def groupkfold(groups: NDArray, n_splits: int = 5) -> Iterator[Tuple[Any, Any]]:
    """ Like GroupKFold but ordered """

    ord_unq_grps = groups[np.sort(np.unique(groups, return_index=True)[1])]
    folds_unq_grps = np.array_split(ord_unq_grps, n_splits)

    for unq_grps in folds_unq_grps:
        mask = np.isin(groups, unq_grps)
        test_idxs = np.nonzero(mask)
        train_idxs = np.nonzero(~mask)
        yield train_idxs, test_idxs


def get_cv_scores(
    yt: NDArray,
    yp: NDArray,
    cv_test_idxs: CVIndices,
    sample_weight: Optional[NDArray] = None,
    scorer_type: Literal['classif', 'regress'] = 'classif',
) -> Tuple[Dict[str, List[float]], ScoreSummary]:

    classif_scorers = {
        'accuracy': metrics.accuracy_score,
        'f1': lambda yt, yp, sample_weight=None: metrics.f1_score(yt, yp, sample_weight=sample_weight, zero_division=0),
        'precision': lambda yt, yp, sample_weight=None: metrics.precision_score(yt, yp, sample_weight=sample_weight, zero_division=0),
        'recall': lambda yt, yp, sample_weight=None: metrics.recall_score(yt, yp, sample_weight=sample_weight, zero_division=0),
        'balanced_accuracy': lambda yt, yp, sample_weight=None: metrics.balanced_accuracy_score(yt, yp, sample_weight=sample_weight)
    }

    regress_scorers = {
        'mae': lambda yt, yp, sample_weight: metrics.mean_absolute_error(yt, yp, sample_weight=sample_weight),
        'rmse': lambda yt, yp, sample_weight: metrics.mean_squared_error(yt, yp, sample_weight=sample_weight, squared=False),
        'mape': lambda yt, yp, sample_weight: smooth_mean_absolute_percentage_error(yt, yp, sample_weight=sample_weight),
    }

    def smooth_mean_absolute_percentage_error(
        yt: NDArray,
        yp: NDArray,
        sample_weight: Optional[NDArray] = None,
    ) -> float:
        yt, yp = yt.copy(), yp.copy()
        # Avoid undefined MAPE values when the target is zero.
        whr = yt == 0
        yt[whr] += 1
        yp[whr] += 1
        return float(metrics.mean_absolute_percentage_error(yt, yp, sample_weight=sample_weight))

    if scorer_type == 'classif':
        scorers = classif_scorers
    elif scorer_type == 'regress':
        scorers = regress_scorers
    else:
        raise ValueError(f"Unknown {scorer_type=}")

    raw_scores: defaultdict[str, list[float]] = defaultdict(list)

    for idxs in cv_test_idxs:
        yt_, yp_ = yt[idxs], yp[idxs]
        sample_weight_ = sample_weight[idxs] if sample_weight is not None else None
        for scorer_name, scorer_fn in scorers.items():
            raw_scores[scorer_name].append(float(scorer_fn(yt_, yp_, sample_weight=sample_weight_)))

    summary: ScoreSummary = {}
    for key, val in raw_scores.items():
        q0, q25, q50, q75, q100 = np.quantile(val, (0, .25, .5, .75, 1))
        avg, std = np.mean(val), np.std(val)
        summary[key] = {
            'min': q0, 'Q1': q25, 'med': q50, 'Q3': q75, 'max': q100,
            'mean': avg, 'std': std,
        }

    return raw_scores, summary


@overload
def batch_extract_features(
    X: NDArray, sample_rate: float, to_numpy: Literal[True] = True,
    n_jobs: int = 1, verbose: bool = False,
) -> NDArray:
    ...


@overload
def batch_extract_features(
    X: NDArray, sample_rate: float, to_numpy: Literal[False],
    n_jobs: int = 1, verbose: bool = False,
) -> pd.DataFrame:
    ...


def batch_extract_features(
    X: NDArray, sample_rate: float, to_numpy: bool = True,
    n_jobs: int = 1, verbose: bool = False,
) -> Union[NDArray, pd.DataFrame]:
    """ Extract features for a list or array of windows """


    feature_rows: list[FeatureDict] = Parallel(n_jobs=n_jobs)(
        delayed(features.extract_features)(x, sample_rate)
        for x in tqdm(X, total=len(X), mininterval=5, disable=not verbose, bar_format='Extracting features: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]')
    )
    X_feats = pd.DataFrame(feature_rows)

    if to_numpy:
        return X_feats.to_numpy()
    return X_feats


@overload
def batch_count_peaks(
    X: NDArray, sample_rate: float, lowpass_hz: float, params: PeakParams,
    return_peaks: Literal[False] = False,
) -> NDArray:
    ...


@overload
def batch_count_peaks(
    X: NDArray, sample_rate: float, lowpass_hz: float, params: PeakParams,
    return_peaks: Literal[True],
) -> Tuple[NDArray, List[NDArray]]:
    ...


def batch_count_peaks(
    X: NDArray, sample_rate: float, lowpass_hz: float, params: PeakParams,
    return_peaks: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[NDArray]]]:
    """ Count number of peaks for an array of signals """
    V = toV(X, sample_rate, lowpass_hz)
    if return_peaks:
        return batch_count_peaks_from_V(V, sample_rate, params, True)
    return batch_count_peaks_from_V(V, sample_rate, params, False)


@overload
def batch_count_peaks_from_V(
    V: NDArray, sample_rate: float, params: PeakParams,
    return_peaks: Literal[False] = False,
) -> NDArray:
    ...


@overload
def batch_count_peaks_from_V(
    V: NDArray, sample_rate: float, params: PeakParams,
    return_peaks: Literal[True],
) -> Tuple[NDArray, List[NDArray]]:
    ...


def batch_count_peaks_from_V(
    V: NDArray, sample_rate: float, params: PeakParams,
    return_peaks: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[NDArray]]]:
    """ Count number of peaks for an array of signals """

    batch_peaks = batch_find_peaks_from_V(V, sample_rate, params)

    Y = np.asarray([len(peaks) for peaks in batch_peaks])

    if return_peaks:
        return Y, batch_peaks
    return Y


def batch_find_peaks_from_V(
    V: NDArray,
    sample_rate: float,
    params: PeakParams,
) -> List[NDArray]:
    """Find the peaks for an array of signals"""

    batch_peaks = [
        find_peaks(
            v,
            distance=params["distance"] * sample_rate,
            prominence=params["prominence"],
        )[0] / sample_rate  # convert indices to seconds
        for v in V
    ]

    return batch_peaks


def toV(x: NDArray, sample_rate: float, lowpass_hz: float) -> NDArray:
    V = cast(NDArray, np.linalg.norm(x, axis=-1))
    V = V - 1
    V = np.clip(V, -2, 2)
    V = features.butterfilt(V, lowpass_hz, sample_rate, axis=-1)
    return V


def calc_sample_weight(yt: NDArray, pnr: Optional[float] = None) -> NDArray:
    sample_weight = np.ones_like(yt, dtype='float')
    if pnr is None:
        return sample_weight
    sample_weight[yt == 0] = (yt == 1).sum() / (pnr * (yt == 0).sum())
    return sample_weight


def classification_report(yt: NDArray, yp: NDArray, pnr: float = 1.0) -> str:
    return cast(str, metrics.classification_report(yt, yp, sample_weight=calc_sample_weight(yt, pnr=pnr)))


def calibrate(
    yp: NDArray,
    yt: NDArray,
    pnr: float = 1.0,
    precision_tol: float = 0.9,
    recall_tol: float = 0.9,
) -> Dict[str, Any]:
    sample_weight = calc_sample_weight(yt, pnr)
    precision, recall, thresh_pr = metrics.precision_recall_curve(yt, yp, sample_weight=sample_weight)
    fpr, tpr, thresh_roc = metrics.roc_curve(yt, yp, sample_weight=sample_weight)
    f1 = stats.hmean(np.asarray([precision, recall]), axis=0)
    balanced_accuracy = (tpr + (1 - fpr)) / 2

    # Select the threshold that maximizes balanced accuracy.
    balanced_accuracy_idx = np.argmax(balanced_accuracy)
    balanced_accuracy_thresh = thresh_roc[balanced_accuracy_idx]
    best_balanced_accuracy = {
        'thresh': balanced_accuracy_thresh,
        'balanced_accuracy': balanced_accuracy[balanced_accuracy_idx],
        'tpr': tpr[balanced_accuracy_idx],
        'fpr': fpr[balanced_accuracy_idx],
        'predicted': (yp >= balanced_accuracy_thresh).astype('int'),
    }

    # Select the threshold that maximizes F1.
    f1_idx = np.argmax(f1[:-1])
    f1_thresh = thresh_pr[f1_idx]
    best_f1 = {
        'thresh': f1_thresh,
        'f1': f1[f1_idx],
        'precision': precision[f1_idx],
        'recall': recall[f1_idx],
        'predicted': (yp > f1_thresh).astype('int'),
    }

    # Select the first threshold meeting the precision target.
    precision_idx = np.argmax(precision[:-1] > precision_tol)
    precision_thresh = thresh_pr[precision_idx]
    best_precision = {
        'thresh': precision_thresh,
        'f1': f1[precision_idx],
        'precision': precision[precision_idx],
        'recall': recall[precision_idx],
        'predicted': (yp > precision_thresh).astype('int'),
    }

    # Select the first threshold meeting the recall target.
    recall_idx = np.argmax(recall[:-1] > recall_tol)
    recall_thresh = thresh_pr[recall_idx]
    best_recall = {
        'thresh': recall_thresh,
        'f1': f1[recall_idx],
        'precision': precision[recall_idx],
        'recall': recall[recall_idx],
        'predicted': (yp > recall_thresh).astype('int'),
    }

    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'thresh_pr': thresh_pr,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'best_f1': best_f1,
        'tpr': tpr,
        'fpr': fpr,
        'balanced_accuracy': balanced_accuracy,
        'thresh_roc': thresh_roc,
        'best_balanced_accuracy': best_balanced_accuracy,
    }

    return results


def print_report() -> None:
    pass
