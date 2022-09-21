from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy import stats
from joblib import Parallel, delayed
from sklearn import metrics
from imblearn.ensemble import BalancedRandomForestClassifier
from stepcount import hmm_utils
from stepcount import features
from tqdm.auto import tqdm


class StepCounter():

    def __init__(
        self,
        window_sec=5,
        sample_rate=100,
        lowpass_hz=5,
        steptol=3,
        pnr=0.1,
        n_jobs=-1,
        wd_params=None,
        verbose=False
    ):
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.lowpass_hz = lowpass_hz
        self.steptol = steptol
        self.pnr = pnr
        self.n_jobs = n_jobs
        self.verbose = verbose

        wd_params = wd_params or dict()
        wd_defaults = dict(sample_rate=sample_rate, n_jobs=n_jobs, verbose=verbose, pnr=pnr)
        for key, value in wd_defaults.items():
            if key not in wd_params:
                wd_params[key] = value
        self.wd_params = wd_params
        self.wd = WalkDetector(**self.wd_params)

        self.window_len = int(np.ceil(window_sec * sample_rate))
        self.find_peaks_params = None
        self.cv_scores = None

    def fit(self, X, Y, groups=None):

        # define walk/non-walk based on threashold
        W = np.zeros_like(Y)
        W[Y >= self.steptol] = 1

        # train walk detector & cross-val-predict
        if self.verbose:
            print("Running cross_val_predict...")
        self.wd.n_jobs = 1
        Wp, list_of_idxs = cvp(
            self.wd, X, W, groups=groups,
            fit_predict_groups=True,
            n_jobs=self.n_jobs,
            return_indices=True,
        )
        self.wd.n_jobs = self.n_jobs

        Wp = Wp.astype('bool')

        if self.verbose:
            print("Fitting walk detector...")
        self.wd.fit(X, W, groups=groups)

        # train step counter
        Xw, Yw = X[Wp], Y[Wp]
        Vw = toV(Xw, self.sample_rate, self.lowpass_hz)

        sample_weight = calc_sample_weight(W, self.pnr)
        sample_weight_w = sample_weight[Wp]

        def mae(x):
            Ywp = batch_count_peaks_from_V(Vw, self.sample_rate, to_params(x))
            err = metrics.mean_absolute_error(Ywp, Yw, sample_weight=sample_weight_w)
            return err

        def to_params(x):
            params = {
                "distance": x[0],
                "max_width": x[1],
                "prominence": x[2],
            }
            return params

        if self.verbose:
            print("Tuning step counter...")
        res = minimize(
            mae,
            x0=[.5, .5, .5],
            bounds=[
                (.2, 2),  # 0.2s to 2s (4Hz - 0.5Hz)
                (.01, 1),  # 10ms to 1s
                (.1, 1),  # 1g to 1g
            ],
            method='Nelder-Mead'
        )

        self.find_peaks_params = to_params(res.x)

        # performance -- walk detector
        wd_cv_scores, wd_cv_summary = get_cv_scores(
            W, Wp, list_of_idxs,
            sample_weight=sample_weight,
            scorer_type='classification'
        )

        # performance -- step count
        Yp = np.zeros_like(Y)
        Yp[Wp] = batch_count_peaks_from_V(Vw, self.sample_rate, self.find_peaks_params)
        sc_cv_scores, sc_cv_summary = get_cv_scores(
            Y, Yp, list_of_idxs,
            sample_weight=sample_weight,
            scorer_type='regression'
        )

        self.cv_scores = {
            'wd': {'scores': wd_cv_scores, 'summary': wd_cv_summary},
            'sc': {'scores': sc_cv_scores, 'summary': sc_cv_summary},
        }

        return self

    def predict(self, X, groups=None, return_walk=False):

        if self.find_peaks_params is None:
            print("Model not yet trained. Call .fit() first.")
            return

        # check X quality
        whr_ok = ~(np.asarray([np.isnan(x).any() for x in X]))

        X_ = X[whr_ok]
        W_ = self.wd.predict(X_, groups).astype('bool')
        Y_ = np.zeros_like(W_, dtype='float')
        Y_[W_] = batch_count_peaks(X_[W_], self.sample_rate, self.lowpass_hz, self.find_peaks_params)

        Y = np.full(len(X), fill_value=np.nan)
        Y[whr_ok] = Y_

        if return_walk:
            W = np.full(len(X), fill_value=np.nan)
            W[whr_ok] = W_
            return Y, W

        return Y

    def predict_from_frame(self, data, **kwargs):

        def fn(chunk):
            """ Process the chunk. Apply padding if length is not enough. """
            n = len(chunk)
            x = chunk[['x', 'y', 'z']].to_numpy()
            if n < self.window_len:
                m = self.window_len - n
                x = np.pad(x, ((0, m), (0, 0)), mode='wrap')
            return x

        X, T = make_windows(data, self.window_sec, fn=fn, return_index=True)
        X = np.asarray(X)
        Y = self.predict(X, **kwargs)
        Y = pd.Series(Y, index=T)
        return Y

class WalkDetector():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.sample_rate = kwargs['sample_rate']
        self.n_jobs = kwargs.get('n_jobs', -1)
        self.verbose = kwargs.get('verbose', 0)

        self.pnr = kwargs.get('pnr', 0.1)
        self.calib_method = kwargs.get('calib_method', 'precision')
        self.precision_tol = kwargs.get('precision_tol', 0.9)

        self.clf = BalancedRandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 1000),
            replacement=kwargs.get('replacement', True),
            sampling_strategy=kwargs.get('sampling_strategy', 'not minority'),
            random_state=kwargs.get('random_state', 42),
            verbose=0, n_jobs=1,
        )

        self.hmms = hmm_utils.HMMSmoother(
            use_hmmlearn=kwargs.get('use_hmmlearn', True),
            n_iter=kwargs.get('n_iter', 10),
        )

        self.thresh = 0.5

    def fit(self, X, Y, groups=None):

        X_feats = batch_extract_features(X, self.sample_rate, n_jobs=self.n_jobs)

        whr_ok = ~(np.isnan(X_feats).any(1))
        X_feats = X_feats[whr_ok]
        Y = Y[whr_ok]
        groups = groups[whr_ok]

        Yp = cvp(
            self.clf, X_feats, Y, groups,
            method='predict_proba',
            fit_predict_groups=False,
            n_jobs=self.n_jobs,
        )

        self.clf.n_jobs = self.n_jobs
        self.clf.fit(X_feats, Y)
        self.clf.n_jobs = 1

        if self.calib_method is not None:
            calib_ops = calibrate(Yp[:, 1], Y, self.pnr, self.precision_tol)
            self.thresh = calib_ops['best_f1']['thresh']
            Ypp = calib_ops['best_f1']['predicted']

            if self.calib_method == 'precision':
                if calib_ops['best_f1']['precision'] < self.precision_tol:
                    self.thresh = calib_ops['best_precision']['thresh']
                    Ypp = calib_ops['best_precision']['predicted']

        self.hmms.fit(Ypp, Y, groups=groups)

        return self

    def predict(self, X, groups=None):
        X_feats = batch_extract_features(X, self.sample_rate, n_jobs=self.n_jobs)
        whr_ok = ~(np.isnan(X_feats).any(1))
        W = np.zeros(len(X), dtype='int')  # nan defaults to non-walk
        W[whr_ok] = (self.clf.predict_proba(X_feats[whr_ok])[:, 1] > self.thresh).astype('int')
        W = self.hmms.predict(W, groups=groups)
        return W


def make_windows(data, window_sec, fn=None, return_index=False):
    """ Split data into windows """

    if fn is None:
        def fn(x):
            return x

    X = [fn(x) for _, x in data.resample(f"{window_sec}s", origin="start")]

    if return_index:
        T = (
            data.index
            .to_series()
            .resample(f"{window_sec}s", origin="start")
            .first()
        )
        return X, T

    return X


def cvp(
    model, X, Y, groups,
    method='predict',
    fit_predict_groups=False,
    return_indices=False,
    n_splits=5,
    n_jobs=-1,
):
    """ Like cross_val_predict with custom tweaks """

    if n_splits == -1:
        n_splits = len(np.unique(groups))

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    def worker(train_idxs, test_idxs):
        X_train, Y_train, groups_train = X[train_idxs], Y[train_idxs], groups[train_idxs]
        X_test, Y_test, groups_test = X[test_idxs], Y[test_idxs], groups[test_idxs]

        m = deepcopy(model)

        if fit_predict_groups:
            m.fit(X_train, Y_train, groups=groups_train)
            Y_test_pred = getattr(m, method)(X_test, groups=groups_test)
        else:
            m.fit(X_train, Y_train)
            Y_test_pred = getattr(m, method)(X_test)

        return Y_test_pred, test_idxs

    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(train_idxs, test_idxs)
        for train_idxs, test_idxs in groupkfold(groups, n_splits)
    )

    Y_pred = np.concatenate([r[0] for r in results])
    list_of_idxs = [r[1] for r in results]

    if return_indices:
        return Y_pred, list_of_idxs

    return Y_pred


def groupkfold(groups, n_splits=5):
    """ Like GroupKFold but ordered """

    ord_unq_grps = groups[np.sort(np.unique(groups, return_index=True)[1])]
    folds_unq_grps = np.array_split(ord_unq_grps, n_splits)

    for unq_grps in folds_unq_grps:
        mask = np.isin(groups, unq_grps)
        test_idxs = np.nonzero(mask)
        train_idxs = np.nonzero(~mask)
        yield train_idxs, test_idxs


def get_cv_scores(yt, yp, list_of_idxs, sample_weight=None, scorer_type='classification'):

    classif_scorers = {
        'accuracy': metrics.accuracy_score,
        'f1': lambda yt, yp, sample_weight=None: metrics.f1_score(yt, yp, sample_weight=sample_weight, zero_division=0),
        'precision': lambda yt, yp, sample_weight=None: metrics.precision_score(yt, yp, sample_weight=sample_weight, zero_division=0),
        'recall': lambda yt, yp, sample_weight=None: metrics.recall_score(yt, yp, sample_weight=sample_weight, zero_division=0),
    }

    regress_scorers = {
        'mae': metrics.mean_absolute_error,
        'rmse': lambda yt, yp, sample_weight: metrics.mean_squared_error(yt, yp, sample_weight=sample_weight, squared=False),
    }

    if scorer_type == 'classification':
        scorers = classif_scorers
    elif scorer_type == 'regression':
        scorers = regress_scorers
    else:
        raise ValueError(f"Unknown {scorer_type=}")

    raw_scores = defaultdict(list)

    for idxs in list_of_idxs:
        yt_, yp_, sample_weight_ = yt[idxs], yp[idxs], sample_weight[idxs]
        for scorer_name, scorer_fn in scorers.items():
            raw_scores[scorer_name].append(scorer_fn(yt_, yp_, sample_weight=sample_weight_))

    summary = {}
    for key, val in raw_scores.items():
        q0, q25, q50, q75, q100 = np.quantile(val, (0, .25, .5, .75, 1))
        avg, std = np.mean(val), np.std(val)
        summary[key] = {
            'min': q0, 'Q1': q25, 'med': q50, 'Q3': q75, 'max': q100,
            'mean': avg, 'std': std,
        }

    return raw_scores, summary


def batch_extract_features(X, sample_rate, to_numpy=True, n_jobs=1, verbose=False):
    """ Extract features for a list or array of windows """

    X_feats = Parallel(n_jobs=n_jobs)(
        delayed(features.extract_features)(x, sample_rate)
        for x in tqdm(X, disable=not verbose)
    )
    X_feats = pd.DataFrame(X_feats)

    if to_numpy:
        X_feats = X_feats.to_numpy()
    return X_feats


def batch_count_peaks(X, sample_rate, lowpass_hz, params):
    """ Count number of peaks for an array of signals """
    V = toV(X, sample_rate, lowpass_hz)
    return batch_count_peaks_from_V(V, sample_rate, params)


def batch_count_peaks_from_V(V, sample_rate, params):
    """ Count number of peaks for an array of signals """
    Y = np.asarray([
        len(find_peaks(
            v,
            distance=params["distance"] * sample_rate,
            width=(1, params["max_width"] * sample_rate),
            prominence=params["prominence"],
        )[0]) for v in V
    ])
    return Y


def toV(x, sample_rate, lowpass_hz):
    V = np.linalg.norm(x, axis=-1)
    V = V - 1
    V = np.clip(V, -2, 2)
    V = features.butterfilt(V, lowpass_hz, sample_rate, axis=-1)
    return V


def calc_sample_weight(yt, pnr=0.1):
    sample_weight = np.ones_like(yt)
    sample_weight[yt == 0] = (yt == 1).sum() / (pnr * (yt == 0).sum())
    return sample_weight


def classification_report(yt, yp, pnr=0.1):
    return metrics.classification_report(yt, yp, sample_weight=calc_sample_weight(yt, pnr=pnr))


def calibrate(yp, yt, pnr=0.1, precision_tol=0.9):
    sample_weight = calc_sample_weight(yt, pnr)
    precision, recall, thresholds = metrics.precision_recall_curve(yt, yp, sample_weight=sample_weight)
    f1 = stats.hmean(np.asarray([precision, recall]), axis=0)

    # optimize for F1
    f1_idx = np.argmax(f1)
    f1_thresh = thresholds[f1_idx]
    f1_f1 = f1[f1_idx]
    f1_precision = precision[f1_idx]
    f1_recall = recall[f1_idx]
    f1_pred = (yp > f1_thresh).astype('int')

    # optimize for precision
    precision_idx = np.argmax(precision > precision_tol)
    precision_thresh = thresholds[precision_idx]
    precision_f1 = f1[precision_idx]
    precision_precision = precision[precision_idx]
    precision_recall = recall[precision_idx]
    precision_pred = (yp > precision_thresh).astype('int')

    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'thresholds': thresholds,
        'best_f1': {
            'thresh': f1_thresh,
            'f1': f1_f1,
            'precision': f1_precision,
            'recall': f1_recall,
            'predicted': f1_pred,
        },
        'best_precision': {
            'thresh': precision_thresh,
            'f1': precision_f1,
            'precision': precision_precision,
            'recall': precision_recall,
            'predicted': precision_pred,
        },
    }

    return results


def print_report():
    pass
