from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import minimize
from joblib import Parallel, delayed
from sklearn import metrics
from imblearn.ensemble import BalancedRandomForestClassifier
import hmm_utils
import features
from tqdm.auto import tqdm


class StepCounter():

    def __init__(self, window_sec=5, sample_rate=100, lowpass_hz=5, steptol=3, wd_params=None, verbose=False):
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.lowpass_hz = lowpass_hz
        self.steptol = steptol
        wd_params = wd_params or dict()
        self.wd = WalkDetector(**wd_params)
        self.verbose = verbose
        self.find_peaks_params = None

    def fit(self, X, Y, groups=None):

        # define walk/non-walk based on threashold
        W = np.zeros_like(Y)
        W[Y >= self.steptol] = 1

        # train walk detector & cross-val-predict
        if self.verbose:
            print("Running cross_val_predict...")
        Wp = cvp(
            self.wd, X, W, groups=groups,
            fit_predict_groups=True,
            n_jobs=-1,
        ).astype('bool')

        if self.verbose:
            print("Fitting walk detector...")
        self.wd.fit(X, W, groups=groups)

        Xw, Yw = X[Wp], Y[Wp]

        def rmse(x):
            Ywp = batch_count_peaks(Xw, self.sample_rate, self.lowpass_hz, to_params(x))
            err = metrics.mean_squared_error(Ywp, Yw, squared=False)
            return err

        def to_params(x):
            params = {
                "distance": x[0] * self.sample_rate,
                "max_width": x[1] * self.sample_rate,
                "max_plateau": x[2] * self.sample_rate,
                "prominence": x[3],
            }
            return params

        if self.verbose:
            print("Tuning step counter...")
        res = minimize(
            rmse,
            x0=[.5, .5, .5, .5],
            bounds=[
                (1 / self.sample_rate, 1),
                (1 / self.sample_rate, 1),
                (1 / self.sample_rate, 1),
                (0, 1),
            ],
            method='Nelder-Mead'
        )

        self.find_peaks_params = to_params(res.x)

        return self

    def predict(self, X, groups=None, return_walk=False):

        if self.find_peaks_params is None:
            print("Model not yet trained. Call .fit() first.")
            return

        # check X quality
        is_ok = ~(np.asarray([np.isnan(x).any() for x in X]))

        X_ = X[is_ok]
        W_ = self.wd.predict(X_, groups).astype('bool')
        Y_ = np.zeros_like(W_, dtype='float')
        Y_[W_] = batch_count_peaks(X_[W_], self.sample_rate, self.lowpass_hz, self.find_peaks_params)

        Y = np.full(len(X), fill_value=np.nan)
        Y[is_ok] = Y_
        W = np.full(len(X), fill_value=np.nan)
        W[is_ok] = W_

        if return_walk:
            return Y, W
        return Y


class WalkDetector():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.clf = BalancedRandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 1000),
            replacement=kwargs.get('replacement', True),
            sampling_strategy=kwargs.get('sampling_strategy', 'not minority'),
            random_state=kwargs.get('random_state', 42),
            verbose=kwargs.get('verbose', 0),
            n_jobs=1,
        )

        self.hmms = hmm_utils.HMMSmoother(
            use_hmmlearn=kwargs.get('use_hmmlearn', True),
            n_iter=kwargs.get('n_iter', 10),
        )

    def fit(self, X, Y, groups=None):
        X_feats = batch_extract_features(X)
        Yp = cvp(
            self.clf, X_feats, Y, groups,
            method='predict_proba',
            fit_predict_groups=False,
            n_jobs=self.kwargs.get('n_jobs', -1),
        )
        self.clf.fit(X_feats, Y)
        self.hmms.fit(Yp, Y, groups=groups)
        return self

    def predict(self, X, groups=None):
        X_feats = batch_extract_features(X, n_jobs=4, verbose=True)
        is_ok = ~(np.isnan(X_feats).any(1))
        W = np.zeros(len(X), dtype='int')  # nan defaults to non-walk
        W[is_ok] = self.clf.predict(X_feats[is_ok])
        W = self.hmms.predict(W, groups=groups)
        return W


def cvp(
    model, X, Y, groups,
    method='predict',
    fit_predict_groups=False,
    n_splits=5,
    n_jobs=-1
):
    """ Like cross_val_predict with custom tweaks """

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

        return Y_test_pred

    Y_out = Parallel(n_jobs=n_jobs)(
        delayed(worker)(train_idxs, test_idxs)
        for train_idxs, test_idxs in groupkfold(groups, n_splits)
    )
    Y_out = np.concatenate(Y_out)

    return Y_out


def groupkfold(groups, n_splits=5):
    """ Like GroupKFold but ordered """

    ord_unq_grps = groups[np.sort(np.unique(groups, return_index=True)[1])]
    folds_unq_grps = np.array_split(ord_unq_grps, n_splits)

    for unq_grps in folds_unq_grps:
        mask = np.isin(groups, unq_grps)
        test_idxs = np.nonzero(mask)
        train_idxs = np.nonzero(~mask)
        yield train_idxs, test_idxs


def batch_extract_features(X, to_numpy=True, n_jobs=1, verbose=False):
    """ Extract features for a list or array of windows """

    X_feats = Parallel(n_jobs=n_jobs)(
        delayed(features.extract_features)(x)
        for x in tqdm(X, disable=not verbose)
    )
    X_feats = pd.DataFrame(X_feats)

    if to_numpy:
        X_feats = X_feats.to_numpy()
    return X_feats


def batch_count_peaks(X, sample_rate, lowpass_hz, params):
    """ Count number of peaks for a list of signals """
    return np.asarray([
        count_peaks(x, sample_rate, lowpass_hz, params)
        for x in X
    ])


def count_peaks(x, sample_rate, lowpass_hz, params):
    """ Count number of peaks in signal """
    v = np.linalg.norm(x, axis=-1)
    v = v - 1
    v = np.clip(v, -2, 2)
    v = features.butterfilt(v, lowpass_hz, sample_rate)
    y = len(find_peaks(
        v,
        distance=params["distance"],
        width=(1, params["max_width"]),
        plateau_size=(1, params["max_plateau"]),
        prominence=params["prominence"],
    )[0])
    return y


def print_report():
    pass
