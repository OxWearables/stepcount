from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import minimize
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
        n_jobs=-1,
        wd_params=None,
        verbose=False
    ):
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.lowpass_hz = lowpass_hz
        self.steptol = steptol
        self.n_jobs = n_jobs
        self.verbose = verbose

        wd_params = wd_params or dict()
        wd_defaults = dict(sample_rate=sample_rate, n_jobs=n_jobs, verbose=verbose)
        for key, value in wd_defaults.items():
            if key not in wd_params:
                wd_params[key] = value
        self.wd_params = wd_params
        self.wd = WalkDetector(**self.wd_params)

        self.window_len = int(np.ceil(window_sec * sample_rate))
        self.find_peaks_params = None

    def fit(self, X, Y, groups=None):

        # define walk/non-walk based on threashold
        W = np.zeros_like(Y)
        W[Y >= self.steptol] = 1

        # train walk detector & cross-val-predict
        if self.verbose:
            print("Running cross_val_predict...")
        self.wd.n_jobs = 1
        Wp = cvp(
            self.wd, X, W, groups=groups,
            fit_predict_groups=True,
            n_jobs=self.n_jobs,
        ).astype('bool')
        self.wd.n_jobs = self.n_jobs

        if self.verbose:
            print("Fitting walk detector...")
        self.wd.fit(X, W, groups=groups)

        Xw, Yw = X[Wp], Y[Wp]
        Vw = toV(Xw, self.sample_rate, self.lowpass_hz)

        def mae(x):
            Ywp = batch_count_peaks_from_V(Vw, to_ticks_params(x))
            err = metrics.mean_absolute_error(Ywp, Yw)
            return err

        def to_ticks_params(x):
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
            mae,
            x0=[.5, .5, .5, .5],
            bounds=[
                (1 / self.sample_rate, 1),
                (1 / self.sample_rate, 1),
                (1 / self.sample_rate, 1),
                (0, 1),
            ],
            method='Nelder-Mead'
        )

        self.find_peaks_params = to_ticks_params(res.x)

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
        self.hmms.fit(Yp, Y, groups=groups)
        return self

    def predict(self, X, groups=None):
        X_feats = batch_extract_features(X, self.sample_rate, n_jobs=self.n_jobs)
        whr_ok = ~(np.isnan(X_feats).any(1))
        W = np.zeros(len(X), dtype='int')  # nan defaults to non-walk
        W[whr_ok] = self.clf.predict(X_feats[whr_ok])
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
    n_splits=5,
    n_jobs=-1
):
    """ Like cross_val_predict with custom tweaks """

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
    return batch_count_peaks_from_V(V, params)


def batch_count_peaks_from_V(V, params):
    """ Count number of peaks for an array of signals """
    Y = np.asarray([
        len(find_peaks(
            v,
            distance=params["distance"],
            width=(1, params["max_width"]),
            plateau_size=(1, params["max_plateau"]),
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



def print_report():
    pass
