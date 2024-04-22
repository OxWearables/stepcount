import warnings
from copy import deepcopy
from collections import defaultdict, Counter
import torch
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


class StepCounter:

    def __init__(
        self,
        window_sec=5,
        sample_rate=100,
        steptol=3,
        pnr=1.0,
        lowpass_hz=5,
        cv=5,
        wd_type='ssl',
        wd_params=None,
        n_jobs=-1,
        verbose=False
    ):
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
            wd_defaults = {
                'device': 'cpu',
                'batch_size': 100,
                'verbose': verbose
            }
            wd = WalkDetectorSSL
            # ssl is pretrained with 10s/30hz
            self.window_sec = 10
            self.sample_rate = 30
            # disable multiprocessing, pytorch will already use all available cores,
            # and when using gpu we can only do 1 process at a time anyway
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
        self.wd = wd(**self.wd_params)

        self.window_len = int(np.ceil(self.window_sec * self.sample_rate))
        self.find_peaks_params = None
        self.cv_scores = None

    def fit(self, X, Y, groups=None):

        # define walk/non-walk based on threshold
        W = np.zeros_like(Y)
        W[Y >= self.steptol] = 1
        # zero-out windows below the threshold
        Y = Y.copy()
        Y[Y < self.steptol] = 0

        # train walk detector & cross-val-predict
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

        # train step counter
        Xw, Yw = X[whr_walk_pred], Y[whr_walk_pred]
        Vw = toV(Xw, self.sample_rate, self.lowpass_hz)

        sample_weight = calc_sample_weight(W, self.pnr)
        sample_weight_w = sample_weight[whr_walk_pred]

        def mae(x):
            Ywp = batch_count_peaks_from_V(Vw, self.sample_rate, to_params(x))
            err = metrics.mean_absolute_error(Yw, Ywp, sample_weight=sample_weight_w)
            return err

        def to_params(x):
            params = {
                "distance": x[0],
                "prominence": x[1],
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

        # performance -- walk detector
        _, wd_scores = get_cv_scores(
            W, Wp, cv_test_idxs,
            sample_weight=sample_weight,
            scorer_type='classif'
        )

        # performance -- step count
        Yp = np.zeros_like(Y)
        Yp[whr_walk_pred] = batch_count_peaks_from_V(Vw, self.sample_rate, self.find_peaks_params)
        _, sc_scores = get_cv_scores(
            Y, Yp, cv_test_idxs,
            sample_weight=sample_weight,
            scorer_type='regress'
        )

        # performance -- step count, walk periods only
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

    def predict(self, X, groups=None, return_walk=False, return_step_times=False):

        if self.find_peaks_params is None:
            print("Model not yet trained. Call .fit() first.")
            return

        # check X quality
        ok = np.flatnonzero(~np.asarray([np.isnan(x).any() for x in X]))

        X_ = X[ok]
        W_ = self.wd.predict(X_, groups).astype('bool')
        Y_ = np.zeros_like(W_, dtype='float')
        Z_ = np.full_like(W_, fill_value=None, dtype=np.ndarray)

        w_ = np.flatnonzero(W_)
        (Y_[w_], Z_[w_]) = batch_count_peaks(
            X_[w_],
            self.sample_rate,
            self.lowpass_hz,
            self.find_peaks_params,
            return_peaks=True
        )
        # zero-out windows below the threshold
        Y_[Y_ < self.steptol] = 0

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

    def predict_from_frame(self, data):

        def fn(chunk):
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

        Y, W, Z = self.predict(X, return_walk=True, return_step_times=True)

        Y = pd.Series(Y, index=T, name='Steps')
        W = pd.Series(W, index=T, name='Walk')

        T_steps = []
        for t, z in zip(T, Z):
            if z is not None:
                # convert the local window timestamps to global timestamps
                T_steps.extend([t + pd.Timedelta(seconds=dt) for dt in z])
        T_steps = pd.Series(T_steps, name='time')

        return Y, W, T_steps


class WalkDetectorRF:
    def __init__(
        self,
        sample_rate=100,
        pnr=1.0,
        calib_method='balanced_accuracy',
        precision_tol=.9,
        recall_tol=.9,
        cv=5,
        clf_params=None,
        hmm_params=None,
        n_jobs=-1,
        verbose=False,
    ):

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

        self.clf = BalancedRandomForestClassifier(
            n_estimators=clf_params.get('n_estimators', 1000),
            replacement=clf_params.get('replacement', True),
            sampling_strategy=clf_params.get('sampling_strategy', 'not minority'),
            random_state=clf_params.get('random_state', 42),
            verbose=0, n_jobs=1,
        )

        self.hmms = hmm_utils.HMMSmoother(**hmm_params)

        self.thresh = 0.5

    def fit(self, X, Y, groups=None):

        X_feats = batch_extract_features(X, self.sample_rate, n_jobs=self.n_jobs, verbose=self.verbose)

        whr_ok = ~(np.isnan(X_feats).any(1))
        X_feats = X_feats[whr_ok]
        Y = Y[whr_ok]
        groups = groups[whr_ok]

        Yp = cvp(
            self.clf, X_feats, Y, groups,
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

        self.hmms.fit(Ypp, Y, groups=groups)

        return self

    def predict(self, X, groups=None):

        if len(X) == 0:
            warnings.warn("No data to predict")
            return np.array([], dtype='int')

        W = np.zeros(len(X), dtype='int')  # nan defaults to non-walk
        X_feats = batch_extract_features(X, self.sample_rate, n_jobs=self.n_jobs, verbose=self.verbose)
        ok = ~(np.isnan(X_feats).any(1))
        if ok.any():
            W[ok] = (self.clf.predict_proba(X_feats[ok])[:, 1] > self.thresh).astype('int')
        W = self.hmms.predict(W, groups=groups)

        return W


class WalkDetectorSSL:
    def __init__(
            self,
            device='cpu',
            batch_size=100,
            weights_path='state_dict.pt',
            repo_tag='v1.0.0',
            hmm_params=None,
            verbose=False,
    ):
        self.device = device
        self.weights_path = weights_path
        self.repo_tag = repo_tag
        self.batch_size = batch_size
        self.state_dict = None

        self.verbose = verbose

        hmm_params = hmm_params or dict()
        self.hmms = hmm_utils.HMMSmoother(**hmm_params)

    def fit(self, X, Y, groups=None):
        sslmodel.verbose = self.verbose

        if self.verbose:
            print('Training SSL')

        # prepare training and validation sets
        folds = GroupShuffleSplit(
            1, test_size=0.2, random_state=41
        ).split(X, Y, groups=groups)
        train_idx, val_idx = next(folds)

        x_train = X[train_idx]
        x_val = X[val_idx]

        y_train = Y[train_idx]
        y_val = Y[val_idx]

        group_train = groups[train_idx]
        group_val = groups[val_idx]

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

        # balancing to 90% notwalk, 10% walk
        c = Counter(y_train)
        notwalk = c[0]
        walk = c[1]
        class_weights = [(walk * 9.0) / notwalk, 1.0]

        model = sslmodel.get_sslnet(tag=self.repo_tag, pretrained=True)
        model.to(self.device)

        sslmodel.train(model, train_loader, val_loader, self.device, class_weights, weights_path=self.weights_path)
        model.load_state_dict(torch.load(self.weights_path, self.device))

        if self.verbose:
            print('Training HMM')

        # train HMM with predictions of the validation set
        y_val, y_val_pred, group_val = sslmodel.predict(model, val_loader, self.device, output_logits=True)
        y_val_pred_sf = softmax(y_val_pred, axis=1)

        self.hmms.fit(y_val_pred_sf, y_val, groups=group_val)

        # move model to cpu to get a device-less state dict (prevents device conflicts when loading on cpu/gpu later)
        model.to('cpu')
        self.state_dict = model.state_dict()

        return self

    def predict(self, X, groups=None):

        if len(X) == 0:
            warnings.warn("No data to predict")
            return np.array([], dtype='int')

        sslmodel.verbose = self.verbose

        dataset = sslmodel.NormalDataset(X, name='prediction')
        dataloader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
            num_workers=0,
        )

        model = sslmodel.get_sslnet(tag=self.repo_tag, pretrained=False)
        model.load_state_dict(self.state_dict)
        model.to(self.device)

        _, y_pred, _ = sslmodel.predict(model, dataloader, self.device, output_logits=False)

        y_pred = self.hmms.predict(y_pred, groups=groups)

        return y_pred


def make_windows(data, window_sec, fn=None, return_index=False, verbose=True):
    """ Split data into windows """

    if verbose:
        print("Defining windows...")

    if fn is None:
        def fn(x):
            return x

    X, T = [], []
    for t, x in tqdm(data.resample(f"{window_sec}s", origin="start"), mininterval=5, disable=not verbose):
        x = fn(x)
        X.append(x)
        T.append(t)

    X = np.asarray(X)

    if return_index:
        T = pd.DatetimeIndex(T, name=data.index.name)
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
        m.n_jobs = 1

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
    cv_test_idxs = [r[1] for r in results]

    if return_indices:
        return Y_pred, cv_test_idxs

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


def get_cv_scores(yt, yp, cv_test_idxs, sample_weight=None, scorer_type='classif'):

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

    def smooth_mean_absolute_percentage_error(yt, yp, sample_weight=None):
        yt, yp = yt.copy(), yp.copy()
        # add 1 where zero to smooth the mape
        whr = yt == 0
        yt[whr] += 1
        yp[whr] += 1
        return metrics.mean_absolute_percentage_error(yt, yp, sample_weight=sample_weight)

    if scorer_type == 'classif':
        scorers = classif_scorers
    elif scorer_type == 'regress':
        scorers = regress_scorers
    else:
        raise ValueError(f"Unknown {scorer_type=}")

    raw_scores = defaultdict(list)

    for idxs in cv_test_idxs:
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

    if verbose:
        print("Extracting features...")

    X_feats = Parallel(n_jobs=n_jobs)(
        delayed(features.extract_features)(x, sample_rate)
        for x in tqdm(X, mininterval=5, disable=not verbose)
    )
    X_feats = pd.DataFrame(X_feats)

    if to_numpy:
        X_feats = X_feats.to_numpy()
    return X_feats


def batch_count_peaks(X, sample_rate, lowpass_hz, params, return_peaks=False):
    """ Count number of peaks for an array of signals """
    V = toV(X, sample_rate, lowpass_hz)
    return batch_count_peaks_from_V(V, sample_rate, params, return_peaks)


def batch_count_peaks_from_V(V, sample_rate, params, return_peaks=False):
    """ Count number of peaks for an array of signals """

    batch_peaks = batch_find_peaks_from_V(V, sample_rate, params)

    Y = np.asarray([len(peaks) for peaks in batch_peaks])

    if return_peaks:
        return Y, batch_peaks
    return Y


def batch_find_peaks_from_V(V, sample_rate, params):
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


def toV(x, sample_rate, lowpass_hz):
    V = np.linalg.norm(x, axis=-1)
    V = V - 1
    V = np.clip(V, -2, 2)
    V = features.butterfilt(V, lowpass_hz, sample_rate, axis=-1)
    return V


def calc_sample_weight(yt, pnr=None):
    sample_weight = np.ones_like(yt, dtype='float')
    if pnr is None:
        return sample_weight
    sample_weight[yt == 0] = (yt == 1).sum() / (pnr * (yt == 0).sum())
    return sample_weight


def classification_report(yt, yp, pnr=1.0):
    return metrics.classification_report(yt, yp, sample_weight=calc_sample_weight(yt, pnr=pnr))


def calibrate(yp, yt, pnr=1.0, precision_tol=0.9, recall_tol=0.9):
    sample_weight = calc_sample_weight(yt, pnr)
    precision, recall, thresh_pr = metrics.precision_recall_curve(yt, yp, sample_weight=sample_weight)
    fpr, tpr, thresh_roc = metrics.roc_curve(yt, yp, sample_weight=sample_weight)
    f1 = stats.hmean(np.asarray([precision, recall]), axis=0)
    balanced_accuracy = (tpr + (1 - fpr)) / 2

    # optimize for balanced accuracy
    balanced_accuracy_idx = np.argmax(balanced_accuracy)
    balanced_accuracy_thresh = thresh_roc[balanced_accuracy_idx]
    best_balanced_accuracy = {
        'thresh': balanced_accuracy_thresh,
        'balanced_accuracy': balanced_accuracy[balanced_accuracy_idx],
        'tpr': tpr[balanced_accuracy_idx],
        'fpr': fpr[balanced_accuracy_idx],
        'predicted': (yp >= balanced_accuracy_thresh).astype('int'),
    }

    # optimize for F1
    f1_idx = np.argmax(f1[:-1])
    f1_thresh = thresh_pr[f1_idx]
    best_f1 = {
        'thresh': f1_thresh,
        'f1': f1[f1_idx],
        'precision': precision[f1_idx],
        'recall': recall[f1_idx],
        'predicted': (yp > f1_thresh).astype('int'),
    }

    # optimize for precision
    precision_idx = np.argmax(precision[:-1] > precision_tol)
    precision_thresh = thresh_pr[precision_idx]
    best_precision = {
        'thresh': precision_thresh,
        'f1': f1[precision_idx],
        'precision': precision[precision_idx],
        'recall': recall[precision_idx],
        'predicted': (yp > precision_thresh).astype('int'),
    }

    # optimize for recall
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


def print_report():
    pass
