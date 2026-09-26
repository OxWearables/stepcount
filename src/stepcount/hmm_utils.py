from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from hmmlearn.hmm import CategoricalHMM

from stepcount._types import HMMParams, NDArray


class HMMSmoother:
    def __init__(
        self,
        use_hmmlearn: bool = False,
        n_components: Optional[int] = None,
        train_test_split: bool = False,
        ste: str = "st",
        startprob: Optional[NDArray] = None,
        emissionprob: Optional[NDArray] = None,
        transmat: Optional[NDArray] = None,
        n_iter: int = 100,
        n_trials: int = 100,
        random_state: int = 123,
        stratify_groups: bool = True,
    ) -> None:
        self.use_hmmlearn = use_hmmlearn
        if use_hmmlearn:
            assert n_components is not None, "Must specify n_components when use_hmmlearn=True"
        self.n_components = n_components
        self.train_test_split = train_test_split
        self.ste = ste
        self.startprob = startprob
        self.emissionprob = emissionprob
        self.transmat = transmat
        self.n_iter = n_iter
        self.n_trials = n_trials
        self.random_state = random_state
        self.stratify_groups = stratify_groups
        self.labels: Optional[NDArray] = None
        self.hmm: Any = None
        self.score: Any = None

    def fit(
        self,
        Y_pred: NDArray,
        Y_true: NDArray,
        groups: Optional[NDArray] = None,
    ) -> HMMSmoother:
        self.labels = np.unique(Y_true)
        if self.use_hmmlearn:
            assert len(self.labels) == self.n_components, f"n_components ({self.n_components}) doesn't match number of labels ({len(self.labels)})"
        if self.startprob is None:
            self.startprob = compute_prior(Y_true, self.labels)
        if self.emissionprob is None:
            self.emissionprob = compute_emission(Y_pred, Y_true, self.labels)
        if self.transmat is None:
            self.transmat = compute_transition(Y_true, self.labels, groups)
        return self

    def predict(self, Y: NDArray, groups: Optional[NDArray] = None) -> NDArray:
        if self.use_hmmlearn:
            return self.hmmlearn_fit_predict(Y, groups=groups, method='predict')
        return self.viterbi(Y, groups)

    def predict_proba(self, Y: NDArray, groups: Optional[NDArray] = None) -> NDArray:
        if self.use_hmmlearn:
            return self.hmmlearn_fit_predict(Y, groups=groups, method='predict_proba')
        raise NotImplementedError

    def viterbi(self, Y: NDArray, groups: Optional[NDArray] = None) -> NDArray:
        if (
            self.startprob is None
            or self.emissionprob is None
            or self.transmat is None
            or self.labels is None
        ):
            raise RuntimeError("HMMSmoother must be fitted before prediction")
        params: HMMParams = {
            'prior': self.startprob,
            'emission': self.emissionprob,
            'transition': self.transmat,
            'labels': self.labels,
        }
        if groups is None:
            Y_vit = viterbi(Y, params)
        else:
            groups = np.asarray(groups)
            if len(groups) != len(Y):
                raise ValueError("groups must have the same length as Y")
            Y_vit = np.empty(len(Y), dtype=self.labels.dtype)
            for group in ordered_unique(groups):
                mask = groups == group
                Y_vit[mask] = viterbi(Y[mask], params)
        return Y_vit

    def hmmlearn_fit_predict(
        self,
        Y: NDArray,
        groups: Optional[NDArray] = None,
        method: Literal['predict', 'predict_proba'] = 'predict',
    ) -> NDArray:

        if self.n_components is None:
            raise RuntimeError("n_components is required when using hmmlearn")

        hmm_params: Dict[str, Any] = {
            "n_components": self.n_components,
            "method": method,
            "train_test_split": self.train_test_split,
            "ste": self.ste,
            "startprob": self.startprob,
            "emissionprob": self.emissionprob,
            "transmat": self.transmat,
            "n_iter": self.n_iter,
            "n_trials": self.n_trials,
            "random_state": self.random_state,
        }

        if Y.ndim == 1:
            Y = Y[:, None]

        if groups is None:
            groups = np.ones(len(Y))
        else:
            groups = np.asarray(groups)
            if len(groups) != len(Y):
                raise ValueError("groups must have the same length as Y")

        if self.stratify_groups:
            Y_pred_parts: list[NDArray] = []
            group_masks: list[NDArray] = []
            scores: list[float] = []
            hmms: list[Any] = []
            for g in ordered_unique(groups):
                mask = groups == g
                _hmm, _score, _Y_pred = hmmlearn_fit_predict(
                    Y[mask],
                    groups=None,
                    **hmm_params,
                )
                Y_pred_parts.append(_Y_pred)
                group_masks.append(mask)
                scores.append(_score)
                hmms.append(_hmm)
            Y_pred = np.empty_like(np.concatenate(Y_pred_parts))
            for mask, part in zip(group_masks, Y_pred_parts):
                Y_pred[mask] = part
            score: Any = scores
            hmm: Any = hmms

        else:
            # hmmlearn receives group lengths, so samples from each group must be contiguous.
            hmm, score, Y_pred = hmmlearn_fit_predict(
                Y,
                groups=groups,
                **hmm_params,
            )

        self.hmm = hmm
        self.score = score

        return Y_pred


def hmmlearn_fit_predict(
    Y: NDArray,
    groups: Optional[NDArray] = None,
    n_components: int = 2,
    method: Literal['predict', 'predict_proba'] = 'predict',
    train_test_split: bool = False,
    ste: str = "st",
    startprob: Optional[NDArray] = None,
    emissionprob: Optional[NDArray] = None,
    transmat: Optional[NDArray] = None,
    n_iter: int = 100,
    n_trials: int = 100,
    random_state: int = 123,
    **kwargs: Any,
) -> Tuple[Any, float, NDArray]:

    if method == 'predict_proba' and len(np.unique(Y)) < 2:
        # TODO: Support probability output when observations contain one label.
        raise NotImplementedError

    np.random.seed(random_state)

    # TODO: Split train and test at group boundaries to avoid leakage.
    if train_test_split:
        n = len(Y)
        Y_train, Y_test = Y[: n // 2], Y[n // 2 :]
        if groups is None:
            groups_train = groups_test = None
        else:
            groups_train, groups_test = groups[: n // 2], groups[n // 2 :]
    else:
        Y_train = Y_test = Y
        groups_train = groups_test = groups

    best_score: Optional[float] = None
    best_hmm: Any = None

    for idx in range(n_trials + 1):

        hmm = CategoricalHMM(
            n_components=n_components,
            params=ste,
            init_params="",
            n_iter=n_iter,
            random_state=idx,
            **kwargs,
        )
        # Explicitly set n_features because Y may omit one of the possible labels.
        # https://github.com/hmmlearn/hmmlearn/issues/423
        hmm.n_features = n_components

        if n_trials > 0 and "s" in ste:
            if startprob is None:
                hmm.startprob_ = np.random.rand(n_components)
            elif isinstance(startprob, np.ndarray):
                hmm.startprob_ = np.random.dirichlet(startprob)
        else:
            hmm.startprob_ = startprob

        if n_trials > 0 and "t" in ste:
            if transmat is None:
                hmm.transmat_ = np.random.rand(n_components, n_components)
            elif isinstance(transmat, np.ndarray):
                hmm.transmat_ = np.array([
                    np.random.dirichlet(transmat[0]),
                    np.random.dirichlet(transmat[1]),
                ])
        else:
            hmm.transmat_ = transmat

        if n_trials > 0 and "e" in ste:
            if emissionprob is None:
                hmm.emissionprob_ = np.random.rand(n_components, n_components)
            elif isinstance(emissionprob, np.ndarray):
                hmm.emissionprob_ = np.array([
                    np.random.dirichlet(emissionprob[0]),
                    np.random.dirichlet(emissionprob[1]),
                ])
        else:
            hmm.emissionprob_ = emissionprob

        hmm.fit(Y_train, lengths_from_groups(groups_train))
        score = float(hmm.score(Y_test, lengths_from_groups(groups_test)))

        if best_score is None or score > best_score:
            best_score = score
            best_hmm = hmm

    if best_hmm is None or best_score is None:
        raise RuntimeError("HMM fitting did not run; n_trials must be non-negative")

    if method == 'predict_proba':
        Y_pred = best_hmm.predict_proba(Y, lengths_from_groups(groups))
    else:
        Y_pred = best_hmm.predict(Y, lengths_from_groups(groups))

    return best_hmm, best_score, Y_pred


def compute_transition(
    Y: NDArray,
    labels: Optional[NDArray] = None,
    groups: Optional[NDArray] = None,
) -> NDArray:
    """ Compute transition matrix from sequence """

    if labels is None:
        labels = np.unique(Y)

    def _compute_transition(values: NDArray) -> NDArray:
        transition = np.vstack([
            np.sum(values[1:][(values == label)[:-1]].reshape(-1, 1) == labels, axis=0)
            for label in labels
        ])
        return transition

    if groups is None:
        transition = _compute_transition(Y)
    else:
        transition = np.zeros((len(labels), len(labels)), dtype=float)
        for g in ordered_unique(groups):
            transition += _compute_transition(Y[groups == g])

    row_sums = np.sum(transition, axis=1)
    zero_rows = np.flatnonzero(row_sums == 0)
    transition[zero_rows, zero_rows] = 1
    row_sums[zero_rows] = 1
    transition = transition / row_sums.reshape(-1, 1)

    return transition


def compute_emission(
    Y_pred: NDArray,
    Y_true: NDArray,
    labels: Optional[NDArray] = None,
) -> NDArray:
    """ Compute emission matrix from predicted and true sequences """

    if labels is None:
        labels = np.unique(Y_true)

    if Y_pred.ndim == 1:
        Y_pred = np.hstack([
            (Y_pred == label).astype('float')[:, None]
            for label in labels
        ])

    emission = np.vstack(
        [np.mean(Y_pred[Y_true == label], axis=0) for label in labels]
    )

    return emission


def compute_prior(
    Y_true: NDArray,
    labels: Optional[NDArray] = None,
    uniform: bool = True,
) -> NDArray:
    """ Compute prior probabilities from sequence """

    if labels is None:
        labels = np.unique(Y_true)

    if uniform:
        # all labels with equal probability
        prior = np.ones(len(labels)) / len(labels)

    else:
        # label probability equals observed rate
        prior = np.mean(Y_true.reshape(-1, 1) == labels, axis=0)

    return prior


def viterbi(Y: NDArray, hmm_params: HMMParams) -> NDArray:
    ''' https://en.wikipedia.org/wiki/Viterbi_algorithm '''

    if len(Y) == 0:
        return np.empty_like(Y)

    def log(x: Any) -> Any:
        SMALL_NUMBER = 1e-16
        return np.log(x + SMALL_NUMBER)

    prior = hmm_params['prior']
    emission = hmm_params['emission']
    transition = hmm_params['transition']
    labels = hmm_params['labels']

    nobs = len(Y)
    nlabels = len(labels)

    Y = np.where(Y.reshape(-1, 1) == labels)[1]  # to numeric

    probs = np.zeros((nobs, nlabels))
    probs[0, :] = log(prior) + log(emission[:, Y[0]])
    for j in range(1, nobs):
        for i in range(nlabels):
            probs[j, i] = np.max(
                log(emission[i, Y[j]]) +
                log(transition[:, i]) +
                probs[j - 1, :])  # probs already in log scale
    viterbi_path = np.zeros_like(Y)
    viterbi_path[-1] = np.argmax(probs[-1, :])
    for j in reversed(range(nobs - 1)):
        viterbi_path[j] = np.argmax(
            log(transition[:, viterbi_path[j + 1]]) +
            probs[j, :])  # probs already in log scale

    viterbi_path = labels[viterbi_path]  # to labels

    return viterbi_path


def ordered_unique(x: NDArray) -> NDArray:
    """ np.unique without sorting """
    return x[np.sort(np.unique(x, return_index=True)[1])]


def lengths_from_groups(groups: Optional[NDArray]) -> Optional[NDArray]:
    if groups is None or len(np.unique(groups)) == 1:
        lengths = None
    else:
        _, ind, cnt = np.unique(groups, return_index=True, return_counts=True)
        lengths = cnt[np.argsort(ind)]
    return lengths
