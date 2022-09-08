import numpy as np
from hmmlearn import hmm


class HMMSmoother():

    def __init__(self, use_hmmlearn=False, **hmmlearn_params) -> None:
        self.use_hmmlearn = use_hmmlearn
        self.hmmlearn_params = hmmlearn_params

    def fit(self, Y_pred, Y_true, groups=None):
        self.labels = np.unique(Y_true)
        self.n_components = len(self.labels)
        self.prior = compute_prior(Y_true, self.labels)
        self.emission = compute_emission(Y_pred, Y_true, self.labels)
        self.transition = compute_transition(Y_true, self.labels, groups)
        return self

    def predict(self, Y, groups=None):
        if self.use_hmmlearn:
            return self._hmmlearn_fit_predict(Y, groups=None)
        return self.viterbi(Y, groups)

    def predict_proba(self, Y):
        if self.use_hmmlearn:
            return self._hmmlearn_fit_predict(Y, groups=None, method='predict_proba')
        raise NotImplementedError

    def viterbi(self, Y, groups=None):
        params = {
            'prior': self.prior,
            'emission': self.emission,
            'transition': self.transition,
            'labels': self.labels,
        }
        if groups is None:
            Y_vit = viterbi(Y, params)
        else:
            Y_vit = np.concatenate([
                viterbi(Y[groups == g], params)
                for g in ordered_unique(groups)
            ])
        return Y_vit

    def _hmmlearn_fit_predict(self, Y, groups=None, method='predict'):

        if len(np.unique(Y)) < 2:
            if method == 'predict_proba':
                # TODO:
                raise NotImplementedError
            return Y

        Y = HMMSmoother.reshapeY(Y)

        hmm_ = hmm.MultinomialHMM(
            n_components=self.n_components,
            params="te",
            init_params="",
            **self.hmmlearn_params,
        )
        hmm_.startprob_ = self.prior
        hmm_.emissionprob_ = self.emission
        hmm_.transmat_ = self.transition

        lengths = None
        if groups is not None:
            _, ind, cnt = np.unique(groups, return_index=True, return_counts=True)
            lengths = cnt[np.argsort(ind)]

        hmm_.fit(Y, lengths)

        if method == 'predict_proba':
            return hmm_.predict_proba(Y, lengths)
        return hmm_.predict(Y, lengths)

    @staticmethod
    def reshapeY(Y):
        if len(Y.shape) == 1:
            Y = Y[:, None]
        return Y


def compute_transition(Y, labels=None, groups=None):
    """ Compute transition matrix from sequence """

    if labels is None:
        labels = np.unique(Y)

    def _compute_transition(Y):
        transition = np.vstack([
            np.sum(Y[1:][(Y == label)[:-1]].reshape(-1, 1) == labels, axis=0)
            for label in labels
        ])
        return transition

    if groups is None:
        transition = _compute_transition(Y)
    else:
        transition = sum((
            _compute_transition(Y[groups == g])
            for g in ordered_unique(groups)
        ))

    transition = transition / np.sum(transition, axis=1).reshape(-1, 1)

    return transition


def compute_emission(Y_pred, Y_true, labels=None):
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


def compute_prior(Y_true, labels=None, uniform=True):
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


def viterbi(Y, hmm_params):
    ''' https://en.wikipedia.org/wiki/Viterbi_algorithm '''

    def log(x):
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


def ordered_unique(x):
    """ np.unique without sorting """
    return x[np.sort(np.unique(x, return_index=True)[1])]
