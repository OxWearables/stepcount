import numpy as np
from hmmlearn import hmm


class HMMSmoother():
    """ HMM model based on hmmlearn """

    def __init__(self, transition=None, **hmm_params) -> None:

        self.transition = transition
        self.hmm_params = hmm_params

        self.h3m = None
        self.labels = None
        self.n_components = None
        self.emission = None

    def fit(self, clf, Y_true):
        self.h3m = None
        self.labels = np.unique(Y_true)
        self.n_components = len(self.labels)
        self.emission = HMMSmoother.calc_clf_emission(clf, Y_true, self.labels)
        return self

    def _h3m_fit(self, Y):

        if self.h3m is not None:
            return self.h3m

        Y = HMMSmoother.reshapeY(Y)

        h3m = hmm.MultinomialHMM(
            n_components=self.n_components,
            params="st",
            init_params="st" if self.transition is None else "s",
            **self.hmm_params,
        )
        h3m.emissionprob_ = self.emission

        if self.transition is not None:
            h3m.transmat_ = self.transition

        h3m.fit(Y)

        return h3m

    def predict(self, Y):
        Y = HMMSmoother.reshapeY(Y)
        return self._h3m_fit(Y).predict(Y)

    def predict_proba(self, Y):
        Y = HMMSmoother.reshapeY(Y)
        return self._h3m_fit(Y).predict_proba(Y)

    def reset(self):
        self.h3m = None
        self.labels = None
        self.n_components = None
        self.emission = None

    @staticmethod
    def reshapeY(Y):
        if len(Y.shape) == 1:
            Y = Y[:, None]
        return Y

    @staticmethod
    def calc_clf_emission(clf, Y_true, labels=None):
        if labels is None:
            labels = np.unique(Y_true)
        Y_prob = clf.oob_decision_function_
        emission = np.vstack(
            [np.mean(Y_prob[Y_true == label], axis=0) for label in labels]
        )
        return emission


def train_hmm(Y_pred, Y_true, labels=None, uniform_prior=True):
    """ Estimate prior, transition, and emission matrices based on the true
    sequence of labels and the model's corresponding predictions """

    if labels is None:
        labels = np.unique(Y_true)

    prior = compute_prior(Y_true, labels, uniform=uniform_prior)
    emission = compute_emission(Y_pred, Y_true, labels)
    transition = compute_transition(Y_true, labels)

    params = {
        'prior': prior,
        'emission': emission,
        'transition': transition,
        'labels': labels
    }

    return params


def compute_transition(Y_true, labels=None):
    """ Compute transition matrix from sequence """

    if labels is None:
        labels = np.unique(Y_true)

    transition = np.vstack([
        np.mean(Y_true[1:][(Y_true == label)[:-1]].reshape(-1, 1) == labels, axis=0)
        for label in labels
    ])

    return transition


def compute_emission(Y_pred, Y_true, labels=None):
    """ Compute emission matrix from predicted and true sequences """

    if labels is None:
        labels = np.unique(Y_true)

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


def viterbi(Y_obs, hmm_params):
    ''' https://en.wikipedia.org/wiki/Viterbi_algorithm '''

    def log(x):
        SMALL_NUMBER = 1e-16
        return np.log(x + SMALL_NUMBER)

    prior = hmm_params['prior']
    emission = hmm_params['emission']
    transition = hmm_params['transition']
    labels = hmm_params['labels']

    nobs = len(Y_obs)
    nlabels = len(labels)

    Y_obs = np.where(Y_obs.reshape(-1, 1) == labels)[1]  # to numeric

    probs = np.zeros((nobs, nlabels))
    probs[0, :] = log(prior) + log(emission[:, Y_obs[0]])
    for j in range(1, nobs):
        for i in range(nlabels):
            probs[j, i] = np.max(
                log(emission[i, Y_obs[j]]) +
                log(transition[:, i]) +
                probs[j - 1, :])  # probs already in log scale
    viterbi_path = np.zeros_like(Y_obs)
    viterbi_path[-1] = np.argmax(probs[-1, :])
    for j in reversed(range(nobs - 1)):
        viterbi_path[j] = np.argmax(
            log(transition[:, viterbi_path[j + 1]]) +
            probs[j, :])  # probs already in log scale

    viterbi_path = labels[viterbi_path]  # to labels

    return viterbi_path
