import numpy as np
from hmmlearn.hmm import CategoricalHMM


class HMMSmoother():
    def __init__(
        self,
        use_hmmlearn=False,
        n_components=None,
        train_test_split=False,
        ste="st",
        startprob=None,
        emissionprob=None,
        transmat=None,
        n_iter=100,
        n_trials=100,
        random_state=123,
        stratify_groups=True,
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

    def fit(self, Y_pred, Y_true, groups=None):
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

    def predict(self, Y, groups=None):
        if self.use_hmmlearn:
            return self.hmmlearn_fit_predict(Y, groups=groups, method='predict')
        return self.viterbi(Y, groups)

    def predict_proba(self, Y, groups=None):
        if self.use_hmmlearn:
            return self.hmmlearn_fit_predict(Y, groups=groups, method='predict_proba')
        raise NotImplementedError

    def viterbi(self, Y, groups=None):
        params = {
            'prior': self.startprob,
            'emission': self.emissionprob,
            'transition': self.transmat,
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

    def hmmlearn_fit_predict(self, Y, groups=None, method='predict'):

        hmm_params = dict(
            n_components=self.n_components,
            method=method,
            train_test_split=self.train_test_split,
            ste=self.ste,
            startprob=self.startprob,
            emissionprob=self.emissionprob,
            transmat=self.transmat,
            n_iter=self.n_iter,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )

        if Y.ndim == 1:
            Y = Y[:, None]

        if groups is None:
            groups = np.ones(len(Y))

        if self.stratify_groups:
            Y_pred = []
            score = []
            hmm = []
            for g in ordered_unique(groups):
                _hmm, _score, _Y_pred = hmmlearn_fit_predict(Y[groups == g], groups=None, **hmm_params)
                Y_pred.append(_Y_pred)
                score.append(_score)
                hmm.append(_hmm)
            Y_pred = np.concatenate(Y_pred)

        else:
            hmm, score, Y_pred = hmmlearn_fit_predict(Y, groups=groups, **hmm_params)

        self.hmm = hmm
        self.score = score

        return Y_pred


def hmmlearn_fit_predict(
    Y,
    groups=None,
    n_components=2,
    method='predict',
    train_test_split=False,
    ste="st",
    startprob=None,
    emissionprob=None,
    transmat=None,
    n_iter=100,
    n_trials=100,
    random_state=123,
    **kwargs,
):

    if method == 'predict_proba' and len(np.unique(Y)) < 2:
        # TODO:
        raise NotImplementedError

    np.random.seed(random_state)

    # TODO: it's better to split by group
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

    best_score = best_hmm = None

    for idx in range(n_trials + 1):

        hmm = CategoricalHMM(
            n_components=n_components,
            params=ste,
            init_params="",
            n_iter=n_iter,
            random_state=idx,
            **kwargs,
        )
        # Manually set n_features = n_components, otherwise this is set
        # automatically to np.unique(Y) which can result in a mismatch when Y
        # doesn't contain all the possible labels.
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
        score = hmm.score(Y_test, lengths_from_groups(groups_test))

        if best_score is None or score > best_score:
            best_score = score
            best_hmm = hmm

    if method == 'predict_proba':
        Y_pred = best_hmm.predict_proba(Y, lengths_from_groups(groups))
    else:
        Y_pred = best_hmm.predict(Y, lengths_from_groups(groups))

    return best_hmm, best_score, Y_pred


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

    if len(Y) == 0:
        return np.empty_like(Y)

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


def lengths_from_groups(groups):
    if groups is None or len(np.unique(groups)) == 1:
        lengths = None
    else:
        _, ind, cnt = np.unique(groups, return_index=True, return_counts=True)
        lengths = cnt[np.argsort(ind)]
    return lengths
