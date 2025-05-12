# Author: Jiaqi Li <xuewei4d@gmail.com>
# License: BSD 3 clause

import copy
import itertools
import re
import sys
import warnings
from io import StringIO
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import linalg, stats
from scipy.special import ive as expBessel

from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.extmath import fast_logdet

import spsklearn
from spsklearn.mixture import vonMisesFisherMixture
from spsklearn.mixture._von_mises_fisher_mixture import (
    _estimate_von_mises_fisher_parameters,
)


class RandomData:
    def __init__(self, rng, n_samples=1500, n_components=3, n_features=2, scale=50):
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_features = n_features

        # weights
        self.weights = rng.rand(n_components)
        self.weights /= np.sum(self.weights)
        # generate mean direction
        self.means = rng.randn(n_components, n_features)
        self.means /= np.linalg.norm(self.means, ord=2, axis=1, keepdims=True)
        # concentration parameters kappa
        self.kappas = scale * rng.rand(n_components)

        self.X = []
        for _, (w, m, k) in enumerate(zip(self.weights, self.means, self.kappas)):
            vmf = stats.vonmises_fisher(m, k)
            self.X.append(vmf.rvs(int(np.round(w * n_samples)), random_state=rng))
        self.X = np.vstack(self.X)

        self.Y = np.hstack(
            [
                np.full(int(np.round(w * n_samples)), k, dtype=int)
                for k, w in enumerate(self.weights)
            ]
        )


def test_von_mises_fisher_mixture_attributes():
    # test bad parameters
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, n_samples=10, n_components=2, n_features=2, scale=20)
    X = rand_data.X

    # test good parameters
    n_components, tol, n_init, max_iter = (
        2,
        1e-4,
        3,
        30,
    )
    init_params = "random"
    vmf = vonMisesFisherMixture(
        n_components=n_components,
        tol=tol,
        n_init=n_init,
        max_iter=max_iter,
        init_params=init_params,
    ).fit(X)

    assert vmf.n_components == n_components
    assert vmf.tol == tol
    assert vmf.max_iter == max_iter
    assert vmf.n_init == n_init
    assert vmf.init_params == init_params


def test_check_weights():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)

    n_components = rand_data.n_components
    X = rand_data.X

    vmf = vonMisesFisherMixture(n_components=n_components)

    # Check bad shape
    weights_bad_shape = rng.rand(n_components, 1)
    vmf.weights_init = weights_bad_shape
    msg = re.escape(
        "The parameter 'weights' should have the shape of "
        f"({n_components},), but got {str(weights_bad_shape.shape)}"
    )
    with pytest.raises(ValueError, match=msg):
        vmf.fit(X)

    # Check bad range
    weights_bad_range = rng.rand(n_components) + 1
    vmf.weights_init = weights_bad_range
    msg = re.escape(
        "The parameter 'weights' should be in the range [0, 1], but got"
        f" max value {np.min(weights_bad_range):.5f}, "
        f"min value {np.max(weights_bad_range):.5f}"
    )
    with pytest.raises(ValueError, match=msg):
        vmf.fit(X)

    # Check bad normalization
    weights_bad_norm = rng.rand(n_components)
    weights_bad_norm = weights_bad_norm / (weights_bad_norm.sum() + 1)
    vmf.weights_init = weights_bad_norm
    msg = re.escape(
        "The parameter 'weights' should be normalized, "
        f"but got sum(weights) = {np.sum(weights_bad_norm):.5f}"
    )
    with pytest.raises(ValueError, match=msg):
        vmf.fit(X)

    # Check good weights matrix
    weights = rand_data.weights
    print("****** weights", weights)
    vmf = vonMisesFisherMixture(weights_init=weights, n_components=n_components)
    vmf.fit(X)
    assert_array_equal(weights, vmf.weights_init)


def test_check_means():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)

    n_components, n_features = rand_data.n_components, rand_data.n_features
    X = rand_data.X

    vmf = vonMisesFisherMixture(n_components=n_components)

    # Check means bad shape
    means_bad_shape = rng.rand(n_components + 1, n_features)
    vmf.means_init = means_bad_shape
    msg = "The parameter 'means' should have the shape of "
    with pytest.raises(ValueError, match=msg):
        vmf.fit(X)

    # Check good means matrix
    means = rand_data.means
    vmf.means_init = means
    vmf.fit(X)
    assert_array_equal(means, vmf.means_init)


def _naive_lvmfpdf(X, means, kappas):
    resp = np.zeros((len(X), len(kappas)))
    for i, (mean, kappa) in enumerate(zip(means, kappas)):
        resp[:, i] = stats.vonmises_fisher.logpdf(X, mean, kappa)
    return resp


def test_von_mises_fisher_mixture_log_probabilities():
    from spsklearn.mixture._von_mises_fisher_mixture import (
        _estimate_log_von_mises_fisher_prob,
    )

    # test against with _naive_lvmfpdf
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)

    means = rand_data.means
    kappas = rand_data.kappas
    X = rand_data.X
    log_prob_naive = _naive_lvmfpdf(X, means, kappas)

    log_prob = _estimate_log_von_mises_fisher_prob(X, means, kappas)
    assert_array_almost_equal(log_prob, log_prob_naive)


def test_von_mises_fisher_mixture_estimate_log_prob_resp():
    # test whether responsibilities are normalized
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=5)
    n_samples = rand_data.n_samples
    n_features = rand_data.n_features
    n_components = rand_data.n_components

    X = rand_data.X
    weights = rand_data.weights
    means = rand_data.means
    kappas = rand_data.kappas
    vmf = vonMisesFisherMixture(
        n_components=n_components,
        random_state=rng,
        weights_init=weights,
        means_init=means,
        kappas_init=kappas,
    )
    vmf.fit(X)
    resp = vmf.predict_proba(X)
    assert_array_almost_equal(resp.sum(axis=1), np.ones(n_samples))
    assert_array_equal(vmf.weights_init, weights)
    assert_array_equal(vmf.means_init, means)
    assert_array_equal(vmf.kappas_init, kappas)


def test_von_mises_fisher_mixture_predict_predict_proba():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    X = rand_data.X
    Y = rand_data.Y
    vmf = vonMisesFisherMixture(
        n_components=rand_data.n_components,
        random_state=rng,
        weights_init=rand_data.weights,
        means_init=rand_data.means,
        kappas_init=rand_data.kappas,
        verbose=2,
    )

    # Check a warning message arrive if we don't do fit
    msg = (
        "This vonMisesFisherMixture instance is not fitted yet. Call 'fit' "
        "with appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        vmf.predict(X)

    vmf.fit(X)

    Y_pred = vmf.predict(X)
    resp_pred = vmf.predict_proba(X)

    # import matplotlib.pyplot as plt

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    # ax1.scatter(X[Y == 0, 0], X[Y == 0, 1], s=0.5, c="green", label="Y=0")
    # ax1.scatter(X[Y == 1, 0], X[Y == 1, 1], s=0.5, c="orange", label="Y=1")
    # ax1.scatter(X[Y == 2, 0], X[Y == 2, 1], s=0.5, c="purple", label="Y=2")
    # ax1.scatter(
    #     rand_data.means[:, 0],
    #     rand_data.means[:, 1],
    #     marker="x",
    #     s=0.5 * 200,
    #     c="red",
    #     label="vMF centers",
    # )
    # ax1.scatter(0, 0, marker=".", c="black", label="Origin")
    # ax1.set_xlim(-1.2, 1.2)
    # ax1.set_ylim(-1.2, 1.2)
    # ax1.legend()

    # ax2.scatter(X[Y_pred == 0, 0], X[Y_pred == 0, 1], s=0.5, c="green", label="pred_Y=0")
    # ax2.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], s=0.5, c="orange", label="pred_Y=1")
    # ax2.scatter(X[Y_pred == 2, 0], X[Y_pred == 2, 1], s=0.5, c="purple", label="pred_Y=2")
    # ax2.scatter(
    #     vmf.means_[:, 0],
    #     vmf.means_[:, 1],
    #     marker="x",
    #     s=0.5 * 200,
    #     c="red",
    #     label="clustering centers",
    # )
    # ax2.set_xlim(-1.2, 1.2)
    # ax2.set_ylim(-1.2, 1.2)
    # ax2.legend()
    # fig.savefig("test_von_mises_fisher_mixture_predict_predict_proba.pdf")
    assert_array_equal(Y_pred, resp_pred.argmax(axis=1))

    assert adjusted_rand_score(Y, Y_pred) > 0.90


@pytest.mark.filterwarnings("ignore:.*did not converge.*")
@pytest.mark.parametrize(
    "seed, max_iter, tol",
    [
        (0, 2, 1e-7),  # strict non-convergence
        (1, 2, 1e-1),  # loose non-convergence
        (3, 300, 1e-7),  # strict convergence
        (4, 300, 1e-1),  # loose convergence
    ],
)
def test_von_mises_fisher_mixture_fit_predict(seed, max_iter, tol):
    rng = np.random.RandomState(seed)
    rand_data = RandomData(rng)
    X = rand_data.X
    Y = rand_data.Y
    vmf = vonMisesFisherMixture(
        n_components=rand_data.n_components,
        random_state=rng,
        weights_init=rand_data.weights,
        means_init=rand_data.means,
        kappas_init=rand_data.kappas,
        max_iter=max_iter,
        tol=tol,
    )

    # check if fit_predict(X) is equivalent to fit(X).predict(X)
    f = copy.deepcopy(vmf)
    Y_pred1 = f.fit(X).predict(X)
    Y_pred2 = vmf.fit_predict(X)
    assert_array_equal(Y_pred1, Y_pred2)
    # assert adjusted_rand_score(Y, Y_pred2) > 0.90


def test_von_mises_fisher_mixture_fit_predict_n_init():
    # Check that fit_predict is equivalent to fit.predict, when n_init > 1
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, n_samples=1000, n_components=5, n_features=5, scale=50)
    X = rand_data.X
    vmf = vonMisesFisherMixture(n_components=5, n_init=5, random_state=0)
    y_pred1 = vmf.fit_predict(X)
    y_pred2 = vmf.predict(X)
    assert_array_equal(y_pred1, y_pred2)


# TODO: cannot pass on decimal=7
def test_von_mises_fisher_mixture_fit_best_params():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    n_init = 10
    X = rand_data.X
    vmf = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        random_state=rng,
        max_iter=1000,
    )
    ll = []
    for _ in range(n_init):
        vmf.fit(X)
        ll.append(vmf.score(X))
    ll = np.array(ll)
    vmf_best = vonMisesFisherMixture(
        n_components=n_components,
        n_init=n_init,
        random_state=rng,
    )
    vmf_best.fit(X)
    assert_almost_equal(ll.min(), vmf_best.score(X))


# TODO: does not warn
def test_von_mises_fisher_mixture_fit_convergence_warning():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=100)
    n_components = rand_data.n_components
    max_iter = 1
    X = rand_data.X
    vmf = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        max_iter=max_iter,
        random_state=rng,
    )
    msg = (
        "Best performing initialization did not converge. "
        "Try different init parameters, or increase max_iter, "
        "tol, or check for degenerate data."
    )
    with pytest.warns(ConvergenceWarning, match=msg):
        vmf.fit(X)


def test_multiple_init():
    # Test that multiple inits does not much worse than a single one
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 50, 5, 2
    rand_data = RandomData(rng, n_samples=n_samples, n_features=n_features, scale=20)
    X = rand_data.X
    train1 = (
        vonMisesFisherMixture(n_components=n_components, random_state=0).fit(X).score(X)
    )
    train2 = (
        vonMisesFisherMixture(
            n_components=n_components,
            random_state=0,
            n_init=5,
        )
        .fit(X)
        .score(X)
    )
    assert train2 >= train1


def test_von_mises_fisher_mixture_n_parameters():
    # Test that the right number of parameters is estimated
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 50, 5, 2
    rand_data = RandomData(
        rng,
        n_components=n_components,
        n_samples=n_samples,
        n_features=n_features,
        scale=20,
    )
    X = rand_data.X
    n_params = n_components * n_features + 2 * n_components - 1
    n_params = 13
    vmf = vonMisesFisherMixture(n_components=n_components, random_state=rng).fit(X)
    assert vmf._n_parameters() == n_params


def test_von_mises_fisher_mixture_aic_bic():
    # Test the aic and bic criteria
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 50, 3, 2
    X = stats.vonmises_fisher.rvs(
        mu=np.array([0, 0, 1]), kappa=50, size=n_samples, random_state=rng
    )
    svmfh = stats.vonmises_fisher.entropy(mu=np.array([0, 0, 1]), kappa=50)
    vmf = vonMisesFisherMixture(
        n_components=n_components,
        random_state=rng,
        max_iter=200,
    )
    vmf.fit(X)
    aic = 2 * n_samples * svmfh + 2 * vmf._n_parameters()
    bic = 2 * n_samples * svmfh + np.log(n_samples) * vmf._n_parameters()
    bound = n_features / np.sqrt(n_samples)
    assert (vmf.aic(X) - aic) / n_samples < bound
    assert (vmf.bic(X) - bic) / n_samples < bound


def test_von_mises_fisher_mixture_verbose():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    X = rand_data.X
    g = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        random_state=rng,
        verbose=1,
    )
    h = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        random_state=rng,
        verbose=2,
    )
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        g.fit(X)
        h.fit(X)
    finally:
        sys.stdout = old_stdout


@pytest.mark.filterwarnings("ignore:.*did not converge.*")
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_warm_start(seed):
    random_state = seed
    rng = np.random.RandomState(random_state)
    n_samples, n_features, n_components = 500, 2, 2
    rand_data = RandomData(
        rng,
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        scale=20,
    )
    X = rand_data.X

    # Assert the warm_start give the same result for the same number of iter
    g = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        max_iter=2,
        random_state=random_state,
        warm_start=False,
    )
    h = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        max_iter=1,
        random_state=random_state,
        warm_start=True,
    )

    g.fit(X)
    score1 = h.fit(X).score(X)
    score2 = h.fit(X).score(X)

    assert_almost_equal(g.weights_, h.weights_)
    assert_almost_equal(g.means_, h.means_)
    assert_almost_equal(g.kappas_, h.kappas_)
    assert score2 > score1

    # Assert that by using warm_start we can converge to a good solution
    g = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        max_iter=2,
        random_state=random_state,
        warm_start=False,
        tol=1e-6,
    )
    h = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        max_iter=2,
        random_state=random_state,
        warm_start=True,
        tol=1e-6,
    )

    g.fit(X)
    h.fit(X)
    assert not g.converged_

    h.fit(X)
    # depending on the data there is large variability in the number of
    # refit necessary to converge due to the complete randomness of the
    # data
    for _ in range(1000):
        h.fit(X)
        if h.converged_:
            break
    assert h.converged_


@ignore_warnings(category=ConvergenceWarning)
def test_convergence_detected_with_warm_start():
    # We check that convergence is detected when warm_start=True
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components = rand_data.n_components
    X = rand_data.X

    for max_iter in (1, 2, 50):
        vmf = vonMisesFisherMixture(
            n_components=n_components,
            warm_start=True,
            max_iter=max_iter,
            random_state=rng,
        )
        for _ in range(100):
            vmf.fit(X)
            if vmf.converged_:
                break
        assert vmf.converged_
        assert max_iter >= vmf.n_iter_


def test_score():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components
    X = rand_data.X

    # Check the error message if we don't call fit
    vmf = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        max_iter=1,
        random_state=rng,
    )
    msg = (
        "This vonMisesFisherMixture instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        vmf.score(X)

    # Check score value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        vmf.fit(X)
    vmf_score = vmf.score(X)
    vmf_score_proba = vmf.score_samples(X).mean()
    assert_almost_equal(vmf_score, vmf_score_proba)

    # Check if the score increase
    vmf2 = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        random_state=rng,
    ).fit(X)
    assert vmf2.score(X) > vmf.score(X)


def test_score_samples():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components
    X = rand_data.X

    # Check the error message if we don't call fit
    vmf = vonMisesFisherMixture(
        n_components=n_components,
        n_init=1,
        random_state=rng,
    )
    msg = (
        "This vonMisesFisherMixture instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        vmf.score_samples(X)

    vmf_score_samples = vmf.fit(X).score_samples(X)
    assert vmf_score_samples.shape[0] == rand_data.n_samples


def test_monotonic_likelihood():
    # We check that each step of the EM without regularization improve
    # monotonically the training set likelihood
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=7)
    n_components = rand_data.n_components
    X = rand_data.X

    vmf = vonMisesFisherMixture(
        n_components=n_components,
        warm_start=True,
        max_iter=1000,
        random_state=rng,
        tol=1e-7,
    )
    current_log_likelihood = -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        # Do one training iteration at a time so we can make sure that the
        # training log likelihood increases after each iteration.
        for _ in range(600):
            prev_log_likelihood = current_log_likelihood
            current_log_likelihood = vmf.fit(X).score(X)
            assert current_log_likelihood >= prev_log_likelihood

            if vmf.converged_:
                break

        assert vmf.converged_


@ignore_warnings(category=ConvergenceWarning)
def test_init():
    # We check that by increasing the n_init number we have a better solution
    for random_state in range(15):
        rand_data = RandomData(
            np.random.RandomState(random_state), n_samples=50, scale=20
        )
        n_components = rand_data.n_components
        X = rand_data.X

        vmf1 = vonMisesFisherMixture(
            n_components=n_components, n_init=1, max_iter=1, random_state=random_state
        ).fit(X)
        vmf2 = vonMisesFisherMixture(
            n_components=n_components, n_init=10, max_iter=1, random_state=random_state
        ).fit(X)

        assert vmf2.lower_bound_ >= vmf1.lower_bound_


@pytest.mark.parametrize(
    "init_params",
    ["random", "random_from_data", "spherical-k-means++", "spherical-k-means"],
)
def test_init_means_not_duplicated(init_params):
    # Check that all initialisations provide not duplicated starting means
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=5)
    n_components = rand_data.n_components
    X = rand_data.X

    vmf = vonMisesFisherMixture(
        n_components=n_components, init_params=init_params, random_state=rng, max_iter=0
    )
    vmf.fit(X)

    means = vmf.means_
    for i_mean, j_mean in itertools.combinations(means, r=2):
        assert not np.allclose(i_mean, j_mean)


@pytest.mark.parametrize(
    "init_params",
    ["random", "random_from_data", "spherical-k-means++", "spherical-k-means"],
)
def test_means_for_all_inits(init_params):
    # Check fitted means properties for all initializations
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=5)
    n_components = rand_data.n_components
    X = rand_data.X

    vmf = vonMisesFisherMixture(
        n_components=n_components, init_params=init_params, random_state=rng
    )
    vmf.fit(X)

    assert vmf.means_.shape == (n_components, X.shape[1])
    assert np.all(X.min(axis=0) <= vmf.means_)
    assert np.all(vmf.means_ <= X.max(axis=0))
    assert vmf.converged_


def test_max_iter_zero():
    # Check that max_iter=0 returns initialisation as expected
    # Pick arbitrary initial means and check equal to max_iter=0
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, scale=5, n_components=2, n_features=2)
    n_components = rand_data.n_components
    X = rand_data.X
    means_init = [[20, 30], [30, 25]]
    means_init /= np.linalg.norm(means_init, ord=2, axis=1, keepdims=True)
    vmf = vonMisesFisherMixture(
        n_components=n_components,
        random_state=rng,
        means_init=means_init,
        tol=1e-06,
        max_iter=0,
    )
    vmf.fit(X)

    assert_allclose(vmf.means_, means_init)


def _generate_data(seed, n_samples, n_features, n_components):
    """Randomly generate samples and responsibilities."""
    rs = np.random.RandomState(seed)
    X = rs.random_sample((n_samples, n_features))
    resp = rs.random_sample((n_samples, n_components))
    resp /= resp.sum(axis=1)[:, np.newaxis]
    return X, resp


def test_von_mises_fisher_mixture_single_component_stable():
    """
    Non-regression test for #23032 ensuring 1-component GM works on only a
    few samples.
    """
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng, n_samples=3, n_components=1, n_features=2, scale=20)
    X = rand_data.X
    vmf = vonMisesFisherMixture(n_components=1)
    vmf.fit(X)
    vmf.sample()


def test_von_mises_fisher_mixture_all_init_does_not_estimate_von_mises_fisher_parameters(
    monkeypatch,
):
    """When all init parameters are provided, the von Mises-Fisher parameters
    are not estimated.

    Non-regression test for gh-26015.
    """

    mock = Mock(side_effect=_estimate_von_mises_fisher_parameters)
    monkeypatch.setattr(
        spsklearn.mixture._von_mises_fisher_mixture,
        "_estimate_von_mises_fisher_parameters",
        mock,
    )

    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)

    vmf = vonMisesFisherMixture(
        n_components=rand_data.n_components,
        weights_init=rand_data.weights,
        means_init=rand_data.means,
        kappas_init=rand_data.kappas,
        random_state=rng,
    )
    vmf.fit(rand_data.X)
    # The initial von_mises_fisher parameters are not estimated. They are estimated for every
    # m_step.
    assert mock.call_count == vmf.n_iter_


# if __name__ == '__main__':
#     pytest.main([__file__])
