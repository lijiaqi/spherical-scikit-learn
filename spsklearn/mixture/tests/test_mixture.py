# Author: Jiaqi Li <lijiaqi.academic@outlook.com>
# License: BSD 3 clause

import numpy as np
import pytest

from spsklearn.mixture import vonMisesFisherMixture
import numpy as np


@pytest.mark.parametrize("estimator", [vonMisesFisherMixture()])
def test_von_mises_fisher_mixture_n_iter(estimator):
    # check that n_iter is the number of iteration performed.
    rng = np.random.RandomState(0)
    X = rng.rand(10, 5)
    X /= np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    max_iter = 1
    estimator.set_params(max_iter=max_iter)
    estimator.fit(X)
    assert estimator.n_iter_ == max_iter


@pytest.mark.parametrize("estimator", [vonMisesFisherMixture()])
def test_mixture_n_components_greater_than_n_samples_error(estimator):
    """Check error when n_components <= n_samples"""
    rng = np.random.RandomState(0)
    X = rng.rand(10, 5)
    X /= np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    estimator.set_params(n_components=12)

    msg = "Expected n_samples >= n_components"
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X)
