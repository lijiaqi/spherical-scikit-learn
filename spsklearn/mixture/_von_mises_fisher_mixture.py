"""von Mises-Fisher Mixture Model."""

# Author: Jiaqi Li <lijiaqi.academic@outlook.com>
# License: BSD 3 clause
import warnings
import numpy as np
import scipy as sp
from scipy import linalg
from scipy.special import ive as expBessel
from scipy.special import logsumexp
from scipy.stats import vonmises_fisher
from scipy import optimize

from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.extmath import row_norms
from sklearn.mixture._base import BaseMixture, _check_shape
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, DensityMixin, _fit_context
from sklearn.exceptions import ConvergenceWarning
from ..cluster import SphericalKMeans, spherical_k_means_plusplus
from ..utils import check_spherical_array

###############################################################################
# von Mises-Fisher mixture shape checkers used by the vonMisesFisherMixture class

def check_nan(array):
    return np.isnan(array).any()

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like of shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(weights, (n_components,), "weights")

    # check range
    if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
        raise ValueError(
            "The parameter 'weights' should be in the range "
            "[0, 1], but got max value %.5f, min value %.5f"
            % (np.min(weights), np.max(weights))
        )

    # check normalization
    if not np.allclose(np.abs(1.0 - np.sum(weights)), 0.0):
        raise ValueError(
            "The parameter 'weights' should be normalized, but got sum(weights) = %.5f"
            % np.sum(weights)
        )
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), "means")
    check_spherical_array(means, spherical_axis=1)
    return means


def _check_kappas(kappas, n_components):
    """Check a kappas vector is positive."""
    kappas = check_array(kappas, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(kappas, (n_components,), "kappas")
    if np.any(np.less_equal(kappas, 0.0)):
        raise ValueError("'kappas' should be all positive")
    return kappas


###############################################################################
# von Mises-Fisher mixture parameters estimators (used by the M-Step)

def _estimate_von_mises_fisher_parameters(X, resp):
    """Estimate the von Mises-Fisher distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    kappas : array-like
        The concentration parameter kappas of the current components.
        
    References: 
    [1] Clustering on the Unit Hypersphere using von Mises-Fisher Distributions. JMLR 2025.
    """
    n_features = X.shape[1]
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    resultant = np.dot(resp.T, X) / nk[:, np.newaxis]
    means, res_len = normalize(resultant, norm="l2", axis=1, return_norm=True)
    ### kappa is the solution to the equation:
    # r = I[d/2](kappa) / I[d/2 -1](kappa)
    #   = I[d/2](kappa) * exp(-kappa) / I[d/2 -1](kappa) * exp(-kappa)
    #   = ive(d/2, kappa) / ive(d/2 -1, kappa)
    ### if solve with Newton's method, according to [1]
    #    let A[d](kappa) = I[d/2](kappa) / I[d/2 -1](kappa)
    #    then the differientiation  A^{\prime}[d](kappa) = 1 - A[d](kappa) **2 - (d-1)*A[d](kappa)/kappa
    ####### the following is from scipy.stats.vonmises_fisher.fit()
    ####### expBessel(d, k) will be 0 (underflow) for large d (e.g., d=300)
    # kappas = np.zeros_like(res_len)
    # for idx, r in enumerate(res_len):
    #     def eq_for_kappa(k):
    #         return expBessel(n_features/2, k)/expBessel(n_features/2-1, k) - r
    #     sol = optimize.root_scalar(eq_for_kappa, method="brentq", bracket=(1e-8, 1e9))
    #     kappas[idx] = sol.root

    # according to [1], kappa = (r*d-r**3)/(1-r**2) 
    kappas = (res_len * n_features - res_len**3) / (1 - res_len**2)
    return nk, means, kappas


def _flipudlr(array):
    """Reverse the rows and columns of an array."""
    return np.flipud(np.fliplr(array))


###############################################################################
# von Mises-Fisher mixture probability estimators
def _estimate_log_von_mises_fisher_prob(X, means, kappas):
    """Estimate the log von Mises-Fisher probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    kappas : array-like of shape (n_components,)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
        C = kappa**(d/2-1)/( (2*pi)**(d/2) * I_{d/2-1}(kappa)  )
          = kappa**(d/2-1) * exp(-kappa) 
                /( (2*pi)**(d/2) * I_{d/2-1}(kappa)*exp(-kappa)  )
            = kappa**(d/2-1) * exp(-kappa) 
                /( (2*pi)**(d/2) * scipy.special.ive(d/2-1, kappa) )
        log C = (d/2-1)*\log(kappa) - kappa - (d/2)*\log(2*pi) - ive(d/2-1, kappa)
        log p(x; mu, kappa) = kappa * X.dot(mu.T) + log_C
    Notes:
        ive(r, z) = iv(r, z) * exp(-abs(z.real))
        ive is useful for large arguments z: 
            for these, iv easily overflows, 
            while ive does not due to the exponential scaling.
    """
    assert means.shape[0]==kappas.shape[0], "means.shape[0]!=kappa.shape[0] !"
    assert means.shape[1]==X.shape[1], "means.shape[1]!=X.shape[1] !"
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    log_C = (n_features/2-1)*np.log(kappas) - kappas - n_features/2 * np.log(2*np.pi) \
        -  np.log(expBessel(n_features/2-1, kappas))
    log_prob = kappas[np.newaxis, :] * np.dot(X, means.T) + log_C[np.newaxis, :]
    return log_prob


class vonMisesFisherMixture(BaseMixture):
    """von Mises-Fisher Mixture.

    Representation of a von Mises-Fisher mixture model probability distribution.
    This class allows to estimate the parameters of a von Mises-Fisher mixture
    distribution.

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.

    init_params : {'spherical-k-means', 'spherical-k-means++', 'random', 'random_from_data'}, \
    default='spherical-k-means++'
        The method used to initialize the weights, the means and the
        precisions.
        String must be one of:

        - 'spherical-k-means' : responsibilities are initialized using spherical k-means.
        - 'spherical-k-means++' : use the spherical k-means++ method to initialize.
        - 'random' : responsibilities are initialized randomly.
        - 'random_from_data' : initial means are randomly selected data points.

    weights_init : array-like of shape (n_components, ), default=None
        The user-provided initial weights.
        If it is None, weights are initialized using the `init_params` method.

    means_init : array-like of shape (n_components, n_features), default=None
        The user-provided initial means,
        If it is None, means are initialized using the `init_params` method.

    kappas_init : array-like, default=None
        The user-provided initial concentration parameter kappas.
        If it is None, kappas are initialized using the 'init_params'
        method.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    kappas_ : array-like
        The concentration parameter kappas of each mixture component.

    converged_ : bool
        True when convergence of the best fit of EM was reached, False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> import numpy as np
    >>> from spsklearn.mixture import vonMisesFisherMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> vmf = vonMisesFisherMixture(n_components=2, random_state=0).fit(X)
    >>> vmf.means_
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> vmf.predict([[0, 0], [12, 3]])
    array([1, 0])
    """

    _parameter_constraints: dict = {
        **BaseMixture._parameter_constraints,
        "weights_init": ["array-like", None],
        "means_init": ["array-like", None],
        "kappas_init": ["array-like", None],
    }
    _parameter_constraints.update({
        "init_params": [
            StrOptions({"spherical-k-means", "random", "random_from_data", "spherical-k-means++"})
        ],
    })

    def __init__(
        self,
        n_components=1,
        *,
        tol=1e-3,
        max_iter=100,
        n_init=1,
        init_params="spherical-k-means++",
        weights_init=None,
        means_init=None,
        kappas_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
            reg_covar=0, # discarded in vMF
        )

        self.weights_init = weights_init
        self.means_init = means_init
        self.kappas_init = kappas_init

    def _check_parameters(self, X):
        """Check the von Mises-Fisher mixture parameters are well defined."""
        _, n_features = X.shape

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init, self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init, self.n_components, n_features)

        if self.kappas_init is not None:
            self.kappas_init = _check_kappas(
                self.kappas_init,
                self.n_components,
            )

    def _initialize_parameters(self, X, random_state):
        # If all the initial parameters are all provided, then there is no need to run
        # the initialization.
        compute_resp = (
            self.weights_init is None
            or self.means_init is None
            or self.kappas_init is None
        )
        if compute_resp:
            self._initialize_parameters_algorithm(X, random_state)
        else:
            self._initialize(X, None)

    def _initialize_parameters_algorithm(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape

        if self.init_params == "spherical-k-means":
            resp = np.zeros((n_samples, self.n_components))
            spkmeans = SphericalKMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                ).fit(X)
            indices = spkmeans.labels_
            resp[np.arange(n_samples), indices] = 1
        elif self.init_params == "random":
            resp = random_state.uniform(size=(n_samples, self.n_components))
            resp /= resp.sum(axis=1)[:, np.newaxis]
        elif self.init_params == "random_from_data":
            resp = np.zeros((n_samples, self.n_components))
            center_indices = random_state.choice(
                n_samples, size=self.n_components, replace=False
            )
            centers = X[center_indices, :]
            indices = np.argmax(np.dot(X, centers.T), axis=1)
            resp[np.arange(n_samples), indices] = 1
        elif self.init_params == "spherical-k-means++":
            resp = np.zeros((n_samples, self.n_components))
            centers, _ = spherical_k_means_plusplus(
                X,
                self.n_components,
                random_state=random_state,
            )
            indices = np.argmax(np.dot(X, centers.T), axis=1)
            resp[np.arange(n_samples), indices] = 1

        self._initialize(X, resp)

    def _initialize(self, X, resp):
        """Initialization of the von Mises-Fisher mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape
        weights, means, kappas = None, None, None
        if resp is not None:
            weights, means, kappas = _estimate_von_mises_fisher_parameters(X, resp)
            if self.weights_init is None:
                weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init
        self.kappas_ = kappas if self.kappas_init is None else self.kappas_init

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        check_spherical_array(X, spherical_axis=1)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm_avg, log_resp = self._e_step(X)
                    # log_resp = log (alpha_i * P(x_j|z=i))
                    # log_prob_norm_avg = average_j( log \sum_i (alpha_i * P(x_j|z=i)) )
                    self._m_step(X, log_resp)
                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm_avg)

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        self.converged_ = True
                        break

                self._print_verbose_msg_init_end(lower_bound)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X
                average_j( \log \sum_{i=1}^C (\alpha_i * P(x_j|z=i)) )

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
                \log (\alpha_i * P(x_j|z=i))
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        self.weights_, self.means_, self.kappas_ = _estimate_von_mises_fisher_parameters(
            X, np.exp(log_resp)
        )
        self.weights_ /= self.weights_.sum()

    def _estimate_log_prob(self, X):
        return _estimate_log_von_mises_fisher_prob(
            X, self.means_, self.kappas_
        )

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            \log \sum_{i=1}^C (\alpha_i * P(x_j|z=i))

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
            \log (\alpha_i * P(x_j|z=i))
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        log_prob = self._estimate_log_prob(X)
        log_weights = self._estimate_log_weights()
        return log_prob + log_weights[np.newaxis, :]

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.kappas_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.kappas_,
        ) = params

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        kappa_params = self.n_components
        mean_params = n_features * self.n_components
        return int(kappa_params + mean_params + self.n_components - 1)
    ### fir BIC and AIC, 
    ###     ref: https://scikit-learn.org/stable/modules/linear_model.html#mathematical-details
    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the BIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        bic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(X.shape[0])

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the AIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

    def sample(self, n_samples=1):
        """Generate random samples from the fitted von Mises-Fisher mixture distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        y : array, shape (nsamples,)
            Component labels.
        """
        check_is_fitted(self)

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)
        X = np.vstack(
            [
                vonmises_fisher.rvs(mean, kappa, size=int(sample), random_state=rng)
                for (mean, kappa, sample) in zip(
                    self.means_, self.kappas_, n_samples_comp
                )
            ]
        )

        y = np.concatenate(
            [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )

        return (X, y)