import random
import numpy as np
import scipy
from scipy.stats import vonmises_fisher
from .validation import check_spherical_array

seed = 1
np.random.seed(seed)
random.seed(seed)


def draw_von_mises_fisher(mu, kappa, n_samples, seed=0):
    if kappa <= 0:
        raise ValueError("Concentration 'kappa' must be positive (>0)!")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("'n_samples' must be a positive integer!")
    mu_norm = np.linalg.norm(mu, ord=2, keepdims=True)
    if not np.allclose(mu_norm, 1.0):
        raise ValueError("Centroid 'mu' must be on a unit sphere (|\mu|=1)!")
    vmf = vonmises_fisher(mu=mu, kappa=kappa, seed=seed)
    samples = vmf.rvs(n_samples, random_state=seed)  # shape=(n_samples, n_features)
    return samples


def draw_von_mises_fisher_mixture(mus, kappas, n_samples, weights, seed=0):
    """_summary_

    Args:
        mus: shape=(n_components, n_features)
        kappas: shape=(n_components,)
        n_samples (int):
        weights: shape=(n_components,)
    """
    assert mus.shape[0] == len(kappas)
    assert len(weights) == len(kappas)
    check_spherical_array(mus, spherical_axis=1)
    if not np.allclose(np.sum(weights), 1.0):
        raise ValueError("'weights' should be summed to one.")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("'n_samples' must be a positive integer!")
    samples = []
    for _, (w, m, k) in enumerate(zip(weights, mus, kappas)):
        samples.append(
            draw_von_mises_fisher(m, k, n_samples=int(np.round(w * n_samples)), seed=seed)
        )
    samples = np.concatenate(samples, axis=0)

    labels = np.hstack(
        [
            np.full(int(np.round(w * n_samples)), k, dtype=int)
            for k, w in enumerate(weights)
        ]
    )
    return samples, labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D = 2
    N_per_k = 500
    K = 3
    N = N_per_k * K
    marker_s = 0.3

    # Set 1
    mu1 = [1, 1]
    kappa1 = 50
    # Set 2
    mu2 = [-1, 1]
    kappa2 = 20
    # Set 3
    mu3 = [-1, 0]
    kappa3 = 100

    mus = np.stack([mu1, mu2, mu3], axis=0)
    mus /= np.linalg.norm(mus, axis=1, keepdims=True)
    kappas = np.stack([kappa1, kappa3, kappa3], axis=0)
    weights = np.array([1.0, 1.0, 1.0])
    weights /= weights.sum()

    samples, labels = draw_von_mises_fisher_mixture(mus, kappas, n_samples=N, weights=weights, seed=0)
    print(samples.shape)

    ## visualization
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].scatter(0, 0, marker=".", c="black", label="Origin")
    ax[0].scatter(
        mus[:, 0], mus[:, 1], marker="x", s=marker_s * 200, c="red", label="vMF centers"
    )
    ax[0].scatter(samples[labels==0, 0], samples[labels==0, 1], s=marker_s, c="purple", label="set 1")
    ax[0].scatter(samples[labels==1, 0], samples[labels==1, 1], s=marker_s, c="green", label="set 2")
    ax[0].scatter(samples[labels==2, 0], samples[labels==2, 1], s=marker_s, c="orange", label="set 3")
    ax[0].set_xlim(-1.2, 1.2)
    ax[0].set_ylim(-1.2, 1.2)
    ax[0].legend()

    ax[1].scatter(0, 0, marker=".", c="black", label="Origin")
    ax[1].scatter(
        mus[:, 0], mus[:, 1], marker="x", s=marker_s * 200, c="red", label="vMF centers"
    )
    ax[1].scatter(samples[:, 0], samples[:, 1], s=marker_s)
    ax[1].set_xlim(-1.2, 1.2)
    ax[1].set_ylim(-1.2, 1.2)
    ax[1].legend()
    # fig.savefig("draw_vmf_mixture.pdf")
