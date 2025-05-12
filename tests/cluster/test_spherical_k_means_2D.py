# Author: Jiaqi Li <xuewei4d@gmail.com>
# License: BSD 3 clause
import numpy as np
import pytest

import spsklearn
from spsklearn.cluster import (
    SphericalKMeans,
    spherical_k_means,
    spherical_k_means_plusplus,
)
from spsklearn.utils import draw_von_mises_fisher, draw_von_mises_fisher_mixture


class RandomData:
    def __init__(self, rng, n_samples=1500, n_components=3, n_features=2, scale=20):
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_features = n_features

        self.means = rng.randn(n_components, n_features)
        self.means /= np.linalg.norm(self.means, ord=2, axis=1, keepdims=True)
        self.kappas = rng.rand(n_components) * scale
        self.weights = rng.rand(n_components)
        self.weights /= self.weight.sum()
        self.X, self.Y = draw_von_mises_fisher_mixture(
            self.means, self.kappas, self.n_samples, self.weights, rng
        )


class SemiRandomData:
    def __init__(self, rng, n_samples=1500):
        self.n_samples = n_samples
        self.n_components = 3
        self.n_features = 2

        # Set 1
        mu1 = [1, 1]
        kappa1 = 50
        # Set 2
        mu2 = [-1, 1]
        kappa2 = 20
        # Set 3
        mu3 = [-1, 0]
        kappa3 = 100
        self.means = np.stack([mu1, mu2, mu3], axis=0, dtype=float)
        self.kappas = np.stack([kappa1, kappa3, kappa3], axis=0, dtype=float)
        self.weights = np.array([1.0, 1.0, 1.0], dtype=float)

        self.weights = self.weights / self.weights.sum()
        self.means /= np.linalg.norm(self.means, ord=2, axis=1, keepdims=True)
        self.X, self.Y = draw_von_mises_fisher_mixture(
            self.means, self.kappas, self.n_samples, self.weights, rng
        )


def test_spherical_k_means_plusplus():
    n_samples = 1500
    rng = np.random.RandomState(0)
    rand_data = SemiRandomData(rng, n_samples=n_samples)
    X = rand_data.X
    Y = rand_data.Y
    centers, indices = spherical_k_means_plusplus(
        X=X,
        n_clusters=rand_data.n_components,
        sample_weight=np.ones(n_samples),
        random_state=rng,
    )
    print(centers)
    print(indices)
    ### visualization
    # import matplotlib.pyplot as plt
    # marker_s = 0.3
    # fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    # ax[0].scatter(0, 0, marker=".", c="black", label="Origin")
    # ax[0].scatter(
    #     rand_data.means[:, 0], rand_data.means[:, 1], marker="s", s=marker_s * 200, c="red", label="vMF centers"
    # )
    # ax[0].scatter(X[Y==0, 0], X[Y==0, 1], s=marker_s, c="purple", label="set 1")
    # ax[0].scatter(X[Y==1, 0], X[Y==1, 1], s=marker_s, c="green", label="set 2")
    # ax[0].scatter(X[Y==2, 0], X[Y==2, 1], s=marker_s, c="orange", label="set 3")
    # ax[0].set_xlim(-1.2, 1.2)
    # ax[0].set_ylim(-1.2, 1.2)
    # ax[0].legend()

    # ax[1].scatter(0, 0, marker=".", c="black", label="Origin")
    # ax[1].scatter(
    #     centers[:, 0], centers[:, 1], marker="x", s=marker_s * 200, c="red", label="init with k-means++"
    # )
    # ax[1].scatter(X[:, 0], X[:, 1], s=marker_s)
    # ax[1].set_xlim(-1.2, 1.2)
    # ax[1].set_ylim(-1.2, 1.2)
    # ax[1].legend()
    # plt.show()
    # # fig.savefig("draw_SKM_init_kmeans_plusplus.pdf")


def test_SKM_init_random():
    n_samples = 1500
    rng = np.random.RandomState(1)
    rand_data = SemiRandomData(rng, n_samples=n_samples)
    X = rand_data.X
    Y = rand_data.Y

    skm = SphericalKMeans(n_clusters=rand_data.n_components, init="random", verbose=1)
    skm.fit(X)
    print("cluster_centers_", skm.cluster_centers_)
    print("labels_", skm.labels_)
    print("inertia_", skm.inertia_)
    print("n_iter_", skm.n_iter_)
    print("n_features_in_", skm.n_features_in_)
    

def test_spherical_k_means():
    n_samples = 1500
    rng = np.random.RandomState(1)
    rand_data = SemiRandomData(rng, n_samples=n_samples)
    X = rand_data.X
    Y = rand_data.Y

    cluster_centers_, labels_, inertia_, n_iter_ = spherical_k_means(
        X, 
        n_clusters=rand_data.n_components, 
        init="spherical-k-means++", 
        verbose=1, return_n_iter=True)
    print("cluster_centers_", cluster_centers_)
    print("labels_", labels_)
    print("inertia_", inertia_)
    print("n_iter_", n_iter_)

def test_SKM_init_spherical_kmpp():
    n_samples = 1500
    rng = np.random.RandomState(1)
    rand_data = SemiRandomData(rng, n_samples=n_samples)
    X = rand_data.X
    Y = rand_data.Y

    skm = SphericalKMeans(n_clusters=rand_data.n_components, init="spherical-k-means++", verbose=1)
    skm.fit(X)
    print("cluster_centers_", skm.cluster_centers_)
    print("labels_", skm.labels_)
    print("inertia_", skm.inertia_)
    print("n_iter_", skm.n_iter_)
    print("n_features_in_", skm.n_features_in_)
    
    # ## visualization
    # import matplotlib.pyplot as plt
    # marker_s = 0.3
    # fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    # ax[0].scatter(0, 0, marker=".", c="black", label="Origin")
    # ax[0].scatter(
    #     rand_data.means[:, 0], rand_data.means[:, 1], marker="s", s=marker_s * 200, c="red", label="vMF centers"
    # )
    # ax[0].scatter(X[Y==0, 0], X[Y==0, 1], s=marker_s, c="purple", label="set 1")
    # ax[0].scatter(X[Y==1, 0], X[Y==1, 1], s=marker_s, c="green", label="set 2")
    # ax[0].scatter(X[Y==2, 0], X[Y==2, 1], s=marker_s, c="orange", label="set 3")
    # ax[0].set_xlim(-1.2, 1.2)
    # ax[0].set_ylim(-1.2, 1.2)
    # ax[0].legend()

    # ax[1].scatter(0, 0, marker=".", c="black", label="Origin")
    # ax[1].scatter(
    #     skm.cluster_centers_[:, 0], skm.cluster_centers_[:, 1], marker="x", s=marker_s * 200, c="red", label="cluster centers"
    # )
    # ax[1].scatter(X[skm.labels_==0, 0], X[skm.labels_==0, 1], s=marker_s, c="purple", label="set 1")
    # ax[1].scatter(X[skm.labels_==1, 0], X[skm.labels_==1, 1], s=marker_s, c="green", label="set 2")
    # ax[1].scatter(X[skm.labels_==2, 0], X[skm.labels_==2, 1], s=marker_s, c="orange", label="set 3")
    # ax[1].set_xlim(-1.2, 1.2)
    # ax[1].set_ylim(-1.2, 1.2)
    # ax[1].legend()
    # plt.show()
    # fig.savefig("draw_SKM_results_spherical_kmeans_plusplus.pdf")


