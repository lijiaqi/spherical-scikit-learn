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

import spsklearn
from spsklearn.cluster import (
    SphericalKMeans,
    spherical_k_means,
    spherical_k_means_plusplus,
)
from spsklearn.utils import draw_von_mises_fisher, draw_von_mises_fisher_mixture
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def plot_3d_scatter(data,ax=None,colour='#1f77b4',sz=30,el=20,az=50,sph=True,sph_colour="gray",sph_alpha=0.03,
                    eq_line=True,pol_line=True,grd=False):
    """
        plot_3d_scatter()
        =================

        Plots 3D samples on the surface of a sphere.

        INPUT:

            * data (array of floats of shape (N,3)) - samples of a spherical distribution such as von Mises-Fisher.
            * ax (axes) - axes on which the plot is constructed.
            * colour (string) - colour of the scatter plot.
            * sz (float) - size of points.
            * el (float) - elevation angle of the plot.
            * az (float) - azimuthal angle of the plot.
            * sph (boolean) - whether or not to inclde a sphere.
            * sph_colour (string) - colour of the sphere if included.
            * sph_alpha (float) - the opacity/alpha value of the sphere.
            * eq_line (boolean) - whether or not to include an equatorial line.
            * pol_line (boolean) - whether or not to include a polar line.
            * grd (boolean) - whether or not to include a grid.

        OUTPUT:

            * ax (axes) - axes on which the plot is contructed.
            * Plot of 3D samples on the surface of a sphere.

    """


    # The polar axis
    if ax is None:
        ax = plt.axes(projection='3d')

    # Check that data is 3D (data should be Nx3)
    d = np.shape(data)[1]
    if d != 3:
        raise Exception("data should be of shape Nx3, i.e., each data point should be 3D.")

    ax.scatter(data[:,0],data[:,1],data[:,2],s=5,c=colour)
    ax.view_init(el, az)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_zlim(-1.5,1.5)

    # Add a shaded unit sphere
    if sph:
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color=sph_colour,alpha=sph_alpha)

    # Add an equitorial line
    if eq_line:
        # t = theta, p = phi
        eqt = np.linspace(0,2*np.pi,50,endpoint=False)
        eqp = np.linspace(0,2*np.pi,50,endpoint=False)
        eqx = 2*np.sin(eqt)*np.cos(eqp)
        eqy = 2*np.sin(eqt)*np.sin(eqp) - 1
        eqz = np.zeros(50)

        # Equator line
        ax.plot(eqx,eqy,eqz,color="k",lw=1)

    # Add a polar line
    if pol_line:
        # t = theta, p = phi
        eqt = np.linspace(0,2*np.pi,50,endpoint=False)
        eqp = np.linspace(0,2*np.pi,50,endpoint=False)
        eqx = 2*np.sin(eqt)*np.cos(eqp)
        eqy = 2*np.sin(eqt)*np.sin(eqp) - 1
        eqz = np.zeros(50)

        # Polar line
        ax.plot(eqx,eqz,eqy,color="k",lw=1)

    # Draw a centre point
    ax.scatter([0], [0], [0], color="k", s=sz)    

    # Turn off grid
    ax.grid(grd)

    # Ticks
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    ax.set_zticks([-1,0,1])

    return ax

# Drawing a fancy vector see Ref. [7] 
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_arrow(point,ax,colour="red"):
    """
        plot_arrow(point,ax,colour="red")
        ==============================
        Plots a 3D arrow on the axes ax from the origin to the point mu. 
        INPUT: 
        
            * point (array of floats of shape (3,1)) - a 3D point.
            * ax (axes) - axes on which the plot is constructed.
            * colour (string) - colour of the arrow. 
    """
    # Can use quiver for a simple arrow
    #ax.quiver(0,0,0,point[0],point[1],point[2],length=1.0,color=colour,pivot="tail")
    
    # Fancy arrow 
    a = Arrow3D([0, point[0]], [0, point[1]], [0, point[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color=colour)
    ax.add_artist(a)
    
    return ax

class RandomData:
    def __init__(self, rng, n_samples=1500, n_components=4, n_features=3, scale=20):
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
    def __init__(self, rng, n_samples=2000):
        self.n_samples = n_samples
        self.n_components = 4
        self.n_features = 3

        # Set 1
        mu1 = [1,1,0]
        kappa1 = 50
        # Set 2
        mu2 = [0,0,1]
        kappa2 = 20
        # Set 3
        mu3 = [0,0,-1]
        kappa3 = 100
        # set 4
        mu4 = [-10,0,-1]
        kappa4 = 200

        self.means = np.stack([mu1, mu2, mu3, mu4], axis=0, dtype=float)
        self.kappas = np.stack([kappa1, kappa3, kappa3, kappa4], axis=0, dtype=float)
        self.weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

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

    centers, labels, inertia_, n_iter_ = spherical_k_means(
        X, 
        n_clusters=rand_data.n_components, 
        init="spherical-k-means++", 
        verbose=1, return_n_iter=True)
    print("cluster_centers_", centers)
    print("labels_", labels)
    print("inertia_", inertia_)
    print("n_iter_", n_iter_)
    ## visualization
    marker_s = 0.3
    fig = plt.figure(figsize=(9,4))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_3d_scatter(X[Y==0], ax0, colour='purple', sz=marker_s)
    plot_3d_scatter(X[Y==1], ax0, colour='green', sz=marker_s)
    plot_3d_scatter(X[Y==2], ax0, colour='orange', sz=marker_s)
    plot_3d_scatter(X[Y==3], ax0, colour='pink', sz=marker_s)
    ax0.scatter(rand_data.means[:,0], rand_data.means[:,1], rand_data.means[:,2], marker='s', c='red')

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_3d_scatter(X[labels==0], ax1, colour='purple', sz=marker_s)
    plot_3d_scatter(X[labels==1], ax1, colour='green', sz=marker_s)
    plot_3d_scatter(X[labels==2], ax1, colour='orange', sz=marker_s)
    plot_3d_scatter(X[labels==3], ax1, colour='pink', sz=marker_s)
    ax1.scatter(centers[:,0], centers[:,1], centers[:,2], marker='x', c='red')
    plt.show()

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

