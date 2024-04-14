from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Build import cythonize

import numpy as np

extensions = [
    Extension(
        "spsklearn.cluster._spherical_k_means_common",
        sources=["spsklearn/cluster/_spherical_k_means_common.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
    ),
    Extension(
        "spsklearn.cluster._spherical_k_means_lloyd",
        sources=["spsklearn/cluster/_spherical_k_means_lloyd.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="spherical-scikit-learn",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    version="0.1.0",
    install_requires=["scikit-learn>=1.4.1.post1", "numpy", "scipy>=0.11.0"],
    zip_safe=False,
    packages=find_packages(),
)
