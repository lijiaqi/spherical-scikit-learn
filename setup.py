from setuptools import setup, Extension, find_packages
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
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
    zip_safe=False,
)