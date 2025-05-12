# Spherical-Scikit-Learn: a toolkit for spherical k-means and spherical von Mises-Fisher mixture model.

## Short Introduction
This package was unintentionally built when I explored a problem about spherical clustering. Compared to some existing packages [spherecluster](https://github.com/jasonlaska/spherecluster), this toolkit was implemented with `Cython`-like basic operators similar to [scikit-learn](https://scikit-learn.org/stable/). The APIs are `scikit-learn`-like and the documentations can be found at [https://lijiaqi.github.io/spherical-scikit-learn/](https://lijiaqi.github.io/spherical-scikit-learn/).

Currently, this package supports:

- **Spherical K-Means**: Conduct spherical clustering on a hypersphere:
```python
from spsklearn.cluster import SphericalKMeans
spkm = SphericalKMeans()
spkm.fit(data)
...
```

- **von Mises-Fisher Mixture Model**: use a mixture of von Mises-Fisher distributions to model the data on a hypersphere.
```python
from spsklearn.mixture import vonMisesFisherMixture
vmfmm = vonMisesFisherMixture(n_components=3)
vmfmm.fit(data)
...
```

## Build and Install
```
## build the package (into .wheel) and install
git clone git@github.com:lijiaqi/spherical-scikit-learn.git
cd spherical-scikit-learn
pip install -r requirements.txt

python -m build
pip install dist/spherical_scikit_learn-x.x.x-xxx.whl

## Or, directly install
pip install .
```

<!-- ## Generate documentations
```
cd doc
sphinx-apidoc -f -o ./source ../spsklearn/
make clean
make html
``` -->