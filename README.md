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

### Quick Install (Recommended)
```bash
git clone git@github.com:lijiaqi/spherical-scikit-learn.git
cd spherical-scikit-learn
pip install -e .[dev]  # Install with development dependencies
```

### Alternative Installation Methods
```bash
# Install only runtime dependencies
pip install -e .

# Install with test dependencies
pip install -e .[test]

# Build wheel and install
python -m build
pip install dist/spherical_scikit_learn-*.whl
```

<!-- ## Generate documentations
```
cd doc
sphinx-apidoc -f -o ./source ../spsklearn/
make clean
make html
``` -->