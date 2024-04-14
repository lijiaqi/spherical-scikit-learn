## Spherical-Scikit-Learn: a toolkit for spherical k-means and spherical von Mises-Fisher mixture model.

For spherical k-means:
```
import spsklearn
spkm = spsklearn.cluster.SphericalKMeans()
```

For von Mises-Fisher Mixture Model:
```
import spsklearn
vmm = spsklearn.mixture.vonMisesFisherMixture()
```

## Install
```
python setup.py build_ext --inplace
python setup.py install
```