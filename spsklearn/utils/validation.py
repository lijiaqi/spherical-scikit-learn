import numpy as np
from sklearn.preprocessing import normalize

def check_spherical_array(array, spherical_axis=1):
    _, array_norms = normalize(array, norm='l2', axis=spherical_axis, return_norm=True)
    if not np.allclose(array_norms, 1.0):
        raise ValueError(
            "Array should be L2-normalized to 1 among axis={:d}, but got max_norm = {:.5f}"
            "  and min_norm = {:.5f}.".format(
                spherical_axis, 
                np.max(array_norms), 
                np.min(array_norms))
        )