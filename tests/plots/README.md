# Test Plots Directory

This directory contains plots generated during test execution.

## Usage

### Running Tests with Plot Generation
By default, tests will save plots to this directory instead of displaying them:

```bash
pytest tests/cluster/test_spherical_k_means_3D.py::test_spherical_k_means -v
```

### Displaying Plots During Development
To display plots instead of saving them (useful for development), set the `SHOW_PLOTS` environment variable:

```bash
SHOW_PLOTS=1 pytest tests/cluster/test_spherical_k_means_3D.py::test_spherical_k_means -v
```

## File Organization

- All test-generated plots are saved here to avoid cluttering the project root
- Files are automatically ignored by git (see `.gitignore`)
- The `plot_utils.py` module provides consistent plot handling across all tests

## Generated Files

- `test_spherical_k_means_3d_result.png` - 3D visualization of spherical k-means clustering results
