# Aerodynamic Data Cache

This directory contains cached aerodynamic polar data to improve performance for repeated VSM runs.

## How It Works

When you run VSM with expensive airfoil computations (masure_regression, neuralfoil), the system automatically:

1. **Generates a unique cache key** based on your configuration (airfoil parameters, alpha range, Reynolds number)
2. **Saves computed polar data** to a cache file named: `aerodynamic_cache_YYYYMMDD_<hash>.pkl`
3. **Loads from cache** on subsequent runs with the same configuration

## Performance Benefits

- **First run**: Normal computation time (e.g., 5-10 seconds)
- **Subsequent runs**: Ultra-fast loading (e.g., 0.03 seconds) - **~300x speedup!**

## Cache Files

Cache files are automatically generated with the naming pattern:
```
aerodynamic_cache_YYYYMMDD_<hash>.pkl
```

Where:
- `YYYYMMDD` is the creation date
- `<hash>` is a unique identifier based on the configuration parameters

## Cache Management

### Automatic Cleanup
- Cache files are **date-based** - files from previous days are automatically deleted
- This ensures fresh computation when you resume work on a different day
- No manual cleanup required

### Cache Invalidation
Cache automatically invalidates when you change:
- Airfoil parameters (t, eta, kappa, delta, lambda, phi)
- Alpha range (min, max, step)
- Reynolds number
- Airfoil types or IDs

## What Gets Cached

Only expensive aerodynamic computations are cached:
- ✅ `masure_regression` - Machine learning model predictions
- ✅ `neuralfoil` - Neural network aerodynamic analysis

Fast computations are not cached:
- ❌ `breukels_regression` - Polynomial calculations  
- ❌ `inviscid` - Simple analytical calculations
- ❌ `polars` - CSV file loading

## Troubleshooting

### Cache Not Working
If caching isn't working as expected:
1. Check that `file_path` is provided to `BodyAerodynamics.instantiate()`
2. Ensure you're using cacheable airfoil types (masure_regression, neuralfoil)
3. Verify the cache directory has write permissions

### Force Fresh Computation
To force fresh computation (bypass cache):
1. **Change the date**: Cache auto-expires daily
2. **Modify parameters**: Any parameter change invalidates cache
3. **Delete cache files**: Manually remove `aerodynamic_cache_*.pkl` files

### Cache Size
- Typical cache file size: 10-100 KB per airfoil configuration
- Large alpha ranges or many airfoils will create larger files
- Consider disk space if running many different configurations

## Cache Invalidation

Cache files are automatically invalidated when:
- Any configuration parameter changes (different hash)
- The date changes (different date in filename)
- Alpha range or Reynolds number changes

## File Format

Cache files contain:
```python
{
    'metadata': {
        'creation_time': '2025-07-08T10:30:00',
        'config_hash': 'abc123...',
        'alpha_range': [-10, 30, 1],
        'reynolds': 5e6,
        'airfoil_types': ['masure_regression', 'neuralfoil', ...]
    },
    'airfoil_polars': {
        'airfoil_1': numpy_array,  # Shape: (n_alpha, 4) [alpha, CL, CD, CM]
        'airfoil_2': numpy_array,
        ...
    }
}
```

## Performance Impact

Expected speedup for repeated runs:
- **First run**: Normal time (3-6 seconds)
- **Subsequent runs**: ~0.1-0.2 seconds (25-50x faster)
