# Masure Regression Models

This directory contains the trained regression models for the Masure aerodynamic coefficient prediction.

## Required Files

Place the following three model files in this directory:

- `ET_re5e6.pkl` - Extra Trees model trained for Reynolds number 5×10⁶
- `ET_re1e6.pkl` - Extra Trees model trained for Reynolds number 1×10⁶  
- `ET_re2e7.pkl` - Extra Trees model trained for Reynolds number 2×10⁷

## File Sources

These model files are **not included** in the git repository due to their large size. They should be obtained from the original source or trained separately.

## Performance Optimizations

### Model Caching
- Models are **automatically cached** in memory after first load
- Subsequent uses of the same Reynolds number reuse the cached model
- **Significant speedup**: First load ~5s, subsequent loads ~0.1s

### Batch Processing
- Multiple airfoils with same Reynolds number are processed in batches
- **Batch advantage**: More efficient than individual processing
- Automatically used when multiple `masure_regression` airfoils are present

### Aerodynamic Data Caching
- Computed polar data is cached to `data/cache/` for ultra-fast repeated runs
- **Dramatic speedup**: ~300x faster on cache hits (5s → 0.03s)
- Cache automatically invalidates when parameters change

## Usage

The models are automatically loaded when using the `masure_regression` airfoil type in the configuration files. The appropriate model is selected based on the Reynolds number specified in the configuration.

## Model Input Format

The models expect input in the following format:
- Input: `[t, cx, cy, r, LE, camTE, a]` where:
  - `t`: thickness parameter
  - `cx`: eta parameter  
  - `cy`: kappa parameter
  - `r`: delta parameter
  - `LE`: lambda parameter
  - `camTE`: phi parameter
  - `a`: angle of attack in degrees

## Model Output Format

The models output aerodynamic coefficients in the following order:
- Output: `[Cd, Cl, CmPitch]` where:
  - `Cd`: drag coefficient
  - `Cl`: lift coefficient
  - `CmPitch`: pitching moment coefficient

## Compatibility

The models were trained with an older version of scikit-learn. The code includes compatibility patches to handle version differences, but for best results, consider retraining the models with the current sklearn version if compatibility issues arise.

## Troubleshooting

### Model Loading Issues
If you encounter model loading errors:

1. **File not found**: Ensure all three `.pkl` files are in `data/models/`
2. **Version compatibility**: Models were trained with older scikit-learn
   - Code includes automatic compatibility patches
   - Consider retraining models if issues persist
3. **Memory issues**: Models are large; ensure sufficient RAM available
4. **Permissions**: Verify read permissions on model files

### Performance Issues
If model performance is slow:
- **First run slowness**: Normal - models are cached after first load
- **Repeated slowness**: Check if caching is working (should see cache messages)
- **Memory usage**: Each model consumes significant memory when loaded

### Configuration Examples
Example masure_regression configuration:
```yaml
wing_airfoils:
  reynolds: !!float 5e6  # Must match available model
  data:
    - [1, masure_regression, {t: 0.07, eta: 0.2, kappa: 0.95, delta: -2, lambda: 0.65, phi: 0.25}]
```

Available Reynolds numbers: `5e6`, `1e6`, `2e7`
