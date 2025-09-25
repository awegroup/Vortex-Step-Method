import numpy as np
from .utils import jit_cross, jit_norm, jit_dot

# Example vectors for pre-compilation (dummy values)
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Force JIT compilation during package initialization
jit_cross(a, b)
jit_norm(a)
jit_dot(a, b)

# Expose these functions at the package level
__all__ = ["jit_cross", "jit_norm", "jit_dot"]
