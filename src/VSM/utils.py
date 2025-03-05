from numba import jit
import numpy as np


@jit(nopython=True)
def jit_cross(a, b):
    return np.cross(a, b)


@jit(nopython=True)
def jit_norm(value):
    # value = value.astype(np.float64)  # Convert to float64 (needed for pytest)
    return np.linalg.norm(value.astype(np.float64))


@jit(nopython=True)
def jit_dot(a, b):
    # a = a.astype(np.float64)  # Convert to float64 (needed for pytest)
    # b = b.astype(np.float64)  # Convert to float64 (needed for pytest)
    return np.dot(a.astype(np.float64), b.astype(np.float64))
    # return np.dot(a, b)
