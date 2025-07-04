from numba import jit
import numpy as np


@jit(nopython=True)
def jit_cross(a, b):
    return np.cross(a, b)


@jit(nopython=True)
def jit_norm(value):
    return np.linalg.norm(value.astype(np.float64))


@jit(nopython=True)
def jit_dot(a, b):
    return np.dot(a.astype(np.float64), b.astype(np.float64))


def intersect_line_with_plane(x_cp, F_unit, plane_point, plane_normal):
    numerator = np.dot(plane_normal, (plane_point - x_cp))
    denominator = np.dot(plane_normal, F_unit)
    if np.abs(denominator) < 1e-6:
        return None  # No intersection
    lam = numerator / denominator
    intersection = x_cp + lam * F_unit
    return intersection

def point_in_triangle(pt, v0, v1, v2, tol=1e-8):
    # Barycentric coordinate check with tolerance
    u = v1 - v0
    v = v2 - v0
    w = pt - v0

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    denom = uv * uv - uu * vv
    if np.abs(denom) < 1e-12:
        return False  # Degenerate triangle

    s = (uv * wv - vv * wu) / denom
    t = (uv * wu - uu * wv) / denom

    return (s >= -tol) and (t >= -tol) and (s + t <= 1 + tol)


def point_in_quad(pt, corners):
    """
    Checks if a point is inside a quadrilateral.
    corners: array of shape (4, 3)
    """
    return point_in_triangle(pt, corners[0], corners[1], corners[2]) or point_in_triangle(pt, corners[0], corners[2], corners[3])
