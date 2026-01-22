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


def compute_effective_section_axes(y_airf_array, z_airf_array, eps=1e-12):  #TODO: NEW
    """Project chordwise directions into the plane normal to spanwise direction.

    Returns effective (x, y) axes and the cosine of local sweep (projection magnitude).
    """
    y_airf_array = np.asarray(y_airf_array)  #TODO: NEW
    z_airf_array = np.asarray(z_airf_array)  #TODO: NEW
    y_dot_z = np.sum(y_airf_array * z_airf_array, axis=1)  #TODO: NEW
    y_proj = y_airf_array - y_dot_z[:, None] * z_airf_array  #TODO: NEW
    y_proj_norm = np.linalg.norm(y_proj, axis=1)  #TODO: NEW
    safe_norm = np.where(y_proj_norm > eps, y_proj_norm, 1.0)  #TODO: NEW
    y_eff = y_proj / safe_norm[:, None]  #TODO: NEW

    x_eff = np.cross(y_eff, z_airf_array)  #TODO: NEW
    x_eff_norm = np.linalg.norm(x_eff, axis=1)  #TODO: NEW
    safe_x_norm = np.where(x_eff_norm > eps, x_eff_norm, 1.0)  #TODO: NEW
    x_eff = x_eff / safe_x_norm[:, None]  #TODO: NEW
    return x_eff, y_eff, y_proj_norm  #TODO: NEW
