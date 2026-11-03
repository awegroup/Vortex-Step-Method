"""Chordwise pressure-coefficient (Cp) database for the fancy Plotly plot.

A directory of ``cp_AOA_<deg>.dat`` files (whitespace columns ``x/c, y/c, Cp``
around the 2D airfoil section) is treated as a small database keyed by angle of
attack. Given an operating angle of attack, the closest available distribution is
selected and its **upper (suction) surface** ``|Cp|(x/c)`` is used to scale the
magnitude of the chordwise force vectors on the canopy.

Per the modelling assumption in use, the single mid-span section distribution is
applied to every spanwise station (all airfoils share this geometry).
"""

import re
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np


def load_upper_surface_cp(dat_file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the upper (suction) surface ``Cp(x/c)`` from a Cp ``.dat`` loop.

    The file lists ``x/c, y/c, Cp`` around the closed airfoil section. The upper
    surface is the leading-edge-to-trailing-edge branch with the more negative
    (suction) mean Cp; it is returned sorted by ``x/c``.

    Args:
        dat_file_path (Path): Path to a ``cp_AOA_*.dat`` file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``x/c`` (ascending) and ``Cp`` arrays.
    """
    data = np.loadtxt(dat_file_path)
    x = data[:, 0]
    cp = data[:, 2]
    n = len(x)
    i_le = int(np.argmin(x))
    i_te = int(np.argmax(x))

    def branch(a: int, b: int) -> np.ndarray:
        if a <= b:
            return np.arange(a, b + 1)
        return np.concatenate([np.arange(a, n), np.arange(0, b + 1)])

    branch_a = branch(i_le, i_te)
    branch_b = branch(i_te, i_le)
    upper = branch_a if cp[branch_a].mean() < cp[branch_b].mean() else branch_b

    x_upper = x[upper]
    cp_upper = cp[upper]
    order = np.argsort(x_upper)
    return x_upper[order], cp_upper[order]


def load_cp_database(cp_dir: Path) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Load all ``cp_AOA_<deg>.dat`` files, keyed by angle of attack in degrees.

    Args:
        cp_dir (Path): Directory of ``cp_AOA_*.dat`` files.

    Returns:
        Dict[float, Tuple[np.ndarray, np.ndarray]]: Mapping ``aoa_deg -> (x/c, Cp)``
            for the upper surface.
    """
    database = {}
    for path in sorted(Path(cp_dir).glob("cp_AOA_*.dat")):
        match = re.search(r"AOA_(-?\d+(?:\.\d+)?)", path.stem)
        if not match:
            continue
        database[float(match.group(1))] = load_upper_surface_cp(path)
    if not database:
        raise FileNotFoundError(f"No 'cp_AOA_*.dat' files found in {cp_dir}.")
    return database


def build_cp_magnitude_fn(
    cp_dir: Path, angle_of_attack: float
) -> Tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Return a ``|Cp|(x/c)`` interpolator for the closest available AoA.

    Args:
        cp_dir (Path): Directory of ``cp_AOA_*.dat`` files.
        angle_of_attack (float): Operating angle of attack in degrees.

    Returns:
        Tuple[Callable, float]: A function mapping ``x/c`` (array) to ``|Cp|`` and
            the matched database angle of attack.
    """
    database = load_cp_database(cp_dir)
    available = np.array(sorted(database))
    matched_aoa = float(available[np.argmin(np.abs(available - angle_of_attack))])
    x_upper, cp_upper = database[matched_aoa]

    def magnitude(x_over_c: np.ndarray) -> np.ndarray:
        return np.abs(np.interp(x_over_c, x_upper, cp_upper))

    return magnitude, matched_aoa
