import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
from VSM.WingGeometry import Wing
from VSM.WingAerodynamics import WingAerodynamics
from VSM.Solver import Solver
from VSM.plotting import plot_polars, plot_distribution, plot_geometry
from VSM.interactive import interactive_plot

# Find the root directory of the repository
root_dir = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
    root_dir = os.path.abspath(os.path.join(root_dir, ".."))
    if root_dir == "/":
        raise FileNotFoundError("Could not find the root directory of the repository.")
save_folder = Path(root_dir) / "results" / "TUDELFT_V3_LEI_KITE"

default_kite = rows = [
    [
        np.array([-0.207302, 1.470985, 0.218733]),
        np.array([0.420886, 1.438128, 0.216666]),
        ["lei_airfoil_breukels", [np.float64(0.09000078168237204), 0.09]],
    ],
    [
        np.array([-0.111246, 1.46245, 0.038158]),
        np.array([0.353482, 1.441267, 0.036624]),
        ["lei_airfoil_breukels", [np.float64(0.10868124653310406), 0.0449861]],
    ],
    [
        np.array([0.055167, 1.445922, -0.081867]),
        np.array([0.308862, 1.443344, -0.082558]),
        ["lei_airfoil_breukels", [np.float64(0.09496237630319553), 0.0449861]],
    ],
    [
        np.array([-0.261826, 1.429789, 0.561858]),
        np.array([0.531586, 1.389118, 0.553203]),
        ["lei_airfoil_breukels", [np.float64(0.08999990083595241), 0.09]],
    ],
    [
        np.array([-0.307829, 1.332461, 0.894339]),
        np.array([0.624986, 1.287208, 0.875843]),
        ["lei_airfoil_breukels", [np.float64(0.09000046673001391), 0.09]],
    ],
    [
        np.array([-0.345378, 1.172471, 1.202549]),
        np.array([0.701222, 1.126971, 1.171919]),
        ["lei_airfoil_breukels", [np.float64(0.08999974848332609), 0.09]],
    ],
    [
        np.array([-0.374528, 0.949267, 1.469545]),
        np.array([0.760405, 0.9086, 1.42614]),
        ["lei_airfoil_breukels", [np.float64(0.08999987043487782), 0.09]],
    ],
    [
        np.array([-0.395319, 0.669341, 1.677332]),
        np.array([0.802618, 0.638624, 1.622579]),
        ["lei_airfoil_breukels", [np.float64(0.08999976120391483), 0.09]],
    ],
    [
        np.array([-0.407782, 0.346336, 1.809671]),
        np.array([0.827922, 0.329802, 1.747056]),
        ["lei_airfoil_breukels", [np.float64(0.09000001096237602), 0.09]],
    ],
    [
        np.array([-0.411935, 0.0, 1.854922]),
        np.array([0.836352, 0.0, 1.789502]),
        ["lei_airfoil_breukels", [np.float64(0.08999999392985342), 0.09]],
    ],
    [
        np.array([-0.407782, -0.346336, 1.809671]),
        np.array([0.827922, -0.329802, 1.747056]),
        ["lei_airfoil_breukels", [np.float64(0.09000001096237602), 0.09]],
    ],
    [
        np.array([-0.395319, -0.669341, 1.677332]),
        np.array([0.802618, -0.638624, 1.622579]),
        ["lei_airfoil_breukels", [np.float64(0.08999976120391483), 0.09]],
    ],
    [
        np.array([-0.374528, -0.949267, 1.469545]),
        np.array([0.760405, -0.9086, 1.42614]),
        ["lei_airfoil_breukels", [np.float64(0.08999987043487782), 0.09]],
    ],
    [
        np.array([-0.345378, -1.172471, 1.202549]),
        np.array([0.701222, -1.126971, 1.171919]),
        ["lei_airfoil_breukels", [np.float64(0.08999974848332609), 0.09]],
    ],
    [
        np.array([-0.307829, -1.332461, 0.894339]),
        np.array([0.624986, -1.287208, 0.875843]),
        ["lei_airfoil_breukels", [np.float64(0.09000046673001391), 0.09]],
    ],
    [
        np.array([-0.261826, -1.429789, 0.561858]),
        np.array([0.531586, -1.389118, 0.553203]),
        ["lei_airfoil_breukels", [np.float64(0.08999990083595241), 0.09]],
    ],
    [
        np.array([0.055167, -1.445922, -0.081867]),
        np.array([0.308862, -1.443344, -0.082558]),
        ["lei_airfoil_breukels", [np.float64(0.09496237630319553), 0.0449861]],
    ],
    [
        np.array([-0.111246, -1.46245, 0.038158]),
        np.array([0.353482, -1.441267, 0.036624]),
        ["lei_airfoil_breukels", [np.float64(0.10868124653310406), 0.0449861]],
    ],
    [
        np.array([-0.207302, -1.470985, 0.218733]),
        np.array([0.420886, -1.438128, 0.216666]),
        ["lei_airfoil_breukels", [np.float64(0.09000078168237204), 0.09]],
    ],
]
"""
"polar_data",[alpha,CL,CD,CM]]: Polar data aerodynamics
                    Where alpha, CL, CD, and CM are arrays of the same length
                        - alpha: Angle of attack in radians
                        - CL: Lift coefficient
                        - CD: Drag coefficient
                        - CM: Moment coefficient
"""

n_panels = len(default_kite)
wing = Wing(n_panels + 3, "linear")

alpha = np.deg2rad(np.linspace(-5, 20, 26))
cl = 2 * np.pi * (alpha)
cd = 0.01 + 0.1 * np.abs(alpha)
cm = 0.03 * np.ones_like(alpha)

polar_data = ["polar_data", np.array([alpha, cl, cd, cm])]

print(f"polar_data: {polar_data[1].shape} {polar_data[1]}")

for rib in default_kite:
    wing.add_section(rib[0], rib[1], polar_data)
wing_aero = WingAerodynamics([wing])


# interactive plot
interactive_plot(
    wing_aero,
    vel=3.15,
    angle_of_attack=6.75,
    side_slip=0,
    yaw_rate=0,
    is_with_aerodynamic_details=True,
)
