from pathlib import Path
from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.Solver import Solver
from VSM.plotting import (
    process_panel_coefficients,
)
from VSM.interactive import interactive_plot

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent


## Processing panel coefficients
file_path = (
    Path(PROJECT_DIR) / "data" / "TUDELFT_V3_LEI_KITE" / "geometry_corrected.csv"
)
n_panels = 35
spanwise_panel_distribution = "unchanged"
wing_instance = Wing(n_panels, spanwise_panel_distribution)
body_aero = BodyAerodynamics.from_file(
    wing_instance, file_path, is_with_corrected_polar=False
)
process_panel_coefficients(
    body_aero,
    PROJECT_DIR,
    n_panels,
    polar_folder_path=Path(
        PROJECT_DIR, "examples", "TUDELFT_V3_LEI_KITE", "polar_engineering"
    ),
)
