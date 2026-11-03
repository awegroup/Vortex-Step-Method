"""Interactive "fancy" 3D Plotly plot of the TU Delft V3 kite.

Renders the physical inflatable structure of the LEI kite -- the leading-edge
tube, the chordwise strut tubes and the curved single-skin canopy -- reconstructed
from a raw Surfplan export, together with the VSM aerodynamic loading drawn as
chord-normal force vectors distributed over the chord by a measured Cp(x/c).

Requirements (already handled by the data shipped in the repo):
  - a Surfplan export directory: a ``<name>.txt`` plus a ``profiles/`` directory,
  - (optional) a directory of ``cp_AOA_<deg>.dat`` chordwise Cp distributions.

SurfplanAdapter must be installed (a dependency of VSM) -- it converts the raw
export into the VSM geometry the plot needs (cached on first run).
"""

from pathlib import Path

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.plot_geometry_plotly import interactive_plot, save_high_res_render

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "TUDELFT_V3_KITE"
RESULTS_DIR = PROJECT_DIR / "results" / "TUDELFT_V3_KITE"

# Set True to also write a high-resolution PNG (results/.../plotly_render.png)
SAVE_HIGH_RES_PNG = True


def main():
    # --- inflow / operating point -------------------------------------------
    Umag = 10.0  # [m/s]
    angle_of_attack = 8.0  # [deg]
    side_slip = 0.0  # [deg]

    # --- build the aerodynamic model ----------------------------------------
    # The Breukels-regression CAD geometry needs no ML models, so it runs
    # out of the box. Any other aero_geometry_*.yaml works too.
    aero_geometry = (
        DATA_DIR / "CAD_derived_geometry" / "aero_geometry_CAD_breukels_regression.yaml"
    )
    body_aero = BodyAerodynamics.instantiate(
        n_panels=36,
        file_path=aero_geometry,
        spanwise_panel_distribution="uniform",
    )

    # --- data for the fancy plot --------------------------------------------
    # Raw Surfplan export (tubes + airfoil profiles) and the Cp database.
    surfplan_dir = DATA_DIR / "Surfplan_export"
    cp_distributions_dir = DATA_DIR / "cpx_distributions"

    # --- interactive fancy plot ---------------------------------------------
    # Passing ``surfplan_dir`` switches on the inflatable tubes + curved canopy.
    # Passing ``cp_distributions_dir`` distributes each panel's VSM force over the
    # chord using the closest-AoA measured Cp(x/c). Omit either to fall back.
    fig = interactive_plot(
        body_aero,
        vel=Umag,
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        title="TUDELFT_V3_KITE",
        surfplan_dir=surfplan_dir,
        cp_distributions_dir=cp_distributions_dir,
        is_with_canopy=True,  # curved canopy over the tubes (False -> tubes only)
        is_with_tube_rings=False,  # True -> also draw the tube construction rings
        is_show=True,  # open the interactive plot in a browser
        # is_save=True,  # uncomment to also write an interactive .html
        # save_path=str(RESULTS_DIR / "fancy_plot.html"),
    )

    # --- optional: high-resolution PNG (matches the interactive view) --------
    if SAVE_HIGH_RES_PNG:
        save_high_res_render(
            fig,
            RESULTS_DIR / "plotly_render.png",
            base_size=2000,  # square canvas -> undistorted; then cropped to the kite
            scale=4,
            show_legend=False,
            show_axes=False,
            show_annotations=False,
        )


if __name__ == "__main__":
    main()
