**The airfoils in this folder are the best airfoils found during the TU Delft V3 kite project.**
- `2D_airfoil_sliced_from_CFD_CAD`: represent the airfoils sliced from the CAD model used for CFD simulations, the one reported in (Vire et al. 2020, 2022)
- `2D_airfoils_used_for_2D_CFD_generated_parametrically`: represent the airfoils used for 2D RANS CFD simulations by Kasper, the airfoils were generated parametrically using Masure's scripts, don't match the CAD exactly but are close approximations.
- `2D_polars_CFD_re5e5_converged_kasper_run_3_raw`: raw polars obtained from 2D RANS CFD simulations at Re=5e5 performed by Kasper for the airfoils in 2D_airfoils_used_for_2D_CFD_generated_parametrically, no smoothing or fitting applied.
- `2D_polars_CFD_re5e5_converged_kasper_run_3`: raw polars from `2D_polars_CFD_re5e5_converged_kasper_run_3_raw` fitted using PCHIP interpolation for better usability in simulations.
- Wind tunnel flat plate approximate from: 
https://digital.library.unt.edu/ark:/67531/metadc57443/