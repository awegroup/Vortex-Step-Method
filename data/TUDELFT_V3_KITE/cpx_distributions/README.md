# TU Delft V3 kite — 2D Cp(x/c) distributions

Chordwise pressure-coefficient (`cp_AOA_*.dat`) and skin-friction (`cf_AOA_*.dat`)
distributions for the **TU Delft V3 LEI kite midspan airfoil section** (rigid 2D),
from RANS CFD (Thijs/Kasper).

File format: whitespace columns `# x y Cp` (or `... Cf`), i.e. column 0 = x/c,
column 1 = y/c, column 2 = Cp (or Cf).

## Contents

| Angle of attack | Cp file | Cf file |
|---|---|---|
| 2° | `cp_AOA_2.dat` | `cf_AOA_2.dat` |
| 6° | `cp_AOA_6.dat` | `cf_AOA_6.dat` |
| 8° | `cp_AOA_8.dat` | `cf_AOA_8.dat` |

These are the only angles for which the genuine V3 midspan section Cp distribution
is available locally.

## Provenance

Copied from `WES_aero_sim_for_kite_design/data/`:
- AoA 2° and 8°: `cp_cf_kasper/`
- AoA 6°: `THIJS_CFD_alpha_6_V3_turbulent/postProcessing/surfaces/624/`

The V3 section is identified by its airfoil geometry embedded in the files:
~13.4% thickness, max camber y ≈ 0.0953 (chord normalised to 1).

## Not included

`Pointwise-Openfoam-toolchain/.../cp_AOA_10.dat` was **excluded**: it is a generic
demo LEI profile from the Pointwise→OpenFOAM toolchain (~12.4% thickness,
`tube_size=0.08, c_x=0.15, c_y=0.08`), **not** the V3 section.
