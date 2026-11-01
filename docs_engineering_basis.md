# Engineering basis and source traceability

## Values carried into `haws/config.py`

| Parameter | Backend value | Source / qualification |
|---|---:|---|
| Fully laden vehicle mass | 330 kg | SEVC 2026 Design Report, drivetrain calculation; also repeated in HAWS proposal p.2 |
| Rolling resistance coefficient | 0.025 | SEVC 2026 Design Report, drivetrain calculation |
| Body drag coefficient | 0.18 | SEVC 2026 Design Report / HAWS proposal p.2; yaw dependence is not modelled |
| Frontal area | 1.4 m² | HAWS proposal p.2; initial estimate only |
| Wheel radius | 0.2334 m | SEVC 2026 Design Report, wheel/tractive torque calculation |
| Wheelbase | 1.753 m | HAWS proposal p.2 |
| Track width | 0.980 m | HAWS proposal p.2 |
| CG height | 0.350 m | HAWS proposal p.2 initial value; confirm loaded condition (the design report also gives 0.364 m in its brake calculation) |
| Battery nominal energy | 5,040 Wh | SEVC 2026 Design Report: 72 V × 70 Ah; report also specifies 80% DOD and 95% SOH |
| Target / maximum speed | 60 / 85 km/h | HAWS proposal p.2 |
| Air density | 1.225 kg/m³ | Retained from existing Three.js/Python simulation assumption; not a HELIARK-specific measurement |

## Efficiency discrepancy to resolve

The supplied `config.py` used 0.82 drivetrain efficiency and described it as the report's range-calculation assumption. The Design Report's drivetrain section also states 0.85 in its electrical-power calculation and separately describes 92% motor efficiency. These may refer to different system boundaries/operating points. The backend retains 0.82 to preserve the supplied implementation, but the team must resolve the intended end-to-end efficiency before race-energy comparisons are treated as authoritative.

## Explicit placeholders retained from the visualization

`fin_x_m=-1.45 m`, `fin_base_height_m=0.508 m`, and `cg_x_m=-0.05 m` are scene-derived placeholders, not confirmed CAD measurements. Fin geometry A–D and default span/chord are candidate visualization parameters, not final engineering dimensions. Confirm the current vehicle CAD, mounting envelope, fin centre of pressure, actual loaded CG, and rules before using moment or packaging outputs.

## Proposal constraints and missing evidence

The HAWS R&D proposal (25 October 2026) marks the work as concept/feasibility stage and says no CFD results, fin dimensions, or performance gains have been validated. It calls for whole-vehicle baseline-versus-fin force deltas (ΔFx, ΔFy, ΔFz) and yaw/pitch/roll moments about a documented reference point. The CFD schema in this backend therefore requires all six force/moment outputs. It also requires convergence and provenance review; loading a CSV does not make it validated.

The proposal also notes that the searchable design report does not provide a complete tabulated external body length × width × height or a ready-to-run watertight outer-body CAD surface. A fin/body interaction model, structural loads, vehicle dynamics, sensor uncertainty, and competition-rule approval remain future validation work.
