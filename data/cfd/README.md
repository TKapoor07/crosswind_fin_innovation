# CFD dataset

Upload a reviewed CSV through `/api/v1/data/cfd/upload`. Required columns:

`geometry,span_m,chord_m,v_kmh,wind_ms,wind_from_deg,fin_angle_deg,fx_n,fy_n,fz_n,mx_nm,my_nm,mz_nm`

- `geometry`: A/B/C/D/E
- Wind angle uses HAWS convention: 0° headwind, 90° wind from right, 180° tailwind, 270° wind from left.
- Forces and moments are whole-vehicle increments versus the no-fin baseline, ISO vehicle axes: +x forward, +y left, +z up; moments are about the documented vehicle reference point.
- Include optional `case_group` to keep related mesh/refinement cases together in the train/test split.
- Add source/run IDs, solver, mesh, convergence and reference-point metadata in your project data dictionary. Do not discard those records in your research archive.

Do not upload the toy model's generated values and label them as CFD.
