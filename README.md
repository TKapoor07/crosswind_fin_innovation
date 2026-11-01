# HELIARK Adaptive Wind Strategy (HAWS) — Backend v1

A Python/FastAPI research simulation backend for the existing Three.js visualization. The uploaded `session.py`, `controller.py`, `energy.py`, `geometry.py`, `kinematics.py`, `config.py`, `base.py`, and `toy.py` were retained and organized into a runnable package. Missing package structure, API, data ingestion, ML workflow, race simulator, tests and deployment files were added.

> **Engineering status:** Concept-stage research software. The only included aerodynamic model is a low-order isolated-fin toy approximation. No HELIARK-specific CFD or wind-tunnel data was supplied. The API must not be connected to a physical actuator or used as a safety-critical controller.

## Run locally (Windows PowerShell)

```powershell
cd HAWS_Backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` for the current visualization and `http://127.0.0.1:8000/docs` for interactive API docs. Keep the terminal running while using the frontend.

## API

- `GET /api/v1/health` — service/model status
- `GET /api/v1/models` — registered model provenance
- `POST /api/v1/evaluate` — one independent evaluation
- `POST /api/v1/sessions` — create stateful controller session
- `POST /api/v1/sessions/{id}/step` — advance controller and energy ledger
- `POST /api/v1/sessions/{id}/reset` — reset session/ledger
- `DELETE /api/v1/sessions/{id}` — delete session
- `WS /ws/sessions/{id}` — streaming session interface
- `POST /api/v1/data/cfd/upload` — load a CFD CSV
- `POST /api/v1/ml/train` — train ExtraTrees surrogate from loaded CFD data
- `POST /api/v1/race/simulate` — compare race energy strategies

Example step payload:

```json
{
  "v_kmh": 60,
  "wind_ms": 8,
  "wind_from_deg": 70,
  "geometry": "B",
  "mode": "auto",
  "manual_deg": 15,
  "side_force_limit_n": 200,
  "model": "toy_v2",
  "dt": 0.1
}
```

Use `/api/v1/sessions` first for a stateful simulation; use its returned session ID for each step. The stateless `/evaluate` endpoint intentionally does not preserve hysteresis or cumulative energy.

## CFD data and ML

1. Run no-fin and fin CFD cases under identical geometry, wind, speed and solver settings.
2. Store *whole-vehicle* force and moment increments against the no-fin baseline, not fin-only isolated force, per the R&D proposal. The moment reference point must be documented.
3. Upload the reviewed CSV. See `data/cfd/README.md` for schema and axis conventions.
4. Train via `POST /api/v1/ml/train`.
5. Compare held-out R²/RMSE/MAE and residuals. Include a `case_group` column to avoid splitting related mesh variants across train/test.
6. The trained model predicts all three force and three moment increments and is domain-gated. It remains `validated=false`. The training endpoint does not certify CFD quality or physical validity.

Recommended ML targets after enough CFD runs: ΔFx, ΔFy, ΔFz and yaw/pitch/roll moment increments. A multi-output ExtraTrees baseline is included because it handles nonlinear tabular data and offers a tree-ensemble spread estimate. Compare it against nearest-neighbour interpolation, Random Forest, Gradient Boosting and CatBoost/LightGBM if available. Use grouped holdout by geometry/configuration and a final unseen CFD validation matrix; do not use random row splitting where near-duplicate CFD runs leak across folds.

## Safety and engineering caveats

- Corrected controller mode is the default (`legacy=false`). Hard safety gates can retract immediately; dwell time is only for non-safety transitions.
- Manual mode bypasses controller safety checks and is for visualization/testing only.
- The existing controller's yaw/roll moment estimates use scene placeholders for fin x and CG x. Confirm CAD-derived CG, fin centre of pressure, and reference axes before trusting moments.
- The toy model is unvalidated. Its envelope is not a certified safe operating envelope.
- Current whole-body force interaction, fin structural loads, aeroelasticity, steering response, tyre loads, actuator failure dynamics and competition-rule compliance are not established.
- Energy outputs are first-order tractive-energy estimates. Added mass and actuator power are configurable, but body-drag changes and real solar/SOC/drive-cycle dynamics require validated inputs.
- Do not connect to a real actuator. Use a mechanically safe stowed position, independent hardware interlocks, watchdog, redundant sensing and formal safety review for any future physical prototype.

## Tests

```powershell
pip install pytest httpx
pytest -q
```

## Deploy

For local team use, run behind a trusted network. Before public deployment: configure exact CORS origins, authentication/API key middleware, HTTPS, rate limits, persistent data storage, structured logs and session cleanup. Do not expose unrestricted training/upload endpoints publicly.
