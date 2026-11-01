# Three.js integration notes

The bundled `heliark_haws_v2.html` now:

- creates a backend session on page load;
- sends inputs to `POST /api/v1/sessions/{session_id}/step` at approximately 10 Hz;
- uses backend state, angle, forces, power, cumulative energy and force-angle curve for the dashboard and fin animation;
- polls `/api/v1/models` so CFD/ML models become selectable after data upload/training;
- retains Three.js rendering, camera controls, airflow particles and vehicle animation in the browser.

The browser still computes a local apparent-wind vector for particle/compass animation. The displayed engineering decision and force/power/energy results come from the backend. The backend uses the exact same wind convention documented in `kinematics.py`.

For local use, serve the HTML from FastAPI at `http://127.0.0.1:8000/`. Do not open the HTML with `file://`; relative API requests require the server origin. The Three.js CDN script requires internet access unless you vendor the library locally.
