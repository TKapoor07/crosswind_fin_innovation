"""Central configuration.

Every number here is either (a) taken from the SEVC 2026 design report / the existing
Three.js page, or (b) an explicitly labelled PLACEHOLDER that the team must confirm.
Nothing in this file is a validated aerodynamic result.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VehicleParams:
    # --- from the design report (Transmission / Motor sections) -------------------
    mass_kg: float = 330.0            # fully laden car
    crr: float = 0.025                # coefficient of rolling resistance
    cd: float = 0.18                  # body drag coefficient (report value; yaw dependence NOT modelled)
    frontal_area_m2: float = 1.4
    wheel_radius_m: float = 0.2334    # 120/70-12 tyre
    g: float = 9.81
    track_m: float = 0.98
    wheelbase_m: float = 1.753
    cg_height_m: float = 0.35
    battery_nominal_wh: float = 5040.0  # 72 V x 70 Ah
    # report's own range calculation uses 0.82 electrical-to-mechanical efficiency at 31 km/h
    drive_efficiency: float = 0.82
    # --- same constant the Three.js page uses ---------------------------------------
    air_density: float = 1.225
    # --- PLACEHOLDERS (taken from the Three.js scene, NOT from CAD) ----------------
    fin_x_m: float = -1.45            # fin pivot, vehicle frame, +x forward
    fin_base_height_m: float = 0.508  # fin root height above ground
    cg_x_m: float = -0.05             # body-centre in the scene; real CG x is unknown


@dataclass(frozen=True)
class EnergyAssumptions:
    """Things the browser toy model ignores. Defaults reproduce the page exactly (all zero)."""
    actuator_power_w: float = 0.0     # average electrical power drawn by the fin actuator while deployed
    fin_extra_mass_kg: float = 0.0    # fin + hinge + actuator + wiring, carried whether deployed or not
    solar_power_w: float = 0.0        # constant solar input for SOC bookkeeping in race sims


@dataclass(frozen=True)
class ControllerConfig:
    # candidate fin angle grid (the page searches -30..+30 in 1 degree steps)
    angle_min_deg: float = -30.0
    angle_max_deg: float = 30.0
    angle_step_deg: float = 1.0
    # validated wind envelope (apparent wind speed, m/s)
    envelope_min_va: float = 4.0
    envelope_max_va: float = 32.0
    # decision thresholds
    enter_fx_n: float = 5.0           # deploy only if the best feasible force is at least this
    exit_fx_n: float = 2.5            # corrected mode: dead-band, stay deployed until force falls below this
    dwell_s: float = 1.5              # minimum time between state changes (non-safety transitions only)
    angle_improve_n: float = 4.0      # re-aim only if it gains at least this much force
    side_hold_threshold_ms: float = 0.05
    # uncertainty handling (only matters for models that report sigma)
    z_sigma: float = 1.64             # one-sided ~95 %
    # optional hard limits on moments (None = not enforced)
    max_yaw_moment_nm: float | None = None
    max_roll_moment_nm: float | None = None
    max_lift_n: float | None = None
    # True reproduces the page's controller bit-for-bit (including its safety gaps)
    legacy: bool = False
    max_step_dt_s: float = 0.5        # server-side clamp on the client-supplied dt


@dataclass(frozen=True)
class ValidationPolicy:
    """Gate for calling a learned model 'validated'. These thresholds are PLACEHOLDERS: the team
    must set them from its own accuracy requirements before relying on the flag."""
    allowed_sources: tuple[str, ...] = ("cfd", "windtunnel")
    min_cases: int = 100
    max_rmse_fx_n: float = 3.0
    max_rmse_fy_n: float = 10.0
    coverage_range: tuple[float, float] = (0.85, 1.0)   # fraction of CV errors inside +-1.96 sigma


@dataclass(frozen=True)
class Settings:
    data_dir: str = field(default_factory=lambda: os.environ.get("HELIARK_DATA_DIR", "./data"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("HELIARK_API_KEY") or None)
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in os.environ.get("HELIARK_CORS_ORIGINS", "*").split(",") if o.strip()))
    max_sessions: int = field(default_factory=lambda: int(os.environ.get("HELIARK_MAX_SESSIONS", "256")))
    session_ttl_s: float = field(default_factory=lambda: float(os.environ.get("HELIARK_SESSION_TTL_S", "900")))
    max_sim_steps: int = field(default_factory=lambda: int(os.environ.get("HELIARK_MAX_SIM_STEPS", "400000")))


VEHICLE = VehicleParams()
CONTROLLER = ControllerConfig()
POLICY = ValidationPolicy()
