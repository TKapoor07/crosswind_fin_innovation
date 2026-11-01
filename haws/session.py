"""Stateful session: owns controller state, simulated time, side-hold and the energy ledger.

``Session.step`` is the single entry point used by the REST step endpoint, the WebSocket, the
stateless evaluator and the race simulator, so all of them run the *same* code path.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, replace

import numpy as np

from .aero.base import AeroModel
from .config import CONTROLLER, VEHICLE, ControllerConfig, EnergyAssumptions, VehicleParams
from .controller import Controller, Decision
from .energy import EnergyLedger, power_no_fin, power_with_fin
from .geometry import CANDIDATES, resolve
from .kinematics import Flow, apparent_wind, apply_gust


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, AeroModel] = {}
        self._lock = threading.RLock()

    def register(self, model: AeroModel) -> None:
        with self._lock:
            self._models[model.name] = model

    def unregister(self, name: str) -> None:
        with self._lock:
            self._models.pop(name, None)

    def get(self, name: str) -> AeroModel:
        with self._lock:
            try:
                return self._models[name]
            except KeyError:
                raise KeyError(f"unknown aero model {name!r}; available: {sorted(self._models)}") from None

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._models)

    def info(self) -> list[dict]:
        with self._lock:
            return [m.info() for m in self._models.values()]


@dataclass
class StepInput:
    v_kmh: float = 60.0
    wind_ms: float = 8.0
    wind_from_deg: float = 70.0
    gust: bool = False
    geometry: str = "B"
    span_m: float | None = None
    chord_m: float | None = None
    mode: str = "auto"                 # auto | ret | man
    manual_deg: float = 15.0
    side_force_limit_n: float = 200.0
    fault: bool = False
    model: str = "toy_v2"
    legacy: bool | None = None         # None = keep the session's controller mode
    dt: float = 0.0


def _num(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _arr(a: np.ndarray, nd: int = 3) -> list[float]:
    return np.round(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), nd).tolist()


@dataclass
class StepResult:
    t: float
    dt: float
    inp: StepInput
    flow: Flow
    dec: Decision
    p0_w: float
    p1_w: float
    pfix_w: float
    ledger: EnergyLedger
    eta: float
    model_info: dict
    legacy: bool
    geom_span: float
    geom_chord: float

    def to_dict(self) -> dict:
        d, f, fl = self.dec, self.dec.f, self.flow
        fy_right = fl.side * f.fy
        g = d.grid
        finite = g.finite() and np.isfinite(d.fixed_f.fx)
        warnings = list(d.warnings)
        if not finite:
            warnings.append("non_finite_model_output")
        if not self.model_info.get("validated", False):
            warnings.append(f"model_unvalidated:{self.model_info.get('provenance')}")
        if self.inp.mode == "man":
            warnings.append("manual_mode_bypasses_safety_checks")
        batt_a, batt_f = self.ledger.battery_equivalent(self.eta)
        return {
            "t": round(self.t, 4), "dt": round(self.dt, 4),
            "state": d.name, "reason": d.reason, "code": d.code, "held": d.held,
            "controller": "legacy" if self.legacy else "corrected",
            "model": self.model_info,
            "fin": {"deployed": d.on, "angle_deg": d.delta, "physical_angle_deg": fl.side * d.delta,
                    "aoa_deg": _num(f.aoa) if d.on else 0.0, "geometry": self.inp.geometry,
                    "span_m": self.geom_span, "chord_m": self.geom_chord},
            "flow": {"va_ms": fl.va, "apparent_angle_deg": fl.gam_deg, "side": fl.side,
                     "side_label": "right" if fl.side > 0 else "left", "sx": fl.sx, "sz": fl.sz,
                     "q_pa": fl.q, "wind_ms": fl.wind_ms, "wind_from_deg": fl.wind_from_deg},
            "forces": {"fx_n": _num(f.fx), "fy_n": _num(f.fy), "fy_abs_n": abs(_num(f.fy)),
                       "fy_right_n": _num(fy_right), "lateral_dir": "left" if fy_right < 0 else "right",
                       "fz_n": _num(f.fz), "sigma_fx_n": _num(f.sigma_fx), "sigma_fy_n": _num(f.sigma_fy),
                       "yaw_moment_nm": _num(d.yaw_moment_nm), "pitch_moment_nm": _num(d.pitch_moment_nm), "roll_moment_nm": _num(d.roll_moment_nm)},
            "power": {"no_fin_w": self.p0_w, "with_fin_w": self.p1_w,
                      "delta_pct": ((self.p1_w - self.p0_w) / self.p0_w * 100.0) if self.p0_w > 50 else None},
            "energy": {"adaptive_wh": self.ledger.adaptive_wh, "fixed_wh": self.ledger.fixed_wh,
                       "adaptive_battery_wh": batt_a, "fixed_battery_wh": batt_f,
                       "seconds": self.ledger.seconds, "basis": "mechanical tractive energy (Wh); battery_* divides by drive efficiency"},
            "fixed_fin": {"angle_deg": float(self.inp.manual_deg), "fx_n": _num(d.fixed_f.fx),
                          "fy_abs_n": abs(_num(d.fixed_f.fy)),
                          "over_limit": bool(abs(d.fixed_f.fy) > self.inp.side_force_limit_n)},
            "best": {"angle_deg": d.best_d, "fx_n": d.best_fx},
            "curve": {"d": _arr(g.d, 2), "fx": _arr(g.fx), "fy_abs": _arr(np.abs(g.fy)),
                      "over_limit": (np.abs(g.fy) > self.inp.side_force_limit_n).astype(int).tolist(),
                      "allowed": d.allowed.astype(int).tolist(),
                      "sigma_fx": _arr(g.sigma_fx)},
            "violations": d.violations, "warnings": warnings,
        }


class Session:
    def __init__(self, registry: ModelRegistry, cfg: ControllerConfig = CONTROLLER,
                 vehicle: VehicleParams = VEHICLE, assume: EnergyAssumptions | None = None,
                 session_id: str | None = None):
        self.id = session_id or uuid.uuid4().hex
        self.registry, self.vehicle = registry, vehicle
        self.assume = assume or EnergyAssumptions()
        self.ctrl = Controller(cfg)
        self.ledger = EnergyLedger()
        self.t = 0.0
        self.side_hold = 1
        self.last_used = time.monotonic()
        self.lock = threading.Lock()

    def reset(self, *, energy_only: bool = False) -> None:
        self.ledger.reset()
        if not energy_only:
            self.ctrl.reset()
            self.t = 0.0
            self.side_hold = 1

    def step(self, inp: StepInput) -> StepResult:
        self.last_used = time.monotonic()
        if inp.legacy is not None and inp.legacy != self.ctrl.cfg.legacy:
            self.ctrl.cfg = replace(self.ctrl.cfg, legacy=bool(inp.legacy))
        cfg, veh = self.ctrl.cfg, self.vehicle

        dt = min(max(float(inp.dt), 0.0), cfg.max_step_dt_s)
        self.t += dt                                           # page: tt += dt happens first
        w, b = inp.wind_ms, inp.wind_from_deg
        if inp.gust and w > 0:
            w, b = apply_gust(w, b, self.t)
        v = inp.v_kmh / 3.6
        flow = apparent_wind(v, w, b, side_hold=self.side_hold, rho=veh.air_density,
                             side_threshold=cfg.side_hold_threshold_ms)
        self.side_hold = flow.side

        geom = resolve(inp.geometry, inp.span_m, inp.chord_m)
        model = self.registry.get(inp.model)
        res = self.ctrl.evaluate(model, geom, flow, inp.manual_deg)
        dec = self.ctrl.decide(flow=flow, res=res, mode=inp.mode, manual_deg=inp.manual_deg,
                               lim=inp.side_force_limit_n, fault=inp.fault, t=self.t, vehicle=veh,
                               fin_x_m=veh.fin_x_m, fin_height_m=veh.fin_base_height_m, span_m=geom.span_m)

        p0 = power_no_fin(flow, veh)
        p1 = power_with_fin(flow, veh, dec.f.fx, dec.on, self.assume)
        pf = power_with_fin(flow, veh, dec.fixed_f.fx, True, self.assume)
        self.ledger.add(p0, p1, pf, dt)
        return StepResult(self.t, dt, inp, flow, dec, p0, p1, pf, self.ledger, veh.drive_efficiency,
                          model.info(), cfg.legacy, geom.span_m, geom.chord_m)


class SessionManager:
    """Bounded, TTL-evicting store so a misbehaving client cannot exhaust memory."""

    def __init__(self, registry: ModelRegistry, max_sessions: int = 256, ttl_s: float = 900.0):
        self.registry, self.max_sessions, self.ttl_s = registry, max_sessions, ttl_s
        self._s: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _purge_locked(self) -> None:
        now = time.monotonic()
        for k in [k for k, s in self._s.items() if now - s.last_used > self.ttl_s]:
            del self._s[k]

    def create(self, **kw) -> Session:
        with self._lock:
            self._purge_locked()
            if len(self._s) >= self.max_sessions:
                raise RuntimeError("too many active sessions")
            s = Session(self.registry, **kw)
            self._s[s.id] = s
            return s

    def get(self, sid: str) -> Session:
        with self._lock:
            self._purge_locked()
            try:
                s = self._s[sid]
            except KeyError:
                raise KeyError(f"unknown session {sid!r} (expired or never created)") from None
            s.last_used = time.monotonic()
            return s

    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._s.pop(sid, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._s)
