"""Fin controller.

Two modes selected by ``ControllerConfig.legacy``:

legacy=True   Bit-for-bit port of the page's decision logic (used by the parity tests against the
              page's own JavaScript). It has three safety gaps that the harness demonstrated:
                1. the minimum-dwell timer also delays RETRACT when the wind leaves the validated
                   envelope or every angle breaks the side-force limit;
                2. a held angle is never re-checked against the side-force limit;
                3. a single 5 N threshold (no dead-band) is used to both enter and leave PROPULSION.
legacy=False  Same decision rules, but hard-constraint violations bypass dwell/hysteresis, a held
              angle is re-validated every step, there is an enter/exit dead-band, a post-fault /
              post-override re-deploy hold, uncertainty-aware feasibility (mean +- z*sigma) and an
              out-of-domain gate for data-driven models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aero.base import AeroModel, AeroResult
from .config import ControllerConfig, VehicleParams
from .geometry import FinGeometry
from .kinematics import Flow

RETRACT, PROPULSION, MANUAL, SAFE = "RETRACT", "PROPULSION", "MANUAL", "SAFE / FAULT"

MSG_ENVELOPE = "Outside the model\u2019s validated wind envelope."
MSG_SIDE = "Every fin angle exceeds the side-force limit."
MSG_NOBEN = "No feasible angle gives a useful forward force."
MSG_OK = "Best feasible angle within the side-force limit."
MSG_HOLD = "Holding state for the minimum dwell time (no rapid switching)."
MSG_FAULT = "Sensor or actuator data invalid. Fin driven to its stowed safe position."
MSG_RET = "Manual override: fin held stowed."
MSG_MAN = "Angle set by hand. Side-force and envelope checks are bypassed."
MSG_DOMAIN = "Operating point is outside the aerodynamic model\u2019s data range."
MSG_UNCERT = "Predicted benefit is within the model\u2019s uncertainty."
MSG_BADNUM = "Aerodynamic model returned invalid numbers. Fin driven to its stowed safe position."
MSG_ANGLE = "Held angle no longer satisfies the constraints; re-aimed immediately."


@dataclass
class CtrlState:
    state: str = RETRACT
    delta: float = 0.0
    t: float = -9.0          # time of the last state change (same initial value as the page)


@dataclass
class Forces:
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    aoa: float = 0.0
    sigma_fx: float = 0.0
    sigma_fy: float = 0.0
    in_domain: bool = True
    post_stall: bool = False
    mx_nm: float | None = None
    my_nm: float | None = None
    mz_nm: float | None = None


@dataclass
class Decision:
    name: str
    reason: str
    code: str
    on: bool
    delta: float
    held: bool
    f: Forces
    fixed_f: Forces                  # fin at the manual angle, always deployed (page's comparator)
    grid: AeroResult                 # full angle sweep for the curve plot
    allowed: np.ndarray              # bool mask: angles the controller may choose
    best_d: float | None
    best_fx: float | None
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    yaw_moment_nm: float = 0.0       # signed, ISO 8855 (positive = nose left); needs real CG x from CAD
    roll_moment_nm: float = 0.0      # ISO +x roll moment
    pitch_moment_nm: float = 0.0     # ISO +y pitch moment


def angle_grid(cfg: ControllerConfig) -> np.ndarray:
    n = int(round((cfg.angle_max_deg - cfg.angle_min_deg) / cfg.angle_step_deg)) + 1
    return cfg.angle_min_deg + cfg.angle_step_deg * np.arange(n, dtype=float)


def _forces_at(res: AeroResult, i: int) -> Forces:
    return Forces(float(res.fx[i]), float(res.fy[i]), float(res.fz[i]), float(res.aoa[i]),
                  float(res.sigma_fx[i]), float(res.sigma_fy[i]), bool(res.in_domain[i]),
                  bool(res.post_stall[i]),
                  None if res.mx_nm is None else float(res.mx_nm[i]),
                  None if res.my_nm is None else float(res.my_nm[i]),
                  None if res.mz_nm is None else float(res.mz_nm[i]))


class Controller:
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.state = CtrlState()
        self.grid_d = angle_grid(cfg)

    def reset(self) -> None:
        self.state = CtrlState()

    # ------------------------------------------------------------------ helpers
    def _grid_index(self, delta: float) -> int:
        i = int(round((delta - self.cfg.angle_min_deg) / self.cfg.angle_step_deg))
        return min(max(i, 0), len(self.grid_d) - 1)

    def evaluate(self, model: AeroModel, geom: FinGeometry, flow: Flow, manual_deg: float) -> AeroResult:
        """One model call for the whole grid plus the manual angle (last element)."""
        manual = float(np.clip(manual_deg, self.cfg.angle_min_deg, self.cfg.angle_max_deg))
        return model.predict(geom, flow, np.concatenate([self.grid_d, [manual]]))

    # ------------------------------------------------------------------ main step
    def decide(self, *, flow: Flow, res: AeroResult, mode: str, manual_deg: float, lim: float,
               fault: bool, t: float, vehicle: VehicleParams, fin_x_m: float, fin_height_m: float,
               span_m: float) -> Decision:
        cfg, st = self.cfg, self.state
        legacy = cfg.legacy
        n = len(self.grid_d)
        z = 0.0 if legacy else cfg.z_sigma
        g = AeroResult(res.d[:n], res.aoa[:n], res.fx[:n], res.fy[:n], res.fz[:n], res.sigma_fx[:n],
                       res.sigma_fy[:n], res.in_domain[:n], res.post_stall[:n],
                       None if res.mx_nm is None else res.mx_nm[:n],
                       None if res.my_nm is None else res.my_nm[:n],
                       None if res.mz_nm is None else res.mz_nm[:n])
        zero = Forces()
        fixed_f = _forces_at(res, n)

        over_limit = np.abs(g.fy) > lim                                # what the page paints red
        if legacy:
            allowed = ~over_limit
        else:
            allowed = (np.abs(g.fy) + z * g.sigma_fy <= lim) & g.in_domain
            # Enforce configured moment/lift limits during candidate selection, not only in diagnostics.
            f_right_grid = flow.side * g.fy
            yaw_grid = g.mz_nm if g.mz_nm is not None else f_right_grid * (vehicle.cg_x_m - fin_x_m)
            roll_grid = np.abs(g.mx_nm) if g.mx_nm is not None else np.abs(f_right_grid) * (fin_height_m + 0.5 * span_m)
            if cfg.max_yaw_moment_nm is not None: allowed &= np.abs(yaw_grid) <= cfg.max_yaw_moment_nm
            if cfg.max_roll_moment_nm is not None: allowed &= roll_grid <= cfg.max_roll_moment_nm
            if cfg.max_lift_n is not None: allowed &= np.abs(g.fz) <= cfg.max_lift_n
        score = g.fx - z * g.sigma_fx

        bi = None
        if allowed.any():
            bi = int(np.argmax(np.where(allowed, score, -np.inf)))     # first maximum == page's strict '>'
        best_d = float(g.d[bi]) if bi is not None else None
        best_fx = float(g.fx[bi]) if bi is not None else None
        best_score = float(score[bi]) if bi is not None else None

        def make(name, reason, code, on, delta, held=False, **kw):
            if on:
                f = fixed_f if (mode == "man") else _forces_at(g, self._grid_index(delta))
            else:
                f = zero
            dec = Decision(name, reason, code, on, delta if on else 0.0, held, f, fixed_f, g, allowed,
                           best_d, best_fx)
            self._annotate(dec, flow, lim, vehicle, fin_x_m, fin_height_m, span_m)
            return dec

        # ---- non-finite numbers from a model => fail safe -----------------------
        if not res.finite():
            st.state = RETRACT
            if not legacy:
                st.t = t
            return make(SAFE, MSG_BADNUM, "BAD_NUMBERS", False, 0.0)

        if fault:
            st.state = RETRACT
            if not legacy:
                st.t = t
            return make(SAFE, MSG_FAULT, "FAULT", False, 0.0)

        if mode == "ret":
            st.state = RETRACT
            if not legacy:
                st.t = t
            return make(RETRACT, MSG_RET, "OVERRIDE", False, 0.0)

        if mode == "man":
            if not legacy:
                st.state, st.t = RETRACT, t
            return make(MANUAL, MSG_MAN, "MANUAL", True, float(manual_deg))

        # ---- automatic mode ------------------------------------------------------
        enter = cfg.enter_fx_n
        leave = cfg.enter_fx_n if legacy else cfg.exit_fx_n
        thr = leave if st.state == PROPULSION else enter

        if flow.va < cfg.envelope_min_va or flow.va > cfg.envelope_max_va:
            cand, why, code, safety, c_delta = RETRACT, MSG_ENVELOPE, "ENVELOPE", True, None
        elif not legacy and not g.in_domain.any():
            cand, why, code, safety, c_delta = RETRACT, MSG_DOMAIN, "OUT_OF_DOMAIN", True, None
        elif bi is None:
            cand, why, code, safety, c_delta = RETRACT, MSG_SIDE, "SIDE_FORCE", True, None
        elif best_score < thr:
            unc = (not legacy) and best_fx >= thr
            cand, why, code, safety, c_delta = (RETRACT, MSG_UNCERT if unc else MSG_NOBEN,
                                                "UNCERTAIN" if unc else "NO_BENEFIT", False, None)
        else:
            cand, why, code, safety, c_delta = PROPULSION, MSG_OK, "OK", False, best_d

        # is the currently commanded angle still acceptable? (the page never asks)
        cur_ok = True
        if not legacy and st.state == PROPULSION:
            ci = self._grid_index(st.delta)
            cur_ok = bool(allowed[ci])

        held = False
        if cand != st.state:
            immediate = (not legacy) and (safety or (st.state == PROPULSION and not cur_ok))
            if (not immediate) and (t - st.t < cfg.dwell_s):
                held = True
            else:
                st.state, st.t = cand, t
                if c_delta is not None:
                    st.delta = c_delta
        elif cand == PROPULSION:
            if not legacy and not cur_ok:
                st.delta, why, code = c_delta, MSG_ANGLE, "ANGLE_INFEASIBLE"
            elif c_delta != st.delta:
                cur_score = float(score[self._grid_index(st.delta)]) if not legacy else \
                    float(g.fx[self._grid_index(st.delta)])
                gain = (best_score if not legacy else best_fx) - cur_score
                if gain > cfg.angle_improve_n:
                    st.delta = c_delta

        if held:
            why, code = MSG_HOLD, "DWELL_HOLD"
        on = st.state == PROPULSION
        return make(st.state, why, code, on, st.delta, held)

    # ------------------------------------------------------------------ diagnostics
    def _annotate(self, dec: Decision, flow: Flow, lim: float, veh: VehicleParams, fin_x_m: float,
                  fin_h_m: float, span_m: float) -> None:
        cfg = self.cfg
        if flow.va < cfg.envelope_min_va or flow.va > cfg.envelope_max_va:
            dec.violations.append("envelope")
        if dec.on:
            if abs(dec.f.fy) > lim:
                dec.violations.append("side_force_limit")
            if not dec.f.in_domain:
                dec.violations.append("out_of_model_domain")
            if dec.f.post_stall:
                dec.warnings.append("post_stall_extrapolation")
            f_right = flow.side * dec.f.fy                      # physical rightward force
            dec.yaw_moment_nm = (dec.f.mz_nm if dec.f.mz_nm is not None else
                                 f_right * (veh.cg_x_m - fin_x_m))
            dec.roll_moment_nm = (abs(dec.f.mx_nm) if dec.f.mx_nm is not None else
                                  abs(f_right) * (fin_h_m + 0.5 * span_m))
            dec.pitch_moment_nm = dec.f.my_nm if dec.f.my_nm is not None else 0.0
            if cfg.max_yaw_moment_nm is not None and abs(dec.yaw_moment_nm) > cfg.max_yaw_moment_nm:
                dec.violations.append("yaw_moment_limit")
            if cfg.max_roll_moment_nm is not None and dec.roll_moment_nm > cfg.max_roll_moment_nm:
                dec.violations.append("roll_moment_limit")
        if abs(dec.fixed_f.fy) > lim:
            dec.warnings.append("fixed_fin_exceeds_side_force_limit")

