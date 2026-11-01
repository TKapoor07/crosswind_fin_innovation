"""Tractive power and energy bookkeeping (the page's 'race-level toy numbers', made explicit).

With default assumptions this reproduces the page exactly:
    Fd  = 0.5 rho Cd A Vx |Vx|           Vx = W cos(B) + v   (axial apparent wind, signed)
    Ft0 = Crr m g + Fd                    P0 = Ft0 v           (no fin)
    P1  = (Ft0 - fin_fx) v                                     (with fin)
Energy is MECHANICAL tractive energy; ``battery_equivalent`` divides by the drivetrain efficiency.

What this does NOT include unless you set ``EnergyAssumptions``: actuator power, the fin's mass,
body-drag change from fin/body interaction, and the body's yaw-dependent drag/side force.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import EnergyAssumptions, VehicleParams
from .kinematics import Flow


def baseline_tractive_force(flow: Flow, veh: VehicleParams) -> float:
    fd = 0.5 * veh.air_density * veh.cd * veh.frontal_area_m2 * flow.sx * abs(flow.sx)
    return veh.crr * veh.mass_kg * veh.g + fd


def power_no_fin(flow: Flow, veh: VehicleParams) -> float:
    return baseline_tractive_force(flow, veh) * flow.v_ms


def power_with_fin(flow: Flow, veh: VehicleParams, fin_fx: float, deployed: bool,
                   assume: EnergyAssumptions) -> float:
    """Tractive power with the fin installed. Extra mass raises rolling resistance whether or not
    the fin is deployed; actuator power is charged only while it is deployed."""
    ft = baseline_tractive_force(flow, veh) + veh.crr * assume.fin_extra_mass_kg * veh.g - fin_fx
    p = ft * flow.v_ms
    if deployed:
        p += assume.actuator_power_w * veh.drive_efficiency  # actuator draws electrical power; express as mechanical-equivalent
    return p


@dataclass
class EnergyLedger:
    """Running totals in Wh. 'gain' = energy saved relative to the no-fin car (positive is good)."""
    adaptive_wh: float = 0.0
    fixed_wh: float = 0.0
    seconds: float = 0.0

    def add(self, p_none_w: float, p_adaptive_w: float, p_fixed_w: float, dt: float) -> None:
        self.adaptive_wh += (p_none_w - p_adaptive_w) * dt / 3600.0
        self.fixed_wh += (p_none_w - p_fixed_w) * dt / 3600.0
        self.seconds += dt

    def reset(self) -> None:
        self.adaptive_wh = self.fixed_wh = self.seconds = 0.0

    def battery_equivalent(self, eta: float) -> tuple[float, float]:
        return self.adaptive_wh / eta, self.fixed_wh / eta
