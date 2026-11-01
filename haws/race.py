"""Deterministic longitudinal race-energy comparison; inputs are scenarios, not predictions of race results."""
from __future__ import annotations
import math
import numpy as np
from .config import VEHICLE, EnergyAssumptions
from .kinematics import apparent_wind
from .geometry import resolve
from .aero.toy import ToyModelV2
from .energy import baseline_tractive_force, power_with_fin

def simulate_race(payload: dict):
    steps=payload.get("steps",[])
    if not steps or len(steps)>20000: raise ValueError("steps must contain 1..20000 samples")
    dt_default=float(payload.get("dt_s",1.0)); eta=float(payload.get("drive_efficiency",VEHICLE.drive_efficiency)); initial_soc=float(payload.get("initial_soc_pct",100.0))
    if not 0.05<=eta<=1: raise ValueError("drive_efficiency must be between 0.05 and 1")
    if not 0<=initial_soc<=100: raise ValueError("initial_soc_pct must be between 0 and 100")
    assume=EnergyAssumptions(actuator_power_w=float(payload.get("actuator_power_w",0)),
        fin_extra_mass_kg=float(payload.get("fin_extra_mass_kg",0)),solar_power_w=float(payload.get("solar_power_w",0)))
    model=ToyModelV2(); totals={k:0. for k in ("baseline_wh","fixed_wh","adaptive_wh","solar_wh")}; elapsed=0.; distance=0.
    rows=[]
    for s in steps:
        v=float(s.get("v_kmh",60))/3.6; wind=float(s.get("wind_ms",0)); bearing=float(s.get("wind_from_deg",0)); dt=float(s.get("dt_s",dt_default))
        if not 0<dt<=60 or not 0<=v<=40 or not 0<=wind<=40: raise ValueError("invalid race step bounds")
        flow=apparent_wind(v,wind,bearing,rho=VEHICLE.air_density)
        geom=resolve(s.get("geometry","B"),s.get("span_m"),s.get("chord_m"))
        angle=float(s.get("fin_angle_deg",0)); arr=model.predict(geom,flow,np.array([angle]))
        fx=float(arr.fx[0]); deployed=bool(s.get("deployed",True))
        p0=baseline_tractive_force(flow,VEHICLE)*v
        pfix=power_with_fin(flow,VEHICLE,fx,deployed,assume)
        # adaptive strategy uses fin only if explicit per-step flag; otherwise safe stowed baseline
        ad_on=bool(s.get("adaptive_deployed",False)); pa=power_with_fin(flow,VEHICLE,fx if ad_on else 0,ad_on,assume)
        solar=assume.solar_power_w
        totals["baseline_wh"]+=p0*dt/3600/eta
        totals["fixed_wh"]+=pfix*dt/3600/eta
        totals["adaptive_wh"]+=pa*dt/3600/eta
        totals["solar_wh"]+=solar*dt/3600
        elapsed+=dt; distance+=v*dt
        rows.append({"time_s":elapsed,"speed_kmh":v*3.6,"baseline_power_w":p0,"fixed_power_w":pfix,"adaptive_power_w":pa})
    totals["baseline_net_wh"]=totals["baseline_wh"]-totals["solar_wh"]
    totals["fixed_net_wh"]=totals["fixed_wh"]-totals["solar_wh"]
    totals["adaptive_net_wh"]=totals["adaptive_wh"]-totals["solar_wh"]
    totals["distance_km"]=distance/1000; totals["elapsed_s"]=elapsed
    cap=VEHICLE.battery_nominal_wh; start_wh=cap*initial_soc/100
    totals["initial_soc_pct"]=initial_soc
    for strategy,key in (("baseline","baseline_net_wh"),("fixed","fixed_net_wh"),("adaptive","adaptive_net_wh")):
        totals[f"{strategy}_final_soc_pct"]=100*(start_wh-totals[key])/cap
    return {"assumptions":{"drive_efficiency":eta,"actuator_power_w":assume.actuator_power_w,"fin_extra_mass_kg":assume.fin_extra_mass_kg,
                           "solar_power_w":solar,"aero_model":"toy_v2 (unvalidated; isolated fin approximation)"},"totals":totals,"trace":rows}
