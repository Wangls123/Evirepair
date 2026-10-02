from __future__ import annotations

from typing import Any

from baselines.autotap.property_compiler import compile_ss_property

def _single_rule(ir: dict[str, Any], props: list[dict[str, Any]]) -> dict[str, list]:
    obs = ir.get("observations") or {}
    acts = ir.get("actions") or []
    win = str(obs.get("window_state") or "").lower()
    mot = str(obs.get("motion_state") or obs.get("occupancy") or "").lower()
    out: dict[str, list] = {}
    templates = {str(p.get("template") or "") for p in props}

    climate_act = next((a for a in acts if str(a.get("service") or "").startswith("climate")), None)
    light_act = next((a for a in acts if str(a.get("service") or "").startswith("light")), None)

    if "never_ac_on_and_window_open" in templates:
        svc = str((climate_act or {}).get("service") or "")
        val = "off" if "off" in svc or "turn_off" in svc else "on"
        trig = "window.switch=on" if win in {"open", "on", "true"} else "window.switch=off"
        out["airConditioner.switch"] = [[[trig], ["none"], [val], [0, "none"]]]
    if "motion_light_safety" in templates:
        svc = str((light_act or {}).get("service") or "")
        val = "on" if "turn_on" in svc or svc.endswith(".on") else "off"
        trig = "motionSensor.motion=active" if mot in {"on", "true", "active", "detected"} else "motionSensor.motion=inactive"
        out["light.switch"] = [[[trig], ["none"], [val], [0, "none"]]]
    return out

def _tapfixer_spec(prop: dict[str, Any]) -> list:
    tmpl = str(prop.get("template") or "")
    if tmpl == "never_ac_on_and_window_open":
        return [r"--LTLSPEC G((window.switch = on)->(airConditioner.switch = off))", "state-state"]
    if tmpl == "motion_light_safety":
        return [
            r"--LTLSPEC G((motionSensor.motion = active & motionSensor.motion != motionSensor.motion_last)->X(light.switch = on))",
            "state-event",
        ]
    ltl = str(prop.get("ltl") or "")
    return [f"--LTLSPEC {ltl}", "state-state"]

def compile_ss_property_tapfixer(ir: dict[str, Any]) -> dict[str, Any]:

    base = compile_ss_property(ir)
    if not base.get("applicable"):
        return {
            "applicable": False,
            "status": str(base.get("status") or "NOT_APPLICABLE"),
            "properties": [],
            "rules": {},
            "specs": [],
            "reason": str(base.get("reason") or "ss_not_mappable_onto_tapfixer_predefined_or_level2_property"),
            "n_rules": 0,
        }
    props = list(base.get("properties") or [])
    rules = _single_rule(ir, props)
    if not rules:
        return {
            "applicable": False,
            "status": "PROPERTY_UNAVAILABLE",
            "properties": props,
            "rules": {},
            "specs": [],
            "reason": "single_rule_could_not_be_encoded_in_tapfixer_rule_dict",
            "n_rules": 0,
        }
    n_rules = sum(len(v) for v in rules.values())
    if n_rules != 1:

        return {
            "applicable": False,
            "status": "PROPERTY_UNAVAILABLE",
            "properties": props,
            "rules": rules,
            "specs": [],
            "reason": f"ss_maps_to_{n_rules}_tapfixer_rules_single_automation_required",
            "n_rules": n_rules,
        }
    specs = [_tapfixer_spec(p) for p in props[:1]]
    return {
        "applicable": True,
        "status": "APPLICABLE",
        "properties": props[:1],
        "rules": rules,
        "specs": specs,
        "reason": "single_rule_tapfixer_model_from_explicit_ss_observation",
        "n_rules": 1,
        "level": "level2_explicit_observation",
    }
