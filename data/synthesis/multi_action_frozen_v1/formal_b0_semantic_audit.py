from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.b0_adapter import blueprint_yaml_bridge
from smarthome_mdf.multi_action_frozen_v1.frozen_instance_resolver import resolve_frozen_instance
from smarthome_mdf.multi_action_frozen_v1.frozen_runtime_projector import project_component_runtime
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path
from smarthome_mdf.synthesis_v3.blueprint_executor import execute_blueprint
from smarthome_mdf.synthesis_v3.blueprint_parser import parse_blueprint_yaml
from smarthome_mdf.synthesis_v3.scenario_generators.base import _executor_observed, _executor_runtime

FOCUS_BLUEPRINTS = (
    "appliance_power_google_sheets",
    "camera_frigate_intelligent",
    "camera_frigate_vision_llm",
    "lighting_motion_advanced_v22",
    "presence_holiday_away_lighting",
)

def yaml_predicate_eligible(
    bid: str,
    inputs: dict,
    tc: dict,
    obs: dict,
    ss: dict,
    rs: dict,
) -> tuple[bool, str]:

    if bid == "lighting_motion_advanced_v22":
        ms = str(obs.get("motion_state") or tc.get("trigger_id") or "").lower()
        lux = obs.get("illuminance_lux")
        if lux is None:
            lux = obs.get("illuminance")
        lux_min = float(inputs.get("lux_min") or 0)
        lux_max = float(inputs.get("lux_max") or 100000)
        motion_on = ms in ("on", "detected", "true") or tc.get("trigger_id") == "motion_on"
        lux_ok = lux is not None and lux_min <= float(lux) <= lux_max
        has_target = bool(inputs.get("light_target"))
        has_scene = any(
            inputs.get(k) not in (None, "scene.none", "")
            for k in ("scene_morning", "scene_day", "scene_evening", "scene_night", "scene_sleep_mode")
        )
        if motion_on and lux_ok and (has_target or has_scene):
            return True, "motion_on_lux_ok_with_action_target"
        return False, f"motion={motion_on} lux_ok={lux_ok} target={has_target} scene={has_scene}"

    if bid == "presence_holiday_away_lighting":
        sys_st = dict(rs.get("system_state") or ss.get("system_state") or {})
        away = bool(tc.get("away_mode") or tc.get("holiday_mode") or sys_st.get("away_mode"))
        if away:
            return True, "away_or_holiday_mode"
        return False, "automation_control_not_active_no_away_mode"

    if bid in ("camera_frigate_intelligent", "camera_frigate_vision_llm"):
        camera = inputs.get("in_camera") or inputs.get("camera") or obs.get("camera_entity")
        event = tc.get("frigate_event_type") or obs.get("frigate_event_type")
        if camera and event in ("new", "end"):
            return True, "camera_and_frigate_event"
        return False, f"camera={bool(camera)} event={event}"

    if bid == "appliance_power_google_sheets":
        power = obs.get("current_power_w") or obs.get("current_power")
        idle = float(inputs.get("idle_power_threshold") or 5)
        working = float(inputs.get("working_power_threshold") or 200)
        if power is None:
            return False, "no_power_reading"
        p = float(power)
        if p >= working or (p > idle and inputs.get("is_working_boolean")):
            return True, "power_state_transition_eligible"
        if p >= idle:
            return True, "power_above_idle_yaml_semantics"
        return False, f"power={p} below_idle={idle}"

    return False, "unknown"

def audit_action_eligible_miss(
    corpus: list[dict[str, Any]],
    frozen_index: dict[str, dict[str, Any]],
    *,
    focus_only: bool = False,
) -> dict[str, Any]:

    misses: list[dict[str, Any]] = []
    per_bp: dict[str, dict[str, int]] = defaultdict(lambda: {"eligible": 0, "miss": 0, "actions": 0})

    with blueprint_yaml_bridge():
        for ma in corpus:
            for comp in ma.get("components") or []:
                bid = str(comp.get("blueprint_id") or "")
                if focus_only and bid not in FOCUS_BLUEPRINTS:
                    continue
                ss_sample = frozen_index.get(str(comp.get("single_scene_sample_id") or ""), {})
                scene = str(comp.get("scene") or ss_sample.get("scene_type") or "")
                iid = str(comp.get("grounded_instance") or "")
                inst = resolve_frozen_instance(iid, blueprint_id=bid, scene=scene)
                inputs = inst.get("blueprint_inputs") or {}
                rs = comp.get("runtime_state") or {}
                proj = project_component_runtime(
                    scene=scene,
                    single_scene_sample=ss_sample,
                    component_runtime=rs,
                    blueprint_inputs=inputs,
                )
                ypath = resolve_yaml_path(bid)
                spec = parse_blueprint_yaml(bid, ypath, inputs)
                inst_dict = {
                    "blueprint_id": bid,
                    "scene_type": scene,
                    "blueprint_inputs": inputs,
                    "automation_instance_id": iid,
                }
                exec_obs = _executor_observed(
                    proj["observed"],
                    proj["derived_observation"],
                    proj["synthetic_runtime_state"],
                    scene,
                    automation_instance=inst_dict,
                )
                exec_rt = _executor_runtime(proj["runtime_memory"], proj["synthetic_runtime_state"], scene)
                oracle = execute_blueprint(
                    spec,
                    inst_dict,
                    trigger_context=proj["trigger_context"],
                    observed=exec_obs,
                    runtime_memory=exec_rt,
                    entity_states=proj.get("entity_states"),
                )
                acts = len(oracle.get("actions") or [])
                per_bp[bid]["actions"] += acts
                elig, reason = yaml_predicate_eligible(bid, inputs, proj["trigger_context"], exec_obs, ss_sample, rs)
                if elig:
                    per_bp[bid]["eligible"] += 1
                    if acts == 0:
                        per_bp[bid]["miss"] += 1
                        misses.append(
                            {
                                "multi_action_id": ma.get("multi_action_id"),
                                "component_id": comp.get("component_id"),
                                "blueprint_id": bid,
                                "reason": reason,
                                "failed_conditions": oracle.get("failed_conditions"),
                            }
                        )

    return {
        "B0_ACTION_ELIGIBLE_MISS": len(misses),
        "miss_samples": misses[:20],
        "per_blueprint": dict(per_bp),
    }

def run_counterfactual_tests() -> dict[str, Any]:
    results: dict[str, Any] = {}
    with blueprint_yaml_bridge():
        bid = "lighting_motion_advanced_v22"
        yp = resolve_yaml_path(bid)
        inputs = {
            "motion_entity": "binary_sensor.motion",
            "light_target": {"entity_id": "light.kitchen"},
            "no_motion_wait": 120,
        }
        spec = parse_blueprint_yaml(bid, yp, inputs)
        inst = {"blueprint_id": bid, "scene_type": "advanced_lighting", "blueprint_inputs": inputs}
        oracle = execute_blueprint(
            spec,
            inst,
            trigger_context={"trigger_id": "motion_on", "local_time": "12:00:00"},
            observed={"illuminance_lux": 100},
            runtime_memory={},
        )
        results[bid] = {
            "actions": len(oracle.get("actions") or []),
            "failed": oracle.get("failed_conditions"),
            "pass": len(oracle.get("actions") or []) > 0,
        }

        bid = "appliance_power_google_sheets"
        yp = resolve_yaml_path(bid)
        inputs = {"power_sensor": "sensor.power", "working_power_threshold": 200, "idle_power_threshold": 5}
        spec = parse_blueprint_yaml(bid, yp, inputs)
        inst = {"blueprint_id": bid, "scene_type": "periodic_task_scheduling", "blueprint_inputs": inputs}
        oracle = execute_blueprint(
            spec,
            inst,
            trigger_context={"trigger_id": "power_state_change"},
            observed={"current_power_w": 350},
            runtime_memory={},
        )
        results[bid] = {
            "actions": len(oracle.get("actions") or []),
            "failed": oracle.get("failed_conditions"),
            "pass": len(oracle.get("actions") or []) > 0,
        }

        for cam_bid in ("camera_frigate_vision_llm", "camera_frigate_intelligent"):
            yp = resolve_yaml_path(cam_bid)
            inputs = {"camera": "camera.front", "notify_device": "notify.mobile_app_phone"}
            spec = parse_blueprint_yaml(cam_bid, yp, inputs)
            inst = {"blueprint_id": cam_bid, "scene_type": "visual_fusion", "blueprint_inputs": inputs}
            oracle = execute_blueprint(
                spec,
                inst,
                trigger_context={
                    "frigate_event_type": "new",
                    "objects_match": True,
                    "zone_match": True,
                    "severity_match": True,
                    "notify_targets": ["mobile_app_phone"],
                    "event_id": "evt1",
                },
                observed={"object_type": "person"},
                runtime_memory={},
            )
            results[cam_bid] = {
                "actions": len(oracle.get("actions") or []),
                "services": [a.get("service") for a in oracle.get("actions") or []],
                "pass": len(oracle.get("actions") or []) > 0,
            }

        bid = "presence_holiday_away_lighting"
        yp = resolve_yaml_path(bid)
        inputs = {"holiday_scene": "scene.away"}
        spec = parse_blueprint_yaml(bid, yp, inputs)
        inst = {"blueprint_id": bid, "scene_type": "scene_schedule_override", "blueprint_inputs": inputs}
        oracle = execute_blueprint(
            spec,
            inst,
            trigger_context={"away_mode": True, "holiday_mode": True},
            observed={},
            runtime_memory={},
        )
        results[bid] = {
            "actions": len(oracle.get("actions") or []),
            "pass": len(oracle.get("actions") or []) > 0,
        }

    return results

def audit_zero_action_blueprints_from_b0(
    corpus: list[dict[str, Any]],
    b0_records: list[dict[str, Any]],
    frozen_index: dict[str, dict[str, Any]],
    *,
    formal_b0_sha256: str | None = None,
) -> dict[str, Any]:

    b0_by_id = {str(r["multi_action_id"]): r for r in b0_records}
    per_bp: dict[str, dict[str, int]] = {bid: defaultdict(int) for bid in FOCUS_BLUEPRINTS}
    classifications: dict[str, Counter] = {bid: Counter() for bid in FOCUS_BLUEPRINTS}

    elig_audit = audit_action_eligible_miss(corpus, frozen_index, focus_only=True)

    for ma in corpus:
        rec = b0_by_id.get(str(ma.get("multi_action_id")), {})
        per_comp = {str(c.get("component_id")): c for c in rec.get("per_component_b0") or []}
        for comp in ma.get("components") or []:
            bid = str(comp.get("blueprint_id") or "")
            if bid not in FOCUS_BLUEPRINTS:
                continue
            per_bp[bid]["formal_executions"] += 1
            pc = per_comp.get(str(comp.get("component_id")), {})
            acts = len(pc.get("semantic_actions") or [])
            per_bp[bid]["executor_actions"] += acts
            if acts > 0:
                classifications[bid]["ACTIONS_PRODUCED"] += 1
            else:
                classifications[bid]["NO_ACTION"] += 1

    for bid in FOCUS_BLUEPRINTS:
        bp_stats = elig_audit["per_blueprint"].get(bid, {})
        per_bp[bid]["predicate_action_eligible"] = bp_stats.get("eligible", 0)
        per_bp[bid]["predicate_no_action"] = per_bp[bid]["formal_executions"] - bp_stats.get("eligible", 0)

    counterfactual = run_counterfactual_tests()
    cf_fail = sum(1 for v in counterfactual.values() if not v.get("pass"))

    unexplained = sum(
        1
        for bid in FOCUS_BLUEPRINTS
        if per_bp[bid].get("predicate_action_eligible", 0) > 0 and per_bp[bid].get("executor_actions", 0) == 0
    )

    defect = unexplained > 0 or cf_fail > 0 or elig_audit["B0_ACTION_ELIGIBLE_MISS"] > 0
    verdict = "FORMAL_B0_V2_SEMANTIC_SANITY_CONFIRMED" if not defect else "FORMAL_B0_V2_SEMANTIC_DEFECT_FOUND"

    return {
        "verdict": verdict,
        "formal_b0_frozen_sha256": formal_b0_sha256,
        "per_blueprint": {
            bid: {**dict(per_bp[bid]), "classifications": dict(classifications[bid])} for bid in FOCUS_BLUEPRINTS
        },
        "ZERO_ACTION_BLUEPRINT_UNEXPLAINED": unexplained,
        "COUNTERFACTUAL_ACTION_TEST_FAIL": cf_fail,
        "B0_ACTION_ELIGIBLE_MISS": elig_audit["B0_ACTION_ELIGIBLE_MISS"],
        "counterfactual_tests": counterfactual,
    }
