from __future__ import annotations

import copy
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.blueprint_instance_defaults import REPAIR_BLUEPRINT_IDS

_FRIGATE_EVENTS = ("new", "update", "end")
_FRIGATE_OBJECTS = ("person", "car", "dog", "package", "cat")

def _merge_trigger_context(entity_obs: list[dict], patch: dict[str, Any]) -> list[dict]:
    if not entity_obs:
        entity_obs = [
            {
                "entity_id": "sensor.grounded",
                "domain": "sensor",
                "state": "on",
                "attributes": {},
            }
        ]
    out = copy.deepcopy(entity_obs)
    target = out[0]
    attrs = dict(target.get("attributes") or {})
    tc = dict(attrs.get("trigger_context") or {})
    tc.update(patch)
    attrs["trigger_context"] = tc
    target["attributes"] = attrs
    return out

def enrich_sample_evidence(
    sample: dict,
    *,
    blueprint_id: str,
    instance: dict,
    attempt_index: int = 0,
    visual_observed: dict | None = None,
) -> dict:
    if blueprint_id not in REPAIR_BLUEPRINT_IDS:
        return sample

    s = copy.deepcopy(sample)
    inputs = dict(instance.get("blueprint_inputs") or {})
    idx = attempt_index

    if blueprint_id == "appliance_power_google_sheets":
        power_entity = str(inputs.get("power_sensor") or "sensor.power_dishwasher")
        observed = dict(s.get("observed") or {})
        power_w = observed.get("current_power_w")
        if power_w is None:
            for eo in s.get("entity_observations") or []:
                attrs = eo.get("attributes") or {}
                if attrs.get("current_power_w") is not None:
                    power_w = attrs.get("current_power_w")
                    break
        working = float(inputs.get("working_power_threshold") or 200)
        idle = float(inputs.get("idle_power_threshold") or 5)
        transition = "working_start" if power_w is not None and float(power_w) >= working else "idle_to_working"
        tc = {
            "trigger_id": "power_state_change",
            "power_sensor": power_entity,
            "working_power_threshold": working,
            "idle_power_threshold": idle,
            "power_transition": transition,
        }
        if power_w is not None:
            tc["current_power_w"] = power_w
        s["entity_observations"] = _merge_trigger_context(s.get("entity_observations") or [], tc)
        if s["entity_observations"]:
            s["entity_observations"][0]["entity_id"] = power_entity
            s["entity_observations"][0]["domain"] = "sensor"
        s["observed"] = {**observed, "current_power_w": power_w if power_w is not None else working + 50}
        bb = dict(s.get("blueprint_binding") or {})
        snap = dict(bb.get("grounded_instance") or {})
        snap["blueprint_inputs"] = inputs
        snap["bound_entities"] = sorted(
            {
                v
                for v in inputs.values()
                if isinstance(v, str) and "." in v
            }
        )
        bb["grounded_instance"] = snap
        bb["entities"] = snap["bound_entities"]
        s["blueprint_binding"] = bb

    elif blueprint_id == "presence_holiday_away_lighting":
        tc = {
            "away_mode": True,
            "holiday_mode": True,
            "zone_empty": True,
            "automation_control": "enable_zone",
            "automation_control_zone": "zone.home",
        }
        s["entity_observations"] = _merge_trigger_context(s.get("entity_observations") or [], tc)
        sys_st = dict(s.get("system_state") or {})
        sys_st.update({"away_mode": True, "holiday_mode": True, "zone_empty": True})
        s["system_state"] = sys_st
        bb = dict(s.get("blueprint_binding") or {})
        snap = dict(bb.get("grounded_instance") or {})
        snap["blueprint_inputs"] = inputs
        entities = list(inputs.get("light_entities") or inputs.get("include_entity") or [])
        snap["bound_entities"] = [e for e in entities if isinstance(e, str)]
        bb["grounded_instance"] = snap
        bb["entities"] = snap["bound_entities"]
        s["blueprint_binding"] = bb

    elif blueprint_id in ("camera_frigate_vision_llm", "camera_frigate_intelligent"):
        camera = str(inputs.get("camera") or "camera.living_room")
        event_type = _FRIGATE_EVENTS[idx % len(_FRIGATE_EVENTS)]
        object_type = _FRIGATE_OBJECTS[idx % len(_FRIGATE_OBJECTS)]
        if visual_observed:
            object_type = str(visual_observed.get("object_type") or visual_observed.get("detected_class") or object_type)
        event_id = f"evt_{idx:05d}"
        review_id = f"review_{idx:05d}"
        tc = {
            "trigger_id": "frigate_event",
            "frigate_event_type": event_type,
            "objects_match": True,
            "zone_match": True,
            "severity_match": True,
            "camera": camera,
            "event_id": event_id,
            "review_id": review_id,
            "object_type": object_type,
            "notify_targets": ["mobile_app_phone"],
        }
        s["entity_observations"] = _merge_trigger_context(s.get("entity_observations") or [], tc)
        observed = dict(s.get("observed") or {})
        if visual_observed:
            observed = {**visual_observed, **observed}
        observed.update(
            {
                "frigate_event_type": event_type,
                "object_type": object_type,
                "event_id": event_id,
                "review_id": review_id,
                "camera_entity": camera,
            }
        )
        s["observed"] = observed
        meta = dict(s.get("synthesis_metadata") or {})
        meta["temporal_pattern"] = "frigate_event"
        s["synthesis_metadata"] = meta

    return s

def repair_sample_evidence_valid(sample: dict, blueprint_id: str) -> tuple[bool, str]:
    if blueprint_id not in REPAIR_BLUEPRINT_IDS:
        return True, "ok"

    tc: dict[str, Any] = {}
    for eo in sample.get("entity_observations") or []:
        t = (eo.get("attributes") or {}).get("trigger_context")
        if isinstance(t, dict):
            tc.update(t)
    sys_st = dict(sample.get("system_state") or {})
    obs = dict(sample.get("observed") or {})
    inputs = ((sample.get("blueprint_binding") or {}).get("grounded_instance") or {}).get("blueprint_inputs") or {}

    if blueprint_id == "appliance_power_google_sheets":
        if not (inputs.get("power_sensor") or inputs.get("working_power_threshold")):
            return False, "missing_power_monitor_inputs"
        if tc.get("trigger_id") != "power_state_change" and not tc.get("power_transition"):
            return False, "missing_power_state_change_trigger"
        if obs.get("current_power_w") is None and not any(
            (eo.get("attributes") or {}).get("current_power_w") is not None for eo in sample.get("entity_observations") or []
        ):
            return False, "missing_power_evidence"
        return True, "ok"

    if blueprint_id == "presence_holiday_away_lighting":
        away = tc.get("away_mode") or sys_st.get("away_mode")
        holiday = tc.get("holiday_mode") or sys_st.get("holiday_mode")
        if not (away and holiday):
            return False, "missing_away_holiday_evidence"
        return True, "ok"

    if blueprint_id in ("camera_frigate_vision_llm", "camera_frigate_intelligent"):
        if not (tc.get("frigate_event_type") or obs.get("frigate_event_type")):
            return False, "missing_frigate_event"
        if not (inputs.get("camera") or ((sample.get("blueprint_binding") or {}).get("entities"))):
            return False, "missing_camera_binding"
        return True, "ok"

    return True, "ok"
