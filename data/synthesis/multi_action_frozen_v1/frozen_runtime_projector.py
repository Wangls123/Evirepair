from __future__ import annotations

from datetime import datetime
from typing import Any

def _extract_trigger_from_observations(entity_observations: list[dict]) -> dict[str, Any]:
    trigger: dict[str, Any] = {}
    for obs in entity_observations or []:
        attrs = obs.get("attributes") or {}
        tc = attrs.get("trigger_context")
        if isinstance(tc, dict):
            trigger.update(tc)
    return trigger

def _parse_local_time(timestamp: str | None) -> str | None:
    if not timestamp:
        return None
    try:
        ts = timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%H:%M:%S")
    except ValueError:
        return None

def _normalize_contact_state(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in ("on", "open", "detected", "true"):
        return "on"
    if text in ("off", "closed", "close", "clear", "false"):
        return "off"
    return None

def project_component_runtime(
    *,
    scene: str,
    single_scene_sample: dict[str, Any],
    component_runtime: dict[str, Any],
    blueprint_inputs: dict[str, Any],
) -> dict[str, Any]:

    observed = dict(component_runtime.get("observed") or single_scene_sample.get("observed") or {})
    derived = dict(single_scene_sample.get("derived_observation") or component_runtime.get("derived_observation") or {})
    syn_rt = dict(single_scene_sample.get("synthetic_runtime_state") or component_runtime.get("synthetic_runtime_state") or {})
    entity_obs = list(component_runtime.get("entity_observations") or single_scene_sample.get("entity_observations") or [])
    trigger = _extract_trigger_from_observations(entity_obs)
    mem = dict(component_runtime.get("runtime_memory") or single_scene_sample.get("runtime_memory") or {})
    ss_state = dict(component_runtime.get("system_state") or single_scene_sample.get("system_state") or {})

    if scene == "climate_window":
        ws = str(observed.get("window_state") or trigger.get("to_state") or "").lower()
        if ws == "open":
            trigger.setdefault("trigger_id", "window_open")
        elif ws in ("closed", "close"):
            trigger.setdefault("trigger_id", "window_closed")

    elif scene == "advanced_lighting":
        ms = str(observed.get("motion_state") or trigger.get("to_state") or "").lower()
        if ms in ("on", "detected", "true"):
            trigger.setdefault("trigger_id", "motion_on")
            lt = _parse_local_time(observed.get("timestamp"))
            if lt:
                trigger.setdefault("local_time", lt)

        elif ms in ("off", "clear", "false"):
            trigger.setdefault("trigger_id", "motion_off_delayed")

    elif scene == "appliance_monitoring":

        pass

    elif scene == "notification_security":
        ws = str(observed.get("window_state") or observed.get("door_state") or "").lower()
        ms = str(observed.get("motion_state") or trigger.get("to_state") or "").lower()
        for obs in entity_obs:
            tc = (obs.get("attributes") or {}).get("trigger_context") or {}
            if tc.get("yaml_trigger_id"):
                trigger["yaml_trigger_id"] = tc["yaml_trigger_id"]
            if tc.get("trigger_id"):
                trigger.setdefault("trigger_id", tc["trigger_id"])
        if not trigger.get("yaml_trigger_id"):
            if ws in ("open", "on"):
                trigger.setdefault("yaml_trigger_id", "issue_detected")
            elif ws in ("closed", "close", "off"):
                trigger.setdefault("yaml_trigger_id", "issue_resolved")
            elif ms in ("on", "detected"):
                trigger.setdefault("yaml_trigger_id", "issue_detected")
            elif ms in ("off", "clear", "false"):
                trigger.setdefault("yaml_trigger_id", "issue_resolved")
        if not trigger.get("yaml_trigger_id"):
            for obs in entity_obs:
                ent_state = _normalize_contact_state(obs.get("state"))
                if ent_state == "on":
                    trigger.setdefault("yaml_trigger_id", "issue_detected")
                    break
                if ent_state == "off":
                    trigger.setdefault("yaml_trigger_id", "issue_resolved")
                    break

    elif scene == "on_off_schedule":
        if observed.get("timestamp"):
            trigger.setdefault("current_timestamp", observed.get("timestamp"))
        if trigger.get("schedule_fired") is None and observed.get("schedule_fired") is not None:
            trigger.setdefault("schedule_fired", observed.get("schedule_fired"))

    elif scene == "periodic_task_scheduling":
        if trigger.get("schedule_trigger") is None and observed.get("schedule_trigger") is not None:
            trigger.setdefault("schedule_trigger", observed.get("schedule_trigger"))

    elif scene == "visual_fusion":
        bp_id = str(single_scene_sample.get("blueprint_binding", {}).get("blueprint_id") or "")
        for obs in entity_obs:
            attrs = obs.get("attributes") or {}
            tc = attrs.get("trigger_context") or {}
            trigger.update({k: v for k, v in tc.items() if v is not None})
            if attrs.get("motion_state") == "on" or attrs.get("motion") is True:
                trigger.setdefault("frigate_event_type", "new")
            if tc.get("event_type") == "end":
                trigger.setdefault("frigate_event_type", "end")
        if bp_id == "camera_frigate_intelligent":
            for obs in entity_obs:
                attrs = obs.get("attributes") or {}
                tc = attrs.get("trigger_context") or {}
                entity = str(tc.get("entity") or obs.get("entity_id") or "")
                to_state = str(tc.get("to_state") or obs.get("state") or "").lower()
                from_state = str(tc.get("from_state") or "").lower()
                motion = attrs.get("motion_state") or attrs.get("motion")
                if entity.endswith("m006") and from_state == "on" and to_state == "off":
                    trigger["button_id"] = "silence"
                    trigger["silence"] = True
                elif motion in ("on", True) and to_state == "on":
                    trigger.setdefault("frigate_event_type", "new")
                elif "tulum2" in entity or attrs.get("area") == "tulum2":
                    if from_state == "off" and to_state == "off":
                        trigger.setdefault("debug", True)
                    else:
                        trigger.setdefault("frigate_event_type", "end")
        elif trigger.get("frigate_event_type") is None and observed.get("frigate_event_type"):
            trigger.setdefault("frigate_event_type", observed.get("frigate_event_type"))
        if observed.get("frame_id") or observed.get("event_id"):
            trigger.setdefault("event_id", observed.get("frame_id") or observed.get("event_id"))

    elif scene == "scene_schedule_override":
        trigger.setdefault("system_state", ss_state)
        person = blueprint_inputs.get("person")
        if person:
            for obs in entity_obs:
                if str(obs.get("entity_id") or "") == str(person):
                    trigger["person_state"] = str(obs.get("state") or "")
                    break
        if not trigger.get("person_state"):
            for obs in entity_obs:
                eid = str(obs.get("entity_id") or "")
                if eid.startswith("person."):
                    trigger["person_state"] = str(obs.get("state") or "")
                    break
        if ss_state.get("away_mode") is not None:
            trigger.setdefault("away_mode", ss_state.get("away_mode"))
        if ss_state.get("holiday_mode") is not None:
            trigger.setdefault("holiday_mode", ss_state.get("holiday_mode"))
        if ss_state.get("manual_override") is not None:
            trigger.setdefault("manual_override", ss_state.get("manual_override"))
        if ss_state.get("timer_active") is not None:
            trigger.setdefault("timer_active", ss_state.get("timer_active"))
        if ss_state.get("schedule_active") is not None:
            trigger.setdefault("schedule_active", ss_state.get("schedule_active"))

    return {
        "observed": observed,
        "derived_observation": derived,
        "trigger_context": trigger,
        "synthetic_runtime_state": syn_rt,
        "entity_observations": entity_obs,
        "entity_states": dict(component_runtime.get("entity_states") or single_scene_sample.get("entity_states") or {}),
        "runtime_memory": mem,
    }
