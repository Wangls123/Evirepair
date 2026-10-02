from __future__ import annotations

from typing import Any

CLASSIFIER_VERSION = "final_gold_y_action_type_v1"

DEVICE_PREFIXES = (
    "light.",
    "switch.",
    "climate.",
    "cover.",
    "fan.",
    "lock.",
    "scene.",
    "alarm_control_panel.",
    "humidifier.",
    "water_heater.",
    "vacuum.",
    "remote.",
    "siren.",
    "media_player.",
    "valve.",
)

EVENT_PREFIXES = (
    "notify.",
    "persistent_notification.",
    "system_log.",
    "logbook.",
    "google_sheets.",
    "camera.",
    "tts.",
    "schedule.",
    "rest_command.",
)

EVENT_EXACT = {
    "system_log.write",
    "logbook.log",
    "google_sheets.append_sheet",
    "persistent_notification.create",
    "persistent_notification.dismiss",
    "input_text.set_value",
    "schedule.activate",
}

EVENT_PRIORITY = (
    "system_log.write",
    "logbook.log",
    "google_sheets.append_sheet",
    "persistent_notification.create",
    "persistent_notification.dismiss",
    "notify.mobile_app",
    "notify.notify",
    "input_text.set_value",
    "schedule.activate",
)

STATE_PRIORITY_PREFIXES = (
    "climate.",
    "light.",
    "switch.",
    "scene.",
    "cover.",
    "fan.",
    "lock.",
    "input_boolean.",
)

def service_kind(service: str) -> str:
    s = str(service or "").strip()
    if not s:
        return "OTHER"
    if s in EVENT_EXACT or s.startswith(EVENT_PREFIXES):
        return "EVENT_ACTION"
    if s.startswith(DEVICE_PREFIXES) or s.startswith("input_boolean."):
        return "STATE_ACTION"
    return "OTHER"

def _walk_services(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        svc = obj.get("service")
        if svc:
            out.append(str(svc))
        for v in obj.values():
            _walk_services(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _walk_services(item, out)

def extract_services_from_payload(payload: dict[str, Any]) -> list[str]:
    found: list[str] = []
    intended = payload.get("intended_action_definition") or {}
    spec = payload.get("automation_specification") or {}
    for s in intended.get("candidate_services") or []:
        if s:
            found.append(str(s))
    for s in spec.get("services") or []:
        if s:
            found.append(str(s))
    _walk_services(spec.get("control_flow"), found)

    seen: set[str] = set()
    out: list[str] = []
    for s in found:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _pick_event(services: list[str]) -> str:
    rank = {n: i for i, n in enumerate(EVENT_PRIORITY)}
    events = [s for s in services if service_kind(s) == "EVENT_ACTION"]
    if not events:
        return ""
    events.sort(key=lambda s: rank.get(s, 100 if not s.startswith("notify.") else 50))
    return events[0]

def _pick_state(services: list[str]) -> str:
    for prefix in STATE_PRIORITY_PREFIXES:
        for s in services:
            if s.startswith(prefix):
                return s
    for s in services:
        if service_kind(s) == "STATE_ACTION":
            return s
    return services[0] if services else ""

def classify_services(services: list[str]) -> dict[str, Any]:

    svcs = [str(s) for s in services if s]
    device = [s for s in svcs if any(s.startswith(p) for p in DEVICE_PREFIXES)]
    event = [s for s in svcs if service_kind(s) == "EVENT_ACTION"]
    helpers = [
        s
        for s in svcs
        if s.startswith(("input_boolean.", "input_datetime.", "input_number.", "input_select."))
    ]
    if device:
        primary = _pick_state(device)
        return {
            "action_type": "STATE_ACTION",
            "declared_effect_service": primary,
            "services": svcs,
            "reason": "device_domain_present",
        }
    if event:
        primary = _pick_event(event)
        return {
            "action_type": "EVENT_ACTION",
            "declared_effect_service": primary,
            "services": svcs,
            "reason": "event_effect_without_device_domain",
        }
    if helpers:
        return {
            "action_type": "STATE_ACTION",
            "declared_effect_service": helpers[0],
            "services": svcs,
            "reason": "helper_state_only",
        }
    if svcs:
        return {
            "action_type": "EVENT_ACTION",
            "declared_effect_service": svcs[0],
            "services": svcs,
            "reason": "unknown_treated_as_event_effect",
        }
    return {
        "action_type": "STATE_ACTION",
        "declared_effect_service": "",
        "services": [],
        "reason": "empty_service_list",
    }

def classify_payload(payload: dict[str, Any]) -> dict[str, Any]:
    services = extract_services_from_payload(payload)
    result = classify_services(services)
    result["classifier_version"] = CLASSIFIER_VERSION
    result["action_type_source"] = "intended_action_definition+automation_specification"
    return result
