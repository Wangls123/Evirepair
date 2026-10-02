from __future__ import annotations

from typing import Any

ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "motion": frozenset(
        {
            "motion_state",
            "motion",
            "person_detected",
            "entity_id",
            "device_class",
            "area",
            "trigger_context",
            "detection_confidence",
        }
    ),
    "power": frozenset(
        {
            "current_power_w",
            "energy",
            "appliance_id",
            "channel_id",
            "entity_id",
            "device_class",
            "area",
            "trigger_context",
        }
    ),
    "temperature": frozenset(
        {"temperature", "indoor_temperature", "entity_id", "device_class", "area", "trigger_context"}
    ),
    "illuminance": frozenset({"illuminance_lux", "illuminance", "entity_id", "device_class", "area", "trigger_context"}),
    "window": frozenset({"window_state", "door_state", "entity_id", "device_class", "area", "trigger_context"}),
    "visual": frozenset(
        {
            "participant_id",
            "sequence_id",
            "frame_id",
            "person_detected",
            "detection_confidence",
            "illuminance_lux",
            "entity_id",
            "device_class",
            "trigger_context",
        }
    ),
}

CONTAMINATION_PAIRS = [
    ("motion", "current_power_w"),
    ("motion", "temperature"),
    ("motion", "indoor_temperature"),
    ("motion", "illuminance_lux"),
    ("binary_sensor.motion", "current_power_w"),
]

def infer_primary_semantic(attrs: dict[str, Any], device_class: str | None = None) -> str:
    dc = (device_class or attrs.get("device_class") or "").lower()
    if dc == "motion" or attrs.get("motion_state") is not None or attrs.get("motion") is not None:
        return "motion"
    if attrs.get("current_power_w") is not None:
        return "power"
    if attrs.get("indoor_temperature") is not None or attrs.get("temperature") is not None:
        return "temperature"
    if attrs.get("illuminance_lux") is not None:
        return "illuminance"
    if attrs.get("window_state") is not None or attrs.get("door_state") is not None:
        return "window"
    if attrs.get("frame_id") is not None:
        return "visual"
    if attrs.get("person_detected") is not None and attrs.get("frame_id"):
        return "visual"
    return "motion" if dc == "motion" else "unknown"

def sanitize_attributes(attrs: dict[str, Any], *, device_class: str | None = None) -> tuple[dict[str, Any], list[str]]:

    if not attrs:
        return {}, []
    primary = infer_primary_semantic(attrs, device_class)
    allowed = ALLOWED_ATTRS.get(primary, frozenset(attrs.keys()))
    removed: list[str] = []
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        if k in allowed or k == "trigger_context":
            out[k] = v
        else:
            removed.append(k)
    return out, removed

def detect_semantic_contamination(sample: dict) -> list[dict[str, Any]]:

    hits: list[dict[str, Any]] = []
    for eo in sample.get("entity_observations") or []:
        attrs = dict(eo.get("attributes") or {})
        dc = attrs.get("device_class") or eo.get("domain")
        primary = infer_primary_semantic(attrs, str(dc) if dc else None)
        _, removed = sanitize_attributes(attrs, device_class=str(dc) if dc else None)
        if removed:
            hits.append(
                {
                    "entity_id": eo.get("entity_id"),
                    "primary_semantic": primary,
                    "invalid_fields": removed,
                    "device_class": dc,
                }
            )
    return hits

def contamination_invalidates_match(hits: list[dict]) -> bool:

    critical = {"current_power_w", "temperature", "indoor_temperature", "illuminance_lux"}
    for h in hits:
        if critical & set(h.get("invalid_fields") or []):
            if h.get("primary_semantic") == "motion":
                return True
    return False
