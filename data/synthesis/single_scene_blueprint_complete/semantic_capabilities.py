from __future__ import annotations

from typing import Any

from smarthome_mdf.multi_action_vnext.source_adapters import SourceObservation
from smarthome_mdf.single_scene_blueprint_complete.semantic_sanitize import infer_primary_semantic

REQUIREMENT_CAPABILITY: dict[str, frozenset[str]] = {
    "power": frozenset({"power"}),
    "energy": frozenset({"power", "energy"}),
    "motion": frozenset({"motion"}),
    "person_detected": frozenset({"motion", "person_detected", "visual_frame"}),
    "temperature": frozenset({"temperature"}),
    "illuminance": frozenset({"illuminance"}),
    "window": frozenset({"window", "door"}),
    "door": frozenset({"door", "window"}),
    "schedule": frozenset({"schedule", "time"}),
    "time": frozenset({"time", "schedule"}),
    "vibration": frozenset({"vibration"}),
    "observed": frozenset({"observed"}),
    "trigger_context": frozenset({"trigger_context", "motion", "window", "door", "power"}),
}

def infer_semantic_capabilities(attrs: dict[str, Any], *, device_class: str | None = None, dataset: str | None = None) -> frozenset[str]:
    caps: set[str] = set()
    primary = infer_primary_semantic(attrs, device_class)
    dc = (device_class or attrs.get("device_class") or "").lower()
    if primary == "motion" or dc == "motion" or attrs.get("motion_state") is not None or attrs.get("motion") is not None:
        caps.add("motion")
    if attrs.get("person_detected") is not None and not attrs.get("frame_id"):
        caps.add("motion")
    if primary == "power" and (attrs.get("current_power_w") is not None or attrs.get("energy") is not None):
        caps.add("power")
    if primary == "temperature" and (attrs.get("temperature") is not None or attrs.get("indoor_temperature") is not None):
        caps.add("temperature")
    if primary == "illuminance" and (attrs.get("illuminance_lux") is not None or attrs.get("illuminance") is not None):
        caps.add("illuminance")
    if primary in ("window", "unknown") and (attrs.get("window_state") is not None or attrs.get("door_state") is not None):
        caps.add("window")
        caps.add("door")
    if attrs.get("frame_id") is not None:
        caps.add("visual_frame")
    if attrs.get("trigger_context"):
        caps.add("trigger_context")
    if not caps and dataset == "UK-DALE":
        caps.add("power")
    return frozenset(caps)

def attach_capabilities(obs: SourceObservation) -> SourceObservation:
    attrs = dict(obs.attributes or {})
    caps = infer_semantic_capabilities(attrs, device_class=str(attrs.get("device_class") or ""), dataset=obs.dataset)
    obs.capabilities = {**(obs.capabilities or {}), "semantic_capabilities": sorted(caps)}
    meta = dict(obs.source_metadata or {})
    meta["raw_sensor_type"] = attrs.get("device_class") or obs.measurement
    meta["semantic_capabilities"] = sorted(caps)
    obs.source_metadata = meta
    return obs

def observation_has_capability(obs: SourceObservation, measurement: str) -> bool:
    required = REQUIREMENT_CAPABILITY.get(measurement, frozenset({measurement}))
    caps = frozenset((obs.capabilities or {}).get("semantic_capabilities") or [])
    if not caps:
        caps = infer_semantic_capabilities(obs.attributes or {}, device_class=str((obs.attributes or {}).get("device_class") or ""), dataset=obs.dataset)
    if caps & required:
        return True

    if measurement == "power":
        return "power" in caps and (obs.attributes or {}).get("current_power_w") is not None
    if measurement in ("motion", "person_detected"):
        return bool(caps & {"motion", "person_detected", "visual_frame"})
    if measurement == "temperature":
        return "temperature" in caps
    if measurement in ("window", "door"):
        return bool(caps & {"window", "door"})
    if measurement == "observed":
        return bool(caps) or bool(obs.attributes) or bool(obs.history) or obs.state is not None
    return measurement in caps
