from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from smarthome_mdf.multi_action_vnext.source_adapters import CAPABILITY_INDEX, SourceObservation, _primary_measurement
from smarthome_mdf.single_scene_blueprint_complete.semantic_sanitize import infer_primary_semantic
from smarthome_mdf.single_scene_blueprint_complete.semantic_capabilities import attach_capabilities, infer_semantic_capabilities
from smarthome_mdf.single_scene_blueprint_complete.semantic_sanitize import sanitize_attributes
from smarthome_mdf.paths import DATA_DIR
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

CANONICAL_PATH = DATA_DIR / "v4_synthesis" / "intermediate" / "canonical_observations.jsonl"
EVIDENCE_WINDOWS_PATH = DATA_DIR / "v4_synthesis" / "intermediate" / "evidence_windows.jsonl"

DATASET_SCENE_COMPAT: Dict[str, List[str]] = {
    "CASAS": [
        "advanced_lighting",
        "climate_window",
        "notification_security",
        "on_off_schedule",
        "periodic_task_scheduling",
        "scene_schedule_override",
        "visual_fusion",
    ],
    "UK-DALE": ["appliance_monitoring", "periodic_task_scheduling"],
    "YouHome": ["advanced_lighting", "visual_fusion"],
    "REFIT": ["appliance_monitoring"],
}

def _entity_domain_from_canonical(entity: dict) -> str | None:
    return str(entity.get("domain") or "") or None

def _observation_from_canonical(row: dict) -> SourceObservation | None:
    source = row.get("source") or {}
    entity = row.get("entity") or {}
    obs = row.get("observation") or {}
    dataset = str(source.get("dataset") or "UNKNOWN")
    cap = CAPABILITY_INDEX.get(dataset)
    obs_type = str(obs.get("type") or "")
    value = obs.get("value")
    measurement: str | None = None
    state: Any = None
    attrs: dict[str, Any] = {}

    if obs_type == "measurement":
        unit = str(obs.get("unit") or "").lower()
        if unit in ("w", "kw"):
            measurement = "power"
            attrs["current_power_w"] = value
        elif "c" in unit or unit == "°c":
            measurement = "temperature"
            attrs["temperature"] = value
            attrs["indoor_temperature"] = value
        else:
            measurement = "observed"
            attrs["value"] = value
    elif obs_type == "state":
        state = value
        device_class = str(entity.get("device_class") or "")
        if device_class == "motion":
            measurement = "motion"
            attrs["motion_state"] = value
            attrs["motion"] = value == "on"
        elif device_class in ("door", "window"):
            measurement = "window"
            attrs["window_state"] = "open" if value == "open" else "closed"
        else:
            measurement = "observed"
            attrs["state"] = value
    elif obs_type == "event" and isinstance(value, dict):
        if value.get("person_detected") is not None:
            measurement = "motion"
            attrs["person_detected"] = bool(value["person_detected"])
            attrs["motion_state"] = "on" if value["person_detected"] else "off"
        if value.get("illuminance_lux") is not None:
            attrs["illuminance_lux"] = value["illuminance_lux"]
            if measurement is None:
                measurement = "illuminance"
        for k, v in value.items():
            attrs[k] = v

    domain = _entity_domain_from_canonical(entity)
    synthetic = {
        "timestamp": row.get("timestamp"),
        **attrs,
    }
    if measurement is None:
        measurement = _primary_measurement(synthetic, scene=None, trigger_ctx=None)

    caps = infer_semantic_capabilities(attrs, device_class=str(entity.get("device_class") or ""), dataset=dataset)
    obs = SourceObservation(
        dataset=dataset,
        record_id=str(row.get("observation_id") or source.get("record_id") or ""),
        original_timestamp=str(row.get("timestamp") or "") or None,
        entity_domain=domain,
        measurement=measurement,
        value=value if obs_type == "measurement" else attrs.get("illuminance_lux") or state,
        state=state,
        attributes=attrs,
        history=[],
        source_metadata={
            "source_type": "canonical_observation",
            "canonical_entity_id": entity.get("canonical_entity_id"),
            "device_class": entity.get("device_class"),
        },
        provenance={
            "source_dataset": dataset,
            "source_record_id": source.get("record_id"),
            "source_type": "GROUNDED",
            "source_file": source.get("source_file"),
        },
        scene_type="",
        capabilities={
            "entity_domains": list(cap.entity_domains) if cap else ([domain] if domain else []),
            "measurements": list(cap.measurements) if cap else ([measurement] if measurement else []),
            "history_available": bool(cap.history_available if cap else False),
            "semantic_capabilities": sorted(caps),
        },
    )
    return attach_capabilities(obs)

def _observation_from_evidence_window(row: dict, obs_by_id: Dict[str, dict]) -> SourceObservation | None:
    snapshot = row.get("current_snapshot") or {}
    entities = snapshot.get("entities") or []
    measurements = list(row.get("measurements") or [])
    dataset = (row.get("source_datasets") or ["CASAS"])[0]
    cap = CAPABILITY_INDEX.get(str(dataset))

    ent = entities[0] if entities else {}
    domain = str(ent.get("domain") or "sensor")
    device_class = str(ent.get("device_class") or "")
    state = ent.get("state")
    measurement: str | None = None
    attrs: dict[str, Any] = {
        "entity_id": ent.get("entity_id"),
        "device_class": device_class,
        "area": ent.get("area"),
    }

    if device_class == "motion":
        measurement = "motion"
        attrs["motion_state"] = state
        attrs["motion"] = state == "on"
    elif device_class in ("door", "window"):
        measurement = "window"
        attrs["window_state"] = state

    power_history: list[dict] = []
    if measurements:
        measurement = "power"
        last = measurements[-1]
        attrs["current_power_w"] = last.get("value")
        power_history = [
            {"timestamp": m.get("timestamp"), "value": m.get("value"), "entity": m.get("entity"), "event_type": "measurement"}
            for m in measurements
        ]
        if len(measurements) >= 2 and float(measurements[-1].get("value") or 0) > float(measurements[0].get("value") or 0) * 1.5:
            attrs.setdefault("trigger_context", {})
            if isinstance(attrs["trigger_context"], dict):
                attrs["trigger_context"].update(
                    {"event_type": "state_changed", "to_state": "on", "trigger_id": "power_state_change"}
                )

    for m in measurements:
        if "temperature" in str(m.get("entity", "")).lower() or m.get("unit") == "°C":
            attrs["temperature"] = m.get("value")
            attrs["indoor_temperature"] = m.get("value")

    clean_attrs, removed = sanitize_attributes(attrs, device_class=device_class)
    if removed:
        attrs = clean_attrs

    event_history = list(row.get("event_history") or [])
    state_history = list(row.get("state_history") or [])
    history = event_history or state_history or power_history
    tc: dict[str, Any] = dict(attrs.get("trigger_context") or {})
    if event_history:
        last = event_history[-1]
        tc = {
            "event_type": last.get("event_type") or "state_change",
            "to_state": last.get("value"),
            "entity": last.get("entity"),
        }
        if len(event_history) >= 2:
            tc["from_state"] = event_history[-2].get("value")

    attrs["trigger_context"] = tc
    if not state and power_history:
        state = "on" if float(attrs.get("current_power_w") or 0) > 10 else "off"

    return SourceObservation(
        dataset=str(dataset),
        record_id=str(row.get("window_id") or ""),
        original_timestamp=str(snapshot.get("timestamp") or row.get("end_time") or "") or None,
        entity_domain=domain if domain else ("sensor" if measurement == "power" else "binary_sensor"),
        measurement=measurement or _primary_measurement(attrs, trigger_ctx=tc),
        value=attrs.get("current_power_w") or state,
        state=state,
        attributes=attrs,
        history=history,
        source_metadata={
            "source_type": "evidence_window",
            "trigger_context": tc,
            "window_id": row.get("window_id"),
        },
        provenance={
            "source_dataset": dataset,
            "source_record_id": row.get("window_id"),
            "source_type": "GROUNDED",
            "observation_ids": row.get("observation_ids") or [],
        },
        scene_type="",
        capabilities={
            "entity_domains": list(cap.entity_domains) if cap else [domain or "sensor"],
            "measurements": list(cap.measurements) if cap else [],
            "history_available": True,
        },
    )

def _load_casas_contact_observations(max_events: int = 120) -> List[SourceObservation]:

    try:
        from smarthome_mdf.synthesis.loaders import load_casas_home_index
        from smarthome_mdf.v4_synthesis.canonical_schema import casas_to_canonical
    except ImportError:
        return []

    index = load_casas_home_index(max_per_home=20000)
    out: List[SourceObservation] = []
    idx = 0
    for _home, events in index.items():
        for ev in events:
            if ev.sensor_type != "D":
                continue
            row = casas_to_canonical(ev, idx).to_dict()
            so = _observation_from_canonical(row)
            if so is not None:
                out.append(so)
                idx += 1
            if len(out) >= max_events:
                return out
    return out

def load_public_observations(
    canonical_path: Path | None = None,
    windows_path: Path | None = None,
) -> Dict[str, List[SourceObservation]]:

    canonical_path = canonical_path or CANONICAL_PATH
    windows_path = windows_path or EVIDENCE_WINDOWS_PATH
    by_scene: Dict[str, List[SourceObservation]] = {}
    seen: Set[str] = set()

    if canonical_path.is_file():
        for row in read_jsonl(canonical_path):
            so = _observation_from_canonical(row)
            if so is None or so.record_id in seen:
                continue
            seen.add(so.record_id)
            for scene in DATASET_SCENE_COMPAT.get(so.dataset, []):
                so_copy = SourceObservation(**{**so.__dict__, "scene_type": scene})
                by_scene.setdefault(scene, []).append(so_copy)

    obs_by_id: Dict[str, dict] = {}
    if canonical_path.is_file():
        for row in read_jsonl(canonical_path):
            obs_by_id[str(row.get("observation_id"))] = row

    if windows_path.is_file():
        for row in read_jsonl(windows_path):
            so = _observation_from_evidence_window(row, obs_by_id)
            if so is None or so.record_id in seen:
                continue
            seen.add(so.record_id)
            for scene in DATASET_SCENE_COMPAT.get(so.dataset, []):
                so_copy = SourceObservation(**{**so.__dict__, "scene_type": scene})
                by_scene.setdefault(scene, []).append(so_copy)

    for so in _load_casas_contact_observations():
        if so.record_id in seen:
            continue
        seen.add(so.record_id)
        for scene in DATASET_SCENE_COMPAT.get(so.dataset, []):
            so_copy = SourceObservation(**{**so.__dict__, "scene_type": scene})
            by_scene.setdefault(scene, []).append(so_copy)

    return by_scene

def merge_public_into_pool(pool: SourcePool, scenes: List[str]) -> SourcePool:

    public = load_public_observations()
    if not hasattr(pool, "_public_by_scene"):
        pool._public_by_scene = public
    else:
        pool._public_by_scene = public

    orig = pool.candidates_for_scene

    def _merged_candidates(scene: str) -> List[SourceObservation]:
        legacy = orig(scene)
        pub = public.get(scene, [])
        if not pub:
            return legacy
        legacy_ids = {o.record_id for o in legacy}
        extra = [o for o in pub if o.record_id not in legacy_ids]
        return legacy + extra

    pool.candidates_for_scene = _merged_candidates
    return pool

def probe_raw_datasets() -> dict[str, Any]:

    from smarthome_mdf.v4_synthesis.public_data_adapter import check_dataset_availability

    statuses = check_dataset_availability()
    canonical_count = 0
    window_count = 0
    by_dataset: Dict[str, int] = {}
    if CANONICAL_PATH.is_file():
        for row in read_jsonl(CANONICAL_PATH):
            canonical_count += 1
            ds = (row.get("source") or {}).get("dataset", "UNKNOWN")
            by_dataset[ds] = by_dataset.get(ds, 0) + 1
    if EVIDENCE_WINDOWS_PATH.is_file():
        window_count = sum(1 for _ in read_jsonl(EVIDENCE_WINDOWS_PATH))

    return {
        "raw_dataset_status": [s.__dict__ for s in statuses],
        "canonical_observations_count": canonical_count,
        "evidence_windows_count": window_count,
        "canonical_by_dataset": by_dataset,
        "paths": {
            "canonical_observations": str(CANONICAL_PATH),
            "evidence_windows": str(EVIDENCE_WINDOWS_PATH),
        },
    }
