from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.config import FROZEN_INVENTORY

def _entity_role(entity_id: str) -> str:
    low = entity_id.lower()
    if "motion" in low or "occupancy" in low:
        return "motion"
    if "lux" in low or "illumin" in low:
        return "lux"
    if "window" in low:
        return "window"
    if "door" in low or "contact" in low:
        return "contact"
    if "power" in low or "energy" in low:
        return "power"
    if "climate" in low or "thermostat" in low or "temp" in low:
        return "climate"
    if "light" in low:
        return "light"
    if "camera" in low or "frigate" in low:
        return "camera"
    if "person" in low or "device_tracker" in low or "presence" in low:
        return "presence"
    if "notify" in low:
        return "notify"
    return entity_id.split(".")[0] if "." in entity_id else "entity"

def _action_domain(action_path: str) -> str:
    if not action_path:
        return "unknown"
    svc = action_path.split("|")[0].split("→")[-1]
    return svc.split(".")[0] if "." in svc else svc

def _runtime_requirements(sample: dict) -> set[str]:
    reqs: set[str] = set()
    for obs in sample.get("entity_observations") or []:
        dom = obs.get("domain") or (obs.get("entity_id", "").split(".")[0])
        if dom:
            reqs.add(f"entity:{dom}")
    tp = (sample.get("synthesis_metadata") or {}).get("temporal_pattern") or {}
    if isinstance(tp, dict):
        if tp.get("event_type"):
            reqs.add(f"temporal:{tp['event_type']}")
        if tp.get("trigger_id"):
            reqs.add(f"trigger:{tp['trigger_id']}")
    elif isinstance(tp, str) and tp:
        reqs.add(f"temporal:{tp}")
    obs = sample.get("observed") or {}
    if "current_power_w" in obs:
        reqs.add("measurement:power")
    if "timestamp" in obs:
        reqs.add("observation:timestamp")
    return reqs

def _history_requirements(blueprint_id: str, action_domain: str) -> list[str]:
    hist: list[str] = []
    if "appliance" in blueprint_id or action_domain in {"input_number", "input_text"}:
        hist.append("power_history")
    if "climate" in blueprint_id or action_domain == "climate":
        hist.append("climate_state")
    return hist

@dataclass
class BlueprintProfile:
    blueprint_id: str
    scene: str
    action_domain: str
    entity_roles: list[str]
    runtime_requirement_types: list[str]
    temporal_requirements: list[str]
    history_requirements: list[str]
    source_dataset_capabilities: list[str]
    grounded_instance_count: int
    sample_count: int
    status: str = "GROUNDED_CAPABLE"
    action_path_signatures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def build_blueprint_profiles(samples: list[dict]) -> dict[str, BlueprintProfile]:
    inventory: dict[str, dict] = {}
    if FROZEN_INVENTORY.is_file():
        for row in json.loads(FROZEN_INVENTORY.read_text(encoding="utf-8")):
            inventory[row["blueprint_id"]] = row

    by_bp: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        bp_id = s["blueprint_binding"]["blueprint_id"]
        by_bp[bp_id].append(s)

    profiles: dict[str, BlueprintProfile] = {}
    for bp_id, inv in inventory.items():
        rows = by_bp.get(bp_id, [])
        roles: set[str] = set()
        runtimes: set[str] = set()
        temporal: set[str] = set()
        actions: set[str] = set()
        sources: set[str] = set()
        instances: set[str] = set()
        action_dom = "unknown"

        for s in rows:
            bb = s["blueprint_binding"]
            instances.add(bb.get("automation_instance_id", ""))
            prov = s.get("provenance") or {}
            sources.add(prov.get("source_dataset", ""))
            meta = s.get("synthesis_metadata") or {}
            ap = meta.get("action_path_signature") or ""
            if ap:
                actions.add(ap)
                action_dom = _action_domain(ap)
            for ent in bb.get("entities") or []:
                roles.add(_entity_role(ent))
            runtimes |= _runtime_requirements(s)
            tp = meta.get("temporal_pattern") or {}
            if isinstance(tp, dict):
                if tp.get("event_type"):
                    temporal.add(tp["event_type"])
                if tp.get("trigger_id"):
                    temporal.add(tp["trigger_id"])
            elif isinstance(tp, str) and tp:
                temporal.add(tp)

        if rows and action_dom == "unknown":
            action_dom = _action_domain(next(iter(actions), ""))

        profiles[bp_id] = BlueprintProfile(
            blueprint_id=bp_id,
            scene=inv.get("scene") or (rows[0]["scene_type"] if rows else "unknown"),
            action_domain=action_dom,
            entity_roles=sorted(roles),
            runtime_requirement_types=sorted(runtimes),
            temporal_requirements=sorted(temporal),
            history_requirements=_history_requirements(bp_id, action_dom),
            source_dataset_capabilities=sorted(s for s in sources if s),
            grounded_instance_count=len(instances),
            sample_count=len(rows),
            status=inv.get("status", "GROUNDED_CAPABLE"),
            action_path_signatures=sorted(actions),
        )
    return profiles
