from __future__ import annotations

import json
import re
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.y_canonical_semantics import build_canonical_blueprint_semantics
from smarthome_mdf.multi_action_frozen_v1.y_compact_prompt import (
    FROZEN_INDEPENDENT_Y_V5_SS_SYSTEM_PROMPT,
    build_compact_ss_user_prompt,
)
from smarthome_mdf.multi_action_frozen_v1.y_labeling_view import (
    _entity_catalog_for_component,
    _sanitize_evidence_dict,
    build_candidate_actions_ha,
    build_frozen_y_user_prompt,
    build_y_labeling_view,
)

_ENTITY_ID_RE = re.compile(r"^[a-z][a-z0-9_]+\.[a-zA-Z0-9_]+$")

def _walk_entity_ids(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        eid = obj.get("entity_id")
        if isinstance(eid, str) and _ENTITY_ID_RE.match(eid):
            found.append(eid)
        elif isinstance(eid, list):
            found.extend(str(x) for x in eid if isinstance(x, str) and _ENTITY_ID_RE.match(x))
        for value in obj.values():
            found.extend(_walk_entity_ids(value))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str) and _ENTITY_ID_RE.match(item):
                found.append(item)
            else:
                found.extend(_walk_entity_ids(item))
    elif isinstance(obj, str) and _ENTITY_ID_RE.match(obj):
        found.append(obj)
    return found

def _entity_binding_from_ss(sample: dict[str, Any]) -> dict[str, Any]:
    bb = sample.get("blueprint_binding") or {}
    entities = list(bb.get("entities") or [])
    records: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(ent: Any, role: str = "entity") -> None:
        eid = str(ent or "").strip()
        if not eid or eid in seen:
            return
        seen.add(eid)
        records.append({"component_local_entity": eid, "role": role})

    for ent in entities:
        _add(ent)
    for eo in sample.get("entity_observations") or []:
        _add(eo.get("entity_id"))
    for ent in _walk_entity_ids(_blueprint_inputs_from_ss(sample)):
        role = "entity"
        if ent.startswith("light."):
            role = "light_target"
        elif ent.startswith("scene."):
            role = "scene"
        elif ent.startswith("climate."):
            role = "climate"
        _add(ent, role)
    return {"records": records, "component_entity_map": {}}

def _blueprint_inputs_from_ss(sample: dict[str, Any]) -> dict[str, Any]:
    bb = sample.get("blueprint_binding") or {}
    gi = bb.get("grounded_instance") or {}
    return dict(gi.get("blueprint_inputs") or bb.get("blueprint_inputs") or {})

def _ss_as_component(sample: dict[str, Any]) -> dict[str, Any]:
    bb = sample.get("blueprint_binding") or {}
    return {
        "component_id": "c0",
        "blueprint_id": bb.get("blueprint_id"),
        "scene": sample.get("scene_type"),
        "blueprint_inputs": _blueprint_inputs_from_ss(sample),
        "entity_binding": _entity_binding_from_ss(sample),
        "temporal_alignment": None,
        "runtime_state": _sanitize_evidence_dict(
            {
                "observed": sample.get("observed") or {},
                "system_state": sample.get("system_state") or {},
                "entity_observations": sample.get("entity_observations") or [],
            }
        ),
        "single_scene_sample_id": sample.get("sample_id"),
    }

def build_single_scene_y_labeling_view(sample: dict[str, Any]) -> dict[str, Any]:

    comp = _ss_as_component(sample)
    bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
    bp_inputs = comp.get("blueprint_inputs") or {}
    semantics = build_canonical_blueprint_semantics(bp, bp_inputs)
    candidate_ha, candidate_audit = build_candidate_actions_ha([comp])
    entity_catalog = _entity_catalog_for_component(comp)

    return {
        "sample_id": sample.get("sample_id"),
        "single_scene": True,
        "component_count": 1,
        "blueprint_id": bp,
        "scene_type": sample.get("scene_type"),
        "components": [comp],
        "canonical_blueprint_semantics": {bp: semantics},
        "candidate_actions_ha": candidate_ha,
        "candidate_action_domain": candidate_audit.get("per_component_domain") or {},
        "entity_catalog": entity_catalog,
        "candidate_actions_audit": candidate_audit,
        "evidence": _sanitize_evidence_dict(
            {
                "observed": sample.get("observed") or {},
                "system_state": sample.get("system_state") or {},
                "entity_observations": sample.get("entity_observations") or [],
            }
        ),
        "labeling_mode": "independent_single_scene_frozen_y2_forced_decision",
    }

def build_single_scene_y_user_prompt(view: dict[str, Any], sample: dict[str, Any] | None = None) -> str:

    if sample is None:
        sample = {
            "observed": (view.get("evidence") or {}).get("observed") or {},
            "entity_observations": (view.get("evidence") or {}).get("entity_observations") or [],
        }
    return build_compact_ss_user_prompt(view, sample)

def build_ma_y_labeling_view(ma: dict[str, Any], frozen_index: dict[str, dict[str, Any]]) -> dict[str, Any]:

    return build_y_labeling_view(ma, frozen_index)

def build_multi_action_y_user_prompt(view: dict[str, Any], _ma: dict[str, Any] | None = None) -> str:

    return build_frozen_y_user_prompt(view)

FROZEN_INDEPENDENT_Y_SS_SYSTEM_PROMPT = FROZEN_INDEPENDENT_Y_V5_SS_SYSTEM_PROMPT
