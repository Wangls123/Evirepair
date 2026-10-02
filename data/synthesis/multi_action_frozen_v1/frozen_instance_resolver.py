from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.config import FROZEN_V1_DIR
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import _default_inputs

_FROZEN_SS_ROOT: Path | None = None

def set_frozen_ss_root(path: Path | None) -> None:

    global _FROZEN_SS_ROOT
    _FROZEN_SS_ROOT = path
    _load_instance_index.cache_clear()

def _frozen_ss_root() -> Path:
    return _FROZEN_SS_ROOT or FROZEN_V1_DIR

def _grounded_candidates_path() -> Path:
    return _frozen_ss_root() / "grounded" / "grounded_instance_candidates.json"

GROUNDED_CANDIDATES_PATH = FROZEN_V1_DIR / "grounded" / "grounded_instance_candidates.json"
TEMPLATES_PATH = FROZEN_V1_DIR.parent / "automation_instance_templates.json"

@lru_cache(maxsize=1)
def _load_instance_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    candidates_path = _grounded_candidates_path()
    if candidates_path.is_file():
        raw = json.loads(candidates_path.read_text(encoding="utf-8"))
        for _bp, rows in (raw or {}).items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                iid = str(row.get("automation_instance_id") or "")
                if iid:
                    index[iid] = row
    if TEMPLATES_PATH.is_file():
        doc = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
        for _bp, rows in (doc.get("instances") or {}).items():
            for row in rows or []:
                iid = str(row.get("automation_instance_id") or "")
                if iid and iid not in index:
                    index[iid] = {
                        "automation_instance_id": iid,
                        "blueprint_id": row.get("blueprint_id"),
                        "scene": row.get("scene"),
                        "input_bindings": dict(row.get("blueprint_inputs") or {}),
                        "bound_entities": list(row.get("bound_entities") or []),
                    }
    return index

def resolve_frozen_instance(
    automation_instance_id: str,
    *,
    blueprint_id: str,
    scene: str,
) -> dict[str, Any]:

    index = _load_instance_index()
    row = index.get(automation_instance_id)
    if row:
        inputs = dict(row.get("input_bindings") or row.get("blueprint_inputs") or {})
        return {
            "automation_instance_id": automation_instance_id,
            "blueprint_id": row.get("blueprint_id") or blueprint_id,
            "scene": row.get("scene") or scene,
            "blueprint_inputs": inputs,
            "bound_entities": list(row.get("bound_entities") or row.get("entity_bindings") or []),
            "configuration_signature": row.get("configuration_signature"),
            "resolution_source": "grounded_instance_candidates",
        }
    defaults = _default_inputs(scene, blueprint_id)
    return {
        "automation_instance_id": automation_instance_id,
        "blueprint_id": blueprint_id,
        "scene": scene,
        "blueprint_inputs": dict(defaults),
        "bound_entities": [],
        "resolution_source": "default_inputs_fallback",
        "resolution_warning": "instance_not_in_grounded_cache",
    }

def audit_instance_reconstruction(
    automation_instance_id: str,
    *,
    blueprint_id: str,
    scene: str,
) -> dict[str, Any]:
    inst = resolve_frozen_instance(automation_instance_id, blueprint_id=blueprint_id, scene=scene)
    invalid = (
        inst.get("resolution_source") == "default_inputs_fallback"
        and automation_instance_id in _load_instance_index()
    ) or (not inst.get("blueprint_inputs") and automation_instance_id.endswith("_inst_"))
    return {
        "automation_instance_id": automation_instance_id,
        "blueprint_id": blueprint_id,
        "invalid": invalid,
        "resolution_source": inst.get("resolution_source"),
        "blueprint_inputs_nonempty": bool(inst.get("blueprint_inputs")),
    }
