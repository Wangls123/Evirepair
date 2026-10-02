from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.config import (
    ALL_BLUEPRINT_IDS,
    OUTPUT_DIR,
    SCENE_BLUEPRINT_MAP,
)
from smarthome_mdf.single_scene_blueprint_complete.inventory import build_full_inventory
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path, yaml_exists

def yaml_hash(path: Path) -> str:
    if not path.is_file():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="replace")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def build_registry() -> dict[str, Any]:
    inv = build_full_inventory()
    blueprints: dict[str, dict] = {}
    for row in inv["blueprints"]:
        bp_id = row["blueprint_id"]
        scene = row["scene"]
        ypath = resolve_yaml_path(bp_id)
        abs_path = str(ypath.resolve()) if ypath else None
        blueprints[bp_id] = {
            "blueprint_id": bp_id,
            "blueprint_name": row["blueprint_name"],
            "blueprint_path": abs_path,
            "scene_types": [scene],
            "yaml_exists": yaml_exists(bp_id),
            "contract_source": row["contract_path"],
            "registry_role": "canonical_22",
        }
    return {
        "version": "22_blueprint_registry_v1",
        "description": "Complete 22-blueprint registry for single-scene synthesis. Not the Phase-1 7-entry subset.",
        "total_blueprints": len(blueprints),
        "blueprints": blueprints,
        "scene_to_blueprints": SCENE_BLUEPRINT_MAP,
    }

def write_registry(out_dir: Path | None = None) -> Path:
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    reg = build_registry()
    path = out / "blueprint_registry.json"
    path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def load_registry(path: Path | None = None) -> dict:
    p = path or (OUTPUT_DIR / "blueprint_registry.json")
    if not p.is_file():
        write_registry(p.parent)
    return json.loads(p.read_text(encoding="utf-8"))

def get_blueprint_meta(blueprint_id: str, registry: dict | None = None) -> dict:
    reg = registry or load_registry()
    meta = (reg.get("blueprints") or {}).get(blueprint_id)
    if not meta:
        raise KeyError(f"Unknown blueprint_id in 22-registry: {blueprint_id}")
    return meta

def instances_for_blueprint(blueprint_id: str, instances_doc: dict | None = None) -> list[dict]:
    if instances_doc is None:
        path = OUTPUT_DIR / "automation_instance_templates.json"
        if not path.is_file():
            return []
        instances_doc = json.loads(path.read_text(encoding="utf-8"))
    return list((instances_doc.get("instances") or {}).get(blueprint_id) or [])
