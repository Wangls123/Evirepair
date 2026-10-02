from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smarthome_mdf.paths import MDF_ROOT
from smarthome_mdf.single_scene_blueprint_complete.config import (
    ALL_BLUEPRINT_IDS,
    BLUEPRINT_COLLECTION_PATH,
    CONTRACTS_PATH,
    OUTPUT_DIR,
    PROJECT_SCENES,
    SCENE_BLUEPRINT_MAP,
)
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path, yaml_exists

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _contract_for(bp_id: str, contracts: dict) -> dict:
    return (contracts.get("contracts") or {}).get(bp_id) or {}

def build_full_inventory() -> dict[str, Any]:
    collection = _load_json(BLUEPRINT_COLLECTION_PATH) if BLUEPRINT_COLLECTION_PATH.is_file() else {}
    contracts_doc = _load_json(CONTRACTS_PATH) if CONTRACTS_PATH.is_file() else {}
    contracts = contracts_doc.get("contracts") or {}

    entries: list[dict[str, Any]] = []
    for scene in PROJECT_SCENES:
        for bp_id in SCENE_BLUEPRINT_MAP.get(scene, []):
            c = _contract_for(bp_id, contracts_doc)
            ypath = resolve_yaml_path(bp_id)
            entries.append(
                {
                    "scene": scene,
                    "blueprint_id": bp_id,
                    "blueprint_name": c.get("blueprint_name") or bp_id,
                    "source": "blueprint_collection.json + community_blueprint_contracts.json",
                    "yaml_path": str(ypath) if ypath else None,
                    "yaml_exists": yaml_exists(bp_id),
                    "contract_path": str(CONTRACTS_PATH),
                    "category": c.get("category"),
                    "automation_instance_templates": f"data/single_scene_blueprint_complete/automation_instances/{bp_id}.json",
                    "status": "YAML_OK" if yaml_exists(bp_id) else "YAML_MISSING",
                }
            )

    return {
        "version": "22_blueprint_inventory_v1",
        "total_scenes": len(PROJECT_SCENES),
        "total_blueprint_ids": len(entries),
        "project_scenes": PROJECT_SCENES,
        "legacy_single_scene_dir": "data/samples_v3/full",
        "legacy_label": "legacy scene-driven grounded corpus",
        "note": "Authoritative inventory — NOT derived from configs_v3/blueprint_registry.json (7 entries).",
        "blueprints": entries,
        "scene_blueprint_counts": {s: len(SCENE_BLUEPRINT_MAP[s]) for s in PROJECT_SCENES},
    }

def write_inventory(out_dir: Path | None = None) -> Path:
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    inv = build_full_inventory()
    path = out / "full_blueprint_inventory.json"
    path.write_text(json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def blueprint_to_scene(bp_id: str) -> str | None:
    for scene, ids in SCENE_BLUEPRINT_MAP.items():
        if bp_id in ids:
            return scene
    return None
