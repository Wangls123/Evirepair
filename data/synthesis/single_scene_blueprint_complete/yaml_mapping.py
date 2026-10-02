from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.config import ALL_BLUEPRINT_IDS, OUTPUT_DIR
from smarthome_mdf.single_scene_blueprint_complete.yaml_acquisition import (
    build_one_to_one_mapping,
    resolve_canonical_path,
    yaml_hash,
)

def resolve_yaml_path(blueprint_id: str) -> Path | None:
    return resolve_canonical_path(blueprint_id)

def yaml_exists(blueprint_id: str) -> bool:
    p = resolve_yaml_path(blueprint_id)
    return bool(p and p.is_file())

def build_yaml_mapping_audit() -> dict[str, Any]:
    return build_one_to_one_mapping()

def write_yaml_mapping(out_dir: Path | None = None) -> Path:
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    audit = build_yaml_mapping_audit()
    path = out / "blueprint_yaml_mapping.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
