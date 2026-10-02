from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import (
    build_action_domains,
    build_all_behavior_specs,
    run_parser_audit,
)
from smarthome_mdf.single_scene_blueprint_complete.config import (
    ALL_BLUEPRINT_IDS,
    OUTPUT_DIR,
    PROJECT_SCENES,
    SCENE_BLUEPRINT_MAP,
)
from smarthome_mdf.single_scene_blueprint_complete.information_flow import audit_package_imports
from smarthome_mdf.single_scene_blueprint_complete.inventory import build_full_inventory
from smarthome_mdf.single_scene_blueprint_complete.runtime_requirements import build_all_runtime_requirements
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import build_yaml_mapping_audit

def evaluate_generation_gate(out_dir: Path | None = None) -> dict[str, Any]:

    out = out_dir or OUTPUT_DIR
    inv = build_full_inventory()
    yaml_audit = build_yaml_mapping_audit()
    parser_audit = run_parser_audit()
    specs = build_all_behavior_specs()
    domains = build_action_domains(specs)
    runtime_reqs = build_all_runtime_requirements()
    flow_violations = audit_package_imports()

    yaml_rows = {r["blueprint_id"]: r for r in yaml_audit["rows"]}
    parser_rows = {r["blueprint_id"]: r for r in parser_audit["rows"]}

    blockers: list[str] = []
    if inv["total_scenes"] != 8:
        blockers.append(f"scene count {inv['total_scenes']} != 8")
    if inv["total_blueprint_ids"] != 22:
        blockers.append(f"blueprint count {inv['total_blueprint_ids']} != 22")
    if not yaml_audit.get("complete"):
        blockers.append(f"YAML mapping incomplete: {yaml_audit['yaml_present']}/22")
    for bp_id in ALL_BLUEPRINT_IDS:
        row = yaml_rows.get(bp_id) or {}
        if row.get("mapping_status") == "PROXY_YAML":
            blockers.append(f"proxy YAML forbidden: {bp_id}")
    if parser_audit.get("implementation_gap", 0) > 0:
        blockers.append(f"IMPLEMENTATION_GAP: {parser_audit['implementation_gap']}/22")
    if len(domains.get("blueprints") or []) < 22:
        blockers.append(f"action domains incomplete: {len(domains.get('blueprints') or [])}/22")
    if len({s["blueprint_id"] for s in specs}) < 22:
        blockers.append(f"behavior specs incomplete: {len({s['blueprint_id'] for s in specs})}/22")
    if len(runtime_reqs.get("blueprints") or []) < 22:
        blockers.append(f"runtime requirements incomplete: {len(runtime_reqs.get('blueprints') or [])}/22")
    if flow_violations:
        blockers.append(f"information flow violations: {len(flow_violations)}")

    proxy_pairs = _detect_proxy_yaml(yaml_rows)
    if proxy_pairs:
        blockers.append(f"duplicate YAML across blueprint IDs: {len(proxy_pairs)}")

    verdict = "READY_FOR_GENERATION" if not blockers else "BLOCKED_BEFORE_GENERATION"
    result = {
        "verdict": verdict,
        "blockers": blockers,
        "steps": {
            "A_inventory": inv["total_blueprint_ids"] == 22,
            "B_yaml_mapping": yaml_audit.get("complete"),
            "C_parser_support": parser_audit.get("implementation_gap") == 0,
            "D_behavior_specs": len({s["blueprint_id"] for s in specs}) >= 22,
            "E_action_domains": len(domains.get("blueprints") or []) >= 22,
            "F_runtime_requirements": len(runtime_reqs.get("blueprints") or []) >= 22,
            "G_runtime_matching": verdict == "READY_FOR_GENERATION",
        },
        "yaml_present": yaml_audit["yaml_present"],
        "parse_ok": parser_audit["parse_ok"],
        "implementation_gap": parser_audit["implementation_gap"],
        "post_hoc_binding": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "generation_gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

def _detect_proxy_yaml(yaml_rows: dict[str, dict]) -> list[tuple[str, str]]:
    by_path: dict[str, list[str]] = {}
    for bp_id, row in yaml_rows.items():
        path = row.get("yaml_path")
        if path and row.get("yaml_exists"):
            by_path.setdefault(str(path), []).append(bp_id)
    return [(a, b) for path, ids in by_path.items() if len(ids) > 1 for a in ids for b in ids if a < b]

def write_scene_blueprint_mapping(out_dir: Path | None = None) -> Path:
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "total_scenes": len(PROJECT_SCENES),
        "total_blueprints": len(ALL_BLUEPRINT_IDS),
        "scenes": PROJECT_SCENES,
        "mapping": SCENE_BLUEPRINT_MAP,
    }
    path = out / "scene_blueprint_mapping.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
