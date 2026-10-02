from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import build_all_behavior_specs, write_behavior_artifacts
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR, SCENE_BLUEPRINT_MAP
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

@dataclass
class ExecutableActionPath:
    blueprint_id: str
    scene: str
    branch_id: str
    service: str
    action_path_signature: str
    action_template_key: str
    behavior_target: str
    trigger_id: str | None = None
    conditions: list[str] = field(default_factory=list)
    target_entity: str | None = None
    parameter_keys: list[str] = field(default_factory=list)

    @property
    def path_key(self) -> str:
        return f"{self.branch_id}::{self.action_path_signature}::{self.action_template_key}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _template_key(tmpl: dict) -> str:
    svc = tmpl.get("service") or "valid.no_action"
    target = tmpl.get("target_entity")
    params = sorted((tmpl.get("parameters") or {}).keys())
    return f"{svc}|{target}|{params}"

def enumerate_paths_from_spec(spec: dict) -> list[ExecutableActionPath]:

    bp = spec["blueprint_id"]
    scene = spec["scene"]
    out: list[ExecutableActionPath] = []
    seen: set[str] = set()
    branches = spec.get("branches") or []
    if not branches:
        branches = [{"branch_id": "default", "semantic_action_templates": spec.get("semantic_action_templates") or []}]
    for br in branches:
        bid = str(br.get("branch_id") or "default")
        conditions = list(br.get("conditions") or [])
        trigger_id = br.get("trigger_id")
        templates = br.get("semantic_action_templates") or []
        if not templates:
            templates = [{"service": "valid.no_action", "branch_id": bid}]
        for tmpl in templates:
            svc = str(tmpl.get("service") or "valid.no_action")
            sig = "|".join(sorted({svc}))
            tmpl_key = _template_key(tmpl)
            key = f"{bid}::{sig}::{tmpl_key}"
            if key in seen:
                continue
            seen.add(key)
            out.append(
                ExecutableActionPath(
                    blueprint_id=bp,
                    scene=scene,
                    branch_id=bid,
                    service=svc,
                    action_path_signature=sig,
                    action_template_key=tmpl_key,
                    behavior_target=f"{bp}::{bid}::{sig}",
                    trigger_id=tmpl.get("trigger_id") or trigger_id,
                    conditions=conditions,
                    target_entity=tmpl.get("target_entity"),
                    parameter_keys=sorted((tmpl.get("parameters") or {}).keys()),
                )
            )
    return out

def build_blueprint_action_inventory(
    specs: list[dict] | None = None,
) -> dict[str, Any]:

    specs = specs or build_all_behavior_specs()
    by_bp: dict[str, dict[str, Any]] = {}
    for spec in specs:
        bp = spec["blueprint_id"]
        if bp not in by_bp:
            by_bp[bp] = {
                "blueprint_id": bp,
                "scene": spec["scene"],
                "parse_status": spec.get("parse_status"),
                "yaml_exists": spec.get("yaml_exists"),
                "instance_count": 0,
                "paths": {},
            }
        by_bp[bp]["instance_count"] += 1
        for path in enumerate_paths_from_spec(spec):
            by_bp[bp]["paths"][path.path_key] = path.to_dict()

    blueprints: list[dict[str, Any]] = []
    for bp_id in sorted(by_bp):
        row = by_bp[bp_id]
        paths = list(row["paths"].values())
        services = sorted({p["service"] for p in paths})
        blueprints.append(
            {
                "blueprint_id": bp_id,
                "scene": row["scene"],
                "parse_status": row["parse_status"],
                "yaml_exists": row["yaml_exists"],
                "instance_count": row["instance_count"],
                "executable_action_count": len(paths),
                "unique_services": services,
                "executable_actions": sorted(paths, key=lambda p: (p["branch_id"], p["service"], p["action_template_key"])),
            }
        )

    return {
        "version": "action_path_inventory_v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "behavior_spec.action_paths via build_all_behavior_specs()",
        "total_blueprints": len(blueprints),
        "total_executable_paths": sum(b["executable_action_count"] for b in blueprints),
        "blueprints": blueprints,
    }

def strict_path_key_from_sample(sample: dict) -> str | None:

    meta = sample.get("synthesis_metadata") or {}
    branch = str(meta.get("branch_id") or meta.get("stable_branch_id") or "")
    sig = str(meta.get("action_path_signature") or "")
    if not branch or not sig:
        bt = str(meta.get("behavior_target") or "")
        parts = bt.split("::")
        if len(parts) >= 3:
            branch = branch or parts[1]
            sig = sig or parts[2]
    if not branch or not sig:
        return None
    svc = sig.split("|")[0] if "|" in sig else sig
    target = meta.get("target_entity")
    param_keys: list[str] = []
    actions = sample.get("semantic_actions") or sample.get("expected_actions") or []
    if actions and isinstance(actions[0], dict):
        a0 = actions[0]
        target = target or a0.get("target_entity") or a0.get("entity_id")
        params = a0.get("parameters") or a0.get("data") or {}
        if isinstance(params, dict):
            param_keys = sorted(params.keys())
    tmpl_key = f"{svc}|{target}|{param_keys}"
    return f"{branch}::{sig}::{tmpl_key}"

def _sample_path_key(sample: dict) -> str | None:
    return strict_path_key_from_sample(sample)

def _sample_path_key_loose(sample: dict) -> tuple[str, str] | None:

    meta = sample.get("synthesis_metadata") or {}
    branch = str(meta.get("branch_id") or meta.get("stable_branch_id") or "")
    sig = str(meta.get("action_path_signature") or "")
    if not branch or not sig:
        bt = str(meta.get("behavior_target") or "")
        parts = bt.split("::")
        if len(parts) >= 3:
            branch = branch or parts[1]
            sig = sig or parts[2]
    if not branch or not sig:
        return None
    return branch, sig

def audit_action_path_coverage(
    samples: list[dict],
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:

    inventory = inventory or build_blueprint_action_inventory()
    inv_by_bp = {b["blueprint_id"]: b for b in inventory["blueprints"]}

    counts_by_bp: dict[str, Counter[str]] = defaultdict(Counter)
    loose_counts_by_bp: dict[str, Counter[str]] = defaultdict(Counter)
    sample_totals: Counter[str] = Counter()

    for sample in samples:
        bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if not bp:
            continue
        sample_totals[bp] += 1
        key = _sample_path_key(sample)
        if key:
            counts_by_bp[bp][key] += 1
        loose = _sample_path_key_loose(sample)
        if loose:
            loose_counts_by_bp[bp][f"{loose[0]}::{loose[1]}"] += 1

    blueprint_reports: list[dict[str, Any]] = []
    total_paths = 0
    total_covered = 0
    total_missing = 0

    for bp_id, inv in inv_by_bp.items():
        paths = inv["executable_actions"]
        path_details: list[dict[str, Any]] = []
        covered = 0
        for path in paths:
            total_paths += 1
            strict_key = f"{path['branch_id']}::{path['action_path_signature']}::{path['action_template_key']}"
            loose_key = f"{path['branch_id']}::{path['action_path_signature']}"
            count = counts_by_bp[bp_id].get(strict_key, 0)
            if count == 0:
                count = loose_counts_by_bp[bp_id].get(loose_key, 0)
            if count > 0:
                covered += 1
                total_covered += 1
            else:
                total_missing += 1
            path_details.append(
                {
                    **path,
                    "sample_count": count,
                    "covered": count > 0,
                }
            )

        n_paths = len(paths)
        blueprint_reports.append(
            {
                "blueprint_id": bp_id,
                "scene": inv["scene"],
                "sample_count": sample_totals.get(bp_id, 0),
                "executable_action_count": n_paths,
                "covered_action_count": covered,
                "missing_action_count": n_paths - covered,
                "coverage_ratio": round(covered / n_paths, 4) if n_paths else 1.0,
                "executable_actions": path_details,
                "missing_actions": [p for p in path_details if not p["covered"]],
            }
        )

    return {
        "version": "action_path_coverage_audit_v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_samples": len(samples),
        "blueprints_with_samples": len(sample_totals),
        "total_executable_paths": total_paths,
        "total_covered_paths": total_covered,
        "total_missing_paths": total_missing,
        "overall_coverage_ratio": round(total_covered / total_paths, 4) if total_paths else 1.0,
        "blueprints": blueprint_reports,
    }

def write_action_path_artifacts(
    out_dir: Path | None = None,
    *,
    samples_jsonl: Path | None = None,
) -> dict[str, Path]:
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    validation = out / "validation"
    validation.mkdir(parents=True, exist_ok=True)

    behavior_paths = write_behavior_artifacts(out)
    inventory = build_blueprint_action_inventory()
    inv_path = validation / "executable_action_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    paths: dict[str, Path] = {
        "behavior_specs": behavior_paths["behavior_specs"],
        "action_domains": behavior_paths["action_domains"],
        "executable_action_inventory": inv_path,
    }

    if samples_jsonl and samples_jsonl.is_file():
        samples = read_jsonl(samples_jsonl)
        coverage = audit_action_path_coverage(samples, inventory)
        cov_path = validation / "action_path_coverage_audit.json"
        cov_path.write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
        paths["action_path_coverage_audit"] = cov_path

    return paths

def match_sample_to_path(sample: dict, paths: list[ExecutableActionPath]) -> ExecutableActionPath | None:

    loose = _sample_path_key_loose(sample)
    if not loose:
        return None
    branch, sig = loose
    candidates = [p for p in paths if p.branch_id == branch and p.action_path_signature == sig]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    meta = sample.get("synthesis_metadata") or {}
    bt = str(meta.get("behavior_target") or "")
    for p in candidates:
        if p.behavior_target == bt or bt.startswith(p.behavior_target):
            return p
    return candidates[0]

def compute_balanced_path_quotas(
    paths: list[ExecutableActionPath],
    total_count: int,
) -> dict[str, int]:

    if not paths:
        return {}
    base = total_count // len(paths)
    rem = total_count % len(paths)
    out: dict[str, int] = {}
    for i, path in enumerate(paths):
        out[path.path_key] = base + (1 if i < rem else 0)
    return out

def path_balance_skew(counts: list[int]) -> float:
    if not counts:
        return 0.0
    mn = min(counts)
    if mn <= 0:
        return float("inf") if max(counts) > 0 else 0.0
    return max(counts) / mn

def blueprint_ids_in_project() -> list[str]:
    ids: list[str] = []
    for scene in SCENE_BLUEPRINT_MAP:
        ids.extend(SCENE_BLUEPRINT_MAP[scene])
    return ids
