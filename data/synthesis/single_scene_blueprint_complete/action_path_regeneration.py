from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import (
    ExecutableActionPath,
    enumerate_paths_from_spec,
)
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance, build_all_behavior_specs
from smarthome_mdf.single_scene_blueprint_complete.blueprint_evidence_enrichment import repair_sample_evidence_valid
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR, PROJECT_SCENES, SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.grounded_instances import enumerate_grounded_instances, instances_to_templates
from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.sample_acceptance_gate import validate_sample_accept
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import canonical_sample_signature
from smarthome_mdf.single_scene_blueprint_complete.synthesizer import synthesize_single_scene
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

CONTRACT_AFFECTED_BLUEPRINTS = frozenset(
    {
        "appliance_notifications_actions",
        "appliance_power_state_detect",
        "climate_production_grade",
        "lighting_motion_advanced_v22",
        "lighting_motion_maestro_48",
    }
)

DEFAULT_OUT_DIR = OUTPUT_DIR / "frozen_v1_ss_repair"

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _scene_for(blueprint_id: str) -> str:
    for scene, ids in SCENE_BLUEPRINT_MAP.items():
        if blueprint_id in ids:
            return scene
    raise KeyError(blueprint_id)

def _allocate_path_quotas(paths: list[ExecutableActionPath], target_count: int) -> list[tuple[ExecutableActionPath, int]]:
    if not paths:
        raise RuntimeError("No executable action paths for blueprint")
    base = target_count // len(paths)
    rem = target_count % len(paths)
    quotas: list[tuple[ExecutableActionPath, int]] = []
    for i, path in enumerate(paths):
        q = base + (1 if i < rem else 0)
        quotas.append((path, q))
    return quotas

def regenerate_blueprint_by_action_paths(
    *,
    blueprint_id: str,
    target_count: int,
    pool,
    spec: dict,
    instance: dict,
    reg,
    reuse_counts: dict[str, int],
    verbose: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    scene = _scene_for(blueprint_id)
    iid = instance["automation_instance_id"]
    nb = _normalize_from_instance(scene, instance, reg)
    paths = enumerate_paths_from_spec(spec)
    if not paths:
        raise RuntimeError(f"No action paths in behavior spec for {blueprint_id}")

    quotas = _allocate_path_quotas(paths, target_count)
    remaining: Counter[str] = Counter(p.path_key for p, q in quotas for _ in range(q))
    path_by_key = {p.path_key: p for p, _ in quotas}

    accepted: list[dict] = []
    rejects: Counter[str] = Counter()
    signatures: set[str] = set()
    per_path_accepted: Counter[str] = Counter()
    attempts = 0
    max_attempts = max(target_count * 50, 5000)

    while len(accepted) < target_count and attempts < max_attempts:
        if not remaining:
            break
        pk = max(remaining.keys(), key=lambda k: remaining[k])
        path = path_by_key[pk]
        status, sample, reason = synthesize_single_scene(
            scene=scene,
            blueprint_id=blueprint_id,
            automation_instance_id=iid,
            behavior_target=path.behavior_target,
            branch_id=path.branch_id,
            action_path_signature=path.action_path_signature,
            instance=instance,
            normalized_blueprint=nb,
            behavior_spec=spec,
            source_pool=pool,
            reuse_counts=reuse_counts,
            attempt_index=attempts,
            visual_frame_index=attempts,
        )
        attempts += 1
        if status != "ACCEPT" or not sample:
            rejects[status or reason or "REJECT"] += 1
            continue

        ok_ev, ev_reason = repair_sample_evidence_valid(sample, blueprint_id)
        if not ok_ev:
            rejects[f"REPAIR_EVIDENCE_FAIL:{ev_reason}"] += 1
            continue

        ok, reject_reason = validate_sample_accept(sample, pool=pool, normalized_blueprint=nb)
        if not ok:
            rejects[reject_reason] += 1
            continue

        sig = canonical_sample_signature(sample)
        if sig in signatures:
            rejects["DUPLICATE_SIGNATURE"] += 1
            continue
        signatures.add(sig)

        meta = dict(sample.get("synthesis_metadata") or {})
        meta.pop("expected_contract_action", None)
        meta["action_path_scheduling"] = "behavior_spec_action_paths_v1"
        sample["synthesis_metadata"] = meta
        ss = dict(sample.get("system_state") or {})
        ss.pop("expected_contract_action", None)
        sample["system_state"] = ss
        md = dict(sample.get("metadata") or {})
        md["ss_repair"] = "action_path_regeneration_v1"
        md.pop("contract_action_target", None)
        sample["metadata"] = md

        accepted.append(sample)
        per_path_accepted[path.path_key] += 1
        remaining[pk] -= 1
        if remaining[pk] <= 0:
            del remaining[pk]

    report = {
        "blueprint_id": blueprint_id,
        "scene": scene,
        "target_count": target_count,
        "executable_path_count": len(paths),
        "path_quotas": {p.path_key: q for p, q in quotas},
        "accepted_count": len(accepted),
        "accepted_by_path": dict(per_path_accepted),
        "attempts": attempts,
        "top_rejects": rejects.most_common(10),
        "complete": len(accepted) >= target_count,
        "executable_paths": [p.to_dict() for p in paths],
    }
    if verbose:
        print(
            f"  {blueprint_id}: {len(accepted)}/{target_count} "
            f"paths={len(paths)} attempts={attempts} rejects={dict(rejects.most_common(3))}",
            flush=True,
        )
    if len(accepted) < target_count:
        raise RuntimeError(f"Failed to regenerate {blueprint_id}: {report}")
    return accepted, report

def merge_regenerated_corpus(
    source_samples: list[dict],
    regenerated_by_bp: dict[str, deque[dict]],
) -> tuple[list[dict], dict[str, int]]:
    merged: list[dict] = []
    replaced = Counter()
    retained = Counter()
    for sample in source_samples:
        bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if bp in regenerated_by_bp and regenerated_by_bp[bp]:
            merged.append(regenerated_by_bp[bp].popleft())
            replaced[bp] += 1
        else:
            merged.append(sample)
            if bp:
                retained[bp] += 1
    for bp, q in regenerated_by_bp.items():
        if q:
            raise RuntimeError(f"Unused regenerated samples for {bp}: {len(q)}")
    return merged, {"replaced": dict(replaced), "retained": dict(retained)}

def run_action_path_regeneration(
    *,
    source_jsonl: Path,
    out_dir: Path | None = None,
    blueprint_ids: set[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    blueprint_ids = blueprint_ids or set(CONTRACT_AFFECTED_BLUEPRINTS)

    if verbose:
        print(f"Loading source corpus: {source_jsonl}", flush=True)
    source_samples = read_jsonl(source_jsonl)
    source_sha = _sha256(source_jsonl)

    need_by_bp: Counter[str] = Counter()
    for sample in source_samples:
        bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if bp in blueprint_ids:
            need_by_bp[bp] += 1
    if not need_by_bp:
        raise RuntimeError(f"No samples found for blueprints: {sorted(blueprint_ids)}")

    reg = build_registry()
    pool = build_legacy_source_pool(PROJECT_SCENES, include_public=True)

    if verbose:
        print("Enumerating grounded instances...", flush=True)
    grounded_all = enumerate_grounded_instances(pool, reg=reg, max_probe=300)
    inst_idx = instances_to_templates(grounded_all)
    specs = build_all_behavior_specs(reg, instance_overrides=inst_idx)
    spec_by_bp = {s["blueprint_id"]: s for s in specs}
    reuse_counts: dict[str, int] = {}

    regenerated_by_bp: dict[str, deque[dict]] = {}
    regen_reports: list[dict] = []
    if verbose:
        print("Regenerating blueprints by action_paths...", flush=True)
    for bp_id in sorted(need_by_bp.keys()):
        target = need_by_bp[bp_id]
        spec = spec_by_bp.get(bp_id)
        if not spec:
            raise RuntimeError(f"No behavior spec for {bp_id}")
        inst_list = inst_idx.get(bp_id) or []
        if not inst_list:
            raise RuntimeError(f"No grounded instances for {bp_id}")
        samples, report = regenerate_blueprint_by_action_paths(
            blueprint_id=bp_id,
            target_count=target,
            pool=pool,
            spec=spec,
            instance=inst_list[0],
            reg=reg,
            reuse_counts=reuse_counts,
            verbose=verbose,
        )
        regenerated_by_bp[bp_id] = deque(samples)
        regen_reports.append(report)

    merged, merge_stats = merge_regenerated_corpus(source_samples, regenerated_by_bp)
    out_jsonl = out_dir / "single_scene_samples.jsonl"
    _write_jsonl(out_jsonl, merged)
    out_sha = _sha256(out_jsonl)

    validation_dir = out_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": "ACTION_PATH_REGEN_COMPLETE",
        "regenerated_at": _utc_now(),
        "scheduling_target": "behavior_spec.action_paths",
        "source_jsonl": str(source_jsonl),
        "source_sha256": source_sha,
        "output_jsonl": str(out_jsonl),
        "output_sha256": out_sha,
        "total_samples": len(merged),
        "regenerated_blueprints": dict(need_by_bp),
        "regeneration_reports": regen_reports,
        "merge_stats": merge_stats,
    }
    summary_path = validation_dir / "action_path_regeneration_report.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(json.dumps({"verdict": summary["verdict"], "output_sha256": out_sha, "total": len(merged)}, indent=2), flush=True)
    return summary
