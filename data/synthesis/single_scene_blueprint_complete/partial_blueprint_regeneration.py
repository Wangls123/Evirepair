from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance, build_all_behavior_specs
from smarthome_mdf.single_scene_blueprint_complete.blueprint_evidence_enrichment import repair_sample_evidence_valid
from smarthome_mdf.single_scene_blueprint_complete.blueprint_instance_defaults import REPAIR_BLUEPRINT_IDS
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR, PROJECT_SCENES, SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.grounded_instances import enumerate_grounded_instances, instances_to_templates
from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.sample_acceptance_gate import validate_sample_accept
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import canonical_sample_signature
from smarthome_mdf.single_scene_blueprint_complete.synthesizer import synthesize_single_scene
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

REPAIR_TARGET_COUNTS: dict[str, int] = {
    "appliance_power_google_sheets": 700,
    "presence_holiday_away_lighting": 76,
    "camera_frigate_vision_llm": 224,
    "camera_frigate_intelligent": 160,
}

DEFAULT_SOURCE_JSONL = OUTPUT_DIR / "frozen_v1" / "single_scene_samples.jsonl"
DEFAULT_OUT_DIR = OUTPUT_DIR / "frozen_v1_ss_repair"

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _scene_for(blueprint_id: str) -> str:
    for scene, ids in SCENE_BLUEPRINT_MAP.items():
        if blueprint_id in ids:
            return scene
    raise KeyError(blueprint_id)

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _behavior_targets(spec: dict) -> list[tuple[str, str, str]]:

    out: list[tuple[str, str, str]] = []
    bp = spec["blueprint_id"]
    templates = spec.get("semantic_action_templates") or []
    branches = spec.get("branches") or []
    if templates:
        for tmpl in templates:
            branch = str(tmpl.get("branch_id") or "default")
            action_sig = str(tmpl.get("service") or tmpl.get("action_path_signature") or "default")
            out.append((branch, action_sig, f"{bp}::{branch}::{action_sig}"))
    elif branches:
        for br in branches:
            branch = str(br.get("branch_id") or "default")
            action_sig = str(br.get("action_path_signature") or "default")
            out.append((branch, action_sig, f"{bp}::{branch}::{action_sig}"))
    else:
        out.append(("default", "default", f"{bp}::default::default"))
    return out

def regenerate_blueprint_samples(
    *,
    blueprint_id: str,
    target_count: int,
    pool,
    specs: list[dict],
    inst_idx: dict[str, list[dict]],
    reg,
    reuse_counts: dict[str, int],
    verbose: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    scene = _scene_for(blueprint_id)
    spec = next((s for s in specs if s["blueprint_id"] == blueprint_id), None)
    if spec is None:
        raise RuntimeError(f"No behavior spec for {blueprint_id}")

    inst_list = inst_idx.get(blueprint_id) or []
    if not inst_list:
        raise RuntimeError(f"No grounded instances for {blueprint_id}")
    instance = inst_list[0]
    iid = instance["automation_instance_id"]
    nb_cache_key = f"{blueprint_id}::{iid}"
    nb = _normalize_from_instance(scene, instance, reg)

    targets = _behavior_targets(spec)
    accepted: list[dict] = []
    rejects: Counter[str] = Counter()
    signatures: set[str] = set()
    attempts = 0
    max_attempts = max(target_count * 40, 4000)

    while len(accepted) < target_count and attempts < max_attempts:
        branch_id, action_sig, bt = targets[attempts % len(targets)]
        status, sample, reason = synthesize_single_scene(
            scene=scene,
            blueprint_id=blueprint_id,
            automation_instance_id=iid,
            behavior_target=bt,
            branch_id=branch_id,
            action_path_signature=action_sig,
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
        md = dict(sample.get("metadata") or {})
        md["ss_repair"] = "partial_blueprint_regeneration_v1"
        sample["metadata"] = md
        accepted.append(sample)

    report = {
        "blueprint_id": blueprint_id,
        "scene": scene,
        "target_count": target_count,
        "accepted_count": len(accepted),
        "attempts": attempts,
        "top_rejects": rejects.most_common(8),
        "complete": len(accepted) >= target_count,
    }
    if verbose:
        print(
            f"  {blueprint_id}: {len(accepted)}/{target_count} "
            f"(attempts={attempts}, rejects={dict(rejects.most_common(3))})",
            flush=True,
        )
    if len(accepted) < target_count:
        raise RuntimeError(f"Failed to regenerate {blueprint_id}: {report}")
    return accepted, report

def merge_repaired_corpus(
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
        elif bp in REPAIR_BLUEPRINT_IDS:
            raise RuntimeError(f"Missing replacement for repair blueprint slot: {bp}")
        else:
            merged.append(sample)
            if bp:
                retained[bp] += 1
    for bp, q in regenerated_by_bp.items():
        if q:
            raise RuntimeError(f"Unused regenerated samples for {bp}: {len(q)}")
    return merged, {"replaced": dict(replaced), "retained": dict(retained)}

def run_partial_blueprint_regeneration(
    *,
    source_jsonl: Path | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    source_jsonl = source_jsonl or DEFAULT_SOURCE_JSONL
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading source corpus: {source_jsonl}", flush=True)
    source_samples = read_jsonl(source_jsonl)
    source_sha = _sha256(source_jsonl)

    need_by_bp: Counter[str] = Counter()
    for sample in source_samples:
        bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if bp in REPAIR_TARGET_COUNTS:
            need_by_bp[bp] += 1
    for bp, target in REPAIR_TARGET_COUNTS.items():
        if need_by_bp[bp] != target:
            raise RuntimeError(f"Source count mismatch for {bp}: have {need_by_bp[bp]}, expected {target}")

    reg = build_registry()
    pool = build_legacy_source_pool(PROJECT_SCENES, include_public=True)

    if verbose:
        print("Re-enumerating grounded instances for repair blueprints...", flush=True)
    grounded_all = enumerate_grounded_instances(pool, reg=reg, max_probe=300)
    inst_idx = instances_to_templates(grounded_all)

    specs = build_all_behavior_specs(reg, instance_overrides=inst_idx)
    reuse_counts: dict[str, int] = {}

    regenerated_by_bp: dict[str, deque[dict]] = {}
    regen_reports: list[dict] = []
    if verbose:
        print("Regenerating repaired blueprints...", flush=True)
    for bp_id in sorted(REPAIR_TARGET_COUNTS.keys()):
        samples, report = regenerate_blueprint_samples(
            blueprint_id=bp_id,
            target_count=REPAIR_TARGET_COUNTS[bp_id],
            pool=pool,
            specs=specs,
            inst_idx=inst_idx,
            reg=reg,
            reuse_counts=reuse_counts,
            verbose=verbose,
        )
        regenerated_by_bp[bp_id] = deque(samples)
        regen_reports.append(report)

    merged, merge_stats = merge_repaired_corpus(source_samples, regenerated_by_bp)
    out_jsonl = out_dir / "single_scene_samples.jsonl"
    _write_jsonl(out_jsonl, merged)
    out_sha = _sha256(out_jsonl)

    src_grounded = OUTPUT_DIR / "frozen_v1" / "grounded" / "grounded_instance_candidates.json"
    grounded_doc = json.loads(src_grounded.read_text(encoding="utf-8"))
    for bp_id in REPAIR_BLUEPRINT_IDS:
        cands = grounded_all.get(bp_id) or []
        grounded_doc[bp_id] = [
            {
                "automation_instance_id": c.automation_instance_id,
                "blueprint_id": c.blueprint_id,
                "scene": c.scene,
                "input_bindings": c.input_bindings,
                "entity_bindings": c.entity_bindings,
                "parameter_bindings": c.parameter_bindings,
                "configuration_signature": c.configuration_signature,
                "display_name": c.display_name,
                "matcher_viable": c.matcher_viable,
                "supporting_source_count": c.supporting_source_count,
            }
            for c in cands
        ]
    grounded_dir = out_dir / "grounded"
    grounded_dir.mkdir(parents=True, exist_ok=True)
    grounded_path = grounded_dir / "grounded_instance_candidates.json"
    grounded_path.write_text(json.dumps(grounded_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    validation_dir = out_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": "SS_REPAIR_COMPLETE",
        "repaired_at": _utc_now(),
        "source_jsonl": str(source_jsonl),
        "source_sha256": source_sha,
        "output_jsonl": str(out_jsonl),
        "output_sha256": out_sha,
        "total_samples": len(merged),
        "repair_blueprints": dict(REPAIR_TARGET_COUNTS),
        "regeneration_reports": regen_reports,
        "merge_stats": merge_stats,
        "unchanged_sample_count": len(merged) - sum(REPAIR_TARGET_COUNTS.values()),
    }
    summary_path = validation_dir / "partial_blueprint_regeneration_report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(json.dumps({"verdict": summary["verdict"], "output_sha256": out_sha, "total": len(merged)}, indent=2), flush=True)
    return summary
