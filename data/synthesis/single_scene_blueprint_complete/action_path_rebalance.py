from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import (
    ExecutableActionPath,
    compute_balanced_path_quotas,
    enumerate_paths_from_spec,
    match_sample_to_path,
    path_balance_skew,
)
from smarthome_mdf.single_scene_blueprint_complete.action_path_regeneration import (
    DEFAULT_OUT_DIR,
    _allocate_path_quotas,
    _scene_for,
    _sha256,
    _utc_now,
    _write_jsonl,
    merge_regenerated_corpus,
)
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance, build_all_behavior_specs
from smarthome_mdf.single_scene_blueprint_complete.blueprint_evidence_enrichment import repair_sample_evidence_valid
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR, PROJECT_SCENES
from smarthome_mdf.single_scene_blueprint_complete.grounded_instances import enumerate_grounded_instances, instances_to_templates
from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.sample_acceptance_gate import validate_sample_accept
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import canonical_sample_signature
from smarthome_mdf.single_scene_blueprint_complete.synthesizer import synthesize_single_scene
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

def _finalize_sample(sample: dict) -> dict:
    meta = dict(sample.get("synthesis_metadata") or {})
    meta.pop("expected_contract_action", None)
    meta["action_path_scheduling"] = "behavior_spec_action_paths_v1"
    sample["synthesis_metadata"] = meta
    ss = dict(sample.get("system_state") or {})
    ss.pop("expected_contract_action", None)
    sample["system_state"] = ss
    md = dict(sample.get("metadata") or {})
    md["ss_repair"] = "action_path_rebalance_v1"
    md.pop("contract_action_target", None)
    sample["metadata"] = md
    return sample

def _group_samples_by_path(
    samples: list[dict],
    paths: list[ExecutableActionPath],
) -> tuple[dict[str, list[dict]], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    unmatched: list[dict] = []
    for sample in samples:
        matched = match_sample_to_path(sample, paths)
        if matched is None:
            unmatched.append(sample)
        else:
            grouped[matched.path_key].append(sample)
    return grouped, unmatched

def rebalance_blueprint_samples(
    *,
    blueprint_id: str,
    existing_samples: list[dict],
    pool,
    spec: dict,
    instance: dict,
    reg,
    reuse_counts: dict[str, int],
    skew_threshold: float = 2.0,
    allow_partial: bool = True,
    max_attempts_multiplier: int = 5,
    max_attempts_cap: int = 3000,
    progress_interval: int = 100,
    verbose: bool = True,
) -> tuple[list[dict], dict[str, Any]]:

    scene = _scene_for(blueprint_id)
    iid = instance["automation_instance_id"]
    nb = _normalize_from_instance(scene, instance, reg)
    paths = enumerate_paths_from_spec(spec)
    if not paths:
        raise RuntimeError(f"No action paths for {blueprint_id}")

    total = len(existing_samples)
    quotas = compute_balanced_path_quotas(paths, total)
    grouped, unmatched = _group_samples_by_path(existing_samples, paths)
    before_counts = {pk: len(grouped.get(pk, [])) for pk in quotas}
    skew_before = path_balance_skew(list(before_counts.values()))

    if skew_before < skew_threshold:
        report = {
            "blueprint_id": blueprint_id,
            "skipped": True,
            "reason": "ALREADY_BALANCED",
            "skew_before": skew_before,
            "path_counts_before": before_counts,
            "path_quotas": quotas,
        }
        if verbose:
            print(f"  {blueprint_id}: skip (skew={skew_before:.1f} < {skew_threshold})", flush=True)
        return list(existing_samples), report

    kept: list[dict] = []
    discard_pool: list[dict] = list(unmatched)
    deficits: list[ExecutableActionPath] = []

    for path in paths:
        q = quotas[path.path_key]
        bucket = list(grouped.get(path.path_key, []))
        kept.extend(bucket[:q])
        discard_pool.extend(bucket[q:])
        deficit = q - min(len(bucket), q)
        deficits.extend([path] * deficit)

    discard_pool.extend([])

    remaining_deficit: Counter[str] = Counter(p.path_key for p in deficits)
    path_by_key = {p.path_key: p for p in paths}
    new_samples: list[dict] = []
    rejects: Counter[str] = Counter()
    signatures = {canonical_sample_signature(s) for s in kept}
    attempts = 0
    deficit_slots = len(deficits)
    max_attempts = min(
        max(deficit_slots * max_attempts_multiplier, min(400, deficit_slots * 2 + 50)),
        max_attempts_cap,
    )

    if verbose and deficit_slots > 0:
        print(
            f"    {blueprint_id}: synthesizing up to {deficit_slots} path slots "
            f"(max_attempts={max_attempts})",
            flush=True,
        )

    while remaining_deficit and attempts < max_attempts:

        pk = max(remaining_deficit.keys(), key=lambda k: remaining_deficit[k])
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
        if verbose and progress_interval > 0 and attempts % progress_interval == 0:
            filled = deficit_slots - sum(remaining_deficit.values())
            print(
                f"    {blueprint_id}: attempt {attempts}/{max_attempts} "
                f"filled={filled}/{deficit_slots} new={len(new_samples)}",
                flush=True,
            )
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
        new_samples.append(_finalize_sample(sample))
        remaining_deficit[pk] -= 1
        if remaining_deficit[pk] <= 0:
            del remaining_deficit[pk]

    slots_needed = total - len(kept) - len(new_samples)
    if slots_needed > 0:
        kept.extend(discard_pool[:slots_needed])
        discard_pool = discard_pool[slots_needed:]

    merged = kept + new_samples
    if len(merged) != total:
        raise RuntimeError(f"{blueprint_id}: count mismatch {len(merged)} != {total}")

    after_grouped, _ = _group_samples_by_path(merged, paths)
    after_counts = {pk: len(after_grouped.get(pk, [])) for pk in quotas}
    skew_after = path_balance_skew(list(after_counts.values()))

    report = {
        "blueprint_id": blueprint_id,
        "scene": scene,
        "skipped": False,
        "total_count": total,
        "path_count": len(paths),
        "path_quotas": quotas,
        "path_counts_before": before_counts,
        "path_counts_after": after_counts,
        "skew_before": round(skew_before, 2),
        "skew_after": round(skew_after, 2),
        "kept_existing": len(kept) - len(new_samples),
        "newly_synthesized": len(new_samples),
        "unfilled_deficits": dict(remaining_deficit),
        "attempts": attempts,
        "max_attempts": max_attempts,
        "deficit_slots": deficit_slots,
        "attempt_budget_exhausted": bool(remaining_deficit) and attempts >= max_attempts,
        "top_rejects": rejects.most_common(8),
        "complete": not remaining_deficit,
        "partial_ok": allow_partial and bool(remaining_deficit),
    }
    if verbose:
        print(
            f"  {blueprint_id}: skew {skew_before:.1f}->{skew_after:.1f} "
            f"new={len(new_samples)} unfilled={sum(remaining_deficit.values())}",
            flush=True,
        )
    if remaining_deficit and not allow_partial:
        raise RuntimeError(f"Failed to rebalance {blueprint_id}: {report}")
    return merged, report

def run_action_path_rebalance(
    *,
    source_jsonl: Path,
    out_dir: Path | None = None,
    blueprint_ids: set[str] | None = None,
    skew_threshold: float = 2.0,
    allow_partial: bool = True,
    max_attempts_multiplier: int = 5,
    max_attempts_cap: int = 3000,
    progress_interval: int = 100,
    checkpoint_each_blueprint: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    source_samples = read_jsonl(source_jsonl)
    source_sha = _sha256(source_jsonl)

    by_bp: dict[str, list[dict]] = defaultdict(list)
    for sample in source_samples:
        bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if bp:
            by_bp[bp].append(sample)

    if blueprint_ids is None:
        blueprint_ids = set(by_bp.keys())

    reg = build_registry()
    pool = build_legacy_source_pool(PROJECT_SCENES, include_public=True)
    grounded_all = enumerate_grounded_instances(pool, reg=reg, max_probe=300)
    inst_idx = instances_to_templates(grounded_all)
    specs = build_all_behavior_specs(reg, instance_overrides=inst_idx)
    spec_by_bp = {s["blueprint_id"]: s for s in specs}
    reuse_counts: dict[str, int] = {}

    out_jsonl = out_dir / "single_scene_samples.jsonl"
    backup = out_dir / "single_scene_samples.jsonl.pre_path_rebalance.bak"
    validation_dir = out_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    summary_path = validation_dir / "action_path_rebalance_report.json"

    def _write_checkpoint(*, reports: list[dict], rebalanced: dict[str, deque[dict]], final: bool) -> None:
        merged_corpus, merge_stats = merge_regenerated_corpus(source_samples, rebalanced)
        if out_jsonl.is_file() and not backup.is_file():
            backup.write_bytes(out_jsonl.read_bytes())
        _write_jsonl(out_jsonl, merged_corpus)
        out_sha = _sha256(out_jsonl)
        summary = {
            "verdict": "ACTION_PATH_REBALANCE_COMPLETE" if final else "ACTION_PATH_REBALANCE_IN_PROGRESS",
            "rebalanced_at": _utc_now(),
            "scheduling_target": "behavior_spec.action_paths_balanced",
            "skew_threshold": skew_threshold,
            "allow_partial": allow_partial,
            "max_attempts_multiplier": max_attempts_multiplier,
            "max_attempts_cap": max_attempts_cap,
            "source_jsonl": str(source_jsonl),
            "source_sha256": source_sha,
            "output_jsonl": str(out_jsonl),
            "output_sha256": out_sha,
            "total_samples": len(merged_corpus),
            "rebalanced_blueprints": sorted(rebalanced.keys()),
            "reports": reports,
            "merge_stats": merge_stats,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    rebalanced_by_bp: dict[str, deque[dict]] = {}
    reports: list[dict] = []
    if verbose:
        print(f"Rebalancing path quotas (threshold skew>{skew_threshold})...", flush=True)

    for bp_id in sorted(blueprint_ids):
        if bp_id not in by_bp:
            continue
        spec = spec_by_bp.get(bp_id)
        if not spec:
            continue
        inst_list = inst_idx.get(bp_id) or []
        if not inst_list:
            continue
        merged, report = rebalance_blueprint_samples(
            blueprint_id=bp_id,
            existing_samples=by_bp[bp_id],
            pool=pool,
            spec=spec,
            instance=inst_list[0],
            reg=reg,
            reuse_counts=reuse_counts,
            skew_threshold=skew_threshold,
            allow_partial=allow_partial,
            max_attempts_multiplier=max_attempts_multiplier,
            max_attempts_cap=max_attempts_cap,
            progress_interval=progress_interval,
            verbose=verbose,
        )
        if not report.get("skipped"):
            rebalanced_by_bp[bp_id] = deque(merged)
        reports.append(report)
        if checkpoint_each_blueprint:
            _write_checkpoint(reports=reports, rebalanced=rebalanced_by_bp, final=False)

    merged_corpus, merge_stats = merge_regenerated_corpus(source_samples, rebalanced_by_bp)
    if out_jsonl.is_file() and not backup.is_file():
        backup.write_bytes(out_jsonl.read_bytes())

    _write_jsonl(out_jsonl, merged_corpus)
    out_sha = _sha256(out_jsonl)

    summary = {
        "verdict": "ACTION_PATH_REBALANCE_COMPLETE",
        "rebalanced_at": _utc_now(),
        "scheduling_target": "behavior_spec.action_paths_balanced",
        "skew_threshold": skew_threshold,
        "allow_partial": allow_partial,
        "max_attempts_multiplier": max_attempts_multiplier,
        "max_attempts_cap": max_attempts_cap,
        "source_jsonl": str(source_jsonl),
        "source_sha256": source_sha,
        "output_jsonl": str(out_jsonl),
        "output_sha256": out_sha,
        "total_samples": len(merged_corpus),
        "rebalanced_blueprints": sorted(rebalanced_by_bp.keys()),
        "reports": reports,
        "merge_stats": merge_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(json.dumps({"verdict": summary["verdict"], "rebalanced": summary["rebalanced_blueprints"]}, indent=2), flush=True)
    return summary
