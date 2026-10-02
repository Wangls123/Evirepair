from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import GROUNDED_CAPABLE_BLUEPRINTS
from smarthome_mdf.single_scene_blueprint_complete.final_scheduler import FinalGenerationScheduler
from smarthome_mdf.single_scene_blueprint_complete.grounded_instances import enumerate_grounded_instances, instances_to_templates
from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.config import PROJECT_SCENES
from smarthome_mdf.single_scene_blueprint_complete.p1_integrity_audit import audit_visual_fusion, classify_sample_validity
from smarthome_mdf.single_scene_blueprint_complete.pre_freeze_verification import (
    _RematchContext,
    classify_temporal_near_dup,
    rematch_sample,
)
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import (
    behavior_source_dedup_key,
    canonical_behavior_hash,
    canonical_sample_signature,
    is_timestamp_only_duplicate,
)

def audit_runtime_match_all(samples: list[dict], pool: Any | None = None, *, instance_templates: dict | None = None) -> dict[str, Any]:
    pool = pool or build_legacy_source_pool(PROJECT_SCENES, include_public=True)
    ctx = _RematchContext(reg=build_registry(), pool=pool, instance_templates=instance_templates)
    rows: list[dict] = []
    fail = 0
    for s in samples:
        r = rematch_sample(s, ctx)
        if r.get("status") != "MATCH_OK":
            fail += 1
        rows.append({"sample_id": s.get("sample_id"), "status": r.get("status"), "reason": r.get("reason")})
    return {
        "total": len(samples),
        "MATCH_OK": len(samples) - fail,
        "MATCH_FAIL": fail,
        "POST_GENERATION_RUNTIME_MATCH_FAIL": fail,
        "all_match_ok": fail == 0,
        "failures": [r for r in rows if r["status"] != "MATCH_OK"][:50],
    }

def audit_instance_coverage(
    samples: list[dict],
    *,
    grounded_templates: dict[str, list[dict]] | None = None,
    pool: Any | None = None,
) -> dict[str, Any]:
    pool = pool or build_legacy_source_pool(PROJECT_SCENES, include_public=True)
    if grounded_templates is None:
        grounded = enumerate_grounded_instances(pool, reg=build_registry())
        grounded_templates = instances_to_templates(grounded)

    rows: list[dict] = []
    gap_count = 0
    for bp_id in GROUNDED_CAPABLE_BLUEPRINTS:
        reachable = {i["automation_instance_id"] for i in (grounded_templates.get(bp_id) or [])}
        observed = {
            (s.get("blueprint_binding") or {}).get("automation_instance_id")
            for s in samples
            if (s.get("blueprint_binding") or {}).get("blueprint_id") == bp_id
        }
        observed.discard(None)
        missing = sorted(reachable - observed)
        covered = len(observed & reachable)
        total = len(reachable) or max(len(observed), 1)
        gap = bool(missing and len(reachable) > 1)
        if gap:
            gap_count += 1
        rows.append(
            {
                "blueprint_id": bp_id,
                "reachable_instance_count": len(reachable),
                "observed_instance_count": len(observed),
                "instance_coverage": f"{covered}/{total}",
                "missing_instance_ids": missing,
                "GENERATOR_INSTANCE_COVERAGE_GAP": gap,
            }
        )
    return {
        "blueprints": rows,
        "GENERATOR_INSTANCE_COVERAGE_GAP": gap_count,
        "all_instances_covered": gap_count == 0,
    }

def audit_semantic_integrity(samples: list[dict]) -> dict[str, Any]:
    vf_audit = audit_visual_fusion([s for s in samples if s.get("scene_type") == "visual_fusion"])
    invalid = 0
    hits: list[dict] = []
    for s in samples:
        validity, reason = classify_sample_validity(s, vf_audit=vf_audit)
        if validity != "VALID":
            invalid += 1
            hits.append({"sample_id": s.get("sample_id"), "validity": validity, "reason": reason})
    return {"SEMANTIC_INVALID": invalid, "all_valid": invalid == 0, "invalid_samples": hits[:50]}

def audit_visual_provenance(samples: list[dict]) -> dict[str, Any]:
    vf = [s for s in samples if s.get("scene_type") == "visual_fusion"]
    audit = audit_visual_fusion(vf)
    invalid = audit.get("provenance_invalid_count", 0)
    return {
        "PROVENANCE_INVALID": invalid,
        "all_valid": invalid == 0,
        **audit,
    }

def audit_duplicates(samples: list[dict], *, generation_rejects: dict[str, int] | None = None) -> dict[str, Any]:
    exact_canonical: Counter = Counter()
    behavior_source: dict[str, dict] = {}
    ts_only = 0
    same_source_exact = 0
    for s in samples:
        exact_canonical[canonical_sample_signature(s)] += 1
        key = behavior_source_dedup_key(s)
        prior = behavior_source.get(key)
        if prior is not None:
            if is_timestamp_only_duplicate(s, prior):
                ts_only += 1
            if canonical_sample_signature(s) == canonical_sample_signature(prior):
                same_source_exact += 1
        behavior_source[key] = s

    temporal = classify_temporal_near_dup(samples)
    exact_dup_pairs = sum(c - 1 for c in exact_canonical.values() if c > 1)
    return {
        "exact_canonical_duplicate_count": exact_dup_pairs,
        "timestamp_only_duplicate_count": ts_only,
        "same_source_exact_duplicate_count": same_source_exact,
        "generation_time_timestamp_only_rejects": (generation_rejects or {}).get("TIMESTAMP_ONLY_DUPLICATE", 0),
        "temporal_near_dup_audit": temporal,
        "canonical_behavior_unique_count": len({canonical_behavior_hash(s) for s in samples}),
    }

def write_v3_audit_artifacts(
    out_dir: Path,
    samples: list[dict],
    scheduler: FinalGenerationScheduler,
    *,
    grounded_templates: dict[str, list[dict]] | None = None,
    rejects: dict[str, int] | None = None,
    pool: Any | None = None,
) -> dict[str, Any]:
    pool = pool or build_legacy_source_pool(PROJECT_SCENES, include_public=True)
    runtime = audit_runtime_match_all(samples, pool, instance_templates=grounded_templates)
    instance = audit_instance_coverage(samples, grounded_templates=grounded_templates, pool=pool)
    semantic = audit_semantic_integrity(samples)
    visual = audit_visual_provenance(samples)
    duplicate = audit_duplicates(samples, generation_rejects=rejects)
    source_dist = Counter((s.get("provenance") or {}).get("source_dataset") for s in samples)

    artifacts = {
        "runtime_match_audit.json": runtime,
        "instance_coverage.json": instance,
        "semantic_integrity_audit.json": semantic,
        "visual_provenance_audit.json": visual,
        "duplicate_audit.json": duplicate,
        "source_diversity_audit.json": {
            "source_dataset_distribution": dict(source_dist),
            "unique_source_records": len({(s.get("provenance") or {}).get("source_record_id") for s in samples}),
            "scheduler_instance_gaps": scheduler.instance_coverage_gaps(),
        },
        "blueprint_inventory.json": {
            "inventory_blueprints": 22,
            "grounded_capable": 21,
            "true_evidence_limited": 1,
        },
    }
    for name, obj in artifacts.items():
        (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifacts
