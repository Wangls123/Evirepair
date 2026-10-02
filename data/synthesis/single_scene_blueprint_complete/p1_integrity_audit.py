from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.config import SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (
    GROUNDED_CAPABLE_BLUEPRINTS,
    TRUE_EVIDENCE_LIMITED_BLUEPRINTS,
)
from smarthome_mdf.single_scene_blueprint_complete.semantic_sanitize import (
    ALLOWED_ATTRS,
    contamination_invalidates_match,
    detect_semantic_contamination,
    infer_primary_semantic,
)
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import (
    canonical_behavior_hash,
    canonical_behavior_signature,
    is_dynamic_behavior_target,
    numeric_only_near_duplicate_stable,
    same_source_observation_duplicate,
    strip_dynamic_behavior_suffix,
    temporal_shift_near_duplicate_stable,
)
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

V1_DIR_NAME = "final_regeneration"
V2_DIR_NAME = "final_regeneration_v2"
VERDICT_BLOCKED = "BLOCKED_BEFORE_SINGLE_SCENE_FREEZE"

def _action_share(samples: list[dict], bp_id: str) -> dict[str, Any]:
    subset = [s for s in samples if (s.get("blueprint_binding") or {}).get("blueprint_id") == bp_id]
    if not subset:
        return {"blueprint_id": bp_id, "sample_count": 0, "actions": {}}
    acts = Counter((s.get("synthesis_metadata") or {}).get("action_path_signature") for s in subset)
    total = len(subset)
    return {
        "blueprint_id": bp_id,
        "sample_count": total,
        "actions": {
            k: {"count": v, "share": round(v / total, 4)} for k, v in acts.most_common()
        },
        "dominant_action": acts.most_common(1)[0][0] if acts else None,
        "dominant_share": round(acts.most_common(1)[0][1] / total, 4) if acts else 0,
    }

def audit_stable_behavior_signatures(samples: list[dict]) -> dict[str, Any]:
    raw_bt = Counter((s.get("synthesis_metadata") or {}).get("behavior_target") for s in samples)
    stable_bt = Counter(strip_dynamic_behavior_suffix((s.get("synthesis_metadata") or {}).get("behavior_target")) for s in samples)
    canon = Counter(canonical_behavior_hash(s) for s in samples)
    dynamic_count = sum(1 for s in samples if is_dynamic_behavior_target((s.get("synthesis_metadata") or {}).get("behavior_target")))
    exact_dup_groups = sum(1 for c in canon.values() if c > 1)
    exact_dup_samples = sum(c - 1 for c in canon.values() if c > 1)
    return {
        "original_unique_behavior_target_count": len(raw_bt),
        "stable_canonical_behavior_count": len(canon),
        "unique_stable_behavior_target_count": len(stable_bt),
        "dynamic_a_suffix_sample_count": dynamic_count,
        "dynamic_a_suffix_is_generation_time_id": dynamic_count == len(samples),
        "exact_canonical_duplicate_groups": exact_dup_groups,
        "exact_canonical_duplicate_samples": exact_dup_samples,
        "exact_canonical_duplicate_rate": round(exact_dup_samples / max(len(samples), 1), 4),
        "previous_duplicate_audit_invalid": True,
        "previous_duplicate_audit_reason": "Prior audit used diversity_signature including ::aXXXX attempt suffix; reported 0.0 duplicate rate is invalid",
    }

def audit_duplicates_reaudit(samples: list[dict]) -> dict[str, Any]:
    by_sig: dict[str, dict] = {}
    numeric = temporal = same_src = 0
    for s in samples:
        sig = canonical_behavior_hash(s)
        prior = by_sig.get(sig)
        if prior:
            if same_source_observation_duplicate(s, prior):
                same_src += 1
            elif temporal_shift_near_duplicate_stable(s, prior):
                temporal += 1
            elif numeric_only_near_duplicate_stable(s, prior):
                numeric += 1
        by_sig[sig] = s
    n = len(samples)
    return {
        "numeric_only_near_duplicate_count": numeric,
        "numeric_only_near_duplicate_rate": round(numeric / max(n, 1), 4),
        "temporal_shift_near_duplicate_count": temporal,
        "temporal_shift_near_duplicate_rate": round(temporal / max(n, 1), 4),
        "same_source_observation_duplicate_count": same_src,
        "same_source_observation_duplicate_rate": round(same_src / max(n, 1), 4),
        "previous_duplicate_audit_invalid": True,
    }

def audit_visual_fusion(samples: list[dict]) -> dict[str, Any]:
    vf = [s for s in samples if s.get("scene_type") == "visual_fusion"]
    frames = Counter((s.get("observed") or {}).get("frame_id") for s in vf)
    participants = Counter((s.get("observed") or {}).get("participant_id") for s in vf)
    sequences = Counter((s.get("observed") or {}).get("sequence_id") for s in vf)
    runtime_ds = Counter((s.get("provenance") or {}).get("source_dataset") for s in vf)
    visual_ds = Counter(
        ((s.get("provenance") or {}).get("visual_source") or {}).get("dataset")
        or ((s.get("observed") or {}).get("frame_id") and "YouHome")
        for s in vf
    )
    cross = []
    for s in vf:
        prov = s.get("provenance") or {}
        runtime = prov.get("source_dataset") or prov.get("runtime_source", {}).get("dataset")
        obs = s.get("observed") or {}
        has_youhome_frame = bool(obs.get("frame_id"))
        if runtime == "CASAS" and has_youhome_frame and not prov.get("composition_type"):
            cross.append(s.get("sample_id"))
    largest_frame = frames.most_common(1)[0] if frames else (None, 0)
    return {
        "total_visual_fusion_samples": len(vf),
        "unique_image_frames": len(frames),
        "unique_participants": len(participants),
        "unique_sequences": len(sequences),
        "frame_reuse_distribution_top10": frames.most_common(10),
        "largest_frame_reuse_count": largest_frame[1],
        "largest_frame_id": largest_frame[0],
        "single_frame_domination": largest_frame[1] > len(vf) * 0.5 if vf else False,
        "runtime_source_dataset_counts": dict(runtime_ds),
        "visual_source_dataset_counts": dict(visual_ds),
        "cross_dataset_without_explicit_composition_count": len(cross),
        "cross_dataset_sample_ids_sample": cross[:20],
        "invalid_for_freeze_count": len(vf) if largest_frame[1] == len(vf) and len(vf) > 1 else len(cross),
    }

def audit_visual_provenance(samples: list[dict]) -> dict[str, Any]:
    vf = [s for s in samples if s.get("scene_type") == "visual_fusion"]
    issues = []
    for s in vf:
        prov = s.get("provenance") or {}
        obs = s.get("observed") or {}
        runtime_ds = prov.get("source_dataset")
        has_frame = bool(obs.get("frame_id"))
        explicit = prov.get("composition_type")
        if has_frame and runtime_ds == "CASAS" and explicit != "CROSS_DATASET_SYNTHETIC":
            issues.append(
                {
                    "sample_id": s.get("sample_id"),
                    "issue": "CASAS runtime with YouHome frame without explicit cross-dataset composition",
                    "runtime_dataset": runtime_ds,
                    "frame_id": obs.get("frame_id"),
                }
            )
    return {
        "samples_audited": len(vf),
        "provenance_issue_count": len(issues),
        "issues_sample": issues[:50],
        "all_have_runtime_and_visual_separated": sum(1 for s in vf if (s.get("provenance") or {}).get("runtime_source")),
    }

def audit_raw_to_adapter_semantic(samples: list[dict]) -> dict[str, Any]:
    by_type: dict[str, Counter] = defaultdict(Counter)
    violations: list[dict] = []
    for s in samples:
        for eo in s.get("entity_observations") or []:
            attrs = eo.get("attributes") or {}
            primary = infer_primary_semantic(attrs, str(attrs.get("device_class") or ""))
            for k in attrs:
                if k == "trigger_context":
                    continue
                allowed = ALLOWED_ATTRS.get(primary, frozenset(attrs.keys()))
                if k not in allowed:
                    by_type[primary][k] += 1
                    if len(violations) < 100:
                        violations.append(
                            {
                                "sample_id": s.get("sample_id"),
                                "primary": primary,
                                "field": k,
                                "value": attrs[k],
                            }
                        )
    return {
        "semantic_multiplexing_by_primary_type": {k: dict(v) for k, v in by_type.items()},
        "violation_samples": violations,
        "motion_with_power_count": by_type["motion"].get("current_power_w", 0),
        "motion_with_temperature_count": by_type["motion"].get("temperature", 0) + by_type["motion"].get("indoor_temperature", 0),
    }

def audit_semantic_contamination_impact(samples: list[dict]) -> dict[str, Any]:
    per_bp: dict[str, list] = defaultdict(list)
    for s in samples:
        bp = (s.get("blueprint_binding") or {}).get("blueprint_id")
        hits = detect_semantic_contamination(s)
        if not hits:
            per_bp[bp].append({"sample_id": s.get("sample_id"), "impact_status": "UNAFFECTED"})
        elif contamination_invalidates_match(hits):
            per_bp[bp].append({"sample_id": s.get("sample_id"), "impact_status": "INVALID_MATCH", "hits": hits})
        else:
            per_bp[bp].append({"sample_id": s.get("sample_id"), "impact_status": "AFFECTED_BUT_RECOVERABLE", "hits": hits})
    summary = {}
    for bp, rows in per_bp.items():
        status_counts = Counter(r["impact_status"] for r in rows)
        summary[bp] = dict(status_counts)
    return {"per_blueprint_impact_summary": summary, "details_sample": {k: v[:5] for k, v in per_bp.items()}}

def audit_automation_instance_diversity(samples: list[dict]) -> dict[str, Any]:
    per_bp: dict[str, Any] = {}
    for bp in GROUNDED_CAPABLE_BLUEPRINTS:
        subset = [s for s in samples if (s.get("blueprint_binding") or {}).get("blueprint_id") == bp]
        insts = Counter((s.get("blueprint_binding") or {}).get("automation_instance_id") for s in subset)
        per_bp[bp] = {
            "blueprint_id": bp,
            "sample_count": len(subset),
            "unique_automation_instance_ids": len(insts),
            "instance_ids": dict(insts),
            "grounded_instance_capacity_note": "Derived from instance_factory; only inst_default unless alt instances grounded",
        }
    return {"per_blueprint": per_bp, "total_unique_instances": len(set((s.get("blueprint_binding") or {}).get("automation_instance_id") for s in samples))}

def audit_entity_binding_diversity(samples: list[dict]) -> dict[str, Any]:
    per_bp: dict[str, Any] = {}
    for bp in GROUNDED_CAPABLE_BLUEPRINTS:
        subset = [s for s in samples if (s.get("blueprint_binding") or {}).get("blueprint_id") == bp]
        bindings = Counter((s.get("synthesis_metadata") or {}).get("entity_binding_signature") for s in subset)
        per_bp[bp] = {
            "blueprint_id": bp,
            "observed_binding_count": len(bindings),
            "bindings": dict(bindings),
            "grounded_binding_capacity": len(bindings),
        }
    return {"per_blueprint": per_bp}

def audit_canonical_ir_identity(samples: list[dict]) -> dict[str, Any]:
    raw_branches = Counter((s.get("synthesis_metadata") or {}).get("branch_id") for s in samples)
    suspicious = [b for b in raw_branches if b and (str(b).startswith("{") or str(b) in ("and", "or", "template"))]
    return {
        "raw_branch_id_distribution_top20": raw_branches.most_common(20),
        "suspicious_raw_branch_ids": suspicious[:30],
        "recommendation": "Use stable_branch_id / ir_branch_path from Canonical IR; raw expressions stored separately",
    }

def audit_action_distribution(samples: list[dict]) -> dict[str, Any]:
    per_bp = [_action_share(samples, bp) for bp in GROUNDED_CAPABLE_BLUEPRINTS]
    dominant = [p for p in per_bp if p.get("dominant_share", 0) > 0.8 and p.get("sample_count", 0) > 10]
    return {
        "per_blueprint": per_bp,
        "dominant_action_over_80pct": dominant,
    }

def classify_sample_validity(sample: dict, *, vf_audit: dict) -> tuple[str, str]:
    reasons: list[str] = []
    sid = sample.get("sample_id")
    scene = sample.get("scene_type")
    if scene == "visual_fusion":
        obs = sample.get("observed") or {}
        prov = sample.get("provenance") or {}
        frame = obs.get("frame_id")
        if vf_audit.get("single_frame_domination"):
            reasons.append("single_frame_domination")
        if prov.get("source_dataset") == "CASAS" and frame and prov.get("composition_type") != "CROSS_DATASET_SYNTHETIC":
            reasons.append("cross_dataset_provenance_implicit")
        if reasons:
            return "INVALID_VISUAL_PROVENANCE", ";".join(reasons)
    hits = detect_semantic_contamination(sample)
    if hits and contamination_invalidates_match(hits):
        return "INVALID_SEMANTIC_MAPPING", "motion_multiplexing"
    if hits:
        return "VALID_SAMPLE", "INVALID_PREVIOUS_DIVERSITY_METRIC;semantic_recoverable"
    if is_dynamic_behavior_target((sample.get("synthesis_metadata") or {}).get("behavior_target")):
        return "VALID_SAMPLE", "INVALID_PREVIOUS_DIVERSITY_METRIC;dynamic_behavior_target_suffix"
    return "VALID", "ok"

def run_full_integrity_audit(samples: list[dict]) -> dict[str, Any]:
    vf_audit = audit_visual_fusion(samples)
    validity_rows = []
    invalid_ids = []
    for s in samples:
        status, reason = classify_sample_validity(s, vf_audit=vf_audit)
        validity_rows.append({"sample_id": s.get("sample_id"), "validity": status, "reason": reason})
        if status.startswith("INVALID"):
            invalid_ids.append(s.get("sample_id"))

    return {
        "verdict": VERDICT_BLOCKED,
        "total_samples": len(samples),
        "stable_behavior_signature_audit": audit_stable_behavior_signatures(samples),
        "duplicate_reaudit": audit_duplicates_reaudit(samples),
        "visual_fusion_grounding_audit": vf_audit,
        "visual_provenance_audit": audit_visual_provenance(samples),
        "raw_to_adapter_semantic_audit": audit_raw_to_adapter_semantic(samples),
        "semantic_contamination_impact": audit_semantic_contamination_impact(samples),
        "automation_instance_diversity_audit": audit_automation_instance_diversity(samples),
        "entity_binding_diversity_audit": audit_entity_binding_diversity(samples),
        "canonical_ir_identity_audit": audit_canonical_ir_identity(samples),
        "action_distribution_audit": audit_action_distribution(samples),
        "sample_validity_rows": validity_rows,
        "invalid_sample_ids": invalid_ids,
        "invalid_count": len(invalid_ids),
        "valid_count": len(samples) - len(invalid_ids),
    }

def write_audit_artifacts(out_dir: Path, audit: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "stable_behavior_signature_audit.json": audit["stable_behavior_signature_audit"],
        "duplicate_reaudit.json": audit["duplicate_reaudit"],
        "visual_fusion_grounding_audit.json": audit["visual_fusion_grounding_audit"],
        "visual_provenance_audit.json": audit["visual_provenance_audit"],
        "raw_to_adapter_semantic_audit.json": audit["raw_to_adapter_semantic_audit"],
        "semantic_contamination_impact.json": audit["semantic_contamination_impact"],
        "automation_instance_diversity_audit.json": audit["automation_instance_diversity_audit"],
        "entity_binding_diversity_audit.json": audit["entity_binding_diversity_audit"],
        "canonical_ir_identity_audit.json": audit["canonical_ir_identity_audit"],
        "action_distribution_audit.json": audit["action_distribution_audit"],
        "invalid_sample_ids.json": {"ids": audit["invalid_sample_ids"], "count": audit["invalid_count"]},
    }
    for name, obj in mapping.items():
        (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "sample_validity_audit.jsonl").open("w", encoding="utf-8") as f:
        for row in audit["sample_validity_rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_v1_samples(v1_dir: Path) -> list[dict]:
    return read_jsonl(v1_dir / "single_scene_samples.jsonl")
