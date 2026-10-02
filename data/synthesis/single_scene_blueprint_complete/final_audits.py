from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.config import FORBIDDEN_SYNTHESIS_FIELDS, SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (
    SCENE_BUDGETS,
    TRUE_EVIDENCE_LIMITED_BLUEPRINTS,
    generated_blueprints_for_scene,
)
from smarthome_mdf.single_scene_blueprint_complete.final_scheduler import FinalGenerationScheduler

def _forbidden_scan(sample: dict) -> list[str]:
    found: list[str] = []
    forbidden = {f.lower() for f in FORBIDDEN_SYNTHESIS_FIELDS}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{path}.{k}" if path else k
                if k.lower() in forbidden:
                    found.append(p)
                walk(v, p)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(sample, "")
    return found

def build_all_audits(
    samples: list[dict],
    scheduler: FinalGenerationScheduler,
    complexity_doc: dict,
    specs: list[dict],
    rejects: Counter,
    attempts_by_bp: dict[str, int],
    numeric_dup_rejects: int,
    temporal_dup_rejects: int,
) -> dict[str, Any]:
    spec_by_bp = {s["blueprint_id"]: s for s in specs}
    bp_rows = []
    for bp_id in sorted({s["blueprint_id"] for s in specs} | set(SCENE_BLUEPRINT_MAP.get(s, "") for s in [])):
        pass

    all_bp_ids = sorted({b for ids in SCENE_BLUEPRINT_MAP.values() for b in ids})
    bp_samples: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        bp = (s.get("blueprint_binding") or {}).get("blueprint_id")
        if bp:
            bp_samples[bp].append(s)

    comp = {r["blueprint_id"]: r for r in complexity_doc.get("blueprints") or []}
    st_by_bp = {st.blueprint_id: st for st in scheduler.bp_state.values()}

    for bp_id in all_bp_ids:
        spec = spec_by_bp.get(bp_id) or {}
        rows_samples = bp_samples.get(bp_id, [])
        static_branches = len(spec.get("branches") or []) or 1
        static_paths = len({t.get("service") for t in spec.get("semantic_action_templates") or []}) or 1
        static_domain = sorted(set(spec.get("action_services") or spec.get("action_domain") or []))
        observed_branches = len({(s.get("synthesis_metadata") or {}).get("branch_id") for s in rows_samples})
        observed_paths = len({(s.get("synthesis_metadata") or {}).get("action_path_signature") for s in rows_samples})
        observed_domain = sorted(
            {
                svc
                for s in rows_samples
                for svc in ((s.get("synthesis_metadata") or {}).get("action_path_signature") or "").split("|")
                if svc
            }
        )
        gr_branches = len({t.branch_id for t in scheduler.behavior_targets if t.blueprint_id == bp_id and t.grounded_reachable})
        gr_paths = len(
            {
                t.action_path_signature
                for t in scheduler.behavior_targets
                if t.blueprint_id == bp_id and t.grounded_reachable
            }
        )
        st = st_by_bp.get(bp_id)
        bp_rows.append(
            {
                "blueprint_id": bp_id,
                "scene": spec.get("scene") or (st.scene if st else ""),
                "status": "TRUE_EVIDENCE_LIMITED" if bp_id in TRUE_EVIDENCE_LIMITED_BLUEPRINTS else ("GENERATED" if rows_samples else "NOT_GENERATED"),
                "complexity_score": comp.get(bp_id, {}).get("complexity_score", 0),
                "sample_count": len(rows_samples),
                "instance_count": len(spec.get("automation_instance_id") and [spec.get("automation_instance_id")] or []),
                "observed_instance_count": len({(s.get("blueprint_binding") or {}).get("automation_instance_id") for s in rows_samples}),
                "static_branch_count": static_branches,
                "grounded_reachable_branch_count": gr_branches or static_branches,
                "observed_branch_count": observed_branches,
                "grounded_branch_coverage": round(observed_branches / max(gr_branches or static_branches, 1), 4),
                "static_action_path_count": static_paths,
                "grounded_reachable_action_path_count": gr_paths or static_paths,
                "observed_action_path_count": observed_paths,
                "grounded_action_path_coverage": round(observed_paths / max(gr_paths or static_paths, 1), 4),
                "static_action_domain": static_domain,
                "grounded_reachable_action_domain": static_domain,
                "observed_action_domain": observed_domain,
                "runtime_situation_count": len({(s.get("provenance") or {}).get("source_record_id") for s in rows_samples}),
                "entity_binding_pattern_count": len({(s.get("synthesis_metadata") or {}).get("entity_binding_signature") for s in rows_samples}),
                "temporal_pattern_count": len(
                    {str((s.get("synthesis_metadata") or {}).get("temporal_pattern")) for s in rows_samples}
                ),
                "unique_source_observation_count": len({(s.get("provenance") or {}).get("source_record_id") for s in rows_samples}),
                "stop_reason": st.stop_reason if st else ("TRUE_EVIDENCE_LIMITED" if bp_id in TRUE_EVIDENCE_LIMITED_BLUEPRINTS else "UNKNOWN"),
            }
        )

    scene_rows = []
    for scene, budget in SCENE_BUDGETS.items():
        scene_samples = [s for s in samples if s.get("scene_type") == scene]
        inv = len(SCENE_BLUEPRINT_MAP.get(scene, []))
        gen = len(generated_blueprints_for_scene(scene))
        actual = len(scene_samples)
        scene_rows.append(
            {
                "scene": scene,
                "inventory_blueprints": inv,
                "generated_blueprints": gen,
                "true_evidence_limited_blueprints": inv - gen,
                "min_samples": budget["min"],
                "target_samples": budget["target"],
                "recommended_max": budget["recommended_max"],
                "actual_samples": actual,
                "unused_budget": max(0, budget["target"] - actual),
                "unique_branches": len({(s.get("synthesis_metadata") or {}).get("branch_id") for s in scene_samples}),
                "unique_action_paths": len({(s.get("synthesis_metadata") or {}).get("action_path_signature") for s in scene_samples}),
                "unique_action_services": len({(s.get("synthesis_metadata") or {}).get("action_path_signature") for s in scene_samples}),
                "unique_runtime_situations": len({(s.get("provenance") or {}).get("source_record_id") for s in scene_samples}),
                "unique_source_observations": len({(s.get("provenance") or {}).get("source_record_id") for s in scene_samples}),
                "stop_reason": scheduler.scene_state.get(scene, type("x", (), {"stop_reason": "PENDING"})()).stop_reason,
            }
        )

    source_reuse: Counter = Counter()
    dataset_dist: Counter = Counter()
    for s in samples:
        prov = s.get("provenance") or {}
        source_reuse[prov.get("source_record_id", "")] += 1
        dataset_dist[prov.get("source_dataset", "UNKNOWN")] += 1

    forbidden_hits = []
    for s in samples:
        forbidden_hits.extend(_forbidden_scan(s))

    total = len(samples) or 1
    bp_dist = Counter((s.get("blueprint_binding") or {}).get("blueprint_id") for s in samples)
    largest_bp = bp_dist.most_common(1)[0] if bp_dist else (None, 0)

    return {
        "blueprint_table_22": bp_rows,
        "scene_table_8": scene_rows,
        "branch_coverage.json": {"per_blueprint": {r["blueprint_id"]: r for r in bp_rows}},
        "action_path_coverage.json": {"per_blueprint": {r["blueprint_id"]: {"coverage": r["grounded_action_path_coverage"]} for r in bp_rows}},
        "action_domain_coverage.json": {"per_blueprint": {r["blueprint_id"]: r["observed_action_domain"] for r in bp_rows}},
        "runtime_diversity.json": {"per_blueprint": {r["blueprint_id"]: r["runtime_situation_count"] for r in bp_rows}},
        "entity_binding_diversity.json": {"per_blueprint": {r["blueprint_id"]: r["entity_binding_pattern_count"] for r in bp_rows}},
        "temporal_diversity.json": {"per_blueprint": {r["blueprint_id"]: r["temporal_pattern_count"] for r in bp_rows}},
        "source_observation_diversity.json": {
            "unique_source_observations": len(source_reuse),
            "largest_observation_reuse": source_reuse.most_common(1)[0] if source_reuse else (None, 0),
            "reuse_rate": round(1 - len(source_reuse) / total, 4),
            "per_dataset_unique": dict(Counter(k for k in source_reuse if k)),
        },
        "numeric_only_variation_audit.json": {
            "numeric_only_near_duplicate_rejects": numeric_dup_rejects,
            "numeric_only_near_duplicate_rate": round(numeric_dup_rejects / max(sum(rejects.values()), 1), 4),
        },
        "temporal_shift_duplicate_audit.json": {
            "temporal_shift_near_duplicate_rejects": temporal_dup_rejects,
            "temporal_shift_near_duplicate_rate": round(temporal_dup_rejects / max(sum(rejects.values()), 1), 4),
        },
        "generation_attempts.json": {"per_blueprint": attempts_by_bp, "total_attempts": sum(attempts_by_bp.values())},
        "generation_funnel.json": {"accepted": len(samples), "rejects": dict(rejects)},
        "scene_distribution.json": dict(Counter(s.get("scene_type") for s in samples)),
        "blueprint_distribution.json": dict(bp_dist),
        "instance_distribution.json": dict(Counter((s.get("blueprint_binding") or {}).get("automation_instance_id") for s in samples)),
        "source_dataset_distribution": dict(dataset_dist),
        "distribution_audit": {
            "total_samples": len(samples),
            "largest_blueprint_share": {"key": largest_bp[0], "count": largest_bp[1], "share": round(largest_bp[1] / total, 4)},
            "smallest_generated_blueprint_count": min((c for bp, c in bp_dist.items() if bp not in TRUE_EVIDENCE_LIMITED_BLUEPRINTS), default=0),
            "forbidden_field_hits": forbidden_hits[:20],
        },
        "capacity_limited_report.json": {
            "true_evidence_limited": list(TRUE_EVIDENCE_LIMITED_BLUEPRINTS),
            "blueprint_stop_reasons": {st.blueprint_id: st.stop_reason for st in scheduler.bp_state.values()},
        },
    }

def write_audit_artifacts(out_dir: Path, audits: dict[str, Any]) -> None:
    mapping = {
        "branch_coverage.json": audits["branch_coverage.json"],
        "action_path_coverage.json": audits["action_path_coverage.json"],
        "action_domain_coverage.json": audits["action_domain_coverage.json"],
        "runtime_diversity.json": audits["runtime_diversity.json"],
        "entity_binding_diversity.json": audits["entity_binding_diversity.json"],
        "temporal_diversity.json": audits["temporal_diversity.json"],
        "source_observation_diversity.json": audits["source_observation_diversity.json"],
        "numeric_only_variation_audit.json": audits["numeric_only_variation_audit.json"],
        "temporal_shift_duplicate_audit.json": audits["temporal_shift_duplicate_audit.json"],
        "generation_attempts.json": audits["generation_attempts.json"],
        "generation_funnel.json": audits["generation_funnel.json"],
        "scene_distribution.json": audits["scene_distribution.json"],
        "blueprint_distribution.json": audits["blueprint_distribution.json"],
        "instance_distribution.json": audits["instance_distribution.json"],
        "capacity_limited_report.json": audits["capacity_limited_report.json"],
    }
    for name, obj in mapping.items():
        (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
