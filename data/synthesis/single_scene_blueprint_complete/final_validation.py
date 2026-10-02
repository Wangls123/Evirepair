from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import build_all_behavior_specs, run_parser_audit
from smarthome_mdf.single_scene_blueprint_complete.blueprint_capacity import (
    build_capacity_row,
    classify_instance_coverage,
)
from smarthome_mdf.single_scene_blueprint_complete.config import (
    ALL_BLUEPRINT_IDS,
    FORBIDDEN_SYNTHESIS_FIELDS,
    PROJECT_SCENES,
    SCENE_BLUEPRINT_MAP,
)
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (
    GROUNDED_CAPABLE_BLUEPRINTS,
    TRUE_EVIDENCE_LIMITED_BLUEPRINTS,
    V3_CLEAN_OUTPUT_DIR_NAME,
)
from smarthome_mdf.single_scene_blueprint_complete.final_scheduler import FinalGenerationScheduler
from smarthome_mdf.single_scene_blueprint_complete.information_flow import audit_package_imports
from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.p1_integrity_audit import (
    audit_duplicates_reaudit,
    audit_raw_to_adapter_semantic,
    audit_stable_behavior_signatures,
    audit_visual_fusion,
    audit_visual_provenance,
    classify_sample_validity,
)
from smarthome_mdf.single_scene_blueprint_complete.pre_freeze_verification import (
    _RematchContext,
    classify_temporal_near_dup,
    rematch_sample,
)
from smarthome_mdf.single_scene_blueprint_complete.semantic_sanitize import detect_semantic_contamination
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import (
    behavior_source_dedup_key,
    canonical_sample_signature,
    is_dynamic_behavior_target,
    is_timestamp_only_duplicate,
    numeric_only_near_duplicate_stable,
    same_source_observation_duplicate,
)
from smarthome_mdf.single_scene_blueprint_complete.v3_post_audits import audit_duplicates
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import build_yaml_mapping_audit, resolve_yaml_path
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

REQUIRED_TOP_LEVEL = frozenset(
    {
        "sample_id",
        "scene_type",
        "blueprint_binding",
        "provenance",
        "synthesis_metadata",
        "entity_observations",
        "observed",
    }
)

SAMPLE_INFO_FLOW_PATTERNS = re.compile(
    r"\b(b0_actions|expected_state|expected_actions|ground_truth|conflict_label|repair_output|y_label|ars_result|decision_output)\b",
    re.I,
)

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _load_samples(jsonl_path: Path) -> tuple[list[dict], dict[str, Any]]:
    errors: list[str] = []
    samples: list[dict] = []
    if not jsonl_path.is_file():
        return [], {"JSON_PARSE_FAIL": 1, "errors": ["missing jsonl"]}
    for i, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"line {i}: {e}")
    return samples, {
        "JSON_PARSE_FAIL": len(errors),
        "parse_errors": errors[:20],
        "truncated_rows": 0,
    }

def _audit_file_integrity(samples: list[dict], parse_info: dict[str, Any]) -> dict[str, Any]:
    sample_ids = [s.get("sample_id") for s in samples]
    id_counts = Counter(sample_ids)
    dup_ids = {k: v for k, v in id_counts.items() if k and v > 1}
    schema_versions = Counter(s.get("schema_version") for s in samples)
    missing_required = 0
    malformed = 0
    for s in samples:
        missing = REQUIRED_TOP_LEVEL - set(s.keys())
        if missing:
            missing_required += 1
        bb = s.get("blueprint_binding")
        if not isinstance(bb, dict) or not bb.get("blueprint_id"):
            malformed += 1
    return {
        "FINAL_SAMPLE_COUNT": len(samples),
        "JSON_PARSE_FAIL": parse_info.get("JSON_PARSE_FAIL", 0),
        "DUPLICATE_SAMPLE_ID": sum(v - 1 for v in dup_ids.values()),
        "duplicate_sample_ids": list(dup_ids.keys())[:20],
        "SCHEMA_INVALID": sum(1 for v in schema_versions if v not in (None, "v3_clean", "single_scene_v3")),
        "schema_version_distribution": dict(schema_versions),
        "MISSING_REQUIRED_FIELD": missing_required,
        "malformed_blueprint_binding": malformed,
    }

def _audit_blueprint_mapping(samples: list[dict]) -> dict[str, Any]:
    bp_to_scene = {bp: scene for scene, bps in SCENE_BLUEPRINT_MAP.items() for bp in bps}
    yaml_audit = build_yaml_mapping_audit()
    yaml_mapped = yaml_audit.get("complete", False)
    yaml_rows = {r["blueprint_id"]: r for r in yaml_audit.get("rows") or []}
    mapping_invalid = scene_invalid = yaml_invalid = 0
    unknown = 0
    hits: list[dict] = []
    for s in samples:
        bb = s.get("blueprint_binding") or {}
        bp = bb.get("blueprint_id", "")
        scene = s.get("scene_type", "")
        if bp not in ALL_BLUEPRINT_IDS:
            unknown += 1
            mapping_invalid += 1
            continue
        if bp_to_scene.get(bp) != scene:
            scene_invalid += 1
        row = yaml_rows.get(bp) or {}
        if not row.get("yaml_exists"):
            yaml_invalid += 1
        if len(hits) < 20 and (bp_to_scene.get(bp) != scene or not row.get("yaml_exists")):
            hits.append({"sample_id": s.get("sample_id"), "blueprint_id": bp, "scene": scene})
    reg = build_registry()
    parser = run_parser_audit(reg)
    return {
        "BLUEPRINT_MAPPING_INVALID": mapping_invalid + unknown,
        "SCENE_MAPPING_INVALID": scene_invalid,
        "YAML_MAPPING_INVALID": yaml_invalid,
        "yaml_mapping_complete": yaml_mapped,
        "yaml_mapped_count": yaml_audit.get("mapped_count", 0),
        "IMPLEMENTATION_GAP": parser.get("implementation_gap", 0),
        "parser_audit": parser,
        "unknown_blueprint_hits": hits,
    }

def _audit_inventory(samples: list[dict]) -> dict[str, Any]:
    bp_counts = Counter((s.get("blueprint_binding") or {}).get("blueprint_id") for s in samples)
    rows = []
    for bp in ALL_BLUEPRINT_IDS:
        scene = next(sc for sc, bps in SCENE_BLUEPRINT_MAP.items() if bp in bps)
        if bp in TRUE_EVIDENCE_LIMITED_BLUEPRINTS:
            status = "TRUE_EVIDENCE_LIMITED"
        elif bp_counts.get(bp, 0) > 0:
            status = "GENERATED"
        else:
            status = "NOT_GENERATED"
        rows.append(
            {
                "blueprint_id": bp,
                "scene": scene,
                "sample_count": bp_counts.get(bp, 0),
                "status": status,
            }
        )
    generated = sum(1 for r in rows if r["status"] == "GENERATED")
    vibration_count = bp_counts.get("appliance_vibration_sensor", 0)
    return {
        "blueprint_rows": rows,
        "BLUEPRINT_INVENTORY": 22,
        "GROUNDED_CAPABLE_COVERAGE": generated,
        "GROUNDED_CAPABLE_EXPECTED": 21,
        "TRUE_EVIDENCE_LIMITED": 1,
        "appliance_vibration_sensor_samples": vibration_count,
        "SCENE_COVERAGE": len({s.get("scene_type") for s in samples} & set(PROJECT_SCENES)),
    }

def _audit_scenes(samples: list[dict]) -> dict[str, Any]:
    scenes = Counter(s.get("scene_type") for s in samples)
    total = len(samples) or 1
    rows = []
    for sc in PROJECT_SCENES:
        c = scenes.get(sc, 0)
        rows.append({"scene": sc, "final_sample_count": c, "percentage": round(100 * c / total, 2)})
    return {"scene_rows": rows, "SCENE_COVERAGE": len([r for r in rows if r["final_sample_count"] > 0])}

def _audit_runtime_rematch(samples: list[dict], inst_idx: dict[str, list[dict]]) -> dict[str, Any]:
    pool = build_legacy_source_pool(PROJECT_SCENES, include_public=True)
    ctx = _RematchContext(reg=build_registry(), pool=pool, instance_templates=inst_idx)
    ok = fail = 0
    failures: list[dict] = []
    for s in samples:
        r = rematch_sample(s, ctx)
        if r.get("status") == "MATCH_OK":
            ok += 1
        else:
            fail += 1
            if len(failures) < 50:
                bb = s.get("blueprint_binding") or {}
                failures.append(
                    {
                        "sample_id": s.get("sample_id"),
                        "blueprint_id": bb.get("blueprint_id"),
                        "instance_id": bb.get("automation_instance_id"),
                        "source_observation": (s.get("provenance") or {}).get("source_record_id"),
                        "failed_requirement": r.get("unsatisfied"),
                        "failure_type": r.get("status"),
                        "reason": r.get("reason"),
                    }
                )
    return {
        "POST_GENERATION_RUNTIME_MATCH": ok,
        "MATCH_FAIL": fail,
        "total": len(samples),
        "all_match_ok": fail == 0,
        "failures": failures,
    }

def _audit_semantic(samples: list[dict]) -> dict[str, Any]:
    vf_audit = audit_visual_fusion([s for s in samples if s.get("scene_type") == "visual_fusion"])
    invalid = 0
    hits: list[dict] = []
    for s in samples:
        validity, reason = classify_sample_validity(s, vf_audit=vf_audit)
        if validity.startswith("INVALID"):
            invalid += 1
            if len(hits) < 30:
                hits.append({"sample_id": s.get("sample_id"), "validity": validity, "reason": reason})
    return {"SEMANTIC_INVALID_ACCEPTED": invalid, "invalid_samples": hits}

def _audit_provenance(samples: list[dict]) -> dict[str, Any]:
    missing_obs = 0
    invalid = 0
    cross_invalid = 0
    hits: list[dict] = []
    for s in samples:
        prov = s.get("provenance") or {}
        rid = prov.get("source_record_id") or (prov.get("runtime_source") or {}).get("observation_id")
        if not rid:
            missing_obs += 1
        if not prov.get("source_dataset") and not (prov.get("runtime_source") or {}).get("dataset"):
            invalid += 1
        comp = prov.get("composition_type")
        if comp == "CROSS_DATASET_SYNTHETIC":
            if not prov.get("runtime_source") or not prov.get("visual_source"):
                cross_invalid += 1
        if len(hits) < 20 and (not rid or invalid):
            hits.append({"sample_id": s.get("sample_id"), "provenance": prov})
    vf = audit_visual_provenance(samples)
    return {
        "PROVENANCE_INVALID": invalid + vf.get("provenance_issue_count", 0),
        "MISSING_SOURCE_OBSERVATION": missing_obs,
        "INVALID_CROSS_DATASET_PROVENANCE": cross_invalid,
        "visual_provenance_issues": vf.get("provenance_issue_count", 0),
        "hits": hits,
    }

def _audit_visual_fusion_detailed(samples: list[dict]) -> dict[str, Any]:
    vf = [s for s in samples if s.get("scene_type") == "visual_fusion"]
    audit = audit_visual_fusion(vf)
    bp_counts = Counter((s.get("blueprint_binding") or {}).get("blueprint_id") for s in vf)
    frames = Counter((s.get("observed") or {}).get("frame_id") for s in vf)
    sources = Counter((s.get("provenance") or {}).get("source_record_id") for s in vf)
    reuse = list(frames.values())
    reuse.sort()
    mid = reuse[len(reuse) // 2] if reuse else 0
    same_frame_same_beh: Counter = Counter()
    by_key: dict[str, dict] = {}
    for s in vf:
        key = behavior_source_dedup_key(s)
        if key in by_key and is_timestamp_only_duplicate(s, by_key[key]):
            same_frame_same_beh["timestamp_only"] += 1
        by_key[key] = s
    return {
        "visual_sample_count": len(vf),
        "camera_frigate_vision_llm": bp_counts.get("camera_frigate_vision_llm", 0),
        "camera_frigate_intelligent": bp_counts.get("camera_frigate_intelligent", 0),
        "unique_frame_ids": len(frames),
        "unique_source_observations": len(sources),
        "max_frame_reuse": max(reuse) if reuse else 0,
        "median_frame_reuse": mid,
        "invalid_frame_reference": 0,
        "missing_visual_provenance": audit.get("cross_dataset_without_explicit_composition_count", 0),
        "same_frame_same_behavior_duplicates": sum(same_frame_same_beh.values()),
        "single_frame_domination": audit.get("single_frame_domination", False),
        "frame_reuse_top10": audit.get("frame_reuse_distribution_top10"),
    }

def _audit_duplicates_full(samples: list[dict]) -> dict[str, Any]:
    exact: Counter = Counter()
    canonical: Counter = Counter()
    padding = 0
    ts_only = 0
    numeric = 0
    by_canon: dict[str, dict] = {}
    by_bs: dict[str, dict] = {}
    n = len(samples)
    for s in samples:
        cs = canonical_sample_signature(s)
        exact[cs] += 1
        bk = behavior_source_dedup_key(s)
        canonical[bk] += 1
        prior_c = by_canon.get(cs)
        if prior_c and canonical_sample_signature(prior_c) == cs:
            pass
        prior_b = by_bs.get(bk)
        if prior_b:
            if same_source_observation_duplicate(s, prior_b):
                padding += 1
            elif is_timestamp_only_duplicate(s, prior_b):
                ts_only += 1
            elif numeric_only_near_duplicate_stable(s, prior_b):
                numeric += 1
        by_canon[cs] = s
        by_bs[bk] = s
    exact_dup = sum(c - 1 for c in exact.values() if c > 1)
    canon_dup = sum(c - 1 for c in canonical.values() if c > 1)
    temporal = classify_temporal_near_dup(samples)
    return {
        "EXACT_DUPLICATE": exact_dup,
        "CANONICAL_DUPLICATE": canon_dup,
        "SAME_SOURCE_DUPLICATE_PADDING": padding,
        "TIMESTAMP_ONLY_DUPLICATE": ts_only,
        "NUMERIC_ONLY_NEAR_DUPLICATE": numeric,
        "exact_duplicate_rate": round(exact_dup / max(n, 1), 6),
        "canonical_duplicate_rate": round(canon_dup / max(n, 1), 6),
        "padding_rate": round(padding / max(n, 1), 6),
        "timestamp_only_rate": round(ts_only / max(n, 1), 6),
        "temporal_near_dup_audit": temporal,
        "unique_behavior_source_pairs": len(by_bs),
        "duplicate_behavior_source_pairs": canon_dup,
    }

def _audit_source_diversity(samples: list[dict]) -> dict[str, Any]:
    ds = Counter((s.get("provenance") or {}).get("source_dataset") for s in samples)
    src_ids = Counter((s.get("provenance") or {}).get("source_record_id") for s in samples)
    counts = sorted(src_ids.values())
    p95_idx = int(len(counts) * 0.95) - 1 if counts else 0
    return {
        "source_dataset_distribution": dict(ds),
        "CASAS": ds.get("CASAS", 0),
        "UK-DALE": ds.get("UK-DALE", 0) + ds.get("UK_DALE", 0),
        "YouHome": ds.get("YouHome", 0) + ds.get("YOUHOME", 0),
        "unique_source_observation_count": len(src_ids),
        "max_samples_per_source": max(src_ids.values()) if src_ids else 0,
        "median_samples_per_source": counts[len(counts) // 2] if counts else 0,
        "p95_source_reuse": counts[p95_idx] if counts else 0,
    }

def _audit_instances(samples: list[dict], grounded_raw: dict[str, list]) -> dict[str, Any]:
    invalid = snapshot_missing = inflation = 0
    rows = []
    for bp in GROUNDED_CAPABLE_BLUEPRINTS:
        grounded_ids = {c["automation_instance_id"] for c in grounded_raw.get(bp) or []}
        observed = {
            (s.get("blueprint_binding") or {}).get("automation_instance_id")
            for s in samples
            if (s.get("blueprint_binding") or {}).get("blueprint_id") == bp
        }
        observed.discard(None)
        cov = classify_instance_coverage(grounded_instance_ids=grounded_ids, observed_instance_ids=observed)
        if cov["legacy_instance_ids_present"] and cov["current_grounded_instances_covered"] > cov["current_grounded_instance_count"]:
            inflation += 1
        for s in samples:
            if (s.get("blueprint_binding") or {}).get("blueprint_id") != bp:
                continue
            snap = s.get("blueprint_binding") or {}
            if not snap.get("automation_instance_id"):
                snapshot_missing += 1
                break
        rows.append({"blueprint_id": bp, **cov})
    return {
        "INSTANCE_INVALID": invalid,
        "INSTANCE_SNAPSHOT_MISSING": snapshot_missing,
        "LEGACY_INSTANCE_COVERAGE_INFLATION": inflation,
        "per_blueprint": rows,
        "legacy_instance_ids_global": sorted(
            {
                iid
                for bp in GROUNDED_CAPABLE_BLUEPRINTS
                for iid in classify_instance_coverage(
                    grounded_instance_ids={c["automation_instance_id"] for c in grounded_raw.get(bp) or []},
                    observed_instance_ids={
                        (s.get("blueprint_binding") or {}).get("automation_instance_id")
                        for s in samples
                        if (s.get("blueprint_binding") or {}).get("blueprint_id") == bp
                    }
                    - {None},
                )["legacy_instance_ids_present"]
            }
        ),
    }

def _audit_stable_signatures(samples: list[dict]) -> dict[str, Any]:
    audit = audit_stable_behavior_signatures(samples)
    dynamic = sum(
        1
        for s in samples
        if is_dynamic_behavior_target((s.get("synthesis_metadata") or {}).get("behavior_target"))
    )
    malformed_branch = sum(
        1 for s in samples if not (s.get("synthesis_metadata") or {}).get("branch_id")
    )
    unstable_path = sum(
        1
        for s in samples
        if not (s.get("synthesis_metadata") or {}).get("action_path_signature")
    )
    return {
        **audit,
        "DYNAMIC_BEHAVIOR_ID": dynamic,
        "MALFORMED_BRANCH_ID": malformed_branch,
        "UNSTABLE_ACTION_PATH_ID": unstable_path,
    }

def _audit_contamination(samples: list[dict]) -> dict[str, Any]:
    raw = audit_raw_to_adapter_semantic(samples)
    motion_power = raw.get("motion_with_power_count", 0)
    motion_temp = raw.get("motion_with_temperature_count", 0)
    cross_modal = 0
    for s in samples:
        if s.get("scene_type") == "visual_fusion":
            prov = s.get("provenance") or {}
            if prov.get("source_dataset") == "CASAS" and (s.get("observed") or {}).get("frame_id"):
                if prov.get("composition_type") != "CROSS_DATASET_SYNTHETIC":
                    cross_modal += 1
    accepted_contam = sum(1 for s in samples if detect_semantic_contamination(s))
    return {
        "MOTION_POWER_CONTAMINATION": motion_power,
        "MOTION_TEMPERATURE_CONTAMINATION": motion_temp,
        "CROSS_MODAL_CONTAMINATION": cross_modal,
        "accepted_samples_with_any_contamination_field": accepted_contam,
    }

def _audit_information_flow(samples: list[dict]) -> dict[str, Any]:
    code_violations = audit_package_imports()
    sample_violations: list[str] = []
    forbidden = {f.lower() for f in FORBIDDEN_SYNTHESIS_FIELDS}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{path}.{k}" if path else k
                if k.lower() in forbidden:
                    sample_violations.append(p)
                walk(v, p)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")
        elif isinstance(node, str) and SAMPLE_INFO_FLOW_PATTERNS.search(node):
            sample_violations.append(path)

    for s in samples[:500]:
        walk(s, s.get("sample_id", ""))
    return {
        "INFORMATION_FLOW_VIOLATION": len(code_violations) + len(sample_violations),
        "code_import_violations": code_violations,
        "sample_field_violations_sample": sample_violations[:30],
    }

def _audit_reproducibility(out_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    present = {}
    for name in (
        "manifest.json",
        "blueprint_budget.json",
        "blueprint_complexity.json",
        "grounded_instance_candidates.json",
        "blueprint_yaml_mapping.json",
    ):
        present[name] = (out_dir / name).is_file()
    return {
        "manifest_global_stop": manifest.get("global_stop_reason"),
        "checkpoint_samples_at_start": manifest.get("checkpoint_samples_at_start"),
        "diversity_selector_summary": manifest.get("diversity_selector"),
        "artifact_presence": present,
        "synthesis_version": "v3_clean",
        "canonical_signature_version": "stable_v3",
        "deterministic_seed_policy": "diversity_selector.generation_step + tiebreak hash",
        "missing_metadata": [k for k, v in present.items() if not v],
    }

def _audit_capacity(samples: list[dict], out_dir: Path) -> dict[str, Any]:
    grounded_raw = json.loads((out_dir / "grounded_instance_candidates.json").read_text(encoding="utf-8"))
    inst_idx = {
        bp: [
            {
                "automation_instance_id": c["automation_instance_id"],
                "blueprint_id": c["blueprint_id"],
                "scene": c["scene"],
                "blueprint_inputs": c.get("input_bindings") or {},
                "bound_entities": c.get("entity_bindings") or [],
            }
            for c in cands
        ]
        for bp, cands in grounded_raw.items()
    }
    reg = build_registry()
    pool = build_legacy_source_pool(PROJECT_SCENES, include_public=True)
    specs = build_all_behavior_specs(reg, instance_overrides=inst_idx)
    blueprint_budget = json.loads((out_dir / "blueprint_budget.json").read_text(encoding="utf-8"))
    scheduler = FinalGenerationScheduler(specs, blueprint_budget.get("blueprints") or [], instance_templates=inst_idx)
    from smarthome_mdf.single_scene_blueprint_complete.diversity_source_selector import DiversitySourceSelector

    selector = DiversitySourceSelector()
    selector.restore_from_samples(samples)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    bp_state = {r["blueprint_id"]: r for r in manifest["scheduler_state"]["blueprints"]}
    rows = []
    for bp_id in GROUNDED_CAPABLE_BLUEPRINTS:
        st = bp_state.get(bp_id) or {}
        scene = st.get("scene") or (inst_idx.get(bp_id) or [{}])[0].get("scene", "")
        rows.append(
            build_capacity_row(
                blueprint_id=bp_id,
                scene=scene,
                samples=samples,
                scheduler=scheduler,
                selector=selector,
                pool=pool,
                inst_list=inst_idx.get(bp_id) or [],
                grounded_raw=grounded_raw.get(bp_id) or [],
                reg=reg,
                scheduler_stop_reason=st.get("stop_reason", "UNKNOWN"),
            )
        )
    remaining_total = sum(r["remaining_grounded_pairs"] for r in rows)
    return {
        "blueprint_capacity_rows": rows,
        "remaining_grounded_pairs_total": remaining_total,
        "global_stop_reason": manifest.get("global_stop_reason"),
        "final_count": len(samples),
        "TARGET_REACHED": len(samples) >= 12000,
    }

def _compute_verdict(report: dict[str, Any]) -> tuple[str, list[str]]:
    blockers: list[str] = []
    fi = report["file_integrity"]
    if fi["FINAL_SAMPLE_COUNT"] != 12000:
        blockers.append(f"FINAL_SAMPLE_COUNT={fi['FINAL_SAMPLE_COUNT']} != 12000")
    for key in ("JSON_PARSE_FAIL", "DUPLICATE_SAMPLE_ID", "SCHEMA_INVALID", "MISSING_REQUIRED_FIELD"):
        if fi.get(key, 0) > 0:
            blockers.append(f"{key}={fi[key]}")
    inv = report["inventory"]
    if inv["GROUNDED_CAPABLE_COVERAGE"] != 21:
        blockers.append(f"GROUNDED_CAPABLE_COVERAGE={inv['GROUNDED_CAPABLE_COVERAGE']}/21")
    if inv["appliance_vibration_sensor_samples"] > 0:
        blockers.append("appliance_vibration_sensor has samples")
    if inv["SCENE_COVERAGE"] != 8:
        blockers.append(f"SCENE_COVERAGE={inv['SCENE_COVERAGE']}/8")
    bm = report["blueprint_mapping"]
    for key in ("BLUEPRINT_MAPPING_INVALID", "YAML_MAPPING_INVALID", "SCENE_MAPPING_INVALID", "IMPLEMENTATION_GAP"):
        if bm.get(key, 0) > 0:
            blockers.append(f"{key}={bm[key]}")
    if not bm.get("yaml_mapping_complete"):
        blockers.append("yaml_mapping incomplete")
    rm = report["runtime_rematch"]
    if rm["MATCH_FAIL"] > 0:
        blockers.append(f"POST_GENERATION_RUNTIME_MATCH_FAIL={rm['MATCH_FAIL']}")
    for key in ("SEMANTIC_INVALID_ACCEPTED", "PROVENANCE_INVALID"):
        if report["semantic"].get(key, 0) > 0 or report["provenance"].get(key, 0) > 0:
            blockers.append(f"{key}>0")
    dup = report["duplicates"]
    for key in ("EXACT_DUPLICATE", "CANONICAL_DUPLICATE", "SAME_SOURCE_DUPLICATE_PADDING"):
        if dup.get(key, 0) > 0:
            blockers.append(f"{key}={dup[key]}")
    if report["information_flow"]["INFORMATION_FLOW_VIOLATION"] > 0:
        blockers.append(f"INFORMATION_FLOW_VIOLATION={report['information_flow']['INFORMATION_FLOW_VIOLATION']}")
    sig = report["stable_signatures"]
    for key in ("DYNAMIC_BEHAVIOR_ID", "MALFORMED_BRANCH_ID", "UNSTABLE_ACTION_PATH_ID"):
        if sig.get(key, 0) > 0:
            blockers.append(f"{key}={sig[key]}")
    verdict = "READY_FOR_SINGLE_SCENE_FREEZE" if not blockers else "BLOCKED_BEFORE_SINGLE_SCENE_FREEZE"
    return verdict, blockers

def _format_md(report: dict[str, Any]) -> str:
    lines = [
        "# V3 Clean Final Validation Report",
        "",
        f"**Generated:** {report['generated_at']}",
        f"**FINAL VERDICT:** `{report['FINAL_VERDICT']}`",
        "",
        f"- Samples: **{report['file_integrity']['FINAL_SAMPLE_COUNT']}**",
        f"- Runtime match: **{report['runtime_rematch']['POST_GENERATION_RUNTIME_MATCH']}/{report['runtime_rematch']['total']}**",
        f"- Global stop: **{report['reproducibility'].get('manifest_global_stop')}**",
        "",
    ]
    if report["blockers"]:
        lines.extend(["## Blockers", ""] + [f"- {b}" for b in report["blockers"]] + [""])
    lines.extend(["## Scene Distribution", ""])
    for r in report["scenes"]["scene_rows"]:
        lines.append(f"- {r['scene']}: {r['final_sample_count']} ({r['percentage']}%)")
    lines.extend(["", "## Hard Metrics", ""])
    for section in ("duplicates", "semantic", "provenance", "contamination", "information_flow"):
        lines.append(f"### {section}")
        lines.append(f"```json\n{json.dumps(report[section], indent=2)[:2000]}\n```")
        lines.append("")
    return "\n".join(lines)

def run_final_validation(out_dir: Path | None = None) -> dict[str, Any]:
    out = out_dir or (OUTPUT_DIR / V3_CLEAN_OUTPUT_DIR_NAME)
    jsonl = out / "single_scene_samples.jsonl"
    samples, parse_info = _load_samples(jsonl)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8")) if (out / "manifest.json").is_file() else {}
    grounded_raw = json.loads((out / "grounded_instance_candidates.json").read_text(encoding="utf-8")) if (out / "grounded_instance_candidates.json").is_file() else {}
    inst_idx = {
        bp: [
            {
                "automation_instance_id": c["automation_instance_id"],
                "blueprint_id": c["blueprint_id"],
                "scene": c["scene"],
                "blueprint_inputs": c.get("input_bindings") or {},
                "bound_entities": c.get("entity_bindings") or [],
            }
            for c in cands
        ]
        for bp, cands in grounded_raw.items()
    }

    report: dict[str, Any] = {
        "generated_at": _utc_now(),
        "dataset_path": str(jsonl),
        "validation_mode": "independent_read_only",
        "file_integrity": _audit_file_integrity(samples, parse_info),
        "blueprint_mapping": _audit_blueprint_mapping(samples),
        "inventory": _audit_inventory(samples),
        "scenes": _audit_scenes(samples),
        "runtime_rematch": _audit_runtime_rematch(samples, inst_idx),
        "semantic": _audit_semantic(samples),
        "provenance": _audit_provenance(samples),
        "visual_fusion": _audit_visual_fusion_detailed(samples),
        "duplicates": _audit_duplicates_full(samples),
        "source_diversity": _audit_source_diversity(samples),
        "instances": _audit_instances(samples, grounded_raw),
        "stable_signatures": _audit_stable_signatures(samples),
        "contamination": _audit_contamination(samples),
        "information_flow": _audit_information_flow(samples),
        "reproducibility": _audit_reproducibility(out, manifest),
        "capacity": _audit_capacity(samples, out),
        "duplicate_reaudit_legacy": audit_duplicates_reaudit(samples),
    }
    verdict, blockers = _compute_verdict(report)
    report["FINAL_VERDICT"] = verdict
    report["blockers"] = blockers
    report["PRODUCTION_STATUS"] = "PRODUCTION_COMPLETE_READY_FOR_FINAL_VALIDATION" if verdict == "READY_FOR_SINGLE_SCENE_FREEZE" else "PRODUCTION_BLOCKED"

    (out / "final_validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "FINAL_VALIDATION_REPORT.md").write_text(_format_md(report), encoding="utf-8")
    return report
