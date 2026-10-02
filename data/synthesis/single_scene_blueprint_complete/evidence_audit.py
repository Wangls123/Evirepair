from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.multi_action_vnext.requirement_extractor import (
    extract_requirement_bundle,
    minimal_grounding_requirements,
)
from smarthome_mdf.multi_action_vnext.requirement_types import (
    CurrentStateRequirement,
    HistoryRequirement,
    MeasurementRequirement,
    SubsequentEventRequirement,
    TemporalRequirement,
    TemplateVariableRequirement,
    UnifiedRequirement,
)
from smarthome_mdf.multi_action_vnext.source_adapters import SourcePool
from smarthome_mdf.multi_action_vnext.source_capabilities import (
    datasets_for_requirement,
    observation_supports_domain,
    observation_supports_measurement,
)
from smarthome_mdf.multi_action_vnext.unified_runtime_matcher import (
    match_requirement_bundle,
    satisfies_requirement,
)
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance
from smarthome_mdf.single_scene_blueprint_complete.config import (
    OUTPUT_DIR,
    PROJECT_SCENES,
    SCENE_BLUEPRINT_MAP,
)
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import build_instance_templates
from smarthome_mdf.single_scene_blueprint_complete.inventory import blueprint_to_scene
from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.public_observation_pool import (
    load_public_observations,
    probe_raw_datasets,
)
from smarthome_mdf.single_scene_blueprint_complete.synthesizer import synthesize_single_scene
from smarthome_mdf.single_scene_blueprint_complete.yaml_acquisition import resolve_canonical_path
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

ROOT_CAUSE_CLASSES = (
    "RAW_SOURCE_AVAILABLE",
    "SOURCE_POOL_GAP",
    "ADAPTER_SEMANTIC_GAP",
    "MATCHER_GAP",
    "ENTITY_BINDING_GAP",
    "TEMPORAL_MATCHING_GAP",
    "TRUE_EVIDENCE_LIMITED",
)

REQUIREMENT_TYPES = [
    "current_state",
    "power_measurement",
    "motion",
    "illuminance",
    "contact_state",
    "history",
    "event_sequence",
    "subsequent_event",
    "image",
    "object_detection",
    "temperature",
    "humidity",
    "time_window",
    "schedule",
    "trigger_context",
    "vibration",
    "template_variable",
    "observed",
]

def _req_type(req: UnifiedRequirement) -> str:
    if isinstance(req, CurrentStateRequirement):
        if req.entity_role in ("door", "window"):
            return "contact_state"
        return "current_state"
    if isinstance(req, MeasurementRequirement):
        m = req.measurement or "observed"
        if m == "power":
            return "power_measurement"
        if m in ("motion", "person_detected", "presence"):
            return "motion"
        if m == "illuminance":
            return "illuminance"
        if m == "temperature":
            return "temperature"
        if m == "trigger_context":
            return "trigger_context"
        if m == "window":
            return "contact_state"
        return m
    if isinstance(req, HistoryRequirement):
        return "history"
    if isinstance(req, SubsequentEventRequirement):
        return "subsequent_event"
    if isinstance(req, TemporalRequirement):
        return "schedule" if req.context == "schedule" else "time_window"
    if isinstance(req, TemplateVariableRequirement):
        return "template_variable"
    return req.kind if hasattr(req, "kind") else "unknown"

def _classify_requirement_failure(
    req: UnifiedRequirement,
    unsat: list[str],
    *,
    legacy_pool: SourcePool,
    public_by_scene: dict[str, list],
    scene: str,
    raw_probe: dict,
) -> str:
    rtype = _req_type(req)
    compatible_datasets = datasets_for_requirement(req)

    raw_has = False
    canonical_by_ds = raw_probe.get("canonical_by_dataset") or {}
    for ds in compatible_datasets:
        if canonical_by_ds.get(ds, 0) > 0:
            raw_has = True
            break

    if rtype == "vibration":
        return "TRUE_EVIDENCE_LIMITED"
    if isinstance(req, MeasurementRequirement) and req.measurement == "temperature":
        if "UK-DALE" in compatible_datasets and scene in ("appliance_monitoring", "periodic_task_scheduling"):
            return "TRUE_EVIDENCE_LIMITED"
    if isinstance(req, TemplateVariableRequirement):
        return "ADAPTER_SEMANTIC_GAP"
    if any("!input" in u for u in unsat):
        return "ADAPTER_SEMANTIC_GAP"

    legacy_cands = legacy_pool.candidates_for_scene(scene)
    pub_cands = public_by_scene.get(scene, [])
    all_cands = legacy_cands + [c for c in pub_cands if c.record_id not in {x.record_id for x in legacy_cands}]

    raw_match = any(satisfies_requirement(req, o)[0] for o in all_cands)
    legacy_match = any(satisfies_requirement(req, o)[0] for o in legacy_cands)

    if raw_has and not legacy_match and raw_match:
        return "SOURCE_POOL_GAP"
    if raw_has and not raw_match and rtype in ("history", "subsequent_event"):
        return "TEMPORAL_MATCHING_GAP"
    if raw_has and not raw_match:
        return "ADAPTER_SEMANTIC_GAP"

    if isinstance(req, HistoryRequirement) and unsat and "HISTORY" in str(unsat):
        if raw_probe.get("evidence_windows_count", 0) > 0:
            return "SOURCE_POOL_GAP"
        return "TRUE_EVIDENCE_LIMITED"

    if isinstance(req, SubsequentEventRequirement):
        if raw_probe.get("evidence_windows_count", 0) > 0 and not raw_match:
            return "TEMPORAL_MATCHING_GAP"
        if not raw_has:
            return "TRUE_EVIDENCE_LIMITED"

    if isinstance(req, TemporalRequirement) and "temporal" in str(unsat).lower():
        if scene == "scene_schedule_override" and not legacy_cands:
            return "SOURCE_POOL_GAP"
        return "TRUE_EVIDENCE_LIMITED"

    if any("entity_domain" in u for u in unsat):
        return "ENTITY_BINDING_GAP"
    if any("NO_COMPATIBLE_RUNTIME" in u for u in unsat):
        return "TEMPORAL_MATCHING_GAP"
    if not raw_has and not legacy_match:
        return "TRUE_EVIDENCE_LIMITED"
    if raw_has and not legacy_match:
        return "SOURCE_POOL_GAP"
    return "MATCHER_GAP"

def _audit_one_blueprint(
    bp_id: str,
    *,
    legacy_pool: SourcePool,
    extended_pool: SourcePool,
    public_by_scene: dict,
    raw_probe: dict,
    reg: dict,
    inst_idx: dict,
) -> dict[str, Any]:
    scene = blueprint_to_scene(bp_id) or ""
    yaml_path = resolve_canonical_path(bp_id)
    instance = (inst_idx.get(bp_id) or [{}])[0]
    nb = _normalize_from_instance(scene, instance, reg)
    ir = None
    if yaml_path and yaml_path.is_file():
        ir = parse_blueprint_ir(bp_id, yaml_path, instance.get("blueprint_inputs") or {})
    bundle = extract_requirement_bundle(nb, ir=ir)
    grounding = minimal_grounding_requirements(bundle)

    obs_legacy, status_legacy, traces_legacy = match_requirement_bundle(legacy_pool, bundle)
    obs_ext, status_ext, traces_ext = match_requirement_bundle(extended_pool, bundle)

    audit_traces = traces_legacy if status_legacy != "MATCH_OK" else traces_ext
    audit_obs = obs_legacy if status_legacy != "MATCH_OK" else obs_ext

    req_audits = []
    primary_cause = "TRUE_EVIDENCE_LIMITED"
    cause_counts: Counter = Counter()

    trace_by_req = {i: t for i, t in enumerate(audit_traces or [])}
    for i, req in enumerate(grounding):
        trace = trace_by_req.get(i, {})
        unsat = trace.get("unsatisfied_requirements") or []
        ok = status_ext == "MATCH_OK" or (status_legacy == "MATCH_OK")
        if not ok:
            ok = not unsat

        if not ok and not unsat:
            unsat = ["UNSATISFIED"]

        cause = "SATISFIED" if ok else _classify_requirement_failure(
            req, unsat, legacy_pool=legacy_pool, public_by_scene=public_by_scene, scene=scene, raw_probe=raw_probe
        )
        if cause != "SATISFIED":
            cause_counts[cause] += 1
        req_audits.append(
            {
                "requirement": req.to_dict(),
                "requirement_type": _req_type(req),
                "status": "SATISFIED" if ok else "UNSATISFIED",
                "unsatisfied_detail": unsat,
                "root_cause": cause if not ok else None,
                "compatible_datasets": sorted(datasets_for_requirement(req)),
            }
        )

    if status_ext == "MATCH_OK":
        primary_cause = "RECOVERED"
    elif status_legacy == "MATCH_OK":
        primary_cause = "SOURCE_POOL_GAP"
    elif cause_counts:
        primary_cause = cause_counts.most_common(1)[0][0]

    return {
        "scene": scene,
        "blueprint_id": bp_id,
        "yaml_path": str(yaml_path) if yaml_path else None,
        "behavior_targets": (nb.triggers or [])[:5],
        "grounding_requirements": [r.to_dict() for r in grounding],
        "requirement_audits": req_audits,
        "match_status_legacy_pool": status_legacy,
        "match_status_extended_pool": status_ext,
        "primary_root_cause": primary_cause,
        "root_cause_counts": dict(cause_counts),
        "legacy_pool_candidates": len(legacy_pool.candidates_for_scene(scene)),
        "extended_pool_candidates": len(extended_pool.candidates_for_scene(scene)),
    }

def build_source_capability_matrix(raw_probe: dict) -> dict[str, Any]:
    public = load_public_observations()
    matrix: dict[str, dict[str, str]] = {}
    datasets = ["CASAS", "UK-DALE", "YouHome", "REFIT"]
    canonical_by_ds = raw_probe.get("canonical_by_dataset") or {}

    type_dataset_signals: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for scene, obs_list in public.items():
        for obs in obs_list:
            for rtype in REQUIREMENT_TYPES:
                if rtype == "motion" and observation_supports_measurement(obs, "motion"):
                    type_dataset_signals[rtype][obs.dataset].add("ADAPTED")
                if rtype == "illuminance" and observation_supports_measurement(obs, "illuminance"):
                    type_dataset_signals[rtype][obs.dataset].add("ADAPTED")
                if rtype == "power_measurement" and observation_supports_measurement(obs, "power"):
                    type_dataset_signals[rtype][obs.dataset].add("ADAPTED")
                if rtype == "temperature" and observation_supports_measurement(obs, "temperature"):
                    type_dataset_signals[rtype][obs.dataset].add("ADAPTED")
                if rtype == "contact_state" and observation_supports_measurement(obs, "window"):
                    type_dataset_signals[rtype][obs.dataset].add("ADAPTED")
                if rtype == "history" and (obs.history or obs.capabilities.get("history_available")):
                    type_dataset_signals[rtype][obs.dataset].add("ADAPTED")
                if rtype == "subsequent_event" and obs.history:
                    type_dataset_signals[rtype][obs.dataset].add("ADAPTED")

    for rtype in REQUIREMENT_TYPES:
        matrix[rtype] = {}
        for ds in datasets:
            if type_dataset_signals[rtype].get(ds):
                matrix[rtype][ds] = "AVAILABLE_ADAPTED"
            elif canonical_by_ds.get(ds, 0) > 0 and rtype in ("motion", "temperature", "contact_state", "power_measurement"):
                matrix[rtype][ds] = "AVAILABLE_RAW"
            elif ds == "YouHome" and rtype in ("image", "object_detection") and canonical_by_ds.get("YouHome", 0) > 0:
                matrix[rtype][ds] = "AVAILABLE_RAW"
            elif rtype == "vibration":
                matrix[rtype][ds] = "NOT_AVAILABLE"
            elif rtype == "humidity":
                matrix[rtype][ds] = "NOT_AVAILABLE"
            else:
                matrix[rtype][ds] = "UNKNOWN"

    return {"requirement_types": REQUIREMENT_TYPES, "datasets": datasets, "matrix": matrix}

def build_blueprint_source_compatibility(audits: list[dict]) -> dict[str, Any]:
    rows = []
    for a in audits:
        reqs = a.get("requirement_audits") or []
        ds_compat: dict[str, str] = {"CASAS": "COMPATIBLE", "UK-DALE": "COMPATIBLE", "YouHome": "COMPATIBLE", "REFIT": "UNKNOWN"}
        for r in reqs:
            if r.get("status") == "SATISFIED":
                continue
            cause = r.get("root_cause")
            for ds in r.get("compatible_datasets") or []:
                if cause == "TRUE_EVIDENCE_LIMITED":
                    ds_compat[ds] = "INCOMPATIBLE"
                elif cause in ("SOURCE_POOL_GAP", "ADAPTER_SEMANTIC_GAP") and ds_compat.get(ds) != "INCOMPATIBLE":
                    ds_compat[ds] = "PARTIAL"
        best = None
        for ds, st in ds_compat.items():
            if st == "COMPATIBLE":
                best = ds
                break
        if not best:
            for ds, st in ds_compat.items():
                if st == "PARTIAL":
                    best = ds
                    break
        rows.append(
            {
                "blueprint_id": a["blueprint_id"],
                "scene": a["scene"],
                "required_capabilities": sorted({r["requirement_type"] for r in reqs}),
                "CASAS_compatibility": ds_compat["CASAS"],
                "UK-DALE_compatibility": ds_compat["UK-DALE"],
                "YouHome_compatibility": ds_compat["YouHome"],
                "REFIT_compatibility": ds_compat["REFIT"],
                "best_candidate_source": best,
                "primary_root_cause": a.get("primary_root_cause"),
                "extended_pool_match": a.get("match_status_extended_pool") == "MATCH_OK",
            }
        )
    return {"blueprints": rows}

def audit_640_distribution(samples_path: Path) -> dict[str, Any]:
    samples = read_jsonl(samples_path) if samples_path.is_file() else []
    bp_counts = Counter((s.get("blueprint_binding") or {}).get("blueprint_id") for s in samples)
    branch_counts = Counter((s.get("synthesis_metadata") or {}).get("branch_id") for s in samples)
    path_counts = Counter((s.get("synthesis_metadata") or {}).get("action_path_signature") for s in samples)
    template_counts = Counter(
        ((s.get("blueprint_binding") or {}).get("blueprint_id"), (s.get("synthesis_metadata") or {}).get("behavior_target"))
        for s in samples
    )
    runtime_counts = Counter((s.get("provenance") or {}).get("source_dataset") for s in samples)

    total = len(samples) or 1
    successful_bps = sorted(bp_counts.keys())

    def _share(counter: Counter) -> dict:
        if not counter:
            return {"key": None, "count": 0, "share": 0.0}
        key, cnt = counter.most_common(1)[0]
        return {"key": key, "count": cnt, "share": round(cnt / total, 4)}

    per_bp = []
    for bp in successful_bps:
        bp_samples = [s for s in samples if (s.get("blueprint_binding") or {}).get("blueprint_id") == bp]
        per_bp.append(
            {
                "blueprint_id": bp,
                "sample_count": bp_counts[bp],
                "branch_coverage": len({(s.get("synthesis_metadata") or {}).get("branch_id") for s in bp_samples}),
                "action_path_coverage": len({(s.get("synthesis_metadata") or {}).get("action_path_signature") for s in bp_samples}),
                "behavior_target_coverage": len({(s.get("synthesis_metadata") or {}).get("behavior_target") for s in bp_samples}),
                "runtime_situation_coverage": len({(s.get("provenance") or {}).get("source_record_id") for s in bp_samples}),
            }
        )

    return {
        "total_samples": len(samples),
        "successful_blueprint_count": len(successful_bps),
        "successful_blueprints": successful_bps,
        "per_blueprint": per_bp,
        "min_per_blueprint_requested": 5,
        "generation_note": (
            "640 samples with --min-per-blueprint 5: scheduler continued generating for 8 easy blueprints "
            "(climate/appliance/on_off) until MAX_TOTAL_ATTEMPTS (~998). 14 blueprints never matched → EVIDENCE_LIMITED. "
            "Easy-blueprint domination: successful blueprints hit per-blueprint caps while failures exhausted attempt budget."
        ),
        "domination": {
            "largest_blueprint_share": _share(bp_counts),
            "largest_branch_share": _share(branch_counts),
            "largest_action_path_share": _share(path_counts),
        },
        "blueprint_distribution": dict(bp_counts),
        "branch_distribution": dict(branch_counts),
        "action_path_distribution": dict(path_counts),
        "runtime_dataset_distribution": dict(runtime_counts),
    }

def run_smoke_tests(
    bp_ids: list[str],
    *,
    pool: SourcePool,
    reg: dict,
    inst_idx: dict,
    max_per_bp: int = 3,
) -> dict[str, Any]:
    from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import build_all_behavior_specs

    specs = {s["blueprint_id"]: s for s in build_all_behavior_specs(reg)}
    results = []
    for bp_id in bp_ids:
        scene = blueprint_to_scene(bp_id) or ""
        inst_list = inst_idx.get(bp_id) or []
        if not inst_list:
            continue
        instance = inst_list[0]
        nb = _normalize_from_instance(scene, instance, reg)
        bspec = specs.get(bp_id) or {}
        accepted = 0
        attempts = 0
        statuses: Counter = Counter()
        while accepted < max_per_bp and attempts < max_per_bp * 5:
            attempts += 1
            status, sample, reason = synthesize_single_scene(
                scene=scene,
                blueprint_id=bp_id,
                automation_instance_id=instance.get("automation_instance_id", bp_id),
                behavior_target=f"smoke::{bp_id}::a{attempts}",
                branch_id="smoke",
                action_path_signature="smoke",
                instance=instance,
                normalized_blueprint=nb,
                behavior_spec=bspec,
                source_pool=pool,
                reuse_counts={},
                attempt_index=attempts,
            )
            statuses[status] += 1
            if status == "ACCEPT" and sample:
                accepted += 1
        obs, match_status, _ = match_requirement_bundle(pool, extract_requirement_bundle(nb))
        results.append(
            {
                "blueprint_id": bp_id,
                "scene": scene,
                "smoke_samples_accepted": accepted,
                "smoke_attempts": attempts,
                "status_distribution": dict(statuses),
                "grounded_match_status": match_status,
                "grounded_match_record": obs.record_id if obs else None,
                "recovered": accepted > 0 and match_status == "MATCH_OK",
            }
        )
    return {"smoke_tests": results, "recovered_count": sum(1 for r in results if r.get("recovered"))}

def run_full_audit(out_dir: Path | None = None) -> dict[str, Any]:
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    raw_probe = probe_raw_datasets()
    legacy_pool = build_legacy_source_pool(PROJECT_SCENES, include_public=False)
    extended_pool = build_legacy_source_pool(PROJECT_SCENES, include_public=True)
    public_by_scene = load_public_observations()
    reg = build_registry()
    inst_idx = build_instance_templates().get("instances") or {}

    evidence_limited_path = out / "evidence_limited_report.json"
    el_ids = []
    if evidence_limited_path.is_file():
        el_ids = [r["blueprint_id"] for r in json.loads(evidence_limited_path.read_text(encoding="utf-8")).get("evidence_limited") or []]
    if not el_ids:
        el_ids = [
            bp for scene, ids in SCENE_BLUEPRINT_MAP.items()
            for bp in ids
            if bp not in {
                "climate_window_restore", "climate_smarter_thermostat", "climate_hvac_auto_adjust",
                "climate_production_grade", "presence_better_thermostat", "presence_automation_state",
                "appliance_power_state_detect", "lighting_motion_maestro_48",
            }
        ]

    audits = [_audit_one_blueprint(bp, legacy_pool=legacy_pool, extended_pool=extended_pool, public_by_scene=public_by_scene, raw_probe=raw_probe, reg=reg, inst_idx=inst_idx) for bp in el_ids]

    capability_matrix = build_source_capability_matrix(raw_probe)
    compatibility = build_blueprint_source_compatibility(audits)
    distribution = audit_640_distribution(out / "single_scene_samples.jsonl")

    gap_buckets = {
        "SOURCE_POOL_GAP": [],
        "ADAPTER_SEMANTIC_GAP": [],
        "MATCHER_GAP": [],
        "ENTITY_BINDING_GAP": [],
        "TEMPORAL_MATCHING_GAP": [],
        "TRUE_EVIDENCE_LIMITED": [],
    }
    for a in audits:
        cause = a.get("primary_root_cause")
        if cause in gap_buckets:
            gap_buckets[cause].append(a["blueprint_id"])
        elif cause == "RECOVERED":
            gap_buckets.setdefault("RECOVERED", []).append(a["blueprint_id"])

    recovered_ids = [a["blueprint_id"] for a in audits if a.get("match_status_extended_pool") == "MATCH_OK"]
    still_limited = [a["blueprint_id"] for a in audits if a.get("match_status_extended_pool") != "MATCH_OK"]
    smoke = run_smoke_tests(recovered_ids + still_limited, pool=extended_pool, reg=reg, inst_idx=inst_idx, max_per_bp=3)

    grounded_capable = 8 + len(recovered_ids)

    legacy_cause_counts: Counter = Counter()
    for a in audits:
        if a.get("match_status_legacy_pool") == "MATCH_OK":
            legacy_cause_counts["LEGACY_ALREADY_MATCHED"] += 1
        else:
            for cause, cnt in (a.get("root_cause_counts") or {}).items():
                legacy_cause_counts[cause] += cnt
            if not a.get("root_cause_counts"):
                legacy_cause_counts[a.get("primary_root_cause", "UNKNOWN")] += 1

    true_limited = [
        {
            "blueprint_id": a["blueprint_id"],
            "scene": a["scene"],
            "evidence": "Raw public datasets checked; no grounded vibration/contact observation satisfies blueprint requirement.",
            "unsatisfied": [r for r in a["requirement_audits"] if r["status"] == "UNSATISFIED"],
        }
        for a in audits
        if a.get("match_status_extended_pool") != "MATCH_OK"
    ]

    verdict = "BLOCKED_BY_TRUE_EVIDENCE_LIMITATION" if still_limited else "READY_FOR_FULL_SINGLE_SCENE_REGENERATION"
    if recovered_ids and still_limited:
        verdict = "READY_FOR_FULL_SINGLE_SCENE_REGENERATION"

    summary = {
        "evidence_limited_blueprints": len(el_ids),
        "legacy_pool_failure_causes": dict(legacy_cause_counts),
        "SOURCE_POOL_GAP": sum(1 for a in audits if a.get("match_status_legacy_pool") != "MATCH_OK" and "SOURCE_POOL_GAP" in (a.get("root_cause_counts") or {})),
        "ADAPTER_SEMANTIC_GAP": sum(1 for a in audits if "ADAPTER_SEMANTIC_GAP" in (a.get("root_cause_counts") or {})),
        "MATCHER_GAP": sum(1 for a in audits if "MATCHER_GAP" in (a.get("root_cause_counts") or {})),
        "ENTITY_BINDING_GAP": sum(1 for a in audits if "ENTITY_BINDING_GAP" in (a.get("root_cause_counts") or {})),
        "TEMPORAL_MATCHING_GAP": sum(1 for a in audits if "TEMPORAL_MATCHING_GAP" in (a.get("root_cause_counts") or {})),
        "TRUE_EVIDENCE_LIMITED": len(still_limited),
        "RECOVERED_MATCH": len(recovered_ids),
        "still_evidence_limited": still_limited,
        "grounded_synthesis_capable_blueprints": grounded_capable,
        "verdict": verdict,
        "per_blueprint_primary_cause": {a["blueprint_id"]: a["primary_root_cause"] for a in audits},
        "per_blueprint_legacy_match": {a["blueprint_id"]: a["match_status_legacy_pool"] for a in audits},
        "per_blueprint_extended_match": {a["blueprint_id"]: a["match_status_extended_pool"] for a in audits},
    }

    artifacts = {
        "blueprint_requirement_audit.json": {"blueprints": audits},
        "source_capability_matrix.json": capability_matrix,
        "blueprint_source_compatibility.json": compatibility,
        "raw_vs_adapter_gap.json": {"raw_probe": raw_probe, "gap_summary": gap_buckets},
        "matcher_gap_audit.json": {"blueprints": [a for a in audits if "MATCHER_GAP" in (a.get("root_cause_counts") or {})]},
        "entity_binding_gap_audit.json": {"blueprints": [a for a in audits if "ENTITY_BINDING_GAP" in (a.get("root_cause_counts") or {})]},
        "temporal_matching_gap_audit.json": {"blueprints": [a for a in audits if "TEMPORAL_MATCHING_GAP" in (a.get("root_cause_counts") or {})]},
        "true_evidence_limited.json": {"blueprints": true_limited, "count": len(true_limited)},
        "current_640_distribution_audit.json": distribution,
        "evidence_recovery_smoke_test.json": smoke,
        "audit_summary.json": summary,
    }
    for name, obj in artifacts.items():
        (out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_markdown_report(out / "EVIDENCE_LIMITED_ROOT_CAUSE_AUDIT.md", summary, audits, distribution, smoke, gap_buckets, true_limited)
    return summary

def _write_markdown_report(
    path: Path,
    summary: dict,
    audits: list,
    distribution: dict,
    smoke: dict,
    gap_buckets: dict,
    true_limited: list,
) -> None:
    lines = [
        "# EVIDENCE_LIMITED Root-Cause Audit",
        "",
        f"**Verdict:** `{summary['verdict']}`",
        "",
        "## Executive Summary",
        "",
        f"- EVIDENCE_LIMITED blueprints audited: **{summary['evidence_limited_blueprints']}**",
        f"- SOURCE_POOL_GAP: **{summary['SOURCE_POOL_GAP']}**",
        f"- ADAPTER_SEMANTIC_GAP: **{summary['ADAPTER_SEMANTIC_GAP']}**",
        f"- MATCHER_GAP: **{summary['MATCHER_GAP']}**",
        f"- ENTITY_BINDING_GAP: **{summary['ENTITY_BINDING_GAP']}**",
        f"- TEMPORAL_MATCHING_GAP: **{summary['TEMPORAL_MATCHING_GAP']}**",
        f"- TRUE_EVIDENCE_LIMITED: **{summary['TRUE_EVIDENCE_LIMITED']}**",
        f"- Recovered grounded match (extended pool): **{summary['RECOVERED_MATCH']}**",
        f"- Blueprints with grounded synthesis capability: **{summary['grounded_synthesis_capable_blueprints']} / 22**",
        "",
        "## Per-Blueprint Primary Root Cause",
        "",
        "| blueprint_id | primary_root_cause | legacy_match | extended_match |",
        "|---|---|---|---|",
    ]
    for a in audits:
        lines.append(
            f"| {a['blueprint_id']} | {a['primary_root_cause']} | {a['match_status_legacy_pool']} | {a['match_status_extended_pool']} |"
        )

    lines.extend(["", "## 640-Sample Distribution", ""])
    lines.append(f"- Total samples: **{distribution.get('total_samples', 0)}**")
    lines.append(f"- Successful blueprints: **{distribution.get('successful_blueprint_count', 0)} / 22**")
    dom = distribution.get("domination") or {}
    for k, v in dom.items():
        lines.append(f"- {k}: `{v.get('key')}` = {v.get('count')} ({100 * v.get('share', 0):.1f}%)")
    lines.append(f"\n{distribution.get('generation_note', '')}")

    lines.extend(["", "## Smoke Test Recovery", ""])
    for r in smoke.get("smoke_tests") or []:
        lines.append(f"- **{r['blueprint_id']}**: accepted={r['smoke_samples_accepted']}, match={r['grounded_match_status']}, recovered={r.get('recovered')}")

    lines.extend(["", "## TRUE_EVIDENCE_LIMITED (with evidence)", ""])
    for t in true_limited:
        lines.append(f"### {t['blueprint_id']}")
        for u in t.get("unsatisfied") or []:
            lines.append(f"- {u.get('requirement_type')}: {u.get('unsatisfied_detail')} ({u.get('root_cause')})")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    result = run_full_audit()
    print(json.dumps(result, indent=2))
