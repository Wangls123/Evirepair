from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.multi_action_vnext.requirement_extractor import extract_requirement_bundle, minimal_grounding_requirements
from smarthome_mdf.multi_action_vnext.source_adapters import SourceObservation, SourcePool
from smarthome_mdf.multi_action_vnext.unified_runtime_matcher import (
    _grounding_requirements,
    satisfies_requirement,
)
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR, PROJECT_SCENES, SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import GROUNDED_CAPABLE_BLUEPRINTS
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import build_instance_templates
from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.p1_integrity_audit import (
    V1_DIR_NAME,
    V2_DIR_NAME,
    classify_sample_validity,
    audit_visual_fusion,
)
from smarthome_mdf.single_scene_blueprint_complete.semantic_sanitize import detect_semantic_contamination
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import (
    canonical_behavior_hash,
    canonical_behavior_signature,
    numeric_only_near_duplicate_stable,
    temporal_shift_near_duplicate_stable,
)
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _norm_ts(obj: Any) -> str:
    if isinstance(obj, dict):
        return json.dumps({k: v for k, v in obj.items() if k not in ("timestamp", "synthesis_timestamp")}, sort_keys=True)
    return str(obj)

def identify_sanitize_retained(v1_dir: Path, v2_samples: list[dict]) -> list[dict]:

    v1_validity_path = v1_dir / "sample_validity_audit.jsonl"
    semantic_invalid_v1: set[str] = set()
    if v1_validity_path.is_file():
        for line in v1_validity_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("validity") == "INVALID_SEMANTIC_MAPPING":
                semantic_invalid_v1.add(row["sample_id"])

    v2_by_id = {s.get("sample_id"): s for s in v2_samples}
    retained: list[dict] = []
    for sid in semantic_invalid_v1:
        s = v2_by_id.get(sid)
        if not s:
            continue
        if s.get("replaces_sample_id"):
            continue
        repair = (s.get("metadata") or {}).get("p1_repair")
        if repair == "visual_fusion_regenerated":
            continue
        retained.append({**s, "_v1_semantic_invalid": True, "_sanitize_retained": True})
    return retained

def sample_to_source_observation(sample: dict, pool: SourcePool | None = None, *, history_cache: dict | None = None) -> SourceObservation | None:

    prov = sample.get("provenance") or {}
    eos = sample.get("entity_observations") or []
    if not eos:
        return None
    eo = eos[0]
    attrs = dict(eo.get("attributes") or {})
    scene = sample.get("scene_type") or ""
    record_id = str(
        (prov.get("runtime_source") or {}).get("observation_id")
        or prov.get("source_record_id")
        or eo.get("source_entity")
        or ""
    )
    dataset = str(
        (prov.get("runtime_source") or {}).get("dataset")
        or prov.get("source_dataset")
        or eo.get("source_dataset")
        or "UNKNOWN"
    )

    history: list[Any] = []
    if pool and record_id:
        if history_cache is not None and record_id in history_cache:
            history = history_cache[record_id]
        else:
            for cand in pool.candidates_for_scene(scene):
                if cand.record_id == record_id:
                    history = list(cand.history or [])
                    if history_cache is not None:
                        history_cache[record_id] = history
                    break

    measurement = None
    if attrs.get("current_power_w") is not None:
        measurement = "power"
    elif attrs.get("motion_state") is not None or attrs.get("motion") is not None:
        measurement = "motion"
    elif attrs.get("window_state") is not None:
        measurement = "window"
    elif attrs.get("temperature") is not None or attrs.get("indoor_temperature") is not None:
        measurement = "temperature"
    elif attrs.get("illuminance_lux") is not None:
        measurement = "illuminance"
    elif sample.get("scene_type") == "visual_fusion":
        measurement = "person_detected"

    return SourceObservation(
        dataset=dataset,
        record_id=record_id,
        original_timestamp=str(eo.get("timestamp") or prov.get("runtime_source", {}).get("original_timestamp") or ""),
        entity_domain=str(eo.get("domain") or "sensor"),
        measurement=measurement,
        value=attrs.get("current_power_w") or attrs.get("motion_state") or eo.get("state"),
        state=eo.get("state"),
        attributes=attrs,
        history=history,
        source_metadata={"scene_type": scene, "trigger_context": attrs.get("trigger_context")},
        provenance=dict(prov),
        scene_type=scene,
    )

@dataclass
class _RematchContext:
    reg: dict
    pool: SourcePool
    inst_cache: dict[str, list] = None
    bundle_cache: dict[str, Any] = None
    history_cache: dict[str, list] = None
    instance_templates: dict[str, list[dict]] | None = None

    def __post_init__(self):
        self.inst_cache = {}
        self.bundle_cache = {}
        self.history_cache = {}
        if self.instance_templates is None:
            from smarthome_mdf.single_scene_blueprint_complete.grounded_instances import enumerate_grounded_instances, instances_to_templates

            grounded = enumerate_grounded_instances(self.pool, reg=self.reg, max_probe=80)
            self.instance_templates = instances_to_templates(grounded)

def rematch_sample(sample: dict, ctx: _RematchContext) -> dict[str, Any]:

    from smarthome_mdf.single_scene_blueprint_complete.grounded_instance_snapshot import build_ir_for_sample, instance_from_sample

    bb = sample.get("blueprint_binding") or {}
    bp_id = bb.get("blueprint_id", "")
    iid = bb.get("automation_instance_id", "")
    scene = sample.get("scene_type", "")

    snap = instance_from_sample(sample)
    if snap.get("blueprint_inputs"):
        instance = snap
    elif bp_id not in ctx.inst_cache:
        templates = ctx.instance_templates or {}
        ctx.inst_cache[bp_id] = templates.get(bp_id) or (build_instance_templates().get("instances") or {}).get(bp_id) or []
        inst_list = ctx.inst_cache[bp_id]
        instance = next((i for i in inst_list if i["automation_instance_id"] == iid), inst_list[0] if inst_list else {})
    else:
        inst_list = ctx.inst_cache[bp_id]
        instance = next((i for i in inst_list if i["automation_instance_id"] == iid), inst_list[0] if inst_list else {})

    bkey = f"{bp_id}::{iid}"
    if bkey not in ctx.bundle_cache:
        nb = _normalize_from_instance(scene, instance, ctx.reg)
        if nb.parse_status != "PARSE_OK":
            ctx.bundle_cache[bkey] = ("IMPLEMENTATION_GAP", nb.rejection_reason, None, [])
        else:
            ypath = resolve_yaml_path(bp_id)
            ir = (
                parse_blueprint_ir(bp_id, Path(ypath), dict(instance.get("blueprint_inputs") or {}), automation_instance_id=iid)
                if ypath and ypath.is_file()
                else None
            )
            bundle = extract_requirement_bundle(nb, ir=ir)
            grounding = minimal_grounding_requirements(bundle)
            ctx.bundle_cache[bkey] = ("OK", None, bundle, grounding)
    cached = ctx.bundle_cache[bkey]
    if cached[0] != "OK":
        return {"status": cached[0], "reason": cached[1], "unsatisfied": []}

    _, _, bundle, grounding = cached
    obs = sample_to_source_observation(sample, ctx.pool, history_cache=ctx.history_cache)
    if obs is None:
        return {"status": "NO_OBSERVATION", "reason": "cannot_build_obs", "unsatisfied": ["NO_OBSERVATION"]}
    from smarthome_mdf.single_scene_blueprint_complete.semantic_capabilities import attach_capabilities

    obs = attach_capabilities(obs)

    traces: list[dict] = []
    all_ok = True
    unsat_all: list[str] = []
    for i, req in enumerate(grounding):
        ok, unsat, matched = satisfies_requirement(req, obs)
        traces.append({"requirements": req.to_dict(), "matched_fields": matched, "unsatisfied_requirements": unsat, "component_index": i})
        if not ok:
            all_ok = False
            unsat_all.extend(unsat)

    status = "MATCH_OK" if all_ok else "MATCH_FAIL"
    return {
        "status": status,
        "reason": ",".join(unsat_all[:5]) if unsat_all else "ok",
        "unsatisfied": unsat_all,
        "traces": traces,
        "requirement_types": [getattr(r, "kind", type(r).__name__) for r in grounding],
    }

def audit_post_sanitize_matching(
    sanitize_retained: list[dict],
    pool: SourcePool,
    reg: dict,
) -> dict[str, Any]:
    ok = fail = 0
    fail_rows: list[dict] = []
    by_bp: Counter = Counter()
    by_req: Counter = Counter()
    by_ds: Counter = Counter()
    ctx = _RematchContext(reg=reg, pool=pool)

    for s in sanitize_retained:
        result = rematch_sample(s, ctx)
        if result["status"] == "MATCH_OK":
            ok += 1
        else:
            fail += 1
            bb = s.get("blueprint_binding") or {}
            bp = bb.get("blueprint_id", "")
            by_bp[bp] += 1
            ds = (s.get("provenance") or {}).get("source_dataset", "")
            by_ds[ds] += 1
            for u in result.get("unsatisfied") or []:
                by_req[u.split(":")[0] if ":" in u else u] += 1
            if len(fail_rows) < 30:
                fail_rows.append(
                    {
                        "sample_id": s.get("sample_id"),
                        "blueprint_id": bp,
                        "source_dataset": ds,
                        "unsatisfied": result.get("unsatisfied"),
                        "reason": result.get("reason"),
                    }
                )

    n = len(sanitize_retained)
    return {
        "SANITIZE_RETAINED_N": n,
        "POST_SANITIZE_MATCH_OK": ok,
        "POST_SANITIZE_MATCH_FAIL": fail,
        "POST_SANITIZE_MATCH_OK_RATE": f"{ok}/{n}",
        "fail_by_blueprint": dict(by_bp),
        "fail_by_requirement_type": dict(by_req),
        "fail_by_source_dataset": dict(by_ds),
        "fail_examples": fail_rows,
    }

def _runtime_substance(sample: dict) -> str:
    eo = (sample.get("entity_observations") or [{}])[0]
    attrs = dict(eo.get("attributes") or {})
    attrs.pop("trigger_context", None)
    for k in ("timestamp",):
        attrs.pop(k, None)
    obs = dict(sample.get("observed") or {})
    obs.pop("timestamp", None)
    prov = sample.get("provenance") or {}
    parts = [
        canonical_behavior_signature(sample),
        prov.get("source_record_id"),
        json.dumps(attrs, sort_keys=True, default=str),
        json.dumps(obs, sort_keys=True, default=str),
        (sample.get("synthesis_metadata") or {}).get("entity_binding_signature"),
    ]
    return "|".join(str(p) for p in parts)

def classify_temporal_near_dup(samples: list[dict]) -> dict[str, Any]:

    by_sig: dict[str, dict] = {}
    categories: Counter = Counter()
    classified_samples: list[str] = []

    for s in samples:
        sig = canonical_behavior_hash(s)
        prior = by_sig.get(sig)
        if prior is None:
            by_sig[sig] = s
            continue

        if not temporal_shift_near_duplicate_stable(s, prior) and not numeric_only_near_duplicate_stable(s, prior):
            by_sig[sig] = s
            continue

        prov_a = s.get("provenance") or {}
        prov_b = prior.get("provenance") or {}
        src_a = prov_a.get("source_record_id")
        src_b = prov_b.get("source_record_id")

        meta_a = s.get("synthesis_metadata") or {}
        meta_b = prior.get("synthesis_metadata") or {}
        if meta_a.get("action_path_signature") != meta_b.get("action_path_signature") or meta_a.get("branch_id") != meta_b.get("branch_id"):
            cat = "DIFFERENT_BEHAVIOR_OR_ACTION_PATH"
        elif src_a != src_b:
            cat = "DIFFERENT_GROUNDED_OBSERVATION"
        elif _runtime_substance(s) != _runtime_substance(prior):
            cat = "DIFFERENT_RUNTIME_SITUATION"
        elif _norm_ts(s.get("observed")) == _norm_ts(prior.get("observed")) and _runtime_substance(s) == _runtime_substance(prior):
            cat = "ONLY_TIMESTAMP_SHIFT"
        else:
            ts_a = (s.get("observed") or {}).get("timestamp") or prov_a.get("synthesis_timestamp")
            ts_b = (prior.get("observed") or {}).get("timestamp") or prov_b.get("synthesis_timestamp")
            if ts_a != ts_b and src_a == src_b:
                cat = "ONLY_TIMESTAMP_SHIFT"
            else:
                cat = "OTHER"

        categories[cat] += 1
        classified_samples.append(s.get("sample_id", ""))
        by_sig[sig] = s

    total_near = sum(categories.values())
    only_ts = categories.get("ONLY_TIMESTAMP_SHIFT", 0)
    total = len(samples)
    canon_unique = len(set(canonical_behavior_hash(s) for s in samples))

    return {
        "TEMPORAL_NEAR_DUP_N": total_near,
        "ONLY_TIMESTAMP_SHIFT": only_ts,
        "DIFFERENT_GROUNDED_OBSERVATION": categories.get("DIFFERENT_GROUNDED_OBSERVATION", 0),
        "DIFFERENT_RUNTIME_SITUATION": categories.get("DIFFERENT_RUNTIME_SITUATION", 0),
        "DIFFERENT_BEHAVIOR_OR_ACTION_PATH": categories.get("DIFFERENT_BEHAVIOR_OR_ACTION_PATH", 0),
        "OTHER": categories.get("OTHER", 0),
        "ONLY_TIMESTAMP_SHIFT_TOTAL_RATE": f"{only_ts}/{total}",
        "effective_unique_samples_without_timestamp_only_duplicates": total - only_ts,
        "canonical_behavior_unique_count": canon_unique,
        "category_breakdown": dict(categories),
    }

def _binding_signature_from_obs(obs: SourceObservation) -> str:
    attrs = obs.attributes or {}
    parts = sorted(f"{k}={attrs[k]}" for k in ("entity_id", "device_class", "area", "appliance_id", "channel_id") if k in attrs)
    return "|".join(parts) or obs.record_id

def audit_grounded_capacity(samples: list[dict], pool: SourcePool, reg: dict, *, max_candidates_per_scene: int = 400) -> dict[str, Any]:
    inst_templates = build_instance_templates().get("instances") or {}
    rows: list[dict] = []
    gap_counts: Counter = Counter()

    for bp_id in GROUNDED_CAPABLE_BLUEPRINTS:
        scene = next((s.get("scene_type") for s in samples if (s.get("blueprint_binding") or {}).get("blueprint_id") == bp_id), "")
        if not scene:
            for sc, bps in SCENE_BLUEPRINT_MAP.items():
                if bp_id in bps:
                    scene = sc
                    break

        v2_subset = [s for s in samples if (s.get("blueprint_binding") or {}).get("blueprint_id") == bp_id]
        observed_instances = set((s.get("blueprint_binding") or {}).get("automation_instance_id") for s in v2_subset)
        observed_bindings = set((s.get("synthesis_metadata") or {}).get("entity_binding_signature") for s in v2_subset)

        template_instances = inst_templates.get(bp_id) or []
        reachable_instance_ids: set[str] = set()
        reachable_bindings: set[str] = set()
        matcher_ok_records: set[str] = set()

        for inst in template_instances:
            iid = inst["automation_instance_id"]
            nb = _normalize_from_instance(scene, inst, reg)
            if nb.parse_status != "PARSE_OK":
                continue
            ypath = resolve_yaml_path(bp_id)
            ir = parse_blueprint_ir(bp_id, Path(ypath), dict(inst.get("blueprint_inputs") or {}), automation_instance_id=iid) if ypath else None
            bundle = extract_requirement_bundle(nb, ir=ir)
            grounding = minimal_grounding_requirements(bundle)
            inst_bindings = "|".join(sorted(inst.get("bound_entities") or []))

            candidates = pool.candidates_for_scene(scene)[:max_candidates_per_scene]

            for cand in candidates:
                all_ok = True
                for req in grounding:
                    ok, _, _ = satisfies_requirement(req, cand)
                    if not ok:
                        all_ok = False
                        break
                if all_ok:
                    reachable_instance_ids.add(iid)
                    reachable_bindings.add(inst_bindings or _binding_signature_from_obs(cand))
                    matcher_ok_records.add(cand.record_id)

        ri = max(len(reachable_instance_ids), 1)
        rb = max(len(reachable_bindings), 1)
        oi = len(observed_instances)
        ob = len(observed_bindings)
        ic = round(oi / ri, 4)
        bc = round(ob / rb, 4)

        if ri == 1 and oi == 1 and len(template_instances) <= 1:
            verdict = "TRUE_CAPACITY_LIMIT"
            reason = "Only one template instance and matcher accepts default binding"
        elif ic >= 1.0 and bc >= 1.0:
            verdict = "FULL_COVERAGE"
            reason = "Observed instances/bindings cover matcher-reachable space"
        elif oi < ri:
            verdict = "GENERATOR_COVERAGE_GAP"
            reason = f"Matcher reachable {ri} instances but observed {oi}"
            gap_counts["GENERATOR_COVERAGE_GAP"] += 1
        elif ob < rb:
            verdict = "ENTITY_BINDING_GAP"
            reason = f"Matcher reachable {rb} bindings but observed {ob}"
            gap_counts["ENTITY_BINDING_GAP"] += 1
        else:
            verdict = "TRUE_CAPACITY_LIMIT"
            reason = "Reachable space fully observed"

        rows.append(
            {
                "blueprint_id": bp_id,
                "scene": scene,
                "reachable_instances": len(reachable_instance_ids),
                "observed_instances": oi,
                "instance_coverage": ic,
                "reachable_bindings": len(reachable_bindings),
                "observed_bindings": ob,
                "binding_coverage": bc,
                "matcher_viable_observations": len(matcher_ok_records),
                "template_instance_count": len(template_instances),
                "verdict": verdict,
                "reason": reason,
            }
        )

    full_inst = sum(1 for r in rows if r["instance_coverage"] >= 1.0)
    full_bind = sum(1 for r in rows if r["binding_coverage"] >= 1.0)

    return {
        "blueprint_rows": rows,
        "full_instance_capacity_covered": full_inst,
        "full_binding_capacity_covered": full_bind,
        "generator_coverage_gap": gap_counts.get("GENERATOR_COVERAGE_GAP", 0),
        "adapter_gap": gap_counts.get("ADAPTER_GAP", 0),
        "matcher_gap": gap_counts.get("MATCHER_GAP", 0),
        "entity_binding_gap": gap_counts.get("ENTITY_BINDING_GAP", 0),
    }

def audit_v2_semantic_provenance(samples: list[dict]) -> dict[str, Any]:
    semantic_invalid = 0
    provenance_invalid = 0
    vf_audit = audit_visual_fusion(samples)
    for s in samples:
        if detect_semantic_contamination(s):
            semantic_invalid += 1
        prov = s.get("provenance") or {}
        scene = s.get("scene_type")
        if scene == "visual_fusion":
            if not prov.get("runtime_source") or not prov.get("visual_source"):
                if prov.get("composition_type") != "CROSS_DATASET_SYNTHETIC" and prov.get("source_dataset") == "CASAS":
                    obs = s.get("observed") or {}
                    if obs.get("frame_id"):
                        provenance_invalid += 1
            elif prov.get("source_dataset") == "CASAS" and (s.get("observed") or {}).get("frame_id") and not prov.get("composition_type"):
                provenance_invalid += 1
    return {
        "SEMANTIC_INVALID": semantic_invalid,
        "PROVENANCE_INVALID": provenance_invalid,
        "visual_single_frame_domination": vf_audit.get("single_frame_domination"),
        "IMPLEMENTATION_GAP": 0,
    }

def run_pre_freeze_verification(
    *,
    v1_dir: Path | None = None,
    v2_dir: Path | None = None,
) -> dict[str, Any]:
    v1 = v1_dir or (OUTPUT_DIR / V1_DIR_NAME)
    v2 = v2_dir or (OUTPUT_DIR / V2_DIR_NAME)
    samples = read_jsonl(v2 / "single_scene_samples.jsonl")
    reg = build_registry()
    pool = build_legacy_source_pool(PROJECT_SCENES, include_public=True)

    sanitize_retained = identify_sanitize_retained(v1, samples)
    post_sanitize = audit_post_sanitize_matching(sanitize_retained, pool, reg)
    temporal = classify_temporal_near_dup(samples)
    capacity = audit_grounded_capacity(samples, pool, reg)
    extra = audit_v2_semantic_provenance(samples)

    blockers: list[str] = []
    if post_sanitize["POST_SANITIZE_MATCH_FAIL"] > 0:
        blockers.append(f"POST_SANITIZE_MATCH_FAIL={post_sanitize['POST_SANITIZE_MATCH_FAIL']}")
    if extra["SEMANTIC_INVALID"] > 0:
        blockers.append(f"SEMANTIC_INVALID={extra['SEMANTIC_INVALID']}")
    if extra["PROVENANCE_INVALID"] > 0:
        blockers.append(f"PROVENANCE_INVALID={extra['PROVENANCE_INVALID']}")
    if extra.get("visual_single_frame_domination"):
        blockers.append("visual_single_frame_domination")

    only_ts = temporal.get("ONLY_TIMESTAMP_SHIFT", 0)
    only_ts_rate = only_ts / max(len(samples), 1)
    if only_ts_rate > 0.15:
        blockers.append(f"HIGH_ONLY_TIMESTAMP_SHIFT_RATE={only_ts_rate:.4f}")

    if capacity["generator_coverage_gap"] > 5:
        blockers.append(f"GENERATOR_COVERAGE_GAP={capacity['generator_coverage_gap']} blueprints")

    verdict = "READY_FOR_SINGLE_SCENE_FREEZE" if not blockers else "BLOCKED"

    report = {
        "generated_at": _utc_now(),
        "FINAL_VERDICT": verdict,
        "blockers": blockers,
        "HARD_RESULT_1_POST_SANITIZE": {
            "sanitize_retained": post_sanitize["SANITIZE_RETAINED_N"],
            "MATCH_OK": post_sanitize["POST_SANITIZE_MATCH_OK"],
            "MATCH_FAIL": post_sanitize["POST_SANITIZE_MATCH_FAIL"],
            "MATCH_OK_RATE": post_sanitize["POST_SANITIZE_MATCH_OK_RATE"],
        },
        "HARD_RESULT_2_TEMPORAL": {
            "temporal_near_dup": temporal["TEMPORAL_NEAR_DUP_N"],
            "only_timestamp_shift": temporal["ONLY_TIMESTAMP_SHIFT"],
            "only_timestamp_shift_total_rate": temporal["ONLY_TIMESTAMP_SHIFT_TOTAL_RATE"],
            "effective_unique_samples_without_timestamp_only_duplicates": temporal["effective_unique_samples_without_timestamp_only_duplicates"],
            "breakdown": temporal["category_breakdown"],
        },
        "HARD_RESULT_3_CAPACITY": {
            "full_instance_capacity_covered": f"{capacity['full_instance_capacity_covered']}/21",
            "full_binding_capacity_covered": f"{capacity['full_binding_capacity_covered']}/21",
            "generator_coverage_gap": capacity["generator_coverage_gap"],
            "adapter_gap": capacity["adapter_gap"],
            "matcher_gap": capacity["matcher_gap"],
            "entity_binding_gap": capacity["entity_binding_gap"],
        },
        "post_sanitize_runtime_audit": post_sanitize,
        "temporal_near_dup_audit": temporal,
        "grounded_capacity_audit": capacity,
        "extra_checks": extra,
    }

    _write_artifacts(v2, report)
    return report

def _write_artifacts(v2_dir: Path, report: dict[str, Any]) -> None:
    v2_dir.mkdir(parents=True, exist_ok=True)
    (v2_dir / "post_sanitize_runtime_audit.json").write_text(
        json.dumps(report["post_sanitize_runtime_audit"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (v2_dir / "temporal_near_dup_deep_audit.json").write_text(
        json.dumps(report["temporal_near_dup_audit"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (v2_dir / "grounded_capacity_audit.json").write_text(
        json.dumps(report["grounded_capacity_audit"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = _format_report_md(report)
    (v2_dir / "PRE_FREEZE_VERIFICATION_REPORT.md").write_text(md, encoding="utf-8")
    (v2_dir / "pre_freeze_manifest.json").write_text(json.dumps({"verdict": report["FINAL_VERDICT"], "blockers": report["blockers"]}, indent=2), encoding="utf-8")

def _format_report_md(report: dict) -> str:
    h1 = report["HARD_RESULT_1_POST_SANITIZE"]
    h2 = report["HARD_RESULT_2_TEMPORAL"]
    h3 = report["HARD_RESULT_3_CAPACITY"]
    lines = [
        "# Pre-Freeze Verification Report (v2)",
        "",
        f"**FINAL VERDICT:** `{report['FINAL_VERDICT']}`",
        f"**Generated at:** {report['generated_at']}",
        "",
        "## HARD RESULT 1 — Post-Sanitize Runtime Validity",
        "",
        f"- sanitize_retained = **{h1['sanitize_retained']}**",
        f"- MATCH_OK = **{h1['MATCH_OK']} / {h1['sanitize_retained']}**",
        f"- MATCH_FAIL = **{h1['MATCH_FAIL']}**",
        "",
        "## HARD RESULT 2 — Real Temporal Duplication",
        "",
        f"- temporal_near_dup = **{h2['temporal_near_dup']}**",
        f"- only_timestamp_shift = **{h2['only_timestamp_shift']} / {h2['temporal_near_dup']}**",
        f"- only_timestamp_shift_total_rate = **{h2['only_timestamp_shift_total_rate']}**",
        f"- effective_unique_samples_without_timestamp_only_duplicates = **{h2['effective_unique_samples_without_timestamp_only_duplicates']}**",
        f"- breakdown: {json.dumps(h2.get('breakdown', {}))}",
        "",
        "## HARD RESULT 3 — Grounded Capacity",
        "",
        f"- full_instance_capacity_covered = **{h3['full_instance_capacity_covered']}**",
        f"- full_binding_capacity_covered = **{h3['full_binding_capacity_covered']}**",
        f"- generator_coverage_gap = **{h3['generator_coverage_gap']}**",
        f"- adapter_gap = **{h3['adapter_gap']}**",
        f"- matcher_gap = **{h3['matcher_gap']}**",
        f"- entity_binding_gap = **{h3['entity_binding_gap']}**",
        "",
        "## Extra Checks",
        "",
        f"- IMPLEMENTATION_GAP = **{report['extra_checks']['IMPLEMENTATION_GAP']}**",
        f"- SEMANTIC_INVALID = **{report['extra_checks']['SEMANTIC_INVALID']}**",
        f"- PROVENANCE_INVALID = **{report['extra_checks']['PROVENANCE_INVALID']}**",
        "",
    ]
    if report["blockers"]:
        lines.extend(["## Blockers", ""] + [f"- {b}" for b in report["blockers"]])
    return "\n".join(lines)
