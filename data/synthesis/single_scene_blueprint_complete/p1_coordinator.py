from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance, build_all_behavior_specs
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR, PROJECT_SCENES, SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import GROUNDED_CAPABLE_BLUEPRINTS
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import build_instance_templates
from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool
from smarthome_mdf.single_scene_blueprint_complete.p1_integrity_audit import (
    V1_DIR_NAME,
    V2_DIR_NAME,
    VERDICT_BLOCKED,
    audit_visual_fusion,
    run_full_integrity_audit,
    write_audit_artifacts,
)
from smarthome_mdf.single_scene_blueprint_complete.semantic_sanitize import detect_semantic_contamination, sanitize_attributes
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import (
    canonical_behavior_hash,
    stable_action_node_id,
    strip_dynamic_behavior_suffix,
)
from smarthome_mdf.single_scene_blueprint_complete.synthesizer import synthesize_single_scene
from smarthome_mdf.single_scene_blueprint_complete.final_scheduler import FinalGenerationScheduler
from smarthome_mdf.single_scene_blueprint_complete.final_complexity import build_blueprint_complexity
from smarthome_mdf.single_scene_blueprint_complete.final_budgets import build_blueprint_budget
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

VF_BLUEPRINTS = SCENE_BLUEPRINT_MAP.get("visual_fusion", [])

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _write_jsonl(path: Path, samples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

def _sanitize_sample_metadata(sample: dict) -> dict:

    s = dict(sample)
    meta = dict(s.get("synthesis_metadata") or {})
    bb = s.get("blueprint_binding") or {}
    bp = bb.get("blueprint_id", "")
    canonical_bt = strip_dynamic_behavior_suffix(meta.get("behavior_target") or (s.get("system_state") or {}).get("behavior_target"))
    meta["behavior_target"] = canonical_bt
    meta["canonical_behavior_target"] = canonical_bt
    meta["stable_action_node_id"] = stable_action_node_id(
        blueprint_id=str(bp),
        branch_id=str(meta.get("branch_id") or ""),
        action_path_signature=str(meta.get("action_path_signature") or ""),
    )
    s["synthesis_metadata"] = meta
    if s.get("system_state"):
        s["system_state"] = {**s["system_state"], "behavior_target": canonical_bt}
    eos = []
    for eo in s.get("entity_observations") or []:
        eo = dict(eo)
        attrs = dict(eo.get("attributes") or {})
        dc = attrs.get("device_class") or eo.get("domain")
        clean, _ = sanitize_attributes(attrs, device_class=str(dc) if dc else None)
        eo["attributes"] = clean
        eos.append(eo)
    s["entity_observations"] = eos
    s["canonical_behavior_signature"] = canonical_behavior_hash(s)
    s["diversity_signature"] = canonical_behavior_hash(s)
    md = dict(s.get("metadata") or {})
    md["schema_version"] = "single_scene_blueprint_complete_v2"
    md["p1_repair"] = "metadata_sanitize"
    s["metadata"] = md
    return s

def _regenerate_visual_fusion(
    *,
    target_count: int,
    pool,
    specs: list[dict],
    inst_idx: dict,
    reg,
    reuse_counts: dict,
) -> tuple[list[dict], list[dict]]:

    replacements: list[dict] = []
    report_rows: list[dict] = []
    spec_idx = {s["blueprint_id"]: s for s in specs}
    nb_cache: dict[str, Any] = {}
    attempts = 0
    frame_cursor = 0

    while len(replacements) < target_count and attempts < target_count * 20:
        bp_id = VF_BLUEPRINTS[attempts % len(VF_BLUEPRINTS)]
        inst_list = inst_idx.get(bp_id) or []
        instance = inst_list[0] if inst_list else {}
        iid = instance.get("automation_instance_id", f"{bp_id}__inst_default")
        cache_key = f"{bp_id}::{iid}"
        if cache_key not in nb_cache:
            nb_cache[cache_key] = _normalize_from_instance("visual_fusion", instance, reg)
        bspec = spec_idx.get(bp_id) or {}
        targets = [t for t in (bspec.get("branches") or [])]
        branch_id = "default"
        action_sig = "notify"
        if bspec.get("semantic_action_templates"):
            tmpl = bspec["semantic_action_templates"][attempts % len(bspec["semantic_action_templates"])]
            action_sig = tmpl.get("service") or action_sig
            branch_id = tmpl.get("branch_id") or branch_id
        bt = f"{bp_id}::{branch_id}::{action_sig}"

        status, sample, reason = synthesize_single_scene(
            scene="visual_fusion",
            blueprint_id=bp_id,
            automation_instance_id=iid,
            behavior_target=bt,
            branch_id=branch_id,
            action_path_signature=action_sig,
            instance=instance,
            normalized_blueprint=nb_cache[cache_key],
            behavior_spec=bspec,
            source_pool=pool,
            reuse_counts=reuse_counts,
            attempt_index=attempts,
            visual_frame_index=frame_cursor,
        )
        attempts += 1
        frame_cursor += 1
        if status != "ACCEPT" or not sample:
            continue
        sample["metadata"] = {**(sample.get("metadata") or {}), "p1_repair": "visual_fusion_regenerated"}
        replacements.append(sample)
        report_rows.append({"blueprint_id": bp_id, "frame_id": (sample.get("observed") or {}).get("frame_id"), "status": "regenerated"})

    return replacements, report_rows

def run_p1_repair(
    *,
    v1_dir: Path | None = None,
    v2_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    v1 = v1_dir or (OUTPUT_DIR / V1_DIR_NAME)
    v2 = v2_dir or (OUTPUT_DIR / V2_DIR_NAME)
    v2.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(v1 / "single_scene_samples.jsonl")
    if verbose:
        print(f"P1 Step 1: Auditing v1 ({len(samples)} samples)...", flush=True)

    audit = run_full_integrity_audit(samples)
    write_audit_artifacts(v1, audit)
    (v1 / "manifest.json").write_text(
        json.dumps(
            {
                "verdict": VERDICT_BLOCKED,
                "prior_verdict": "READY_FOR_SINGLE_SCENE_FREEZE",
                "blockers": ["P1 integrity audit failed — see P1_SINGLE_SCENE_DATA_INTEGRITY_REPORT.md"],
                "total_samples": len(samples),
                "invalid_count": audit["invalid_count"],
                "audit_at": _utc_now(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    invalid_ids = set(audit["invalid_sample_ids"])
    vf_invalid = {s["sample_id"] for s in samples if s.get("scene_type") == "visual_fusion"}
    semantic_invalid = set()
    for row in audit["sample_validity_rows"]:
        if row["validity"] == "INVALID_SEMANTIC_MAPPING":
            semantic_invalid.add(row["sample_id"])

    retained = [s for s in samples if s.get("sample_id") not in invalid_ids]
    if verbose:
        print(f"P1 Step 2: Retained {len(retained)}, invalid {len(invalid_ids)} (VF={len(vf_invalid)})", flush=True)

    reg = build_registry()
    specs = build_all_behavior_specs(reg)
    pool = build_legacy_source_pool(PROJECT_SCENES, include_public=True)
    inst_idx = build_instance_templates().get("instances") or {}
    reuse_counts: dict[str, int] = {}

    vf_target = len([s for s in samples if s.get("scene_type") == "visual_fusion"])
    if verbose:
        print(f"P1 Step 3: Regenerating visual_fusion ({vf_target})...", flush=True)
    vf_new, vf_report = _regenerate_visual_fusion(
        target_count=vf_target,
        pool=pool,
        specs=specs,
        inst_idx=inst_idx,
        reg=reg,
        reuse_counts=reuse_counts,
    )

    replacement_map: list[dict] = []
    for old in samples:
        if old.get("sample_id") in vf_invalid:
            replacement_map.append({"old_sample_id": old.get("sample_id"), "reason": "INVALID_VISUAL_PROVENANCE", "status": "pending"})

    v2_samples: list[dict] = []
    vf_iter = iter(vf_new)
    for old in samples:
        sid = old.get("sample_id")
        if sid in vf_invalid:
            new_s = next(vf_iter, None)
            if new_s:
                new_s = dict(new_s)
                new_s["replaces_sample_id"] = sid
                new_s["regeneration_reason"] = "INVALID_VISUAL_PROVENANCE"
                v2_samples.append(new_s)
                replacement_map.append({"old_sample_id": sid, "new_sample_id": new_s.get("sample_id"), "reason": "INVALID_VISUAL_PROVENANCE"})
        elif sid in semantic_invalid:
            replacement_map.append({"old_sample_id": sid, "reason": "INVALID_SEMANTIC_MAPPING", "status": "needs_manual_review"})
            v2_samples.append(_sanitize_sample_metadata(old))
        else:
            v2_samples.append(_sanitize_sample_metadata(old))

    _write_jsonl(v2 / "single_scene_samples.jsonl", v2_samples)

    v2_audit = run_full_integrity_audit(v2_samples)
    write_audit_artifacts(v2, v2_audit)

    vf_v2 = audit_visual_fusion(v2_samples)
    verdict = "READY_FOR_SINGLE_SCENE_FREEZE" if v2_audit["invalid_count"] == 0 and not vf_v2.get("single_frame_domination") else "BLOCKED"

    targeted_report = {
        "visual_fusion_regenerated": len(vf_new),
        "visual_fusion_target": vf_target,
        "retained_sanitized": len(retained),
        "vf_regeneration_report_sample": vf_report[:20],
    }
    (v2 / "targeted_regeneration_report.json").write_text(json.dumps(targeted_report, indent=2), encoding="utf-8")
    (v2 / "v1_to_v2_replacement_map.json").write_text(json.dumps(replacement_map, indent=2), encoding="utf-8")

    report_md = _format_p1_report(v1_audit=audit, v2_audit=v2_audit, v2_samples=v2_samples, verdict=verdict, vf_v2=vf_v2)
    (v2 / "P1_SINGLE_SCENE_DATA_INTEGRITY_REPORT.md").write_text(report_md, encoding="utf-8")

    manifest = {
        "verdict": verdict,
        "dataset_version": "final_regeneration_v2",
        "prior_version": "final_regeneration_v1",
        "total_samples": len(v2_samples),
        "v1_total": len(samples),
        "v1_invalid": audit["invalid_count"],
        "targeted_regeneration": targeted_report,
        "completed_at": _utc_now(),
    }
    (v2 / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verbose:
        print(f"P1 complete: v2={len(v2_samples)} verdict={verdict}", flush=True)
    return manifest

def _format_p1_report(*, v1_audit, v2_audit, v2_samples, verdict, vf_v2) -> str:
    sba = v1_audit["stable_behavior_signature_audit"]
    dup = v1_audit["duplicate_reaudit"]
    vf1 = v1_audit["visual_fusion_grounding_audit"]
    scene_counts = Counter(s.get("scene_type") for s in v2_samples)
    bp_counts = Counter((s.get("blueprint_binding") or {}).get("blueprint_id") for s in v2_samples)
    lines = [
        "# P1 Single-Scene Data Integrity Report",
        "",
        f"**V1 Verdict:** `{VERDICT_BLOCKED}`",
        f"**V2 Verdict:** `{verdict}`",
        f"**Generated at:** {_utc_now()}",
        "",
        "## Key Answers",
        "",
        f"1. dynamic aXXXX is generation-time ID: **Yes** ({sba.get('dynamic_a_suffix_sample_count')}/{v1_audit['total_samples']})",
        f"2. previous duplicate rate=0 valid: **No** — {dup.get('previous_duplicate_audit_invalid')}",
        f"3. stable canonical behavior count (v1): **{sba.get('stable_canonical_behavior_count')}**",
        f"4. v1 visual unique frames: **{vf1.get('unique_image_frames')}** (largest reuse: {vf1.get('largest_frame_reuse_count')})",
        f"5. CASAS/YouHome provenance mix (v1): **{vf1.get('cross_dataset_without_explicit_composition_count')}** implicit cross-dataset samples",
        f"6. v1 visual_fusion invalid: **{vf1.get('total_visual_fusion_samples')}** (single-frame domination)",
        f"7. motion+power/temperature fake semantics (v1): motion_with_power={v1_audit['raw_to_adapter_semantic_audit'].get('motion_with_power_count')}",
        f"8. semantic contamination adapters: entity_observations from V3 pool with multiplexed attrs",
        f"9. INVALID_MATCH blueprints: see semantic_contamination_impact.json",
        f"10. v1 truly invalid: **{v1_audit['invalid_count']}**",
        f"11. scenes targeted: **visual_fusion**",
        f"12-15. instance/binding capacity: see automation_instance_diversity_audit.json (21 inst_default; binding per blueprint in entity_binding audit)",
        f"16. dominant actions: see action_distribution_audit.json (may reflect YAML behavior concentration)",
        f"17. VF replacements: **{len([s for s in v2_samples if s.get('replaces_sample_id')])}**",
        f"18. v2 total samples: **{len(v2_samples)}**",
        f"19. v2 scenes: **{len(scene_counts)}/8** — {dict(scene_counts)}",
        f"20. v2 grounded blueprints: **{len([b for b in GROUNDED_CAPABLE_BLUEPRINTS if bp_counts.get(b)])}/21**",
        f"21. v2 visual unique frames: **{vf_v2.get('unique_image_frames')}**",
        f"22. freeze ready: **{verdict == 'READY_FOR_SINGLE_SCENE_FREEZE'}**",
        "",
        "## V2 Duplicate Reaudit",
        "",
        json.dumps(v2_audit.get("duplicate_reaudit", {}), indent=2),
    ]
    return "\n".join(lines)
