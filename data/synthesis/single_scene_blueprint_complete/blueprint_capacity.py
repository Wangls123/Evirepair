from __future__ import annotations

from typing import Any

from smarthome_mdf.multi_action_vnext.requirement_extractor import extract_requirement_bundle
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (
    TRUE_EVIDENCE_LIMITED_BLUEPRINTS,
)
from smarthome_mdf.single_scene_blueprint_complete.final_scheduler import FinalGenerationScheduler
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import behavior_source_dedup_key
from smarthome_mdf.single_scene_blueprint_complete.synthesizer import _get_ir

LEGACY_INSTANCE_SUFFIX = "__inst_default"

def is_legacy_instance_id(instance_id: str) -> bool:
    return LEGACY_INSTANCE_SUFFIX in instance_id or instance_id.endswith("_default")

def classify_instance_coverage(
    *,
    grounded_instance_ids: set[str],
    observed_instance_ids: set[str],
) -> dict[str, Any]:
    current_grounded = set(grounded_instance_ids)
    legacy_present = sorted(iid for iid in observed_instance_ids if iid not in current_grounded)
    current_covered = sorted(iid for iid in observed_instance_ids if iid in current_grounded)
    return {
        "current_grounded_instance_count": len(current_grounded),
        "current_grounded_instances_covered": len(current_covered),
        "legacy_instance_ids_present": legacy_present,
        "covered_instance_count_all": len(observed_instance_ids),
    }

def normalize_blueprint_state(stop_reason: str, *, remaining_grounded_pairs: int) -> str:

    if stop_reason == "PENDING":
        return "ACTIVE"
    if stop_reason in ("SAMPLE_TARGET_AND_COVERAGE_REACHED", "ALLOCATED_TARGET_REACHED"):
        return "ALLOCATED_TARGET_REACHED"
    if stop_reason in ("TRUE_EVIDENCE_LIMITED",):
        return "TRUE_EVIDENCE_LIMITED"
    if remaining_grounded_pairs > 0 and stop_reason in (
        "BLUEPRINT_GROUNDED_SPACE_SATURATED",
        "GROUNDED_BEHAVIOR_SPACE_SMALL",
        "ALL_GROUNDED_BEHAVIORS_COVERED",
        "EVIDENCE_CAPACITY_EXHAUSTED",
    ):
        return "ALLOCATED_TARGET_REACHED"
    if remaining_grounded_pairs == 0 and stop_reason in (
        "BLUEPRINT_GROUNDED_SPACE_SATURATED",
        "GROUNDED_BEHAVIOR_SPACE_SMALL",
        "ALL_GROUNDED_BEHAVIORS_COVERED",
        "EVIDENCE_CAPACITY_EXHAUSTED",
    ):
        return "GROUNDED_SPACE_SATURATED"
    if stop_reason == "BLUEPRINT_GROUNDED_SPACE_SATURATED":
        return "GROUNDED_SPACE_SATURATED" if remaining_grounded_pairs == 0 else "ACTIVE"
    if remaining_grounded_pairs == 0:
        return "GROUNDED_SPACE_SATURATED"
    return "ALLOCATED_TARGET_REACHED"

def compute_remaining_behavior_source_pairs(
    *,
    blueprint_id: str,
    scheduler: FinalGenerationScheduler,
    selector: Any,
    pool: Any,
    inst_list: list[dict],
    reg: Any,
) -> int:

    total = 0
    behavior_rows = [
        {
            "blueprint_id": t.blueprint_id,
            "automation_instance_id": t.automation_instance_id,
            "branch_id": t.branch_id,
            "action_path_signature": t.action_path_signature,
        }
        for t in scheduler.behavior_targets
        if t.blueprint_id == blueprint_id and t.grounded_reachable and not t.static_only
    ]
    if not behavior_rows:
        return 0
    scene = scheduler.bp_state[blueprint_id].scene if blueprint_id in scheduler.bp_state else ""
    for inst in inst_list:
        iid = inst["automation_instance_id"]
        nb = _normalize_from_instance(scene, inst, reg)
        try:
            ir = _get_ir(blueprint_id, inst, iid)
        except Exception:
            ir = None
        bundle = extract_requirement_bundle(nb, ir=ir)
        entity_sig = "|".join(sorted(inst.get("bound_entities") or []))
        for bt in behavior_rows:
            if bt["automation_instance_id"] != iid:
                continue
            key = selector.behavior_temporal_key_for_target(
                blueprint_id=bt["blueprint_id"],
                automation_instance_id=bt["automation_instance_id"],
                branch_id=bt["branch_id"],
                action_path_signature=bt["action_path_signature"],
                entity_binding_signature=entity_sig,
            )
            total += selector.unused_compatible_count(
                pool,
                bundle,
                iid,
                key,
                blueprint_id=blueprint_id,
                branch_id=bt["branch_id"],
                action_path_signature=bt["action_path_signature"],
            )
    return total

def compute_all_remaining_pairs(
    *,
    scheduler: FinalGenerationScheduler,
    selector: Any,
    pool: Any,
    inst_idx: dict[str, list[dict]],
    reg: Any,
) -> dict[str, int]:
    remaining: dict[str, int] = {}
    for bp_id in scheduler.bp_state:
        if bp_id in TRUE_EVIDENCE_LIMITED_BLUEPRINTS:
            remaining[bp_id] = 0
            continue
        inst_list = inst_idx.get(bp_id) or []
        remaining[bp_id] = compute_remaining_behavior_source_pairs(
            blueprint_id=bp_id,
            scheduler=scheduler,
            selector=selector,
            pool=pool,
            inst_list=inst_list,
            reg=reg,
        )
    return remaining

def accepted_pairs_from_samples(samples: list[dict], blueprint_id: str) -> int:
    keys = {behavior_source_dedup_key(s) for s in samples if (s.get("blueprint_binding") or {}).get("blueprint_id") == blueprint_id}
    return len(keys)

def build_capacity_row(
    *,
    blueprint_id: str,
    scene: str,
    samples: list[dict],
    scheduler: FinalGenerationScheduler,
    selector: Any,
    pool: Any,
    inst_list: list[dict],
    grounded_raw: list[dict],
    reg: Any,
    scheduler_stop_reason: str = "UNKNOWN",
) -> dict[str, Any]:
    from collections import Counter, defaultdict

    accepted_by_bp: Counter = Counter()
    instances_seen: dict[str, set[str]] = defaultdict(set)
    behavior_keys_seen: dict[str, set[str]] = defaultdict(set)
    sources_used: dict[str, set[str]] = defaultdict(set)
    pairs_accepted: dict[str, set[str]] = defaultdict(set)

    for s in samples:
        bb = s.get("blueprint_binding") or {}
        bp = bb.get("blueprint_id") or ""
        if bp != blueprint_id:
            continue
        accepted_by_bp[bp] += 1
        iid = bb.get("automation_instance_id")
        if iid:
            instances_seen[bp].add(iid)
        meta = s.get("synthesis_metadata") or {}
        behavior_keys_seen[bp].add(f"{meta.get('branch_id')}::{meta.get('action_path_signature')}")
        prov = s.get("provenance") or {}
        rid = prov.get("source_record_id")
        if rid:
            sources_used[bp].add(str(rid))
        pairs_accepted[bp].add(behavior_source_dedup_key(s))

    grounded_ids = {c["automation_instance_id"] for c in grounded_raw}
    inst_cov = classify_instance_coverage(
        grounded_instance_ids=grounded_ids,
        observed_instance_ids=instances_seen.get(blueprint_id) or set(),
    )

    compatible_sources: set[str] = set()
    for inst in inst_list:
        iid = inst["automation_instance_id"]
        nb = _normalize_from_instance(scene, inst, reg)
        try:
            ir = _get_ir(blueprint_id, inst, iid)
        except Exception:
            ir = None
        bundle = extract_requirement_bundle(nb, ir=ir)
        for obs, _ in selector.viable_for_bundle(pool, bundle, iid):
            compatible_sources.add(obs.record_id)

    remaining = compute_remaining_behavior_source_pairs(
        blueprint_id=blueprint_id,
        scheduler=scheduler,
        selector=selector,
        pool=pool,
        inst_list=inst_list,
        reg=reg,
    )
    pairs_acc = len(pairs_accepted.get(blueprint_id) or set())
    cap_est = remaining + pairs_acc
    bp_state = normalize_blueprint_state(scheduler_stop_reason, remaining_grounded_pairs=remaining)

    reachable = sum(
        1
        for t in scheduler.behavior_targets
        if t.blueprint_id == blueprint_id and t.grounded_reachable and not t.static_only
    )

    return {
        "blueprint_id": blueprint_id,
        "scene": scene,
        "blueprint_state": bp_state,
        "accepted_count": accepted_by_bp.get(blueprint_id, 0),
        "current_grounded_instance_count": inst_cov["current_grounded_instance_count"],
        "current_grounded_instances_covered": inst_cov["current_grounded_instances_covered"],
        "legacy_instance_ids_present": inst_cov["legacy_instance_ids_present"],
        "grounded_instance_count": inst_cov["current_grounded_instance_count"],
        "covered_instance_count": inst_cov["current_grounded_instances_covered"],
        "reachable_behavior_count": reachable,
        "covered_behavior_count": len(behavior_keys_seen.get(blueprint_id) or set()),
        "compatible_source_count": len(compatible_sources),
        "used_source_count": len(sources_used.get(blueprint_id) or set()),
        "grounded_behavior_source_capacity": cap_est,
        "accepted_behavior_source_pairs": pairs_acc,
        "unused_behavior_source_pairs": remaining,
        "remaining_grounded_pairs": remaining,
        "behavior_source_capacity_estimate": cap_est,
        "behavior_source_pairs_accepted": pairs_acc,
        "saturation_reason": scheduler_stop_reason,
        "scheduler_stop_reason": scheduler_stop_reason,
    }
