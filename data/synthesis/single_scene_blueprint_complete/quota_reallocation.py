from __future__ import annotations

from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.blueprint_capacity import normalize_blueprint_state
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (
    MAX_BLUEPRINT_SHARE_WITHIN_SCENE,
    PHASE2_INCREMENT_BASE,
    PHASE2_MAX_BP_ABSOLUTE,
    PHASE2_MAX_SHARE_DELTA,
    TARGET_TOTAL,
)
from smarthome_mdf.single_scene_blueprint_complete.final_scheduler import FinalGenerationScheduler

def _quota_stop_reasons() -> frozenset[str]:
    return frozenset({"SAMPLE_TARGET_AND_COVERAGE_REACHED", "ALLOCATED_TARGET_REACHED"})

def released_quota_from_blueprint(st: Any) -> int:

    initial = getattr(st, "initial_allocated_target", st.allocated_target)
    state = normalize_blueprint_state(
        st.stop_reason,
        remaining_grounded_pairs=getattr(st, "cached_remaining_pairs", 0),
    )
    if state == "GROUNDED_SPACE_SATURATED":
        return max(0, initial - st.accepted_count)
    if state == "ALLOCATED_TARGET_REACHED" and getattr(st, "cached_remaining_pairs", 0) == 0:
        return max(0, initial - st.accepted_count)
    return 0

def compute_reallocation_priority(
    *,
    blueprint_id: str,
    scheduler: FinalGenerationScheduler,
    remaining_pairs: int,
    complexity_score: float = 0.0,
) -> tuple:

    st = scheduler.bp_state[blueprint_id]
    uncovered_behaviors = 0
    uncovered_instances = len(st.reachable_instance_ids - st.observed_instance_ids)
    uncovered_branches: set[str] = set()
    covered_branches: set[str] = set()
    uncovered_paths: set[str] = set()
    covered_paths: set[str] = set()
    uncovered_templates: set[str] = set()
    covered_templates: set[str] = set()
    for t in scheduler.behavior_targets:
        if t.blueprint_id != blueprint_id or not t.grounded_reachable or t.static_only:
            continue
        key = t.strict_path_key or f"{t.branch_id}::{t.action_path_signature}::{t.action_template_key}"
        path_count = scheduler.path_coverage.path_count(blueprint_id, key)
        if path_count <= 0:
            uncovered_behaviors += 1
            uncovered_branches.add(t.branch_id)
            uncovered_paths.add(t.action_path_signature)
            uncovered_templates.add(t.action_template_key)
        else:
            covered_branches.add(t.branch_id)
            covered_paths.add(t.action_path_signature)
            covered_templates.add(t.action_template_key)
    unused_pair_flag = 0 if remaining_pairs > 0 else 1
    min_source_reuse = min(st.source_obs_ids.values()) if st.source_obs_ids else 0
    return (
        uncovered_behaviors > 0,
        uncovered_instances > 0,
        len(uncovered_branches) > 0,
        len(uncovered_paths) > 0,
        len(uncovered_templates) > 0,
        unused_pair_flag == 0,
        min_source_reuse,
        st.accepted_count,
        -complexity_score,
        blueprint_id,
    )

def _soft_cap_for_blueprint(
    *,
    blueprint_id: str,
    scheduler: FinalGenerationScheduler,
    phase2_base_share: dict[str, float],
) -> int:
    st = scheduler.bp_state[blueprint_id]
    scene_total = max(1, scheduler.scene_count(st.scene))
    base_share = phase2_base_share.get(blueprint_id, st.accepted_count / scene_total)
    max_share = min(MAX_BLUEPRINT_SHARE_WITHIN_SCENE + PHASE2_MAX_SHARE_DELTA, base_share + PHASE2_MAX_SHARE_DELTA)
    share_cap = int(scene_total * max_share)
    return min(PHASE2_MAX_BP_ABSOLUTE, max(st.accepted_count + PHASE2_INCREMENT_BASE, share_cap))

def try_phase2_reallocation(
    *,
    scheduler: FinalGenerationScheduler,
    remaining_by_bp: dict[str, int],
    complexity_by_bp: dict[str, float] | None = None,
    global_total: int,
    verbose: bool = False,
) -> dict[str, Any]:

    complexity_by_bp = complexity_by_bp or {}
    for bp_id, st in scheduler.bp_state.items():
        st.cached_remaining_pairs = remaining_by_bp.get(bp_id, 0)

    headroom = TARGET_TOTAL - global_total
    if headroom <= 0:
        return {"activated": [], "released_quota": 0, "phase": scheduler.generation_phase}

    released = sum(released_quota_from_blueprint(st) for st in scheduler.bp_state.values())
    eligible: list[tuple[tuple, str, int]] = []
    for bp_id, st in scheduler.bp_state.items():
        remaining = remaining_by_bp.get(bp_id, 0)
        if remaining <= 0:
            continue
        state = normalize_blueprint_state(st.stop_reason, remaining_grounded_pairs=remaining)
        if state == "GROUNDED_SPACE_SATURATED":
            continue
        if st.stop_reason == "PENDING" and st.generation_phase >= 2:
            continue
        if st.stop_reason == "PENDING" and st.accepted_count < st.effective_target:
            continue
        prio = compute_reallocation_priority(
            blueprint_id=bp_id,
            scheduler=scheduler,
            remaining_pairs=remaining,
            complexity_score=complexity_by_bp.get(bp_id, 0.0),
        )
        eligible.append((prio, bp_id, remaining))

    if not eligible and scheduler.generation_phase < 2:

        for bp_id, st in scheduler.bp_state.items():
            remaining = remaining_by_bp.get(bp_id, 0)
            if remaining <= 0:
                continue
            if st.stop_reason in _quota_stop_reasons() or (
                st.stop_reason == "PENDING" and st.accepted_count >= st.initial_allocated_target
            ):
                prio = compute_reallocation_priority(
                    blueprint_id=bp_id,
                    scheduler=scheduler,
                    remaining_pairs=remaining,
                    complexity_score=complexity_by_bp.get(bp_id, 0.0),
                )
                eligible.append((prio, bp_id, remaining))

    if not eligible:
        return {"activated": [], "released_quota": released, "phase": scheduler.generation_phase}

    if scheduler.generation_phase < 2:
        scheduler.generation_phase = 2
        for sc in scheduler.scene_state.values():
            if sc.stop_reason in _quota_stop_reasons():
                sc.stop_reason = "PENDING"

    eligible.sort(key=lambda x: x[0])
    phase2_base_share: dict[str, float] = {}
    for bp_id, st in scheduler.bp_state.items():
        scene_total = max(1, scheduler.scene_count(st.scene))
        phase2_base_share[bp_id] = st.accepted_count / scene_total

    budget_pool = max(headroom, released)
    per_round = max(PHASE2_INCREMENT_BASE, budget_pool // max(len(eligible), 1))
    activated: list[dict[str, Any]] = []

    for _prio, bp_id, remaining in eligible:
        st = scheduler.bp_state[bp_id]
        soft_cap = _soft_cap_for_blueprint(
            blueprint_id=bp_id, scheduler=scheduler, phase2_base_share=phase2_base_share
        )
        increment = min(
            remaining,
            per_round,
            soft_cap - st.accepted_count,
            TARGET_TOTAL - global_total - sum(a["increment"] for a in activated),
        )
        if increment <= 0:
            continue
        prev_target = st.effective_target
        st.effective_target = st.accepted_count + increment
        st.generation_phase = 2
        st.stop_reason = "PENDING"
        st.consecutive_fail = 0
        st.attempt_budget = max(st.attempt_budget, st.attempt_count + increment * 3)
        activated.append(
            {
                "blueprint_id": bp_id,
                "increment": increment,
                "effective_target": st.effective_target,
                "previous_effective_target": prev_target,
                "remaining_pairs": remaining,
            }
        )
        if verbose:
            print(
                f"[phase2] reactivated {bp_id}: +{increment} -> effective_target={st.effective_target} "
                f"(remaining_pairs={remaining})",
                flush=True,
            )

    return {
        "activated": activated,
        "released_quota": released,
        "phase": scheduler.generation_phase,
        "eligible_count": len(eligible),
    }

def resume_reactivate_quota_limited(
    *,
    scheduler: FinalGenerationScheduler,
    remaining_by_bp: dict[str, int],
    complexity_by_bp: dict[str, float] | None = None,
    global_total: int,
    verbose: bool = False,
) -> dict[str, Any]:

    for bp_id, st in scheduler.bp_state.items():
        remaining = remaining_by_bp.get(bp_id, 0)
        st.cached_remaining_pairs = remaining
        state = normalize_blueprint_state(st.stop_reason, remaining_grounded_pairs=remaining)
        if remaining > 0 and state in ("ALLOCATED_TARGET_REACHED", "ACTIVE"):
            if st.stop_reason != "PENDING":
                st.stop_reason = "PENDING"
                st.consecutive_fail = 0
            if st.generation_phase < 2 and st.accepted_count >= st.initial_allocated_target:
                st.generation_phase = 2
        elif remaining == 0 and st.stop_reason == "PENDING":
            st.stop_reason = "GROUNDED_SPACE_SATURATED"
        elif remaining == 0 and st.stop_reason in _quota_stop_reasons():
            st.stop_reason = "GROUNDED_SPACE_SATURATED"

    return try_phase2_reallocation(
        scheduler=scheduler,
        remaining_by_bp=remaining_by_bp,
        complexity_by_bp=complexity_by_bp,
        global_total=global_total,
        verbose=verbose,
    )
