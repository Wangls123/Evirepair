from __future__ import annotations

from typing import Any, Callable

from smarthome_mdf.formal_b0.types import WalkContext

def wait_for_trigger_evidence(step: dict, ctx: WalkContext) -> bool | None:
    if ctx.observed.get("wait_trigger_satisfied") is True:
        return True
    if ctx.observed.get("appliance_cycle_complete") is True:
        return True
    below = ctx.observed.get("current_power_w")
    if below is None:
        below = ctx.observed.get("current_power")
    finish = float(ctx.inputs.get("end_appliance_power") or ctx.inputs.get("finish_threshold") or 5)
    if below is not None and float(below) < finish:
        dur = ctx.observed.get("below_threshold_duration_sec")
        need_min = float(ctx.inputs.get("end_time_delay") or 1) * 60
        if dur is not None and float(dur) >= need_min:
            return True
    if ctx.observed.get("wait_trigger_timed_out") is True:
        return False
    return None

def evaluate_wait_step(step: dict, ctx: WalkContext) -> bool | None:
    return wait_for_trigger_evidence(step, ctx)

def evaluate_wait_in_repeat(
    inner: list,
    ctx: WalkContext,
    *,
    branch_id: str,
    walk_sequence: Callable[..., bool],
    actions: list[dict],
) -> bool:
    evidence = wait_for_trigger_evidence(inner[0], ctx)
    if evidence is None:
        ctx.blocked_future = True
        ctx.control_flow_decisions.append(
            {"stage": "wait_for_trigger", "passed": None, "detail": "missing_stop_evidence"}
        )
        return False
    ctx.control_flow_decisions.append(
        {
            "stage": "wait_for_trigger",
            "passed": evidence,
            "timed_out": ctx.observed.get("wait_trigger_timed_out"),
        }
    )
    if evidence:
        return walk_sequence(inner[1:], ctx, branch_id=f"{branch_id}.after_wait", actions=actions)
    if ctx.observed.get("wait_trigger_timed_out"):
        return walk_sequence(inner[1:], ctx, branch_id=f"{branch_id}.watchdog", actions=actions)
    return True
