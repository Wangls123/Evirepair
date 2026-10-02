from __future__ import annotations

from typing import Any

from smarthome_mdf.formal_b0.action_instantiator import instantiate_action, selector_actions
from smarthome_mdf.formal_b0.condition_evaluator import evaluate_conditions
from smarthome_mdf.formal_b0.extensions.delay import delay_elapsed
from smarthome_mdf.formal_b0.extensions.repeat import execute_repeat
from smarthome_mdf.formal_b0.extensions.wait_for_trigger import evaluate_wait_step
from smarthome_mdf.formal_b0.types import WalkContext

def walk_sequence(
    sequence: list[Any],
    ctx: WalkContext,
    *,
    branch_id: str,
    actions: list[dict],
) -> bool:
    for step in sequence or []:
        if not isinstance(step, dict):
            continue
        if ctx.abstain_reason:
            return False

        if "choose" in step:
            matched = False
            for choice in step.get("choose") or []:
                ok, failed = evaluate_conditions(choice.get("conditions"), ctx)
                ctx.control_flow_decisions.append(
                    {"stage": "choose", "branch_id": branch_id, "passed": ok, "failed": failed}
                )
                if ok is None:
                    ctx.abstain_reason = "insufficient_event_evidence"
                    ctx.abstain_missing.extend(failed)
                    if ctx.output_contract == "v4" and actions:
                        return True
                    return False
                if ok:
                    matched = True
                    seq = choice.get("sequence")
                    if isinstance(seq, str) and seq.startswith("!input"):
                        actions.extend(selector_actions(seq, ctx, branch_id=branch_id, step_no=len(actions) + 1))
                    elif not walk_sequence(seq or [], ctx, branch_id=branch_id, actions=actions):
                        return False
                    break
            if not matched:
                default = step.get("default")
                if default is not None:
                    seq = default if isinstance(default, list) else [default]
                    if not walk_sequence(seq, ctx, branch_id=f"{branch_id}.default", actions=actions):
                        return False
            continue

        if "parallel" in step:
            group = f"{branch_id}.parallel"
            ctx.parallel_group_id = group
            for pi, pseq in enumerate(step.get("parallel") or []):
                if isinstance(pseq, dict):
                    before = len(actions)
                    ok = walk_sequence([pseq], ctx, branch_id=f"{group}_{pi}", actions=actions)
                    resolved = "EXECUTED" if len(actions) > before else ("BLOCKED" if ctx.blocked_future else "NO_ACTION")
                    if ctx.output_contract == "v4":
                        ctx.branch_resolutions.append(
                            {"parallel_group_id": group, "branch_index": pi, "resolution": resolved}
                        )
                        if not ok and ctx.blocked_future:
                            continue
                    elif not ok:
                        return False
            ctx.parallel_group_id = None
            if ctx.output_contract == "v4" and ctx.blocked_future:
                return True
            continue

        if "repeat" in step:
            rep = step.get("repeat") if isinstance(step.get("repeat"), dict) else {}
            if not execute_repeat(
                rep, ctx, branch_id=branch_id, walk_sequence=walk_sequence, actions=actions
            ):
                return False
            continue

        if "delay" in step:
            elapsed = delay_elapsed(step, ctx)
            if elapsed is None:
                ctx.blocked_future = True
                ctx.control_flow_decisions.append({"stage": "delay", "passed": None, "detail": "dead_zone_not_elapsed"})
                return ctx.output_contract == "v4"
            ctx.control_flow_decisions.append({"stage": "delay", "passed": elapsed})
            continue

        if "wait_for_trigger" in step:
            evidence = evaluate_wait_step(step, ctx)
            if evidence is None:
                ctx.blocked_future = True
                ctx.control_flow_decisions.append({"stage": "wait_for_trigger", "passed": None})
                return ctx.output_contract == "v4"
            ctx.control_flow_decisions.append({"stage": "wait_for_trigger", "passed": evidence})
            continue

        if "variables" in step:
            continue

        if "sequence" in step and "action" not in step and "service" not in step:
            if not walk_sequence(step.get("sequence") or [], ctx, branch_id=branch_id, actions=actions):
                return False
            continue

        if "action" in step or "service" in step:
            inst = instantiate_action(step, ctx, branch_id=branch_id, step_no=len(actions) + 1)
            if inst:
                actions.append(inst)
            continue

    return True
