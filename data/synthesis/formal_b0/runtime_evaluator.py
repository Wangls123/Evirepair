from __future__ import annotations

from typing import Any

import yaml

from smarthome_mdf.formal_b0.condition_evaluator import evaluate_conditions
from smarthome_mdf.formal_b0.extensions.appliance_runtime import normalize_inputs, validate_start_trigger
from smarthome_mdf.formal_b0.result import abstain, no_action, success
from smarthome_mdf.formal_b0.sequence_walker import walk_sequence
from smarthome_mdf.formal_b0.types import WalkContext
from smarthome_mdf.formal_b0.yaml_trigger_resolver import resolve_fired_trigger_ids
from smarthome_mdf.synthesis_v3.blueprint_parser import BlueprintSpec, _BlueprintLoader

SEQUENTIAL_BLUEPRINTS = frozenset({"appliance_notifications_actions"})

def evaluate_sequential_yaml(
    spec: BlueprintSpec,
    inputs: dict,
    *,
    trigger_context: dict,
    observed: dict,
    control_flow_decisions: list[dict] | None = None,
    input_normalizer=None,
    trigger_validator=None,
    output_contract: str = "v3",
) -> dict[str, Any]:
    decisions = control_flow_decisions if control_flow_decisions is not None else []
    norm_inputs = input_normalizer(inputs) if input_normalizer else dict(inputs or {})
    obs = dict(observed or {})
    derived = dict(obs.get("derived_observation") or {})
    obs = {**obs, **derived}
    ctx = WalkContext(
        inputs=norm_inputs,
        trigger_context=dict(trigger_context or {}),
        observed=obs,
        control_flow_decisions=decisions,
        output_contract=output_contract,
        raw_yaml=spec.raw_yaml,
    )
    ctx.fired_trigger_ids = resolve_fired_trigger_ids(
        spec.raw_yaml,
        norm_inputs,
        trigger_context=ctx.trigger_context,
        observed=obs,
    )

    if trigger_validator:
        ok, failed = trigger_validator(ctx)
        decisions.append({"stage": "trigger", "passed": ok, "failed": failed})
        if ok is None:
            return abstain("insufficient_event_evidence", failed, spec, decisions)
        if not ok:
            if "current_power" in failed or "above_threshold_duration_sec" in failed:
                return abstain("insufficient_event_evidence", failed, spec, decisions)
            return no_action("t0", failed, spec, decisions)

    doc = yaml.load(spec.raw_yaml, Loader=_BlueprintLoader) if spec.raw_yaml else {}
    actions_yaml = doc.get("actions") or doc.get("action") or []
    if isinstance(actions_yaml, dict):
        actions_yaml = [actions_yaml]

    global_conds = doc.get("conditions") or []
    if isinstance(global_conds, list) and len(global_conds) == 1:
        only = global_conds[0]
        if isinstance(only, dict) and only.get("condition") == "and":
            nested = only.get("conditions")
            if isinstance(nested, str) and nested.startswith("!input"):
                key = nested.replace("!input", "").strip().strip("'\"")
                nested = norm_inputs.get(key) or []
            if not nested:
                g_ok, g_failed = True, []
            else:
                g_ok, g_failed = evaluate_conditions(nested, ctx)
        else:
            g_ok, g_failed = evaluate_conditions(global_conds, ctx)
    else:
        g_ok, g_failed = evaluate_conditions(global_conds, ctx)
    decisions.append({"stage": "global_conditions", "passed": g_ok, "failed": g_failed})
    if g_ok is None:
        return abstain("insufficient_event_evidence", g_failed, spec, decisions)
    if not g_ok:
        return no_action("t0", g_failed, spec, decisions)

    actions: list[dict] = []
    walk_sequence(actions_yaml, ctx, branch_id="main", actions=actions)

    if ctx.abstain_reason and not actions:
        return abstain(ctx.abstain_reason, ctx.abstain_missing or ["template"], spec, decisions)
    if actions:
        branch = actions[0].get("source_branch") or "main"
        result = success("t0", branch, actions, spec, decisions)
        if ctx.blocked_future:
            result["blocked_future"] = True
            result["action_prefix_truncated"] = True
        if ctx.branch_resolutions:
            result["branch_resolutions"] = list(ctx.branch_resolutions)
        return result
    if ctx.blocked_future:
        if output_contract == "v4":
            return no_action("t0", ["snapshot_future_unresolved"], spec, decisions)
        return abstain("insufficient_event_evidence", ["wait_or_delay_evidence"], spec, decisions)
    return no_action("t0", ["no_matching_action_path"], spec, decisions)

def evaluate_unified_yaml_runtime(
    spec: BlueprintSpec,
    inputs: dict,
    *,
    trigger_context: dict,
    observed: dict,
    runtime_memory: dict | None = None,
    entity_states: dict | None = None,
    control_flow_decisions: list[dict] | None = None,
    output_contract: str = "v3",
) -> dict[str, Any]:
    _ = runtime_memory
    obs = dict(observed or {})
    if entity_states:
        obs = {**obs, "entity_states": entity_states}

    if spec.blueprint_id in SEQUENTIAL_BLUEPRINTS:
        if spec.blueprint_id == "appliance_notifications_actions":
            return evaluate_sequential_yaml(
                spec,
                inputs,
                trigger_context=trigger_context,
                observed=obs,
                control_flow_decisions=control_flow_decisions,
                input_normalizer=normalize_inputs,
                trigger_validator=validate_start_trigger,
                output_contract=output_contract,
            )

    if not spec.action_paths and spec.raw_yaml:
        doc_actions = None
        try:
            import yaml as _yaml
            from smarthome_mdf.synthesis_v3.blueprint_parser import _BlueprintLoader

            doc = _yaml.load(spec.raw_yaml, Loader=_BlueprintLoader) if spec.raw_yaml else {}
            doc_actions = doc.get("actions") or doc.get("action")
        except Exception:
            doc_actions = None
        if doc_actions:
            return evaluate_sequential_yaml(
                spec,
                inputs,
                trigger_context=trigger_context,
                observed=obs,
                control_flow_decisions=control_flow_decisions,
                output_contract=output_contract,
            )

    from smarthome_mdf.formal_b0.choose_evaluator import evaluate_choose_paths

    return evaluate_choose_paths(
        spec,
        inputs,
        trigger_context=trigger_context,
        observed=obs,
        runtime_memory=runtime_memory,
    )
