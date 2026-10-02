from __future__ import annotations

import re
from typing import Any

from smarthome_mdf.formal_b0.extensions.appliance_runtime import sensor_float
from smarthome_mdf.formal_b0.types import WalkContext
from smarthome_mdf.synthesis_v3.yaml_template_instantiator import has_unresolved_template, resolve_action_fields

_DEVICE_ATTR_SLUG = re.compile(r"device_attr\(repeat\.item")

def _resolve_states_template(value: Any, ctx: WalkContext) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if "states(" not in text:
        return value
    m = re.search(r"states\(([^)]+)\)", text)
    if not m:
        return value
    sensor = m.group(1).strip().strip("'\"")
    if sensor in ctx.inputs:
        sensor = str(ctx.inputs[sensor])
    reading = sensor_float(ctx, sensor)
    if reading is None and sensor == str(ctx.inputs.get("power_consumption_sensor")):
        reading = sensor_float(ctx, "power_sensor")
    if reading is None:
        return value
    if "| float" in text:
        return reading
    return str(reading)

def _resolve_template_var(value: Any, ctx: WalkContext) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if _DEVICE_ATTR_SLUG.search(text) and ctx.repeat_item is not None:
        return text.replace("repeat.item", str(ctx.repeat_item))
    m = re.match(r"^\{\{\s*(\w+)\s*\}\}$", text)
    if m:
        var = m.group(1)
        if var in ctx.observed:
            return ctx.observed[var]
        if var in ctx.trigger_context:
            return ctx.trigger_context[var]
        if var == "repeat" and ctx.repeat_item is not None:
            return ctx.repeat_item
    if "repeat.item" in text and ctx.repeat_item is not None:
        return text.replace("{{ repeat.item }}", str(ctx.repeat_item)).replace("repeat.item", str(ctx.repeat_item))
    return _resolve_states_template(value, ctx)

def instantiate_action(step: dict, ctx: WalkContext, *, branch_id: str, step_no: int) -> dict | None:
    svc = str(step.get("action") or step.get("service") or "")
    if not svc:
        return None
    if "repeat.item" in svc and ctx.repeat_item is not None:
        svc = svc.replace("{{ repeat.item }}", str(ctx.repeat_item)).replace("repeat.item", str(ctx.repeat_item))
    target = dict(step.get("target") or {})
    data = dict(step.get("data") or step.get("data_template") or step.get("service_data") or {})
    for k, v in list(data.items()):
        data[k] = _resolve_template_var(v, ctx)
    resolved = resolve_action_fields(
        service=svc,
        target=target,
        data=data,
        inputs=ctx.inputs,
        trigger_context=ctx.trigger_context,
        observed=ctx.observed,
    )
    provenance: dict[str, Any] = {}
    if ctx.parallel_group_id:
        provenance["parallel_group_id"] = ctx.parallel_group_id
    if ctx.repeat_iteration is not None:
        provenance["repeat_iteration"] = ctx.repeat_iteration
    if resolved is None:
        if has_unresolved_template(data) or has_unresolved_template(target):
            ctx.control_flow_decisions.append(
                {"stage": "template_skip", "service": svc, "branch_id": branch_id, "reason": "unresolved_template"}
            )
            return None
        clean_data = {k: v for k, v in data.items() if not has_unresolved_template(v)}
        return {
            "step": step_no,
            "service": svc,
            "target": target,
            "data": clean_data,
            "source_branch": branch_id,
            "parallel_group_id": ctx.parallel_group_id,
            "repeat_iteration": ctx.repeat_iteration,
            **provenance,
        }
    svc, tgt, dat = resolved
    return {
        "step": step_no,
        "service": svc,
        "target": tgt,
        "data": dat,
        "source_branch": branch_id,
        "parallel_group_id": ctx.parallel_group_id,
        "repeat_iteration": ctx.repeat_iteration,
        **provenance,
    }

def selector_actions(raw: Any, ctx: WalkContext, *, branch_id: str, step_no: int) -> list[dict]:
    if isinstance(raw, str) and raw.startswith("!input"):
        key = raw.replace("!input", "").strip().strip("'\"")
        raw = ctx.inputs.get(key) or []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for i, act in enumerate(raw, start=step_no):
        if isinstance(act, dict) and (act.get("service") or act.get("action")):
            inst = instantiate_action(act, ctx, branch_id=branch_id, step_no=i)
            if inst:
                out.append(inst)
    return out
