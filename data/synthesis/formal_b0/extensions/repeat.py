from __future__ import annotations

import re
from typing import Any, Callable

from smarthome_mdf.formal_b0.types import WalkContext

_JINJA_VAR = re.compile(r"\{\{\s*([^}]+)\s*\}\}")

def resolve_for_each(raw: Any, ctx: WalkContext) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("!input"):
            key = text.replace("!input", "").strip().strip("'\"")
            val = ctx.inputs.get(key)
            if isinstance(val, list):
                return val
            if val is not None:
                return [val]
            return []
        m = _JINJA_VAR.match(text)
        if m:
            var = m.group(1).strip()
            if var in ctx.inputs:
                val = ctx.inputs[var]
                if isinstance(val, list):
                    return val
                if val is not None:
                    return [val]
            if var in ctx.observed:
                val = ctx.observed[var]
                if isinstance(val, list):
                    return val
            if var in ctx.trigger_context:
                val = ctx.trigger_context[var]
                if isinstance(val, list):
                    return val
        return []
    return []

def execute_repeat(
    rep: dict,
    ctx: WalkContext,
    *,
    branch_id: str,
    walk_sequence: Callable[..., bool],
    actions: list[dict],
) -> bool:
    inner = rep.get("sequence") or []
    if inner and isinstance(inner[0], dict) and "wait_for_trigger" in inner[0]:
        from smarthome_mdf.formal_b0.extensions.wait_for_trigger import evaluate_wait_in_repeat

        return evaluate_wait_in_repeat(inner, ctx, branch_id=branch_id, walk_sequence=walk_sequence, actions=actions)

    for_each = rep.get("for_each")
    if for_each is not None:
        items = resolve_for_each(for_each, ctx)
        if not items and "{{" in str(for_each):
            ctx.abstain_reason = "insufficient_event_evidence"
            ctx.abstain_missing.append("repeat.for_each")
            return False
        for idx, item in enumerate(items):
            prev_item = ctx.repeat_item
            prev_iter = ctx.repeat_iteration
            ctx.repeat_item = item
            ctx.repeat_iteration = idx + 1
            if not walk_sequence(inner, ctx, branch_id=f"{branch_id}.repeat_{idx}", actions=actions):
                ctx.repeat_item = prev_item
                ctx.repeat_iteration = prev_iter
                return False
            ctx.repeat_item = prev_item
            ctx.repeat_iteration = prev_iter
        return True

    count = rep.get("count")
    if count is not None:
        try:
            if isinstance(count, str) and count.startswith("!input"):
                key = count.replace("!input", "").strip().strip("'\"")
                n = int(ctx.inputs.get(key) or 1)
            else:
                n = int(count)
        except (TypeError, ValueError):
            n = 1
        for ri in range(max(n, 0)):
            ctx.repeat_iteration = ri + 1
            if not walk_sequence(inner, ctx, branch_id=f"{branch_id}.repeat_{ri}", actions=actions):
                return False
        ctx.repeat_iteration = None
        return True

    ctx.repeat_iteration = 1
    ok = walk_sequence(inner, ctx, branch_id=f"{branch_id}.repeat", actions=actions)
    ctx.repeat_iteration = None
    return ok
