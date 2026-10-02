from __future__ import annotations

import re
from typing import Any

from smarthome_mdf.formal_b0.extensions.appliance_runtime import sensor_float
from smarthome_mdf.formal_b0.time_utils import resolve_local_time, time_in_window
from smarthome_mdf.formal_b0.types import WalkContext

_COMPARE_RE = re.compile(r"^\s*\{\{\s*(\w+)\s*(==|!=)\s*['\"]([^'\"]+)['\"]\s*\}\}\s*$")
_MEMBERSHIP_RE = re.compile(r"^\s*\{\{\s*['\"]([^'\"]+)['\"]\s+in\s+(\w+)\s*\}\}\s*$")
_NONEMPTY_RE = re.compile(r"^\s*\{\{\s*(\w+)\s*!=\s*\[\]\s*\}\}\s*$")
_WAIT_TRIGGER_RE = re.compile(r"wait\.trigger\s+is\s+none", re.I)
_IS_STATE_RE = re.compile(r"is_state\(([^,]+),\s*['\"]([^'\"]+)['\"]\)", re.I)
_REPEAT_ITEM_RE = re.compile(r"repeat\.item", re.I)

def entity_state(ctx: WalkContext, entity_ref: str) -> str | None:
    key = str(entity_ref or "").strip().strip("'\"")
    if key in ctx.inputs:
        key = str(ctx.inputs[key])
    entity_states = ctx.observed.get("entity_states") or ctx.trigger_context.get("entity_states") or {}
    if isinstance(entity_states, dict) and key in entity_states:
        st = entity_states[key]
        if isinstance(st, dict):
            return str(st.get("state") or "")
        return str(st)
    for obs in ctx.observed.get("entity_observations") or []:
        if str(obs.get("entity_id") or "") == key:
            return str(obs.get("state") or "")
    return None

def evaluate_template_condition(expr: str, ctx: WalkContext) -> bool | None:
    text = str(expr or "").strip()
    if not text:
        return True
    if text in ("time", "or", "state", "and"):
        return True

    m = _COMPARE_RE.match(text)
    if m:
        var, op, want = m.group(1), m.group(2), m.group(3)
        got = str(ctx.inputs.get(var) or "")
        if op == "==":
            return got == want
        return got != want

    m = _MEMBERSHIP_RE.match(text)
    if m:
        needle, var = m.group(1), m.group(2)
        hay = ctx.inputs.get(var) or []
        if isinstance(hay, str):
            return needle in hay
        if isinstance(hay, list):
            return needle in hay
        return False

    m = _NONEMPTY_RE.match(text)
    if m:
        val = ctx.inputs.get(m.group(1))
        if isinstance(val, str):
            return bool(val.strip())
        if isinstance(val, list):
            return len(val) > 0
        return bool(val)

    if _WAIT_TRIGGER_RE.search(text):
        if ctx.observed.get("wait_trigger_timed_out") is True:
            return True
        if ctx.observed.get("wait_trigger_satisfied") is True:
            return False
        return None

    m = _IS_STATE_RE.search(text)
    if m:
        ent_expr, want = m.group(1), m.group(2).lower()
        got = (entity_state(ctx, ent_expr) or "").lower()
        return got == want.lower()

    if "include_time ==" in text.replace(" ", ""):
        return str(ctx.inputs.get("include_time") or "time_disabled") == "time_disabled"

    return None

def evaluate_conditions(raw_conds: Any, ctx: WalkContext) -> tuple[bool | None, list[str]]:
    if raw_conds is None:
        return True, []
    if isinstance(raw_conds, str):
        if raw_conds.startswith("!input"):
            key = raw_conds.replace("!input", "").strip().strip("'\"")
            nested = ctx.inputs.get(key) or []
            if not nested:
                return True, []
            return evaluate_conditions(nested, ctx)
        raw_conds = [raw_conds]

    failed: list[str] = []
    for cond in raw_conds or []:
        if isinstance(cond, dict):
            kind = cond.get("condition")
            if kind == "template":
                tpl = str(cond.get("value_template") or "")
                result = evaluate_template_condition(tpl, ctx)
                if result is None:
                    return None, [tpl]
                if not result:
                    failed.append(tpl)
            elif kind == "time":
                ctx.time_condition_required = True
                after_raw = cond.get("after")
                before_raw = cond.get("before")
                if isinstance(after_raw, str) and after_raw.startswith("!input"):
                    after_raw = ctx.inputs.get(after_raw.replace("!input", "").strip().strip("'\""))
                if isinstance(before_raw, str) and before_raw.startswith("!input"):
                    before_raw = ctx.inputs.get(before_raw.replace("!input", "").strip().strip("'\""))
                after = str(after_raw or "")
                before = str(before_raw or "")
                if not after or not before or after.startswith("!input") or before.startswith("!input"):
                    continue
                lt = resolve_local_time(ctx.trigger_context, ctx.observed, required=True)
                if lt is None:
                    return None, ["local_time"]
                if not time_in_window(lt, after, before):
                    failed.append("time_window")
            elif kind == "and":
                sub_ok, sub_fail = evaluate_conditions(cond.get("conditions"), ctx)
                if sub_ok is None:
                    return None, sub_fail
                if not sub_ok:
                    failed.extend(sub_fail)
            elif kind == "or":
                subs = cond.get("conditions") or []
                any_pass = False
                unknown = False
                for sub in subs:
                    sub_ok, sub_fail = evaluate_conditions([sub], ctx)
                    if sub_ok is None:
                        unknown = True
                    elif sub_ok:
                        any_pass = True
                if unknown and not any_pass:
                    return None, ["local_time"] if ctx.time_condition_required else failed
                if not any_pass:
                    failed.append("or")
            elif kind == "trigger":
                want = cond.get("id")
                wants = [str(w) for w in want] if isinstance(want, list) else [str(want)]
                fired = set(ctx.fired_trigger_ids or ())
                if not fired and ctx.raw_yaml:
                    from smarthome_mdf.formal_b0.yaml_trigger_resolver import resolve_fired_trigger_ids

                    fired = resolve_fired_trigger_ids(
                        ctx.raw_yaml,
                        ctx.inputs,
                        trigger_context=ctx.trigger_context,
                        observed=ctx.observed,
                    )
                    ctx.fired_trigger_ids = set(fired)
                if not fired:
                    return None, ["trigger"]
                if not any(w in fired for w in wants):
                    failed.append(f"trigger:{want}")
            else:
                tpl = str(cond.get("value_template") or cond.get("condition") or cond)
                result = evaluate_template_condition(tpl, ctx)
                if result is None:
                    return None, [tpl]
                if not result:
                    failed.append(tpl)
        else:
            result = evaluate_template_condition(str(cond), ctx)
            if result is None:
                return None, [str(cond)]
            if not result:
                failed.append(str(cond))
    return len(failed) == 0, failed
