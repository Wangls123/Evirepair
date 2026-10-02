from __future__ import annotations

from typing import Any

from smarthome_mdf.formal_b0.types import WalkContext

def resolve_input_scalar(value: Any, ctx: WalkContext, default: float = 0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.startswith("!input"):
            key = value.replace("!input", "").strip().strip("'\"")
            raw = ctx.inputs.get(key, default)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default
        try:
            return float(value)
        except ValueError:
            return default
    return default

def delay_elapsed(step: dict, ctx: WalkContext) -> bool | None:
    delay = step.get("delay") or {}
    if ctx.observed.get("running_dead_zone_elapsed") is True:
        return True
    if ctx.observed.get("skip_delay") is True:
        return True
    minutes = delay.get("minutes")
    if minutes is not None:
        need = resolve_input_scalar(minutes, ctx, default=0) * 60
        if need <= 0:
            return True
        got = ctx.observed.get("running_dead_zone_elapsed_sec")
        if got is not None:
            return float(got) >= need
        return None
    seconds = delay.get("seconds")
    if seconds is not None:
        need = resolve_input_scalar(seconds, ctx, default=0)
        if need <= 0:
            return True
        got = ctx.observed.get("running_dead_zone_elapsed_sec")
        if got is not None:
            return float(got) >= need
        return None
    return None
