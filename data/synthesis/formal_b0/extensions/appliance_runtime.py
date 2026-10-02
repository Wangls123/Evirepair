from __future__ import annotations

from smarthome_mdf.formal_b0.types import WalkContext
from smarthome_mdf.synthesis_v3.appliance_notifications_inputs import normalize_appliance_notifications_inputs

def normalize_inputs(inputs: dict) -> dict:
    return normalize_appliance_notifications_inputs(inputs)

def sensor_float(ctx: WalkContext, sensor_key: str) -> float | None:
    key = str(ctx.inputs.get(sensor_key) or sensor_key)
    entity_states = ctx.observed.get("entity_states") or {}
    if key in entity_states:
        raw = entity_states[key]
        if isinstance(raw, dict):
            raw = raw.get("state")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    if sensor_key == "power_sensor" or key == ctx.inputs.get("power_sensor"):
        for fld in ("current_power_w", "current_power", "power_watt"):
            if fld in ctx.observed:
                try:
                    return float(ctx.observed[fld])
                except (TypeError, ValueError):
                    pass
    for obs in ctx.observed.get("entity_observations") or []:
        if str(obs.get("entity_id") or "") == key:
            attrs = obs.get("attributes") or {}
            for fld in ("current_power_w", "state"):
                if fld in attrs or fld in obs:
                    try:
                        return float(attrs.get(fld) or obs.get(fld))
                    except (TypeError, ValueError):
                        pass
    return None

def validate_start_trigger(ctx: WalkContext) -> tuple[bool | None, list[str]]:

    if ctx.observed.get("wait_trigger_satisfied") or ctx.observed.get("appliance_cycle_complete"):
        return True, []

    if ctx.trigger_context.get("trigger_event_is_post_condition") is True:
        return True, []

    power = sensor_float(ctx, "power_sensor")
    if power is None:
        return None, ["current_power"]

    start_th = float(ctx.inputs.get("start_appliance_power") or ctx.inputs.get("start_threshold") or 10)
    if power <= start_th:
        return False, [f"power_not_above_start:{power}<={start_th}"]

    dur = ctx.observed.get("above_threshold_duration_sec")
    if dur is None:
        dur = ctx.observed.get("start_time_delay_elapsed_sec")

    if dur is None:
        return None, ["above_threshold_duration_sec"]

    need_min = float(ctx.inputs.get("start_time_delay") or 1) * 60
    if float(dur) < need_min:
        return False, [f"start_duration_lt_delay:{dur}<{need_min}"]
    return True, []
