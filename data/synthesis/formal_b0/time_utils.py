from __future__ import annotations

from datetime import datetime, time as dt_time

def parse_hms(val: str) -> dt_time:
    parts = str(val).split(":")
    return dt_time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)

def local_time_from_timestamp(timestamp: str | None) -> str | None:
    if not timestamp:
        return None
    try:
        ts = str(timestamp).replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%H:%M:%S")
    except ValueError:
        return None

def resolve_local_time(
    trigger_context: dict,
    observed: dict,
    *,
    required: bool = False,
) -> dt_time | None:

    lt = trigger_context.get("local_time") or observed.get("local_time")
    if isinstance(lt, dt_time):
        return lt
    if isinstance(lt, str) and lt.strip():
        return parse_hms(lt)
    ts = observed.get("timestamp") or trigger_context.get("timestamp")
    derived = local_time_from_timestamp(ts if isinstance(ts, str) else None)
    if derived:
        return parse_hms(derived)
    if required:
        return None
    return None

def time_in_window(local_time: dt_time, after: str, before: str) -> bool:
    a = parse_hms(after)
    b = parse_hms(before)
    if a <= b:
        return a <= local_time < b
    return local_time >= a or local_time < b
