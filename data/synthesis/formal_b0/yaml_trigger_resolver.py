from __future__ import annotations

from typing import Any

import yaml

from smarthome_mdf.synthesis_v3.blueprint_parser import _BlueprintLoader

_OPEN_STATES = frozenset({"on", "open", "detected", "true", "1"})
_CLOSED_STATES = frozenset({"off", "closed", "close", "clear", "false", "0"})

def _resolve_input(value: Any, inputs: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("!input"):
        key = value.replace("!input", "").strip()
        return inputs.get(key)
    return value

def _normalize_state(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _OPEN_STATES:
        return "on"
    if text in _CLOSED_STATES:
        return "off"
    return text

def _entity_observations(observed: dict[str, Any]) -> list[dict[str, Any]]:
    rows = observed.get("entity_observations") or []
    return [r for r in rows if isinstance(r, dict)]

def _entity_state(entity_id: str, *, inputs: dict, trigger_context: dict, observed: dict) -> str | None:
    key = str(entity_id or "").strip()
    if not key:
        return None
    entity_states = observed.get("entity_states") or trigger_context.get("entity_states") or {}
    if isinstance(entity_states, dict) and key in entity_states:
        st = entity_states[key]
        if isinstance(st, dict):
            return _normalize_state(st.get("state"))
        return _normalize_state(st)
    tc_entity = str(trigger_context.get("entity") or "")
    if tc_entity == key or (not tc_entity and key):
        if trigger_context.get("to_state") is not None:
            return _normalize_state(trigger_context.get("to_state"))
    for obs in _entity_observations(observed):
        if str(obs.get("entity_id") or "") == key:
            attrs = obs.get("attributes") or {}
            for field in ("window_state", "door_state", "motion_state", "state"):
                if field in attrs and attrs[field] is not None:
                    return _normalize_state(attrs[field])
            return _normalize_state(obs.get("state"))
    ws = observed.get("window_state") or observed.get("door_state")
    if ws is not None:
        return _normalize_state(ws)
    ms = observed.get("motion_state")
    if ms is not None:
        return _normalize_state(ms)
    return None

def _transition_states(*, trigger_context: dict, observed: dict, entity_id: str) -> tuple[str | None, str | None]:
    tc_entity = str(trigger_context.get("entity") or "")
    if not tc_entity or tc_entity == entity_id:
        from_state = trigger_context.get("from_state")
        to_state = trigger_context.get("to_state")
        if from_state is not None or to_state is not None:
            return _normalize_state(from_state) if from_state is not None else None, _normalize_state(to_state) if to_state is not None else None
    for obs in _entity_observations(observed):
        if str(obs.get("entity_id") or "") != entity_id:
            continue
        tc = (obs.get("attributes") or {}).get("trigger_context") or {}
        if isinstance(tc, dict) and (tc.get("from_state") is not None or tc.get("to_state") is not None):
            fs = tc.get("from_state")
            ts = tc.get("to_state")
            return (_normalize_state(fs) if fs is not None else None, _normalize_state(ts) if ts is not None else None)
    return None, None

def _duration_required(trigger: dict) -> bool:
    for_spec = trigger.get("for")
    if not for_spec:
        return False
    if isinstance(for_spec, dict):
        return any(for_spec.get(k) is not None for k in ("seconds", "minutes", "hours", "milliseconds"))
    if isinstance(for_spec, str) and for_spec.startswith("!input"):
        return True
    return bool(for_spec)

def _duration_evidence_present(*, trigger_context: dict, observed: dict) -> bool:
    for key in (
        "state_duration_sec",
        "duration_sec",
        "above_threshold_duration_sec",
        "required_delay_sec",
        "for_seconds",
    ):
        if trigger_context.get(key) is not None or observed.get(key) is not None:
            return True
    derived = observed.get("derived_observation") or {}
    if isinstance(derived, dict):
        for key in ("state_duration_sec", "duration_sec"):
            if derived.get(key) is not None:
                return True
    return False

def _trigger_entity_id(trigger: dict, inputs: dict[str, Any], observed: dict) -> str | None:
    entity = _resolve_input(trigger.get("entity_id"), inputs)
    if entity:
        return str(entity)
    bound = inputs.get("bound_entities") or observed.get("bound_entities") or []
    if isinstance(bound, list) and bound:
        return str(bound[0])
    obs = _entity_observations(observed)
    if obs:
        return str(obs[0].get("entity_id") or "")
    return None

def _state_trigger_matches(
    trigger: dict,
    *,
    inputs: dict,
    trigger_context: dict,
    observed: dict,
) -> bool:
    entity_id = _trigger_entity_id(trigger, inputs, observed)
    if not entity_id:
        return False
    want_to = _normalize_state(_resolve_input(trigger.get("to"), inputs)) if trigger.get("to") is not None else None
    want_from = _normalize_state(_resolve_input(trigger.get("from"), inputs)) if trigger.get("from") is not None else None
    current = _entity_state(entity_id, inputs=inputs, trigger_context=trigger_context, observed=observed)
    from_state, to_state = _transition_states(trigger_context=trigger_context, observed=observed, entity_id=entity_id)
    if to_state is None and current is not None:
        to_state = current
    if _duration_required(trigger) and not _duration_evidence_present(trigger_context=trigger_context, observed=observed):
        return False
    if want_to is not None:
        matched_to = to_state == want_to or (to_state is None and current == want_to)
        if not matched_to:
            return False
    if want_from is not None:
        matched_from = from_state == want_from
        if not matched_from and want_to is not None and current == want_to and want_from == "off" and want_to == "on":
            matched_from = True
        if not matched_from and want_to is None and current == want_from:
            matched_from = True
        if not matched_from:
            return False
    if want_to is None and want_from is not None:
        return from_state == want_from or current == want_from
    return want_to is not None or want_from is not None

def _parse_triggers(raw_yaml: str | None) -> list[dict[str, Any]]:
    if not raw_yaml:
        return []
    try:
        doc = yaml.load(raw_yaml, Loader=_BlueprintLoader) or {}
    except Exception:
        return []
    block = doc.get("triggers") or doc.get("trigger") or []
    if isinstance(block, dict):
        return [block]
    if isinstance(block, list):
        return [t for t in block if isinstance(t, dict)]
    return []

def resolve_fired_trigger_ids(
    raw_yaml: str | None,
    inputs: dict[str, Any],
    *,
    trigger_context: dict[str, Any],
    observed: dict[str, Any],
) -> set[str]:

    fired: set[str] = set()
    triggers = _parse_triggers(raw_yaml)

    explicit = str(trigger_context.get("yaml_trigger_id") or trigger_context.get("trigger_id") or "").strip()
    if explicit and explicit not in ("start", "end", "start_state_to"):
        valid_ids = {str(t.get("id")) for t in triggers if t.get("id") is not None}
        if not valid_ids or explicit in valid_ids:
            fired.add(explicit)

    for trigger in triggers:
        tid = trigger.get("id")
        if tid is None:
            continue
        tid_s = str(tid)
        platform = str(trigger.get("platform") or trigger.get("trigger") or "state").lower()
        if platform in ("state", ""):
            if _state_trigger_matches(trigger, inputs=inputs, trigger_context=trigger_context, observed=observed):
                fired.add(tid_s)
        elif platform == "time":
            at = trigger.get("at")
            if at and (trigger_context.get("schedule_fired") or observed.get("schedule_fired")):
                fired.add(tid_s)

    return fired
