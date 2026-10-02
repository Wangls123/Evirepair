from __future__ import annotations

from typing import Any

Y_RUNTIME_SEMANTIC_BUILDER_VERSION = "y_runtime_semantic_v1"

_OSAM_TRIGGER_DEFS: dict[str, dict[str, Any]] = {
    "issue_detected": {"from": "off", "to": "on", "entity_role": "motion"},
    "issue_resolved": {"from": "on", "to": "off", "entity_role": "motion"},
}

_PRESENCE_EVIDENCE_KEYS = ("person_detected", "person_state", "presence_state")
_MOTION_KEYS = ("motion_state",)
_LUX_KEYS = ("illuminance", "lux")
_TIME_KEYS = ("local_time", "current_timestamp", "timestamp")

def _trigger_context_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for obs in sample.get("entity_observations") or []:
        tc = (obs.get("attributes") or {}).get("trigger_context") or {}
        for k, v in tc.items():
            if v is not None and v != "":
                out.setdefault(k, v)
    return out

def _observed(sample: dict[str, Any]) -> dict[str, Any]:
    return dict(sample.get("observed") or {})

def _match_osam_trigger_ids(obs: dict[str, Any], trig: dict[str, Any]) -> list[str]:

    matched: list[str] = []
    raw_tid = trig.get("trigger_id")
    if raw_tid and str(raw_tid) in _OSAM_TRIGGER_DEFS:
        matched.append(str(raw_tid))
    from_state = str(trig.get("from_state") or obs.get("motion_state") or "").lower()
    to_state = str(trig.get("to_state") or obs.get("motion_state") or "").lower()
    for tid, spec in _OSAM_TRIGGER_DEFS.items():
        if tid in matched:
            continue
        spec_from = str(spec.get("from") or "").lower()
        spec_to = str(spec.get("to") or "").lower()
        if from_state == spec_from and to_state == spec_to:
            matched.append(tid)
        elif to_state == spec_to and not from_state and spec_from:
            matched.append(tid)
    return sorted(set(matched))

def _condition_facts(obs: dict[str, Any], trig: dict[str, Any], blueprint_id: str) -> dict[str, str]:

    facts: dict[str, str] = {}
    for key in _MOTION_KEYS:
        if key in obs and obs[key] not in (None, ""):
            facts[f"motion.{key}"] = "KNOWN_TRUE"
    for key in _LUX_KEYS:
        if key in obs and obs[key] is not None:
            facts[f"lux.{key}"] = "KNOWN_TRUE"
    for key in _TIME_KEYS:
        if key in obs or key in trig:
            facts[f"time.{key}"] = "KNOWN_TRUE"
    for key in _PRESENCE_EVIDENCE_KEYS:
        if key in obs and obs[key] not in (None, ""):
            facts[f"presence.{key}"] = "KNOWN_TRUE"
        elif blueprint_id.startswith("presence_"):
            facts[f"presence.{key}"] = "UNKNOWN"
    return facts

def build_y_runtime_semantic_context(
    sample: dict[str, Any],
    *,
    blueprint_id: str,
    blueprint_inputs: dict[str, Any] | None = None,
    canonical_semantics: dict[str, Any] | None = None,
) -> dict[str, Any]:

    obs = _observed(sample)
    trig = _trigger_context_from_sample(sample)
    entity_ids = sorted(
        {
            str(eo.get("entity_id"))
            for eo in (sample.get("entity_observations") or [])
            if eo.get("entity_id")
        }
    )

    trigger_match: list[str] = []
    if blueprint_id == "security_osam_sensor_alert":
        trigger_match = _match_osam_trigger_ids(obs, trig)

    canonical_triggers = list((canonical_semantics or {}).get("canonical_trigger_semantics") or [])
    trigger_ids_in_yaml = [
        str(t.get("id") or t.get("trigger_id") or "")
        for t in canonical_triggers
        if isinstance(t, dict) and (t.get("id") or t.get("trigger_id"))
    ]

    condition_facts = _condition_facts(obs, trig, blueprint_id)
    inputs = dict(blueprint_inputs or {})

    relevant_obs = {k: obs[k] for k in sorted(obs) if obs[k] not in (None, "", [], {})}
    relevant_trig = {k: trig[k] for k in sorted(trig) if trig[k] not in (None, "", [], {})}

    return {
        "builder_version": Y_RUNTIME_SEMANTIC_BUILDER_VERSION,
        "trigger_context": relevant_trig,
        "observed_state": relevant_obs,
        "entity_ids": entity_ids,
        "blueprint_input_bindings": inputs,
        "canonical_trigger_ids": trigger_ids_in_yaml,
        "runtime_trigger_id_match": trigger_match,
        "condition_facts": condition_facts,
        "notes": [],
    }
