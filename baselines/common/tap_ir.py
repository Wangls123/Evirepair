from __future__ import annotations

from typing import Any

VISUAL_SCENES = {"visual_fusion"}
UNSUPPORTED_SERVICE_PREFIXES = ("llmvision.",)

def _obs_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "on", "open", "detected", "1", "yes", "home", "occupied"}:
        return True
    if s in {"false", "off", "closed", "clear", "0", "no", "away", "idle", "unavailable", "unknown"}:
        return False
    return None

def _b0_actions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    fb = sample.get("formal_b0") or {}
    out: list[dict[str, Any]] = []
    for raw in list(fb.get("semantic_actions") or []):
        if not isinstance(raw, dict):
            continue
        svc = str(raw.get("service") or "").strip()
        if not svc:
            continue
        params = dict(raw.get("parameters") or {})
        ent = raw.get("target_entity") or raw.get("entity") or raw.get("entity_id")
        out.append(
            {
                "service": svc,
                "target": None if ent in (None, "", []) else str(ent),
                "parameters": {k: v for k, v in params.items() if v not in (None, "", [], {})},
            }
        )
    return out

def _entities(sample: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(eid: Any, state: Any = None, attrs: dict | None = None) -> None:
        s = str(eid or "").strip()
        if not s or "." not in s or s in seen:
            return
        seen.add(s)
        rec: dict[str, Any] = {"entity_id": s, "domain": s.split(".", 1)[0]}
        if state is not None:
            rec["state"] = state
        if attrs:
            rec["attributes"] = attrs
        found.append(rec)

    bb = sample.get("blueprint_binding") or {}
    for ent in bb.get("entities") or []:
        add(ent)
    for eo in sample.get("entity_observations") or []:
        if isinstance(eo, dict):
            add(eo.get("entity_id"), eo.get("state"), eo.get("attributes") if isinstance(eo.get("attributes"), dict) else None)
    return found

def _triggers(sample: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sm = sample.get("synthesis_metadata") or {}
    tp = sm.get("temporal_pattern") or {}
    if isinstance(tp, str):
        tp = {"trigger_id": tp}
    if isinstance(tp, dict) and tp.get("trigger_id"):
        out.append({"kind": "temporal_pattern", "trigger_id": tp.get("trigger_id"), "raw": tp})
    b0 = sample.get("b0_output") or {}
    trace = b0.get("execution_trace") if isinstance(b0, dict) else {}
    if not isinstance(trace, dict):
        trace = {}
    tr = trace.get("trigger_evaluation") or {}
    if isinstance(tr, dict) and tr:
        out.append({"kind": "b0_trigger_evaluation", "trigger_id": tr.get("trigger_id"), "raw": {k: tr[k] for k in list(tr)[:12]}})
    rs = sm.get("runtime_situation") or {}
    if rs:
        out.append({"kind": "runtime_situation", "raw": rs if isinstance(rs, dict) else {"value": rs}})
    return out

def _conditions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    obs = sample.get("observed") or {}
    for key in ("quiet_hours", "silent_period", "occupancy", "motion_state", "window_state", "door_state"):
        if key in obs:
            out.append({"field": key, "value": obs.get(key), "boolean": _obs_bool(obs.get(key))})
    return out

def _capabilities(actions: list[dict[str, Any]], entities: list[dict[str, Any]]) -> list[str]:
    caps: list[str] = []
    for a in actions:
        svc = str(a.get("service") or "")
        if svc.startswith("light."):
            caps.append("Lighting")
        elif svc.startswith("climate."):
            caps.append("ClimateControl")
        elif svc.startswith("notify."):
            caps.append("Notify")
        elif svc.startswith(("logbook.", "system_log.", "persistent_notification.")):
            caps.append("Log")
        elif svc.startswith("google_sheets."):
            caps.append("Record")
        elif svc.startswith("llmvision."):
            caps.append("Vision")
        elif svc.startswith(("schedule.", "scene.")):
            caps.append("Schedule")
        else:
            caps.append("Other")
    for e in entities:
        d = e.get("domain")
        if d == "light":
            caps.append("Lighting")
        elif d == "climate":
            caps.append("ClimateControl")
        elif d in {"binary_sensor", "sensor"}:
            caps.append("Sensor")
    seen: set[str] = set()
    uniq = []
    for c in caps:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def sample_to_ir(sample: dict[str, Any]) -> dict[str, Any]:

    scene = str(sample.get("scene_type") or "")
    actions = _b0_actions(sample)
    entities = _entities(sample)
    obs = sample.get("observed") or {}
    unsupported = scene in VISUAL_SCENES or any(
        str(a.get("service") or "").startswith(UNSUPPORTED_SERVICE_PREFIXES) for a in actions
    )
    return {
        "id": str(sample.get("sample_id") or ""),
        "dataset": "ss",
        "scene_type": scene,
        "blueprint_id": str((sample.get("blueprint_binding") or {}).get("blueprint_id") or ""),
        "trigger": _triggers(sample),
        "conditions": _conditions(sample),
        "actions": actions,
        "entities": entities,
        "observations": dict(obs),
        "entity_observations": list(sample.get("entity_observations") or []),
        "capabilities": _capabilities(actions, entities),
        "temporal_constraints": [],
        "component_links": [],
        "b0_execution_status": str((sample.get("formal_b0") or {}).get("execution_status") or ""),
        "unsupported_input": unsupported,
        "gold": None,
    }

def parent_to_ir(parent: dict[str, Any], ss_index: dict[str, dict[str, Any]]) -> dict[str, Any]:

    pid = str(parent.get("multi_action_id") or "")
    components = []
    all_actions: list[dict[str, Any]] = []
    for c in parent.get("components") or []:
        sid = str(c.get("single_scene_sample_id") or "")
        ss = ss_index.get(sid) or {}
        ir = sample_to_ir(ss) if ss else {
            "id": sid,
            "dataset": "ss",
            "scene_type": str(c.get("scene") or ""),
            "blueprint_id": str(c.get("blueprint_id") or ""),
            "trigger": [],
            "conditions": [],
            "actions": [],
            "entities": [],
            "observations": (c.get("runtime_state") or {}).get("observed") or {},
            "entity_observations": (c.get("runtime_state") or {}).get("entity_observations") or [],
            "capabilities": [],
            "temporal_constraints": [],
            "component_links": [],
            "unsupported_input": str(c.get("scene") or "") in VISUAL_SCENES,
        }
        ir["component_id"] = c.get("component_id")
        ir["entity_binding"] = c.get("entity_binding")
        components.append(ir)
        for a in ir.get("actions") or []:
            aa = dict(a)
            aa["component_origin"] = c.get("component_id")
            all_actions.append(aa)
    fb = parent.get("formal_b0") or {}
    parent_actions = []
    pnode = fb.get("parent") if isinstance(fb.get("parent"), dict) else {}
    for raw in list((pnode or {}).get("semantic_actions") or fb.get("semantic_actions") or []):
        if isinstance(raw, dict) and raw.get("service"):
            parent_actions.append(
                {
                    "service": raw.get("service"),
                    "target": raw.get("target_entity") or raw.get("entity"),
                    "parameters": dict(raw.get("parameters") or {}),
                    "component_origin": raw.get("component_origin"),
                }
            )
    if not parent_actions:
        parent_actions = all_actions
    unsupported = any(c.get("unsupported_input") for c in components)
    return {
        "id": pid,
        "dataset": "ma",
        "trigger": [t for c in components for t in (c.get("trigger") or [])],
        "conditions": [t for c in components for t in (c.get("conditions") or [])],
        "actions": parent_actions,
        "entities": [e for c in components for e in (c.get("entities") or [])],
        "observations": {},
        "capabilities": sorted({cap for c in components for cap in (c.get("capabilities") or [])}),
        "temporal_constraints": [],
        "component_links": [
            {
                "component_id": c.get("component_id"),
                "sample_id": c.get("id"),
                "scene_type": c.get("scene_type"),
                "blueprint_id": c.get("blueprint_id"),
            }
            for c in components
        ],
        "components": components,
        "unsupported_input": unsupported,
        "gold": None,
    }
