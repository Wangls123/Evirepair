from __future__ import annotations

import json
from typing import Any

def action_to_final(act: dict | None) -> dict | None:
    if not act:
        return None
    svc = str(act.get("service") or "")
    if "." in svc:
        cap, op = svc.split(".", 1)
    else:
        cap, op = "unknown", svc or "none"
    return {
        "capability": cap,
        "operation": op,
        "target": act.get("target") or act.get("target_entity") or act.get("entity"),
        "semantic_payload": dict(act.get("parameters") or {}),
        "service": svc,
    }

def convert_ss_fix(fix_rule: Any, ir: dict[str, Any]) -> dict[str, Any]:

    orig = [action_to_final(a) for a in (ir.get("actions") or []) if a]
    orig = [a for a in orig if a]
    if not fix_rule:
        return {"ok": True, "status": "NATIVE_NO_CHANGE", "final_action": orig[0] if orig else None}
    if isinstance(fix_rule, str):
        texts = fix_rule.lower()
    else:
        texts = json.dumps(fix_rule, ensure_ascii=False).lower()
    mapped = None
    if "airconditioner.switch" in texts:
        climate = next((a for a in orig if str(a.get("service") or "").startswith("climate")), orig[0] if orig else {})
        mapped = {
            "capability": "climate",
            "operation": "turn_off" if "off" in texts else "turn_on",
            "target": (climate or {}).get("target"),
            "semantic_payload": {},
            "service": "climate.turn_off" if "off" in texts else "climate.turn_on",
        }
    elif "light.switch" in texts:
        light = next((a for a in orig if str(a.get("service") or "").startswith("light")), orig[0] if orig else {})
        mapped = {
            "capability": "light",
            "operation": "turn_on" if "on" in texts else "turn_off",
            "target": (light or {}).get("target") or "light.living_room",
            "semantic_payload": {},
            "service": "light.turn_on" if "on" in texts else "light.turn_off",
        }
    if mapped is None:
        return {"ok": False, "status": "OUTPUT_TRANSLATION_FAILURE", "final_action": None}
    return {"ok": True, "status": "NATIVE_REPAIRED", "final_action": mapped}
