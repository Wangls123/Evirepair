from __future__ import annotations

from typing import Any

def _flatten(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out: list[str] = []
        for i in x:
            out.extend(_flatten(i))
        return out
    return [str(x)]

def _norm(s: str) -> str:
    return str(s or "").strip().lower().replace(" ", "")

def _entity_for(ir: dict[str, Any], domains: tuple[str, ...]) -> str | None:
    for e in ir.get("entities") or []:
        eid = str(e.get("entity_id") or "")
        dom = str(e.get("domain") or (eid.split(".", 1)[0] if "." in eid else ""))
        if dom in domains and eid:
            return eid
    acts = ir.get("actions") or []
    for a in acts:
        tgt = a.get("target") or a.get("target_entity") or a.get("entity")
        if tgt and str(tgt).split(".", 1)[0] in domains:
            return str(tgt)
        svc = str(a.get("service") or "")
        if svc.split(".", 1)[0] in domains:
            return tgt
    return None

def convert_taps(taps: list[dict[str, Any]], ir: dict[str, Any]) -> dict[str, Any]:

    texts: list[str] = []
    for tap in taps or []:
        texts.extend(_flatten(tap.get("action")))
        texts.extend(_flatten(tap.get("trigger")))
        texts.extend(_flatten(tap.get("condition")))
    blob = " ".join(_norm(t) for t in texts)

    climate_ent = _entity_for(ir, ("climate",))
    light_ent = _entity_for(ir, ("light", "switch"))

    if "thermostat.ac=false" in blob or "thermostat.acsetfalse" in blob or "thermostat.acistrue" in blob and "!" in blob:
        return {
            "ok": True,
            "final_action": {
                "capability": "climate",
                "operation": "turn_off",
                "target": climate_ent,
                "semantic_payload": {},
                "service": "climate.turn_off",
            },
            "mapped_channel": "thermostat.ac=false",
        }
    if "thermostat.ac=true" in blob or "thermostat.acsettrue" in blob:
        return {
            "ok": True,
            "final_action": {
                "capability": "climate",
                "operation": "turn_on",
                "target": climate_ent,
                "semantic_payload": {},
                "service": "climate.turn_on",
            },
            "mapped_channel": "thermostat.ac=true",
        }
    if "hue_light.power=true" in blob or "hue_light.powersettrue" in blob:
        return {
            "ok": True,
            "final_action": {
                "capability": "light",
                "operation": "turn_on",
                "target": light_ent or "light.living_room",
                "semantic_payload": {},
                "service": "light.turn_on",
            },
            "mapped_channel": "hue_light.power=true",
        }
    if "hue_light.power=false" in blob or "hue_light.powersetfalse" in blob:
        return {
            "ok": True,
            "final_action": {
                "capability": "light",
                "operation": "turn_off",
                "target": light_ent or "light.living_room",
                "semantic_payload": {},
                "service": "light.turn_off",
            },
            "mapped_channel": "hue_light.power=false",
        }
    return {"ok": False, "final_action": None, "mapped_channel": None, "reason": "no_ha_channel_in_native_taps"}

def convert_tap_program(taps: list[dict[str, Any]], ir: dict[str, Any]) -> dict[str, Any]:

    acts: list[dict[str, Any]] = []
    mapped: list[str] = []
    seen: set[tuple] = set()
    for tap in taps or []:
        one = convert_taps([tap], ir)
        fa = one.get("final_action")
        if not one.get("ok") or not fa:
            continue
        key = (str(fa.get("service") or ""), str(fa.get("target") or ""), str(fa.get("operation") or ""))
        if key in seen:
            continue
        seen.add(key)
        acts.append(fa)
        mapped.append(str(one.get("mapped_channel") or ""))
    if not acts:
        return {"ok": False, "final_action_set": [], "mapped_channels": mapped, "reason": "no_ha_channel_in_native_taps"}
    return {"ok": True, "final_action_set": acts, "mapped_channels": mapped}
