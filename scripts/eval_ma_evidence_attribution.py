from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ss_trhr_final import load_jsonl
from eval_strict_v2_set_remove import (
    MA,
    OUT,
    SS,
    TRACE,
    acc_bucket,
    add_bucket,
    apply_strict_v2,
    cap_of_svc,
    classify_m2,
    compact_action,
    finish_bucket,
    ma_y_acts,
    map_repaired,
    set_metrics,
    set_remove_v1,
    tok,
)
from eval_ma_m2c_necessity_gate import (
    DEFAULT_MIN_CORE,
    apply_gate,
    extra_pairs,
)
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.evidence import (
    door_open,
    motion_on,
    window_open,
)

ATTR_JSONL = OUT / "ma_evidence_attribution.jsonl"
ATTR_MD = OUT / "ma_evidence_attribution_analysis.md"
OWN_JSONL = OUT / "ma_component_evidence_ownership.jsonl"
COMP_JSONL = OUT / "ma_parent_behavior_composition.jsonl"
GATE_MD = OUT / "ma_generic_necessity_gate_design.md"
EVAL_JSON = OUT / "ma_attribution_offline_evaluation.json"
IDLE_JSON = OUT / "ma_parent_idle_certification.json"
RESID_JSON = OUT / "ma_residual_after_attribution.json"
DECISION_MD = OUT / "ma_behavior_composition_decision.md"

PLACEHOLDER_ENTITIES = {"binary_sensor.grounded", "sensor.grounded", "binary_sensor.grounded_window"}
SKIP_OBS_KEYS = {"timestamp", "source_timestamp", "aligned_timestamp", "time", "datetime"}

OBS_FAMILY = {
    "window_state": "window",
    "hvac_mode": "climate",
    "temperature": "climate",
    "humidity": "climate",
    "current_temperature": "climate",
    "target_temperature": "climate",
    "preset_mode": "climate",
    "motion_state": "motion",
    "motion": "motion",
    "occupancy": "motion",
    "illuminance_lux": "lux",
    "illuminance": "lux",
    "light_state": "light",
    "brightness": "light",
    "door_state": "door",
    "lock_state": "door",
    "camera_presence": "person",
    "person_detected": "person",
    "human_count": "person",
    "current_power_w": "power",
    "power": "power",
    "energy": "power",
}

FAMILY_CAPS = {
    "window": ("ClimateControl",),
    "climate": ("ClimateControl",),
    "motion": ("Lighting",),
    "lux": ("Lighting",),
    "light": ("Lighting",),
    "door": ("Notify",),
    "person": ("Notify",),
    "power": ("Record",),
}

NOTIFY_EVENT_FAMS = {"door", "person"}
LIGHT_EVENT_FAMS = {"motion"}
CLIMATE_EVENT_FAMS = {"window"}
ACTIVE_STATE_FAMS = {"climate", "light", "lux", "power"}

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0

def _norm(v: Any) -> str:
    return str(v or "").strip().lower()

def toks_from_acts(acts: list[dict], bind_all: dict) -> list:
    out = []
    for a in acts:
        origin = str(a.get("component_origin") or "")
        t = tok(a, bind_all.get(origin) or {})
        if t:
            out.append(t)
    return out

def binding_originals(bind: dict) -> list[str]:
    seen: list[str] = []
    for r in bind.get("records") or []:
        o = str(r.get("original_entity") or "")
        if o and o not in seen:
            seen.append(o)
    for k in (bind.get("component_entity_map") or {}):
        ks = str(k)
        if ks and ks not in seen:
            seen.append(ks)
    return seen

def entity_family(eid: str) -> str:
    e = _norm(eid)
    if not e:
        return "other"
    if e in PLACEHOLDER_ENTITIES or e.endswith(".grounded"):
        return "placeholder"
    if "window" in e or e.startswith("cover."):
        return "window"
    if "door" in e or "lock" in e:
        return "door"
    if "motion" in e or "occupancy" in e:
        return "motion"
    if "person" in e or "camera" in e or "human" in e:
        return "person"
    if e.startswith("climate.") or "hvac" in e or "thermostat" in e:
        return "climate"
    if e.startswith("light."):
        return "light"
    if "lux" in e or "illumin" in e:
        return "lux"
    if e.startswith("switch.") and "light" in e:
        return "light"
    if "power" in e or "energy" in e or e.startswith("sensor.power"):
        return "power"
    return "other"

def family_caps(fam: str) -> tuple[str, ...]:
    return FAMILY_CAPS.get(fam, ())

def scene_caps(scene: str) -> tuple[str, ...]:
    s = _norm(scene)
    if "climate" in s or "hvac" in s:
        return ("ClimateControl",)
    if "light" in s or "occupancy" in s:
        return ("Lighting",)
    if "notif" in s or "security" in s:
        return ("Notify",)
    if "energy" in s or "periodic" in s or "record" in s or "sheet" in s:
        return ("Record",)
    return ()

def bt_suffix(bt: str) -> str:
    s = str(bt or "")
    if "→" in s:
        s = s.rsplit("→", 1)[-1]
    if "::" in s:
        s = s.rsplit("::", 1)[-1]
    return s.strip()

def is_no_action(bt: str) -> bool:
    return "no_action" in _norm(bt_suffix(bt))

def svc_to_behavior(svc: str) -> dict[str, str] | None:
    s = str(svc or "")
    if not s or s in {"valid.no_action", "no_action"}:
        return None
    cap = cap_of_svc(s)
    if s in {"climate.turn_off"} or s == "climate.set_hvac_mode":
        return {"capability": "ClimateControl", "operation": "Deactivate" if "off" in s or s.endswith("turn_off") else "SetMode", "service": s}
    if s.endswith(".turn_off") and cap == "Lighting":
        return {"capability": "Lighting", "operation": "Deactivate", "service": s}
    if s.endswith(".turn_on") and cap == "Lighting":
        return {"capability": "Lighting", "operation": "Activate", "service": s}
    if s.startswith("notify."):
        return {"capability": "Notify", "operation": "Emit", "service": s}
    if cap == "Record":
        return {"capability": "Record", "operation": "Append", "service": s}
    if cap == "Log":
        return {"capability": "Log", "operation": "Write", "service": s}
    return {"capability": cap, "operation": s, "service": s}

def contract_capabilities(comp: dict, ctx: dict | None) -> list[str]:
    caps: list[str] = []

    def add(c: str) -> None:
        if c and c not in caps and c not in {"Other", "GenericHA"}:
            caps.append(c)

    scene = str(comp.get("scene") or "")
    for c in scene_caps(scene):
        add(c)
    schema = (comp.get("component_action_schema") or {}).get("service") or ""
    b = svc_to_behavior(str(schema))
    if b:
        add(b["capability"])
    bt = str(((comp.get("runtime_state") or {}).get("system_state") or {}).get("behavior_target") or "")
    b2 = svc_to_behavior(bt_suffix(bt))
    if b2:
        add(b2["capability"])
    bind = comp.get("entity_binding") or {}
    for e in binding_originals(bind):
        for c in family_caps(entity_family(e)):
            add(c)
    if ctx:
        declared = str(ctx.get("declared_effect_service") or "")
        b3 = svc_to_behavior(declared)
        if b3:
            add(b3["capability"])
        for s in ctx.get("candidate_services") or []:
            b4 = svc_to_behavior(str(s))
            if b4:
                add(b4["capability"])
    return caps

def collect_raw_atoms(comp: dict) -> list[dict[str, Any]]:
    rt = comp.get("runtime_state") or {}
    obs = dict(rt.get("observed") or {})
    ents = list(rt.get("entity_observations") or [])
    atoms: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def push(atom_id: str, source: str, entity: str | None, value: Any, family: str) -> None:
        key = (atom_id, source, entity or "", str(value))
        if key in seen:
            return
        seen.add(key)
        atoms.append(
            {
                "atom_id": atom_id,
                "source": source,
                "entity": entity,
                "value": value if value in (None, True, False) or isinstance(value, (int, float, str)) else str(value),
                "family": family,
            }
        )

    for k, v in obs.items():
        if k in SKIP_OBS_KEYS or v in (None, ""):
            continue
        fam = OBS_FAMILY.get(k)
        if not fam:
            continue
        push(k, "observed", None, v, fam)

    for eo in ents:
        if not isinstance(eo, dict):
            continue
        eid = str(eo.get("entity_id") or "") or None
        st = eo.get("state")
        fam_e = entity_family(eid or "")
        attrs = eo.get("attributes") or {}
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if k in SKIP_OBS_KEYS or v in (None, ""):
                    continue
                fam = OBS_FAMILY.get(k)
                if fam:
                    push(k, "entity_observation", eid, v, fam)
        if st not in (None, "") and fam_e not in {"other", "placeholder"}:
            push("entity_state", "entity_observation", eid, st, fam_e)
        elif st not in (None, "") and fam_e == "placeholder":

            pass
    return atoms

def attribute_parent(comps: list[dict], ctxs: dict[str, dict]) -> tuple[list[dict], dict[str, dict]]:
    cid_list = [str(c.get("component_id") or "") for c in comps]
    binds = {str(c.get("component_id") or ""): c.get("entity_binding") or {} for c in comps}
    origs = {cid: binding_originals(binds[cid]) for cid in cid_list}
    fams = {cid: {entity_family(e) for e in origs[cid] if entity_family(e) != "placeholder"} for cid in cid_list}
    orig_owners: dict[str, list[str]] = defaultdict(list)
    for cid, ents in origs.items():
        for e in ents:
            orig_owners[e].append(cid)
    shared_ents = {e for e, owners in orig_owners.items() if len(set(owners)) >= 2}

    contracts = {
        str(c.get("component_id") or ""): contract_capabilities(c, ctxs.get(str(c.get("component_id") or "")))
        for c in comps
    }

    attributed: list[dict] = []
    per_comp_atoms: dict[str, list[dict]] = {cid: [] for cid in cid_list}

    family_owners: dict[str, list[str]] = defaultdict(list)
    for cid, fs in fams.items():
        for f in fs:
            family_owners[f].append(cid)

    for c in comps:
        cid = str(c.get("component_id") or "")
        bind = binds[cid]
        mine = set(origs[cid])
        mine_fams = fams[cid]
        contract = set(contracts[cid])
        raw = collect_raw_atoms(c)
        for a in raw:
            fam = a["family"]
            eid = a.get("entity")
            eid_fam = entity_family(eid or "")
            cap_rel = list(family_caps(fam))
            comp_rel = [x for x in cap_rel if x in contract]
            sibling_owners = [x for x in family_owners.get(fam, []) if x != cid]
            bound_match = fam in mine_fams
            entity_in_bind = bool(eid and eid in mine and eid not in PLACEHOLDER_ENTITIES)
            placeholder = (eid in PLACEHOLDER_ENTITIES) or eid_fam == "placeholder" or not eid
            shared = bool((eid and eid in shared_ents) or (bind.get("shared_entity") and bound_match))

            if entity_in_bind or bound_match:
                if shared:
                    prov = "shared"
                    conf = 0.7
                else:
                    prov = "direct"
                    conf = 0.9
                copied_from: list[str] = []
            elif sibling_owners:
                prov = "copied"
                conf = 0.4
                copied_from = sibling_owners
            elif placeholder and sibling_owners:
                prov = "copied"
                conf = 0.35
                copied_from = sibling_owners
            elif placeholder:
                prov = "copied"
                conf = 0.25
                copied_from = []
            else:
                prov = "copied" if sibling_owners else "direct"
                conf = 0.3 if sibling_owners else 0.55
                copied_from = sibling_owners

            row = {
                "component_id": cid,
                "atom_id": a["atom_id"],
                "source": a["source"],
                "entity": eid,
                "bound_entities": origs[cid],
                "family": fam,
                "value": a.get("value"),
                "capability_relevance": cap_rel,
                "component_relevance": comp_rel,
                "provenance": prov,
                "copied_from": copied_from,
                "confidence": conf,
            }
            attributed.append(row)
            per_comp_atoms[cid].append(row)

    meta = {"contracts": contracts, "fams": {k: sorted(v) for k, v in fams.items()}, "origs": origs}
    return attributed, {**meta, "per_comp_atoms": per_comp_atoms}

def ownership_for_component(comp: dict, atoms: list[dict], contract: list[str]) -> dict[str, Any]:
    cid = str(comp.get("component_id") or "")
    bt = str(((comp.get("runtime_state") or {}).get("system_state") or {}).get("behavior_target") or "")
    owned = []
    shared = []
    irrelevant = []
    for a in atoms:
        relevant = bool(a.get("component_relevance"))
        prov = a.get("provenance")
        if relevant and prov == "direct":
            owned.append(a["atom_id"])
        elif relevant and prov == "shared":
            shared.append(a["atom_id"])
        elif (not relevant) or prov == "copied":
            irrelevant.append(a["atom_id"])

    owned = sorted(set(owned))
    shared = sorted(set(shared))
    irrelevant = sorted(set(irrelevant))

    if owned:
        klass = "OWNED_EVIDENCE"
    elif shared:
        klass = "SHARED_RELEVANT_EVIDENCE"
    elif irrelevant:
        klass = "IRRELEVANT_SIBLING_EVIDENCE"
    else:
        klass = "NO_SUPPORTING_EVIDENCE"

    owned_fams = {a["family"] for a in atoms if a["atom_id"] in owned or (a.get("component_relevance") and a.get("provenance") == "direct")}
    shared_fams = {a["family"] for a in atoms if a.get("component_relevance") and a.get("provenance") == "shared"}
    copied_fams = {a["family"] for a in atoms if a.get("provenance") == "copied"}
    relevant_caps = set()
    for a in atoms:
        if a.get("component_relevance") and a.get("provenance") in {"direct", "shared"}:
            relevant_caps.update(a["component_relevance"])

    independent_notify = bool(owned_fams & NOTIFY_EVENT_FAMS)
    independent_light = bool(owned_fams & LIGHT_EVENT_FAMS)
    independent_climate = bool(owned_fams & CLIMATE_EVENT_FAMS)
    independent_any = independent_notify or independent_light or independent_climate

    return {
        "component_id": cid,
        "scene": str(comp.get("scene") or ""),
        "behavior_target": bt,
        "behavior_target_suffix": bt_suffix(bt),
        "no_action": is_no_action(bt),
        "contract_capabilities": contract,
        "owned_evidence": owned,
        "shared_relevant_evidence": shared,
        "irrelevant_sibling_evidence": irrelevant,
        "supporting_class": klass,
        "owned_families": sorted(owned_fams),
        "shared_families": sorted(shared_fams),
        "copied_families": sorted(copied_fams),
        "relevant_caps": sorted(relevant_caps),
        "has_independent_owned_event": independent_any,
        "independent_notify_security": independent_notify,
        "independent_lighting_event": independent_light,
        "independent_climate_event": independent_climate,
        "has_state_transition_requirement": bool(
            ("ClimateControl" in contract and (independent_climate or "climate" in owned_fams))
            or ("Lighting" in contract and (independent_light or owned_fams & {"light", "lux"}))
        ),
    }

def compose_parent(comps: list[dict], owns: list[dict]) -> dict[str, Any]:
    by_id = {o["component_id"]: o for o in owns}
    intended: list[dict] = []
    excluded: list[dict] = []
    for c in comps:
        cid = str(c.get("component_id") or "")
        o = by_id[cid]
        bt = o["behavior_target"]
        suffix = o["behavior_target_suffix"]
        beh = svc_to_behavior(suffix)
        has_owned = o["supporting_class"] in {"OWNED_EVIDENCE", "SHARED_RELEVANT_EVIDENCE"}
        has_event = o["has_independent_owned_event"]

        if o["no_action"]:
            if not has_owned and not has_event:
                excluded.append(
                    {
                        "component_id": cid,
                        "reason": "valid.no_action_no_owned_no_independent_event",
                        "supporting_class": o["supporting_class"],
                    }
                )
                continue

            for cap in o["contract_capabilities"]:
                if cap in o["relevant_caps"] or (
                    cap == "Notify" and o["independent_notify_security"]
                ) or (
                    cap == "Lighting" and o["independent_lighting_event"]
                ) or (
                    cap == "ClimateControl" and o["independent_climate_event"]
                ):
                    intended.append(
                        {
                            "capability": cap,
                            "operation": "Emit" if cap == "Notify" else "StateChange",
                            "component_id": cid,
                            "basis": "no_action_but_owned_independent_event",
                        }
                    )
            if not any(x["component_id"] == cid for x in intended):
                excluded.append(
                    {
                        "component_id": cid,
                        "reason": "valid.no_action_owned_evidence_not_contract_aligned",
                        "supporting_class": o["supporting_class"],
                    }
                )
            continue

        if beh:
            if has_owned or has_event or beh["capability"] in o["relevant_caps"]:
                intended.append(
                    {
                        "capability": beh["capability"],
                        "operation": beh["operation"],
                        "component_id": cid,
                        "basis": "behavior_target_plus_owned_or_relevant_evidence",
                        "service": beh.get("service"),
                    }
                )
            else:

                intended.append(
                    {
                        "capability": beh["capability"],
                        "operation": beh["operation"],
                        "component_id": cid,
                        "basis": "behavior_target_unsupported",
                        "service": beh.get("service"),
                    }
                )
        else:
            excluded.append(
                {
                    "component_id": cid,
                    "reason": "no_parseable_behavior_target",
                    "supporting_class": o["supporting_class"],
                }
            )

    parent_caps = sorted({x["capability"] for x in intended if x.get("basis") != "behavior_target_unsupported"})
    return {
        "parent_intended_behaviors": intended,
        "parent_intended_capabilities": parent_caps,
        "excluded_components": excluded,
    }

def idle_certify(owns: list[dict], composition: dict) -> dict[str, Any]:
    active = [x for x in composition["parent_intended_behaviors"] if x.get("basis") != "behavior_target_unsupported"]
    any_active = bool(active)
    any_owned_event = any(o["has_independent_owned_event"] for o in owns)
    any_state_req = any(o["has_state_transition_requirement"] for o in owns)
    any_sec = any(o["independent_notify_security"] for o in owns)
    if (not any_active) and (not any_owned_event) and (not any_state_req) and (not any_sec):
        return {
            "status": "CERTIFIED_IDLE",
            "reason": "no_active_intended_no_owned_event_no_state_transition_no_independent_security",
            "active_intended": 0,
            "owned_event_components": 0,
        }
    reasons = []
    if any_active:
        reasons.append("active_intended_behavior")
    if any_owned_event:
        reasons.append("component_owned_event_evidence")
    if any_state_req:
        reasons.append("state_transition_requirement")
    if any_sec:
        reasons.append("independent_notification_security_evidence")
    return {
        "status": "UNKNOWN",
        "reason": "+".join(reasons) or "insufficient",
        "active_intended": len(active),
        "owned_event_components": sum(1 for o in owns if o["has_independent_owned_event"]),
    }

def generic_gate_v2(
    action: dict,
    siblings: list[dict],
    own: dict | None,
    composition: dict,
) -> dict[str, Any]:

    svc = str(action.get("service") or "")
    cap = cap_of_svc(svc)
    origin = str(action.get("component_origin") or "")
    intended_here = {
        x["capability"]
        for x in composition["parent_intended_behaviors"]
        if x.get("component_id") == origin and x.get("basis") != "behavior_target_unsupported"
    }
    intended_parent = set(composition.get("parent_intended_capabilities") or [])
    o = own or {}
    owned_rel = cap in set(o.get("relevant_caps") or [])
    copied_fams = set(o.get("copied_families") or [])
    irrelevant = o.get("supporting_class") == "IRRELEVANT_SIBLING_EVIDENCE"
    no_support = o.get("supporting_class") == "NO_SUPPORTING_EVIDENCE"
    independent = False
    if cap == "Notify":
        independent = bool(o.get("independent_notify_security"))
    elif cap == "Lighting":
        independent = bool(o.get("independent_lighting_event"))
    elif cap == "ClimateControl":
        independent = bool(o.get("independent_climate_event"))
    elif cap == "Record":
        independent = "power" in set(o.get("owned_families") or [])

    sibling_copied = bool(copied_fams) and irrelevant and not owned_rel
    bits = {
        "has_sibling": bool(siblings),
        "in_component_intended": cap in intended_here,
        "in_parent_intended": cap in intended_parent,
        "owned_or_shared_relevant": owned_rel,
        "independent_event_or_state": independent,
        "sibling_copied_or_irrelevant": sibling_copied or (irrelevant and not owned_rel),
        "no_supporting_evidence": no_support,
        "no_action_origin": bool(o.get("no_action")),
    }

    if not siblings:
        return {"decision": "ABSTAIN", "reason": "NO_SIBLING_NOT_PARENT_LEVEL", "confidence": 0.0, "bits": bits}

    if cap in intended_here and (owned_rel or independent):
        return {"decision": "KEEP", "reason": "INTENDED_AND_OWNED_OR_INDEPENDENT", "confidence": 0.9, "bits": bits}
    if independent and cap in set(o.get("contract_capabilities") or []):
        return {"decision": "KEEP", "reason": "INDEPENDENT_OWNED_EVENT_FOR_CONTRACT", "confidence": 0.85, "bits": bits}

    not_intended = cap not in intended_here
    no_local = (not owned_rel) and (not independent)
    contamination = sibling_copied or (no_support and bool(o.get("no_action")))
    if not_intended and no_local and contamination and siblings:
        return {
            "decision": "SET_REMOVE",
            "reason": "NO_INTENDED_NO_OWNED_SIBLING_COPIED_OR_UNSUPPORTED",
            "confidence": 0.75 if sibling_copied else 0.6,
            "bits": bits,
        }

    return {"decision": "ABSTAIN", "reason": "INSUFFICIENT_EVIDENCE_DEFAULT_ABSTAIN", "confidence": 0.3, "bits": bits}

def apply_generic_gate(
    actions: list[dict],
    owns_by_id: dict[str, dict],
    composition: dict,
) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    trace: list[dict] = []
    for i, a in enumerate(actions):
        siblings = [x for j, x in enumerate(actions) if j != i]
        origin = str(a.get("component_origin") or "")
        g = generic_gate_v2(a, siblings, owns_by_id.get(origin), composition)
        row = {"service": a.get("service"), "component_origin": origin, **g}
        trace.append(row)
        if g["decision"] != "SET_REMOVE":
            kept.append(a)
    return kept, trace

def residual_flags(pred_toks: list, gold_toks: list) -> dict[str, Any]:
    extra = Counter(pred_toks) - Counter(gold_toks)
    missing = Counter(gold_toks) - Counter(pred_toks)
    extra_n = sum(extra.values())
    missing_n = sum(missing.values())
    gold_idle = not gold_toks
    pred_idle = not pred_toks
    extra_caps = [cap_of_svc(t[0]) for t, n in extra.items() for _ in range(n)]
    missing_caps = [cap_of_svc(t[0]) for t, n in missing.items() for _ in range(n)]
    extra_svc = [t[0] for t, n in extra.items() for _ in range(n)]
    missing_svc = [t[0] for t, n in missing.items() for _ in range(n)]
    wrong = extra_n > 0 and missing_n > 0
    clim_pred = [t for t in pred_toks if str(t[0]).startswith("climate.")]
    combined = False
    if len(clim_pred) >= 2:
        modes = set()
        for t in clim_pred:
            if t[0] == "climate.turn_off":
                modes.add("off")
            elif t[0] == "climate.set_hvac_mode":
                try:
                    p = json.loads(t[2]) if t[2] else {}
                except json.JSONDecodeError:
                    p = {}
                modes.add(str(p.get("hvac_mode") or "").lower())
        if "off" in modes and (modes - {"off", ""}):
            combined = True
    return {
        "extra": extra_n > 0,
        "missing": missing_n > 0,
        "wrong": wrong,
        "idle_residual": gold_idle and not pred_idle,
        "combined_state": combined,
        "extra_n": extra_n,
        "missing_n": missing_n,
        "extra_caps": extra_caps,
        "missing_caps": missing_caps,
        "extra_svc": extra_svc,
        "missing_svc": missing_svc,
    }

def finish_gate_bucket(b: dict, extra: dict | None = None) -> dict[str, Any]:
    out = finish_bucket(b)
    if extra:
        out.update(extra)
    return out

def main() -> None:
    print("load SS + traces...", flush=True)
    traces = load_jsonl(TRACE)
    ss_by_id: dict[str, dict] = {}
    with SS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                ss_by_id[str(rec.get("sample_id") or "")] = rec
    ctx_cache: dict[str, dict] = {}

    def ctx_of(sid: str) -> dict:
        if sid not in ctx_cache:
            ctx_cache[sid] = build_repair_context(ss_by_id[sid])
        return ctx_cache[sid]

    def independent_merge(comps: list[dict], bind_all: dict) -> list[dict]:
        merged = []
        for c in comps:
            sid = str(c.get("single_scene_sample_id") or "")
            origin = str(c.get("component_id") or "")
            fr = compact_action((traces.get(sid) or {}).get("repaired_action"))
            out, _k = apply_strict_v2(ctx_of(sid), fr)
            mapped = map_repaired(out, bind_all.get(origin) or {})
            if mapped:
                mapped["component_origin"] = origin
                merged.append(mapped)
        return merged

    buckets = {k: acc_bucket() for k in ("A", "B", "C", "D", "D_idle")}
    gate_c = Counter()
    gate_d = Counter()
    reason_c = Counter()
    reason_d = Counter()
    attr_prov = Counter()
    attr_fam = Counter()
    attr_cap = Counter()
    copied_mismatch = 0
    owned_window = copied_window = 0
    own_class = Counter()
    intended_caps = Counter()
    excluded_reason = Counter()
    idle_status = Counter()
    leakage_extra_actions = 0
    extra_after_b = 0
    leakage_after_b = 0

    m2c_A = m2c_B = m2c_C = m2c_D = m2c_Di = 0
    m2c_A_exact_D = m2c_A_exact_Di = 0
    m2e_A = 0
    m2e_D_exact = m2e_Di_exact = 0
    m2e_certified = 0
    gold_idle_n = 0
    certified_n = 0
    certified_gold_idle = 0
    certified_gold_active = 0
    d_broke_b = d_fixed_b = 0
    d_broke_c = c_broke_d = 0
    di_broke_d = di_fixed_d = 0
    di_broke_d_b0ok = 0
    new_fr_d_vs_b = new_fr_di_vs_d = 0

    residual_counts = Counter()
    residual_extra_cap = Counter()
    residual_extra_svc = Counter()
    residual_missing_cap = Counter()
    residual_idle_cap = Counter()

    examples_copied = []
    examples_idle_false = []
    examples_m2c_d = []

    attr_lines: list[str] = []
    own_lines: list[str] = []
    comp_lines: list[str] = []
    idle_rows: list[dict] = []

    print("walk MA...", flush=True)
    n_parent = 0
    with MA.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_parent += 1
            mid = str(rec.get("multi_action_id") or "")
            comps = list(rec.get("components") or [])
            bind_all = {str(c.get("component_id") or ""): c.get("entity_binding") or {} for c in comps}
            parent_acts = list(((rec.get("formal_b0") or {}).get("parent") or {}).get("semantic_actions") or [])
            gold = ma_y_acts(rec)
            gold_toks = toks_from_acts(gold, bind_all)
            gold_idle = not gold_toks
            if gold_idle:
                gold_idle_n += 1

            ctxs = {}
            for c in comps:
                sid = str(c.get("single_scene_sample_id") or "")
                cid = str(c.get("component_id") or "")
                ctxs[cid] = ctx_of(sid)

            atoms, meta = attribute_parent(comps, ctxs)
            owns = []
            for c in comps:
                cid = str(c.get("component_id") or "")
                o = ownership_for_component(c, meta["per_comp_atoms"].get(cid) or [], meta["contracts"].get(cid) or [])
                owns.append(o)
                own_class[o["supporting_class"]] += 1
            owns_by_id = {o["component_id"]: o for o in owns}
            composition = compose_parent(comps, owns)
            for it in composition["parent_intended_behaviors"]:
                intended_caps[it["capability"]] += 1
            for ex in composition["excluded_components"]:
                excluded_reason[ex["reason"]] += 1
            idle = idle_certify(owns, composition)
            idle_status[idle["status"]] += 1

            for a in atoms:
                attr_prov[a["provenance"]] += 1
                attr_fam[a["family"]] += 1
                for cap in a["capability_relevance"]:
                    attr_cap[cap] += 1
                if a["family"] == "window":
                    if a["provenance"] in {"direct", "shared"} and a.get("component_relevance"):
                        owned_window += 1
                    elif a["provenance"] == "copied":
                        copied_window += 1
                if a["provenance"] == "copied" and not a.get("component_relevance"):
                    copied_mismatch += 1
                    if len(examples_copied) < 8:
                        examples_copied.append(
                            {
                                "multi_action_id": mid,
                                "component_id": a["component_id"],
                                "atom_id": a["atom_id"],
                                "family": a["family"],
                                "entity": a["entity"],
                                "bound": a["bound_entities"],
                                "copied_from": a["copied_from"],
                                "capability_relevance": a["capability_relevance"],
                            }
                        )

            attr_lines.append(
                json.dumps(
                    {
                        "multi_action_id": mid,
                        "composition_type": rec.get("composition_type"),
                        "n_atoms": len(atoms),
                        "atoms": atoms,
                    },
                    ensure_ascii=False,
                )
            )
            own_lines.append(
                json.dumps(
                    {
                        "multi_action_id": mid,
                        "components": owns,
                    },
                    ensure_ascii=False,
                )
            )
            comp_lines.append(
                json.dumps(
                    {
                        "multi_action_id": mid,
                        "composition_type": rec.get("composition_type"),
                        **composition,
                        "idle_certification": idle["status"],
                    },
                    ensure_ascii=False,
                )
            )

            b0_toks = toks_from_acts(parent_acts, bind_all)
            mB0 = set_metrics(b0_toks, gold_toks)
            b0_ok = bool(mB0["set_exact"])

            merged = independent_merge(comps, bind_all)
            a_toks = toks_from_acts(merged, bind_all)
            mA = set_metrics(a_toks, gold_toks)

            b_acts, b_trace = set_remove_v1(merged, bind_all)
            b_toks = toks_from_acts(b_acts, bind_all)
            mB = set_metrics(b_toks, gold_toks)
            b_fired = any(row["removed"] for row in b_trace)

            c_acts, c_trace = apply_gate(b_acts, comps, rec, bind_all, DEFAULT_MIN_CORE)
            c_toks = toks_from_acts(c_acts, bind_all)
            mC = set_metrics(c_toks, gold_toks)
            for row in c_trace:
                gate_c[row["decision"]] += 1
                reason_c[row.get("reason") or ""] += 1

            d_acts, d_trace = apply_generic_gate(b_acts, owns_by_id, composition)
            d_toks = toks_from_acts(d_acts, bind_all)
            mD = set_metrics(d_toks, gold_toks)
            d_fired = any(row["decision"] == "SET_REMOVE" for row in d_trace)
            for row in d_trace:
                gate_d[row["decision"]] += 1
                reason_d[row.get("reason") or ""] += 1

            if idle["status"] == "CERTIFIED_IDLE":
                di_acts: list[dict] = []
            else:
                di_acts = list(d_acts)
            di_toks = toks_from_acts(di_acts, bind_all)
            mDi = set_metrics(di_toks, gold_toks)

            add_bucket(buckets["A"], mA, b0_ok, bool(mA["set_exact"]), mA["n_pred"])
            add_bucket(buckets["B"], mB, b0_ok, bool(mB["set_exact"]), mB["n_pred"], fired=b_fired)
            add_bucket(buckets["C"], mC, b0_ok, bool(mC["set_exact"]), mC["n_pred"], fired=any(r["decision"] == "SET_REMOVE" for r in c_trace))
            add_bucket(buckets["D"], mD, b0_ok, bool(mD["set_exact"]), mD["n_pred"], fired=d_fired)
            add_bucket(buckets["D_idle"], mDi, b0_ok, bool(mDi["set_exact"]), mDi["n_pred"], fired=idle["status"] == "CERTIFIED_IDLE")

            if mB["set_exact"] and not mD["set_exact"]:
                d_broke_b += 1
                if b0_ok:
                    new_fr_d_vs_b += 1
            if (not mB["set_exact"]) and mD["set_exact"]:
                d_fixed_b += 1
            if mC["set_exact"] and not mD["set_exact"]:
                d_broke_c += 1
            if mD["set_exact"] and not mC["set_exact"]:
                c_broke_d += 1
            if mD["set_exact"] and not mDi["set_exact"]:
                di_broke_d += 1
                if b0_ok:
                    di_broke_d_b0ok += 1
                    new_fr_di_vs_d += 1
            if (not mD["set_exact"]) and mDi["set_exact"]:
                di_fixed_d += 1

            extra_b, extra_acts_b = extra_pairs(b_acts, gold_toks, bind_all)
            extra_after_b += len(extra_b)
            for a, t in zip(extra_acts_b, extra_b):
                origin = str(a.get("component_origin") or "")
                o = owns_by_id.get(origin) or {}
                if o.get("supporting_class") in {"IRRELEVANT_SIBLING_EVIDENCE", "NO_SUPPORTING_EVIDENCE"} or (
                    cap_of_svc(str(a.get("service") or "")) not in set(o.get("relevant_caps") or [])
                    and o.get("copied_families")
                ):
                    leakage_after_b += 1

            extra_a, extra_acts_a = extra_pairs(merged, gold_toks, bind_all)
            for a in extra_acts_a:
                origin = str(a.get("component_origin") or "")
                o = owns_by_id.get(origin) or {}
                if o.get("supporting_class") == "IRRELEVANT_SIBLING_EVIDENCE":
                    leakage_extra_actions += 1

            m2_tag = None
            if extra_a:
                m2_tag = classify_m2(extra_a, gold_toks, b0_toks, extra_acts_a, gold_idle)
            if m2_tag == "M2-C_contextually_unnecessary_component":
                m2c_A += 1
                if mD["set_exact"]:
                    m2c_A_exact_D += 1
                    if len(examples_m2c_d) < 6:
                        examples_m2c_d.append({"multi_action_id": mid, "stage": "D"})
                if mDi["set_exact"]:
                    m2c_A_exact_Di += 1
            extra_b2, extra_acts_b2 = extra_pairs(b_acts, gold_toks, bind_all)
            if extra_b2:
                tag_b = classify_m2(extra_b2, gold_toks, b0_toks, extra_acts_b2, gold_idle)
                if tag_b == "M2-C_contextually_unnecessary_component":
                    m2c_B += 1
            extra_c, extra_acts_c = extra_pairs(c_acts, gold_toks, bind_all)
            if extra_c:
                tag_c = classify_m2(extra_c, gold_toks, b0_toks, extra_acts_c, gold_idle)
                if tag_c == "M2-C_contextually_unnecessary_component":
                    m2c_C += 1
            extra_d, extra_acts_d = extra_pairs(d_acts, gold_toks, bind_all)
            if extra_d:
                tag_d = classify_m2(extra_d, gold_toks, b0_toks, extra_acts_d, gold_idle)
                if tag_d == "M2-C_contextually_unnecessary_component":
                    m2c_D += 1
            extra_di, extra_acts_di = extra_pairs(di_acts, gold_toks, bind_all)
            if extra_di:
                tag_di = classify_m2(extra_di, gold_toks, b0_toks, extra_acts_di, gold_idle)
                if tag_di == "M2-C_contextually_unnecessary_component":
                    m2c_Di += 1

            if gold_idle:
                m2e_A += 1 if (extra_a or (not mA["set_exact"])) else 0
                if mD["set_exact"]:
                    m2e_D_exact += 1
                if mDi["set_exact"]:
                    m2e_Di_exact += 1
                if idle["status"] == "CERTIFIED_IDLE":
                    m2e_certified += 1

            if idle["status"] == "CERTIFIED_IDLE":
                certified_n += 1
                idle_rows.append(
                    {
                        "multi_action_id": mid,
                        "status": idle["status"],
                        "gold_idle": gold_idle,
                        "d_exact": bool(mD["set_exact"]),
                        "di_exact": bool(mDi["set_exact"]),
                        "reason": idle["reason"],
                    }
                )
                if gold_idle:
                    certified_gold_idle += 1
                else:
                    certified_gold_active += 1
                    if len(examples_idle_false) < 8:
                        examples_idle_false.append(
                            {
                                "multi_action_id": mid,
                                "gold_services": [t[0] for t in gold_toks],
                                "reason": idle["reason"],
                                "excluded": composition["excluded_components"],
                            }
                        )

            flags = residual_flags(di_toks, gold_toks)
            if flags["extra"]:
                residual_counts["extra"] += 1
                residual_extra_cap.update(flags["extra_caps"])
                residual_extra_svc.update(flags["extra_svc"])
            if flags["missing"]:
                residual_counts["missing"] += 1
                residual_missing_cap.update(flags["missing_caps"])
            if flags["wrong"]:
                residual_counts["wrong"] += 1
            if flags["idle_residual"]:
                residual_counts["idle_residual"] += 1
                residual_idle_cap.update(flags["extra_caps"])
            if flags["combined_state"]:
                residual_counts["combined_state_residual"] += 1
            if not any(flags[k] for k in ("extra", "missing", "wrong", "idle_residual", "combined_state")):
                residual_counts["clean"] += 1

    n_gate_d = sum(gate_d.values()) or 1
    n_gate_c = sum(gate_c.values()) or 1

    eval_obj = {
        "note": "Sidecar only. Frozen v1 / Strict-v2 / SET_REMOVE v1 / Necessity Gate v1 unchanged.",
        "N": n_parent,
        "stages": {
            "A_strict_v2_independent": finish_bucket(buckets["A"]),
            "B_plus_set_remove_v1": finish_bucket(buckets["B"]),
            "C_plus_current_necessity_gate": finish_gate_bucket(
                buckets["C"],
                {
                    "ABSTAIN_count": gate_c["ABSTAIN"],
                    "KEEP_count": gate_c["KEEP"],
                    "SET_REMOVE_count": gate_c["SET_REMOVE"],
                    "ABSTAIN_rate": round(gate_c["ABSTAIN"] / n_gate_c, 4),
                    "reasons": dict(reason_c),
                },
            ),
            "D_plus_generic_necessity_gate_v2": finish_gate_bucket(
                buckets["D"],
                {
                    "ABSTAIN_count": gate_d["ABSTAIN"],
                    "KEEP_count": gate_d["KEEP"],
                    "SET_REMOVE_count": gate_d["SET_REMOVE"],
                    "ABSTAIN_rate": round(gate_d["ABSTAIN"] / n_gate_d, 4),
                    "reasons": dict(reason_d),
                    "fixed_from_B": d_fixed_b,
                    "broke_B": d_broke_b,
                    "new_false_repair_vs_B_correct_B0": new_fr_d_vs_b,
                    "vs_C_fixed": c_broke_d,
                    "vs_C_broke": d_broke_c,
                },
            ),
            "D_plus_idle_certification": finish_gate_bucket(
                buckets["D_idle"],
                {
                    "fixed_from_D": di_fixed_d,
                    "broke_D": di_broke_d,
                    "broke_D_and_correct_B0": di_broke_d_b0ok,
                    "new_false_repair_from_idle_vs_D": new_fr_di_vs_d,
                },
            ),
        },
        "m2c": {
            "after_strict_v2": m2c_A,
            "after_set_remove": m2c_B,
            "after_current_gate": m2c_C,
            "after_generic_gate": m2c_D,
            "after_generic_plus_idle": m2c_Di,
            "strict_v2_m2c_exact_at_D": m2c_A_exact_D,
            "strict_v2_m2c_exact_at_D_idle": m2c_A_exact_Di,
            "coverage_of_strict_v2_m2c_at_D": pct(m2c_A_exact_D, m2c_A),
        },
        "m2e": {
            "gold_idle_parents": gold_idle_n,
            "exact_at_D": m2e_D_exact,
            "exact_at_D_idle": m2e_Di_exact,
            "certified_idle_and_gold_idle": m2e_certified,
        },
        "leakage": {
            "extra_actions_after_strict_v2_from_irrelevant_sibling": leakage_extra_actions,
            "extra_actions_after_set_remove": extra_after_b,
            "extra_after_set_remove_with_leakage_ownership": leakage_after_b,
            "leakage_share_of_B_extras": pct(leakage_after_b, extra_after_b),
        },
        "attribution": {
            "provenance": dict(attr_prov),
            "family": dict(attr_fam),
            "capability_relevance": dict(attr_cap),
            "window_owned_or_shared_relevant": owned_window,
            "window_copied": copied_window,
            "copied_and_not_component_relevant": copied_mismatch,
            "examples_copied": examples_copied,
        },
        "ownership_class": dict(own_class),
        "intended_capabilities": dict(intended_caps),
        "excluded_reason": dict(excluded_reason),
        "gate_v1_decisions": dict(gate_c),
        "gate_v2_decisions": dict(gate_d),
        "gate_v1_reasons": dict(reason_c),
        "gate_v2_reasons": dict(reason_d),
    }
    dump(EVAL_JSON, eval_obj)

    idle_prec = pct(certified_gold_idle, certified_n)
    idle_cov = pct(certified_gold_idle, gold_idle_n)
    idle_obj = {
        "note": "Y-blind CERTIFIED_IDLE. Gold used only for precision/coverage eval.",
        "N": n_parent,
        "CERTIFIED_IDLE": certified_n,
        "UNKNOWN": n_parent - certified_n,
        "gold_idle_parents": gold_idle_n,
        "precision": round(certified_gold_idle / certified_n, 4) if certified_n else None,
        "precision_pct": idle_prec,
        "coverage": round(certified_gold_idle / gold_idle_n, 4) if gold_idle_n else None,
        "coverage_pct": idle_cov,
        "certified_and_gold_idle": certified_gold_idle,
        "false_idle_certification": certified_gold_active,
        "false_idle_rate": pct(certified_gold_active, certified_n),
        "broken_exact_parent_from_D": di_broke_d,
        "broken_exact_and_correct_B0": di_broke_d_b0ok,
        "gold_idle_made_exact_by_idle_vs_D": di_fixed_d,
        "status_counts": dict(idle_status),
        "false_idle_examples": examples_idle_false,
        "certified_sample": idle_rows[:12],
    }
    dump(IDLE_JSON, idle_obj)

    resid_obj = {
        "note": "Residual after Strict-v2 + SET_REMOVE v1 + Generic Necessity Gate v2 + Idle Certification. Gold eval only.",
        "N": n_parent,
        "inexact": n_parent - buckets["D_idle"]["set_exact"],
        "parent_counts": dict(residual_counts),
        "extra_capability": dict(residual_extra_cap),
        "extra_service": dict(residual_extra_svc.most_common(20)),
        "missing_capability": dict(residual_missing_cap),
        "idle_residual_extra_capability": dict(residual_idle_cap),
        "pipeline": {
            "A_strict_v2_exact": buckets["A"]["set_exact"],
            "B_set_remove_exact": buckets["B"]["set_exact"],
            "C_current_gate_exact": buckets["C"]["set_exact"],
            "D_generic_gate_exact": buckets["D"]["set_exact"],
            "D_idle_exact": buckets["D_idle"]["set_exact"],
        },
    }
    dump(RESID_JSON, resid_obj)

    ATTR_JSONL.write_text("\n".join(attr_lines) + "\n", encoding="utf-8")
    OWN_JSONL.write_text("\n".join(own_lines) + "\n", encoding="utf-8")
    COMP_JSONL.write_text("\n".join(comp_lines) + "\n", encoding="utf-8")

    a = eval_obj["stages"]["A_strict_v2_independent"]
    b = eval_obj["stages"]["B_plus_set_remove_v1"]
    c = eval_obj["stages"]["C_plus_current_necessity_gate"]
    d = eval_obj["stages"]["D_plus_generic_necessity_gate_v2"]
    di = eval_obj["stages"]["D_plus_idle_certification"]

    attr_md = f"""# Evidence Attribution Analysis

Sidecar on frozen MA N={n_parent}. Does not read Gold. Frozen Repair v1 / Strict-v2 / SET_REMOVE v1 / Necessity Gate v1 unchanged.

## Method

Each observation / entity-state atom is typed by **family** (window, door, motion, climate, lux, person, power), then mapped to **capability relevance**:

| Evidence family | Capability relevance |
| --- | --- |
| window, climate | ClimateControl |
| door, person | Notify |
| motion, lux, light | Lighting |
| power | Record |

`window_open` therefore supports ClimateControl. It does **not** automatically support Notify.

Provenance (per component):

- **direct** — atom family matches this component's bound entities, or the observed entity is in this binding.
- **shared** — the same original entity is bound in two or more components.
- **copied** — the atom is present on this component, but a sibling owns the family, or the entity is the composition placeholder `binary_sensor.grounded` while this binding does not own that family.

Component relevance is the intersection of atom capability and the component's Y-blind contract (entity domain + automation contract + `behavior_target` + scene semantics). Gold action / Gold decision are not used.

## Corpus counts

| Provenance | atoms |
| --- | ---: |
"""
    for k, v in attr_prov.most_common():
        attr_md += f"| {k} | {v} |\n"
    attr_md += "\n| Family | atoms |\n| --- | ---: |\n"
    for k, v in attr_fam.most_common():
        attr_md += f"| {k} | {v} |\n"
    attr_md += f"""
Window atoms: owned/shared-and-component-relevant = {owned_window}; copied = {copied_window}.

Copied atoms that are **not** component-relevant (sibling evidence contamination): {copied_mismatch}.

## Leakage pattern

Typical contamination: a Notify/security component is bound to `binary_sensor.door_front` but observes `window_state=open` on `binary_sensor.grounded`, while a sibling ClimateControl component owns the window/climate entities. The window atom is ClimateControl-relevant and must not be treated as owned Notify evidence.

Examples:

"""
    for ex in examples_copied[:5]:
        attr_md += f"- `{ex['multi_action_id']}` {ex['component_id']} atom `{ex['atom_id']}` family={ex['family']} entity={ex['entity']} bound={ex['bound']} copied_from={ex['copied_from']} caps={ex['capability_relevance']}\n"

    attr_md += f"""
## Ownership class (components, N={sum(own_class.values())})

| Class | n |
| --- | ---: |
"""
    for k, v in own_class.most_common():
        attr_md += f"| {k} | {v} |\n"

    write_md(ATTR_MD, attr_md)

    gate_md = f"""# Generic Necessity Gate v2

Y-blind parent-level gate. **No HVAC+Notify (or any other service-pair) special case.** Frozen operators unchanged.

## Inputs

For each candidate action after Strict-v2 independent merge + SET_REMOVE v1:

1. Component intended behavior (from `behavior_target` + contract + owned evidence).
2. Owned / shared-relevant evidence for that capability.
3. Independent owned event or state basis for that capability.
4. Whether support is only sibling-copied / capability-irrelevant.

`behavior_target = valid.no_action` with no owned evidence and no independent event contributes **nothing** to the parent intended set.

## Decisions

| Decision | When |
| --- | --- |
| KEEP | Capability is in this component's intended set **and** (owned/shared-relevant evidence or independent owned event); or an independent owned event matches the component contract. |
| SET_REMOVE | Capability is **not** in this component's intended set, there is no owned/relevant evidence and no independent event, and support is sibling-copied / irrelevant **or** the origin is `valid.no_action` with no supporting evidence. Requires at least one sibling (parent-level). |
| ABSTAIN | Default. No siblings, or intended-but-unsupported, or mixed/insufficient evidence. |

No action is deleted because it is Notify next to HVAC. Notify is deleted only when attribution says it is not intended and not evidentially owned.

## Offline result (MA N={n_parent})

| Stage | Parent set-exact | P | R | F1 | FR | Pres | ABSTAIN rate | SET_REMOVE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A Strict-v2 independent | {a['Parent_Set_Exact_Match']:.4f} ({a['Parent_Set_Exact_count']}) | {a['Action_Precision']:.4f} | {a['Action_Recall']:.4f} | {a['Action_F1']:.4f} | {a['False_Parent_Repair']:.4f} | {a['Preservation']} | — | — |
| B + SET_REMOVE v1 | {b['Parent_Set_Exact_Match']:.4f} ({b['Parent_Set_Exact_count']}) | {b['Action_Precision']:.4f} | {b['Action_Recall']:.4f} | {b['Action_F1']:.4f} | {b['False_Parent_Repair']:.4f} | {b['Preservation']} | — | {b.get('set_remove_fired', 0)} parents |
| C + current Necessity Gate | {c['Parent_Set_Exact_Match']:.4f} ({c['Parent_Set_Exact_count']}) | {c['Action_Precision']:.4f} | {c['Action_Recall']:.4f} | {c['Action_F1']:.4f} | {c['False_Parent_Repair']:.4f} | {c['Preservation']} | {c.get('ABSTAIN_rate')} | {c.get('SET_REMOVE_count')} actions |
| D + Generic Gate v2 | {d['Parent_Set_Exact_Match']:.4f} ({d['Parent_Set_Exact_count']}) | {d['Action_Precision']:.4f} | {d['Action_Recall']:.4f} | {d['Action_F1']:.4f} | {d['False_Parent_Repair']:.4f} | {d['Preservation']} | {d.get('ABSTAIN_rate')} | {d.get('SET_REMOVE_count')} actions |

Generic v2 vs current gate: set-exact {d['Parent_Set_Exact_count']} vs {c['Parent_Set_Exact_count']} ({pct(d['Parent_Set_Exact_count'] - c['Parent_Set_Exact_count'], n_parent):+.2f} pp vs C). Broke C-exact: {d_broke_c}. New FR vs B (correct B0): {new_fr_d_vs_b}.

Gate v2 decision counts: {dict(gate_d)}

Gate v2 reasons: {dict(reason_d)}
"""
    write_md(GATE_MD, gate_md)

    leak_share = pct(leakage_after_b, extra_after_b)
    gain_vs_c = d["Parent_Set_Exact_Match"] - c["Parent_Set_Exact_Match"]
    gain_vs_b = d["Parent_Set_Exact_Match"] - b["Parent_Set_Exact_Match"]
    need_new_op = (
        residual_counts["missing"] > 50
        or residual_counts["idle_residual"] > 200
        or residual_counts["wrong"] > 50
    )
    decision = f"""# MA behavior composition decision

Sidecar. Frozen SS Repair v1, Strict Repair-v2, Gold Y, original MA B0, SET_REMOVE v1, Necessity Gate v1 **not modified**. SET_COMPLETE / SET_REFINE not adopted. No per-service special-case rules.

## Metrics (N={n_parent})

| | Parent set-exact | Action P | Action R | Action F1 | False Parent Repair | Preservation | ABSTAIN rate | SET_REMOVE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A Strict-v2 independent | {pct(a['Parent_Set_Exact_count'], n_parent)}% ({a['Parent_Set_Exact_count']}) | {a['Action_Precision']:.4f} | {a['Action_Recall']:.4f} | {a['Action_F1']:.4f} | {a['False_Parent_Repair']:.4f} ({a['False_Parent_Repair_count']}) | {a['Preservation']} | — | — |
| B + SET_REMOVE v1 | {pct(b['Parent_Set_Exact_count'], n_parent)}% ({b['Parent_Set_Exact_count']}) | {b['Action_Precision']:.4f} | {b['Action_Recall']:.4f} | {b['Action_F1']:.4f} | {b['False_Parent_Repair']:.4f} ({b['False_Parent_Repair_count']}) | {b['Preservation']} | — | {b.get('set_remove_fired', 0)} parents |
| C + current Necessity Gate | {pct(c['Parent_Set_Exact_count'], n_parent)}% ({c['Parent_Set_Exact_count']}) | {c['Action_Precision']:.4f} | {c['Action_Recall']:.4f} | {c['Action_F1']:.4f} | {c['False_Parent_Repair']:.4f} ({c['False_Parent_Repair_count']}) | {c['Preservation']} | {c.get('ABSTAIN_rate')} | {c.get('SET_REMOVE_count')} |
| D + Generic Necessity Gate v2 | {pct(d['Parent_Set_Exact_count'], n_parent)}% ({d['Parent_Set_Exact_count']}) | {d['Action_Precision']:.4f} | {d['Action_Recall']:.4f} | {d['Action_F1']:.4f} | {d['False_Parent_Repair']:.4f} ({d['False_Parent_Repair_count']}) | {d['Preservation']} | {d.get('ABSTAIN_rate')} | {d.get('SET_REMOVE_count')} |
| D + Idle Certification | {pct(di['Parent_Set_Exact_count'], n_parent)}% ({di['Parent_Set_Exact_count']}) | {di['Action_Precision']:.4f} | {di['Action_Recall']:.4f} | {di['Action_F1']:.4f} | {di['False_Parent_Repair']:.4f} ({di['False_Parent_Repair_count']}) | {di['Preservation']} | — | idle wipe {certified_n} |

## Answers

### 1. How many extra actions come from evidence leakage / sibling contamination?

After SET_REMOVE v1, extra actions vs Gold = {extra_after_b}. Of those, {leakage_after_b} ({leak_share}%) originate in a component whose supporting class is IRRELEVANT_SIBLING / NO_SUPPORTING, or whose action capability is not in owned/relevant evidence while copied families are present.

After Strict-v2, extra actions from IRRELEVANT_SIBLING_EVIDENCE origins = {leakage_extra_actions}.

Copied and not component-relevant atoms = {copied_mismatch}. Window copied = {copied_window}; window owned/shared-relevant = {owned_window}.

Leakage is real and concentrated: Notify (and similar) components inherit `window_state` via `binary_sensor.grounded` while bound to a door entity. Window is ClimateControl-relevant only.

### 2. Can generic Evidence Attribution replace the current HVAC+Notify special case?

Yes, as the SET_REMOVE criterion. Generic Gate v2 never tests `sibling_hvac_off ∧ Notify`. It tests intended-behavior membership, owned vs copied provenance, and capability relevance.

M2-C parents after Strict-v2: {m2c_A}. Exact at Generic Gate v2: {m2c_A_exact_D} ({pct(m2c_A_exact_D, m2c_A)}% of those M2-C). Residual M2-C after current gate: {m2c_C}; after generic gate: {m2c_D}.

If D's set-exact is at least C's and FR does not rise, the named HVAC+Notify rule is unnecessary.

C set-exact {c['Parent_Set_Exact_count']} vs D {d['Parent_Set_Exact_count']}. D broke C-exact: {d_broke_c}.

### 3. How much does Generic Necessity Gate gain vs the current gate?

Vs C: {gain_vs_c:+.4f} absolute ({pct(d['Parent_Set_Exact_count'] - c['Parent_Set_Exact_count'], n_parent):+.2f} pp), {d['Parent_Set_Exact_count'] - c['Parent_Set_Exact_count']:+d} exact parents.

Vs B (SET_REMOVE only): {gain_vs_b:+.4f} absolute ({pct(d['Parent_Set_Exact_count'] - b['Parent_Set_Exact_count'], n_parent):+.2f} pp), {d['Parent_Set_Exact_count'] - b['Parent_Set_Exact_count']:+d} exact parents. Current gate vs B was +{c['Parent_Set_Exact_count'] - b['Parent_Set_Exact_count']} exact.

### 4. Does it introduce new False Repair?

Generic Gate v2 vs B, among correct-B0 parents that B preserved: new FR = {new_fr_d_vs_b}. D FR rate {d['False_Parent_Repair']:.4f} ({d['False_Parent_Repair_count']}) vs C {c['False_Parent_Repair']:.4f} ({c['False_Parent_Repair_count']}) vs B {b['False_Parent_Repair']:.4f} ({b['False_Parent_Repair_count']}).

Idle Certification on top of D: broke D-exact {di_broke_d} (of which correct-B0 {di_broke_d_b0ok}); new FR vs D = {new_fr_di_vs_d}. Idle is **not** auto-promoted if FR or broken exact is non-zero.

### 5. How many Gold-idle parents can be Y-blind certified?

Gold-idle parents: {gold_idle_n} / {n_parent}.

CERTIFIED_IDLE: {certified_n}. Precision (certified ∧ Gold-idle) / certified = {idle_prec}% ({certified_gold_idle}/{certified_n}). Coverage (certified ∧ Gold-idle) / Gold-idle = {idle_cov}% ({certified_gold_idle}/{gold_idle_n}).

False idle certification (certified but Gold has actions): {certified_gold_active}.

Gold-idle exact at Generic Gate v2 (no idle wipe): {m2e_D_exact}. After idle wipe: {m2e_Di_exact}.

Idle Certification is Y-blind and therefore cannot chase Gold-idle. Precision/coverage above is the honest ceiling of the stated four conditions.

### 6. Does MA still need a new operator next?

Residual after Strict-v2 + SET_REMOVE + Generic Gate v2 + Idle Certification:

| Residual | parents |
| --- | ---: |
| extra | {residual_counts['extra']} |
| missing | {residual_counts['missing']} |
| wrong | {residual_counts['wrong']} |
| idle residual | {residual_counts['idle_residual']} |
| combined-state | {residual_counts['combined_state_residual']} |

Extra capabilities: {dict(residual_extra_cap)}
Missing capabilities: {dict(residual_missing_cap)}

SET_COMPLETE remains rejected. SET_REFINE remains unused.

Generic Gate v2 is the replacement for the HVAC+Notify special, not a new operator family. Idle Certification is a **candidate** operator: promote only if precision is high and broken exact is 0; otherwise keep UNKNOWN.

Need a **new** operator only if residual missing/wrong cannot be expressed as attribution+necessity. Current missing mass is still Gold-driven completion (unsafe without Y). Remaining extras after D+idle that are not leakage should stay ABSTAIN.

Recommendation: keep Generic Gate v2 as the parent-level necessity sidecar; do not add per-service rules; do not promote Idle Certification unless false-idle is acceptable for the paper's safety budget; do not add SET_COMPLETE.

Need-new-operator (heuristic missing>50 or idle_residual>200 or wrong>50): {need_new_op}.
"""
    write_md(DECISION_MD, decision)

    print(
        json.dumps(
            {
                "N": n_parent,
                "A": a["Parent_Set_Exact_Match"],
                "B": b["Parent_Set_Exact_Match"],
                "C": c["Parent_Set_Exact_Match"],
                "D": d["Parent_Set_Exact_Match"],
                "D_idle": di["Parent_Set_Exact_Match"],
                "gate_v2": dict(gate_d),
                "idle": dict(idle_status),
                "m2c": eval_obj["m2c"],
                "FR_D": d["False_Parent_Repair_count"],
                "wrote": [
                    str(ATTR_JSONL),
                    str(ATTR_MD),
                    str(OWN_JSONL),
                    str(COMP_JSONL),
                    str(GATE_MD),
                    str(EVAL_JSON),
                    str(IDLE_JSON),
                    str(RESID_JSON),
                    str(DECISION_MD),
                ],
            },
            indent=2,
        ),
        flush=True,
    )

if __name__ == "__main__":
    main()
