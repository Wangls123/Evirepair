from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ss_trhr_final import load_jsonl
from eval_strict_v2_set_remove import (
    ACTIVE_HVAC,
    MA,
    OUT,
    SS,
    TRACE,
    acc_bucket,
    add_bucket,
    apply_strict_v2,
    cap_of_svc,
    compact_action,
    finish_bucket,
    hvac_mode,
    ma_y_acts,
    map_repaired,
    set_metrics,
    set_remove_v1,
)
from eval_ma_m2c_necessity_gate import extra_pairs
from eval_ma_evidence_attribution import (
    apply_generic_gate,
    attribute_parent,
    compose_parent,
    ownership_for_component,
    residual_flags,
    toks_from_acts,
)
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.evidence import (
    device_already_satisfied,
    door_open,
    lux,
    motion_on,
    quiet_blocked,
    window_open,
)

DECOMP = OUT / "ma_owned_extra_decomposition.json"
DECOMP_MD = OUT / "ma_owned_extra_analysis.md"
SUFF_JSONL = OUT / "ma_evidence_sufficiency.jsonl"
VER_MD = OUT / "ma_behavioral_necessity_verifier_design.md"
EVAL_JSON = OUT / "ma_behavioral_necessity_evaluation.json"
CF_JSON = OUT / "ma_necessity_counterfactual_test.json"
IDLE_V2 = OUT / "ma_idle_certification_v2.json"
RESID = OUT / "ma_residual_after_sufficiency.json"
DECISION = OUT / "ma_sufficiency_next_step_decision.md"

AUX_CAPS = {"Log", "Helper", "Schedule", "GenericHA", "Vision"}
EVENT_CAPS = {"Notify", "Lighting", "ClimateControl"}

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0

def _norm(v: Any) -> str:
    return str(v or "").strip().lower()

def component_obs(comp: dict) -> tuple[dict, list]:
    rt = comp.get("runtime_state") or {}
    return dict(rt.get("observed") or {}), list(rt.get("entity_observations") or [])

def trigger_condition(comp: dict) -> dict[str, Any]:
    fb = comp.get("formal_b0_component") or {}
    tr = fb.get("execution_trace") or {}
    te = tr.get("trigger_evaluation") or {}
    ce = tr.get("condition_evaluation") or {}
    return {
        "trigger_matched": te.get("trigger_matched"),
        "conditions_passed": ce.get("conditions_passed"),
        "event_type": tr.get("event_type"),
    }

def owned_relevant(own: dict, cap: str) -> bool:
    return cap in set(own.get("relevant_caps") or []) or own.get("supporting_class") in {
        "OWNED_EVIDENCE",
        "SHARED_RELEVANT_EVIDENCE",
    }

def runtime_signals(comp: dict, action: dict, obs_override: dict | None = None) -> dict[str, Any]:
    obs, ents = component_obs(comp)
    if obs_override:
        obs = dict(obs)
        obs.update(obs_override)
    tc = trigger_condition(comp)
    door = door_open(obs)
    mot = motion_on(obs)
    lx = lux(obs)
    win = window_open(obs, ents)
    hvac = _norm(obs.get("hvac_mode"))
    light = _norm(obs.get("light_state"))
    satisfied = device_already_satisfied(action, obs, ents)
    quiet = quiet_blocked(obs, None)
    return {
        "window_open": win,
        "door_open": door,
        "motion_on": mot,
        "lux": lx,
        "hvac_mode": hvac or None,
        "light_state": light or None,
        "already_satisfied": satisfied,
        "quiet_blocked": quiet,
        "trigger_matched": tc["trigger_matched"],
        "conditions_passed": tc["conditions_passed"],
        "event_type": tc["event_type"],
    }

def event_occurred_for_cap(cap: str, sig: dict) -> bool | None:
    if cap == "Notify":
        if sig["door_open"] is True:
            return True
        if sig["door_open"] is False and not sig.get("event_type"):
            return False
        if sig["door_open"] is False:
            return False
        return None
    if cap == "Lighting":
        if sig["motion_on"] is True:
            return True
        if sig["motion_on"] is False:
            return False
        return None
    if cap == "ClimateControl":
        return True if sig["window_open"] else False if sig["window_open"] is False else None
    if cap == "Record":
        return None
    return None

def action_direction(action: dict) -> str:
    svc = str(action.get("service") or "")
    cap = cap_of_svc(svc)
    if cap == "ClimateControl":
        if svc == "climate.turn_off" or hvac_mode(action) == "off":
            return "deactivate"
        if hvac_mode(action) in ACTIVE_HVAC:
            return "activate"
        return "set"
    if svc.endswith(".turn_off"):
        return "deactivate"
    if svc.endswith(".turn_on"):
        return "activate"
    if svc.startswith("notify."):
        return "emit"
    return "other"

def contradictory(cap: str, action: dict, sig: dict, own: dict) -> bool:

    direction = action_direction(action)
    if cap == "ClimateControl":
        if direction == "activate" and sig["window_open"] is True:
            return True
        if direction == "deactivate" and sig["window_open"] is False and sig.get("hvac_mode") in {"off", "idle"}:
            return True
    if cap == "Lighting":
        if direction == "activate" and sig["motion_on"] is False and sig["lux"] is not None and sig["lux"] >= 80:
            return True
        if direction == "deactivate" and sig["motion_on"] is True and (sig["lux"] is None or sig["lux"] < 80):
            return True
    if cap == "Notify":
        if direction == "emit" and sig["door_open"] is False and own.get("supporting_class") == "OWNED_EVIDENCE":
            return True
    return False

def classify_e_labels(
    action: dict,
    own: dict,
    composition: dict,
    sig: dict,
    cap: str,
) -> list[str]:

    labels: list[str] = []
    intended_parent = set(composition.get("parent_intended_capabilities") or [])
    intended_here = {
        x["capability"]
        for x in composition.get("parent_intended_behaviors") or []
        if x.get("component_id") == own.get("component_id") and x.get("basis") != "behavior_target_unsupported"
    }
    has_owned = own.get("supporting_class") in {"OWNED_EVIDENCE", "SHARED_RELEVANT_EVIDENCE"}
    ev = event_occurred_for_cap(cap, sig)

    if has_owned and ev is False:
        labels.append("E1")
    if has_owned and sig["already_satisfied"]:
        labels.append("E2")
    if has_owned and (
        sig["quiet_blocked"]
        or sig["conditions_passed"] is False
        or (cap == "Lighting" and action_direction(action) == "activate" and sig["lux"] is not None and sig["lux"] >= 80)
    ):
        labels.append("E3")
    if has_owned and ev is not True and not sig["already_satisfied"] and cap in intended_here:
        labels.append("E4")
    if has_owned and ev is True and not sig["already_satisfied"] and cap not in intended_here and cap in EVENT_CAPS:
        labels.append("E4")
    if cap in AUX_CAPS or (cap == "Record" and "power" not in set(own.get("owned_families") or [])):
        if cap not in intended_parent:
            labels.append("E5")
    if own.get("no_action") or cap not in set(own.get("contract_capabilities") or []):
        labels.append("E6")
    if not labels:
        labels.append("E7")
    return sorted(set(labels))

def sufficiency_of(
    action: dict,
    own: dict,
    composition: dict,
    sig: dict,
    labels: list[str],
) -> dict[str, Any]:
    cap = cap_of_svc(str(action.get("service") or ""))
    intended_here = {
        x["capability"]
        for x in composition.get("parent_intended_behaviors") or []
        if x.get("component_id") == own.get("component_id") and x.get("basis") != "behavior_target_unsupported"
    }
    has_owned = own.get("supporting_class") in {"OWNED_EVIDENCE", "SHARED_RELEVANT_EVIDENCE"}
    ev = event_occurred_for_cap(cap, sig)
    if contradictory(cap, action, sig, own):
        return {"sufficiency": "CONTRADICTORY", "reason": "owned_or_runtime_state_conflicts_action_direction"}
    if (
        cap in intended_here
        and has_owned
        and ev is True
        and not sig["already_satisfied"]
        and not sig["quiet_blocked"]
        and sig["conditions_passed"] is not False
    ):
        return {"sufficiency": "SUFFICIENT_FOR_ACTION", "reason": "intended_owned_event_unmet_transition"}
    if has_owned and (
        ev is False
        or sig["already_satisfied"]
        or sig["quiet_blocked"]
        or sig["conditions_passed"] is False
        or "E4" in labels
    ):
        return {"sufficiency": "RELEVANT_BUT_INSUFFICIENT", "reason": "owned_relevant_but_not_necessary"}
    if has_owned:
        return {"sufficiency": "RELEVANT_BUT_INSUFFICIENT", "reason": "owned_without_proven_necessity"}
    return {"sufficiency": "UNKNOWN", "reason": "no_owned_relevant_or_mixed"}

def necessity_verifier(
    action: dict,
    siblings: list[dict],
    own: dict,
    composition: dict,
    sig: dict,
    labels: list[str],
    suff: dict,
) -> dict[str, Any]:

    svc = str(action.get("service") or "")
    cap = cap_of_svc(svc)
    intended_parent = set(composition.get("parent_intended_capabilities") or [])
    has_state_sib = any(
        cap_of_svc(str(s.get("service") or "")) in {"ClimateControl", "Lighting"} for s in siblings
    )
    bits = {
        "sufficiency": suff["sufficiency"],
        "labels": labels,
        "has_sibling": bool(siblings),
        "no_action": bool(own.get("no_action")),
        "in_parent_intended": cap in intended_parent,
    }

    if suff["sufficiency"] == "SUFFICIENT_FOR_ACTION":
        return {"decision": "NECESSARY", "reason": "SUFFICIENT_INTENDED_OWNED_EVENT", "confidence": 0.85, "bits": bits}

    if suff["sufficiency"] == "CONTRADICTORY":
        return {"decision": "UNNECESSARY", "reason": "CONTRADICTORY_EVIDENCE", "confidence": 0.8, "bits": bits}

    if "E5" in labels and cap not in intended_parent and (has_state_sib or bool(intended_parent - AUX_CAPS - {cap})):
        return {
            "decision": "UNNECESSARY",
            "reason": "AUX_EFFECT_NOT_IN_PARENT_INTENDED",
            "confidence": 0.7,
            "bits": bits,
        }

    if (
        "E6" in labels
        and own.get("no_action")
        and cap == "Notify"
        and event_occurred_for_cap("Notify", sig) is False
        and siblings
    ):
        return {
            "decision": "UNNECESSARY",
            "reason": "NO_ACTION_CONTRACT_AND_SECURITY_EVENT_ABSENT",
            "confidence": 0.75,
            "bits": bits,
        }

    return {"decision": "ABSTAIN", "reason": "INSUFFICIENT_TO_PROVE_UNNECESSARY", "confidence": 0.3, "bits": bits}

def apply_verifier(
    actions: list[dict],
    comps_by_id: dict[str, dict],
    owns_by_id: dict[str, dict],
    composition: dict,
    obs_overrides: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    trace: list[dict] = []
    for i, a in enumerate(actions):
        siblings = [x for j, x in enumerate(actions) if j != i]
        origin = str(a.get("component_origin") or "")
        own = owns_by_id.get(origin) or {"component_id": origin}
        comp = comps_by_id.get(origin) or {}
        ov = (obs_overrides or {}).get(origin)
        sig = runtime_signals(comp, a, ov)
        cap = cap_of_svc(str(a.get("service") or ""))
        labels = classify_e_labels(a, own, composition, sig, cap)
        suff = sufficiency_of(a, own, composition, sig, labels)
        v = necessity_verifier(a, siblings, own, composition, sig, labels, suff)
        row = {
            "service": a.get("service"),
            "component_origin": origin,
            "capability": cap,
            "e_labels": labels,
            **suff,
            **v,
        }
        trace.append(row)
        if v["decision"] != "UNNECESSARY":
            kept.append(a)
    return kept, trace

def idle_certify_v2(
    comps: list[dict],
    owns: list[dict],
    composition: dict,
    d_acts: list[dict],
    e_trace: list[dict],
) -> dict[str, Any]:
    by_id = {o["component_id"]: o for o in owns}
    decisions_by = {}
    for row in e_trace:
        decisions_by.setdefault(row["component_origin"], []).append(row["decision"])
    unresolved = []
    ok = []
    for c in comps:
        cid = str(c.get("component_id") or "")
        o = by_id.get(cid) or {}
        decs = decisions_by.get(cid) or []
        if decs:
            if any(d == "ABSTAIN" for d in decs) or any(d == "NECESSARY" for d in decs):
                unresolved.append(cid)
            elif all(d == "UNNECESSARY" for d in decs):
                ok.append(cid)
            else:
                unresolved.append(cid)
        else:
            if o.get("no_action"):
                ok.append(cid)
            else:
                unresolved.append(cid)
    if not unresolved and (ok or not comps):
        return {
            "status": "CERTIFIED_IDLE",
            "reason": "all_components_unnecessary_or_no_action_contract",
            "ok_components": ok,
            "unresolved_components": [],
        }
    return {
        "status": "UNKNOWN",
        "reason": "unresolved_abstain_or_necessary_or_active_contract",
        "ok_components": ok,
        "unresolved_components": unresolved,
    }

def strip_owned_atoms(atoms: list[dict]) -> list[dict]:
    return [a for a in atoms if not (a.get("component_relevance") and a.get("provenance") in {"direct", "shared"})]

def inject_irrelevant_atom(atoms: list[dict], cid: str, cap: str) -> list[dict]:
    out = list(atoms)
    if cap == "ClimateControl":
        fam, aid, val = "motion", "motion_state", "on"
    elif cap == "Lighting":
        fam, aid, val = "window", "window_state", "open"
    elif cap == "Notify":
        fam, aid, val = "power", "current_power_w", 12
    else:
        fam, aid, val = "window", "window_state", "open"
    out.append(
        {
            "component_id": cid,
            "atom_id": aid,
            "source": "observed",
            "entity": None,
            "bound_entities": [],
            "family": fam,
            "value": val,
            "capability_relevance": [],
            "component_relevance": [],
            "provenance": "copied",
            "copied_from": [],
            "confidence": 0.2,
        }
    )
    return out

def contradictory_override(cap: str, action: dict) -> dict:
    direction = action_direction(action)
    if cap == "ClimateControl":
        if direction == "activate":
            return {"window_state": "open"}
        return {"window_state": "closed", "hvac_mode": "off"}
    if cap == "Lighting":
        if direction == "activate":
            return {"motion_state": "off", "illuminance_lux": 400}
        return {"motion_state": "on", "illuminance_lux": 10}
    if cap == "Notify":
        return {"door_state": "closed"}
    return {"quiet_hours": True}

def rebuild_owns(comps: list[dict], per_comp_atoms: dict[str, list], contracts: dict) -> list[dict]:
    owns = []
    for c in comps:
        cid = str(c.get("component_id") or "")
        owns.append(ownership_for_component(c, per_comp_atoms.get(cid) or [], contracts.get(cid) or []))
    return owns

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

    buckets = {k: acc_bucket() for k in ("D", "E", "E_idle")}
    e_dec = Counter()
    e_reason = Counter()
    e_suff = Counter()
    extra_label = Counter()
    extra_suff = Counter()
    extra_own = Counter()
    extra_cap = Counter()
    extra_multi = Counter()
    extra_n_d = extra_n_idle = 0
    extra_owned_d = extra_e4_d = 0
    extra_owned_idle = extra_e4_idle = 0
    extra_actions_idle = 0
    extra_parents_idle = 0
    examples: dict[str, list] = {k: [] for k in ("E1", "E2", "E3", "E4", "E5", "E6", "E7")}

    e_broke_d = e_fixed_d = e_new_fr = 0
    ei_broke_e = ei_fixed_e = 0
    certified = certified_gold_idle = certified_gold_active = 0
    gold_idle_n = 0
    idle_broke_e = 0
    false_idle_ex: list[dict] = []

    residual_counts = Counter()
    residual_extra_cap = Counter()
    residual_extra_svc = Counter()
    residual_missing_cap = Counter()
    residual_why = Counter()

    cf_n = 0
    cf = {
        "remove_owned": Counter(),
        "swap_sibling": Counter(),
        "inject_irrelevant": Counter(),
        "contradictory": Counter(),
    }
    cf_expected = Counter()
    cf_stable = Counter()
    cf_unsafe = Counter()

    suff_lines: list[str] = []
    extra_rows: list[dict] = []

    print("walk MA...", flush=True)
    n_parent = 0
    with MA.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_parent += 1
            mid = str(rec.get("multi_action_id") or "")
            comps = list(rec.get("components") or [])
            bind_all = {str(c.get("component_id") or ""): c.get("entity_binding") or {} for c in comps}
            comps_by_id = {str(c.get("component_id") or ""): c for c in comps}
            gold = ma_y_acts(rec)
            gold_toks = toks_from_acts(gold, bind_all)
            gold_idle = not gold_toks
            if gold_idle:
                gold_idle_n += 1
            parent_acts = list(((rec.get("formal_b0") or {}).get("parent") or {}).get("semantic_actions") or [])
            b0_toks = toks_from_acts(parent_acts, bind_all)
            b0_ok = bool(set_metrics(b0_toks, gold_toks)["set_exact"])

            ctxs = {}
            for c in comps:
                ctxs[str(c.get("component_id") or "")] = ctx_of(str(c.get("single_scene_sample_id") or ""))
            atoms, meta = attribute_parent(comps, ctxs)
            owns = []
            for c in comps:
                cid = str(c.get("component_id") or "")
                owns.append(ownership_for_component(c, meta["per_comp_atoms"].get(cid) or [], meta["contracts"].get(cid) or []))
            owns_by_id = {o["component_id"]: o for o in owns}
            composition = compose_parent(comps, owns)

            merged = independent_merge(comps, bind_all)
            b_acts, _ = set_remove_v1(merged, bind_all)
            d_acts, _dtr = apply_generic_gate(b_acts, owns_by_id, composition)
            e_acts, e_trace = apply_verifier(d_acts, comps_by_id, owns_by_id, composition)
            for row in e_trace:
                e_dec[row["decision"]] += 1
                e_reason[row.get("reason") or ""] += 1
                e_suff[row.get("sufficiency") or ""] += 1

            idle = idle_certify_v2(comps, owns, composition, d_acts, e_trace)
            ei_acts = [] if idle["status"] == "CERTIFIED_IDLE" else list(e_acts)

            d_toks = toks_from_acts(d_acts, bind_all)
            e_toks = toks_from_acts(e_acts, bind_all)
            ei_toks = toks_from_acts(ei_acts, bind_all)
            mD = set_metrics(d_toks, gold_toks)
            mE = set_metrics(e_toks, gold_toks)
            mEi = set_metrics(ei_toks, gold_toks)
            add_bucket(buckets["D"], mD, b0_ok, bool(mD["set_exact"]), mD["n_pred"])
            add_bucket(buckets["E"], mE, b0_ok, bool(mE["set_exact"]), mE["n_pred"], fired=any(r["decision"] == "UNNECESSARY" for r in e_trace))
            add_bucket(buckets["E_idle"], mEi, b0_ok, bool(mEi["set_exact"]), mEi["n_pred"], fired=idle["status"] == "CERTIFIED_IDLE")

            if mD["set_exact"] and not mE["set_exact"]:
                e_broke_d += 1
                if b0_ok:
                    e_new_fr += 1
            if (not mD["set_exact"]) and mE["set_exact"]:
                e_fixed_d += 1
            if mE["set_exact"] and not mEi["set_exact"]:
                ei_broke_e += 1
                idle_broke_e += 1
            if (not mE["set_exact"]) and mEi["set_exact"]:
                ei_fixed_e += 1

            extra_d, extra_acts_d = extra_pairs(d_acts, gold_toks, bind_all)
            extra_n_d += len(extra_d)
            extra_after_e, extra_acts_e = extra_pairs(e_acts, gold_toks, bind_all)

            trace_by = {(r["component_origin"], r.get("service")): r for r in e_trace}
            for a in extra_acts_d:
                origin = str(a.get("component_origin") or "")
                own = owns_by_id.get(origin) or {}
                row = trace_by.get((origin, a.get("service")))
                if row is None:
                    for r in e_trace:
                        if r["component_origin"] == origin and r.get("service") == a.get("service"):
                            row = r
                            break
                labels = list((row or {}).get("e_labels") or [])
                suff = (row or {}).get("sufficiency") or "UNKNOWN"
                extra_label.update(labels)
                extra_suff[suff] += 1
                extra_own[own.get("supporting_class") or "NONE"] += 1
                extra_cap[cap_of_svc(str(a.get("service") or ""))] += 1
                extra_multi["+".join(labels)] += 1
                if own.get("supporting_class") in {"OWNED_EVIDENCE", "SHARED_RELEVANT_EVIDENCE"}:
                    extra_owned_d += 1
                    if "E4" in labels or suff == "RELEVANT_BUT_INSUFFICIENT":
                        extra_e4_d += 1
                for lab in labels:
                    if len(examples[lab]) < 4:
                        examples[lab].append(
                            {
                                "multi_action_id": mid,
                                "service": a.get("service"),
                                "origin": origin,
                                "labels": labels,
                                "sufficiency": suff,
                                "ownership": own.get("supporting_class"),
                            }
                        )
                extra_rows.append(
                    {
                        "multi_action_id": mid,
                        "service": a.get("service"),
                        "capability": cap_of_svc(str(a.get("service") or "")),
                        "ownership": own.get("supporting_class"),
                        "labels": labels,
                        "sufficiency": suff,
                        "gold_idle": gold_idle,
                    }
                )

            flags = residual_flags(e_toks, gold_toks)
            if flags["extra"]:
                residual_counts["extra"] += 1
                residual_extra_cap.update(flags["extra_caps"])
                residual_extra_svc.update(flags["extra_svc"])
                extra_parents_idle += 0
            if flags["missing"]:
                residual_counts["missing"] += 1
                residual_missing_cap.update(flags["missing_caps"])
            if flags["wrong"]:
                residual_counts["wrong"] += 1
            if flags["idle_residual"]:
                residual_counts["idle_residual"] += 1
            if flags["combined_state"]:
                residual_counts["combined_state"] += 1
            if flags["extra"]:

                if extra_acts_e:
                    origin = str(extra_acts_e[0].get("component_origin") or "")
                    own = owns_by_id.get(origin) or {}
                    row = None
                    for r in e_trace:
                        if r["component_origin"] == origin and r.get("service") == extra_acts_e[0].get("service"):
                            row = r
                            break
                    suff = (row or {}).get("sufficiency")
                    if suff == "UNKNOWN":
                        residual_why["evidence_ambiguity"] += 1
                    elif suff == "RELEVANT_BUT_INSUFFICIENT":
                        residual_why["unsupported_behavior"] += 1
                    elif own.get("supporting_class") in {"OWNED_EVIDENCE", "SHARED_RELEVANT_EVIDENCE"}:
                        residual_why["unsupported_behavior"] += 1
                    else:
                        residual_why["representation_mismatch"] += 1
                else:
                    residual_why["representation_mismatch"] += 1

            if idle["status"] == "CERTIFIED_IDLE":
                certified += 1
                if gold_idle:
                    certified_gold_idle += 1
                else:
                    certified_gold_active += 1
                    if len(false_idle_ex) < 8:
                        false_idle_ex.append({"multi_action_id": mid, "gold_services": [t[0] for t in gold_toks]})

            comp_rows = []
            for o in owns:
                cid = o["component_id"]
                acts = [r for r in e_trace if r["component_origin"] == cid]

                if not acts:
                    comp_rows.append(
                        {
                            "component_id": cid,
                            "ownership": o.get("supporting_class"),
                            "sufficiency": "UNKNOWN",
                            "actions": [],
                        }
                    )
                else:

                    sc = Counter(r["sufficiency"] for r in acts)
                    top = sc.most_common(1)[0][0]
                    comp_rows.append(
                        {
                            "component_id": cid,
                            "ownership": o.get("supporting_class"),
                            "sufficiency": top,
                            "actions": [
                                {
                                    "service": r.get("service"),
                                    "sufficiency": r["sufficiency"],
                                    "e_labels": r["e_labels"],
                                    "verifier": r["decision"],
                                }
                                for r in acts
                            ],
                        }
                    )
            suff_lines.append(
                json.dumps(
                    {
                        "multi_action_id": mid,
                        "components": comp_rows,
                        "idle_v2": idle["status"],
                    },
                    ensure_ascii=False,
                )
            )

            if d_acts and len(comps) >= 1:
                cf_n += 1
                origin0 = str(d_acts[0].get("component_origin") or "")
                cap0 = cap_of_svc(str(d_acts[0].get("service") or ""))
                base_dec = None
                for r in e_trace:
                    if r["component_origin"] == origin0 and r.get("service") == d_acts[0].get("service"):
                        base_dec = r["decision"]
                        break

                atoms1 = dict(meta["per_comp_atoms"])
                atoms1[origin0] = strip_owned_atoms(atoms1.get(origin0) or [])
                owns1 = rebuild_owns(comps, atoms1, meta["contracts"])
                owns1_by = {o["component_id"]: o for o in owns1}
                _k1, tr1 = apply_verifier(d_acts, comps_by_id, owns1_by, composition)
                d1 = next((r["decision"] for r in tr1 if r["component_origin"] == origin0 and r.get("service") == d_acts[0].get("service")), None)
                if d1 == base_dec:
                    cf_stable["remove_owned"] += 1
                    cf["remove_owned"]["stable"] += 1
                else:
                    cf["remove_owned"]["changed"] += 1
                    if base_dec == "NECESSARY" and d1 in {"ABSTAIN", "UNNECESSARY"}:
                        cf_expected["remove_owned"] += 1
                    if base_dec != "UNNECESSARY" and d1 == "UNNECESSARY":
                        cf_unsafe["remove_owned"] += 1
                    if base_dec == "NECESSARY" and d1 == "ABSTAIN":
                        cf_expected["remove_owned_necessary_to_abstain"] += 1

                cids = [str(c.get("component_id") or "") for c in comps]
                if len(cids) >= 2 and origin0 in cids:
                    other = next(x for x in cids if x != origin0)
                    atoms2 = dict(meta["per_comp_atoms"])
                    atoms2[origin0], atoms2[other] = list(atoms2.get(other) or []), list(atoms2.get(origin0) or [])
                    owns2 = rebuild_owns(comps, atoms2, meta["contracts"])
                    owns2_by = {o["component_id"]: o for o in owns2}
                    _k2, tr2 = apply_verifier(d_acts, comps_by_id, owns2_by, composition)
                    d2 = next((r["decision"] for r in tr2 if r["component_origin"] == origin0 and r.get("service") == d_acts[0].get("service")), None)
                    if d2 == base_dec:
                        cf_stable["swap_sibling"] += 1
                        cf["swap_sibling"]["stable"] += 1
                    else:
                        cf["swap_sibling"]["changed"] += 1
                        cf_expected["swap_sibling"] += 1
                    if base_dec != "UNNECESSARY" and d2 == "UNNECESSARY":
                        cf_unsafe["swap_sibling"] += 1
                else:
                    cf["swap_sibling"]["skipped"] += 1

                atoms3 = dict(meta["per_comp_atoms"])
                atoms3[origin0] = inject_irrelevant_atom(list(atoms3.get(origin0) or []), origin0, cap0)
                owns3 = rebuild_owns(comps, atoms3, meta["contracts"])
                owns3_by = {o["component_id"]: o for o in owns3}
                _k3, tr3 = apply_verifier(d_acts, comps_by_id, owns3_by, composition)
                d3 = next((r["decision"] for r in tr3 if r["component_origin"] == origin0 and r.get("service") == d_acts[0].get("service")), None)
                if d3 == base_dec:
                    cf_stable["inject_irrelevant"] += 1
                    cf["inject_irrelevant"]["stable"] += 1
                    cf_expected["inject_irrelevant_stable"] += 1
                else:
                    cf["inject_irrelevant"]["changed"] += 1
                    if d3 == "UNNECESSARY" and base_dec != "UNNECESSARY":
                        cf_unsafe["inject_irrelevant"] += 1

                ov = {origin0: contradictory_override(cap0, d_acts[0])}
                _k4, tr4 = apply_verifier(d_acts, comps_by_id, owns_by_id, composition, obs_overrides=ov)
                d4 = next((r["decision"] for r in tr4 if r["component_origin"] == origin0 and r.get("service") == d_acts[0].get("service")), None)
                if d4 == base_dec:
                    cf_stable["contradictory"] += 1
                    cf["contradictory"]["stable"] += 1
                else:
                    cf["contradictory"]["changed"] += 1
                    if base_dec == "NECESSARY" and d4 in {"ABSTAIN", "UNNECESSARY"}:
                        cf_expected["contradictory"] += 1
                if base_dec == "NECESSARY" and d4 == "NECESSARY":
                    cf["contradictory"]["necessary_stuck"] += 1

    n_e = sum(e_dec.values()) or 1
    d = finish_bucket(buckets["D"])
    e = finish_bucket(buckets["E"])
    ei = finish_bucket(buckets["E_idle"])
    e["ABSTAIN_count"] = e_dec["ABSTAIN"]
    e["NECESSARY_count"] = e_dec["NECESSARY"]
    e["UNNECESSARY_count"] = e_dec["UNNECESSARY"]
    e["ABSTAIN_rate"] = round(e_dec["ABSTAIN"] / n_e, 4)
    e["broke_D_exact"] = e_broke_d
    e["fixed_from_D"] = e_fixed_d
    e["new_false_repair_vs_D"] = e_new_fr
    e["reasons"] = dict(e_reason)
    e["sufficiency_on_D_actions"] = dict(e_suff)

    decomp = {
        "note": "E1–E7 assigned Y-blind on remaining Generic Gate v2 actions; Gold used only to mark extras for stats. Ownership layer unchanged.",
        "N": n_parent,
        "extra_actions_after_D": extra_n_d,
        "extra_parents_after_E": residual_counts["extra"],
        "ownership_of_D_extras": dict(extra_own),
        "sufficiency_of_D_extras": dict(extra_suff),
        "labels_of_D_extras": dict(extra_label),
        "label_combinations": dict(extra_multi.most_common(25)),
        "capability_of_D_extras": dict(extra_cap),
        "owned_extras_after_D": extra_owned_d,
        "owned_or_e4_relevant_insufficient": extra_e4_d,
        "relevant_but_insufficient_extras": extra_suff.get("RELEVANT_BUT_INSUFFICIENT", 0),
        "examples": {k: v for k, v in examples.items() if v},
    }
    dump(DECOMP, decomp)

    eval_obj = {
        "note": "D is frozen Generic Necessity Gate v2. E = D + Behavioral Necessity Verifier. Gate v2 not modified.",
        "N": n_parent,
        "D_generic_gate_v2": d,
        "E_plus_necessity_verifier": e,
        "E_plus_idle_v2": {
            **ei,
            "broke_E_exact": ei_broke_e,
            "fixed_from_E": ei_fixed_e,
        },
        "priority_checks": {
            "broken_D_exact": e_broke_d,
            "false_parent_repair_D": d["False_Parent_Repair_count"],
            "false_parent_repair_E": e["False_Parent_Repair_count"],
            "fr_did_not_increase": e["False_Parent_Repair_count"] <= d["False_Parent_Repair_count"],
        },
        "verifier_decisions": dict(e_dec),
        "verifier_reasons": dict(e_reason),
    }
    dump(EVAL_JSON, eval_obj)

    cf_obj = {
        "note": "Perturbations do not change Gold. Verifier re-run on D remaining actions.",
        "parents_with_D_actions": cf_n,
        "perturbations": {k: dict(v) for k, v in cf.items()},
        "expected_decision_change": dict(cf_expected),
        "decision_stability": {
            k: pct(cf_stable[k], cf_n if k != "swap_sibling" else max(cf_n - cf["swap_sibling"]["skipped"], 1))
            for k in ("remove_owned", "swap_sibling", "inject_irrelevant", "contradictory")
        },
        "unsafe_remove_rate": {
            k: pct(cf_unsafe[k], cf_n)
            for k in ("remove_owned", "swap_sibling", "inject_irrelevant")
        },
        "interpretation": {
            "remove_owned_should": "NECESSARY → ABSTAIN (absence is not proof of unnecessity)",
            "inject_irrelevant_should": "stable",
            "contradictory_should": "NECESSARY → ABSTAIN or UNNECESSARY",
            "unsafe_remove": "became UNNECESSARY after removing owned evidence or injecting irrelevant atoms",
        },
    }
    dump(CF_JSON, cf_obj)

    idle_obj = {
        "note": "Proof-style Idle v2. No evidence ≠ idle. Gold eval only.",
        "N": n_parent,
        "CERTIFIED_IDLE": certified,
        "UNKNOWN": n_parent - certified,
        "gold_idle_parents": gold_idle_n,
        "precision": round(certified_gold_idle / certified, 4) if certified else None,
        "precision_pct": pct(certified_gold_idle, certified),
        "coverage": round(certified_gold_idle / gold_idle_n, 4) if gold_idle_n else None,
        "coverage_pct": pct(certified_gold_idle, gold_idle_n),
        "false_idle_certification": certified_gold_active,
        "broken_exact_from_E": idle_broke_e,
        "gold_idle_made_exact_vs_E": ei_fixed_e,
        "false_idle_examples": false_idle_ex,
    }
    dump(IDLE_V2, idle_obj)

    resid_obj = {
        "note": "Residual after Strict-v2 + SET_REMOVE v1 + Generic Gate v2 + Behavioral Necessity Verifier. Gold eval only.",
        "N": n_parent,
        "inexact": n_parent - buckets["E"]["set_exact"],
        "parent_counts": dict(residual_counts),
        "extra_capability": dict(residual_extra_cap),
        "extra_service": dict(residual_extra_svc.most_common(20)),
        "missing_capability": dict(residual_missing_cap),
        "residual_cause": dict(residual_why),
        "pipeline": {
            "D_generic_gate_exact": buckets["D"]["set_exact"],
            "E_verifier_exact": buckets["E"]["set_exact"],
            "E_idle_v2_exact": buckets["E_idle"]["set_exact"],
        },
    }
    dump(RESID, resid_obj)

    SUFF_JSONL.write_text("\n".join(suff_lines) + "\n", encoding="utf-8")

    decomp_md = f"""# Owned extra decomposition

Y-blind E1–E7 on actions that remain after frozen Generic Necessity Gate v2. Gold is used only to mark which remaining actions are extras. Ownership classes are not modified.

## Corpus

Remaining extra **actions** after D: {extra_n_d} (parent extras after E: {residual_counts['extra']}).

Previous residual extra **parents** after D+Idle v1 was 952; this round decomposes extras the verifier actually sees (after D).

## Ownership of D extras

| Class | n |
| --- | ---: |
"""
    for k, v in extra_own.most_common():
        decomp_md += f"| {k} | {v} |\n"
    decomp_md += f"""
Owned or shared extras: {extra_owned_d}. Of those, E4 or RELEVANT_BUT_INSUFFICIENT: {extra_e4_d}.

## Sufficiency of D extras

| Sufficiency | n |
| --- | ---: |
"""
    for k, v in extra_suff.most_common():
        decomp_md += f"| {k} | {v} |\n"
    decomp_md += """
## E-labels (multi-label)

| Label | meaning | n |
| --- | --- | ---: |
"""
    meanings = {
        "E1": "owned evidence, event did not occur",
        "E2": "owned evidence, state already satisfied",
        "E3": "owned evidence, execution condition fails",
        "E4": "owned evidence proves relevance, not necessity",
        "E5": "auxiliary / event effect not in parent intended",
        "E6": "contract / behavior_target inconsistent with action",
        "E7": "other",
    }
    for k in ("E1", "E2", "E3", "E4", "E5", "E6", "E7"):
        decomp_md += f"| {k} | {meanings[k]} | {extra_label.get(k, 0)} |\n"
    decomp_md += "\nCapability of D extras: " + json.dumps(dict(extra_cap)) + "\n"
    write_md(DECOMP_MD, decomp_md)

    ver_md = f"""# Behavioral Necessity Verifier

Appended **after** frozen Generic Necessity Gate v2. Gate v2 is not modified. The verifier never generates actions.

## Inputs

- component intended behavior (parent composition)
- attributed evidence / ownership (unchanged layer)
- evidence sufficiency (new layer)
- sibling behavior
- candidate action
- trigger / condition / entity state

Not used: Gold action, Gold decision, service blacklist, scene_type alone.

## Sufficiency layer

| Label | Meaning |
| --- | --- |
| SUFFICIENT_FOR_ACTION | Intended + owned relevant + event occurred + transition unmet + condition not blocking |
| RELEVANT_BUT_INSUFFICIENT | Owned/relevant but event absent, already satisfied, blocked, or only correlative |
| CONTRADICTORY | Runtime/owned state conflicts with action direction |
| UNKNOWN | No owned relevant evidence, or mixed |

Principle: **relevance ≠ execution necessity**.

## Decisions

| Decision | When |
| --- | --- |
| NECESSARY | Sufficiency is SUFFICIENT_FOR_ACTION |
| UNNECESSARY | CONTRADICTORY; or aux effect not in parent intended with a sibling/parent STATE goal; or `valid.no_action` Notify with security event absent |
| ABSTAIN | Default. Climate/Lighting Gold-idle leftovers stay here unless contradictory |

Only UNNECESSARY may SET_REMOVE.

## Offline (MA N={n_parent})

D set-exact {d['Parent_Set_Exact_count']} ({d['Parent_Set_Exact_Match']}). E set-exact {e['Parent_Set_Exact_count']} ({e['Parent_Set_Exact_Match']}). Broken D-exact {e_broke_d}. FR D {d['False_Parent_Repair_count']} → E {e['False_Parent_Repair_count']}.

Verifier: {dict(e_dec)}
"""
    write_md(VER_MD, ver_md)

    decision = f"""# Sufficiency next-step decision

Frozen operators unchanged, including Generic Necessity Gate v2 (44.29% / 1063). This round only adds a sufficiency layer and a conservative verifier.

## Metrics

| | Set-exact | P | R | F1 | FR | Broken D-exact | ABSTAIN | UNNECESSARY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D Generic Gate v2 | {pct(d['Parent_Set_Exact_count'], n_parent)}% ({d['Parent_Set_Exact_count']}) | {d['Action_Precision']:.4f} | {d['Action_Recall']:.4f} | {d['Action_F1']:.4f} | {d['False_Parent_Repair_count']} | — | — | — |
| E + Necessity Verifier | {pct(e['Parent_Set_Exact_count'], n_parent)}% ({e['Parent_Set_Exact_count']}) | {e['Action_Precision']:.4f} | {e['Action_Recall']:.4f} | {e['Action_F1']:.4f} | {e['False_Parent_Repair_count']} | {e_broke_d} | {e.get('ABSTAIN_rate')} | {e_dec['UNNECESSARY']} |
| E + Idle v2 | {pct(ei['Parent_Set_Exact_count'], n_parent)}% ({ei['Parent_Set_Exact_count']}) | {ei['Action_Precision']:.4f} | {ei['Action_Recall']:.4f} | {ei['Action_F1']:.4f} | {ei['False_Parent_Repair_count']} | idle broke E {idle_broke_e} | — | — |

## Answers

### 1. Of the remaining extras, how many are relevant but insufficient?

D extras (actions the verifier sees): {extra_n_d}. Ownership: {dict(extra_own)}.

Owned/shared extras: {extra_owned_d}. RELEVANT_BUT_INSUFFICIENT extras: {extra_suff.get('RELEVANT_BUT_INSUFFICIENT', 0)}. Owned extras that also carry E4 or that sufficiency label: {extra_e4_d}.

E-labels: {dict(extra_label)}

The previous 952 figure is **parents** with extra after D+Idle v1, not owned-action count. Action-level extras after D are the correct denominator for this question.

### 2. Can Evidence Sufficiency keep lifting parent set-exact?

E vs D: {e['Parent_Set_Exact_count'] - d['Parent_Set_Exact_count']:+d} exact ({pct(e['Parent_Set_Exact_count'] - d['Parent_Set_Exact_count'], n_parent):+.2f} pp).

Sufficiency explains many owned extras as RELEVANT_BUT_INSUFFICIENT, but proving UNNECESSARY without breaking D-exact is rare. Set-exact is not the objective. The layer is diagnostic first.

### 3. Zero or near-zero new False Repair?

Broken D-exact = {e_broke_d}. FR D {d['False_Parent_Repair_count']} → E {e['False_Parent_Repair_count']} (did not increase: {e['False_Parent_Repair_count'] <= d['False_Parent_Repair_count']}). New FR vs D among correct-B0: {e_new_fr}.

### 4. Do counterfactuals show semantic evidence rather than pattern memory?

Parents with D actions: {cf_n}.

| Perturbation | stable % | expected change | unsafe UNNECESSARY % |
| --- | ---: | ---: | ---: |
| remove-owned-evidence | {pct(cf_stable['remove_owned'], cf_n)} | NECESSARY→ABSTAIN {cf_expected.get('remove_owned_necessary_to_abstain', 0)} | {pct(cf_unsafe['remove_owned'], cf_n)} |
| swap-sibling-evidence | {pct(cf_stable['swap_sibling'], max(cf_n - cf['swap_sibling']['skipped'], 1))} | changed {cf['swap_sibling']['changed']} | {pct(cf_unsafe['swap_sibling'], cf_n)} |
| irrelevant injection | {pct(cf_stable['inject_irrelevant'], cf_n)} | should stay | {pct(cf_unsafe['inject_irrelevant'], cf_n)} |
| contradictory evidence | {pct(cf_stable['contradictory'], cf_n)} | NECESSARY drops {cf_expected.get('contradictory', 0)} | — |

Unsafe remove = became UNNECESSARY after deleting owned evidence or injecting irrelevant atoms (absence treated as proof). Low unsafe + high inject stability supports semantic use. High remove→UNNECESSARY would mean the verifier treats missing evidence as idle/delete.

### 5. Is Idle Certification v2 precise enough?

CERTIFIED_IDLE {certified}. Precision {pct(certified_gold_idle, certified)}% ({certified_gold_idle}/{certified or 1}). Coverage {pct(certified_gold_idle, gold_idle_n)}% of {gold_idle_n} Gold-idle. False idle {certified_gold_active}. Broke E-exact {idle_broke_e}.

Precision is the objective. Idle v2 is not frozen unless precision is high and broken exact is 0.

### 6. Does MA still need a new operator?

Residual after D+verifier: extra {residual_counts['extra']}, missing {residual_counts['missing']}, wrong {residual_counts['wrong']}, idle {residual_counts['idle_residual']}, combined-state {residual_counts['combined_state']}.

Cause tags: {dict(residual_why)}

Do **not** add SET_COMPLETE. Do **not** modify Generic Gate v2. Promote the verifier only if broken D-exact is 0 and FR does not rise **and** UNNECESSARY is non-trivial. Otherwise keep it as an analysis sidecar.

### 7. Are remaining errors mostly evidence insufficiency rather than repair-capacity gaps?

Extra capabilities after E: {dict(residual_extra_cap)}
Missing capabilities: {dict(residual_missing_cap)}

If extras are dominated by RELEVANT_BUT_INSUFFICIENT / UNKNOWN, the bottleneck is proving necessity, not missing a new rewrite operator. Missing Notify remains Gold-driven completion and must stay ABSTAIN.
"""
    write_md(DECISION, decision)

    print(
        json.dumps(
            {
                "N": n_parent,
                "D": d["Parent_Set_Exact_Match"],
                "E": e["Parent_Set_Exact_Match"],
                "broke_D": e_broke_d,
                "FR_D": d["False_Parent_Repair_count"],
                "FR_E": e["False_Parent_Repair_count"],
                "verifier": dict(e_dec),
                "extra_after_D": extra_n_d,
                "owned_extras": extra_owned_d,
                "e4_or_rbi": extra_e4_d,
                "idle_v2": certified,
                "idle_prec": pct(certified_gold_idle, certified),
                "cf_unsafe_remove": pct(cf_unsafe["remove_owned"], cf_n),
                "cf_inject_stable": pct(cf_stable["inject_irrelevant"], cf_n),
            },
            indent=2,
        ),
        flush=True,
    )

if __name__ == "__main__":
    main()
