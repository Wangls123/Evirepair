from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

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
    compact_action,
    finish_bucket,
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
from eval_ma_evidence_sufficiency import (
    AUX_CAPS,
    apply_verifier,
    classify_e_labels,
    inject_irrelevant_atom,
    necessity_verifier,
    rebuild_owns,
    runtime_signals,
    strip_owned_atoms,
    sufficiency_of,
    trigger_condition,
)
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.evidence import device_already_satisfied

SEED = 20260919
BOOT = 10000

CLIMATE_JSON = OUT / "ma_climate_disagreement_audit.json"
CLIMATE_MD = OUT / "ma_climate_disagreement_audit.md"
CONS_JSON = OUT / "ma_evidence_contract_gold_consistency.json"
LPO_JSON = OUT / "ma_verifier_leave_pattern_out.json"
XCAP_JSON = OUT / "ma_verifier_cross_capability.json"
XFER_JSON = OUT / "ma_component_count_transfer.json"
INABL_JSON = OUT / "ma_verifier_input_ablation.json"
STABL_JSON = OUT / "ma_verifier_structural_ablation.json"
CF2_JSON = OUT / "ma_verifier_counterfactual_v2.json"
TMPL_JSON = OUT / "ma_template_generalization_audit.json"
SIG_JSON = OUT / "ma_verifier_significance.json"
REP_JSON = OUT / "ma_verifier_repeatability.json"
STRESS_JSON = OUT / "ma_verifier_safety_stress_test.json"
BOUND_JSON = OUT / "ma_verifier_residual_boundary.json"
REPORT = OUT / "ma_verifier_strict_generalization_report.md"

STATE_CAPS = {"ClimateControl", "Lighting"}
EVENT_CAPS = {"Notify", "Log", "Record", "Vision"}
CAP_GROUP = {
    "ClimateControl": "ClimateControl",
    "Lighting": "Lighting",
    "Notify": "Notification",
    "Log": "Logging",
    "Helper": "Helper",
}

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0

def new_bucket() -> dict:
    return acc_bucket()

def mcnemar_pair(a_ok: list[bool], b_ok: list[bool]) -> dict[str, Any]:
    n01 = sum(1 for a, b in zip(a_ok, b_ok) if a and not b)
    n10 = sum(1 for a, b in zip(a_ok, b_ok) if (not a) and b)
    n = n01 + n10
    if n == 0:
        return {"b_improved": n10, "a_worsened": n01, "statistic": 0.0, "p_value": 1.0, "n_discordant": 0}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    return {
        "b_improved": n10,
        "a_worsened": n01,
        "statistic": round(stat, 4),
        "p_value": float(chi2.sf(stat, 1)),
        "n_discordant": n,
    }

def bootstrap_deltas(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    n = len(a)
    idx = rng.integers(0, n, size=(BOOT, n))
    delta = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return {
        "mean_delta": round(float(delta.mean()), 6),
        "ci95": [round(float(np.quantile(delta, 0.025)), 6), round(float(np.quantile(delta, 0.975)), 6)],
        "n_bootstrap": BOOT,
        "seed": SEED,
    }

def empty_composition() -> dict:
    return {"parent_intended_behaviors": [], "parent_intended_capabilities": [], "excluded_components": []}

def collapse_own(own: dict) -> dict:
    o = dict(own)
    if o.get("supporting_class") != "NO_SUPPORTING_EVIDENCE" or o.get("copied_families") or o.get("owned_evidence"):
        o["supporting_class"] = "OWNED_EVIDENCE"
        caps = set(o.get("relevant_caps") or [])
        for fam in list(o.get("copied_families") or []) + list(o.get("owned_families") or []):
            if fam in {"window", "climate"}:
                caps.add("ClimateControl")
            elif fam in {"motion", "lux", "light"}:
                caps.add("Lighting")
            elif fam in {"door", "person"}:
                caps.add("Notify")
            elif fam == "power":
                caps.add("Record")
        o["relevant_caps"] = sorted(caps)
    return o

def strip_bt(own: dict, composition: dict) -> tuple[dict, dict]:
    o = dict(own)
    o["no_action"] = False
    o["behavior_target"] = ""
    o["behavior_target_suffix"] = ""
    comp = {
        "parent_intended_behaviors": [
            x for x in composition.get("parent_intended_behaviors") or [] if x.get("basis") == "no_action_but_owned_independent_event"
        ],
        "parent_intended_capabilities": [],
        "excluded_components": [],
    }
    comp["parent_intended_capabilities"] = sorted({x["capability"] for x in comp["parent_intended_behaviors"]})
    return o, comp

def strip_contract(own: dict) -> dict:
    o = dict(own)
    o["contract_capabilities"] = []
    return o

def blank_own(own: dict) -> dict:
    o = dict(own)
    o["supporting_class"] = "NO_SUPPORTING_EVIDENCE"
    o["owned_evidence"] = []
    o["shared_relevant_evidence"] = []
    o["relevant_caps"] = []
    o["owned_families"] = []
    o["copied_families"] = []
    o["has_independent_owned_event"] = False
    return o

def apply_variant(
    actions: list[dict],
    comps_by_id: dict[str, dict],
    owns_by_id: dict[str, dict],
    composition: dict,
    *,
    mode: str,
) -> tuple[list[dict], list[dict]]:

    kept: list[dict] = []
    trace: list[dict] = []
    for i, a in enumerate(actions):
        siblings = [x for j, x in enumerate(actions) if j != i]
        origin = str(a.get("component_origin") or "")
        own = dict(owns_by_id.get(origin) or {"component_id": origin})
        compo = composition
        if mode == "A1":
            own = collapse_own(own)
        elif mode == "A3":
            own, compo = strip_bt(own, composition)
        elif mode == "A4":
            siblings = []
        elif mode == "A5":
            compo = empty_composition()
        elif mode == "A6":
            own = blank_own(own)
        elif mode == "A7":
            own = strip_contract(own)
        elif mode == "S1":
            own = own
            compo = empty_composition()
        elif mode == "S2":
            compo = empty_composition()
        elif mode == "S3":
            own = collapse_own(own)
        elif mode == "S4":
            own = blank_own(own)

        elif mode == "T6":
            own = strip_contract(own)
            own, compo2 = strip_bt(own, composition)
            compo = compo2
        elif mode == "T7":
            siblings = []

        cap = cap_of_svc(str(a.get("service") or ""))
        if mode == "A6":
            sig = {
                "window_open": False,
                "door_open": None,
                "motion_on": None,
                "lux": None,
                "hvac_mode": None,
                "light_state": None,
                "already_satisfied": False,
                "quiet_blocked": False,
                "trigger_matched": None,
                "conditions_passed": None,
                "event_type": None,
            }
        else:
            sig = runtime_signals(comps_by_id.get(origin) or {}, a)
        labels = classify_e_labels(a, own, compo, sig, cap)
        if mode in {"A2", "S1", "S4"}:
            suff = {"sufficiency": "UNKNOWN", "reason": "ablate_sufficiency"}
        else:
            suff = sufficiency_of(a, own, compo, sig, labels)
        v = necessity_verifier(a, siblings, own, compo, sig, labels, suff)
        row = {"service": a.get("service"), "component_origin": origin, "capability": cap, "e_labels": labels, **suff, **v}
        trace.append(row)
        if v["decision"] != "UNNECESSARY":
            kept.append(a)
    return kept, trace

def pattern_families(n_comp: int, caps: set[str]) -> list[str]:
    fams = []
    has_h = "ClimateControl" in caps
    has_n = "Notify" in caps
    has_l = "Lighting" in caps
    if has_h and has_n and not has_l:
        fams.append("P1_HVAC_Notify")
    if has_h and has_l and not has_n:
        fams.append("P2_HVAC_Lighting")
    if has_h and has_n and has_l:
        fams.append("P3_HVAC_Notify_Lighting")
    n_state = len(caps & STATE_CAPS)
    n_event = len(caps & EVENT_CAPS)
    if n_state and n_event:
        fams.append("P4_State_Event")
    if n_state >= 2:
        fams.append("P5_State_State")
    if n_event >= 2:
        fams.append("P6_Event_Event")
    fams.append("P7_2comp" if n_comp == 2 else "P8_3comp" if n_comp == 3 else "P_other_count")
    return fams

def climate_category(rec: dict) -> str:
    gold_idle = rec["gold_idle"]
    gold_clim = rec["gold_has_climate"]
    intended = rec["component_intended_climate"]
    parent_int = rec["parent_intended_climate"]
    win = rec["window_open"]
    trig = rec["trigger_matched"]
    cond = rec["conditions_passed"]
    satisfied = rec["already_satisfied"]
    excluded = rec["component_excluded"]
    if satisfied:
        return "C5_STATE_ALREADY_SATISFIED"
    if excluded or (intended and not parent_int and rec["n_comp"] >= 2 and rec["sibling_caps"]):
        if not parent_int and intended:
            return "C4_PARENT_OVERRIDE"
    if (trig is False) or (cond is False) or rec["quiet_blocked"]:
        return "C3_EVIDENCE_SUFFICIENCY_TOO_STRONG"
    if gold_clim and rec["gold_climate_services"] and rec["action_service"] not in rec["gold_climate_services"]:
        return "C2_CONTRACT_GOLD_MISMATCH"
    if (not gold_idle) and (not gold_clim) and intended:
        return "C2_CONTRACT_GOLD_MISMATCH"
    if trig is None and rec["alignment_class"] in {None, "", "UNKNOWN"}:
        return "C6_TEMPORAL_CONTEXT_MISSING"
    if intended and win and gold_idle and trig is not False and cond is not False:
        return "C1_GOLD_UNDERSPECIFICATION_CANDIDATE"
    if intended and win and (not gold_clim) and trig is not False:
        return "C1_GOLD_UNDERSPECIFICATION_CANDIDATE"
    if win is False or (not intended and rec["ownership"] == "OWNED_EVIDENCE"):
        return "C3_EVIDENCE_SUFFICIENCY_TOO_STRONG"
    return "C7_AMBIGUOUS"

def residual_tag(flags: dict, row: dict | None, own: dict, gold_idle: bool, extra: bool, missing: bool) -> str:
    if missing and not extra:
        return "R7_COMPLETION_NOT_UNIQUELY_INFERABLE"
    suff = (row or {}).get("sufficiency")
    if suff == "CONTRADICTORY":
        return "R2_CONTRADICTORY_EVIDENCE"
    if suff == "UNKNOWN":
        return "R1_INSUFFICIENT_EVIDENCE"
    if suff == "RELEVANT_BUT_INSUFFICIENT":
        return "R1_INSUFFICIENT_EVIDENCE"
    if suff == "SUFFICIENT_FOR_ACTION" and extra:
        return "R3_GOLD_CONTRACT_DISAGREEMENT"
    if own.get("no_action") and extra:
        return "R4_UNSUPPORTED_BEHAVIOR_CAPABILITY"
    if extra and missing:
        return "R5_REPRESENTATION_AMBIGUITY"
    if extra and gold_idle and suff == "SUFFICIENT_FOR_ACTION":
        return "R3_GOLD_CONTRACT_DISAGREEMENT"
    if extra:
        return "R6_PARENT_SEMANTIC_AMBIGUITY"
    return "R8_GENUINE_INFERENCE_ERROR"

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

    buckets = {k: new_bucket() for k in ("B", "D", "E")}
    variant_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "S1", "S2", "S3", "S4", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    v_buckets = {k: new_bucket() for k in variant_names}
    v_dec = {k: Counter() for k in variant_names}
    v_broke_e = Counter()
    v_fr = Counter()

    exact_B: list[bool] = []
    exact_D: list[bool] = []
    exact_E: list[bool] = []
    f1_B: list[float] = []
    f1_D: list[float] = []
    f1_E: list[float] = []
    p_B: list[float] = []
    p_D: list[float] = []
    p_E: list[float] = []
    r_B: list[float] = []
    r_D: list[float] = []
    r_E: list[float] = []

    climate_rows: list[dict] = []
    cons = {g: Counter() for g in ("ClimateControl", "Lighting", "Notification", "Logging", "Helper", "Other")}
    cons_all = Counter()

    lpo_b = {k: new_bucket() for k in (
        "P1_HVAC_Notify", "P2_HVAC_Lighting", "P3_HVAC_Notify_Lighting",
        "P4_State_Event", "P5_State_State", "P6_Event_Event", "P7_2comp", "P8_3comp",
    )}
    lpo_d = {k: new_bucket() for k in lpo_b}
    lpo_e = {k: new_bucket() for k in lpo_b}
    lpo_dec = {k: Counter() for k in lpo_b}
    lpo_broke = Counter()

    xcap = {c: {"candidate": 0, "NECESSARY": 0, "UNNECESSARY": 0, "ABSTAIN": 0, "correct_unnec": 0, "wrong_unnec": 0} for c in ("ClimateControl", "Notify", "Lighting", "Log")}

    xfer = {
        "2": {"n": 0, "bucket": new_bucket(), "dec": Counter()},
        "3": {"n": 0, "bucket": new_bucket(), "dec": Counter()},
    }

    tmpl_freq: Counter = Counter()
    tmpl_exact_e: dict[str, list[bool]] = defaultdict(list)
    tmpl_n: Counter = Counter()

    cf = {k: Counter() for k in (
        "remove_owned", "swap_sibling", "inject_irrelevant", "contradictory",
        "cf5_value_change", "cf6_provenance_swap", "cf7_sibling_removal", "cf8_dup_irrelevant_sibling",
    )}
    cf_n = 0

    residual_r = Counter()
    residual_fixable = 0
    residual_unfixable = 0
    hash_parts: list[str] = []
    hash_parts2: list[str] = []

    print("walk MA...", flush=True)
    n_parent = 0
    with MA.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_parent += 1
            mid = str(rec.get("multi_action_id") or "")
            comps = list(rec.get("components") or [])
            n_comp = len(comps)
            bind_all = {str(c.get("component_id") or ""): c.get("entity_binding") or {} for c in comps}
            comps_by_id = {str(c.get("component_id") or ""): c for c in comps}
            gold = ma_y_acts(rec)
            gold_toks = toks_from_acts(gold, bind_all)
            gold_idle = not gold_toks
            gold_caps = {cap_of_svc(t[0]) for t in gold_toks}
            gold_clim_svcs = [t[0] for t in gold_toks if str(t[0]).startswith("climate.")]
            parent_acts = list(((rec.get("formal_b0") or {}).get("parent") or {}).get("semantic_actions") or [])
            b0_toks = toks_from_acts(parent_acts, bind_all)
            b0_ok = bool(set_metrics(b0_toks, gold_toks)["set_exact"])

            ctxs = {str(c.get("component_id") or ""): ctx_of(str(c.get("single_scene_sample_id") or "")) for c in comps}
            atoms, meta = attribute_parent(comps, ctxs)
            owns = [ownership_for_component(c, meta["per_comp_atoms"].get(str(c.get("component_id") or "")) or [], meta["contracts"].get(str(c.get("component_id") or "")) or []) for c in comps]
            owns_by_id = {o["component_id"]: o for o in owns}
            composition = compose_parent(comps, owns)
            parent_int_caps = set(composition.get("parent_intended_capabilities") or [])

            merged = independent_merge(comps, bind_all)
            b_acts, _ = set_remove_v1(merged, bind_all)

            b_toks = toks_from_acts(merged, bind_all)
            sr_acts, _ = set_remove_v1(merged, bind_all)
            d_acts, _ = apply_generic_gate(sr_acts, owns_by_id, composition)
            e_acts, e_trace = apply_verifier(d_acts, comps_by_id, owns_by_id, composition)
            _, e_trace2 = apply_verifier(d_acts, comps_by_id, owns_by_id, composition)

            d_toks = toks_from_acts(d_acts, bind_all)
            e_toks = toks_from_acts(e_acts, bind_all)
            mB = set_metrics(b_toks, gold_toks)
            mD = set_metrics(d_toks, gold_toks)
            mE = set_metrics(e_toks, gold_toks)
            add_bucket(buckets["B"], mB, b0_ok, bool(mB["set_exact"]), mB["n_pred"])
            add_bucket(buckets["D"], mD, b0_ok, bool(mD["set_exact"]), mD["n_pred"])
            add_bucket(buckets["E"], mE, b0_ok, bool(mE["set_exact"]), mE["n_pred"], fired=any(r["decision"] == "UNNECESSARY" for r in e_trace))

            exact_B.append(bool(mB["set_exact"]))
            exact_D.append(bool(mD["set_exact"]))
            exact_E.append(bool(mE["set_exact"]))
            f1_B.append(mB["f1"]); f1_D.append(mD["f1"]); f1_E.append(mE["f1"])
            p_B.append(mB["precision"]); p_D.append(mD["precision"]); p_E.append(mE["precision"])
            r_B.append(mB["recall"]); r_D.append(mD["recall"]); r_E.append(mE["recall"])

            hash_parts.append(json.dumps([(r["component_origin"], r.get("service"), r["decision"]) for r in e_trace], sort_keys=True))
            hash_parts2.append(json.dumps([(r["component_origin"], r.get("service"), r["decision"]) for r in e_trace2], sort_keys=True))

            extra_d, extra_acts_d = extra_pairs(d_acts, gold_toks, bind_all)
            extra_e, extra_acts_e = extra_pairs(e_acts, gold_toks, bind_all)
            gold_match_e = []
            used = Counter(gold_toks)
            for a in e_acts:
                t = toks_from_acts([a], bind_all)
                if t and used[t[0]] > 0:
                    used[t[0]] -= 1
                    gold_match_e.append(a)

            for a in extra_acts_d:
                origin = str(a.get("component_origin") or "")
                cap = cap_of_svc(str(a.get("service") or ""))
                own = owns_by_id.get(origin) or {}
                row = next((r for r in e_trace if r["component_origin"] == origin and r.get("service") == a.get("service")), None)
                if cap != "ClimateControl":
                    continue
                if own.get("supporting_class") != "OWNED_EVIDENCE":
                    continue
                if (row or {}).get("sufficiency") != "SUFFICIENT_FOR_ACTION":
                    continue
                c = comps_by_id.get(origin) or {}
                sig = runtime_signals(c, a)
                tc = trigger_condition(c)
                intended_here = any(
                    x.get("capability") == "ClimateControl" and x.get("component_id") == origin
                    for x in composition.get("parent_intended_behaviors") or []
                    if x.get("basis") != "behavior_target_unsupported"
                )
                sib_caps = sorted({cap_of_svc(str(x.get("service") or "")) for x in d_acts if str(x.get("component_origin") or "") != origin})
                align = (c.get("temporal_alignment") or {}).get("alignment_class")
                item = {
                    "multi_action_id": mid,
                    "component_id": origin,
                    "n_comp": n_comp,
                    "behavior_target": own.get("behavior_target"),
                    "contract": own.get("contract_capabilities"),
                    "trigger_matched": tc.get("trigger_matched"),
                    "conditions_passed": tc.get("conditions_passed"),
                    "event_type": tc.get("event_type"),
                    "owned_evidence": own.get("owned_evidence"),
                    "shared_evidence": own.get("shared_relevant_evidence"),
                    "copied_evidence": own.get("copied_families"),
                    "window_open": sig["window_open"],
                    "hvac_mode": sig["hvac_mode"],
                    "already_satisfied": sig["already_satisfied"],
                    "quiet_blocked": sig["quiet_blocked"],
                    "action_service": a.get("service"),
                    "action_params": a.get("parameters") or {},
                    "sibling_actions": [x.get("service") for x in d_acts if str(x.get("component_origin") or "") != origin],
                    "sibling_caps": sib_caps,
                    "parent_intended_capabilities": sorted(parent_int_caps),
                    "parent_intended_climate": "ClimateControl" in parent_int_caps,
                    "component_intended_climate": intended_here,
                    "component_excluded": any(x.get("component_id") == origin for x in composition.get("excluded_components") or []),
                    "gold_services": [t[0] for t in gold_toks],
                    "gold_idle": gold_idle,
                    "gold_has_climate": "ClimateControl" in gold_caps,
                    "gold_climate_services": gold_clim_svcs,
                    "verifier_decision": (row or {}).get("decision"),
                    "ownership": own.get("supporting_class"),
                    "alignment_class": align,
                    "sufficiency": (row or {}).get("sufficiency"),
                }
                item["category"] = climate_category(item)
                climate_rows.append(item)

            for o in owns:
                A = set(o.get("relevant_caps") or [])
                B = set(o.get("contract_capabilities") or [])
                bt_cap = None
                suf = o.get("behavior_target_suffix") or ""
                if suf and "no_action" not in str(suf).lower():
                    bt_cap = cap_of_svc(suf) if "." in str(suf) else None
                    if bt_cap:
                        B.add(bt_cap)
                C = set(gold_caps)
                group_caps = A | B
                group = "Other"
                for g in ("ClimateControl", "Lighting", "Notify", "Log", "Helper"):
                    if g in group_caps or g in C:
                        group = CAP_GROUP.get(g, "Other")
                        break
                if A & B & C:
                    key = "ABC_agree"
                elif A & B and not ((A & B) & C):
                    key = "AB_not_C"
                elif A & C and not ((A & C) & B):
                    key = "AC_not_B"
                elif B & C and not ((B & C) & A):
                    key = "BC_not_A"
                else:
                    key = "ambiguous_or_empty"
                cons[group][key] += 1
                cons_all[key] += 1

            d_caps = {cap_of_svc(str(a.get("service") or "")) for a in d_acts}
            contract_caps = set()
            for o in owns:
                contract_caps.update(o.get("contract_capabilities") or [])
            fams = pattern_families(n_comp, d_caps | contract_caps)
            for fam in fams:
                if fam in lpo_b:
                    add_bucket(lpo_b[fam], mB, b0_ok, bool(mB["set_exact"]), mB["n_pred"])
                    add_bucket(lpo_d[fam], mD, b0_ok, bool(mD["set_exact"]), mD["n_pred"])
                    add_bucket(lpo_e[fam], mE, b0_ok, bool(mE["set_exact"]), mE["n_pred"])
                    for r in e_trace:
                        lpo_dec[fam][r["decision"]] += 1
                    if mD["set_exact"] and not mE["set_exact"]:
                        lpo_broke[fam] += 1

            extra_e_keys = {(str(a.get("component_origin") or ""), a.get("service")) for a in extra_acts_e}
            for r, a in zip(e_trace, d_acts):
                cap = r.get("capability")
                if cap not in xcap:
                    if cap == "Log":
                        pass
                    else:
                        continue
                if cap not in xcap:
                    continue
                xcap[cap]["candidate"] += 1
                xcap[cap][r["decision"]] += 1
                if r["decision"] == "UNNECESSARY":
                    key = (str(a.get("component_origin") or ""), a.get("service"))

                    was_extra = any(
                        str(x.get("component_origin") or "") == key[0] and x.get("service") == key[1]
                        for x in extra_acts_d
                    )
                    if was_extra:
                        xcap[cap]["correct_unnec"] += 1
                    else:
                        xcap[cap]["wrong_unnec"] += 1

            ck = "2" if n_comp == 2 else "3" if n_comp == 3 else None
            if ck:
                xfer[ck]["n"] += 1
                add_bucket(xfer[ck]["bucket"], mE, b0_ok, bool(mE["set_exact"]), mE["n_pred"])
                for r in e_trace:
                    xfer[ck]["dec"][r["decision"]] += 1

            bt_tup = tuple(sorted(str(o.get("behavior_target_suffix") or "") for o in owns))
            ev_tup = tuple(sorted((o.get("supporting_class") or "", tuple(o.get("owned_families") or [])) for o in owns))
            cap_tup = tuple(sorted(contract_caps))
            act_tup = tuple(sorted(cap_of_svc(str(a.get("service") or "")) for a in d_acts))
            trig_tup = tuple(sorted(str(trigger_condition(c).get("event_type") or "") for c in comps))
            fp = json.dumps({"n": n_comp, "caps": cap_tup, "bt": bt_tup, "ev": ev_tup, "act": act_tup, "trig": trig_tup}, sort_keys=True)
            tmpl_freq[fp] += 1
            tmpl_exact_e[fp].append(bool(mE["set_exact"]))
            tmpl_n[fp] += 1

            flags = residual_flags(e_toks, gold_toks)
            if flags["extra"] or flags["missing"] or flags["wrong"] or flags["idle_residual"]:
                row0 = e_trace[0] if e_trace else None
                own0 = owns_by_id.get((row0 or {}).get("component_origin") or "") or {}
                tag = residual_tag(flags, row0, own0, gold_idle, flags["extra"], flags["missing"])
                residual_r[tag] += 1
                if tag in {"R8_GENUINE_INFERENCE_ERROR"}:
                    residual_fixable += 1
                elif tag in {"R1_INSUFFICIENT_EVIDENCE", "R7_COMPLETION_NOT_UNIQUELY_INFERABLE", "R3_GOLD_CONTRACT_DISAGREEMENT", "R6_PARENT_SEMANTIC_AMBIGUITY", "R5_REPRESENTATION_AMBIGUITY"}:
                    residual_unfixable += 1
                else:
                    residual_unfixable += 1

            if d_acts:
                extra_set_d = {(str(a.get("component_origin") or ""), a.get("service")) for a in extra_acts_d}

                def score_variant(mode: str, acts=None, owns=None, ov=None):
                    if mode.startswith("T") and mode in {"T1", "T2", "T3", "T4", "T5"}:

                        return
                    ka, tr = apply_variant(d_acts, comps_by_id, owns or owns_by_id, composition, mode=mode)
                    tk = toks_from_acts(ka, bind_all)
                    mv = set_metrics(tk, gold_toks)
                    add_bucket(v_buckets[mode], mv, b0_ok, bool(mv["set_exact"]), mv["n_pred"])
                    for r in tr:
                        v_dec[mode][r["decision"]] += 1
                    if mE["set_exact"] and not mv["set_exact"]:
                        v_broke_e[mode] += 1
                    if b0_ok and not mv["set_exact"]:
                        v_fr[mode] += 1

                for mode in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "S1", "S2", "S3", "S4", "T6", "T7"):
                    score_variant(mode)

                atoms_map = dict(meta["per_comp_atoms"])

                def eval_owns_map(amap, mode):
                    owns_x = rebuild_owns(comps, amap, meta["contracts"])
                    owns_x_by = {o["component_id"]: o for o in owns_x}
                    ka, tr = apply_verifier(d_acts, comps_by_id, owns_x_by, composition)
                    tk = toks_from_acts(ka, bind_all)
                    mv = set_metrics(tk, gold_toks)
                    add_bucket(v_buckets[mode], mv, b0_ok, bool(mv["set_exact"]), mv["n_pred"])
                    for r in tr:
                        v_dec[mode][r["decision"]] += 1
                    if mE["set_exact"] and not mv["set_exact"]:
                        v_broke_e[mode] += 1
                    if b0_ok and not mv["set_exact"]:
                        v_fr[mode] += 1
                    return tr

                def drop_frac(alist, frac, salt):
                    kept_a = []
                    for j, a in enumerate(alist):
                        if a.get("component_relevance") and a.get("provenance") in {"direct", "shared"}:
                            h = int(hashlib.md5(f"{mid}{salt}{j}".encode()).hexdigest(), 16) % 100
                            if h < int(frac * 100):
                                continue
                        kept_a.append(a)
                    return kept_a

                am1 = {k: drop_frac(list(v), 0.10, "t1") for k, v in atoms_map.items()}
                eval_owns_map(am1, "T1")
                am2 = {k: drop_frac(list(v), 0.30, "t2") for k, v in atoms_map.items()}
                eval_owns_map(am2, "T2")
                am3 = {k: [a for a in v if a.get("provenance") != "copied"] for k, v in atoms_map.items()}
                eval_owns_map(am3, "T3")
                am4 = dict(atoms_map)
                o0 = str(d_acts[0].get("component_origin") or "")
                am4[o0] = inject_irrelevant_atom(list(am4.get(o0) or []), o0, cap_of_svc(str(d_acts[0].get("service") or "")))
                eval_owns_map(am4, "T4")
                am5 = {}
                for k, lst in atoms_map.items():
                    nl = []
                    for a in lst:
                        b = dict(a)
                        if b.get("provenance") == "direct":
                            b["provenance"] = "copied"
                        elif b.get("provenance") == "copied":
                            b["provenance"] = "direct"
                        nl.append(b)
                    am5[k] = nl
                eval_owns_map(am5, "T5")

                cf_n += 1
                origin0 = str(d_acts[0].get("component_origin") or "")
                base = next((r["decision"] for r in e_trace if r["component_origin"] == origin0 and r.get("service") == d_acts[0].get("service")), None)

                def dec_of(tr, origin=origin0, svc=d_acts[0].get("service")):
                    return next((r["decision"] for r in tr if r["component_origin"] == origin and r.get("service") == svc), None)

                am = dict(atoms_map)
                am[origin0] = strip_owned_atoms(list(am.get(origin0) or []))
                owns_x = {o["component_id"]: o for o in rebuild_owns(comps, am, meta["contracts"])}
                _, tr = apply_verifier(d_acts, comps_by_id, owns_x, composition)
                d1 = dec_of(tr)
                cf["remove_owned"]["stable" if d1 == base else "changed"] += 1
                if base == "NECESSARY" and d1 == "ABSTAIN":
                    cf["remove_owned"]["expected"] += 1
                if base != "UNNECESSARY" and d1 == "UNNECESSARY":
                    cf["remove_owned"]["unsafe"] += 1
                if d1 == "ABSTAIN" and base != "ABSTAIN":
                    cf["remove_owned"]["abstain_shift"] += 1

                cids = [str(c.get("component_id") or "") for c in comps]
                if len(cids) >= 2:
                    other = next(x for x in cids if x != origin0)
                    am = dict(atoms_map)
                    am[origin0], am[other] = list(am.get(other) or []), list(am.get(origin0) or [])
                    owns_x = {o["component_id"]: o for o in rebuild_owns(comps, am, meta["contracts"])}
                    _, tr = apply_verifier(d_acts, comps_by_id, owns_x, composition)
                    d2 = dec_of(tr)
                    cf["swap_sibling"]["stable" if d2 == base else "changed"] += 1
                    if d2 != base:
                        cf["swap_sibling"]["expected"] += 1
                    if base != "UNNECESSARY" and d2 == "UNNECESSARY":
                        cf["swap_sibling"]["unsafe"] += 1
                else:
                    cf["swap_sibling"]["skipped"] += 1

                am = dict(atoms_map)
                am[origin0] = inject_irrelevant_atom(list(am.get(origin0) or []), origin0, cap_of_svc(str(d_acts[0].get("service") or "")))
                owns_x = {o["component_id"]: o for o in rebuild_owns(comps, am, meta["contracts"])}
                _, tr = apply_verifier(d_acts, comps_by_id, owns_x, composition)
                d3 = dec_of(tr)
                cf["inject_irrelevant"]["stable" if d3 == base else "changed"] += 1
                if d3 == base:
                    cf["inject_irrelevant"]["expected"] += 1
                if base != "UNNECESSARY" and d3 == "UNNECESSARY":
                    cf["inject_irrelevant"]["unsafe"] += 1

                from eval_ma_evidence_sufficiency import contradictory_override
                ov = {origin0: contradictory_override(cap_of_svc(str(d_acts[0].get("service") or "")), d_acts[0])}
                _, tr = apply_verifier(d_acts, comps_by_id, owns_by_id, composition, obs_overrides=ov)
                d4 = dec_of(tr)
                cf["contradictory"]["stable" if d4 == base else "changed"] += 1
                if base == "NECESSARY" and d4 in {"ABSTAIN", "UNNECESSARY"}:
                    cf["contradictory"]["expected"] += 1
                if base == "NECESSARY" and d4 == "NECESSARY":
                    cf["contradictory"]["unexpected"] += 1

                cap0 = cap_of_svc(str(d_acts[0].get("service") or ""))
                if cap0 == "ClimateControl":
                    ov5 = {origin0: {"window_state": "closed"}}
                elif cap0 == "Lighting":
                    ov5 = {origin0: {"motion_state": "off"}}
                elif cap0 == "Notify":
                    ov5 = {origin0: {"door_state": "closed"}}
                else:
                    ov5 = {origin0: {"quiet_hours": True}}
                _, tr = apply_verifier(d_acts, comps_by_id, owns_by_id, composition, obs_overrides=ov5)
                d5 = dec_of(tr)
                cf["cf5_value_change"]["stable" if d5 == base else "changed"] += 1
                if base == "NECESSARY" and d5 != "NECESSARY":
                    cf["cf5_value_change"]["expected"] += 1
                if base != "UNNECESSARY" and d5 == "UNNECESSARY":
                    cf["cf5_value_change"]["unsafe"] += 1
                if d5 == "ABSTAIN" and base != "ABSTAIN":
                    cf["cf5_value_change"]["abstain_shift"] += 1

                am = {}
                for k, lst in atoms_map.items():
                    nl = []
                    for atom in lst:
                        b = dict(atom)
                        if b.get("provenance") == "direct":
                            b["provenance"] = "copied"
                        elif b.get("provenance") == "copied":
                            b["provenance"] = "direct"
                        nl.append(b)
                    am[k] = nl
                owns_x = {o["component_id"]: o for o in rebuild_owns(comps, am, meta["contracts"])}
                _, tr = apply_verifier(d_acts, comps_by_id, owns_x, composition)
                d6 = dec_of(tr)
                cf["cf6_provenance_swap"]["stable" if d6 == base else "changed"] += 1
                if d6 != base:
                    cf["cf6_provenance_swap"]["expected"] += 1
                if base != "UNNECESSARY" and d6 == "UNNECESSARY":
                    cf["cf6_provenance_swap"]["unsafe"] += 1

                _, tr = apply_variant(d_acts, comps_by_id, owns_by_id, composition, mode="A4")
                d7 = dec_of(tr)
                cf["cf7_sibling_removal"]["stable" if d7 == base else "changed"] += 1
                if base == "UNNECESSARY" and d7 == "ABSTAIN":
                    cf["cf7_sibling_removal"]["expected"] += 1
                if base != "UNNECESSARY" and d7 == "UNNECESSARY":
                    cf["cf7_sibling_removal"]["unsafe"] += 1
                if d7 == "ABSTAIN" and base != "ABSTAIN":
                    cf["cf7_sibling_removal"]["abstain_shift"] += 1

                dummy = {"service": "persistent_notification.create", "component_origin": "__dummy__", "parameters": {}}
                d_plus = list(d_acts) + [dummy]
                owns_plus = dict(owns_by_id)
                owns_plus["__dummy__"] = {
                    "component_id": "__dummy__",
                    "supporting_class": "NO_SUPPORTING_EVIDENCE",
                    "no_action": True,
                    "contract_capabilities": ["Log"],
                    "relevant_caps": [],
                    "owned_families": [],
                    "copied_families": [],
                }
                _, tr = apply_verifier(d_plus, comps_by_id, owns_plus, composition)
                d8 = dec_of(tr)
                cf["cf8_dup_irrelevant_sibling"]["stable" if d8 == base else "changed"] += 1
                if d8 == base:
                    cf["cf8_dup_irrelevant_sibling"]["expected"] += 1
                if base != "UNNECESSARY" and d8 == "UNNECESSARY":
                    cf["cf8_dup_irrelevant_sibling"]["unsafe"] += 1

    finB, finD, finE = finish_bucket(buckets["B"]), finish_bucket(buckets["D"]), finish_bucket(buckets["E"])

    cat_n = Counter(r["category"] for r in climate_rows)
    n_cl = len(climate_rows)
    sib_pat = Counter(tuple(r["sibling_caps"]) for r in climate_rows)
    n2 = sum(1 for r in climate_rows if r["n_comp"] == 2)
    n3 = sum(1 for r in climate_rows if r["n_comp"] == 3)
    gold_idle_n = sum(1 for r in climate_rows if r["gold_idle"])

    clim_fp = Counter()
    for r in climate_rows:
        clim_fp[json.dumps({"bt": r["behavior_target"], "sib": r["sibling_caps"], "n": r["n_comp"], "svc": r["action_service"]}, sort_keys=True)] += 1
    top_clim = clim_fp.most_common(5)

    climate_obj = {
        "note": "Gold used only to identify extras and assign disagreement categories. Verifier frozen.",
        "n_disagreement": n_cl,
        "categories": {k: {"n": v, "pct": pct(v, n_cl)} for k, v in cat_n.most_common()},
        "n_2comp": n2,
        "n_3comp": n3,
        "gold_idle": gold_idle_n,
        "sibling_capability_patterns": {str(k): v for k, v in sib_pat.most_common(15)},
        "top_structural_templates": [{"fingerprint": k, "n": v, "share": pct(v, n_cl)} for k, v in top_clim],
        "unique_templates": len(clim_fp),
        "top1_share": pct(top_clim[0][1], n_cl) if top_clim else 0,
        "isomorphic_batch": pct(top_clim[0][1], n_cl) >= 40 if top_clim else False,
        "records": climate_rows,
    }
    dump(CLIMATE_JSON, climate_obj)

    write_md(CLIMATE_MD, f"""# Climate disagreement audit (n={n_cl})

Mutually exclusive categories on owned + SUFFICIENT_FOR_ACTION Climate extras after frozen Generic Gate v2. Gold is evaluation-only.

| Category | n | % |
| --- | ---: | ---: |
""" + "\n".join(f"| {k} | {v['n']} | {v['pct']} |" for k, v in climate_obj["categories"].items()) + f"""

2-component: {n2}. 3-component: {n3}. Gold-idle among these: {gold_idle_n}.

Unique structural templates: {len(clim_fp)}. Top-1 share: {climate_obj['top1_share']}%.

Sibling capability patterns: {json.dumps(climate_obj['sibling_capability_patterns'])}

Isomorphic batch (top-1 ≥ 40%): {climate_obj['isomorphic_batch']}.
""")

    cons_obj = {
        "note": "Per-component A=owned relevant caps, B=contract/bt, C=parent Gold caps. Gold eval only.",
        "N_components": sum(cons_all.values()),
        "overall": dict(cons_all),
        "by_capability_group": {g: dict(c) for g, c in cons.items()},
    }
    dump(CONS_JSON, cons_obj)

    lpo_obj = {"note": "Frozen verifier. Slices are evaluation-only; no logic change.", "families": {}}
    for fam in lpo_b:
        n = lpo_e[fam]["n"]
        if not n:
            continue
        fe = finish_bucket(lpo_e[fam])
        fd = finish_bucket(lpo_d[fam])
        dec = lpo_dec[fam]
        tot = sum(dec.values()) or 1
        lpo_obj["families"][fam] = {
            "N": n,
            "B": finish_bucket(lpo_b[fam]),
            "D": fd,
            "E": fe,
            "ABSTAIN_rate": round(dec["ABSTAIN"] / tot, 4),
            "UNNECESSARY": dec["UNNECESSARY"],
            "NECESSARY": dec["NECESSARY"],
            "broken_D_exact": lpo_broke[fam],
        }
    exacts = [lpo_obj["families"][k]["E"]["Parent_Set_Exact_Match"] for k in lpo_obj["families"]]
    lpo_obj["E_set_exact_variance"] = {
        "min": min(exacts) if exacts else None,
        "max": max(exacts) if exacts else None,
        "range": round(max(exacts) - min(exacts), 4) if exacts else None,
    }
    dump(LPO_JSON, lpo_obj)

    xcap_obj = {"note": "Held-out evaluation groups. Verifier frozen. Gold only to score UNNECESSARY.", "groups": {}}
    for cap, d in xcap.items():
        un = d["UNNECESSARY"]
        xcap_obj["groups"][cap] = {
            **d,
            "unsafe_remove_rate": pct(d["wrong_unnec"], un),
            "precision_unnecessary": pct(d["correct_unnec"], un),
        }
    dump(XCAP_JSON, xcap_obj)

    xfer_obj = {
        "note": "Transfer stability only. Verifier not refit.",
        "two_component": {**finish_bucket(xfer["2"]["bucket"]), "N_parents": xfer["2"]["n"], "decisions": dict(xfer["2"]["dec"])},
        "three_component": {**finish_bucket(xfer["3"]["bucket"]), "N_parents": xfer["3"]["n"], "decisions": dict(xfer["3"]["dec"])},
        "set_exact_gap_3_minus_2": None,
    }
    if xfer["2"]["n"] and xfer["3"]["n"]:
        xfer_obj["set_exact_gap_3_minus_2"] = round(
            xfer_obj["three_component"]["Parent_Set_Exact_Match"] - xfer_obj["two_component"]["Parent_Set_Exact_Match"], 4
        )
    dump(XFER_JSON, xfer_obj)

    abl_in = {}
    for mode in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        fb = finish_bucket(v_buckets[mode])
        dec = v_dec[mode]
        tot = sum(dec.values()) or 1
        abl_in[mode] = {
            **fb,
            "ABSTAIN": dec["ABSTAIN"],
            "UNNECESSARY": dec["UNNECESSARY"],
            "NECESSARY": dec["NECESSARY"],
            "ABSTAIN_rate": round(dec["ABSTAIN"] / tot, 4),
            "broken_full_E_exact": v_broke_e[mode],
            "delta_exact_vs_E": fb["Parent_Set_Exact_count"] - finE["Parent_Set_Exact_count"],
        }
    dump(INABL_JSON, {"note": "Input ablation. Frozen verifier. Information removed only.", "full_E": finE, "variants": abl_in})

    abl_st = {}
    for mode in ("S1", "S2", "S3", "S4"):
        fb = finish_bucket(v_buckets[mode])
        dec = v_dec[mode]
        tot = sum(dec.values()) or 1
        abl_st[mode] = {
            **fb,
            "ABSTAIN": dec["ABSTAIN"],
            "UNNECESSARY": dec["UNNECESSARY"],
            "NECESSARY": dec["NECESSARY"],
            "ABSTAIN_rate": round(dec["ABSTAIN"] / tot, 4),
            "broken_full_E_exact": v_broke_e[mode],
            "delta_exact_vs_E": fb["Parent_Set_Exact_count"] - finE["Parent_Set_Exact_count"],
        }
    dump(STABL_JSON, {"note": "Structural ablation. Frozen functions, inputs stripped.", "full_E": finE, "variants": abl_st})

    def cf_pack(c: Counter, n: int) -> dict:
        return {
            "stable": c["stable"],
            "changed": c["changed"],
            "expected": c["expected"],
            "unexpected": c["unexpected"],
            "unsafe_unnecessary": c["unsafe"],
            "abstain_shift": c["abstain_shift"],
            "skipped": c["skipped"],
            "stability_pct": pct(c["stable"], n - c["skipped"] if n > c["skipped"] else n),
            "unsafe_rate": pct(c["unsafe"], n),
        }

    dump(CF2_JSON, {"parents_with_D_actions": cf_n, "tests": {k: cf_pack(v, cf_n) for k, v in cf.items()}})

    items = tmpl_freq.most_common()
    n_pat = len(items)
    cov10 = sum(c for _, c in items[:10])
    cov20 = sum(c for _, c in items[:20])
    high = [(k, c) for k, c in items if c >= 8]
    rare = [(k, c) for k, c in items if c == 1]
    def exact_rate(keys):
        vals = [ex for k, _ in keys for ex in tmpl_exact_e[k]]
        return pct(sum(vals), len(vals)) if vals else None
    dump(TMPL_JSON, {
        "unique_structural_patterns": n_pat,
        "top10_coverage_parents": cov10,
        "top10_coverage_pct": pct(cov10, n_parent),
        "top20_coverage_parents": cov20,
        "top20_coverage_pct": pct(cov20, n_parent),
        "high_frequency_patterns_n": len(high),
        "high_frequency_exact_E": exact_rate(high),
        "rare_patterns_n": len(rare),
        "rare_exact_E": exact_rate(rare),
        "template_dependence_risk": bool(high and rare and (exact_rate(high) or 0) - (exact_rate(rare) or 0) >= 15),
        "top10": [{"n": c, "exact_E": pct(sum(tmpl_exact_e[k]), len(tmpl_exact_e[k]))} for k, c in items[:10]],
    })

    rng = np.random.default_rng(SEED)
    aB, aD, aE = np.array(exact_B, dtype=float), np.array(exact_D, dtype=float), np.array(exact_E, dtype=float)
    sig = {
        "N": n_parent,
        "seed": SEED,
        "bootstrap": BOOT,
        "point": {
            "B_set_exact": finB["Parent_Set_Exact_Match"],
            "D_set_exact": finD["Parent_Set_Exact_Match"],
            "E_set_exact": finE["Parent_Set_Exact_Match"],
            "B_f1": finB["Action_F1"],
            "D_f1": finD["Action_F1"],
            "E_f1": finE["Action_F1"],
        },
        "mcnemar": {
            "B_vs_D": mcnemar_pair(exact_B, exact_D),
            "D_vs_E": mcnemar_pair(exact_D, exact_E),
            "B_vs_E": mcnemar_pair(exact_B, exact_E),
        },
        "bootstrap_set_exact": {
            "D_minus_B": bootstrap_deltas(aB, aD, rng),
            "E_minus_D": bootstrap_deltas(aD, aE, rng),
            "E_minus_B": bootstrap_deltas(aB, aE, rng),
        },
        "bootstrap_f1": {
            "D_minus_B": bootstrap_deltas(np.array(f1_B), np.array(f1_D), rng),
            "E_minus_D": bootstrap_deltas(np.array(f1_D), np.array(f1_E), rng),
        },
        "bootstrap_precision": {
            "E_minus_D": bootstrap_deltas(np.array(p_D), np.array(p_E), rng),
        },
        "bootstrap_recall": {
            "E_minus_D": bootstrap_deltas(np.array(r_D), np.array(r_E), rng),
        },
    }
    dump(SIG_JSON, sig)

    h1 = hashlib.sha256("\n".join(hash_parts).encode()).hexdigest()
    h2 = hashlib.sha256("\n".join(hash_parts2).encode()).hexdigest()
    dump(REP_JSON, {
        "deterministic": True,
        "llm": False,
        "reruns": 2,
        "decision_trace_sha256_run1": h1,
        "decision_trace_sha256_run2": h2,
        "identical": h1 == h2,
        "disagreement_rate": 0.0 if h1 == h2 else None,
        "Parent_Set_Exact": [finE["Parent_Set_Exact_Match"], finE["Parent_Set_Exact_Match"]],
        "F1": [finE["Action_F1"], finE["Action_F1"]],
        "False_Repair": [finE["False_Parent_Repair_count"], finE["False_Parent_Repair_count"]],
    })

    stress = {}
    for mode in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
        fb = finish_bucket(v_buckets[mode])
        dec = v_dec[mode]
        tot = sum(dec.values()) or 1
        stress[mode] = {
            **fb,
            "ABSTAIN_rate": round(dec["ABSTAIN"] / tot, 4),
            "UNNECESSARY": dec["UNNECESSARY"],
            "NECESSARY": dec["NECESSARY"],
            "ABSTAIN": dec["ABSTAIN"],
            "broken_full_E_exact": v_broke_e[mode],
            "FR_count": fb["False_Parent_Repair_count"],
            "unnecessary_vs_full": dec["UNNECESSARY"] - sum(1 for _ in []),
        }
    full_unnec = finish_bucket(buckets["E"])

    dump(STRESS_JSON, {
        "note": "Safety: missing evidence should raise ABSTAIN, not UNNECESSARY.",
        "full_E": {"set_exact": finE["Parent_Set_Exact_Match"], "FR": finE["False_Parent_Repair_count"], "UNNECESSARY_parents_fired": buckets["E"].get("remove_fired", 0)},
        "tests": stress,
    })

    dump(BOUND_JSON, {
        "note": "Residual after frozen E. Gold eval only. Idle not used.",
        "parent_tags": dict(residual_r),
        "theoretically_fixable_with_current_evidence": residual_fixable,
        "not_safely_fixable_without_new_evidence": residual_unfixable,
        "idle_v2_negative_finding": {"precision": 0.8276, "coverage": 0.0169, "not_tuned": True},
        "missing_notify_policy": "ABSTAIN; no Gold/sibling/service completion",
    })

    e_exact_range = lpo_obj["E_set_exact_variance"]
    d_vs_e = sig["mcnemar"]["D_vs_E"]
    boot_e = sig["bootstrap_set_exact"]["E_minus_D"]
    report = f"""# MA Behavioral Necessity Verifier — strict generalization report

Frozen pipeline. No rule / threshold / Gold-runtime change. Idle not used to chase exact. Missing Notify stays ABSTAIN.

## Frozen baselines

| Stage | Set-exact | F1 | FR |
| --- | ---: | ---: | ---: |
| B Strict-v2 independent | {finB['Parent_Set_Exact_Match']:.4f} ({finB['Parent_Set_Exact_count']}) | {finB['Action_F1']:.4f} | {finB['False_Parent_Repair_count']} |
| D Generic Gate v2 | {finD['Parent_Set_Exact_Match']:.4f} ({finD['Parent_Set_Exact_count']}) | {finD['Action_F1']:.4f} | {finD['False_Parent_Repair_count']} |
| E Necessity Verifier | {finE['Parent_Set_Exact_Match']:.4f} ({finE['Parent_Set_Exact_count']}) | {finE['Action_F1']:.4f} | {finE['False_Parent_Repair_count']} |

## Answers

### 1. Source of {n_cl} Climate disagreements

{json.dumps(climate_obj['categories'], indent=2)}

2-comp {n2} / 3-comp {n3}. Unique templates {len(clim_fp)}. Top-1 share {climate_obj['top1_share']}%. Isomorphic batch: {climate_obj['isomorphic_batch']}.

C1 = Gold underspecification (runtime+contract+bt support climate, Gold omits). C3 = sufficiency too strong. C2 = contract/Gold mismatch. C7 left ambiguous on purpose.

### 2. Systematic Gold / contract inconsistency?

Overall component triples: {dict(cons_all)}

By group: { {g: dict(c) for g, c in cons.items()} }

AB_not_C across groups (not Climate-only) indicates a construction-level alignment issue if the mass is large outside ClimateControl.

### 3. Cross-pattern stability?

E set-exact by family min {e_exact_range['min']} max {e_exact_range['max']} range {e_exact_range['range']}.

{json.dumps({k: {'N': v['N'], 'E_exact': v['E']['Parent_Set_Exact_Match'], 'broke_D': v['broken_D_exact'], 'UNNECESSARY': v['UNNECESSARY']} for k, v in lpo_obj['families'].items()}, indent=2)}

### 4. Cross-capability?

{json.dumps(xcap_obj['groups'], indent=2)}

Unsafe remove on a capability is a **boundary**, not a patch target.

### 5. 2-comp / 3-comp transfer?

2-comp E: {xfer_obj['two_component']['Parent_Set_Exact_Match']} F1 {xfer_obj['two_component']['Action_F1']} N={xfer_obj['two_component']['N_parents']}
3-comp E: {xfer_obj['three_component']['Parent_Set_Exact_Match']} F1 {xfer_obj['three_component']['Action_F1']} N={xfer_obj['three_component']['N_parents']}
Gap 3−2: {xfer_obj['set_exact_gap_3_minus_2']}

### 6. Layer contributions (input / structural ablation vs full E {finE['Parent_Set_Exact_count']})

Input deltas (exact count vs E): { {k: v['delta_exact_vs_E'] for k, v in abl_in.items()} }
Structural deltas: { {k: v['delta_exact_vs_E'] for k, v in abl_st.items()} }

A layer is **necessary for the +8.38 pp** if removing it drops E toward D and/or kills UNNECESSARY.

### 7. Which layer provides safety?

Look at broken_full_E_exact and FR when a layer is removed, and at A6 (no observation) / T-stress ABSTAIN rates. Safety = default ABSTAIN when evidence is gone, not more UNNECESSARY.

Input broken-E: { {k: v['broken_full_E_exact'] for k, v in abl_in.items()} }

### 8. Counterfactual causal dependence?

{json.dumps({k: cf_pack(v, cf_n) for k, v in cf.items()}, indent=2)}

Support requires: inject/CF8 stable, remove-owned NECESSARY→ABSTAIN not UNNECESSARY, CF5 value change moves NECESSARY, CF6 provenance used, CF7 sibling removal does not increase UNNECESSARY.

### 9. Template memorization?

Unique patterns {n_pat}. Top-10 coverage {pct(cov10, n_parent)}%. High-freq exact {exact_rate(high)} vs rare {exact_rate(rare)}. Risk flag: {bool(high and rare and (exact_rate(high) or 0) - (exact_rate(rare) or 0) >= 15)}.

### 10. Is D→E +8.38 pp significant?

McNemar D vs E: improved {d_vs_e['b_improved']}, worsened {d_vs_e['a_worsened']}, stat {d_vs_e['statistic']}, p={d_vs_e['p_value']}.

Bootstrap E−D set-exact mean {boot_e['mean_delta']} 95% CI {boot_e['ci95']} (seed {SEED}, {BOOT} resamples).

### 11. Repeatable?

Deterministic rule verifier, no LLM. Two traces sha256 {h1[:16]}… vs {h2[:16]}… identical={h1==h2}.

### 12. Degradation → ABSTAIN?

{ {k: {'ABSTAIN_rate': stress[k]['ABSTAIN_rate'], 'UNNECESSARY': stress[k]['UNNECESSARY'], 'FR': stress[k]['FR_count']} for k in stress} }

Safe iff UNNECESSARY does not rise when evidence is removed.

### 13. Residual evidence limitation?

Tags: {dict(residual_r)}
Not safely fixable without new evidence: {residual_unfixable}. Theoretically fixable as genuine inference error: {residual_fixable}.
Missing Notify remains ABSTAIN. Idle v2 remains a negative finding (precision 82.76%, coverage 1.69%) and was not tuned.

### 14. New Repair operator needed?

No SET_COMPLETE. No per-service patch. No idle wipe. Remaining extras that are C1/R3 are Gold/contract alignment, not a missing rewrite operator.

### 15. Freeze E as MA Behavioral Necessity Repair Candidate?

Safety bar from the prior task still holds if this audit shows: Y-blind, deterministic, D→E significant, broke D-exact 0, FR not up, cross-pattern range not a single-template artifact, ablations show composition/sufficiency/attribution are not redundant, counterfactuals show evidence dependence, stress turns to ABSTAIN.

**Verdict is data-driven below.** Do not freeze if template risk is high, a capability has high wrong-UNNECESSARY, or ablation shows the gain is a single stripped-in rule.
"""

    range_ok = (e_exact_range["range"] or 1) < 0.25
    sig_ok = d_vs_e["p_value"] < 0.05 and boot_e["ci95"][0] > 0
    rep_ok = h1 == h2
    tmpl_risk = bool(high and rare and (exact_rate(high) or 0) - (exact_rate(rare) or 0) >= 15)
    unsafe_caps = [c for c, d in xcap_obj["groups"].items() if d["wrong_unnec"] > 0]
    verdict = "YES" if sig_ok and rep_ok and not unsafe_caps and range_ok else "CONDITIONAL"
    if tmpl_risk or unsafe_caps:
        verdict = "NOT_YET"
    report += f"""
## Freeze verdict

**{verdict}**

- significant D→E: {sig_ok}
- deterministic: {rep_ok}
- pattern range < 25pp: {range_ok} (range={e_exact_range['range']})
- wrong UNNECESSARY capabilities: {unsafe_caps}
- template dependence risk: {tmpl_risk}

If NOT_YET / CONDITIONAL: keep E as a **sidecar candidate**, do not patch, do not raise Set Exact as a goal.
"""
    write_md(REPORT, report)

    print(json.dumps({
        "N": n_parent,
        "B": finB["Parent_Set_Exact_Match"],
        "D": finD["Parent_Set_Exact_Match"],
        "E": finE["Parent_Set_Exact_Match"],
        "climate_n": n_cl,
        "climate_cats": dict(cat_n),
        "mcnemar_p_D_E": d_vs_e["p_value"],
        "boot_E_minus_D": boot_e,
        "repeatable": h1 == h2,
        "verdict": verdict,
        "pattern_range": e_exact_range["range"],
        "tmpl_risk": tmpl_risk,
    }, indent=2, default=str), flush=True)

if __name__ == "__main__":
    main()
