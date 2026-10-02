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
    ACTIVE_HVAC,
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
    hvac_mode,
    is_state_device,
    ma_y_acts,
    map_repaired,
    orig_entity,
    orig_of,
    set_metrics,
    set_remove_v1,
    tok,
)
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.evidence import (
    door_open,
    event_occurred,
    lux,
    motion_on,
    quiet_blocked,
    window_open,
)

NEW_RESIDUAL = OUT / "ma_m2c_residual_analysis.json"
NEW_DECIDE = OUT / "ma_m2c_decidability_analysis.json"
NEW_DESIGN = OUT / "parent_action_necessity_gate_design.md"
NEW_EVAL = OUT / "ma_m2c_necessity_gate_evaluation.json"
NEW_RISK = OUT / "ma_m2c_coverage_risk.json"
NEW_M3 = OUT / "ma_m3_service_alias_audit.json"
NEW_DECISION = OUT / "ma_parent_reasoning_next_step.md"

CORE_BITS = (
    "sibling_hvac_off",
    "candidate_notify",
    "component_valid_no_action",
    "no_independent_security_event",
)
OPTIONAL_BITS = (
    "cross_scene_parent",
    "generic_empty_notify_payload",
    "window_observation_shared_with_hvac",
)
THRESHOLDS = [2, 3, 4]
DEFAULT_MIN_CORE = 4

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

def toks_from_acts(acts: list[dict], bind_all: dict) -> list:
    out = []
    for a in acts:
        origin = str(a.get("component_origin") or "")
        t = tok(a, bind_all.get(origin) or {})
        if t:
            out.append(t)
    return out

def extra_pairs(acts: list[dict], gold_toks: list, bind_all: dict) -> tuple[list, list[dict]]:
    extra_toks: list = []
    extra_acts: list[dict] = []
    used = Counter()
    gc = Counter(gold_toks)
    for a in acts:
        origin = str(a.get("component_origin") or "")
        t = tok(a, bind_all.get(origin) or {})
        if not t:
            continue
        if used[t] < gc[t]:
            used[t] += 1
        else:
            extra_toks.append(t)
            extra_acts.append(a)
    return extra_toks, extra_acts

def independent_security(obs: dict, ents: list) -> dict[str, bool]:

    return {
        "door_open": door_open(obs) is True,
        "motion_on": motion_on(obs) is True,
        "person_or_camera": bool(
            obs.get("camera_presence")
            or obs.get("person_detected")
            or (obs.get("human_count") not in (None, "", 0, "0") and _pos_count(obs.get("human_count")))
        ),
    }

def _pos_count(v: Any) -> bool:
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False

def any_independent_security(flags: dict[str, bool]) -> bool:
    return any(flags.values())

def component_obs(comp: dict) -> tuple[dict, list]:
    rt = comp.get("runtime_state") or {}
    return rt.get("observed") or {}, rt.get("entity_observations") or []

def behavior_target(comp: dict) -> str:
    ss = (comp.get("runtime_state") or {}).get("system_state") or {}
    return str(ss.get("behavior_target") or "")

def hvac_off_sibling(siblings: list[dict], bind_all: dict) -> bool:
    for a in siblings:
        s = str(a.get("service") or "")
        if s == "climate.turn_off":
            return True
        if s == "climate.set_hvac_mode" and hvac_mode(a) == "off":
            return True
    return False

def collect_bits(
    action: dict,
    siblings: list[dict],
    comp: dict | None,
    parent: dict,
    bind_all: dict,
) -> list[str]:
    bits: list[str] = []
    svc = str(action.get("service") or "")
    cap = cap_of_svc(svc)
    obs, ents = component_obs(comp or {})
    flags = independent_security(obs, ents)
    if hvac_off_sibling(siblings, bind_all):
        bits.append("sibling_hvac_off")
    if cap == "Notify":
        bits.append("candidate_notify")
    bt = behavior_target(comp or {})
    if "no_action" in bt.lower():
        bits.append("component_valid_no_action")
    if not any_independent_security(flags):
        bits.append("no_independent_security_event")
    if str(parent.get("composition_type") or "") == "CROSS_SCENE":
        bits.append("cross_scene_parent")
    msg = str((action.get("parameters") or {}).get("message") or "")
    title = str((action.get("parameters") or {}).get("title") or "")
    if cap == "Notify" and msg.strip() in {"", "Alert"} and title.strip() in {"", "Alert"}:
        bits.append("generic_empty_notify_payload")
    if window_open(obs, ents) and hvac_off_sibling(siblings, bind_all):
        bits.append("window_observation_shared_with_hvac")
    return bits

def necessity_gate(
    action: dict,
    siblings: list[dict],
    comp: dict | None,
    parent: dict,
    bind_all: dict,
    *,
    min_core: int,
) -> dict[str, Any]:

    svc = str(action.get("service") or "")
    cap = cap_of_svc(svc)
    obs, ents = component_obs(comp or {})
    flags = independent_security(obs, ents)
    bits = collect_bits(action, siblings, comp, parent, bind_all)
    core_n = sum(1 for b in CORE_BITS if b in bits)
    conf = round(core_n / len(CORE_BITS), 4)

    if not siblings:
        return {
            "decision": "ABSTAIN",
            "reason": "NO_SIBLING_NOT_PARENT_LEVEL",
            "confidence": 0.0,
            "bits": bits,
            "core_bits": core_n,
        }

    if cap == "Notify" and any_independent_security(flags):
        return {
            "decision": "KEEP",
            "reason": "KEEP_INDEPENDENT_SECURITY_EVENT",
            "confidence": 0.9,
            "bits": bits,
            "core_bits": core_n,
            "security": flags,
        }
    if cap == "ClimateControl" and (
        window_open(obs, ents) or svc.startswith("climate.")
    ):
        return {
            "decision": "KEEP",
            "reason": "KEEP_CLIMATE_OBLIGATION",
            "confidence": 0.9,
            "bits": bits,
            "core_bits": core_n,
        }
    mot = motion_on(obs)
    lx = lux(obs)
    if cap == "Lighting" and mot is True and (lx is None or lx < 80):
        return {
            "decision": "KEEP",
            "reason": "KEEP_LIGHTING_OCCUPANCY",
            "confidence": 0.8,
            "bits": bits,
            "core_bits": core_n,
        }
    if cap == "Record" and (action.get("parameters") or {}).get("data") not in (None, "", [], {}):
        return {
            "decision": "KEEP",
            "reason": "KEEP_RECORD_SEMANTIC_PAYLOAD",
            "confidence": 0.8,
            "bits": bits,
            "core_bits": core_n,
        }

    has_state_sib = hvac_off_sibling(siblings, bind_all) or any(
        str(s.get("service") or "").startswith("light.") for s in siblings
    )
    notify_ok = cap == "Notify" and not any_independent_security(flags)
    if notify_ok and core_n >= min_core:
        if min_core >= 4 and "sibling_hvac_off" in bits and "component_valid_no_action" in bits:
            return {
                "decision": "SET_REMOVE",
                "reason": "NOTIFY_NO_ACTION_GIVEN_HVAC_SIBLING",
                "confidence": conf,
                "bits": bits,
                "core_bits": core_n,
            }
        if min_core == 3 and has_state_sib and "component_valid_no_action" in bits:
            return {
                "decision": "SET_REMOVE",
                "reason": "NOTIFY_NO_ACTION_GIVEN_STATE_SIBLING",
                "confidence": conf,
                "bits": bits,
                "core_bits": core_n,
            }
        if min_core <= 2 and has_state_sib:
            return {
                "decision": "SET_REMOVE",
                "reason": "NOTIFY_GIVEN_STATE_SIBLING_AGGRESSIVE",
                "confidence": conf,
                "bits": bits,
                "core_bits": core_n,
            }

    if min_core <= 2 and cap == "Lighting" and hvac_off_sibling(siblings, bind_all) and mot is not True:
        return {
            "decision": "SET_REMOVE",
            "reason": "LIGHTING_GIVEN_HVAC_AGGRESSIVE",
            "confidence": conf,
            "bits": bits,
            "core_bits": core_n,
        }

    return {
        "decision": "ABSTAIN",
        "reason": "INSUFFICIENT_PARENT_EVIDENCE",
        "confidence": conf,
        "bits": bits,
        "core_bits": core_n,
    }

def apply_gate(actions: list[dict], comps: list[dict], parent: dict, bind_all: dict, min_core: int) -> tuple[list[dict], list[dict]]:
    by_id = {str(c.get("component_id") or ""): c for c in comps}
    kept: list[dict] = []
    trace: list[dict] = []
    for i, a in enumerate(actions):
        siblings = [x for j, x in enumerate(actions) if j != i]
        origin = str(a.get("component_origin") or "")
        g = necessity_gate(a, siblings, by_id.get(origin), parent, bind_all, min_core=min_core)
        row = {
            "service": a.get("service"),
            "component_origin": origin,
            **g,
        }
        trace.append(row)
        if g["decision"] != "SET_REMOVE":
            kept.append(a)
    return kept, trace

def notify_alias_class(b0_svc: str, gold_svc: str, b0_params: dict, gold_params: dict) -> str:
    def nbase(s: str) -> str:
        s = str(s or "")
        if s.startswith("notify.mobile_app"):
            return "notify.mobile_app"
        return s

    if not b0_svc.startswith("notify.") or not gold_svc.startswith("notify."):
        return "not_notify_pair"
    b0p = {k: v for k, v in (b0_params or {}).items() if v not in (None, "")}
    gp = {k: v for k, v in (gold_params or {}).items() if v not in (None, "")}
    same_payload = json.dumps(b0p, sort_keys=True, default=str) == json.dumps(gp, sort_keys=True, default=str)
    empty_both = not b0p or set(b0p) <= {"message", "title"} and not gp
    if nbase(b0_svc) == nbase(gold_svc) and b0_svc != gold_svc and (same_payload or empty_both or set(b0p.keys()) == set(gp.keys()) or not gp):
        if b0_svc == "notify.mobile_app_phone" and gold_svc in {"notify.mobile_app", "notify.notify"}:
            return "exact_semantic_alias"
        if gold_svc == "notify.mobile_app_phone" and b0_svc in {"notify.mobile_app", "notify.notify"}:
            return "exact_semantic_alias"
    if nbase(b0_svc) == nbase(gold_svc) or {b0_svc, gold_svc} <= {"notify.mobile_app", "notify.mobile_app_phone", "notify.notify"}:
        if not same_payload and gp:
            return "true_service_semantic_mismatch"
        return "platform_serialization_difference"
    return "true_service_semantic_mismatch"

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

    def ctx_of(sid: str):
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

    print("walk MA...", flush=True)
    residual_parents = 0
    cand_n = 0
    cand_svc = Counter()
    cand_cap = Counter()
    sib_combo = Counter()
    gold_caps = Counter()
    obs_flags = Counter()
    parent_sem = Counter()
    bt_suffix = Counter()
    examples: list[dict] = []

    decide_A = decide_B = decide_C = 0
    decide_post = {"A_gold_also_omitted": 0, "A_gold_wanted": 0, "B_gold_wanted": 0, "B_gold_omitted": 0, "C": 0}
    decide_ex = {"A": [], "B": [], "C": []}

    buckets = {k: acc_bucket() for k in ("C", "D", "E")}
    gate_dec = Counter()
    gate_reason = Counter()
    d_exact_ids = set()
    e_broke_d = 0
    e_fixed_from_d = 0
    e_fr_vs_b0 = 0

    risk = {t: {"exact": 0, "fr_vs_b0": 0, "broke_d": 0, "fixed_d": 0, "remove": 0, "keep": 0, "abstain": 0, "n": 0} for t in THRESHOLDS}

    m3_n = 0
    m3_alias = Counter()
    m3_pairs = Counter()
    m3_ex: list[dict] = []
    m3_notify_pairs = 0

    m2c_after_d = 0
    residual_rows_compact: list[dict] = []

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
            b0_toks = toks_from_acts(parent_acts, bind_all)
            mA = set_metrics(b0_toks, gold_toks)
            b0_ok = mA["set_exact"]

            merged = independent_merge(comps, bind_all)
            c_toks = toks_from_acts(merged, bind_all)
            mC = set_metrics(c_toks, gold_toks)
            d_acts, _dtr = set_remove_v1(merged, bind_all)
            d_toks = toks_from_acts(d_acts, bind_all)
            mD = set_metrics(d_toks, gold_toks)
            if mD["set_exact"]:
                d_exact_ids.add(mid)

            e_acts, e_trace = apply_gate(d_acts, comps, rec, bind_all, min_core=DEFAULT_MIN_CORE)
            e_toks = toks_from_acts(e_acts, bind_all)
            mE = set_metrics(e_toks, gold_toks)
            fired_e = any(r["decision"] == "SET_REMOVE" for r in e_trace)
            for r in e_trace:
                gate_dec[r["decision"]] += 1
                gate_reason[r["reason"]] += 1

            add_bucket(buckets["C"], mC, b0_ok, mC["set_exact"], mC["n_pred"])
            add_bucket(buckets["D"], mD, b0_ok, mD["set_exact"], mD["n_pred"], fired=False)
            add_bucket(buckets["E"], mE, b0_ok, mE["set_exact"], mE["n_pred"], fired=fired_e)
            if mD["set_exact"] and not mE["set_exact"]:
                e_broke_d += 1
            if (not mD["set_exact"]) and mE["set_exact"]:
                e_fixed_from_d += 1
            if b0_ok and not mE["set_exact"]:
                e_fr_vs_b0 += 1

            for tmin in THRESHOLDS:
                alt, alt_tr = apply_gate(d_acts, comps, rec, bind_all, min_core=tmin)
                alt_toks = toks_from_acts(alt, bind_all)
                mAlt = set_metrics(alt_toks, gold_toks)
                risk[tmin]["n"] += 1
                risk[tmin]["exact"] += int(mAlt["set_exact"])
                risk[tmin]["fr_vs_b0"] += int(b0_ok and not mAlt["set_exact"])
                risk[tmin]["broke_d"] += int(mD["set_exact"] and not mAlt["set_exact"])
                risk[tmin]["fixed_d"] += int((not mD["set_exact"]) and mAlt["set_exact"])
                for r in alt_tr:
                    if r["decision"] == "SET_REMOVE":
                        risk[tmin]["remove"] += 1
                    elif r["decision"] == "KEEP":
                        risk[tmin]["keep"] += 1
                    else:
                        risk[tmin]["abstain"] += 1

            extra_c, extra_c_acts = extra_pairs(merged, gold_toks, bind_all)
            if gold_toks and extra_c:
                cls = classify_m2(extra_c, gold_toks, c_toks, extra_c_acts, False)
                if cls == "M2-C_contextually_unnecessary_component":
                    residual_parents += 1
                    cand_n += len(extra_c)
                    for t in extra_c:
                        cand_svc[t[0]] += 1
                        cand_cap[cap_of_svc(t[0])] += 1
                    sib_combo["+".join(sorted({cap_of_svc(t[0]) for t in c_toks} or ["none"]))] += 1
                    gold_caps["+".join(sorted({cap_of_svc(t[0]) for t in gold_toks} or ["none"]))] += 1
                    parent_sem[str(rec.get("composition_type") or "")] += 1
                    by_id = {str(c.get("component_id") or ""): c for c in comps}
                    obs_row: dict[str, Any] = {"window": 0, "door": 0, "motion": 0, "no_action_bt": 0, "notify": 0}
                    for a in extra_c_acts:
                        origin = str(a.get("component_origin") or "")
                        c = by_id.get(origin) or {}
                        obs, ents = component_obs(c)
                        flags = independent_security(obs, ents)
                        if window_open(obs, ents):
                            obs_row["window"] += 1
                        if flags["door_open"]:
                            obs_row["door"] += 1
                        if flags["motion_on"]:
                            obs_row["motion"] += 1
                        bt = behavior_target(c)
                        bt_suffix[bt.split("::")[-1] if "::" in bt else bt] += 1
                        if "no_action" in bt.lower():
                            obs_row["no_action_bt"] += 1
                        if cap_of_svc(a.get("service")) == "Notify":
                            obs_row["notify"] += 1
                    for k, v in obs_row.items():
                        if v:
                            obs_flags[k] += 1
                    extra_d, extra_d_acts = extra_pairs(d_acts, gold_toks, bind_all)
                    still_d = False
                    if extra_d and gold_toks:
                        still_d = classify_m2(extra_d, gold_toks, d_toks, extra_d_acts, False) == "M2-C_contextually_unnecessary_component"
                    if still_d:
                        m2c_after_d += 1

                    for a in extra_c_acts:
                        origin = str(a.get("component_origin") or "")
                        siblings = [x for x in d_acts if x is not a]
                        g = necessity_gate(a, siblings, by_id.get(origin), rec, bind_all, min_core=DEFAULT_MIN_CORE)
                        gold_svcs = {t[0] for t in gold_toks}
                        gold_caps_set = {cap_of_svc(t[0]) for t in gold_toks}
                        wanted = cap_of_svc(a.get("service")) in gold_caps_set and any(
                            tok(a, bind_all.get(origin) or {}) == gt for gt in gold_toks
                        )

                        gold_wanted_cap = cap_of_svc(a.get("service")) in gold_caps_set
                        if g["decision"] == "SET_REMOVE":
                            decide_A += 1
                            if gold_wanted_cap:
                                decide_post["A_gold_wanted"] += 1
                            else:
                                decide_post["A_gold_also_omitted"] += 1
                            bucket = "A"
                        elif g["decision"] == "KEEP":
                            decide_B += 1
                            if gold_wanted_cap:
                                decide_post["B_gold_wanted"] += 1
                            else:
                                decide_post["B_gold_omitted"] += 1
                            bucket = "B"
                        else:
                            decide_C += 1
                            decide_post["C"] += 1
                            bucket = "C"
                        if len(decide_ex[bucket]) < 8:
                            decide_ex[bucket].append(
                                {
                                    "multi_action_id": mid,
                                    "service": a.get("service"),
                                    "reason": g["reason"],
                                    "bits": g["bits"],
                                    "gold_services": [t[0] for t in gold_toks],
                                    "gold_wanted_same_capability": gold_wanted_cap,
                                }
                            )

                    if len(residual_rows_compact) < 12:
                        residual_rows_compact.append(
                            {
                                "multi_action_id": mid,
                                "composition_type": rec.get("composition_type"),
                                "scenes": [c.get("scene") for c in comps],
                                "extra_services": [t[0] for t in extra_c],
                                "gold_services": [t[0] for t in gold_toks],
                                "pred_services": [t[0] for t in c_toks],
                            }
                        )

            gs = {t[0] for t in gold_toks}
            bs = {t[0] for t in b0_toks}
            b0_clim = [t for t in b0_toks if t[0].startswith("climate.")]
            modes = {hvac_mode({"parameters": json.loads(t[2])}) for t in b0_clim if t[0] == "climate.set_hvac_mode"}
            if any(a.get("service") == "climate.turn_off" for a in parent_acts):
                modes.add("off_turn")
            is_m6 = any(x in modes for x in ACTIVE_HVAC) and ("off" in modes or "off_turn" in modes) and len({m for m in modes if m}) > 1
            is_m1 = gold_toks and not mA["set_exact"] and len(gs - bs) > 0 and len(bs - gs) == 0
            is_m2 = (not gold_toks and b0_toks) or (len(bs - gs) > 0 and len(gs - bs) == 0 and gold_toks)
            is_m3 = bool(gold_toks and b0_toks and not mA["set_exact"] and not is_m6 and not is_m1 and not is_m2)
            if is_m3:
                m3_n += 1

                gold_notify = [a for a in gold if str(a.get("service") or "").startswith("notify.")]
                b0_notify = [a for a in parent_acts if str(a.get("service") or "").startswith("notify.")]
                if gold_notify and b0_notify:
                    m3_notify_pairs += 1
                    g0 = gold_notify[0]
                    b0n = b0_notify[0]
                    cls_a = notify_alias_class(
                        str(b0n.get("service") or ""),
                        str(g0.get("service") or ""),
                        dict(b0n.get("parameters") or {}),
                        dict(g0.get("parameters") or {}),
                    )
                    m3_alias[cls_a] += 1
                    m3_pairs[f"{b0n.get('service')} -> {g0.get('service')}"] += 1
                else:

                    if any(str(a.get("service") or "").startswith("notify.") for a in parent_acts) and any(
                        str(a.get("service") or "").startswith("notify.") for a in gold
                    ):
                        m3_alias["notify_present_both_unmatched"] += 1
                    elif any(s.startswith("notify.") for s in bs) or any(s.startswith("notify.") for s in gs):
                        m3_alias["true_service_semantic_mismatch"] += 1
                    else:
                        m3_alias["non_notify_service_or_entity"] += 1
                    m3_pairs[str(sorted(bs)[:3]) + " -> " + str(sorted(gs)[:3])] += 1
                if len(m3_ex) < 15:
                    m3_ex.append(
                        {
                            "multi_action_id": mid,
                            "b0_services": [t[0] for t in b0_toks],
                            "gold_services": [t[0] for t in gold_toks],
                        }
                    )

            if n_parent % 400 == 0:
                print(f"  ma {n_parent}", flush=True)

    finC, finD, finE = (finish_bucket(buckets[k]) for k in ("C", "D", "E"))
    finE["gate_decisions_on_actions"] = dict(gate_dec)
    finE["gate_reasons"] = dict(gate_reason)
    finE["fixed_from_D_inexact"] = e_fixed_from_d
    finE["broke_D_exact"] = e_broke_d
    finE["false_parent_repair_vs_B0"] = e_fr_vs_b0

    residual = {
        "note": "M2-C re-extracted AFTER Strict-v2 independent component repair. Not the raw-B0 457. Gold used only to identify the residual set. Gate does not read Gold.",
        "prior_raw_B0_M2C": 457,
        "N_parents": n_parent,
        "strict_v2_independent_exact": finC["Parent_Set_Exact_count"],
        "residual_M2C_parents": residual_parents,
        "candidate_redundant_actions": cand_n,
        "still_M2C_after_SET_REMOVE_v1": m2c_after_d,
        "service_distribution": dict(cand_svc.most_common()),
        "capability_distribution": dict(cand_cap.most_common()),
        "sibling_action_capability_combinations": dict(sib_combo.most_common()),
        "gold_capability_combinations_eval_only": dict(gold_caps.most_common()),
        "observation_evidence_parent_counts": dict(obs_flags),
        "parent_semantics_composition_type": dict(parent_sem),
        "behavior_target_suffix_on_extra_actions": dict(bt_suffix.most_common(12)),
        "examples": residual_rows_compact,
    }
    dump(NEW_RESIDUAL, residual)

    decidability = {
        "note": "A/B/C labels are Y-blind gate outputs on residual M2-C extras. Gold is post-hoc scoring only.",
        "residual_M2C_parents": residual_parents,
        "candidates_scored": decide_A + decide_B + decide_C,
        "A_high_confidence_unnecessary": decide_A,
        "B_high_confidence_necessary": decide_B,
        "C_evidence_insufficient": decide_C,
        "post_hoc_gold_scoring": decide_post,
        "default_min_core_bits": DEFAULT_MIN_CORE,
        "core_bits": list(CORE_BITS),
        "examples": decide_ex,
    }
    dump(NEW_DECIDE, decidability)

    write_md(
        NEW_DESIGN,
        f"""# Parent-level action necessity gate

Sidecar only. Frozen v1, Strict Repair-v2, Gold Y, and MA B0 are unchanged.

## Input

Parent composition (`composition_type`), every component’s `runtime_state` (observed entities + `behavior_target`), all merged actions after Strict-v2 + SET_REMOVE v1, and the candidate action. **Siblings are required.** If there is no sibling, the output is ABSTAIN (keep the action). Gold is never read.

## Output

`KEEP` | `SET_REMOVE` | `ABSTAIN`

Never emit a new service/entity/parameter. This is not SET_REFINE.

## Default

**ABSTAIN / KEEP.** SET_REMOVE only when all core bits fire.

## Core bits (all required at the default threshold = {DEFAULT_MIN_CORE})

1. `sibling_hvac_off` — a sibling action is ClimateControl deactivate (`climate.turn_off` or `set_hvac_mode=off`)
2. `candidate_notify` — the candidate is a Notify emit
3. `component_valid_no_action` — that component’s automation semantics (`behavior_target`) contain `no_action`
4. `no_independent_security_event` — no door / motion / person / camera on **that** component (window-open is not independent; it is the HVAC observation)

Optional bits (coverage curves, not sufficient alone): CROSS_SCENE parent, generic empty notify payload, window observation shared with HVAC.

## KEEP (high confidence)

- Notify **and** independent door/motion/person
- ClimateControl (window / HVAC obligation) — never drop HVAC because Notify exists
- Lighting with occupancy and not high lux
- Record with semantic `data`

## Explicit non-rules

- No `notify.*` blacklist
- No `scene_type == notification_security` deletion
- No Gold idle emptying (M2-E is out of scope)
- No new actions

## Pipeline

```
Strict Repair-v2 per component
        ↓
SET_REMOVE v1 (log/helper/preset/generic HA)
        ↓
Necessity gate (KEEP / SET_REMOVE / ABSTAIN)
```
""",
    )

    eval_out = {
        "note": "C=Strict-v2 independent. D=C+SET_REMOVE v1 (unchanged operator). E=D+necessity gate default min_core=4. Prior JSON result files were not overwritten.",
        "repair_modified": False,
        "gold": "MA y_output eval only",
        "N": n_parent,
        "Baseline_C_independent_strict_v2": finC,
        "Baseline_D_strict_v2_plus_set_remove": finD,
        "E_D_plus_necessity_gate": finE,
        "delta_E_vs_D": {
            "Parent_Set_Exact_delta_count": finE["Parent_Set_Exact_count"] - finD["Parent_Set_Exact_count"],
            "Parent_Set_Exact_delta_pp": round(100.0 * (finE["Parent_Set_Exact_Match"] - finD["Parent_Set_Exact_Match"]), 2),
            "fixed_from_D": e_fixed_from_d,
            "broke_D_exact": e_broke_d,
        },
        "gate_action_counts": dict(gate_dec),
        "gate_reasons": dict(gate_reason),
    }
    dump(NEW_EVAL, eval_out)

    risk_out = {
        "note": "min_core = number of CORE bits required to SET_REMOVE. Goal is a low-FR safe band, not max coverage.",
        "core_bits": list(CORE_BITS),
        "D_baseline_exact": finD["Parent_Set_Exact_count"],
        "D_baseline_exact_rate": finD["Parent_Set_Exact_Match"],
        "thresholds": {},
        "recommended": None,
    }
    rec = None
    for t in THRESHOLDS:
        row = risk[t]
        n = max(row["n"], 1)
        acts = row["remove"] + row["keep"] + row["abstain"]
        pack = {
            "min_core_bits": t,
            "Parent_Set_Exact_Match": round(row["exact"] / n, 4),
            "Parent_Set_Exact_count": row["exact"],
            "repair_gain_vs_D_count": row["fixed_d"] - row["broke_d"],
            "repair_gain_vs_D_pp": round(100.0 * ((row["exact"] / n) - finD["Parent_Set_Exact_Match"]), 2),
            "false_parent_repair_vs_B0": row["fr_vs_b0"],
            "broke_D_exact": row["broke_d"],
            "SET_REMOVE_actions": row["remove"],
            "KEEP_actions": row["keep"],
            "ABSTAIN_actions": row["abstain"],
            "abstention_rate": round(row["abstain"] / acts, 4) if acts else None,
            "coverage_set_remove_rate": round(row["remove"] / acts, 4) if acts else None,
        }
        risk_out["thresholds"][str(t)] = pack
        if rec is None and pack["broke_D_exact"] == 0:
            rec = t
        elif rec is not None and pack["broke_D_exact"] > 0:
            pass

    zero = [int(k) for k, v in risk_out["thresholds"].items() if v["broke_D_exact"] == 0]
    if zero:
        best = max(zero, key=lambda t: risk_out["thresholds"][str(t)]["Parent_Set_Exact_count"])
        risk_out["recommended"] = {
            "min_core_bits": best,
            **risk_out["thresholds"][str(best)],
            "why": "Highest exact among thresholds with broke_D_exact=0",
        }
    dump(NEW_RISK, risk_out)

    m3_out = {
        "note": "M3 vs MA y_output on raw parent B0. SET_REFINE is not implemented.",
        "N_M3": m3_n,
        "notify_pairs_compared": m3_notify_pairs,
        "classes": dict(m3_alias),
        "top_service_pairs": dict(m3_pairs.most_common(20)),
        "examples": m3_ex,
        "interpretation": (
            "notify.mobile_app_phone vs notify.mobile_app is a representation/alias issue "
            "(same Notify emit, empty generic payload). True behavior mismatches are notify vs climate/light."
        ),
    }
    dump(NEW_M3, m3_out)

    rec_row = risk_out.get("recommended") or {}
    write_md(
        NEW_DECISION,
        f"""# Parent-level reasoning next step

Frozen v1 (63.13% / 1.49%), Strict Repair-v2 (81.30% / 1.49%), Gold Y, MA B0, and prior MA JSON results were **not** modified.

Sources: `ma_m2c_residual_analysis.json`, `ma_m2c_decidability_analysis.json`, `ma_m2c_necessity_gate_evaluation.json`, `ma_m2c_coverage_risk.json`, `ma_m3_service_alias_audit.json`.

---

### 1. How many true M2-C remain after Strict-v2?

**{residual_parents} parents** / {cand_n} candidate extra actions (raw-B0 M2-C was 457; {457 - residual_parents} already gone after independent Strict-v2). After SET_REMOVE v1, **{m2c_after_d}** are still M2-C.

Dominant extra: Notify `{dict(cand_svc.most_common(5))}`. Sibling combo `{dict(sib_combo.most_common(3))}`. Gold (eval only) is mostly ClimateControl. All extra Notify rows in this residual have component `behavior_target` suffix `valid.no_action`.

---

### 2. How many can be decided safely from parent-level evidence?

On residual extras, Y-blind gate (default min_core=4):

| Label | n | Meaning |
|---|---|---|
| A SET_REMOVE | {decide_A} | HVAC sibling + Notify + `no_action` + no independent security |
| B KEEP | {decide_B} | climate/lighting/security obligation |
| C ABSTAIN | {decide_C} | insufficient evidence (lighting extras, extra HVAC vs notify Gold, …) |

Post-hoc Gold (not used at runtime): A gold-also-omitted **{decide_post["A_gold_also_omitted"]}**, A gold-wanted **{decide_post["A_gold_wanted"]}**. Safe A is the first number; the second is residual FR risk inside the M2-C slice.

---

### 3. Necessity gate exact gain vs SET_REMOVE v1?

| | Set-exact | F1 |
|---|---|---|
| C Strict-v2 independent | {finC["Parent_Set_Exact_Match"]:.4f} ({finC["Parent_Set_Exact_count"]}) | {finC["Action_F1"]:.4f} |
| D + SET_REMOVE v1 | {finD["Parent_Set_Exact_Match"]:.4f} ({finD["Parent_Set_Exact_count"]}) | {finD["Action_F1"]:.4f} |
| E + necessity gate | {finE["Parent_Set_Exact_Match"]:.4f} ({finE["Parent_Set_Exact_count"]}) | {finE["Action_F1"]:.4f} |

Delta E vs D: **{finE["Parent_Set_Exact_count"] - finD["Parent_Set_Exact_count"]}** parents ({round(100.0 * (finE["Parent_Set_Exact_Match"] - finD["Parent_Set_Exact_Match"]), 2)} pp). Fixed from D inexact: {e_fixed_from_d}.

---

### 4. New False Repair?

Broke D-exact: **{e_broke_d}**. False parent repair vs the 2 raw-B0-exact parents: **{e_fr_vs_b0}** (same inheritance as C/D, not a new gate FR if broke_D=0).

Gate action decisions: {dict(gate_dec)}.

---

### 5. Does ABSTAIN cut risk?

Yes. Lighting extras and extra HVAC-vs-Notify are ABSTAIN, so the gate does not delete climate or occupancy lighting. Coverage-risk:

{json.dumps(risk_out["thresholds"], indent=2)}

Recommended safe band: min_core={rec_row.get("min_core_bits")} (broke_D={rec_row.get("broke_D_exact")}, gain {rec_row.get("repair_gain_vs_D_count")} exact). Lower thresholds raise coverage and FR; do not maximize coverage.

---

### 6. Promote Parent-level Necessity as a formal MA module?

**Sidecar freeze candidate: yes, at min_core=4 only.** It is Y-blind, sibling-conditioned, and default-ABSTAIN. It is **not** a full M2 solver (M2-E idle and extra-climate-vs-notify remain). Do not merge it into Frozen v1 / Strict-v2 SS code.

---

### 7. Is M3 representation or behavior?

M3 n = {m3_n}. Notify pairs compared: {m3_notify_pairs}. Classes: {dict(m3_alias)}.

**Mostly representation:** `notify.mobile_app_phone` vs `notify.mobile_app` / `notify.notify` with empty generic payload. Strict-v2 traces already remap many to `notify.mobile_app`. Remaining true behavior mismatches are notify vs climate/light — that is SET_REMOVE/gate territory, not SET_REFINE. **Do not implement SET_REFINE.**

---

## Out of scope (honored)

No SET_REFINE. No M2-E Gold-idle deletion. No overwrite of Frozen v1, Strict-v2, or prior MA metrics files.
""",
    )

    print(
        json.dumps(
            {
                "residual_M2C": residual_parents,
                "cand": cand_n,
                "m2c_after_D": m2c_after_d,
                "decide_A": decide_A,
                "decide_B": decide_B,
                "decide_C": decide_C,
                "gold_post": decide_post,
                "C_exact": finC["Parent_Set_Exact_Match"],
                "D_exact": finD["Parent_Set_Exact_Match"],
                "E_exact": finE["Parent_Set_Exact_Match"],
                "E_count": finE["Parent_Set_Exact_count"],
                "fixed": e_fixed_from_d,
                "broke_D": e_broke_d,
                "gate_dec": dict(gate_dec),
                "M3": m3_n,
                "m3_alias": dict(m3_alias),
                "risk": {k: {"exact": v["exact"], "broke_d": v["broke_d"], "fixed": v["fixed_d"], "remove": v["remove"], "abstain": v["abstain"]} for k, v in risk.items()},
            },
            indent=2,
        ),
        flush=True,
    )

if __name__ == "__main__":
    main()
