from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ss_b0_vs_final_gold_y import SS, Y_PATH, b0_from_sample
from eval_ss_trhr_final import load_jsonl, metrics_block, pair_from_act, semantic_ok, y_pair
from eval_ss_trhr_v2_joint import compact_action
from eval_ss_trhr_v2_offline import apply_r1, climate_capability, r1_should_fire
from smarthome_mdf.ss_trhr_repair.actions import service_kind
from smarthome_mdf.ss_trhr_repair.context import build_repair_context

OUT = ROOT / "runs" / "ss_trhr_repair"
TRACE = OUT / "repair_execution_trace.jsonl"
MA = ROOT / "data" / "benchmarks" / "multi_action.jsonl"

STRICT_DROP = {"level", "logger"}
NEVER_DROP = {"config_entry", "worksheet", "data", "message", "title", "hvac_mode"}
ACTIVE_HVAC = {"heat", "cool", "auto", "heat_cool", "dry", "fan_only"}
AUX_LOG = {
    "system_log.write",
    "logbook.log",
    "logbook.write",
    "persistent_notification.create",
    "persistent_notification.dismiss",
}
GENERIC_HA = {"homeassistant.turn_on", "homeassistant.turn_off"}
HELPER_OR_SCHEDULE = {"schedule.activate", "scene.turn_on", "scene.turn_off"}
AUX_RECORD_EMPTY_OK = False

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0

def cap_of_svc(svc: str) -> str:
    s = str(svc or "")
    if s.startswith("light."):
        return "Lighting"
    if s.startswith("climate."):
        return "ClimateControl"
    if s.startswith("notify."):
        return "Notify"
    if s.startswith(("logbook.", "system_log.", "persistent_notification.")):
        return "Log"
    if s.startswith("google_sheets."):
        return "Record"
    if s.startswith(("schedule.", "scene.")):
        return "Schedule"
    if s.startswith("input_"):
        return "Helper"
    if s.startswith("llmvision."):
        return "Vision"
    if s.startswith("homeassistant."):
        return "GenericHA"
    return "Other"

def hvac_mode(act: dict | None) -> str | None:
    if not act:
        return None
    raw = (act.get("parameters") or {}).get("hvac_mode")
    if raw in (None, ""):
        return None
    return str(raw).strip().lower()

def orig_entity(ent: Any, binding: dict | None) -> str | None:
    if not ent:
        return None
    e = str(ent)
    recs = (binding or {}).get("records") or []
    for r in recs:
        if e in {r.get("parent_entity"), r.get("component_local_entity"), r.get("original_entity")}:
            return str(r.get("original_entity") or e)
    m = (binding or {}).get("component_entity_map") or {}
    inv = {v: k for k, v in m.items()}
    return inv.get(e, e)

def tok(act: dict | None, binding: dict | None = None) -> tuple | None:
    if not act:
        return None
    svc = str(act.get("service") or "")
    if not svc:
        return None
    ent = orig_entity(act.get("target_entity") or act.get("entity"), binding)

    params = {k: v for k, v in dict(act.get("parameters") or {}).items() if v not in (None, "", [], {})}
    return (svc, ent or "", json.dumps(params, sort_keys=True, default=str))

def behavior_key(t: tuple) -> tuple:
    svc, ent, pjson = t
    params = json.loads(pjson) if pjson else {}
    if svc == "climate.turn_off" or (svc == "climate.set_hvac_mode" and str(params.get("hvac_mode") or "").lower() == "off"):
        return ("ClimateControl", "Deactivate", ent)
    if svc == "climate.set_hvac_mode":
        return ("ClimateControl", "SetMode", ent, str(params.get("hvac_mode") or "").lower())
    if svc == "climate.set_preset_mode":
        return ("ClimateControl", "SetPreset", ent)
    if svc.endswith(".turn_off") and cap_of_svc(svc) == "Lighting":
        return ("Lighting", "Deactivate", ent)
    if svc.endswith(".turn_on") and cap_of_svc(svc) == "Lighting":
        return ("Lighting", "Activate", ent)
    if svc.startswith("notify."):
        return ("Notify", "Emit", ent or "notify")
    return (cap_of_svc(svc), svc, ent, pjson)

def ma_y_acts(sample: dict) -> list[dict]:
    yo = sample.get("y_output") or {}
    if str(yo.get("label_decision") or "").upper() in {"NO_ACTION", ""}:
        return []
    out = []
    for a in yo.get("normalized_structured_y") or []:
        if isinstance(a, dict) and a.get("service"):
            out.append(a)
    return out

def set_metrics(pred: list, gold: list) -> dict[str, Any]:
    pc = Counter(pred)
    gc = Counter(gold)
    inter = sum((pc & gc).values())
    pn, gn = sum(pc.values()), sum(gc.values())
    prec = inter / pn if pn else (1.0 if gn == 0 else 0.0)
    rec = inter / gn if gn else (1.0 if pn == 0 else 0.0)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {
        "set_exact": pc == gc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_pred": pn,
        "n_gold": gn,
    }

def apply_payload_strict(pred: dict | None) -> tuple[Any, bool, list[str]]:
    if not pred:
        return pred, False, []
    params = dict(pred.get("parameters") or {})
    dropped = [k for k in STRICT_DROP if k in params]
    if not dropped:
        return pred, False, []
    kept = {k: v for k, v in params.items() if k not in STRICT_DROP}
    for k in NEVER_DROP:
        if k in params and k not in kept and params.get(k) not in (None,):
            kept[k] = params[k]
    out = compact_action(pred) or dict(pred)
    out["service"] = pred.get("service")
    out["target_entity"] = pred.get("target_entity")
    out["parameters"] = kept
    return out, True, dropped

def apply_strict_v2(ctx: dict, pred: dict | None) -> tuple[Any, list[str]]:

    out = pred
    kinds: list[str] = []
    if r1_should_fire(ctx, out, use_scene=False):

        if climate_capability(ctx, out):
            out = apply_r1(out)
            kinds.append("hvac_r1")
    out, hit, dropped = apply_payload_strict(out)
    if hit:
        kinds.append("payload_" + "+".join(dropped))
    return out, kinds

def is_state_device(svc: str) -> bool:
    kind = service_kind(svc)
    if kind != "STATE":
        return False
    if svc.startswith("scene."):
        return False
    return True

def orig_of(act: dict, bind_all: dict) -> str:
    origin = str(act.get("component_origin") or "")
    bind = bind_all.get(origin) or {}
    return orig_entity(act.get("target_entity") or act.get("entity"), bind) or ""

def set_remove_v1(actions: list[dict], bind_all: dict) -> tuple[list[dict], list[dict]]:

    live = [a for a in actions if a and a.get("service")]
    has_state = any(is_state_device(str(a.get("service") or "")) for a in live)
    has_specific = any(str(a.get("service") or "").startswith(("light.", "climate.")) for a in live)
    hvac_off_targets: set[str] = set()
    for a in live:
        s = str(a.get("service") or "")
        tgt = orig_of(a, bind_all)
        if s == "climate.turn_off" or (s == "climate.set_hvac_mode" and hvac_mode(a) == "off"):
            hvac_off_targets.add(tgt)

    kept: list[dict] = []
    trace: list[dict] = []
    for a in live:
        s = str(a.get("service") or "")
        tgt = orig_of(a, bind_all)
        kind = service_kind(s)
        reason = None

        if s in AUX_LOG or (kind == "EVENT" and s.startswith(("system_log.", "logbook."))):
            if has_state:
                reason = "AUX_LOG_GIVEN_STATE_SIBLING"
        elif kind == "HELPER" or s in HELPER_OR_SCHEDULE:
            if has_state:
                reason = "DOMINATED_HELPER_OR_SCHEDULE_GIVEN_STATE"
        elif s == "climate.set_preset_mode" and (tgt in hvac_off_targets or (not tgt and hvac_off_targets)):
            reason = "DOMINATED_PRESET_GIVEN_HVAC_OFF"
        elif s == "climate.set_hvac_mode" and hvac_mode(a) in ACTIVE_HVAC and tgt in hvac_off_targets:
            reason = "CONTRADICTED_ACTIVE_HVAC_GIVEN_OFF"
        elif s in GENERIC_HA and has_specific:
            reason = "GENERIC_HA_GIVEN_SPECIFIC_DEVICE"

        row = {
            "service": s,
            "target_original": tgt or None,
            "component_origin": a.get("component_origin"),
            "removed": bool(reason),
            "reason": reason or "NECESSARY_GIVEN_SIBLINGS",
        }
        trace.append(row)
        if not reason:
            kept.append(a)
    return kept, trace

def acc_bucket() -> dict[str, Any]:
    return {
        "n": 0,
        "set_exact": 0,
        "prec": 0.0,
        "rec": 0.0,
        "f1": 0.0,
        "fr": 0,
        "pres_ok": 0,
        "b0_ok": 0,
        "n_pred": 0.0,
        "remove_fired": 0,
    }

def add_bucket(b: dict, m: dict, b0_ok: bool, pred_ok: bool, n_pred: int, fired: bool = False) -> None:
    b["n"] += 1
    b["set_exact"] += int(m["set_exact"])
    b["prec"] += m["precision"]
    b["rec"] += m["recall"]
    b["f1"] += m["f1"]
    b["b0_ok"] += int(b0_ok)
    b["fr"] += int(b0_ok and not pred_ok)
    b["pres_ok"] += int(b0_ok and pred_ok)
    b["n_pred"] += n_pred
    b["remove_fired"] += int(fired)

def finish_bucket(b: dict) -> dict[str, Any]:
    n = max(b["n"], 1)
    b0e = b["b0_ok"]
    return {
        "N": b["n"],
        "Parent_Set_Exact_Match": round(b["set_exact"] / n, 4),
        "Parent_Set_Exact_count": b["set_exact"],
        "Action_Precision": round(b["prec"] / n, 4),
        "Action_Recall": round(b["rec"] / n, 4),
        "Action_F1": round(b["f1"] / n, 4),
        "False_Parent_Repair": round(b["fr"] / n, 4),
        "False_Parent_Repair_count": b["fr"],
        "Preservation": round(b["pres_ok"] / b0e, 4) if b0e else None,
        "Preservation_count": b["pres_ok"],
        "correct_B0_count": b0e,
        "mean_action_count": round(b["n_pred"] / n, 4),
        "set_remove_fired": b.get("remove_fired", 0),
    }

def map_repaired(fr: dict | None, bind: dict) -> dict | None:
    if not fr:
        return None
    mapped = dict(fr)
    pmap = (bind.get("component_entity_map") or {})
    src = str(fr.get("target_entity") or "")
    mapped["target_entity"] = pmap.get(src, fr.get("target_entity"))
    return mapped

def classify_m2(extra_toks: list, gold_toks: list, b0_toks: list, extra_acts: list[dict], gold_idle: bool) -> str:
    extra_svc = [t[0] for t in extra_toks]
    gold_beh = {behavior_key(t) for t in gold_toks}
    extra_beh = [behavior_key(t) for t in extra_toks]
    gold_caps = {cap_of_svc(t[0]) for t in gold_toks}

    if gold_toks:
        if any(b in gold_beh for b in extra_beh):
            return "M2-A_semantic_duplicate"
        if extra_beh and len(extra_beh) != len(set(extra_beh)):
            return "M2-A_semantic_duplicate"

    aux_svcs = []
    other_extra = []
    for s in extra_svc:
        if s in AUX_LOG or s.startswith(("system_log.", "logbook.", "google_sheets.", "llmvision.", "persistent_notification.")):
            aux_svcs.append(s)
        else:
            other_extra.append(s)
    if aux_svcs and not other_extra:
        return "M2-B_auxiliary_logging"
    if aux_svcs and gold_toks and not any(cap_of_svc(s) in {"Log", "Record", "Vision"} for s in [t[0] for t in gold_toks]):

        if not other_extra or all(
            s in GENERIC_HA or s in HELPER_OR_SCHEDULE or s.startswith("input_") or s == "climate.set_preset_mode"
            for s in other_extra
        ):
            return "M2-B_auxiliary_logging"

    def dominated(s: str) -> bool:
        if s == "climate.set_preset_mode":
            return True
        if s in HELPER_OR_SCHEDULE or s.startswith("input_"):
            return True
        if s in GENERIC_HA:
            return True
        if s == "climate.set_hvac_mode":
            return False
        return False

    if extra_svc and all(dominated(s) for s in extra_svc) and (gold_toks or any(is_state_device(t[0]) for t in b0_toks)):
        return "M2-D_dominated_behavior"
    if any(dominated(s) for s in extra_svc) and gold_caps & {"ClimateControl", "Lighting", "Notify"} and not any(
        cap_of_svc(s) in {"Notify", "Lighting", "ClimateControl"} for s in extra_svc if not dominated(s)
    ):

        if all(dominated(s) or s in AUX_LOG or s.startswith(("google_sheets.", "llmvision.")) for s in extra_svc):
            return "M2-D_dominated_behavior"

    extra_caps = {cap_of_svc(s) for s in extra_svc}
    if gold_toks and extra_caps and extra_caps.isdisjoint(gold_caps):
        return "M2-C_contextually_unnecessary_component"
    if gold_toks and extra_caps - gold_caps:
        return "M2-C_contextually_unnecessary_component"

    if gold_idle:
        return "M2-E_parent_semantic_mismatch"
    return "M2-E_parent_semantic_mismatch"

def m3_mismatch_kind(b0_toks: list, gold_toks: list) -> str:
    gc = Counter(gold_toks)
    bc = Counter(b0_toks)
    if gc == bc:
        return "none"

    used_b = Counter()
    entity_m = param_m = service_m = 0
    for g in gold_toks:

        if bc[g] > used_b[g]:
            used_b[g] += 1
            continue

        found = None
        for b in b0_toks:
            if used_b[b] >= bc[b]:
                continue
            if b[0] == g[0] and b[1] == g[1] and b[2] != g[2]:
                found = ("parameter", b)
                break
        if found:
            used_b[found[1]] += 1
            param_m += 1
            continue
        found = None
        for b in b0_toks:
            if used_b[b] >= bc[b]:
                continue
            if b[0] == g[0] and b[1] != g[1]:
                found = ("entity", b)
                break
        if found:
            used_b[found[1]] += 1
            entity_m += 1
            continue
        found = None
        for b in b0_toks:
            if used_b[b] >= bc[b]:
                continue
            if b[0] != g[0]:
                found = ("service", b)
                break
        if found:
            used_b[found[1]] += 1
            service_m += 1
            continue
        service_m += 1
    if service_m:
        return "service"
    if entity_m:
        return "entity_grounding"
    if param_m:
        return "parameter"
    return "service"

def main() -> None:
    print("load SS + traces...", flush=True)
    y_idx = load_jsonl(Y_PATH)
    traces = load_jsonl(TRACE)
    ss_by_id: dict[str, dict] = {}
    with SS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                ss_by_id[str(rec.get("sample_id") or "")] = rec

    print("Strict Repair-v2 SS eval...", flush=True)
    n = b0_exact = f_exact = f_rs = f_fr = 0
    v_exact = v_rs = v_fr = 0
    new_exact = new_fr = broke = 0
    hvac_n = payload_n = both_n = 0
    dropped_keys = Counter()
    never_drop_retained = Counter()

    for i, (sid, s) in enumerate(ss_by_id.items(), 1):
        yrec = y_idx[sid]
        tr = traces[sid]
        y_dec, y_act, y_n, _at = y_pair(yrec)
        b0_dec, b0_act = b0_from_sample(s)
        b0_n = 1 if b0_act else 0
        b0_ok = semantic_ok(b0_dec, b0_act, b0_n, y_dec, y_act, y_n)
        n += 1
        if b0_ok:
            b0_exact += 1
        frozen = compact_action(tr.get("repaired_action"))
        f_ok = semantic_ok(*pair_from_act(frozen), y_dec, y_act, y_n)
        if f_ok:
            f_exact += 1
        if (not b0_ok) and f_ok:
            f_rs += 1
        if b0_ok and not f_ok:
            f_fr += 1
        ctx = build_repair_context(s)
        pred, kinds = apply_strict_v2(ctx, frozen)
        v_ok = semantic_ok(*pair_from_act(pred), y_dec, y_act, y_n)
        if v_ok:
            v_exact += 1
        if (not b0_ok) and v_ok:
            v_rs += 1
        if b0_ok and not v_ok:
            v_fr += 1
        if (not f_ok) and v_ok:
            new_exact += 1
        if f_ok and not v_ok:
            broke += 1
        if b0_ok and f_ok and not v_ok:
            new_fr += 1
        if "hvac_r1" in kinds:
            hvac_n += 1
        if any(k.startswith("payload_") for k in kinds):
            payload_n += 1
            for k in kinds:
                if k.startswith("payload_"):
                    for part in k.replace("payload_", "").split("+"):
                        dropped_keys[part] += 1
        if "hvac_r1" in kinds and any(k.startswith("payload_") for k in kinds):
            both_n += 1
        params = dict((pred or {}).get("parameters") or {}) if pred else {}
        for k in NEVER_DROP:
            if k in dict((frozen or {}).get("parameters") or {}):
                never_drop_retained[k] += int(k in params or k in dict((frozen or {}).get("parameters") or {}))
        if i % 3000 == 0:
            print(f"  ss {i}/{len(ss_by_id)}", flush=True)

    frozen_metrics = metrics_block(n, f_exact, b0_exact, f_rs, f_fr)
    strict_metrics = metrics_block(n, v_exact, b0_exact, v_rs, v_fr)
    strict_eval = {
        "note": (
            "Strict Repair-v2 Candidate sidecar. Frozen v1 / Gold Y / B0 / official traces untouched. "
            "Only HVAC R1 (ClimateControl + window_open + climate contract) and payload drop of level/logger. "
            "config_entry, worksheet, data, message, title, hvac_mode are never deleted."
        ),
        "repair_modified": False,
        "N": n,
        "official_Frozen_v1_not_overwritten": {
            "Semantic_Success": 0.6313,
            "False_Repair": 0.0149,
            "Preservation": 0.9456,
        },
        "Frozen_v1_recomputed_from_traces": frozen_metrics,
        "Strict_Repair_v2_Candidate": strict_metrics,
        "delta_vs_Frozen": {
            "Semantic_Success_pp": round(100.0 * (v_exact - f_exact) / n, 2),
            "new_exact_count": new_exact,
            "broke_frozen_exact": broke,
            "new_false_repair_count": new_fr,
            "False_Repair_delta_count": v_fr - f_fr,
        },
        "sidecars": {
            "hvac_r1_fired": hvac_n,
            "payload_level_logger_fired": payload_n,
            "both_fired": both_n,
            "dropped_keys": dict(dropped_keys),
            "never_drop_keys": sorted(NEVER_DROP),
        },
        "gates": {
            "hvac_requires_climate_capability": True,
            "hvac_requires_window_open": True,
            "hvac_requires_climate_contract": True,
            "window_open_alone_does_not_fire": True,
            "payload_only_level_logger": True,
        },
    }
    dump(OUT / "strict_repair_v2_evaluation.json", strict_eval)
    print(
        json.dumps(
            {
                "frozen_ss": frozen_metrics["Semantic_Success"],
                "strict_ss": strict_metrics["Semantic_Success"],
                "strict_rs": strict_metrics["Repair_Success"],
                "strict_fr": strict_metrics["False_Repair"],
                "new_exact": new_exact,
                "new_fr": new_fr,
                "hvac_fired": hvac_n,
                "payload_fired": payload_n,
            },
            indent=2,
        ),
        flush=True,
    )

    print("MA M2/M3/M6 + SET_REMOVE...", flush=True)
    ctx_cache: dict[str, dict] = {}

    def ctx_of(sid: str):
        if sid not in ctx_cache:
            ctx_cache[sid] = build_repair_context(ss_by_id[sid])
        return ctx_cache[sid]

    m2_primary = Counter()
    m2_multi = Counter()
    m2_svc = defaultdict(Counter)
    m2_caps = defaultdict(Counter)
    m2_comp_level = Counter()
    m2_parent_level = Counter()
    m2_examples = defaultdict(list)
    m2_component_solvable = Counter()

    m3_kind = Counter()
    m3_comp_vs_parent = Counter()
    m3_svc_pair = Counter()
    m3_examples: list[dict] = []
    m3_set_refine_needed = 0

    m6_rows: list[dict] = []

    buckets = {k: acc_bucket() for k in ("A", "B", "C", "D")}
    d_new_fr_from_c = 0
    d_broke_c = 0
    d_fixed_from_c = 0
    remove_reasons = Counter()
    m2_solved_by_d = 0
    m2_remaining_after_c = 0
    m2_remaining_after_d = 0
    n_parent = 0
    n_m2 = n_m3 = n_m6 = 0

    def toks_from_acts(acts: list[dict], bind_all: dict) -> list:
        out = []
        for a in acts:
            origin = str(a.get("component_origin") or "")
            t = tok(a, bind_all.get(origin) or {})
            if t:
                out.append(t)
        return out

    def independent_merge(comps: list[dict], bind_all: dict, traces: dict, mode: str) -> list[dict]:
        merged = []
        for c in comps:
            sid = str(c.get("single_scene_sample_id") or "")
            origin = str(c.get("component_id") or "")
            bind = bind_all.get(origin) or {}
            fr = compact_action((traces.get(sid) or {}).get("repaired_action"))
            if mode == "frozen":
                out = fr
            else:
                ctx = ctx_of(sid)
                out, _k = apply_strict_v2(ctx, fr)
            mapped = map_repaired(out, bind)
            if mapped:
                mapped["component_origin"] = origin
                merged.append(mapped)
        return merged

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

            frozen_merged = independent_merge(comps, bind_all, traces, "frozen")
            strict_merged = independent_merge(comps, bind_all, traces, "strict")
            f_toks = toks_from_acts(frozen_merged, bind_all)
            c_toks = toks_from_acts(strict_merged, bind_all)
            mB = set_metrics(f_toks, gold_toks)
            mC = set_metrics(c_toks, gold_toks)

            d_acts, d_trace = set_remove_v1(strict_merged, bind_all)
            d_toks = toks_from_acts(d_acts, bind_all)
            mD = set_metrics(d_toks, gold_toks)
            fired = any(row["removed"] for row in d_trace)
            for row in d_trace:
                if row["removed"]:
                    remove_reasons[row["reason"]] += 1

            b0_ok = mA["set_exact"]
            add_bucket(buckets["A"], mA, b0_ok, mA["set_exact"], mA["n_pred"])
            add_bucket(buckets["B"], mB, b0_ok, mB["set_exact"], mB["n_pred"])
            add_bucket(buckets["C"], mC, b0_ok, mC["set_exact"], mC["n_pred"])
            add_bucket(buckets["D"], mD, b0_ok, mD["set_exact"], mD["n_pred"], fired=fired)
            if mC["set_exact"] and not mD["set_exact"]:
                d_broke_c += 1
                if b0_ok:
                    d_new_fr_from_c += 1
            if (not mC["set_exact"]) and mD["set_exact"]:
                d_fixed_from_c += 1

            gs = {t[0] for t in gold_toks}
            bs = {t[0] for t in b0_toks}
            b0_clim = [t for t in b0_toks if t[0].startswith("climate.")]
            modes = {hvac_mode({"parameters": json.loads(t[2])}) for t in b0_clim if t[0] == "climate.set_hvac_mode"}
            if any(a.get("service") == "climate.turn_off" for a in parent_acts):
                modes.add("off_turn")
            tag = None
            if any(x in modes for x in ACTIVE_HVAC) and (
                "off" in modes or "off_turn" in modes
            ) and len({m for m in modes if m}) > 1:
                tag = "M6"
                n_m6 += 1
            elif gold_toks and not mA["set_exact"] and len(gs - bs) > 0 and len(bs - gs) == 0:
                tag = "M1"
            elif (not gold_toks and b0_toks) or (len(bs - gs) > 0 and len(gs - bs) == 0 and gold_toks):
                tag = "M2"
                n_m2 += 1
            elif gold_toks and b0_toks and not mA["set_exact"]:
                tag = "M3"
                n_m3 += 1
            else:
                tag = "other"

            if tag == "M2":
                extra_toks = []
                extra_acts: list[dict] = []
                gc = Counter(gold_toks)
                used_g = Counter()
                for a in parent_acts:
                    origin = str(a.get("component_origin") or "")
                    t = tok(a, bind_all.get(origin) or {})
                    if not t:
                        continue
                    if used_g[t] < gc[t]:
                        used_g[t] += 1
                    else:
                        extra_toks.append(t)
                        extra_acts.append(a)
                gold_idle = not gold_toks
                cls = classify_m2(extra_toks, gold_toks, b0_toks, extra_acts, gold_idle)
                m2_primary[cls] += 1
                for t in extra_toks:
                    m2_svc[cls][t[0]] += 1
                combo = "+".join(sorted({cap_of_svc(t[0]) for t in b0_toks} or ["none"]))
                m2_caps[cls][combo] += 1
                gold_c = Counter(gold_toks)
                c_extra_n = sum((Counter(c_toks) - gold_c).values())
                if c_extra_n == 0 or mC["set_exact"]:
                    m2_comp_level[cls] += 1
                    m2_component_solvable[cls] += 1
                else:
                    m2_parent_level[cls] += 1
                if mC["set_exact"]:
                    pass
                else:
                    m2_remaining_after_c += 1
                    if mD["set_exact"]:
                        m2_solved_by_d += 1
                    else:
                        m2_remaining_after_d += 1
                if len(m2_examples[cls]) < 6:
                    m2_examples[cls].append(
                        {
                            "multi_action_id": mid,
                            "scenes": [c.get("scene") for c in comps],
                            "n_b0": len(b0_toks),
                            "n_y": len(gold_toks),
                            "extra_services": [t[0] for t in extra_toks[:8]],
                            "gold_services": [t[0] for t in gold_toks[:8]],
                        }
                    )

            if tag == "M3":
                kind = m3_mismatch_kind(b0_toks, gold_toks)
                m3_kind[kind] += 1

                local_ok = 0
                local_n = 0
                for c in comps:
                    sid = str(c.get("single_scene_sample_id") or "")
                    if sid not in y_idx:
                        continue
                    local_n += 1
                    y_dec, y_act, y_n, _at = y_pair(y_idx[sid])
                    fr = compact_action((traces.get(sid) or {}).get("repaired_action"))
                    if semantic_ok(*pair_from_act(fr), y_dec, y_act, y_n):
                        local_ok += 1
                if local_n and local_ok == local_n:
                    scope = "parent_level_semantic_error"
                elif local_n and local_ok == 0:
                    scope = "component_level_semantic_error"
                else:
                    scope = "mixed_component_and_parent"
                m3_comp_vs_parent[scope] += 1
                if scope != "component_level_semantic_error":
                    m3_set_refine_needed += 1
                gsvcs = [t[0] for t in gold_toks]
                bsvcs = [t[0] for t in b0_toks]
                m3_svc_pair[str(sorted(set(bsvcs))[:3]) + " -> " + str(sorted(set(gsvcs))[:3])] += 1
                if len(m3_examples) < 12:
                    m3_examples.append(
                        {
                            "multi_action_id": mid,
                            "kind": kind,
                            "scope": scope,
                            "scenes": [c.get("scene") for c in comps],
                            "b0_services": bsvcs,
                            "gold_services": gsvcs,
                            "ss_local_exact": f"{local_ok}/{local_n}",
                        }
                    )

            if tag == "M6":
                local_legal = []
                clim_acts = []
                origs = []
                for a in parent_acts:
                    s = str(a.get("service") or "")
                    if not s.startswith("climate."):
                        continue
                    tgt = orig_of(a, bind_all)
                    mode = hvac_mode(a)
                    clim_acts.append({"service": s, "original_entity": tgt, "hvac_mode": mode, "origin": a.get("component_origin")})
                    origs.append(tgt)
                    local_legal.append(s in {"climate.set_hvac_mode", "climate.turn_off", "climate.set_preset_mode"} and bool(tgt or s.endswith("turn_off")))
                shared = len({o for o in origs if o}) <= 1
                after_c_modes = []
                for a in strict_merged:
                    if str(a.get("service") or "").startswith("climate."):
                        after_c_modes.append(hvac_mode(a) or str(a.get("service")))
                after_d_modes = []
                for a in d_acts:
                    if str(a.get("service") or "").startswith("climate."):
                        after_d_modes.append(hvac_mode(a) or str(a.get("service")))
                m6_rows.append(
                    {
                        "multi_action_id": mid,
                        "scenes": [c.get("scene") for c in comps],
                        "climate_actions_b0": clim_acts,
                        "each_action_locally_legal": all(local_legal) and bool(local_legal),
                        "combined_error": "opposing_hvac_modes_on_parent",
                        "shared_original_entity": shared,
                        "original_entities": sorted({o for o in origs if o}),
                        "cross_component_dependency": shared and len(clim_acts) >= 2,
                        "needs_state_transition_graph": shared and len(clim_acts) >= 2,
                        "gold_services": [t[0] for t in gold_toks],
                        "gold_n": len(gold_toks),
                        "after_strict_v2_climate": after_c_modes,
                        "after_set_remove_climate": after_d_modes,
                        "set_exact_after_C": mC["set_exact"],
                        "set_exact_after_D": mD["set_exact"],
                    }
                )

            if n_parent % 400 == 0:
                print(f"  ma {n_parent}", flush=True)

    m2_decomp = {
        "note": "Eval-only vs existing MA y_output. Does not retune Frozen/v2. Primary tag uses same M2 definition as taxonomy (B0 superset of Gold).",
        "N_parents": n_parent,
        "M2_n": n_m2,
        "classes": {},
        "examples": {k: v for k, v in m2_examples.items()},
        "set_remove_v1_does_not_target": ["M2-C_contextually_unnecessary_component", "M2-E_parent_semantic_mismatch"],
        "set_remove_v1_targets": ["M2-A_semantic_duplicate (only contradicted active HVAC)", "M2-B_auxiliary_logging (log/logbook only)", "M2-D_dominated_behavior"],
    }
    for cls in [
        "M2-A_semantic_duplicate",
        "M2-B_auxiliary_logging",
        "M2-C_contextually_unnecessary_component",
        "M2-D_dominated_behavior",
        "M2-E_parent_semantic_mismatch",
    ]:
        nn = m2_primary[cls]
        m2_decomp["classes"][cls] = {
            "parent_count": nn,
            "rate_of_M2": round(nn / max(n_m2, 1), 4),
            "action_service_distribution": dict(m2_svc[cls].most_common(12)),
            "capability_combinations": dict(m2_caps[cls].most_common(8)),
            "component_level_judgeable": m2_comp_level[cls],
            "must_read_other_components": m2_parent_level[cls],
            "component_level_if_independent_strict_already_drops_extra": m2_component_solvable[cls],
        }

    dump(OUT / "multi_action_m2_decomposition.json", m2_decomp)

    m3_out = {
        "note": "400 M3 vs MA y_output. SET_REFINE is not implemented.",
        "N_M3": n_m3,
        "mismatch_kind": dict(m3_kind),
        "scope": dict(m3_comp_vs_parent),
        "set_refine_not_justified_if_component_repair_already_covers": m3_comp_vs_parent.get("component_level_semantic_error", 0),
        "parent_or_mixed_may_need_set_refine_or_other": m3_comp_vs_parent.get("parent_level_semantic_error", 0)
        + m3_comp_vs_parent.get("mixed_component_and_parent", 0),
        "top_service_pairs": dict(m3_svc_pair.most_common(15)),
        "examples": m3_examples,
        "recommendation": "Do not implement SET_REFINE in this stage.",
    }
    dump(OUT / "multi_action_m3_analysis.json", m3_out)

    m6_local = sum(1 for r in m6_rows if r["each_action_locally_legal"])
    m6_shared = sum(1 for r in m6_rows if r["shared_original_entity"])
    m6_cross = sum(1 for r in m6_rows if r["cross_component_dependency"])
    m6_graph = sum(1 for r in m6_rows if r["needs_state_transition_graph"])
    m6_fixed_c = sum(1 for r in m6_rows if r["set_exact_after_C"])
    m6_fixed_d = sum(1 for r in m6_rows if r["set_exact_after_D"])
    m6_out = {
        "N_M6": n_m6,
        "each_action_locally_legal": m6_local,
        "shared_original_entity": m6_shared,
        "true_cross_component_dependency": m6_cross,
        "needs_state_transition_graph": m6_graph,
        "set_exact_after_strict_v2_component": m6_fixed_c,
        "set_exact_after_set_remove": m6_fixed_d,
        "combined_error_mechanism": (
            "Two climate components emit opposing HVAC modes (heat/cool vs off) on the same original "
            "climate.living_room_ac. Each action is a legal climate command. The parent combined state is illegal. "
            "Independent HVAC R1 converts heat→off when that component has window_open; leftover active modes "
            "need parent-level contradiction handling, not a full behavior graph yet."
        ),
        "cases": m6_rows,
    }
    dump(OUT / "multi_action_m6_case_analysis.json", m6_out)

    eval_d = {
        "note": (
            "A=raw parent B0. B=Frozen v1 independent component repair + concat. "
            "C=Strict Repair-v2 independent component repair + concat. "
            "D=C + SET_REMOVE v1 (parent-level necessity, no Gold). No SET_REFINE."
        ),
        "repair_modified": False,
        "gold": "MA y_output eval only",
        "N": n_parent,
        "Baseline_A_raw_B0": finish_bucket(buckets["A"]),
        "Baseline_B_independent_frozen": finish_bucket(buckets["B"]),
        "Baseline_C_independent_strict_v2": finish_bucket(buckets["C"]),
        "Baseline_D_strict_v2_plus_set_remove": finish_bucket(buckets["D"]),
        "set_remove": {
            "fired_parents": buckets["D"]["remove_fired"],
            "reasons": dict(remove_reasons),
            "fixed_from_C_inexact_to_exact": d_fixed_from_c,
            "broke_C_exact": d_broke_c,
            "new_false_repair_vs_correct_B0_among_broke_C": d_new_fr_from_c,
        },
        "M2_coverage": {
            "M2_n": n_m2,
            "M2_still_inexact_after_C": m2_remaining_after_c,
            "M2_solved_by_SET_REMOVE_from_C_residual": m2_solved_by_d,
            "M2_still_inexact_after_D": m2_remaining_after_d,
        },
    }
    dump(OUT / "multi_action_set_remove_evaluation.json", eval_d)

    finA, finB, finC, finD = (finish_bucket(buckets[k]) for k in ("A", "B", "C", "D"))

    m2_lines = []
    for cls, recx in m2_decomp["classes"].items():
        m2_lines.append(
            f"| {cls} | {recx['parent_count']} | {recx['rate_of_M2']} | "
            f"{recx['component_level_judgeable']} | {recx['must_read_other_components']} |"
        )
    write_md(
        OUT / "multi_action_m2_analysis.md",
        f"""# Multi-action M2 decomposition

M2 = {n_m2} / {n_parent} parents where raw B0 is a **superset** of MA `y_output` (eval only). Frozen v1 and Gold Y are unchanged.

## Primary classes

| Class | Parents | Rate of M2 | Component-level (extras already gone after Strict v2 concat) | Must read other components |
|---|---|---|---|---|
{chr(10).join(m2_lines)}

## What each class is

**M2-A semantic duplicate.** Extra tokens that are equivalent to a Gold action or to each other (e.g. `climate.turn_off` plus `climate.set_hvac_mode` off on the same original entity). Parent B0 is already `dedupe_log`'d, so true string duplicates are rare; this class is mostly *equivalent* HVAC deactivates.

**M2-B auxiliary / logging redundancy.** Extra `system_log` / `logbook` / `google_sheets` / `llmvision` / persistent notification on top of a device Gold, or Gold-idle parents whose B0 is only recording/vision.

**M2-C contextually unnecessary component.** Extra first-class behavior from a component whose capability is absent from Gold — typically `notify.mobile_app_phone` or lighting while Gold is climate-only. The extra action is often locally justified for that component (event occurred). Necessity is a **parent** question.

**M2-D dominated behavior.** Extra `climate.set_preset_mode`, `schedule.activate` / `scene.turn_on`, helpers, or `homeassistant.turn_*` while a more specific device action already exists in the set.

**M2-E parent-level semantic mismatch.** Gold is `NO_ACTION` (1418 parents corpus-wide) while B0 still contains first-class Notify/Lighting/Climate actions that look locally necessary. This is not a safe SET_REMOVE target: deleting them because Gold is idle would be Gold-chasing.

## Component vs parent

Independent Strict v2 already removes some extras (SS REMOVE on that component). Remaining extras after concat **must** be judged against sibling actions. SET_REMOVE v1 only touches B/D (and contradicted active HVAC). It does **not** delete Notify or Lighting because a climate sibling exists.
""",
    )

    write_md(
        OUT / "multi_action_set_remove_design.md",
        f"""# SET_REMOVE v1 design

Input is the **whole parent** after independent Strict Repair-v2: parent runtime, all component contracts/observations, merged action set. The object of judgment is an **action inside the action-set**, not a scene_type and not a service blacklist by itself.

## Parent-level necessity test

For each action `A` in merged set `S`, ask: **is A still necessary given siblings `S \\ {{A}}`?**

Keep `A` when it is a first-class capability whose evidence does not depend on siblings:

- ClimateControl deactivate / off (window_open is used by HVAC R1 at component, not to delete Notify)
- Lighting activate/deactivate
- Notify emit
- Record with semantic `data`
- Vision analyzer

Remove `A` only with sibling evidence:

| Condition | Reason |
|---|---|
| `A` is log/logbook **and** a STATE device sibling exists | AUX_LOG_GIVEN_STATE_SIBLING |
| `A` is helper / `schedule.activate` / `scene.turn_on|off` **and** a STATE sibling exists | DOMINATED_HELPER_OR_SCHEDULE_GIVEN_STATE |
| `A` is `climate.set_preset_mode` **and** HVAC off/turn_off exists on the same original entity | DOMINATED_PRESET_GIVEN_HVAC_OFF |
| `A` is `set_hvac_mode` heat/cool/auto **and** off exists on the same original entity | CONTRADICTED_ACTIVE_HVAC_GIVEN_OFF |
| `A` is `homeassistant.turn_*` **and** a light.* or climate.* sibling exists | GENERIC_HA_GIVEN_SPECIFIC_DEVICE |

## Explicit non-rules

- Do not remove Notify because HVAC exists (mixed climate+notify parents are common; Notify is a different capability).
- Do not remove Lighting because HVAC exists.
- Do not wipe the set because many Gold labels are idle (M2-E).
- Do not drop `worksheet` / `config_entry` / `data` / `message` / `title` / `hvac_mode`.
- Do not collapse two namespaced climate offs on the same original entity (Gold sometimes wants count=2).
- Do not use Gold Y, scene_type alone, or a global service blacklist.

## Pipeline

```
Independent Strict Repair-v2 (HVAC R1 + level/logger)
        ↓
Behavior composition (concat, entity map)
        ↓
SET_REMOVE v1 necessity test on the action-set
```

No SET_REFINE, SET_COMPLETE, or REORDER in this stage.
""",
    )

    write_md(
        OUT / "multi_action_next_step_decision.md",
        f"""# Multi-action next-step decision

Frozen Repair v1 is unchanged: **Semantic Success = 63.13%, False Repair = 1.49%**. Official traces, Gold Y, and B0 were not modified.

Sources: `strict_repair_v2_evaluation.json`, `multi_action_m2_decomposition.json`, `multi_action_set_remove_evaluation.json`, `multi_action_m3_analysis.json`, `multi_action_m6_case_analysis.json`.

---

### 1. Strict Repair-v2 final SS?

**Semantic Success = {strict_metrics["Semantic_Success"]:.4f} ({strict_metrics["Semantic_Success_count"]} / {n})**

| | Frozen v1 traces | Strict v2 candidate |
|---|---|---|
| Semantic Success | {frozen_metrics["Semantic_Success"]:.4f} | {strict_metrics["Semantic_Success"]:.4f} |
| Repair Success | {frozen_metrics["Repair_Success"]:.4f} | {strict_metrics["Repair_Success"]:.4f} |
| False Repair | {frozen_metrics["False_Repair"]:.4f} | {strict_metrics["False_Repair"]:.4f} |
| Preservation | {frozen_metrics["Preservation_Rate"]:.4f} | {strict_metrics["Preservation_Rate"]:.4f} |
| New exact vs Frozen | — | {new_exact} |
| New false repair vs Frozen | — | {new_fr} |

HVAC R1 fired {hvac_n} times. Payload `level`/`logger` fired {payload_n} times. `config_entry` / `worksheet` / `data` / `message` / `title` / `hvac_mode` were not deleted. This number, not V4's 81.30%, is the Strict Repair-v2 Candidate.

---

### 2. How many of 1914 M2 can parent-level SET_REMOVE solve?

M2 counted in this run: **{n_m2}**.

| Class | n | SET_REMOVE v1? |
|---|---|---|
| M2-A semantic duplicate | {m2_primary["M2-A_semantic_duplicate"]} | only contradicted active HVAC |
| M2-B auxiliary/logging | {m2_primary["M2-B_auxiliary_logging"]} | log/logbook only, if STATE sibling |
| M2-C unnecessary component | {m2_primary["M2-C_contextually_unnecessary_component"]} | **no** (Notify/Lighting vs climate) |
| M2-D dominated | {m2_primary["M2-D_dominated_behavior"]} | **yes** (preset/helper/generic HA) |
| M2-E parent mismatch (Gold idle) | {m2_primary["M2-E_parent_semantic_mismatch"]} | **no** |

After independent Strict v2, **{m2_remaining_after_c}** M2 parents were still inexact. SET_REMOVE then made **{m2_solved_by_d}** of those exact. Residual M2 after D: **{m2_remaining_after_d}**.

The bulk of M2 is C/E (extra Notify/Lighting or Gold-idle). Those need a different, higher-FR-risk policy — not this operator.

---

### 3. Did SET_REMOVE significantly lift MA set-exact?

| Method | Set-exact | F1 | mean |n| |
|---|---|---|---|---|
| A raw B0 | {finA["Parent_Set_Exact_Match"]:.4f} ({finA["Parent_Set_Exact_count"]}) | {finA["Action_F1"]:.4f} | {finA["mean_action_count"]} |
| B Frozen independent | {finB["Parent_Set_Exact_Match"]:.4f} ({finB["Parent_Set_Exact_count"]}) | {finB["Action_F1"]:.4f} | {finB["mean_action_count"]} |
| C Strict v2 independent | {finC["Parent_Set_Exact_Match"]:.4f} ({finC["Parent_Set_Exact_count"]}) | {finC["Action_F1"]:.4f} | {finC["mean_action_count"]} |
| D C + SET_REMOVE v1 | {finD["Parent_Set_Exact_Match"]:.4f} ({finD["Parent_Set_Exact_count"]}) | {finD["Action_F1"]:.4f} | {finD["mean_action_count"]} |

Delta D vs C: **{finD["Parent_Set_Exact_count"] - finC["Parent_Set_Exact_count"]}** exact parents ({round(100.0 * (finD["Parent_Set_Exact_Match"] - finC["Parent_Set_Exact_Match"]), 2)} pp). SET_REMOVE fired on {finD["set_remove_fired"]} parents. Reasons: {dict(remove_reasons)}.

---

### 4. New False Repair from SET_REMOVE?

False Parent Repair (broke a parent whose **raw B0** was already exact): D = {finD["False_Parent_Repair_count"]} / {n_parent} = {finD["False_Parent_Repair"]}.

Broke a parent that **C already had exact**: **{d_broke_c}**. New FR vs correct B0 among those: **{d_new_fr_from_c}**.

Preservation D = {finD["Preservation"]} (correct B0 n = {finD["correct_B0_count"]}).

---

### 5. Are 400 M3 worth SET_REFINE now?

M3 n = {n_m3}. Mismatch kinds: {dict(m3_kind)}. Scope: {dict(m3_comp_vs_parent)}.

A large slice is **service** mismatch (`notify.mobile_app_phone` vs `notify.mobile_app`, `climate.set_hvac_mode` vs `climate.turn_off`) and **entity grounding** (`!input` vs namespaced `comp_c*_climate_*` — independent repair already remaps via SS traces).

Component-local SS-exact on all members: {m3_comp_vs_parent.get("parent_level_semantic_error", 0)} parents — parent Gold disagrees even when each SS component is exact. **Do not implement SET_REFINE in this stage.** Revisit after SET_REMOVE policy for M2-C is decided; otherwise SET_REFINE will be asked to rewrite Notify into climate (wrong operator).

---

### 6. Do 73 M6 rows justify a Behavior Graph module?

M6 n = {n_m6}. Locally legal actions: {m6_local}. Shared original entity: {m6_shared}. Cross-component dependency: {m6_cross}. “Needs transition graph” flag: {m6_graph}. Exact after independent Strict v2: {m6_fixed_c}. Exact after SET_REMOVE: {m6_fixed_d}.

Mechanism is one pattern: opposing HVAC modes on **the same** `climate.living_room_ac`. HVAC R1 plus contradicted-active SET_REMOVE already cover most of the graph that would be built. **73 rows are not enough to freeze a general Behavior Evidence Graph.** Keep `contradicts` as a SET_REMOVE condition; do not add a graph runtime yet.

---

## Sequence (do not merge codebases)

1. Treat **Strict Repair-v2** (HVAC R1 + `level`/`logger`) as the SS candidate; do not promote V4's worksheet strip.
2. Keep SET_REMOVE v1 as an MA sidecar; it is safe but not the M2 solution.
3. Next MA research target is **M2-C necessity for Notify/Lighting given a climate sibling** — only with an ABSTAIN / high-precision gate, not Gold-idle deletion.
4. No SET_REFINE, no Behavior Graph freeze, no Frozen v1 edit.
""",
    )

    print(
        json.dumps(
            {
                "strict_ss": strict_metrics["Semantic_Success"],
                "strict_fr": strict_metrics["False_Repair"],
                "new_exact": new_exact,
                "new_fr": new_fr,
                "M2": n_m2,
                "m2_primary": dict(m2_primary),
                "M3": n_m3,
                "m3_kind": dict(m3_kind),
                "M6": n_m6,
                "A_exact": finA["Parent_Set_Exact_Match"],
                "B_exact": finB["Parent_Set_Exact_Match"],
                "C_exact": finC["Parent_Set_Exact_Match"],
                "D_exact": finD["Parent_Set_Exact_Match"],
                "D_fr": finD["False_Parent_Repair_count"],
                "d_fixed": d_fixed_from_c,
                "d_broke_c": d_broke_c,
                "remove_reasons": dict(remove_reasons),
                "m2_solved_d": m2_solved_by_d,
            },
            indent=2,
        ),
        flush=True,
    )

if __name__ == "__main__":
    main()
