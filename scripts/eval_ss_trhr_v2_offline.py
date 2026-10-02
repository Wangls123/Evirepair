from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ss_b0_vs_final_gold_y import SCENARIO_MAP, SS, Y_PATH, b0_from_sample
from eval_ss_trhr_final import (
    compact_act,
    error_type,
    load_jsonl,
    metrics_block,
    pair_from_act,
    semantic_ok,
    y_pair,
)
from smarthome_mdf.ss_trhr_repair.actions import compact_action
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.evidence import window_open

OUT = ROOT / "runs" / "ss_trhr_repair"
TRACE = OUT / "repair_execution_trace.jsonl"
ACTIVE_HVAC = {"heat", "cool", "auto", "heat_cool", "dry", "fan_only"}
EVENT_PREFIXES = (
    "system_log.",
    "logbook.",
    "google_sheets.",
    "persistent_notification.",
    "notify.",
)
EVENT_EXACT = {"system_log.write", "logbook.log", "google_sheets.append_sheet", "logbook.write"}

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def is_event_svc(svc: str) -> bool:
    s = str(svc or "")
    return s in EVENT_EXACT or s.startswith(EVENT_PREFIXES)

def climate_capability(ctx: dict[str, Any], pred: dict[str, Any] | None) -> bool:
    declared = str(ctx.get("declared_effect_service") or "")
    cands = [str(s) for s in (ctx.get("candidate_services") or []) if s]
    svc = str((pred or {}).get("service") or "")
    return (
        svc.startswith("climate.")
        or declared.startswith("climate.")
        or any(c.startswith("climate.") for c in cands)
    )

def hvac_mode_of(act: dict[str, Any] | None) -> str | None:
    if not act:
        return None
    raw = (act.get("parameters") or {}).get("hvac_mode")
    if raw in (None, ""):
        return None
    return str(raw).strip().lower()

def r1_should_fire(ctx: dict[str, Any], pred: dict[str, Any] | None, *, use_scene: bool) -> bool:

    if not pred:
        return False
    svc = str(pred.get("service") or "")
    if svc != "climate.set_hvac_mode":
        return False
    if use_scene and str(ctx.get("scene_type") or "") != "climate_window":
        return False
    if not use_scene and not climate_capability(ctx, pred):
        return False
    payload = ctx.get("payload") or {}
    if not window_open(payload.get("observation") or {}, payload.get("entity_observations") or []):
        return False
    mode = hvac_mode_of(pred)
    if mode == "off":
        return False
    return mode is None or mode in ACTIVE_HVAC

def apply_r1(pred: dict[str, Any]) -> dict[str, Any]:
    out = compact_action(pred) or dict(pred)
    params = dict(out.get("parameters") or {})
    params["hvac_mode"] = "off"
    out["parameters"] = params
    return out

def canonicalize_event(pred: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pred:
        return pred
    svc = str(pred.get("service") or "")
    if not is_event_svc(svc):
        return pred
    params = dict(pred.get("parameters") or {})
    if not params:
        return pred
    out = compact_action(pred) or dict(pred)
    out["parameters"] = {}
    return out

def main() -> None:
    y_idx = load_jsonl(Y_PATH)
    traces = load_jsonl(TRACE)
    samples: list[dict[str, Any]] = []
    with SS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    n = b0_exact = full_exact = full_rs = full_fr = 0
    v2_exact = v2_rs = v2_fr = 0
    fired = 0
    fired_scene = 0
    repaired_from_fail = 0
    broke_exact = 0
    new_fr = 0
    already_ok = 0
    window_closed_skip = 0
    already_off_skip = 0
    not_climate = 0
    b0_svc = Counter()
    b0_mode = Counter()
    r_mode = Counter()
    y_mode = Counter()
    y_svc = Counter()
    obs_window = Counter()
    obs_hvac = Counter()
    op_on_fire = Counter()
    et_on_fire = Counter()
    gold_transition = Counter()
    examples: list[dict[str, Any]] = []

    ev_strip = 0
    ev_gain = 0
    ev_broke = 0
    ev_new_fr = 0
    ev_exact_kept = 0
    ev_gold_empty = 0
    ev_gold_nonempty = 0
    ev_full_nonempty_exact = 0
    ev_svc = Counter()
    ev_scene = Counter()
    ev_keys = Counter()
    ev_class = Counter()

    print("v2 offline sidecars...", flush=True)
    for i, s in enumerate(samples, 1):
        sid = str(s.get("sample_id") or "")
        yrec = y_idx[sid]
        tr = traces[sid]
        y_dec, y_act, y_n, at = y_pair(yrec)
        b0_dec, b0_act = b0_from_sample(s)
        b0_n = 1 if b0_act else 0
        b0_ok = semantic_ok(b0_dec, b0_act, b0_n, y_dec, y_act, y_n)
        n += 1
        if b0_ok:
            b0_exact += 1
        r_raw = tr.get("repaired_action")
        r_dec, r_act, r_n = pair_from_act(r_raw)
        r_ok = semantic_ok(r_dec, r_act, r_n, y_dec, y_act, y_n)
        if r_ok:
            full_exact += 1
        if (not b0_ok) and r_ok:
            full_rs += 1
        if b0_ok and not r_ok:
            full_fr += 1

        ctx = build_repair_context(s)
        pred = compact_action(r_raw)
        payload = ctx.get("payload") or {}
        obs = payload.get("observation") or {}
        ents = payload.get("entity_observations") or []
        wopen = window_open(obs, ents)

        scene_hit = r1_should_fire(ctx, pred, use_scene=True)
        cap_hit = r1_should_fire(ctx, pred, use_scene=False)
        if scene_hit:
            fired_scene += 1
        if cap_hit:
            fired += 1
            op_on_fire[str(tr.get("operator_used") or "KEEP")] += 1
            et_on_fire[error_type(s, b0_dec, b0_act, y_dec, y_act, at)] += 1
            b0_svc[str((b0_act or {}).get("service") or "NO_ACTION")] += 1
            b0_mode[str(hvac_mode_of(b0_act) or (None if b0_act else "NO_ACTION"))] += 1
            r_mode[str(hvac_mode_of(pred) or "empty")] += 1
            y_mode[str(hvac_mode_of({"parameters": (y_act or {}).get("parameters") or {}}) or ((y_act or {}).get("service") and "empty") or y_dec)] += 1
            y_svc[str((y_act or {}).get("service") or y_dec)] += 1
            obs_window[str(obs.get("window_state") or ("entity" if wopen else "absent"))] += 1
            obs_hvac[str(obs.get("hvac_mode") if obs.get("hvac_mode") not in (None, "") else "missing")] += 1
            y_to = hvac_mode_of({"parameters": (y_act or {}).get("parameters") or {}})
            r_from = hvac_mode_of(pred) or "empty"
            gold_transition[f"{r_from}->{y_to or y_dec}"] += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "sample_id": sid,
                        "operator": tr.get("operator_used"),
                        "b0": compact_act(b0_act),
                        "frozen_repair": compact_act(r_act),
                        "observation_window_state": obs.get("window_state"),
                        "observation_hvac_mode": obs.get("hvac_mode"),
                        "window_open_evidence": wopen,
                        "declared_effect_service": ctx.get("declared_effect_service"),
                        "gold_eval_only": compact_act(y_act) if y_dec == "ACTION" else {"decision": y_dec},
                    }
                )
            v2_pred = apply_r1(pred)
        else:
            v2_pred = pred
            if pred and str(pred.get("service") or "") == "climate.set_hvac_mode":
                if not wopen:
                    window_closed_skip += 1
                elif hvac_mode_of(pred) == "off":
                    already_off_skip += 1
            elif pred and str(pred.get("service") or "").startswith("climate."):
                not_climate += 0
            else:
                not_climate += 1

        v_dec, v_act, v_n = pair_from_act(v2_pred)
        v_ok = semantic_ok(v_dec, v_act, v_n, y_dec, y_act, y_n)
        if v_ok:
            v2_exact += 1
        if (not b0_ok) and v_ok:
            v2_rs += 1
        if b0_ok and not v_ok:
            v2_fr += 1
        if cap_hit:
            if (not r_ok) and v_ok:
                repaired_from_fail += 1
            elif r_ok and v_ok:
                already_ok += 1
            elif r_ok and not v_ok:
                broke_exact += 1
            if (not (b0_ok and not r_ok)) and (b0_ok and not v_ok):
                new_fr += 1

        ev_pred = canonicalize_event(pred)
        stripped = ev_pred is not pred and ev_pred is not None
        if stripped:
            ev_strip += 1
            ev_svc[str(pred.get("service"))] += 1
            ev_scene[SCENARIO_MAP.get(str(s.get("scene_type") or ""), str(s.get("scene_type") or ""))] += 1
            for k in (pred.get("parameters") or {}):
                ev_keys[str(k)] += 1
            y_params = (y_act or {}).get("parameters") or {} if y_dec == "ACTION" else {}
            if y_dec == "ACTION" and not y_params:
                ev_gold_empty += 1
                ev_class["serialization_mismatch"] += 1
            elif y_dec == "ACTION" and y_params:
                ev_gold_nonempty += 1
                same = json.dumps(y_params, sort_keys=True) == json.dumps(pred.get("parameters") or {}, sort_keys=True)
                ev_class["semantic_mismatch" if not same else "gold_also_has_payload"] += 1
            elif y_dec == "NO_ACTION":
                ev_class["gold_idle_payload_irrelevant"] += 1
            e_dec, e_act, e_n = pair_from_act(ev_pred)
            e_ok = semantic_ok(e_dec, e_act, e_n, y_dec, y_act, y_n)
            if (not r_ok) and e_ok:
                ev_gain += 1
            if r_ok and not e_ok:
                ev_broke += 1
            if r_ok:
                ev_full_nonempty_exact += 1
            if b0_ok and r_ok and not e_ok:
                ev_new_fr += 1
            if r_ok and e_ok:
                ev_exact_kept += 1

        if i % 2000 == 0:
            print(f"  {i}/{len(samples)}", flush=True)

    full_block = metrics_block(n, full_exact, b0_exact, full_rs, full_fr)
    v2_block = metrics_block(n, v2_exact, b0_exact, v2_rs, v2_fr)
    dump(
        OUT / "hvac_transition_offline_analysis.json",
        {
            "note": (
                "Sidecar on frozen traces. Rule is Y-blind: climate.set_hvac_mode + window_open "
                "from observation/entity_observations + climate capability from declared/candidates. "
                "Gold is scoring only. Frozen Repair / Gold Y / B0 files not modified. "
                "Official Full remains 63.13% / 1.49%."
            ),
            "N": n,
            "official_Full": {
                "Semantic_Success": full_block["Semantic_Success"],
                "False_Repair": full_block["False_Repair"],
                "not_overwritten": True,
            },
            "rule": {
                "id": "R1_window_implies_hvac_off",
                "reads_gold": False,
                "uses_scene_type": False,
                "evidence": ["observation.window_state", "entity_observations window open", "declared/candidate climate.*"],
                "action": "parameters.hvac_mode=off",
                "does_not_copy_gold_action": True,
            },
            "fired": {
                "capability_predicate": fired,
                "scene_type_climate_window_predicate": fired_scene,
                "scene_vs_capability_delta": fired - fired_scene,
                "operators": dict(op_on_fire),
                "error_types": dict(et_on_fire),
            },
            "b0_on_fired": {
                "services": dict(b0_svc),
                "hvac_mode": dict(b0_mode),
            },
            "frozen_repair_on_fired": {"hvac_mode": dict(r_mode)},
            "gold_eval_only_on_fired": {
                "services": dict(y_svc),
                "hvac_mode_or_decision": dict(y_mode),
                "transition_from_repair_mode": dict(gold_transition),
            },
            "observation_on_fired": {
                "window_state": dict(obs_window),
                "hvac_mode": dict(obs_hvac),
            },
            "offline_outcomes": {
                "potential_repaired": repaired_from_fail,
                "already_exact_unchanged": already_ok,
                "broke_current_exact": broke_exact,
                "new_false_repair": new_fr,
                "window_closed_set_hvac_skipped": window_closed_skip,
                "already_off_skipped": already_off_skip,
            },
            "sidecar_metrics_not_official": {
                "Semantic_Success": v2_block["Semantic_Success"],
                "Semantic_Success_count": v2_exact,
                "False_Repair": v2_block["False_Repair"],
                "False_Repair_count": v2_fr,
                "Repair_Success": v2_block["Repair_Success"],
                "Preservation_Rate": v2_block["Preservation_Rate"],
                "Gain_vs_Full_pp": round(100.0 * (v2_exact - full_exact) / max(n, 1), 2),
            },
            "false_repair_risk": {
                "new_false_repair": new_fr,
                "broke_exact": broke_exact,
                "risk": "none_on_this_corpus" if new_fr == 0 and broke_exact == 0 else "nonzero",
            },
            "examples": examples,
        },
    )
    dump(
        OUT / "event_payload_offline_analysis.json",
        {
            "note": (
                "Sidecar: drop parameters on event/record HA services in frozen Repair output. "
                "Gold used only to score. Official Full not overwritten."
            ),
            "stripped": ev_strip,
            "potential_gain_exact": ev_gain,
            "broke_current_exact": ev_broke,
            "new_false_repair": ev_new_fr,
            "gold_params_empty_among_stripped_action": ev_gold_empty,
            "gold_params_nonempty_among_stripped_action": ev_gold_nonempty,
            "classes": dict(ev_class),
            "services": dict(ev_svc),
            "scenarios": dict(ev_scene),
            "parameter_keys": dict(ev_keys),
            "full_exact_with_nonempty_event_payload": ev_full_nonempty_exact,
            "gain_pp": round(100.0 * ev_gain / max(n, 1), 2),
            "risk": "none_on_this_corpus" if ev_broke == 0 and ev_new_fr == 0 else "nonzero",
        },
    )
    print(
        json.dumps(
            {
                "hvac_fired": fired,
                "hvac_scene_fired": fired_scene,
                "hvac_repaired": repaired_from_fail,
                "hvac_broke": broke_exact,
                "hvac_new_fr": new_fr,
                "hvac_ss": v2_block["Semantic_Success"],
                "hvac_gain_pp": round(100.0 * (v2_exact - full_exact) / max(n, 1), 2),
                "ev_strip": ev_strip,
                "ev_gain": ev_gain,
                "ev_broke": ev_broke,
                "ev_class": dict(ev_class),
                "ev_svc": dict(ev_svc),
            },
            indent=2,
        ),
        flush=True,
    )

if __name__ == "__main__":
    main()
