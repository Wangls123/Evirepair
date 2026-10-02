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
    _llm_stub,
    compact_act,
    error_type,
    load_jsonl,
    metrics_block,
    pair_from_act,
    se_ok,
    semantic_ok,
    y_pair,
)
from eval_ss_trhr_reviewer_evidence import SERVICE_BLACKLIST
from smarthome_mdf.ss_trhr_repair.actions import compact_action, service_kind
from smarthome_mdf.ss_trhr_repair.add import run_add
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.evidence import (
    device_already_satisfied,
    lux,
    motion_on,
    quiet_blocked,
    window_open,
)
from smarthome_mdf.ss_trhr_repair.modify import _collect_entities, _unique_domain_entity, run_modify
from smarthome_mdf.ss_trhr_repair.pipeline import _illegal
from smarthome_mdf.ss_trhr_repair.remove import run_remove

OUT = ROOT / "runs" / "ss_trhr_repair"
TRACE = OUT / "repair_execution_trace.jsonl"
IDS_300 = OUT / "gold_y_audit_300_ids.json"
LOS_SCENES = ["lighting", "climate", "security", "visual", "periodic", "schedule"]

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def run_trhr(ctx, llm_fn) -> tuple[Any, str]:
    original = compact_action(ctx.get("b0_action"))
    repaired = original
    op = "KEEP"
    if original:
        r = run_remove(ctx, original, llm=llm_fn)
        if r.get("changed"):
            repaired, op = r.get("action_out"), "REMOVE"
        else:
            m = run_modify(ctx, original)
            if m.get("changed"):
                repaired, op = m.get("action_out"), "MODIFY"
    else:
        a = run_add(ctx, llm=llm_fn)
        if a.get("changed"):
            repaired, op = a.get("action_out"), "ADD"
    repaired = compact_action(repaired) if repaired else None
    if _illegal(repaired, ctx, original):
        return original, "KEEP"
    return repaired, op

def baseline_blacklist(ctx) -> tuple[Any, str]:
    original = compact_action(ctx.get("b0_action"))
    if not original:
        return None, "KEEP"
    svc = str(original.get("service") or "")
    if svc in SERVICE_BLACKLIST or svc.startswith("input_boolean.") or svc.startswith("notify."):
        return None, "REMOVE"
    return original, "KEEP"

def baseline_condition(ctx) -> tuple[Any, str]:
    original = compact_action(ctx.get("b0_action"))
    if not original:
        return None, "KEEP"
    payload = ctx.get("payload") or {}
    obs = payload.get("observation") or {}
    cond = payload.get("condition") or []
    trig = payload.get("trigger") or []
    if quiet_blocked(obs, cond):
        return None, "REMOVE"
    kind = service_kind(str(original.get("service") or ""))
    if kind == "EVENT" and not trig:
        return None, "REMOVE"
    blob = str(cond).lower()
    if "occupancy" in blob or "presence" in blob:
        occ = str(obs.get("occupancy") or obs.get("presence_state") or "").lower()
        svc = str(original.get("service") or "")
        if occ in {"off", "false", "0", "empty"} and svc.endswith("turn_on"):
            return None, "REMOVE"
    return original, "KEEP"

def baseline_state_only(ctx) -> tuple[Any, str]:
    original = compact_action(ctx.get("b0_action"))
    if not original:
        return None, "KEEP"
    svc = str(original.get("service") or "")
    kind = service_kind(svc)
    if kind in {"EVENT", "HELPER"}:
        return original, "KEEP"
    payload = ctx.get("payload") or {}
    obs = payload.get("observation") or {}
    ents = payload.get("entity_observations") or []
    if device_already_satisfied(original, obs, ents):
        return None, "REMOVE"
    if svc.startswith("light."):
        lx = lux(obs)
        mot = motion_on(obs)
        if svc.endswith("turn_on") and lx is not None and lx >= 80:
            return None, "REMOVE"
        if svc.endswith("turn_on") and mot is False:
            return None, "REMOVE"
    if svc.startswith("climate.") and svc.endswith("turn_off") and not window_open(obs, ents):
        return None, "REMOVE"
    if kind == "STATE" and not original.get("target_entity"):
        domain = svc.split(".", 1)[0]
        ent = _unique_domain_entity(_collect_entities(ctx), domain)
        if ent:
            filled = dict(original)
            filled["target_entity"] = ent
            out = compact_action(filled)
            if not _illegal(out, ctx, original):
                return out, "MODIFY"
    return original, "KEEP"

def score(pred, b0_ok, y_dec, y_act, y_n) -> tuple[bool, int, int]:
    p_dec, p_act, p_n = pair_from_act(pred)
    ok = semantic_ok(p_dec, p_act, p_n, y_dec, y_act, y_n)
    return ok, int((not b0_ok) and ok), int(b0_ok and not ok)

def pack_acc(n, exact, b0_exact, rs, fr, ops) -> dict[str, Any]:
    block = metrics_block(n, exact, b0_exact, rs, fr)
    block["Gain_pp"] = round(100.0 * (block["Semantic_Success"] - (b0_exact / max(n, 1))), 2)
    block["operators"] = dict(ops)
    return block

def fail_boundary(*, et: str, op: str, reason: str, b0_ok: bool, r_ok: bool, r_se: bool, y_dec: str) -> str:
    if b0_ok and not r_ok:
        return "incorrect_inference"
    if op == "REMOVE" and y_dec == "ACTION":
        return "incorrect_inference"
    if et in {"B2_parameter_mismatch", "B3_service_mismatch", "C3_missing_temporal_action"}:
        return "unsupported_behavior_type"
    if et == "C1_missing_state_action":
        return "unsupported_behavior_type"
    if et == "B4_action_target_ambiguity" and r_se:
        return "ambiguity"
    if et == "B4_action_target_ambiguity":
        return "ambiguity"
    if et == "C2_missing_event_action" or op == "ADD":
        return "ambiguity"
    if "AMBIGUOUS" in reason:
        return "ambiguity"
    return "insufficient_evidence"

def main() -> None:
    y_idx = load_jsonl(Y_PATH)
    traces = load_jsonl(TRACE)
    ids300 = json.loads(IDS_300.read_text(encoding="utf-8")).get("sample_ids") or []
    samples: list[dict[str, Any]] = []
    with SS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    methods = {
        "B0": {"exact": 0, "rs": 0, "fr": 0, "ops": Counter()},
        "Baseline-1_service_blacklist": {"exact": 0, "rs": 0, "fr": 0, "ops": Counter()},
        "Baseline-2_condition_based": {"exact": 0, "rs": 0, "fr": 0, "ops": Counter()},
        "Baseline-3_state_only": {"exact": 0, "rs": 0, "fr": 0, "ops": Counter()},
        "Full_Repair": {"exact": 0, "rs": 0, "fr": 0, "ops": Counter()},
    }
    los = {sc: {"full": {"exact": 0, "rs": 0, "fr": 0, "n": 0, "b0": 0}, "generic": {"exact": 0, "rs": 0, "fr": 0, "n": 0, "b0": 0}} for sc in LOS_SCENES}
    bound = Counter()
    bound_et = defaultdict(Counter)
    n = b0_exact_n = fail_n = 0
    gold_rows = []

    print("credibility extras...", flush=True)
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
            b0_exact_n += 1
        r_dec, r_act, r_n = pair_from_act(tr.get("repaired_action"))
        r_ok = semantic_ok(r_dec, r_act, r_n, y_dec, y_act, y_n)
        r_se = se_ok(r_dec, r_act, y_dec, y_act)
        op = str(tr.get("operator_used") or "KEEP")
        reason = str(tr.get("reason") or "")
        et = error_type(s, b0_dec, b0_act, y_dec, y_act, at)
        scene = SCENARIO_MAP.get(str(s.get("scene_type") or ""), str(s.get("scene_type") or "other"))

        if not r_ok:
            fail_n += 1
            bucket = fail_boundary(et=et, op=op, reason=reason, b0_ok=b0_ok, r_ok=r_ok, r_se=r_se, y_dec=y_dec)
            bound[bucket] += 1
            bound_et[bucket][et] += 1
            bound_et[bucket][f"op_{op}"] += 1

        reason_tr = str(tr.get("reason") or "")
        llm_fn = _llm_stub(False if reason_tr == "A3_LLM_UNNECESSARY" else True if reason_tr == "ADD_LLM_NECESSARY" else None)
        ctx = build_repair_context(s)

        preds = {
            "B0": (b0_act, "KEEP" if b0_act else "KEEP"),
            "Baseline-1_service_blacklist": baseline_blacklist(ctx),
            "Baseline-2_condition_based": baseline_condition(ctx),
            "Baseline-3_state_only": baseline_state_only(ctx),
            "Full_Repair": (tr.get("repaired_action"), op),
        }
        for name, (pred, opu) in preds.items():
            ok, rs, fr = score(pred, b0_ok, y_dec, y_act, y_n)
            methods[name]["exact"] += int(ok)
            methods[name]["rs"] += rs
            methods[name]["fr"] += fr
            methods[name]["ops"][opu] += 1

        if scene in los:
            los[scene]["full"]["n"] += 1
            los[scene]["full"]["b0"] += int(b0_ok)
            los[scene]["full"]["exact"] += int(r_ok)
            los[scene]["full"]["rs"] += int((not b0_ok) and r_ok)
            los[scene]["full"]["fr"] += int(b0_ok and not r_ok)
            masked = dict(ctx)
            masked["scene_type"] = ""
            payload = dict(ctx.get("payload") or {})
            payload["scenario"] = ""
            masked["payload"] = payload
            gpred, _gop = run_trhr(masked, llm_fn)
            gok, grs, gfr = score(gpred, b0_ok, y_dec, y_act, y_n)
            los[scene]["generic"]["n"] += 1
            los[scene]["generic"]["b0"] += int(b0_ok)
            los[scene]["generic"]["exact"] += int(gok)
            los[scene]["generic"]["rs"] += grs
            los[scene]["generic"]["fr"] += gfr

        if sid in set(ids300):
            gold_rows.append(
                {
                    "sample_id": sid,
                    "scenario": scene,
                    "action_type": at,
                    "y_decision": y_dec,
                    "action": compact_act(y_act),
                    "human_decision": None,
                    "human_action": None,
                }
            )
        if i % 2000 == 0:
            print(f"  {i}/{len(samples)}", flush=True)

    out_methods = {}
    for name, a in methods.items():
        out_methods[name] = pack_acc(n, a["exact"], b0_exact_n, a["rs"], a["fr"], a["ops"])
    dump(
        OUT / "repair_baseline_comparison.json",
        {
            "note": (
                "Baselines are evaluation-only and do not change frozen TRHR. "
                "Baseline-1: drop blacklisted helper/event services. "
                "Baseline-2: trigger/condition/quiet only, no ADD. "
                "Baseline-3: device-state REMOVE/entity-fill only; events kept. "
                "Full_Repair is official frozen traces."
            ),
            "N": n,
            "y_used_only_for_evaluation": True,
            "repair_modified": False,
            "methods": out_methods,
            "ranking_by_semantic_success": sorted(
                out_methods, key=lambda k: out_methods[k]["Semantic_Success"], reverse=True
            ),
        },
    )

    id_order = {sid: i for i, sid in enumerate(ids300)}
    gold_rows.sort(key=lambda r: id_order.get(r["sample_id"], 10**9))
    with (OUT / "gold_y_validation_sample.jsonl").open("w", encoding="utf-8") as f:
        for row in gold_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    gold_comp = {
        "n": len(gold_rows),
        "seed": 20260918,
        "source_ids": "gold_y_audit_300_ids.json",
        "y_decision": dict(Counter(r["y_decision"] for r in gold_rows)),
        "action_type": dict(Counter(r["action_type"] for r in gold_rows)),
        "scenario": dict(Counter(r["scenario"] for r in gold_rows)),
    }
    dump(OUT / "gold_y_validation_sample_composition.json", gold_comp)

    los_out = {
        "note": (
            "Leave-one-scenario-out for frozen rules: Full is official Repair on that scene. "
            "generic_rules_only blanks scene_type so scene-named branches (A3 lighting/climate, "
            "ADD lighting/schedule disables, periodic sheets) do not fire. Operators unmodified."
        ),
        "scenes": {},
    }
    for sc in LOS_SCENES:
        fu, ge = los[sc]["full"], los[sc]["generic"]
        nn = fu["n"]
        los_out["scenes"][sc] = {
            "n": nn,
            "B0_Semantic_Success": round(fu["b0"] / max(nn, 1), 4),
            "Full_Repair": pack_acc(nn, fu["exact"], fu["b0"], fu["rs"], fu["fr"], Counter()),
            "generic_rules_only": pack_acc(nn, ge["exact"], ge["b0"], ge["rs"], ge["fr"], Counter()),
            "scene_specific_gain_pp": round(100.0 * (fu["exact"] - ge["exact"]) / max(nn, 1), 2),
        }
        for key in ("Full_Repair", "generic_rules_only"):
            los_out["scenes"][sc][key].pop("operators", None)
    dump(OUT / "cross_scenario_generalization.json", los_out)

    dump(
        OUT / "repair_failure_boundary.json",
        {
            "note": "Primary bucket for each inexact Repair row vs Gold (N should be 4432). Frozen traces.",
            "N_inexact": fail_n,
            "buckets": {
                k: {
                    "n": bound[k],
                    "rate": round(bound[k] / max(fail_n, 1), 4),
                    "error_types": dict(bound_et[k]),
                }
                for k in ["insufficient_evidence", "ambiguity", "unsupported_behavior_type", "incorrect_inference"]
            },
        },
    )
    print(json.dumps({
        "baselines_ss": {k: v["Semantic_Success"] for k, v in out_methods.items()},
        "fail_n": fail_n,
        "bound": dict(bound),
        "gold_n": len(gold_rows),
        "los_full": {sc: los_out["scenes"][sc]["Full_Repair"]["Semantic_Success"] for sc in LOS_SCENES},
        "los_generic": {sc: los_out["scenes"][sc]["generic_rules_only"]["Semantic_Success"] for sc in LOS_SCENES},
    }, indent=2))

if __name__ == "__main__":
    main()
