from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ma_evidence_attribution import (
    apply_generic_gate,
    attribute_parent,
    compose_parent,
    ownership_for_component,
    toks_from_acts,
)
from eval_ma_evidence_sufficiency import apply_verifier
from eval_ss_b0_vs_final_gold_y import SS, Y_PATH, b0_from_sample
from eval_ss_trhr_final import (
    error_type,
    load_jsonl,
    pair_from_act,
    semantic_ok,
    y_pair,
)
from eval_strict_v2_set_remove import (
    MA,
    TRACE,
    acc_bucket,
    add_bucket,
    apply_strict_v2,
    compact_action,
    finish_bucket,
    ma_y_acts,
    map_repaired,
    set_metrics,
    set_remove_v1,
)
from smarthome_mdf.ss_trhr_repair import METHOD
from smarthome_mdf.ss_trhr_repair.add import run_add
from smarthome_mdf.ss_trhr_repair.context import audit_context, build_repair_context
from smarthome_mdf.ss_trhr_repair.modify import run_modify
from smarthome_mdf.ss_trhr_repair.pipeline import _illegal
from smarthome_mdf.ss_trhr_repair.remove import run_remove

OUT = ROOT / "results" / "evirepair.json"
RECOVERED_AUX_SERVICES = {"notify.mobile_app", "logbook.log"}

def fin_ss(n: int, exact: int, b0_exact: int, rs: int, fr: int) -> dict[str, Any]:
    wrong = n - b0_exact
    return {
        "N": n,
        "EM": round(exact / n, 4) if n else None,
        "EM_count": f"{exact}/{n}",
        "RSR": round(rs / wrong, 4) if wrong else None,
        "RSR_count": f"{rs}/{wrong}",
        "FRR": round(fr / n, 4) if n else None,
        "FRR_count": f"{fr}/{n}",
        "correct_B0": b0_exact,
    }

def slot_repair(ctx: dict[str, Any]) -> dict[str, Any]:

    original = compact_action(ctx.get("b0_action"))
    modules: list[dict[str, Any]] = []
    operator_used = "KEEP"
    repaired = original
    reason = "KEEP_B0"
    confidence = "HIGH"
    fired = None

    def take(mod: dict[str, Any]) -> None:
        nonlocal operator_used, repaired, reason, confidence, fired
        modules.append({k: mod[k] for k in ("operator", "changed", "reason", "confidence") if k in mod})
        if fired is None and not mod.get("changed"):
            reason = str(mod.get("reason") or reason)
            confidence = str(mod.get("confidence") or confidence)
        if mod.get("changed") and fired is None:
            fired = mod
            operator_used = str(mod.get("operator") or "KEEP")
            repaired = compact_action(mod.get("action_out"))
            reason = str(mod.get("reason") or operator_used)
            confidence = str(mod.get("confidence") or "HIGH")

    if original:
        take(run_remove(ctx, original, llm=None))
        if fired is None:
            take(run_modify(ctx, original))
    else:
        take(run_add(ctx, llm=None))
    illegal = _illegal(repaired, ctx, original)
    if illegal:
        repaired = original
        operator_used = "KEEP"
        reason = f"ROLLBACK_{illegal}"
        confidence = "HIGH"
    audit = ctx.get("input_audit") or audit_context(ctx)
    return {
        "operator_used": operator_used,
        "repaired_action": repaired if repaired else None,
        "reason": reason,
        "confidence": confidence,
        "method": METHOD,
        "y_leakage": bool(audit.get("y_leakage")),
        "illegal_rolled_back": bool(illegal),
    }

def act_key(raw: Any) -> str:
    return json.dumps(compact_action(raw), sort_keys=True, ensure_ascii=False, default=str)

def main() -> None:
    t0 = time.perf_counter()
    print("load indexes", flush=True)
    y_idx = load_jsonl(Y_PATH)
    traces = load_jsonl(TRACE) if TRACE.is_file() else {}
    ma_ids: set[str] = set()
    ma_rows: list[dict[str, Any]] = []
    with MA.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ma_rows.append(rec)
            for c in rec.get("components") or []:
                ma_ids.add(str(c.get("single_scene_sample_id") or ""))

    ss_n = b0_exact = slot_exact = slot_rs = slot_fr = 0
    v2_exact = v2_rs = v2_fr = 0
    trace_exact = trace_rs = trace_fr = 0
    trace_match = trace_mismatch = 0
    ops = Counter()
    reasons = Counter()
    fail_type = Counter()
    success_reason = Counter()
    fail_reason = Counter()
    ctx_cache: dict[str, dict[str, Any]] = {}
    fresh: dict[str, dict[str, Any]] = {}

    print("SS repair_sample", flush=True)
    with SS.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sid = str(sample.get("sample_id") or "")
            yrec = y_idx[sid]
            y_dec, y_act, y_n, at = y_pair(yrec)
            b0_dec, b0_act = b0_from_sample(sample)
            b0_n = 1 if b0_act else 0
            b0_ok = semantic_ok(b0_dec, b0_act, b0_n, y_dec, y_act, y_n)
            ctx = build_repair_context(sample)
            rec = slot_repair(ctx)
            if sid in ma_ids:
                fresh[sid] = rec
                ctx_cache[sid] = ctx
            pred = rec.get("repaired_action")
            p_dec, p_act, p_n = pair_from_act(pred)
            ok = semantic_ok(p_dec, p_act, p_n, y_dec, y_act, y_n)
            v2, _kinds = apply_strict_v2(ctx, compact_action(pred))
            v_dec, v_act, v_n = pair_from_act(v2)
            v_ok = semantic_ok(v_dec, v_act, v_n, y_dec, y_act, y_n)
            tr = traces.get(sid) or {}
            tr_pred = tr.get("repaired_action")
            t_dec, t_act, t_n = pair_from_act(tr_pred)
            t_ok = semantic_ok(t_dec, t_act, t_n, y_dec, y_act, y_n) if tr else False
            if tr:
                if act_key(pred) == act_key(tr_pred) and str(rec.get("operator_used")) == str(tr.get("operator_used")):
                    trace_match += 1
                else:
                    trace_mismatch += 1
            ss_n += 1
            b0_exact += int(b0_ok)
            slot_exact += int(ok)
            v2_exact += int(v_ok)
            trace_exact += int(t_ok)
            if not b0_ok:
                slot_rs += int(ok)
                v2_rs += int(v_ok)
                trace_rs += int(t_ok)
            else:
                slot_fr += int(not ok)
                v2_fr += int(not v_ok)
                trace_fr += int(not t_ok)
            op = str(rec.get("operator_used") or "KEEP")
            reason = str(rec.get("reason") or "")
            ops[op] += 1
            reasons[reason] += 1
            if not b0_ok and ok:
                success_reason[reason] += 1
            elif not b0_ok and not ok:
                fail_reason[reason] += 1
                fail_type[error_type(sample, b0_dec, b0_act, y_dec, y_act, at)] += 1
            if i % 500 == 0:
                print(f"  ss {i} em {slot_exact}/{ss_n} match {trace_match} mismatch {trace_mismatch}", flush=True)

    ss_slot = fin_ss(ss_n, slot_exact, b0_exact, slot_rs, slot_fr)
    ss_v2 = fin_ss(ss_n, v2_exact, b0_exact, v2_rs, v2_fr)
    ss_trace = fin_ss(ss_n, trace_exact, b0_exact, trace_rs, trace_fr)

    print("MA", len(ma_rows), flush=True)
    buckets = {
        "b0": acc_bucket(),
        "slot": acc_bucket(),
        "published": acc_bucket(),
        "trace_published": acc_bucket(),
    }
    stage = Counter()
    for n_parent, rec in enumerate(ma_rows, 1):
        comps = list(rec.get("components") or [])
        bind_all = {str(c.get("component_id") or ""): c.get("entity_binding") or {} for c in comps}
        comps_by_id = {str(c.get("component_id") or ""): c for c in comps}
        gold_toks = toks_from_acts(ma_y_acts(rec), bind_all)
        parent_acts = list(((rec.get("formal_b0") or {}).get("parent") or {}).get("semantic_actions") or [])
        b0_toks = toks_from_acts(parent_acts, bind_all)
        b0_m = set_metrics(b0_toks, gold_toks)
        b0_ok = bool(b0_m["set_exact"])
        add_bucket(buckets["b0"], b0_m, b0_ok, b0_ok, b0_m["n_pred"])
        ctxs: dict[str, dict[str, Any]] = {}
        slot_merged: list[dict[str, Any]] = []
        pub_merged: list[dict[str, Any]] = []
        trace_merged: list[dict[str, Any]] = []
        for c in comps:
            cid = str(c.get("component_id") or "")
            sid = str(c.get("single_scene_sample_id") or "")
            if sid not in ctx_cache:
                raise SystemExit(f"missing context for MA component {sid}")
            ctxs[cid] = ctx_cache[sid]
            fr = compact_action((fresh.get(sid) or {}).get("repaired_action"))
            mapped = map_repaired(fr, bind_all.get(cid) or {})
            if mapped:
                mapped["component_origin"] = cid
                slot_merged.append(mapped)
            out, _k = apply_strict_v2(ctxs[cid], fr)
            mapped_v2 = map_repaired(out, bind_all.get(cid) or {})
            if mapped_v2:
                op = str((fresh.get(sid) or {}).get("operator_used") or "")
                recovered_aux = op == "ADD" and str(mapped_v2.get("service") or "") in RECOVERED_AUX_SERVICES
                if not recovered_aux:
                    mapped_v2["component_origin"] = cid
                    pub_merged.append(mapped_v2)
            tr = compact_action((traces.get(sid) or {}).get("repaired_action"))
            tr_out, _k2 = apply_strict_v2(ctxs[cid], tr)
            mapped_tr = map_repaired(tr_out, bind_all.get(cid) or {})
            if mapped_tr:
                mapped_tr["component_origin"] = cid
                trace_merged.append(mapped_tr)
        _atoms, meta = attribute_parent(comps, ctxs)
        owns = [
            ownership_for_component(
                c,
                meta["per_comp_atoms"].get(str(c.get("component_id") or "")) or [],
                meta["contracts"].get(str(c.get("component_id") or "")) or [],
            )
            for c in comps
        ]
        owns_by_id = {o["component_id"]: o for o in owns}
        composition = compose_parent(comps, owns)

        def finish_path(name: str, merged: list[dict[str, Any]]) -> None:
            sr_acts, _ = set_remove_v1(merged, bind_all)
            if len(sr_acts) < len(merged):
                stage[f"{name}_set_remove"] += 1
            d_acts, _ = apply_generic_gate(sr_acts, owns_by_id, composition)
            if len(d_acts) < len(sr_acts):
                stage[f"{name}_gate"] += 1
            e_acts, _ = apply_verifier(d_acts, comps_by_id, owns_by_id, composition)
            if len(e_acts) < len(d_acts):
                stage[f"{name}_verifier"] += 1
            m = set_metrics(toks_from_acts(e_acts, bind_all), gold_toks)
            add_bucket(buckets[name], m, b0_ok, bool(m["set_exact"]), m["n_pred"], fired=len(sr_acts) < len(merged))

        finish_path("slot", slot_merged)
        finish_path("published", pub_merged)
        finish_path("trace_published", trace_merged)
        if n_parent % 400 == 0:
            print(f"  ma {n_parent}", flush=True)

    published_ma = finish_bucket(buckets["published"])
    summary = {
        "method_id": METHOD,
        "llm": False,
        "entry": "repair_sample(llm=None) then apply_strict_v2; multi-action then map_repaired, drop recovered notify.mobile_app and logbook.log adds, set_remove_v1, apply_generic_gate, apply_verifier",
        "single_scene_benchmark": "data/benchmarks/single_scene.jsonl",
        "single_scene_reference": "data/references/single_scene_gold_y.jsonl",
        "multi_action_benchmark": "data/benchmarks/multi_action.jsonl",
        "multi_action_reference": "embedded on each parent and read by ma_y_acts",
        "dataset_sha256": {
            "single_scene": hashlib.sha256(SS.read_bytes()).hexdigest(),
            "single_scene_reference": hashlib.sha256(Y_PATH.read_bytes()).hexdigest(),
            "multi_action": hashlib.sha256(MA.read_bytes()).hexdigest(),
        },
        "single_scene": ss_v2,
        "multi_action": published_ma,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "single_scene": ss_v2,
        "multi_action": published_ma,
        "elapsed_s": summary["elapsed_s"],
        "wrote": str(OUT),
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
