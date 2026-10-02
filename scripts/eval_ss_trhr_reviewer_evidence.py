from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ss_b0_vs_final_gold_y import (
    SCENARIO_MAP,
    SS,
    Y_PATH,
    _pct,
    _rate,
    b0_from_sample,
    compare_actions,
)
from eval_ss_trhr_final import (
    load_jsonl,
    pair_from_act,
    semantic_ok,
    y_pair,
)
from smarthome_mdf.multi_action_frozen_v1.y_llm_config import get_y_llm_config, y_llm_configured
from smarthome_mdf.ss_trhr_repair.actions import compact_action
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.modify import _collect_entities, _unique_domain_entity
from smarthome_mdf.ss_trhr_repair.pipeline import _illegal
from smarthome_mdf.ss_trhr_repair.remove import run_remove

OUT = ROOT / "runs" / "ss_trhr_repair"
TRACE = OUT / "repair_execution_trace.jsonl"
SEED = 20260918

NAIVE_EVENT = {
    "system_log.write",
    "google_sheets.append_sheet",
    "logbook.log",
}
NAIVE_HELPER = {"input_boolean.turn_off"}

SERVICE_BLACKLIST = NAIVE_EVENT | NAIVE_HELPER | {
    "notify.mobile_app",
    "notify.notify",
    "persistent_notification.create",
    "input_boolean.turn_on",
    "input_text.set_value",
    "schedule.activate",
    "scene.turn_on",
    "scene.turn_off",
}

LLM_ONLY_SYSTEM = """You repair one Home Assistant B0 action. You cannot see Gold labels.
Choose exactly one operator: KEEP, REMOVE, MODIFY, or ADD.
REMOVE if the current B0 action is unnecessary given observation/conditions.
MODIFY if B0 should still run but service or target_entity is wrong; pick from candidate_services.
ADD if B0 is empty and one candidate service is necessary now.
KEEP to leave B0 unchanged.
Return JSON only:
{"operator":"KEEP|REMOVE|MODIFY|ADD","service":null,"target_entity":null}
Do not invent a service outside candidate_services.
"""

GOLD_RELABEL_SYSTEM = """You label one smart-home sample. Output the correct NOW decision.
Return JSON only:
{"decision":"ACTIONS"|"NO_ACTION","service":null,"target_entity":null,"parameters":{}}
ACTIONS requires exactly one service. NO_ACTION requires service null.
Use observation, trigger, condition, and declared_effect_service.
Do not copy B0. Do not mention B0 quality.
"""

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)

def score_pred(pred_raw, b0_dec, b0_act, y_dec, y_act, y_n) -> dict[str, bool]:
    p_dec, p_act, p_n = pair_from_act(pred_raw)
    b0_n = 1 if b0_act else 0
    b0_ok = semantic_ok(b0_dec, b0_act, b0_n, y_dec, y_act, y_n)
    p_ok = semantic_ok(p_dec, p_act, p_n, y_dec, y_act, y_n)
    return {
        "b0_ok": b0_ok,
        "ok": p_ok,
        "rs": (not b0_ok) and p_ok,
        "fr": b0_ok and (not p_ok),
        "p_dec": p_dec,
        "p_act": p_act,
    }

def aggregate(n: int, b0_exact: int, exact_n: int, rs: int, fr: int) -> dict[str, Any]:
    return {
        "N": n,
        "Semantic_Success": _rate(exact_n, n),
        "Semantic_Success_count": exact_n,
        "Repair_Success": _rate(rs, n - b0_exact),
        "Repair_Success_count": rs,
        "False_Repair": _rate(fr, n),
        "False_Repair_count": fr,
        "Preservation_Rate": _rate(b0_exact - fr, b0_exact),
        "Gain": round(_rate(exact_n, n) - _rate(b0_exact, n), 4),
        "Gain_pp": round(100 * (_rate(exact_n, n) - _rate(b0_exact, n)), 2),
    }

def naive_repair(b0_act: dict | None, ctx: dict[str, Any]) -> tuple[Any, str]:
    if not b0_act:
        return None, "KEEP"
    svc = str(b0_act.get("service") or "")
    if svc in NAIVE_EVENT:
        return None, "REMOVE"
    if svc in NAIVE_HELPER:
        return None, "REMOVE"
    act = compact_action(b0_act)
    if act and not act.get("target_entity"):
        domain = svc.split(".", 1)[0] if "." in svc else ""
        if domain in {"climate", "light", "switch", "cover", "fan"}:
            ent = _unique_domain_entity(_collect_entities(ctx), domain)
            if ent:
                act["target_entity"] = ent
                return act, "MODIFY"
    return act, "KEEP"

def r_service_only(b0_act: dict | None) -> tuple[Any, str]:
    if not b0_act:
        return None, "KEEP"
    svc = str(b0_act.get("service") or "")
    if svc in SERVICE_BLACKLIST or svc.startswith("input_boolean.") or svc.startswith("notify."):
        return None, "REMOVE"
    return compact_action(b0_act), "KEEP"

def r_rule_only(b0_act: dict | None) -> tuple[Any, str]:
    if not b0_act:
        return None, "KEEP"
    svc = str(b0_act.get("service") or "")
    ent = b0_act.get("entity") or b0_act.get("target_entity")
    if svc in SERVICE_BLACKLIST or svc.startswith("input_boolean.") or svc.startswith("notify."):
        return None, "REMOVE"
    if svc.startswith(("system_log.", "logbook.", "google_sheets.")):
        return None, "REMOVE"
    if (ent in (None, "")) and svc.startswith(("input_boolean.", "input_text.", "schedule.", "scene.")):
        return None, "REMOVE"
    return compact_action(b0_act), "KEEP"

def llm_chat(system: str, user_obj: dict[str, Any], max_tokens: int = 192) -> dict[str, Any]:
    if not y_llm_configured():
        return {"error": "LLM_NOT_CONFIGURED"}
    cfg = get_y_llm_config()
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.1,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        f"{cfg['base_url']}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"error": "not_object", "raw": text[:200]}
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP_{exc.code}"}
    except Exception as exc:
        return {"error": type(exc).__name__}

def compact_obs(payload: dict[str, Any]) -> dict[str, Any]:
    obs = dict(payload.get("observation") or {})
    keep = {}
    for k in (
        "illuminance_lux",
        "illuminance",
        "motion_state",
        "occupancy",
        "window_state",
        "door_state",
        "hvac_action",
        "current_power_w",
        "binary_state",
    ):
        if obs.get(k) not in (None, "", [], {}):
            keep[k] = obs.get(k)
    return keep

def mcnemar(b: int, c: int) -> dict[str, Any]:
    n_disc = b + c
    if n_disc == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n_disc
        p = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "test": "McNemar_midp_continuity_corrected",
        "b_B0_wrong_Repair_right": b,
        "c_B0_right_Repair_wrong": c,
        "chi2": round(chi2, 4),
        "p_value": p,
        "significant_0.001": bool(p < 0.001),
        "note": "Paired binary Semantic Success on the same 12021 samples.",
    }

def bootstrap_ci(flags: list[int], rng: random.Random, n_boot: int = 5000) -> dict[str, Any]:
    n = len(flags)
    stats = []
    for _ in range(n_boot):
        acc = 0
        for _i in range(n):
            acc += flags[rng.randrange(n)]
        stats.append(acc / n)
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[min(int(0.975 * n_boot), n_boot - 1)]
    mean = sum(stats) / n_boot
    return {
        "n_boot": n_boot,
        "mean": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "ci95_pct": [f"{100 * lo:.2f}%", f"{100 * hi:.2f}%"],
    }

def cohen_kappa(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    n = len(y_true)
    agree = sum(a == b for a, b in zip(y_true, y_pred))
    p0 = agree / max(n, 1)
    labels = sorted(set(y_true) | set(y_pred))
    pe = 0.0
    for lab in labels:
        pe += (y_true.count(lab) / n) * (y_pred.count(lab) / n)
    if abs(1 - pe) < 1e-12:
        k = None
        note = "Kappa undefined (p_e=1)."
    else:
        k = (p0 - pe) / (1 - pe)
        note = "Cohen kappa between frozen Gold Y and independent LLM re-label (not human-human)."
    return {"n": n, "raw_agreement": round(p0, 4), "p_e": round(pe, 4), "cohens_kappa": None if k is None else round(k, 4), "note": note}

def run_core() -> dict[str, Any]:
    y_idx = load_jsonl(Y_PATH)
    traces = load_jsonl(TRACE)
    samples: list[dict[str, Any]] = []
    with SS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    n = 0
    b0_exact = 0
    full_exact = 0
    b_fix = 0
    c_break = 0
    naive_exact = rs_n = fr_n = 0
    naive_ops = Counter()
    r_vars = {
        "R_service_only": {"exact": 0, "rs": 0, "fr": 0, "fired": 0, "success": 0, "fail": 0, "over_delete": 0, "correct_idle": 0},
        "R_rule_only": {"exact": 0, "rs": 0, "fr": 0, "fired": 0, "success": 0, "fail": 0, "over_delete": 0, "correct_idle": 0},
        "R_full": {"exact": 0, "rs": 0, "fr": 0, "fired": 0, "success": 0, "fail": 0, "over_delete": 0, "correct_idle": 0},
    }
    b0_flags: list[int] = []
    rp_flags: list[int] = []
    gain_pairs: list[tuple[int, int]] = []
    llm_stub: dict[str, bool] = {}
    gold_struct = Counter()
    strata: dict[tuple, list[str]] = defaultdict(list)
    rows_meta: list[dict[str, Any]] = []

    print("scoring naive + REMOVE ablations + stats...", flush=True)
    for i, s in enumerate(samples, 1):
        sid = str(s.get("sample_id") or "")
        yrec = y_idx.get(sid)
        tr = traces.get(sid)
        if not yrec or not tr:
            continue
        n += 1
        y_dec, y_act, y_n, at = y_pair(yrec)
        b0_dec, b0_act = b0_from_sample(s)
        fy = yrec.get("formal_y") or {}
        raw_dec = str(fy.get("decision") or "")
        acts = list(fy.get("semantic_actions") or [])
        if raw_dec in {"ACTION", "ACTIONS"}:
            gold_struct["ACTIONS"] += 1
            if len(acts) == 1:
                gold_struct["ACTIONS_n1"] += 1
            else:
                gold_struct["ACTIONS_bad_count"] += 1
        elif raw_dec == "NO_ACTION":
            gold_struct["NO_ACTION"] += 1
            if len(acts) == 0:
                gold_struct["NO_ACTION_empty"] += 1
            else:
                gold_struct["NO_ACTION_nonempty"] += 1
        scene = SCENARIO_MAP.get(str(s.get("scene_type") or ""), "other")
        strata[(scene, y_dec, "STATE" if at == "STATE_ACTION" else "EVENT")].append(sid)

        b0_n = 1 if b0_act else 0
        b0_ok = semantic_ok(b0_dec, b0_act, b0_n, y_dec, y_act, y_n)
        r_dec, r_act, r_n = pair_from_act(tr.get("repaired_action"))
        r_ok = semantic_ok(r_dec, r_act, r_n, y_dec, y_act, y_n)
        if b0_ok:
            b0_exact += 1
        if r_ok:
            full_exact += 1
        if (not b0_ok) and r_ok:
            b_fix += 1
        if b0_ok and (not r_ok):
            c_break += 1
        b0_flags.append(int(b0_ok))
        rp_flags.append(int(r_ok))
        gain_pairs.append((int(b0_ok), int(r_ok)))
        if str(tr.get("reason") or "") == "A3_LLM_UNNECESSARY":
            llm_stub[sid] = False

        ctx = build_repair_context(s)
        original = compact_action(ctx.get("b0_action"))
        nraw, nop = naive_repair(b0_act, ctx)
        naive_ops[nop] += 1
        sc = score_pred(nraw, b0_dec, b0_act, y_dec, y_act, y_n)
        if sc["ok"]:
            naive_exact += 1
        if sc["rs"]:
            rs_n += 1
        if sc["fr"]:
            fr_n += 1

        variants = {
            "R_service_only": r_service_only(original),
            "R_rule_only": r_rule_only(original),
        }
        llm_fn = None
        if llm_stub.get(sid) is False:
            def llm_fn(_p, _v=False):
                return {"necessary": False, "source": "trace_stub"}
        if original:
            rr = run_remove(ctx, original, llm=llm_fn)
            if rr.get("changed"):
                variants["R_full"] = (rr.get("action_out"), "REMOVE")
            else:
                variants["R_full"] = (original, "KEEP")
        else:
            variants["R_full"] = (None, "KEEP")

        for name, (praw, opu) in variants.items():
            st = r_vars[name]
            sc2 = score_pred(praw, b0_dec, b0_act, y_dec, y_act, y_n)
            if sc2["ok"]:
                st["exact"] += 1
            if sc2["rs"]:
                st["rs"] += 1
            if sc2["fr"]:
                st["fr"] += 1
            if opu == "REMOVE":
                st["fired"] += 1
                if (not b0_ok) and sc2["ok"]:
                    st["success"] += 1
                elif (not b0_ok) and (not sc2["ok"]):
                    st["fail"] += 1
                if y_dec == "ACTION" and sc2["p_dec"] == "NO_ACTION":
                    st["over_delete"] += 1
                if y_dec == "NO_ACTION" and sc2["p_dec"] == "NO_ACTION" and b0_dec == "ACTION":
                    st["correct_idle"] += 1

        if i % 2000 == 0:
            print(f"  {i}/{len(samples)}", flush=True)

    naive_metrics = aggregate(n, b0_exact, naive_exact, rs_n, fr_n)
    naive_metrics["operators"] = dict(naive_ops)
    naive_metrics["rules"] = {
        "remove_event_services": sorted(NAIVE_EVENT),
        "remove_helper_services": sorted(NAIVE_HELPER),
        "entity_fill": "unique domain entity for climate/light/switch/cover/fan when target is null",
        "no_ADD": True,
        "y_blind": True,
    }
    dump(OUT / "naive_rule_baseline_results.json", naive_metrics)

    remove_ablation = {
        "note": (
            "Each variant is REMOVE-only (no MODIFY/ADD). "
            "R-service-only is a service blacklist. R-rule-only adds entity emptiness on helpers. "
            "R-full is the frozen TRHR REMOVE module (observation + contract + LLM stub). "
            "False Repair can be near zero because B0 ACTION never exact-matches Gold ACTION; "
            "over_delete (Gold ACTION emptied) is the discriminative failure mode."
        ),
        "variants": {},
    }
    for name, st in r_vars.items():
        block = aggregate(n, b0_exact, st["exact"], st["rs"], st["fr"])
        block.update(
            {
                "REMOVE_fired": st["fired"],
                "REMOVE_success": st["success"],
                "REMOVE_fail": st["fail"],
                "REMOVE_success_rate": _rate(st["success"], st["fired"]),
                "False_Repair_count": st["fr"],
                "over_delete_gold_ACTION": st["over_delete"],
                "correct_idle_from_B0_ACTION": st["correct_idle"],
            }
        )
        remove_ablation["variants"][name] = block
    dump(OUT / "remove_ablation_analysis.json", remove_ablation)

    mc = mcnemar(b_fix, c_break)
    mc["N"] = n
    dump(OUT / "mcnemar_test.json", mc)

    rng = random.Random(SEED)
    b0_ci = bootstrap_ci(b0_flags, rng)
    rng = random.Random(SEED)
    rp_ci = bootstrap_ci(rp_flags, rng)
    rng = random.Random(SEED)
    gains = []
    n_boot = 5000
    for _ in range(n_boot):
        b_acc = r_acc = 0
        for _i in range(n):
            j = rng.randrange(n)
            b_acc += gain_pairs[j][0]
            r_acc += gain_pairs[j][1]
        gains.append((r_acc - b_acc) / n)
    gains.sort()
    boot = {
        "n": n,
        "n_boot": n_boot,
        "seed": SEED,
        "B0_Semantic_Success": {"point": _rate(b0_exact, n), **{k: b0_ci[k] for k in ("mean", "ci95", "ci95_pct")}},
        "Repair_Semantic_Success": {"point": _rate(full_exact, n), **{k: rp_ci[k] for k in ("mean", "ci95", "ci95_pct")}},
        "Gain": {
            "point": round(_rate(full_exact, n) - _rate(b0_exact, n), 4),
            "mean": round(sum(gains) / n_boot, 4),
            "ci95": [round(gains[int(0.025 * n_boot)], 4), round(gains[min(int(0.975 * n_boot), n_boot - 1)], 4)],
        },
    }
    dump(OUT / "bootstrap_confidence_interval.json", boot)

    rng = random.Random(SEED)
    picked: list[str] = []
    keys = sorted(strata.keys())

    for k in keys:
        ids = list(strata[k])
        rng.shuffle(ids)
        take = min(2, len(ids))
        picked.extend(ids[:take])
        strata[k] = ids[take:]
    remain = 300 - len(picked)
    pool = []
    for k, ids in strata.items():
        pool.extend(ids)
    rng.shuffle(pool)
    for sid in pool:
        if len(picked) >= 300:
            break
        picked.append(sid)
    picked = picked[:300]
    dump(OUT / "gold_y_audit_300_ids.json", {"seed": SEED, "n": len(picked), "sample_ids": picked})

    print(
        json.dumps(
            {
                "n": n,
                "naive": naive_metrics["Semantic_Success"],
                "R_service": remove_ablation["variants"]["R_service_only"]["Semantic_Success"],
                "R_rule": remove_ablation["variants"]["R_rule_only"]["Semantic_Success"],
                "R_full": remove_ablation["variants"]["R_full"]["Semantic_Success"],
                "mcnemar_p": mc["p_value"],
                "gold_struct": dict(gold_struct),
                "audit300": len(picked),
            },
            indent=2,
        ),
        flush=True,
    )
    return {
        "n": n,
        "b0_exact": b0_exact,
        "full_exact": full_exact,
        "naive": naive_metrics,
        "remove": remove_ablation,
        "mcnemar": mc,
        "bootstrap": boot,
        "picked": picked,
        "gold_struct": dict(gold_struct),
        "samples": samples,
        "y_idx": y_idx,
        "traces": traces,
        "b_fix": b_fix,
        "c_break": c_break,
    }

def apply_llm_only(sample: dict[str, Any], ans: dict[str, Any]) -> tuple[Any, str]:
    ctx = build_repair_context(sample)
    original = compact_action(ctx.get("b0_action"))
    op = str(ans.get("operator") or "KEEP").upper()
    if op not in {"KEEP", "REMOVE", "MODIFY", "ADD"}:
        op = "KEEP"
    svc = ans.get("service")
    ent = ans.get("target_entity")
    allowed = {str(s) for s in (ctx.get("candidate_services") or []) if s}
    declared = str(ctx.get("declared_effect_service") or "")
    if declared:
        allowed.add(declared)
    if original and original.get("service"):
        allowed.add(str(original["service"]))
    if op == "REMOVE":
        return None, "REMOVE"
    if op == "KEEP":
        return original, "KEEP"
    if op == "MODIFY" and original:
        out = dict(original)
        if svc and str(svc) in allowed:
            out["service"] = str(svc)
        if ent:
            out["target_entity"] = str(ent)
        elif not out.get("target_entity"):
            domain = str(out.get("service") or "").split(".", 1)[0]
            fill = _unique_domain_entity(_collect_entities(ctx), domain)
            if fill:
                out["target_entity"] = fill
        if _illegal(out, ctx, original):
            return original, "KEEP"
        return compact_action(out), "MODIFY"
    if op == "ADD" and not original:
        use = str(svc or declared or "")
        if use not in allowed:
            return None, "KEEP"
        domain = use.split(".", 1)[0]
        fill = ent or _unique_domain_entity(_collect_entities(ctx), domain)
        act = {"service": use, "target_entity": fill, "parameters": {}, "component_origin": "c0", "execution_order": 1}
        if _illegal(act, ctx, None):
            return None, "KEEP"
        return compact_action(act), "ADD"
    return original, "KEEP"

def run_llm_only(samples: list[dict[str, Any]], workers: int = 8) -> None:
    ckpt = OUT / "llm_only_baseline.jsonl"
    done: set[str] = set()
    if ckpt.exists():
        with ckpt.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add(str(rec.get("sample_id") or ""))
    todo = [s for s in samples if str(s.get("sample_id") or "") not in done]
    print(f"LLM-only remaining {len(todo)} (done {len(done)})", flush=True)

    def one(s: dict[str, Any]) -> dict[str, Any]:
        ctx = build_repair_context(s)
        payload = ctx.get("payload") or {}
        original = compact_action(ctx.get("b0_action"))
        user = {
            "scene": ctx.get("scene_type"),
            "action_type": ctx.get("action_type"),
            "declared_effect_service": ctx.get("declared_effect_service"),
            "candidate_services": list(ctx.get("candidate_services") or [])[:8],
            "b0_action": original,
            "observation": compact_obs(payload),
            "condition": (payload.get("condition") or [])[:6],
            "trigger": (payload.get("trigger") or [])[:4],
        }
        ans = llm_chat(LLM_ONLY_SYSTEM, user)
        pred, op = apply_llm_only(s, ans if "error" not in ans else {"operator": "KEEP"})
        return {
            "sample_id": ctx.get("sample_id"),
            "operator": op,
            "repaired_action": pred,
            "llm_error": ans.get("error"),
            "llm_operator": ans.get("operator"),
        }

    with ckpt.open("a", encoding="utf-8") as out:
        if workers <= 1:
            for i, s in enumerate(todo, 1):
                rec = one(s)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if i % 50 == 0:
                    print(f"  llm-only {i}/{len(todo)}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(one, s): s for s in todo}
                done_n = 0
                for fut in as_completed(futs):
                    rec = fut.result()
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    done_n += 1
                    if done_n % 50 == 0:
                        print(f"  llm-only {done_n}/{len(todo)}", flush=True)

def score_llm_only(samples, y_idx) -> dict[str, Any]:
    idx = load_jsonl(OUT / "llm_only_baseline.jsonl")
    n = exact = rs = fr = 0
    b0_exact = 0
    ops = Counter()
    errors = 0
    for s in samples:
        sid = str(s.get("sample_id") or "")
        rec = idx.get(sid)
        yrec = y_idx[sid]
        y_dec, y_act, y_n, _at = y_pair(yrec)
        b0_dec, b0_act = b0_from_sample(s)
        b0_n = 1 if b0_act else 0
        if semantic_ok(b0_dec, b0_act, b0_n, y_dec, y_act, y_n):
            b0_exact += 1
        if not rec:
            continue
        n += 1
        if rec.get("llm_error"):
            errors += 1
        sc = score_pred(rec.get("repaired_action"), b0_dec, b0_act, y_dec, y_act, y_n)
        if sc["ok"]:
            exact += 1
        if sc["rs"]:
            rs += 1
        if sc["fr"]:
            fr += 1
        ops[str(rec.get("operator") or "KEEP")] += 1
    out = aggregate(n, b0_exact, exact, rs, fr)
    out["operators"] = dict(ops)
    out["llm_errors"] = errors
    out["covered"] = n
    out["y_blind"] = True
    out["uses_trhr_pipeline"] = False
    dump(OUT / "llm_only_baseline_results.json", out)
    return out

def run_gold_relabel(samples, y_idx, ids: list[str]) -> dict[str, Any]:
    want = set(ids)
    by_id = {str(s.get("sample_id")): s for s in samples}
    ckpt = OUT / "gold_y_relabel_300.jsonl"
    done: set[str] = set()
    if ckpt.exists():
        with ckpt.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line).get("sample_id"))
    todo = [sid for sid in ids if sid not in done]
    print(f"Gold re-label remaining {len(todo)}", flush=True)

    def one(sid: str) -> dict[str, Any]:
        s = by_id[sid]
        ctx = build_repair_context(s)
        payload = ctx.get("payload") or {}
        user = {
            "scene": ctx.get("scene_type"),
            "action_type": ctx.get("action_type"),
            "declared_effect_service": ctx.get("declared_effect_service"),
            "candidate_services": list(ctx.get("candidate_services") or [])[:8],
            "observation": compact_obs(payload),
            "condition": (payload.get("condition") or [])[:6],
            "trigger": (payload.get("trigger") or [])[:4],
        }
        ans = llm_chat(GOLD_RELABEL_SYSTEM, user)
        y_dec, y_act, y_n, at = y_pair(y_idx[sid])
        pred_dec = str(ans.get("decision") or "").upper()
        if pred_dec in {"ACTION", "ACTIONS"}:
            pred_dec = "ACTION"
        else:
            pred_dec = "NO_ACTION"
        return {
            "sample_id": sid,
            "gold_decision": y_dec,
            "pred_decision": pred_dec,
            "action_type": at,
            "scene": SCENARIO_MAP.get(str(s.get("scene_type") or ""), ""),
            "agree_decision": pred_dec == y_dec,
            "llm_error": ans.get("error"),
        }

    with ckpt.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(one, sid) for sid in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if i % 25 == 0:
                    print(f"  gold-relabel {i}/{len(todo)}", flush=True)

    recs = list(load_jsonl(ckpt).values())
    gold = [r["gold_decision"] for r in recs]
    pred = [r["pred_decision"] for r in recs]
    kap = cohen_kappa(gold, pred)
    by_scene = defaultdict(Counter)
    disagree = []
    for r in recs:
        by_scene[r.get("scene") or "other"]["n"] += 1
        by_scene[r.get("scene") or "other"]["agree"] += int(r.get("agree_decision"))
        if not r.get("agree_decision"):
            disagree.append({"sample_id": r["sample_id"], "gold": r["gold_decision"], "pred": r["pred_decision"], "scene": r.get("scene"), "action_type": r.get("action_type")})
    return {"kappa": kap, "by_scene": {k: dict(v) for k, v in by_scene.items()}, "disagree": disagree[:40], "n": len(recs)}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", action="store_true")
    ap.add_argument("--llm-only", action="store_true")
    ap.add_argument("--gold-relabel", action="store_true")
    ap.add_argument("--score-llm", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not (args.core or args.llm_only or args.gold_relabel or args.score_llm):
        args.core = True

    samples: list[dict[str, Any]] = []
    with SS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    y_idx = load_jsonl(Y_PATH)

    if args.core:
        run_core()
    if args.llm_only:
        run_llm_only(samples, workers=args.workers)
        score_llm_only(samples, y_idx)
    if args.score_llm:
        score_llm_only(samples, y_idx)
    if args.gold_relabel:
        ids = json.loads((OUT / "gold_y_audit_300_ids.json").read_text(encoding="utf-8"))["sample_ids"]
        rel = run_gold_relabel(samples, y_idx, ids)
        dump(OUT / "gold_y_relabel_summary.json", rel)

if __name__ == "__main__":
    main()
