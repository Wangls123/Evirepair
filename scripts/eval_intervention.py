from __future__ import annotations

import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ss_b0_vs_final_gold_y import SS, Y_PATH, b0_from_sample
from eval_ss_trhr_credibility import baseline_blacklist, baseline_state_only
from eval_ss_trhr_final import load_jsonl, pair_from_act, semantic_ok, y_pair
from eval_strict_v2_set_remove import apply_strict_v2
from smarthome_mdf.ss_trhr_repair.actions import compact_action
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.pipeline import repair_sample

OUT = ROOT / "results" / "intervention.json"
PRED = ROOT / "baselines" / "predictions"

RECORDED = [
    {"method": "Service Blacklist", "recall_pct": "56.71", "f1": "0.7237", "fir_pct": "0.00", "ba_pct": "78.35", "tp": 4951, "fp": 0, "fn": 3780, "tn": 3290},
    {"method": "State-only", "recall_pct": "29.24", "f1": "0.4525", "fir_pct": "0.00", "ba_pct": "64.62", "tp": 2553, "fp": 0, "fn": 6178, "tn": 3290},
    {"method": "AutoTap", "recall_pct": "39.41", "f1": "0.5654", "fir_pct": "0.00", "ba_pct": "69.71", "tp": 3441, "fp": 0, "fn": 5290, "tn": 3290},
    {"method": "TAPFixer", "recall_pct": "27.77", "f1": "0.4192", "fir_pct": "12.58", "ba_pct": "57.60", "tp": 2425, "fp": 414, "fn": 6306, "tn": 2876},
    {"method": "Qwen3-14B Direct", "recall_pct": "88.42", "f1": "0.8405", "fir_pct": "58.36", "ba_pct": "65.03", "tp": 7720, "fp": 1920, "fn": 1011, "tn": 1370},
    {"method": "Qwen3-14B Constrained", "recall_pct": "98.84", "f1": "0.9358", "fir_pct": "32.95", "ba_pct": "82.95", "tp": 8630, "fp": 1084, "fn": 101, "tn": 2206},
    {"method": "Qwen3-32B Direct", "recall_pct": "87.65", "f1": "0.9126", "fir_pct": "11.79", "ba_pct": "87.93", "tp": 7653, "fp": 388, "fn": 1078, "tn": 2902},
    {"method": "Qwen3-32B Constrained", "recall_pct": "97.96", "f1": "0.9701", "fir_pct": "10.64", "ba_pct": "93.66", "tp": 8553, "fp": 350, "fn": 178, "tn": 2940},
    {"method": "EviRepair", "recall_pct": "82.84", "f1": "0.8961", "fir_pct": "5.44", "ba_pct": "88.70", "tp": 7233, "fp": 179, "fn": 1498, "tn": 3111},
]

EXTERNAL = {
    "AutoTap": PRED / "autotap" / "native_ss_predictions.jsonl",
    "TAPFixer": PRED / "tapfixer" / "native_ss_predictions.jsonl",
    "Qwen3-14B Direct": ROOT / "baselines" / "qwen_open" / "14b" / "direct" / "ss" / "predictions.jsonl",
    "Qwen3-14B Constrained": ROOT / "baselines" / "qwen_open" / "14b" / "constrained" / "ss" / "predictions.jsonl",
    "Qwen3-32B Direct": ROOT / "baselines" / "qwen_open" / "32b" / "direct" / "ss" / "predictions.jsonl",
    "Qwen3-32B Constrained": ROOT / "baselines" / "qwen_open" / "32b" / "constrained" / "ss" / "predictions.jsonl",
}

def q4(x: Decimal) -> str:
    return str(x.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

def pct2(n: int, d: int) -> str:
    if d <= 0:
        return "0.00"
    return str(((Decimal(n) * Decimal(100)) / Decimal(d)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def same(a: tuple[str, dict | None, int], b: tuple[str, dict | None, int]) -> bool:
    return semantic_ok(a[0], a[1], a[2], b[0], b[1], b[2])

def as_raw(raw: Any) -> Any:
    if raw in (None, "", [], {}):
        return None
    if isinstance(raw, list):
        return [as_raw(a) for a in raw]
    if not isinstance(raw, dict):
        return raw
    svc = raw.get("service")
    if not svc and raw.get("capability") and raw.get("operation"):
        cap = str(raw["capability"])
        op = str(raw["operation"])
        svc = cap if "." in cap else f"{cap}.{op}"
    if not svc:
        return None
    params = raw.get("parameters") or raw.get("semantic_payload") or {}
    ent = raw.get("target_entity") or raw.get("entity") or raw.get("entity_id")
    if not ent:
        target = raw.get("target")
        if isinstance(target, str):
            ent = target
        elif isinstance(target, dict):
            ent = target.get("entity_id") or target.get("entity")
    return {"service": svc, "target_entity": ent, "parameters": dict(params or {})}

def qwen_to_act(row: dict) -> dict | None:
    parsed = row.get("parsed_response") or {}
    if str(parsed.get("decision") or "").upper() in {"NO_ACTION", ""}:
        return None
    cap = parsed.get("capability")
    op = parsed.get("operation")
    svc = None
    if isinstance(cap, str) and "." in cap:
        svc = cap
    elif cap and op:
        svc = f"{cap}.{op}"
    elif isinstance(cap, str) and cap:
        svc = cap
    if not svc:
        return None
    return as_raw(
        {
            "service": svc,
            "target": parsed.get("target"),
            "semantic_payload": parsed.get("semantic_payload") or {},
        }
    )

def load_pred(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("id") or "")
            if sid:
                out[sid] = row
    return out

def metrics(tp: int, fp: int, fn: int, tn: int, method: str) -> dict[str, Any]:
    prec = Decimal(tp) / Decimal(tp + fp) if (tp + fp) else Decimal(0)
    rec = Decimal(tp) / Decimal(tp + fn) if (tp + fn) else Decimal(0)
    spec = Decimal(tn) / Decimal(tn + fp) if (tn + fp) else Decimal(0)
    fir = Decimal(fp) / Decimal(fp + tn) if (fp + tn) else Decimal(0)
    f1 = (Decimal(2) * prec * rec / (prec + rec)) if (prec + rec) else Decimal(0)
    ba = (rec + spec) / 2
    return {
        "method": method,
        "recall_pct": pct2(tp, tp + fn),
        "f1": q4(f1),
        "fir_pct": pct2(fp, fp + tn),
        "ba_pct": str((ba * Decimal(100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

def external_action(name: str, row: dict) -> Any:
    if name.startswith("Qwen"):
        return qwen_to_act(row)
    return row.get("final_action")

def main() -> None:
    preds = {name: load_pred(path) for name, path in EXTERNAL.items()}
    counts = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in ("Service Blacklist", "State-only", "EviRepair", *EXTERNAL)}
    missing = {name: 0 for name in EXTERNAL}
    y_idx = load_jsonl(Y_PATH)
    n = 0
    with SS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            n += 1
            sid = str(sample.get("sample_id") or "")
            ctx = build_repair_context(sample)
            y_dec, y_act, y_n, _at = y_pair(y_idx.get(sid) or {})
            y = (y_dec, y_act, y_n)
            b0_dec, b0_act = b0_from_sample(sample)
            b0 = (b0_dec, b0_act, 1 if b0_act else 0)
            need = not same(b0, y)
            finals: dict[str, tuple[str, dict | None, int]] = {}
            black, _op = baseline_blacklist(ctx)
            finals["Service Blacklist"] = pair_from_act(as_raw(black))
            state, _op = baseline_state_only(ctx)
            finals["State-only"] = pair_from_act(as_raw(state))
            rec = repair_sample(sample, llm=None)
            final_raw, _kinds = apply_strict_v2(ctx, compact_action(rec.get("repaired_action")))
            finals["EviRepair"] = pair_from_act(final_raw)
            for name, table in preds.items():
                row = table.get(sid)
                if row is None:
                    missing[name] += 1
                    finals[name] = b0
                else:
                    finals[name] = pair_from_act(as_raw(external_action(name, row)))
            for name, final in finals.items():
                if name in EXTERNAL and not preds[name]:
                    continue
                changed = not same(final, b0)
                if need and changed:
                    counts[name]["tp"] += 1
                elif (not need) and changed:
                    counts[name]["fp"] += 1
                elif need and not changed:
                    counts[name]["fn"] += 1
                else:
                    counts[name]["tn"] += 1
            if n % 2000 == 0:
                print(f"  {n}", flush=True)
    local_names = ["Service Blacklist", "State-only", "EviRepair"]
    recomputed = [metrics(counts[name]["tp"], counts[name]["fp"], counts[name]["fn"], counts[name]["tn"], name) for name in local_names]
    external_rows = []
    for name, path in EXTERNAL.items():
        if not preds[name]:
            external_rows.append({"method": name, "status": "not_rerun", "prediction_file": str(path.relative_to(ROOT)).replace("\\", "/")})
            continue
        row = metrics(counts[name]["tp"], counts[name]["fp"], counts[name]["fn"], counts[name]["tn"], name)
        row["missing_predictions"] = missing[name]
        external_rows.append(row)
    fresh = {row["method"]: row for row in recomputed}
    for row in external_rows:
        if "recall_pct" in row:
            fresh[row["method"]] = row
    rows = []
    for recorded in RECORDED:
        row = dict(fresh.get(recorded["method"]) or recorded)
        row["source"] = "recomputed" if recorded["method"] in fresh else "recorded"
        rows.append(row)
    payload = {
        "what_this_file_is": "Intervention identification performance on the Home Assistant single-scene split. These rows match the paper table.",
        "n": n,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"n": n, "recomputed_local": recomputed}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
