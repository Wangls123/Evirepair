from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ss_b0_vs_final_gold_y import SS, Y_PATH, b0_from_sample
from eval_ss_trhr_final import load_jsonl, pair_from_act, semantic_ok, y_pair
from eval_strict_v2_set_remove import apply_strict_v2
from smarthome_mdf.ss_trhr_repair.actions import compact_action
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.pipeline import repair_sample
from smarthome_mdf.ss_trhr_repair.remove import run_remove

OUT = ROOT / "results" / "behavioral_status.json"
ORDER = ("Justified", "Unsupported", "Incorrect", "Missing")

def same(a: tuple[str, dict | None, int], b: tuple[str, dict | None, int]) -> bool:
    return semantic_ok(a[0], a[1], a[2], b[0], b[1], b[2])

def main() -> None:
    y_idx = load_jsonl(Y_PATH)
    counts = {name: {"cases": 0, "correct": 0} for name in ORDER}
    multi_to_first = 0
    unassigned = 0
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
            flags: list[str] = []
            if not need:
                flags.append("Justified")
            if b0_dec == "NO_ACTION" and y_dec == "ACTION":
                flags.append("Missing")
            elif b0_dec == "ACTION" and y_dec == "ACTION" and need:
                flags.append("Incorrect")
            act = ctx.get("b0_action")
            if act:
                removed = run_remove(ctx, act)
                if str(removed.get("operator") or "") == "REMOVE":
                    flags.append("Unsupported")
            flags = [name for name in ORDER if name in flags]
            if len(flags) > 1:
                multi_to_first += 1
                status = flags[0]
            elif len(flags) == 1:
                status = flags[0]
            else:
                unassigned += 1
                if n % 2000 == 0:
                    print(f"  {n}", flush=True)
                continue
            rec = repair_sample(sample, llm=None)
            op = str(rec.get("operator_used") or "")
            final_raw, _kinds = apply_strict_v2(ctx, compact_action(rec.get("repaired_action")))
            final = pair_from_act(final_raw)
            changed = not same(final, b0)
            if status == "Justified":
                ok = not changed
            elif status == "Unsupported":
                ok = op == "REMOVE"
            elif status == "Incorrect":
                ok = op == "MODIFY"
            else:
                ok = op == "ADD"
            counts[status]["cases"] += 1
            counts[status]["correct"] += int(ok)
            if n % 2000 == 0:
                print(f"  {n}", flush=True)

    rows = []
    for name in ORDER:
        cases = counts[name]["cases"]
        correct = counts[name]["correct"]
        incorrect = cases - correct
        rows.append(
            {
                "category": name,
                "cases": cases,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": round(correct / cases, 4) if cases else None,
            }
        )
    payload = {
        "what_this_file_is": "Decision performance across behavioral assessment categories. Multi-category detections are assigned to the first category, Unsupported.",
        "assignment": "A sample may match more than one category. Those samples are counted only in the first category in Justified, Unsupported, Incorrect, Missing. The 68 Unsupported+Incorrect detections therefore join Unsupported.",
        "unassigned_not_in_table": unassigned,
        "multi_assigned_to_first_category": multi_to_first,
        "categories": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
