from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "baselines"
sys.path.insert(0, str(ROOT))

from baselines.tapfixer.output_adapter import action_to_final, convert_ss_fix

IR = LAB / "converted" / "tapfixer" / "ss_ir.jsonl"
RUNTIME = LAB / "environments" / "tapfixer" / "runtime"
JOBS_DIR = LAB / "environments" / "tapfixer"
JOBS = JOBS_DIR / "ss_jobs.jsonl"
RESULTS = JOBS_DIR / "ss_results.jsonl"
DONE = JOBS_DIR / "ss_done_ids.txt"
IMAGE = "tapfixer-nuxmv:2.0.0"
LOG = LAB / "logs" / "tapfixer"
PRED = LAB / "predictions" / "tapfixer" / "native_ss_predictions.jsonl"
TRACE = LAB / "predictions" / "tapfixer" / "native_ss_traces.jsonl"
TIMEOUT_SEC = 30
FALLBACK = "B0_NO_REPAIR"
OFFICIAL_USERINPUT = RUNTIME / "UserInput.official.py"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def docker_win_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")

def restore_official_userinput() -> None:
    src = RUNTIME / "UserInput.py"
    if not OFFICIAL_USERINPUT.exists() and src.exists():
        OFFICIAL_USERINPUT.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    if OFFICIAL_USERINPUT.exists():
        src.write_text(OFFICIAL_USERINPUT.read_text(encoding="utf-8"), encoding="utf-8")

def write_jobs(irs: list[dict], limit: int = 0) -> int:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with JOBS.open("w", encoding="utf-8") as f:
        for ir in irs:
            pack = ir.get("property_pack") or {}
            if not pack.get("applicable"):
                continue
            job = {
                "sample_id": ir["id"],
                "parent_id": ir["id"],
                "rules": pack.get("rules") or {},
                "specs": pack.get("specs") or [],
                "properties": pack.get("properties") or [],
            }
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n

def run_worker() -> dict:
    restore_official_userinput()
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{docker_win_path(RUNTIME)}:/work/TAPFixer",
        "-v", f"{docker_win_path(JOBS_DIR)}:/jobs",
        "-e", "TAPFIXER_JOBS=/jobs/ss_jobs.jsonl",
        "-e", "TAPFIXER_RESULTS=/jobs/ss_results.jsonl",
        "-e", "TAPFIXER_DONE=/jobs/ss_done_ids.txt",
        "-e", "TAPFIXER_USERINPUT=/work/TAPFixer/UserInput.py",
        "-e", f"TAPFIXER_TIMEOUT={TIMEOUT_SEC}",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", "PYTHONIOENCODING=utf-8",
        "-w", "/work/TAPFixer",
        IMAGE,
        "python3", "-u", "/jobs/worker.py",
    ]
    t0 = time.perf_counter()
    logf = LOG / "native_ss_worker.log"
    LOG.mkdir(parents=True, exist_ok=True)
    with logf.open("w", encoding="utf-8") as lf:
        lf.write("CMD " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        rc = proc.wait()
    elapsed = round(time.perf_counter() - t0, 3)
    rec = {
        "exit_code": rc,
        "elapsed_time": elapsed,
        "results_exists": RESULTS.exists(),
        "results_lines": sum(1 for _ in RESULTS.open(encoding="utf-8")) if RESULTS.exists() else 0,
        "done_lines": sum(1 for _ in DONE.open(encoding="utf-8")) if DONE.exists() else 0,
        "log": str(logf),
        "command": cmd,
        "concurrency": 1,
    }
    dump(LOG / "native_ss_worker.json", rec)
    restore_official_userinput()
    return rec

def assemble(irs: list[dict], natives: dict[str, dict]) -> None:
    preds = []
    traces = []
    trows = []
    for ir in irs:
        pack = ir.get("property_pack") or {}
        applicable = bool(pack.get("applicable"))
        t_ad0 = time.perf_counter()
        pred = {
            "sample_id": ir["id"],
            "baseline": "TAPFixer-adapted",
            "applicable": applicable,
            "native_used": False,
            "fallback": None,
            "decision": "NO_ACTION",
            "final_action": None,
        }
        t_ad = (time.perf_counter() - t_ad0) * 1000
        native_ms = 0.0
        conv_ms = 0.0
        if not applicable:
            pred["status"] = str(pack.get("status") or "NOT_APPLICABLE")
            pred["fallback"] = FALLBACK
            pred["final_action"] = action_to_final((ir.get("actions") or [None])[0] if ir.get("actions") else None)
            pred["decision"] = "ACTION" if pred["final_action"] else "NO_ACTION"
            pred["unsupported_reason"] = pack.get("reason")
        else:
            nat = natives.get(ir["id"]) or {}
            native_ms = float(nat.get("native_latency_ms") or 0)
            t0 = time.perf_counter()
            st = str(nat.get("native_status") or "NATIVE_FAILURE")
            if st == "NATIVE_REPAIRED":
                mapped = convert_ss_fix(nat.get("native_patch"), ir)
                pred["status"] = mapped["status"]
                pred["final_action"] = mapped.get("final_action") if mapped.get("ok") else None
                pred["native_used"] = True
            elif st == "NATIVE_NO_VIOLATION":
                pred["status"] = "NATIVE_NO_VIOLATION"
                pred["final_action"] = action_to_final((ir.get("actions") or [None])[0] if ir.get("actions") else None)
                pred["native_used"] = True
            elif st == "NATIVE_NO_CHANGE":
                pred["status"] = "NATIVE_NO_CHANGE"
                pred["final_action"] = action_to_final((ir.get("actions") or [None])[0] if ir.get("actions") else None)
                pred["native_used"] = True
            else:
                pred["status"] = st
                pred["final_action"] = None
                pred["native_used"] = bool(nat)
            pred["decision"] = "ACTION" if pred.get("final_action") else "NO_ACTION"
            pred["native_error"] = nat.get("stderr_tail")
            conv_ms = (time.perf_counter() - t0) * 1000
            traces.append({
                "sample_id": ir["id"],
                "correctness_property": pack.get("properties"),
                "native_rule_representation": nat.get("native_input_rules"),
                "native_model_checking_result": nat.get("stdout_tail"),
                "native_repair_patch": nat.get("native_patch"),
                "native_status": pred["status"],
                "native_engine_status": st,
                "native_latency": native_ms,
                "canonical_final_action": pred.get("final_action"),
                "stderr_tail": nat.get("stderr_tail"),
            })
        pred["adapter_latency_ms"] = round(t_ad, 4)
        pred["native_latency_ms"] = round(native_ms, 4)
        pred["output_conversion_latency_ms"] = round(conv_ms, 4)
        pred["end_to_end_latency_ms"] = round(t_ad + native_ms + conv_ms, 4)
        preds.append(pred)
        trows.append({
            "id": ir["id"],
            "baseline": "TAPFixer-adapted",
            "dataset": "ss",
            "applicable": applicable,
            "adapter_latency_ms": pred["adapter_latency_ms"],
            "native_latency_ms": pred["native_latency_ms"],
            "output_conversion_latency_ms": pred["output_conversion_latency_ms"],
            "end_to_end_latency_ms": pred["end_to_end_latency_ms"],
            "status": pred["status"],
        })
    PRED.parent.mkdir(parents=True, exist_ok=True)
    with PRED.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
    with TRACE.open("w", encoding="utf-8") as f:
        for p in traces:
            f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
    tp = LAB / "timing" / "tapfixer_ss_native_timing.jsonl"
    tp.parent.mkdir(parents=True, exist_ok=True)
    with tp.open("w", encoding="utf-8") as f:
        for r in trows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dump(LAB / "predictions" / "tapfixer" / "native_ss_manifest.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_file(PRED),
        "n": len(preds),
        "applicable": sum(1 for p in preds if p.get("applicable")),
        "native_executions": sum(1 for p in preds if p.get("applicable") and p.get("native_used")),
        "status": dict(Counter(p["status"] for p in preds)),
        "GOLD_USED_DURING_GENERATION": "NO",
        "concurrency": 1,
    })
    print(json.dumps({
        "predictions": str(PRED),
        "sha256": sha256_file(PRED),
        "status": dict(Counter(p["status"] for p in preds)),
    }, indent=2))

def load_natives() -> dict[str, dict]:
    natives: dict[str, dict] = {}
    if RESULTS.exists():
        for rec in iter_jsonl(RESULTS):
            natives[str(rec.get("sample_id") or rec.get("parent_id"))] = rec
    return natives

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["jobs", "worker", "assemble", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    frozen_ma = LAB / "predictions" / "tapfixer" / "native_ma_predictions.jsonl"
    if not frozen_ma.exists():
        raise SystemExit("refusing to run: frozen TAPFixer MA predictions missing")
    LOG.mkdir(parents=True, exist_ok=True)
    irs = list(iter_jsonl(IR)) if args.phase != "smoke" else []
    if args.phase in {"jobs", "all"}:
        n = write_jobs(irs, args.limit)
        print(json.dumps({"jobs": n, "path": str(JOBS)}))
    if args.phase in {"worker", "all"}:
        if RESULTS.exists():
            RESULTS.unlink()
        if DONE.exists():
            DONE.unlink()
        print(json.dumps(run_worker(), indent=2))
    if args.phase in {"assemble", "all"}:
        if not irs:
            irs = list(iter_jsonl(IR))
        assemble(irs, load_natives())

if __name__ == "__main__":
    main()
