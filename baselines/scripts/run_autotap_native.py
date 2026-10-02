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

from baselines.autotap.output_adapter import convert_taps
from baselines.common.tap_ir import sample_to_ir

IOT = LAB / "third_party" / "autotap" / "upstream" / "iot-autotap"
IMAGE = "zlfben/autotap_backend:icse"
SPOT_PY = "/jobs:/usr/local/local/lib/python3.10/dist-packages:/root/AutoTap:/usr/local/lib/python3.10/site-packages"
IR = LAB / "converted" / "autotap" / "ss_ir.jsonl"
JOBS_DIR = LAB / "environments" / "autotap"
NATIVE_LOG = LAB / "logs" / "autotap"
PRED = LAB / "predictions" / "autotap" / "native_ss_predictions.jsonl"
TRACE = LAB / "predictions" / "autotap" / "native_ss_traces.jsonl"
TIMEOUT_SEC = 15
FALLBACK = "B0_NO_REPAIR"

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
    s = str(p.resolve())
    return s.replace("\\", "/")

def action_to_final(act: dict | None) -> dict | None:
    if not act:
        return None
    svc = str(act.get("service") or "")
    if "." in svc:
        cap, op = svc.split(".", 1)
    else:
        cap, op = "unknown", svc or "none"
    return {
        "capability": cap,
        "operation": op,
        "target": act.get("target") or act.get("target_entity") or act.get("entity"),
        "semantic_payload": dict(act.get("parameters") or {}),
        "service": svc,
    }

def b0_action(ir: dict) -> dict | None:
    acts = ir.get("actions") or []
    return acts[0] if acts else None

def run_smoke() -> dict:
    NATIVE_LOG.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{docker_win_path(IOT)}:/root/AutoTap:ro",
        "-v", f"{docker_win_path(LAB / 'environments' / 'autotap')}:/jobs:ro",
        "-e", f"PYTHONPATH={SPOT_PY}",
        "-e", "LD_LIBRARY_PATH=/usr/local/lib",
        "-e", "PYTHONUNBUFFERED=1",
        "-w", "/root/AutoTap",
        IMAGE,
        "python3", "/jobs/smoke.py",
    ]
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=180, encoding="utf-8", errors="replace")
        elapsed = round(time.perf_counter() - t0, 3)
    except subprocess.TimeoutExpired as exc:
        rec = {
            "AUTOTAP_UPSTREAM_SMOKE_TEST": "FAIL",
            "command": cmd,
            "exit_code": None,
            "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "timeout",
            "elapsed_time": round(time.perf_counter() - t0, 3),
            "error": "timeout",
        }
        dump(NATIVE_LOG / "upstream_smoke.json", rec)
        return rec
    rec = {
        "AUTOTAP_UPSTREAM_SMOKE_TEST": "PASS" if cp.returncode == 0 and "AUTOTAP_SMOKE_OK" in (cp.stdout or "") else "FAIL",
        "command": cmd,
        "exit_code": cp.returncode,
        "stdout": (cp.stdout or "")[-12000:],
        "stderr": (cp.stderr or "")[-12000:],
        "elapsed_time": elapsed,
        "native_output": (cp.stdout or "")[-4000:],
        "image": IMAGE,
        "autotap_commit": "fe0fdd638a1b170b1ac71b0f2921ab54d547893b",
        "spot": "official_spot-2.7.5_from_lrde.epita.fr",
    }
    dump(NATIVE_LOG / "upstream_smoke.json", rec)
    (NATIVE_LOG / "upstream_smoke.stdout.txt").write_text(cp.stdout or "", encoding="utf-8")
    (NATIVE_LOG / "upstream_smoke.stderr.txt").write_text(cp.stderr or "", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("AUTOTAP_UPSTREAM_SMOKE_TEST", "exit_code", "elapsed_time")}, indent=2))
    return rec

def write_jobs(irs: list[dict]) -> int:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jobs_path = JOBS_DIR / "jobs.jsonl"
    n = 0
    with jobs_path.open("w", encoding="utf-8") as f:
        for ir in irs:
            pack = ir.get("property_pack") or {}
            if not pack.get("applicable"):
                continue
            props = pack.get("properties") or []
            if not props:
                continue
            p0 = props[0]
            job = {
                "sample_id": ir["id"],
                "ltl": p0.get("ltl"),
                "init_value_dict": p0.get("init_value_dict") or {},
                "tap_list": [],
                "property": p0,
            }
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
            n += 1
    return n

def classify_native(nat: dict | None, conv: dict | None) -> str:
    if not nat:
        return "NATIVE_FAILURE"
    st = nat.get("native_status")
    if st == "NATIVE_TIMEOUT":
        return "NATIVE_TIMEOUT"
    if st != "NATIVE_OK":
        return "NATIVE_FAILURE"
    taps = nat.get("native_output") or []
    if not taps:
        return "NATIVE_NO_SOLUTION"
    if conv and conv.get("ok"):
        return "NATIVE_REPAIRED"
    return "OUTPUT_TRANSLATION_FAILURE"

def run_worker() -> dict:
    results = JOBS_DIR / "results.jsonl"
    done = JOBS_DIR / "done_ids.txt"
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{docker_win_path(IOT)}:/root/AutoTap:ro",
        "-v", f"{docker_win_path(LAB / 'environments' / 'autotap')}:/jobs",
        "-e", "AUTOTAP_JOBS=/jobs/jobs.jsonl",
        "-e", "AUTOTAP_RESULTS=/jobs/results.jsonl",
        "-e", "AUTOTAP_DONE=/jobs/done_ids.txt",
        "-e", f"PYTHONPATH={SPOT_PY}",
        "-e", "LD_LIBRARY_PATH=/usr/local/lib",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", f"AUTOTAP_TIMEOUT={TIMEOUT_SEC}",
        "-w", "/root/AutoTap",
        IMAGE,
        "python3", "-u", "/jobs/worker.py",
    ]
    t0 = time.perf_counter()
    logf = NATIVE_LOG / "native_worker.log"
    NATIVE_LOG.mkdir(parents=True, exist_ok=True)
    with logf.open("w", encoding="utf-8") as lf:
        lf.write("CMD " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
    elapsed = round(time.perf_counter() - t0, 3)
    rec = {
        "exit_code": rc,
        "elapsed_time": elapsed,
        "results_exists": results.exists(),
        "results_lines": sum(1 for _ in results.open(encoding="utf-8")) if results.exists() else 0,
        "done_lines": sum(1 for _ in done.open(encoding="utf-8")) if done.exists() else 0,
        "log": str(logf),
    }
    dump(NATIVE_LOG / "native_worker.json", rec)
    return rec

def assemble_predictions(irs: list[dict]) -> None:
    nat_by_id = {}
    res_path = JOBS_DIR / "results.jsonl"
    if res_path.exists():
        for rec in iter_jsonl(res_path):
            nat_by_id[str(rec.get("sample_id"))] = rec
    preds = []
    traces = []
    t_rows = []
    for ir in irs:
        t_ad0 = time.perf_counter()
        pack = ir.get("property_pack") or {}
        applicable = bool(pack.get("applicable"))
        t_ad = (time.perf_counter() - t_ad0) * 1000
        pred = {
            "sample_id": ir["id"],
            "baseline": "AutoTap-adapted",
            "applicable": applicable,
            "property": (pack.get("properties") or [None])[0],
            "native_used": False,
            "fallback": None,
            "final_action": None,
        }
        native_ms = 0.0
        conv_ms = 0.0
        if not applicable:
            pred["status"] = str(pack.get("status") or "NOT_APPLICABLE")
            pred["fallback"] = FALLBACK
            pred["final_action"] = action_to_final(b0_action(ir))
        else:
            nat = nat_by_id.get(ir["id"])
            native_ms = float((nat or {}).get("native_latency_ms") or 0.0)
            t_c0 = time.perf_counter()
            conv = None
            if nat and nat.get("native_status") == "NATIVE_OK":
                conv = convert_taps(nat.get("native_output") or [], ir)
            status = classify_native(nat, conv)
            pred["status"] = status
            pred["native_execution_status"] = (nat or {}).get("native_status")
            pred["native_output"] = (nat or {}).get("native_output")
            pred["native_input"] = (nat or {}).get("native_input")
            pred["native_error"] = (nat or {}).get("error")
            if status == "NATIVE_REPAIRED":
                pred["final_action"] = conv["final_action"]
                pred["native_used"] = True
                pred["converted_canonical_action"] = conv["final_action"]
                pred["mapped_channel"] = conv.get("mapped_channel")
            else:
                pred["final_action"] = None
                pred["native_used"] = bool(nat)
                pred["converted_canonical_action"] = None

            conv_ms = (time.perf_counter() - t_c0) * 1000
            traces.append({
                "sample_id": ir["id"],
                "property": pred["property"],
                "native_AutoTap_input": (nat or {}).get("native_input"),
                "native_AutoTap_output": (nat or {}).get("native_output"),
                "repair_status": status,
                "native_execution_status": (nat or {}).get("native_status"),
                "native_latency": native_ms,
                "converted_canonical_action": pred.get("converted_canonical_action"),
                "error": (nat or {}).get("error"),
            })
        pred["adapter_latency_ms"] = round(t_ad, 4)
        pred["native_latency_ms"] = round(native_ms, 4)
        pred["output_conversion_latency_ms"] = round(conv_ms, 4)
        pred["end_to_end_latency_ms"] = round(t_ad + native_ms + conv_ms, 4)
        preds.append(pred)
        t_rows.append({
            "id": ir["id"],
            "baseline": "AutoTap-adapted",
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
    timing_path = LAB / "timing" / "autotap_native_timing.jsonl"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with timing_path.open("w", encoding="utf-8") as f:
        for r in t_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    native_n = sum(1 for p in preds if p.get("applicable") and p.get("native_used"))
    dump(LAB / "predictions" / "autotap" / "native_ss_manifest.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_file(PRED),
        "n": len(preds),
        "applicable": sum(1 for p in preds if p.get("applicable")),
        "native_executions": native_n,
        "status": dict(Counter(p["status"] for p in preds)),
        "gold_used": False,
        "GOLD_USED_DURING_GENERATION": "NO",
    })
    print(json.dumps({
        "predictions": str(PRED),
        "n": len(preds),
        "native_executions": native_n,
        "status": dict(Counter(p["status"] for p in preds)),
        "sha256": sha256_file(PRED),
    }, indent=2))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "jobs", "worker", "assemble", "all"], default="all")
    args = ap.parse_args()
    irs = list(iter_jsonl(IR)) if args.phase in {"jobs", "assemble", "all"} else []
    if args.phase in {"smoke", "all"}:
        smoke = run_smoke()
        if smoke.get("AUTOTAP_UPSTREAM_SMOKE_TEST") != "PASS":
            print("AUTOTAP_UPSTREAM_SMOKE_TEST=FAIL; not running native batch")
            sys.exit(2)
    if args.phase in {"jobs", "all"}:
        n = write_jobs(irs)
        print(json.dumps({"jobs": n}))
    if args.phase in {"worker", "all"}:
        print(json.dumps(run_worker(), indent=2))
    if args.phase in {"assemble", "all"}:
        if not irs:
            irs = list(iter_jsonl(IR))
        assemble_predictions(irs)

if __name__ == "__main__":
    main()
