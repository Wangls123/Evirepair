from __future__ import annotations

import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ss_b0_vs_final_gold_y import SCENARIO_MAP, SS, Y_PATH, b0_from_sample
from eval_ss_trhr_credibility import run_trhr
from eval_ss_trhr_final import (
    _llm_stub,
    load_jsonl,
    metrics_block,
    pair_from_act,
    semantic_ok,
    y_pair,
)
from eval_ss_trhr_v2_offline import (
    apply_r1,
    climate_capability,
    compact_action,
    r1_should_fire,
)
from smarthome_mdf.ss_trhr_repair.context import build_repair_context

OUT = ROOT / "runs" / "ss_trhr_repair"
TRACE = OUT / "repair_execution_trace.jsonl"
SERIALIZATION_PREFIXES = ("system_log.", "google_sheets.")
SERIALIZATION_EXACT = {"system_log.write", "google_sheets.append_sheet"}
CAP_TO_SCENE = {
    "Lighting": "advanced_lighting",
    "ClimateControl": "climate_window",
    "Schedule": "on_off_schedule",
    "Record": "periodic_task_scheduling",
}
CAP_PRIORITY = ["Lighting", "ClimateControl", "Schedule", "Record"]

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def is_serialization_event(svc: str) -> bool:
    s = str(svc or "")
    return s in SERIALIZATION_EXACT or s.startswith(SERIALIZATION_PREFIXES)

def strip_serialization_payload(pred: dict[str, Any] | None) -> tuple[Any, bool]:

    if not pred:
        return pred, False
    svc = str(pred.get("service") or "")
    if not is_serialization_event(svc):
        return pred, False
    params = dict(pred.get("parameters") or {})
    if not params:
        return pred, False
    out = compact_action(pred) or dict(pred)
    out["service"] = svc
    out["target_entity"] = pred.get("target_entity")
    out["parameters"] = {}
    return out, True

def apply_hvac(ctx: dict[str, Any], pred: dict[str, Any] | None) -> tuple[Any, bool]:
    if r1_should_fire(ctx, pred, use_scene=False):
        return apply_r1(pred), True
    return pred, False

def sidecar(ctx: dict[str, Any], pred: dict[str, Any] | None, *, hvac: bool, payload: bool) -> Any:
    out = pred
    if hvac:
        out, _ = apply_hvac(ctx, out)
    if payload:
        out, _ = strip_serialization_payload(out)
    return out

def infer_capabilities(ctx: dict[str, Any]) -> set[str]:
    tokens = [str(ctx.get("declared_effect_service") or "")]
    tokens.extend(str(s) for s in (ctx.get("candidate_services") or []) if s)
    b0 = ctx.get("b0_action") or {}
    tokens.append(str(b0.get("service") or ""))
    caps: set[str] = set()
    for t in tokens:
        if t.startswith("light."):
            caps.add("Lighting")
        if t.startswith("climate."):
            caps.add("ClimateControl")
        if t.startswith("schedule.") or t in {"scene.turn_on", "scene.turn_off"}:
            caps.add("Schedule")
        if t.startswith("google_sheets."):
            caps.add("Record")
        if t.startswith("notify."):
            caps.add("Notify")
        if t.startswith(("logbook.", "system_log.", "persistent_notification.")):
            caps.add("Log")
    return caps

def scene_from_capabilities(caps: set[str]) -> str:
    for name in CAP_PRIORITY:
        if name in caps:
            return CAP_TO_SCENE[name]
    return ""

def capability_ctx(ctx: dict[str, Any]) -> dict[str, Any]:
    c = dict(ctx)
    c["payload"] = copy.deepcopy(ctx.get("payload") or {})
    c["candidate_services"] = list(ctx.get("candidate_services") or [])
    if ctx.get("b0_action"):
        act = dict(ctx["b0_action"])
        act["parameters"] = dict(act.get("parameters") or {})
        c["b0_action"] = act
    caps = infer_capabilities(c)
    routed = scene_from_capabilities(caps)
    c["scene_type"] = routed
    payload = dict(c["payload"])
    payload["scenario"] = routed
    c["payload"] = payload
    c["inferred_capabilities"] = sorted(caps)
    return c

def acc() -> dict[str, Any]:
    return {"exact": 0, "rs": 0, "fr": 0, "ops": Counter(), "n": 0, "b0": 0, "fired": 0, "fixed": 0}

def add(bucket: dict[str, Any], pred: Any, op: str, b0_ok: bool, y_dec, y_act, y_n, *, fired=False, was_full_ok=False) -> bool:
    p_dec, p_act, p_n = pair_from_act(pred)
    ok = semantic_ok(p_dec, p_act, p_n, y_dec, y_act, y_n)
    bucket["n"] += 1
    bucket["exact"] += int(ok)
    bucket["rs"] += int((not b0_ok) and ok)
    bucket["fr"] += int(b0_ok and not ok)
    bucket["ops"][op] += 1
    bucket["b0"] += int(b0_ok)
    if fired:
        bucket["fired"] += 1
        if (not was_full_ok) and ok:
            bucket["fixed"] += 1
    return ok

def pack(bucket: dict[str, Any], b0_exact: int | None = None) -> dict[str, Any]:
    n = bucket["n"]
    b0e = bucket["b0"] if b0_exact is None else b0_exact
    block = metrics_block(n, bucket["exact"], b0e, bucket["rs"], bucket["fr"])
    block["Gain_pp"] = round(100.0 * (block["Semantic_Success"] - (b0e / max(n, 1))), 2)
    block["operators"] = dict(bucket["ops"])
    if bucket.get("fired") is not None:
        block["sidecar_fired"] = bucket["fired"]
        block["sidecar_fixed_from_full_fail"] = bucket["fixed"]
    return block

def main() -> None:
    y_idx = load_jsonl(Y_PATH)
    traces = load_jsonl(TRACE)
    samples: list[dict[str, Any]] = []
    with SS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    names = ["A_Frozen_Repair", "B_Frozen_HVAC", "C_Frozen_Payload", "D_Frozen_HVAC_Payload", "Capability_routing"]
    overall = {k: acc() for k in names}
    by_scene = {k: defaultdict(acc) for k in names}
    lighting = {k: acc() for k in ("Full_routing", "Capability_routing")}
    periodic = {k: acc() for k in ("Full_routing", "Capability_routing")}
    cap_agree = Counter()
    cap_route = Counter()
    n = b0_exact_n = 0
    hvac_fr_new = payload_fr_new = joint_fr_new = 0
    hvac_broke = payload_broke = joint_broke = 0

    print("v2 joint sidecar eval...", flush=True)
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
        scene = SCENARIO_MAP.get(str(s.get("scene_type") or ""), str(s.get("scene_type") or "other"))
        op = str(tr.get("operator_used") or "KEEP")
        reason = str(tr.get("reason") or "")
        llm_fn = _llm_stub(
            False if reason == "A3_LLM_UNNECESSARY" else True if reason == "ADD_LLM_NECESSARY" else None
        )
        ctx = build_repair_context(s)
        frozen = compact_action(tr.get("repaired_action"))
        f_ok = semantic_ok(*pair_from_act(frozen), y_dec, y_act, y_n)

        hvac_pred, hvac_fired = apply_hvac(ctx, frozen)
        pay_pred, pay_fired = strip_serialization_payload(frozen)
        joint_pred = sidecar(ctx, frozen, hvac=True, payload=True)
        h_ok = semantic_ok(*pair_from_act(hvac_pred), y_dec, y_act, y_n)
        p_ok = semantic_ok(*pair_from_act(pay_pred), y_dec, y_act, y_n)
        j_ok = semantic_ok(*pair_from_act(joint_pred), y_dec, y_act, y_n)
        if f_ok and not h_ok:
            hvac_broke += 1
        if f_ok and not p_ok:
            payload_broke += 1
        if f_ok and not j_ok:
            joint_broke += 1
        if b0_ok and f_ok and not h_ok:
            hvac_fr_new += 1
        if b0_ok and f_ok and not p_ok:
            payload_fr_new += 1
        if b0_ok and f_ok and not j_ok:
            joint_fr_new += 1

        preds = {
            "A_Frozen_Repair": (frozen, op, False),
            "B_Frozen_HVAC": (hvac_pred, op if not hvac_fired else "MODIFY_HVAC_SIDECAR", hvac_fired),
            "C_Frozen_Payload": (pay_pred, op if not pay_fired else "PAYLOAD_SIDECAR", pay_fired),
            "D_Frozen_HVAC_Payload": (
                joint_pred,
                op if not (hvac_fired or pay_fired) else "JOINT_SIDECAR",
                hvac_fired or pay_fired,
            ),
        }
        for name, (pred, opu, fired) in preds.items():
            add(overall[name], pred, opu, b0_ok, y_dec, y_act, y_n, fired=fired, was_full_ok=f_ok)
            add(by_scene[name][scene], pred, opu, b0_ok, y_dec, y_act, y_n, fired=fired, was_full_ok=f_ok)

        add(lighting["Full_routing"], frozen, op, b0_ok, y_dec, y_act, y_n) if scene == "lighting" else None
        add(periodic["Full_routing"], frozen, op, b0_ok, y_dec, y_act, y_n) if scene == "periodic" else None

        cctx = capability_ctx(ctx)
        routed = str(cctx.get("scene_type") or "")
        original_scene = str(s.get("scene_type") or "")
        cap_route[routed or "empty"] += 1
        cap_agree["match" if routed == original_scene else "mismatch"] += 1
        cap_pred, cap_op = run_trhr(cctx, llm_fn)
        add(overall["Capability_routing"], cap_pred, cap_op, b0_ok, y_dec, y_act, y_n)
        add(by_scene["Capability_routing"][scene], cap_pred, cap_op, b0_ok, y_dec, y_act, y_n)
        if scene == "lighting":
            add(lighting["Capability_routing"], cap_pred, cap_op, b0_ok, y_dec, y_act, y_n)
        if scene == "periodic":
            add(periodic["Capability_routing"], cap_pred, cap_op, b0_ok, y_dec, y_act, y_n)

        if i % 2000 == 0:
            print(f"  {i}/{len(samples)}", flush=True)

    def pack_all(name: str) -> dict[str, Any]:
        block = pack(overall[name], b0_exact_n)
        block["vs_Full_pp"] = round(
            100.0 * (overall[name]["exact"] - overall["A_Frozen_Repair"]["exact"]) / max(n, 1), 2
        )
        return block

    out = {
        "note": (
            "Sidecar evaluation on frozen traces. Frozen Repair / Gold Y / B0 / official "
            "63.13% / 1.49% are not modified. HVAC R1 is Y-blind window_open + climate "
            "capability. Payload strip is serialization-only (system_log / google_sheets "
            "parameters → {}), service unchanged. Capability routing infers capabilities "
            "from declared/candidates/B0 and injects a scene adapter for unmodified modules."
        ),
        "N": n,
        "repair_modified": False,
        "gold_modified": False,
        "b0_modified": False,
        "official_Full_frozen": {
            "Semantic_Success": 0.6313,
            "False_Repair": 0.0149,
            "not_overwritten": True,
        },
        "variants": {
            "A_Frozen_Repair": pack_all("A_Frozen_Repair"),
            "B_Frozen_HVAC": pack_all("B_Frozen_HVAC"),
            "C_Frozen_Payload": pack_all("C_Frozen_Payload"),
            "D_Frozen_HVAC_Payload": pack_all("D_Frozen_HVAC_Payload"),
        },
        "sidecar_modules": {
            "HVAC": {
                "fired": overall["B_Frozen_HVAC"]["fired"],
                "fixed_from_full_fail": overall["B_Frozen_HVAC"]["fixed"],
                "new_false_repair_vs_full": hvac_fr_new,
                "broke_full_exact": hvac_broke,
            },
            "Payload": {
                "fired": overall["C_Frozen_Payload"]["fired"],
                "fixed_from_full_fail": overall["C_Frozen_Payload"]["fixed"],
                "new_false_repair_vs_full": payload_fr_new,
                "broke_full_exact": payload_broke,
            },
            "Joint": {
                "fired": overall["D_Frozen_HVAC_Payload"]["fired"],
                "fixed_from_full_fail": overall["D_Frozen_HVAC_Payload"]["fixed"],
                "new_false_repair_vs_full": joint_fr_new,
                "broke_full_exact": joint_broke,
            },
        },
        "capability_routing": {
            "overall": pack_all("Capability_routing"),
            "scene_adapter_agreement_with_corpus_scene_type": dict(cap_agree),
            "routed_scene": dict(cap_route),
            "lighting": {
                "Full_routing": pack(lighting["Full_routing"]),
                "Capability_routing": pack(lighting["Capability_routing"]),
                "delta_pp": round(
                    100.0
                    * (
                        lighting["Capability_routing"]["exact"] / max(lighting["Capability_routing"]["n"], 1)
                        - lighting["Full_routing"]["exact"] / max(lighting["Full_routing"]["n"], 1)
                    ),
                    2,
                ),
            },
            "periodic": {
                "Full_routing": pack(periodic["Full_routing"]),
                "Capability_routing": pack(periodic["Capability_routing"]),
                "delta_pp": round(
                    100.0
                    * (
                        periodic["Capability_routing"]["exact"] / max(periodic["Capability_routing"]["n"], 1)
                        - periodic["Full_routing"]["exact"] / max(periodic["Full_routing"]["n"], 1)
                    ),
                    2,
                ),
            },
        },
        "by_scenario": {
            sc: {k: pack(by_scene[k][sc]) for k in names if by_scene[k][sc]["n"]}
            for sc in sorted({sc for k in names for sc in by_scene[k]})
        },
    }
    dump(OUT / "repair_v2_sidecar_evaluation.json", out)
    print(
        json.dumps(
            {
                "A": out["variants"]["A_Frozen_Repair"]["Semantic_Success"],
                "B": out["variants"]["B_Frozen_HVAC"]["Semantic_Success"],
                "C": out["variants"]["C_Frozen_Payload"]["Semantic_Success"],
                "D": out["variants"]["D_Frozen_HVAC_Payload"]["Semantic_Success"],
                "D_FR": out["variants"]["D_Frozen_HVAC_Payload"]["False_Repair"],
                "hvac_fired": out["sidecar_modules"]["HVAC"],
                "payload_fired": out["sidecar_modules"]["Payload"],
                "joint": out["sidecar_modules"]["Joint"],
                "cap_ss": out["capability_routing"]["overall"]["Semantic_Success"],
                "lighting": out["capability_routing"]["lighting"]["delta_pp"],
                "periodic": out["capability_routing"]["periodic"]["delta_pp"],
                "cap_agree": dict(cap_agree),
            },
            indent=2,
        ),
        flush=True,
    )

if __name__ == "__main__":
    main()
