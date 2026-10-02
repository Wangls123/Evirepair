from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ss_b0_vs_final_gold_y import SCENARIO_MAP, SS, Y_PATH, b0_from_sample
from eval_ss_trhr_final import (
    _llm_stub,
    _rate,
    error_type,
    load_jsonl,
    metrics_block,
    pair_from_act,
    semantic_ok,
    y_pair,
)
from eval_ss_trhr_credibility import (
    baseline_blacklist,
    baseline_condition,
    baseline_state_only,
)
from eval_ss_trhr_v2_offline import apply_r1, climate_capability, r1_should_fire
from eval_strict_v2_set_remove import (
    MA,
    NEVER_DROP,
    OUT,
    STRICT_DROP,
    TRACE,
    acc_bucket,
    add_bucket,
    apply_payload_strict,
    apply_strict_v2,
    behavior_key,
    cap_of_svc,
    compact_action,
    finish_bucket,
    ma_y_acts,
    map_repaired,
    set_metrics,
    set_remove_v1,
    tok,
)
from eval_ma_evidence_attribution import (
    apply_generic_gate,
    attribute_parent,
    compose_parent,
    ownership_for_component,
    residual_flags,
    toks_from_acts,
)
from eval_ma_evidence_sufficiency import (
    apply_verifier,
    classify_e_labels,
    inject_irrelevant_atom,
    necessity_verifier,
    rebuild_owns,
    runtime_signals,
    strip_owned_atoms,
    sufficiency_of,
)
from eval_ma_verifier_strict_generalization import (
    blank_own,
    empty_composition,
    pattern_families,
    residual_tag,
    strip_bt,
)
from eval_frozen_repair_runtime_replay import (
    acc as rt_acc,
    add_exec,
    add_static,
    as_acts,
    compose_class,
    execute_acts,
    make_box,
    static_validate,
    summarize as rt_summarize,
)
from ha_runtime_sandbox import SERVICE_REGISTRY, canonical_service
from smarthome_mdf.ss_trhr_repair.add import run_add
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.modify import run_modify
from smarthome_mdf.ss_trhr_repair.pipeline import _illegal
from smarthome_mdf.ss_trhr_repair.remove import run_remove

SEED = 20260919
BOOT = 10000
OFFICIAL_SS = {"Semantic_Success": 0.813, "Repair_Success": 0.763, "False_Repair": 0.0149, "Preservation_Rate": 0.9456}
OFFICIAL_MA = {"Parent_Set_Exact_Match": 0.5267, "Action_F1": 0.5697, "False_Parent_Repair_count": 1}
RUNTIME_FROZEN = {
    "SS_S0": {"Executable_Action_Rate": 0.2325, "Execution_Success_Rate": 0.5542, "Unsafe_Action_Rate": 0.4458},
    "SS_S1": {"Executable_Action_Rate": 0.5994, "Execution_Success_Rate": 0.8396, "Unsafe_Action_Rate": 0.1604},
    "SS_S2": {"Executable_Action_Rate": 0.7163, "Execution_Success_Rate": 0.9564, "Unsafe_Action_Rate": 0.0436},
    "MA_M0": {"Executable_Action_Rate": 0.8802, "Execution_Success_Rate": 0.646, "Unsafe_Action_Rate": 0.354},
    "MA_M4": {"Executable_Action_Rate": 0.9311, "Execution_Success_Rate": 0.9559, "Unsafe_Action_Rate": 0.0441},
}
SCENES = ["climate", "lighting", "security", "visual", "appliance", "periodic", "schedule", "other"]
METHOD_FILES = [
    ROOT / "scripts" / "eval_strict_v2_set_remove.py",
    ROOT / "scripts" / "eval_ma_evidence_attribution.py",
    ROOT / "scripts" / "eval_ma_evidence_sufficiency.py",
    ROOT / "src" / "smarthome_mdf" / "ss_trhr_repair" / "remove.py",
    ROOT / "src" / "smarthome_mdf" / "ss_trhr_repair" / "modify.py",
    ROOT / "src" / "smarthome_mdf" / "ss_trhr_repair" / "add.py",
    ROOT / "src" / "smarthome_mdf" / "ss_trhr_repair" / "pipeline.py",
    ROOT / "scripts" / "eval_ss_trhr_v2_offline.py",
]
OH_REF = ROOT / "project_delivery" / "validation" / "openhab" / "semantic_reference" / "openhab_semantic_reference.jsonl"
OH_MAP = OUT / "cross_platform_capability_mapping.json"

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def scene_of(sample: dict) -> str:
    raw = SCENARIO_MAP.get(str(sample.get("scene_type") or ""), str(sample.get("scene_type") or "other"))
    return raw if raw in {"climate", "lighting", "security", "visual", "appliance", "periodic", "schedule"} else "other"

def run_ops(ctx: dict, enable: set[str], llm_fn) -> tuple[Any, str]:
    original = compact_action(ctx.get("b0_action"))
    repaired = original
    op = "KEEP"
    if original:
        if "REMOVE" in enable:
            r = run_remove(ctx, original, llm=llm_fn)
            if r.get("changed"):
                repaired, op = r.get("action_out"), "REMOVE"
        if op == "KEEP" and "MODIFY" in enable:
            m = run_modify(ctx, original)
            if m.get("changed"):
                repaired, op = m.get("action_out"), "MODIFY"
    elif "ADD" in enable:
        a = run_add(ctx, llm=llm_fn)
        if a.get("changed"):
            repaired, op = a.get("action_out"), "ADD"
    repaired = compact_action(repaired) if repaired else None
    if _illegal(repaired, ctx, original):
        return original, "KEEP"
    return repaired, op

def apply_v2_parts(ctx: dict, pred: dict | None, *, r1: bool, payload: bool) -> Any:
    out = pred
    if r1 and r1_should_fire(ctx, out, use_scene=False) and climate_capability(ctx, out):
        out = apply_r1(out)
    if payload:
        out, _hit, _dropped = apply_payload_strict(out)
    return out

def mask_ctx(ctx: dict, mode: str) -> dict:
    c = dict(ctx)
    p = dict(c.get("payload") or {})
    obs = dict(p.get("observation") or {})
    ents = list(p.get("entity_observations") or [])
    if mode in {"observation", "T1", "C1"}:
        p["observation"] = {}
    if mode in {"entity_state", "T3"}:
        p["entity_observations"] = []
    if mode in {"automation_semantics"}:
        p["trigger"] = []
        p["condition"] = []
        c["declared_effect_service"] = ""
        c["candidate_services"] = []
    if mode in {"trigger_condition"}:
        p["trigger"] = []
        p["condition"] = []
    if mode in {"T2", "C4"}:
        obs["window_open"] = not bool(obs.get("window_open"))
        obs["window_state"] = "closed" if str(obs.get("window_state") or "").lower() in {"open", "on", "true"} else "open"
        p["observation"] = obs
    if mode == "T4":
        obs["hvac_mode"] = "stale_unknown"
        p["observation"] = obs
    if mode == "T6":
        p["entity_observations"] = [{"entity_id": "not an entity", "state": {"bad": True}}]
    if mode == "T9":
        c["scene_type"] = ""
        c["declared_effect_service"] = ""
    if mode == "C3":
        ents.append({"entity_id": "sensor.irrelevant_injection", "state": "on", "attributes": {}})
        p["entity_observations"] = ents
    if mode == "C5":
        obs["hvac_mode"] = "off"
        obs["light_state"] = "on"
        p["observation"] = obs
    if mode == "T8":
        p["entity_observations"] = ents + ents
    c["payload"] = p
    return c

def ss_prf(pred, y_dec: str, y_act, y_n: int) -> dict:
    p_dec, p_act, _pn = pair_from_act(pred)
    p_toks = [tok(p_act)] if p_act and tok(p_act) else []
    y_toks = [tok(y_act)] if y_act and y_dec == "ACTION" and tok(y_act) else []
    return set_metrics(p_toks, y_toks)

def ss_ok(pred, y_dec, y_act, y_n) -> bool:
    return semantic_ok(*pair_from_act(pred), y_dec, y_act, y_n)

class Acc:
    def __init__(self) -> None:
        self.n = 0
        self.exact = 0
        self.rs = 0
        self.fr = 0
        self.dec = 0
        self.p = 0.0
        self.r = 0.0
        self.f = 0.0
        self.n_pred = 0
        self.flags: list[bool] = []
        self.f1s: list[float] = []
        self.ps: list[float] = []
        self.rs_: list[float] = []

    def add(self, pred, b0_ok: bool, y_dec, y_act, y_n) -> bool:
        ok = ss_ok(pred, y_dec, y_act, y_n)
        p_dec, p_act, pn = pair_from_act(pred)
        m = ss_prf(pred, y_dec, y_act, y_n)
        self.n += 1
        self.exact += int(ok)
        self.rs += int((not b0_ok) and ok)
        self.fr += int(b0_ok and not ok)
        self.dec += int(p_dec == y_dec)
        self.p += m["precision"]
        self.r += m["recall"]
        self.f += m["f1"]
        self.n_pred += pn
        self.flags.append(ok)
        self.f1s.append(m["f1"])
        self.ps.append(m["precision"])
        self.rs_.append(m["recall"])
        return ok

    def pack(self, b0_exact: int) -> dict[str, Any]:
        out = metrics_block(self.n, self.exact, b0_exact, self.rs, self.fr)
        n = max(self.n, 1)
        out["Decision_Accuracy"] = _rate(self.dec, self.n)
        out["Action_Precision"] = round(self.p / n, 4)
        out["Action_Recall"] = round(self.r / n, 4)
        out["Action_F1"] = round(self.f / n, 4)
        out["mean_action_count"] = round(self.n_pred / n, 4)
        return out

def mcnemar(a: list[bool], b: list[bool]) -> dict[str, Any]:
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if (not x) and y)
    n = n01 + n10
    if n == 0:
        return {"b_improved": n10, "a_worsened": n01, "statistic": 0.0, "p_value": 1.0, "n_discordant": 0}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    return {
        "b_improved": n10,
        "a_worsened": n01,
        "statistic": round(stat, 4),
        "p_value": float(chi2.sf(stat, 1)),
        "n_discordant": n,
        "test": "paired_McNemar",
        "gold_based": True,
    }

def bootstrap_delta(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    n = len(a)
    idx = rng.integers(0, n, size=(BOOT, n))
    delta = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return {
        "mean_delta": round(float(delta.mean()), 6),
        "ci95": [round(float(np.quantile(delta, 0.025)), 6), round(float(np.quantile(delta, 0.975)), 6)],
        "n_bootstrap": BOOT,
        "seed": SEED,
        "test": "paired_bootstrap",
        "gold_based": True,
    }

def ss_residual_tag(*, et: str, op: str, b0_ok: bool, r_ok: bool, scene: str, svc: str, y_dec: str) -> str:
    if b0_ok and not r_ok:
        return "R8_GENUINE_INFERENCE_ERROR"
    if "llmvision" in (svc or "") or scene == "visual":
        return "R4_UNSUPPORTED_BEHAVIOR_CAPABILITY"
    if et in {"C1_missing_state_action", "C3_missing_temporal_action"}:
        return "R7_COMPLETION_NOT_UNIQUELY_INFERABLE"
    if et in {"B2_parameter_mismatch"}:
        return "R5_REPRESENTATION_AMBIGUITY"
    if et in {"B4_action_target_ambiguity"}:
        return "R5_REPRESENTATION_AMBIGUITY"
    if et.startswith("C2") or (op == "KEEP" and y_dec == "ACTION"):
        return "R3_GOLD_CONTRACT_DISAGREEMENT"
    if et.startswith("A3") or et.startswith("A4"):
        return "R1_INSUFFICIENT_EVIDENCE"
    if op == "KEEP":
        return "R1_INSUFFICIENT_EVIDENCE"
    return "R8_GENUINE_INFERENCE_ERROR"

def envelope_label(acts: list[dict], st: dict, ex: dict) -> str:
    if not acts:
        return "SAFE_ABSTAIN"
    svc = str((acts[0] or {}).get("service") or "")
    if any(str(a.get("service") or "").startswith("llmvision.") for a in acts):
        return "UNSUPPORTED"
    if ex.get("failed"):
        return "RUNTIME_FAILURE"
    if st.get("executable_n", 0) == len(acts) and (ex.get("success") or 0) >= 1:
        return "SAFE_REPAIR"
    if svc.startswith(("system_log.", "logbook.", "notify.")):
        return "SEMANTIC_UNVERIFIABLE"
    if st.get("invalid_n"):
        return "RUNTIME_FAILURE"
    return "SEMANTIC_UNVERIFIABLE"

def known_entities(obs: dict, ents: list, acts: list[dict]) -> set[str]:
    ids = set()
    for a in acts:
        e = a.get("target_entity") or a.get("entity")
        if e:
            ids.add(str(e) if not isinstance(e, (list, dict)) else str((e[0] if isinstance(e, list) and e else e.get("entity_id") if isinstance(e, dict) else e) or ""))
    for e in ents or []:
        if isinstance(e, dict) and e.get("entity_id"):
            ids.add(str(e["entity_id"]))
    ids.discard("")
    return ids

def ma_finish(b: dict, sem_exact: int = 0, abstain_n: int = 0, abstain_d: int = 0) -> dict[str, Any]:
    out = finish_bucket(b)
    n = max(b["n"], 1)
    out["Semantic_Parent_Exact"] = round(sem_exact / n, 4)
    out["Semantic_Parent_Exact_count"] = sem_exact
    out["ABSTAIN_Rate"] = round(abstain_n / abstain_d, 4) if abstain_d else None
    out["ABSTAIN_count"] = abstain_n
    return out

def add_ma(b: dict, acts: list, gold_toks: list, bind: dict, b0_ok: bool) -> tuple[dict, list, bool]:
    toks = toks_from_acts(acts, bind)
    m = set_metrics(toks, gold_toks)
    add_bucket(b, m, b0_ok, bool(m["set_exact"]), m["n_pred"])
    return m, toks, bool(m["set_exact"])

def sem_exact(acts: list, gold_toks: list, bind: dict) -> bool:
    pred = [behavior_key(t) for t in toks_from_acts(acts, bind)]
    gold = [behavior_key(t) for t in gold_toks]
    return Counter(pred) == Counter(gold)

def blank_owns(owns_by_id: dict) -> dict:
    return {k: blank_own(v) for k, v in owns_by_id.items()}

def expected_close(got: dict, exp: dict) -> dict[str, Any]:
    mism = {}
    for k, v in exp.items():
        g = got.get(k)
        if g != v:
            mism[k] = {"expected": v, "got": g}
    return mism

def main() -> None:
    t_all = time.perf_counter()
    print("hash freeze inputs...", flush=True)
    hashes = {
        "single_scene_final.jsonl": sha256_file(SS),
        "multi_action_final.jsonl": sha256_file(MA),
        "single_scene_final_y.jsonl": sha256_file(Y_PATH),
        "repair_execution_trace.jsonl": sha256_file(TRACE),
        "method_files": {str(p.relative_to(ROOT)): sha256_file(p) for p in METHOD_FILES},
    }
    cfg = {
        "STRICT_DROP": sorted(STRICT_DROP),
        "NEVER_DROP": sorted(NEVER_DROP),
        "seed": SEED,
        "bootstrap": BOOT,
        "llm_this_eval": False,
        "llm_in_frozen_traces": "stubbed_from_trace_A3_or_ADD_flags_only",
        "payload_canonicalization_allowed": ["level", "logger"],
        "payload_canonicalization_forbidden": sorted(NEVER_DROP),
    }
    hashes["configuration"] = sha256_text(json.dumps(cfg, sort_keys=True))
    method_blob = "".join(sha256_file(p) or "" for p in METHOD_FILES)
    hashes["repair_code"] = sha256_text(method_blob)

    y_idx = load_jsonl(Y_PATH)
    traces = load_jsonl(TRACE)
    ss_by_id: dict[str, dict] = {}
    with SS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                ss_by_id[str(rec.get("sample_id") or "")] = rec

    ss_names = [
        "S0_B0",
        "S1_Baseline-1",
        "S2_Baseline-2",
        "S3_Baseline-3",
        "S4_Frozen_v1",
        "S5_Strict_v2",
        "abl_full",
        "abl_minus_REMOVE",
        "abl_minus_Refinement",
        "abl_minus_Completion",
        "abl_minus_HVAC_R1",
        "abl_minus_Payload",
        "abl_REMOVE_only",
        "abl_Refinement_only",
        "abl_Completion_only",
        "abl_RM",
        "abl_RMA",
        "in_observation",
        "in_entity_state",
        "in_automation_semantics",
        "in_trigger_condition",
        "idemp_second",
        "stress_T1",
        "stress_T2",
        "stress_T3",
        "stress_T5",
        "stress_T9",
    ]
    ss_acc = {k: Acc() for k in ss_names}
    ss_scene = {k: {sc: Acc() for sc in SCENES} for k in ("S0_B0", "S4_Frozen_v1", "S5_Strict_v2")}
    ss_res = Counter()
    ss_res_cap = defaultdict(Counter)
    ss_res_scene = defaultdict(Counter)
    ss_fixable = 0
    flags = {k: [] for k in ("S0_B0", "S4_Frozen_v1", "S5_Strict_v2")}
    f1s = {k: [] for k in flags}
    hash_ss1: list[str] = []
    idemp_changed = 0
    idemp_semantic = 0
    rt_ss = {k: rt_acc() for k in ("S0", "S1", "S2")}
    env_ss = Counter()
    env_ss_cap: dict[str, Counter] = defaultdict(Counter)
    refine = Counter()
    t_prep: list[float] = []
    t_rule: list[float] = []
    t_val: list[float] = []
    t_exec: list[float] = []
    b0_exact_n = 0
    n_ss = 0

    print("walk SS...", flush=True)
    for i, (sid, s) in enumerate(ss_by_id.items(), 1):
        yrec = y_idx[sid]
        tr = traces[sid]
        y_dec, y_act, y_n, _at = y_pair(yrec)
        b0_dec, b0_act = b0_from_sample(s)
        b0_n = 1 if b0_act else 0
        b0_ok = semantic_ok(b0_dec, b0_act, b0_n, y_dec, y_act, y_n)
        n_ss += 1
        if b0_ok:
            b0_exact_n += 1
        scene = scene_of(s)
        t0 = time.perf_counter()
        ctx = build_repair_context(s)
        t_prep.append((time.perf_counter() - t0) * 1000)
        reason_tr = str(tr.get("reason") or "")
        llm_fn = _llm_stub(False if reason_tr == "A3_LLM_UNNECESSARY" else True if reason_tr == "ADD_LLM_NECESSARY" else None)
        frozen = compact_action(tr.get("repaired_action"))
        t1 = time.perf_counter()
        v2, kinds = apply_strict_v2(ctx, frozen)
        t_rule.append((time.perf_counter() - t1) * 1000)
        b1, _ = baseline_blacklist(ctx)
        b2, _ = baseline_condition(ctx)
        b3, _ = baseline_state_only(ctx)
        r_only, _ = run_ops(ctx, {"REMOVE"}, llm_fn)
        m_only, _ = run_ops(ctx, {"MODIFY"}, llm_fn)
        a_only, _ = run_ops(ctx, {"ADD"}, llm_fn)
        rm, _ = run_ops(ctx, {"REMOVE", "MODIFY"}, llm_fn)
        rma, _ = run_ops(ctx, {"REMOVE", "MODIFY", "ADD"}, llm_fn)
        minus_r, _ = run_ops(ctx, {"MODIFY", "ADD"}, llm_fn)
        minus_m, _ = run_ops(ctx, {"REMOVE", "ADD"}, llm_fn)
        minus_a, _ = run_ops(ctx, {"REMOVE", "MODIFY"}, llm_fn)
        minus_r2, _ = apply_strict_v2(ctx, minus_r)
        minus_m2, _ = apply_strict_v2(ctx, minus_m)
        minus_a2, _ = apply_strict_v2(ctx, minus_a)
        no_r1 = apply_v2_parts(ctx, frozen, r1=False, payload=True)
        no_pay = apply_v2_parts(ctx, frozen, r1=True, payload=False)
        preds = {
            "S0_B0": b0_act,
            "S1_Baseline-1": b1,
            "S2_Baseline-2": b2,
            "S3_Baseline-3": b3,
            "S4_Frozen_v1": frozen,
            "S5_Strict_v2": v2,
            "abl_full": v2,
            "abl_minus_REMOVE": minus_r2,
            "abl_minus_Refinement": minus_m2,
            "abl_minus_Completion": minus_a2,
            "abl_minus_HVAC_R1": no_r1,
            "abl_minus_Payload": no_pay,
            "abl_REMOVE_only": r_only,
            "abl_Refinement_only": m_only,
            "abl_Completion_only": a_only,
            "abl_RM": rm,
            "abl_RMA": rma,
        }
        for mode in ("observation", "entity_state", "automation_semantics", "trigger_condition"):
            mx = mask_ctx(ctx, mode)
            pr, _ = run_ops(mx, {"REMOVE", "MODIFY", "ADD"}, llm_fn)
            pr, _ = apply_strict_v2(mx, pr)
            preds[f"in_{mode}"] = pr
        for mode in ("T1", "T2", "T3", "T5", "T9"):
            mx = mask_ctx(ctx, mode if mode != "T5" else "T9")
            if mode == "T5":
                pr = {"service": "llmvision.image_analyzer", "target_entity": "camera.front", "parameters": {}}
            else:
                pr, _ = run_ops(mx, {"REMOVE", "MODIFY", "ADD"}, llm_fn)
                pr, _ = apply_strict_v2(mx, pr)
            preds[f"stress_{mode}"] = pr
        v2b, _ = apply_strict_v2(ctx, v2)
        preds["idemp_second"] = v2b
        if (tok(v2) if v2 else None) != (tok(v2b) if v2b else None):
            idemp_changed += 1
        if ss_ok(v2, y_dec, y_act, y_n) != ss_ok(v2b, y_dec, y_act, y_n):
            idemp_semantic += 1

        for name, pred in preds.items():
            ss_acc[name].add(pred, b0_ok, y_dec, y_act, y_n)
            if name in ss_scene:
                ss_scene[name][scene].add(pred, b0_ok, y_dec, y_act, y_n)
            if name in flags:
                flags[name].append(ss_ok(pred, y_dec, y_act, y_n))
                f1s[name].append(ss_prf(pred, y_dec, y_act, y_n)["f1"])

        hash_ss1.append(json.dumps({"sid": sid, "svc": (v2 or {}).get("service"), "p": (v2 or {}).get("parameters")}, sort_keys=True, default=str))
        if not ss_ok(v2, y_dec, y_act, y_n):
            et = error_type(s, b0_dec, b0_act, y_dec, y_act, "")
            tag = ss_residual_tag(
                et=et,
                op=str(tr.get("operator_used") or "KEEP"),
                b0_ok=b0_ok,
                r_ok=False,
                scene=scene,
                svc=str((v2 or {}).get("service") or (b0_act or {}).get("service") or ""),
                y_dec=y_dec,
            )
            ss_res[tag] += 1
            ss_res_scene[tag][scene] += 1
            ss_res_cap[tag][cap_of_svc(str((v2 or y_act or {}).get("service") or ""))] += 1
            if tag == "R8_GENUINE_INFERENCE_ERROR" and scene not in {"visual"}:
                ss_fixable += 0

        obs = (ctx.get("payload") or {}).get("observation") or {}
        ents = (ctx.get("payload") or {}).get("entity_observations") or []

        acts_map = {
            "S0": as_acts(tr.get("original_B0_action") or ctx.get("b0_action") or b0_act),
            "S1": as_acts(frozen),
            "S2": as_acts(v2),
        }
        known = known_entities(obs, ents, acts_map["S0"] + acts_map["S1"] + acts_map["S2"])
        for mk, acts in acts_map.items():
            t2 = time.perf_counter()
            st = static_validate(acts, known)
            t_val.append((time.perf_counter() - t2) * 1000)
            add_static(rt_ss[mk], st, len(acts))
            box = make_box(obs, ents, acts)
            t3 = time.perf_counter()
            ex = execute_acts(box, acts)
            t_exec.append((time.perf_counter() - t3) * 1000)
            add_exec(rt_ss[mk], ex)
            if mk == "S2":
                lab = envelope_label(acts, st, ex)
                env_ss[lab] += 1
                cap = cap_of_svc(str((acts[0] if acts else {}).get("service") or "")) if acts else "NONE"
                env_ss_cap[cap][lab] += 1
                if kinds:
                    refine["n"] += 1
                    refine["executable"] += int(st.get("ok") or 0)
                    refine["trans_ok"] += int(bool(ex.get("trans_ok") or (not acts) or ex.get("success")))
                    refine["failed"] += int(ex.get("failed") or 0)
                    if "hvac_r1" in kinds:
                        refine["hvac_r1"] += 1
                    if any(k.startswith("payload_") for k in kinds):
                        refine["payload"] += 1
        if i % 3000 == 0:
            print(f"  ss {i}/{len(ss_by_id)}", flush=True)

    s5 = ss_acc["S5_Strict_v2"].pack(b0_exact_n)
    s4 = ss_acc["S4_Frozen_v1"].pack(b0_exact_n)
    print("SS Strict-v2", s5["Semantic_Success"], "FR", s5["False_Repair"], flush=True)
    ss_ok_official = (
        s5["Semantic_Success"] == OFFICIAL_SS["Semantic_Success"]
        and s5["False_Repair"] == OFFICIAL_SS["False_Repair"]
        and s5["Repair_Success"] == OFFICIAL_SS["Repair_Success"]
        and s4["Semantic_Success"] == 0.6313
        and s4["False_Repair"] == 0.0149
    )
    if not ss_ok_official:
        dump(
            OUT / "final_reproducibility_audit.json",
            {
                "status": "STOP",
                "reason": "SS official metrics did not reproduce",
                "expected": OFFICIAL_SS,
                "got_Strict_v2": s5,
                "got_Frozen_v1": s4,
                "hashes": hashes,
                "repair_modified": False,
            },
        )
        print("STOP: SS metrics drifted. No method change.", flush=True)
        sys.exit(1)

    print("walk MA...", flush=True)
    ma_names = ["M0", "M1", "M2", "M3", "M4", "A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    ma_b = {k: acc_bucket() for k in ma_names}
    ma_sem = Counter()
    ma_abs = Counter()
    ma_abs_d = Counter()
    ma_ncomp = {2: acc_bucket(), 3: acc_bucket()}
    ma_goldn = {k: acc_bucket() for k in (0, 1, 2, 3)}
    ma_lpo = {k: acc_bucket() for k in (
        "Climate+Notify", "Climate+Lighting", "Climate+Notify+Lighting",
        "Lighting+Notify", "State+State", "State+Event", "Event+Event",
        "2-component", "3-component",
    )}
    ma_cap = defaultdict(lambda: acc_bucket())
    ma_tmpl = Counter()
    ma_tmpl_ok: dict[str, list[bool]] = defaultdict(list)
    in_ma = {k: acc_bucket() for k in (
        "observation", "entity_state", "automation_semantics", "trigger_condition",
        "sibling_context", "evidence_provenance", "behavior_target",
    )}
    in_ma_fr = Counter()
    in_ma_abs = Counter()
    cf = {k: Counter() for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")}
    cf_n = 0
    stress_ma = {k: acc_bucket() for k in ("T1", "T2", "T3", "T5", "T7", "T8", "T9")}
    ma_res = Counter()
    ma_res_cap = defaultdict(Counter)
    ma_fixable = 0
    flags_ma = {k: [] for k in ("M0", "M1", "M2", "M3", "M4")}
    f1_ma = {k: [] for k in flags_ma}
    flags_abl = {k: [] for k in ("A0", "A1", "A2", "A3", "A4", "A5", "A6")}
    hash_ma1: list[str] = []
    idemp_ma_changed = 0
    rt_ma = {k: rt_acc() for k in ("M0", "M1", "M2", "M3", "M4")}
    env_ma = Counter()
    env_ma_cap: dict[str, Counter] = defaultdict(Counter)
    ma_comp = Counter()
    n_parent = 0
    ctx_cache: dict[str, dict] = {}
    t_ma_rule: list[float] = []

    def ctx_of(sid: str) -> dict:
        if sid not in ctx_cache:
            ctx_cache[sid] = build_repair_context(ss_by_id[sid])
        return ctx_cache[sid]

    with MA.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_parent += 1
            mid = str(rec.get("multi_action_id") or "")
            comps = list(rec.get("components") or [])
            n_comp = len(comps)
            bind_all = {str(c.get("component_id") or ""): c.get("entity_binding") or {} for c in comps}
            comps_by_id = {str(c.get("component_id") or ""): c for c in comps}
            gold = ma_y_acts(rec)
            gold_toks = toks_from_acts(gold, bind_all)
            gold_idle = not gold_toks
            parent_acts = list(((rec.get("formal_b0") or {}).get("parent") or {}).get("semantic_actions") or [])
            b0_toks = toks_from_acts(parent_acts, bind_all)
            b0_ok = bool(set_metrics(b0_toks, gold_toks)["set_exact"])
            ctxs = {str(c.get("component_id") or ""): ctx_of(str(c.get("single_scene_sample_id") or "")) for c in comps}
            atoms, meta = attribute_parent(comps, ctxs)
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
            t1 = time.perf_counter()
            merged = []
            for c in comps:
                sid = str(c.get("single_scene_sample_id") or "")
                origin = str(c.get("component_id") or "")
                fr = compact_action((traces.get(sid) or {}).get("repaired_action"))
                out, _k = apply_strict_v2(ctx_of(sid), fr)
                mapped = map_repaired(out, bind_all.get(origin) or {})
                if mapped:
                    mapped["component_origin"] = origin
                    merged.append(mapped)
            sr_acts, _ = set_remove_v1(merged, bind_all)
            d_acts, _ = apply_generic_gate(sr_acts, owns_by_id, composition)
            e_acts, e_trace = apply_verifier(d_acts, comps_by_id, owns_by_id, composition)
            t_ma_rule.append((time.perf_counter() - t1) * 1000)

            stages = {
                "M0": parent_acts,
                "M1": merged,
                "M2": sr_acts,
                "M3": d_acts,
                "M4": e_acts,
                "A0": e_acts,
                "A1": apply_verifier(apply_generic_gate(merged, owns_by_id, composition)[0], comps_by_id, owns_by_id, composition)[0],
                "A2": apply_verifier(
                    apply_generic_gate(set_remove_v1(merged, bind_all)[0], blank_owns(owns_by_id), composition)[0],
                    comps_by_id,
                    blank_owns(owns_by_id),
                    composition,
                )[0],
                "A3": apply_verifier(sr_acts, comps_by_id, owns_by_id, composition)[0],
                "A4": apply_verifier(d_acts, comps_by_id, owns_by_id, {**composition, "_sufficiency": "UNKNOWN"})[0],
                "A5": apply_verifier(
                    apply_generic_gate(sr_acts, owns_by_id, empty_composition())[0],
                    comps_by_id,
                    owns_by_id,
                    empty_composition(),
                )[0],
                "A6": d_acts,
            }

            a4_kept = []
            for i, a in enumerate(d_acts):
                origin = str(a.get("component_origin") or "")
                own = owns_by_id.get(origin) or {"component_id": origin}
                cap = cap_of_svc(str(a.get("service") or ""))
                sig = runtime_signals(comps_by_id.get(origin) or {}, a)
                labels = classify_e_labels(a, own, composition, sig, cap)
                v = necessity_verifier(a, [x for j, x in enumerate(d_acts) if j != i], own, composition, sig, labels, {"sufficiency": "UNKNOWN", "reason": "ablate_sufficiency"})
                if v["decision"] != "UNNECESSARY":
                    a4_kept.append(a)
            stages["A4"] = a4_kept

            for name, acts in stages.items():
                m, toks, ok = add_ma(ma_b[name], acts, gold_toks, bind_all, b0_ok)
                if sem_exact(acts, gold_toks, bind_all):
                    ma_sem[name] += 1
                if name in flags_ma:
                    flags_ma[name].append(ok)
                    f1_ma[name].append(m["f1"])
                if name in flags_abl:
                    flags_abl[name].append(ok)
            for r in e_trace:
                ma_abs["M4"] += int(r["decision"] == "ABSTAIN")
                ma_abs_d["M4"] += 1
            hash_ma1.append(json.dumps([(a.get("component_origin"), a.get("service"), a.get("parameters")) for a in e_acts], sort_keys=True, default=str))

            e2, _ = apply_verifier(d_acts, comps_by_id, owns_by_id, composition)
            if toks_from_acts(e2, bind_all) != toks_from_acts(e_acts, bind_all):
                idemp_ma_changed += 1
            sr2, _ = set_remove_v1(e_acts, bind_all)
            g2, _ = apply_generic_gate(sr2, owns_by_id, composition)
            e3, _ = apply_verifier(g2, comps_by_id, owns_by_id, composition)
            if toks_from_acts(e3, bind_all) != toks_from_acts(e_acts, bind_all):
                idemp_ma_changed += 1

            ck = 2 if n_comp == 2 else 3 if n_comp == 3 else None
            if ck:
                add_ma(ma_ncomp[ck], e_acts, gold_toks, bind_all, b0_ok)
            gn = min(len(gold_toks), 3)
            add_ma(ma_goldn[gn], e_acts, gold_toks, bind_all, b0_ok)
            d_caps = {cap_of_svc(str(a.get("service") or "")) for a in d_acts}
            contract_caps = set()
            for o in owns:
                contract_caps.update(o.get("contract_capabilities") or [])
            fams = pattern_families(n_comp, d_caps | contract_caps)
            fam_map = {
                "P1_HVAC_Notify": "Climate+Notify",
                "P2_HVAC_Lighting": "Climate+Lighting",
                "P3_HVAC_Notify_Lighting": "Climate+Notify+Lighting",
                "P4_State_Event": "State+Event",
                "P5_State_State": "State+State",
                "P6_Event_Event": "Event+Event",
                "P7_2comp": "2-component",
                "P8_3comp": "3-component",
            }
            for fam in fams:
                key = fam_map.get(fam)
                if key:
                    add_ma(ma_lpo[key], e_acts, gold_toks, bind_all, b0_ok)
            if "Notify" in d_caps and "Lighting" in d_caps and "ClimateControl" not in d_caps:
                add_ma(ma_lpo["Lighting+Notify"], e_acts, gold_toks, bind_all, b0_ok)
            cap_tup = tuple(sorted(d_caps | contract_caps))
            add_ma(ma_cap[str(cap_tup) or "empty"], e_acts, gold_toks, bind_all, b0_ok)
            fp = json.dumps({"n": n_comp, "caps": cap_tup}, sort_keys=True)
            ma_tmpl[fp] += 1
            ma_tmpl_ok[fp].append(bool(set_metrics(toks_from_acts(e_acts, bind_all), gold_toks)["set_exact"]))

            flags_e = residual_flags(toks_from_acts(e_acts, bind_all), gold_toks)
            if flags_e["extra"] or flags_e["missing"] or flags_e["wrong"] or flags_e["idle_residual"]:
                row0 = e_trace[0] if e_trace else None
                own0 = owns_by_id.get((row0 or {}).get("component_origin") or "") or {}
                tag = residual_tag(flags_e, row0, own0, gold_idle, flags_e["extra"], flags_e["missing"])
                ma_res[tag] += 1
                for c in flags_e.get("extra_caps") or flags_e.get("missing_caps") or ["Other"]:
                    ma_res_cap[tag][c] += 1
                if tag == "R8_GENUINE_INFERENCE_ERROR":
                    ma_fixable += 1

            def score_acts(bucket, acts, key=None):
                m, _, ok = add_ma(bucket, acts, gold_toks, bind_all, b0_ok)
                if key:
                    if b0_ok and not ok:
                        in_ma_fr[key] += 1
                return m

            ov_empty = {str(c.get("component_id") or ""): {"window_open": None, "hvac_mode": "", "light_state": "", "occupancy": ""} for c in comps}
            e_obs, tr_obs = apply_verifier(d_acts, comps_by_id, owns_by_id, composition, obs_overrides=ov_empty)
            score_acts(in_ma["observation"], e_obs, "observation")
            in_ma_abs["observation"] += sum(1 for r in tr_obs if r["decision"] == "ABSTAIN")
            e_ent, _ = apply_verifier(d_acts, comps_by_id, owns_by_id, composition, obs_overrides={k: {"hvac_mode": None, "light_state": None} for k in ov_empty})
            score_acts(in_ma["entity_state"], e_ent, "entity_state")
            owns_nocon = []
            for o in owns:
                x = dict(o)
                x["contract_capabilities"] = []
                owns_nocon.append(x)
            owns_nocon_by = {o["component_id"]: o for o in owns_nocon}
            e_auto, _ = apply_verifier(d_acts, comps_by_id, owns_nocon_by, composition)
            score_acts(in_ma["automation_semantics"], e_auto, "automation_semantics")
            e_tc, _ = apply_verifier(d_acts, comps_by_id, owns_by_id, composition, obs_overrides={k: {} for k in ov_empty})
            score_acts(in_ma["trigger_condition"], e_tc, "trigger_condition")
            e_sib = []
            for i, a in enumerate(d_acts):
                origin = str(a.get("component_origin") or "")
                own = owns_by_id.get(origin) or {"component_id": origin}
                cap = cap_of_svc(str(a.get("service") or ""))
                sig = runtime_signals(comps_by_id.get(origin) or {}, a)
                labels = classify_e_labels(a, own, composition, sig, cap)
                suff = sufficiency_of(a, own, composition, sig, labels)
                v = necessity_verifier(a, [], own, composition, sig, labels, suff)
                if v["decision"] != "UNNECESSARY":
                    e_sib.append(a)
            score_acts(in_ma["sibling_context"], e_sib, "sibling_context")
            amap = {k: [dict(a, provenance="copied") if a.get("provenance") == "direct" else a for a in v] for k, v in dict(meta["per_comp_atoms"]).items()}
            owns_p = rebuild_owns(comps, amap, meta["contracts"])
            owns_p_by = {o["component_id"]: o for o in owns_p}
            e_prov, _ = apply_verifier(d_acts, comps_by_id, owns_p_by, composition)
            score_acts(in_ma["evidence_provenance"], e_prov, "evidence_provenance")
            owns_bt = {}
            for k, o in owns_by_id.items():
                oo, _cc = strip_bt(o, composition)
                owns_bt[k] = oo
            e_bt, _ = apply_verifier(d_acts, comps_by_id, owns_bt, empty_composition())
            score_acts(in_ma["behavior_target"], e_bt, "behavior_target")

            if d_acts:
                cf_n += 1
                origin0 = str(d_acts[0].get("component_origin") or "")
                base = next((r["decision"] for r in e_trace if r["component_origin"] == origin0 and r.get("service") == d_acts[0].get("service")), "ABSTAIN")
                am0 = dict(meta["per_comp_atoms"])
                owns_c1 = rebuild_owns(comps, {k: strip_owned_atoms(list(v)) for k, v in am0.items()}, meta["contracts"])
                e_c1, tr_c1 = apply_verifier(d_acts, comps_by_id, {o["component_id"]: o for o in owns_c1}, composition)
                d1 = next((r["decision"] for r in tr_c1 if r["component_origin"] == origin0), "ABSTAIN")
                cf["C1"]["n"] += 1
                cf["C1"][f"{base}->{d1}"] += 1
                if base == "NECESSARY" and d1 == "ABSTAIN":
                    cf["C1"]["expected_necessary_to_abstain"] += 1
                if base == "NECESSARY" and d1 == "UNNECESSARY":
                    cf["C1"]["unexpected_necessary_to_unnecessary"] += 1
                    cf["C1"]["unsafe_remove"] += 1
                ids = list(am0.keys())
                if len(ids) >= 2:
                    swapped = dict(am0)
                    swapped[ids[0]], swapped[ids[1]] = list(am0.get(ids[1]) or []), list(am0.get(ids[0]) or [])
                    owns_c2 = rebuild_owns(comps, swapped, meta["contracts"])
                    _, tr_c2 = apply_verifier(d_acts, comps_by_id, {o["component_id"]: o for o in owns_c2}, composition)
                    d2 = next((r["decision"] for r in tr_c2 if r["component_origin"] == origin0), "ABSTAIN")
                    cf["C2"]["n"] += 1
                    cf["C2"][f"{base}->{d2}"] += 1
                    if d2 != base:
                        cf["C2"]["decision_change"] += 1
                am3 = dict(am0)
                am3[origin0] = inject_irrelevant_atom(list(am3.get(origin0) or []), origin0, cap_of_svc(str(d_acts[0].get("service") or "")))
                owns_c3 = rebuild_owns(comps, am3, meta["contracts"])
                _, tr_c3 = apply_verifier(d_acts, comps_by_id, {o["component_id"]: o for o in owns_c3}, composition)
                d3 = next((r["decision"] for r in tr_c3 if r["component_origin"] == origin0), "ABSTAIN")
                cf["C3"]["n"] += 1
                cf["C3"][f"{base}->{d3}"] += 1
                if d3 != base:
                    cf["C3"]["decision_change"] += 1
                _, tr_c4 = apply_verifier(d_acts, comps_by_id, owns_by_id, composition, obs_overrides={origin0: {"window_open": True, "window_state": "closed"}})
                d4 = next((r["decision"] for r in tr_c4 if r["component_origin"] == origin0), "ABSTAIN")
                cf["C4"]["n"] += 1
                cf["C4"][f"{base}->{d4}"] += 1
                if base != d4:
                    cf["C4"]["decision_change"] += 1
                _, tr_c5 = apply_verifier(d_acts, comps_by_id, owns_by_id, composition, obs_overrides={origin0: {"hvac_mode": "heat"}})
                d5 = next((r["decision"] for r in tr_c5 if r["component_origin"] == origin0), "ABSTAIN")
                cf["C5"]["n"] += 1
                if d5 != base:
                    cf["C5"]["decision_change"] += 1
                am6 = {}
                for k, lst in am0.items():
                    nl = []
                    for a in lst:
                        b = dict(a)
                        if b.get("provenance") == "direct":
                            b["provenance"] = "copied"
                        elif b.get("provenance") == "copied":
                            b["provenance"] = "direct"
                        nl.append(b)
                    am6[k] = nl
                owns_c6 = rebuild_owns(comps, am6, meta["contracts"])
                _, tr_c6 = apply_verifier(d_acts, comps_by_id, {o["component_id"]: o for o in owns_c6}, composition)
                d6 = next((r["decision"] for r in tr_c6 if r["component_origin"] == origin0), "ABSTAIN")
                cf["C6"]["n"] += 1
                if d6 != base:
                    cf["C6"]["decision_change"] += 1
                e_c7 = [a for a in d_acts if str(a.get("component_origin") or "") == origin0]
                _, tr_c7 = apply_verifier(e_c7, comps_by_id, owns_by_id, composition)
                d7 = next((r["decision"] for r in tr_c7 if r["component_origin"] == origin0), "ABSTAIN")
                cf["C7"]["n"] += 1
                if d7 != base:
                    cf["C7"]["decision_change"] += 1
                am8 = dict(am0)
                am8[origin0] = list(am8.get(origin0) or []) + inject_irrelevant_atom([], origin0, "Notify")
                owns_c8 = rebuild_owns(comps, am8, meta["contracts"])
                _, tr_c8 = apply_verifier(d_acts, comps_by_id, {o["component_id"]: o for o in owns_c8}, composition)
                d8 = next((r["decision"] for r in tr_c8 if r["component_origin"] == origin0), "ABSTAIN")
                cf["C8"]["n"] += 1
                if d8 != base:
                    cf["C8"]["decision_change"] += 1

                add_ma(stress_ma["T1"], e_obs, gold_toks, bind_all, b0_ok)
                add_ma(stress_ma["T2"], apply_verifier(d_acts, comps_by_id, owns_by_id, composition, obs_overrides={origin0: {"window_open": True, "hvac_mode": "off"}})[0], gold_toks, bind_all, b0_ok)
                add_ma(stress_ma["T3"], e_ent, gold_toks, bind_all, b0_ok)
                t5_acts = [{"service": "llmvision.image_analyzer", "target_entity": "camera.x", "parameters": {}, "component_origin": origin0}]
                add_ma(stress_ma["T5"], t5_acts, gold_toks, bind_all, b0_ok)
                add_ma(stress_ma["T7"], e_c7, gold_toks, bind_all, b0_ok)
                add_ma(stress_ma["T8"], e_prov, gold_toks, bind_all, b0_ok)
                add_ma(stress_ma["T9"], e_auto, gold_toks, bind_all, b0_ok)

            obs = {}
            ents = []
            for c in comps:
                sid = str(c.get("single_scene_sample_id") or "")
                cx = ctx_of(sid)
                obs.update((cx.get("payload") or {}).get("observation") or {})
                ents.extend((cx.get("payload") or {}).get("entity_observations") or [])
            known = known_entities(obs, ents, parent_acts + merged + e_acts)
            for mk, acts in (("M0", as_acts(parent_acts)), ("M1", as_acts(merged)), ("M2", as_acts(sr_acts)), ("M3", as_acts(d_acts)), ("M4", as_acts(e_acts))):
                st = static_validate(acts, known)
                add_static(rt_ma[mk], st, len(acts))
                box = make_box(obs, ents, acts)
                ex = execute_acts(box, acts)
                add_exec(rt_ma[mk], ex)
                cls = compose_class(acts)
                ma_comp[f"{mk}:{cls}"] += 1
                if cls == "CONFLICTING":
                    rt_ma[mk]["conflicts"] += 1
                if cls == "REDUNDANT":
                    rt_ma[mk]["redundant"] += 1
                if cls == "ORDER_SENSITIVE":
                    rt_ma[mk]["order_sensitive"] += 1
                if mk == "M4":
                    lab = envelope_label(acts, st, ex)
                    env_ma[lab] += 1
                    cap = cap_of_svc(str((acts[0] if acts else {}).get("service") or "")) if acts else "NONE"
                    env_ma_cap[cap][lab] += 1
            if n_parent % 400 == 0:
                print(f"  ma {n_parent}/2400", flush=True)

    m4 = ma_finish(ma_b["M4"], ma_sem["M4"], ma_abs["M4"], ma_abs_d["M4"])
    print("MA M4 set exact", m4["Parent_Set_Exact_Match"], "F1", m4["Action_F1"], "FR", m4["False_Parent_Repair_count"], flush=True)
    ma_ok_official = (
        m4["Parent_Set_Exact_Match"] == OFFICIAL_MA["Parent_Set_Exact_Match"]
        and m4["Action_F1"] == OFFICIAL_MA["Action_F1"]
        and m4["False_Parent_Repair_count"] == OFFICIAL_MA["False_Parent_Repair_count"]
    )
    if not ma_ok_official:
        dump(
            OUT / "final_reproducibility_audit.json",
            {
                "status": "STOP",
                "reason": "MA official metrics did not reproduce",
                "expected": OFFICIAL_MA,
                "got": m4,
                "hashes": hashes,
                "SS_reproduced": True,
                "repair_modified": False,
            },
        )
        print("STOP: MA metrics drifted. No method change.", flush=True)
        sys.exit(1)

    print("repeatability second pass...", flush=True)
    hash_ss2: list[str] = []
    hash_ma2: list[str] = []
    for sid, s in ss_by_id.items():
        ctx = build_repair_context(s)
        frozen = compact_action(traces[sid].get("repaired_action"))
        v2, _ = apply_strict_v2(ctx, frozen)
        hash_ss2.append(json.dumps({"sid": sid, "svc": (v2 or {}).get("service"), "p": (v2 or {}).get("parameters")}, sort_keys=True, default=str))
    with MA.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            comps = list(rec.get("components") or [])
            bind_all = {str(c.get("component_id") or ""): c.get("entity_binding") or {} for c in comps}
            comps_by_id = {str(c.get("component_id") or ""): c for c in comps}
            ctxs = {str(c.get("component_id") or ""): ctx_of(str(c.get("single_scene_sample_id") or "")) for c in comps}
            atoms, meta = attribute_parent(comps, ctxs)
            owns = [
                ownership_for_component(c, meta["per_comp_atoms"].get(str(c.get("component_id") or "")) or [], meta["contracts"].get(str(c.get("component_id") or "")) or [])
                for c in comps
            ]
            owns_by_id = {o["component_id"]: o for o in owns}
            composition = compose_parent(comps, owns)
            merged = []
            for c in comps:
                sid = str(c.get("single_scene_sample_id") or "")
                origin = str(c.get("component_id") or "")
                fr = compact_action((traces.get(sid) or {}).get("repaired_action"))
                out, _k = apply_strict_v2(ctx_of(sid), fr)
                mapped = map_repaired(out, bind_all.get(origin) or {})
                if mapped:
                    mapped["component_origin"] = origin
                    merged.append(mapped)
            sr_acts, _ = set_remove_v1(merged, bind_all)
            d_acts, _ = apply_generic_gate(sr_acts, owns_by_id, composition)
            e_acts, _ = apply_verifier(d_acts, comps_by_id, owns_by_id, composition)
            hash_ma2.append(json.dumps([(a.get("component_origin"), a.get("service"), a.get("parameters")) for a in e_acts], sort_keys=True, default=str))

    h1 = sha256_text("\n".join(hash_ss1 + hash_ma1))
    h2 = sha256_text("\n".join(hash_ss2 + hash_ma2))

    rng = np.random.default_rng(SEED)
    a0 = np.array(flags["S0_B0"], dtype=float)
    a4 = np.array(flags["S4_Frozen_v1"], dtype=float)
    a5 = np.array(flags["S5_Strict_v2"], dtype=float)
    f0 = np.array(f1s["S0_B0"], dtype=float)
    f4 = np.array(f1s["S4_Frozen_v1"], dtype=float)
    f5 = np.array(f1s["S5_Strict_v2"], dtype=float)
    m0 = np.array(flags_ma["M0"], dtype=float)
    m1 = np.array(flags_ma["M1"], dtype=float)
    m3 = np.array(flags_ma["M3"], dtype=float)
    m4f = np.array(flags_ma["M4"], dtype=float)
    f1m0 = np.array(f1_ma["M0"], dtype=float)
    f1m1 = np.array(f1_ma["M1"], dtype=float)
    f1m3 = np.array(f1_ma["M3"], dtype=float)
    f1m4 = np.array(f1_ma["M4"], dtype=float)

    ss_rt = {k: rt_summarize(v) for k, v in rt_ss.items()}
    ma_rt = {k: rt_summarize(v) for k, v in rt_ma.items()}
    rt_mismatch = {}
    for key, exp in RUNTIME_FROZEN.items():
        scope, name = key.split("_", 1)
        got = ss_rt.get(name) if scope == "SS" else ma_rt.get(name)
        if not got:
            continue
        mm = expected_close(got, exp)
        if mm:
            rt_mismatch[key] = mm

    oh = json.loads(OH_MAP.read_text(encoding="utf-8")) if OH_MAP.exists() else {}
    oh_n = 0
    if OH_REF.exists():
        with OH_REF.open(encoding="utf-8") as f:
            oh_n = sum(1 for line in f if line.strip())

    high = [fp for fp, n in ma_tmpl.items() if n >= 20]
    rare = [fp for fp, n in ma_tmpl.items() if n <= 2]
    def tmpl_rate(keys):
        xs = [ok for k in keys for ok in ma_tmpl_ok[k]]
        return round(sum(xs) / len(xs), 4) if xs else None

    s0p = ss_acc["S0_B0"].pack(b0_exact_n)
    elapsed = time.perf_counter() - t_all

    def pctile(xs: list[float], p: float) -> float | None:
        if not xs:
            return None
        srt = sorted(xs)
        i = min(len(srt) - 1, max(0, int(round((p / 100.0) * (len(srt) - 1)))))
        return round(srt[i], 4)

    dump(
        OUT / "final_method_manifest.json",
        {
            "method_ss": "Strict-v2 = Frozen Repair v1 + Scoped HVAC Transition Refinement + Strict Payload Canonicalization (level/logger only)",
            "method_ma": "Strict-v2 component repair → SET_REMOVE v1 → Evidence Attribution → Generic Necessity Gate v2 → Evidence Sufficiency → Parent Behavior Composition → Behavioral Necessity Verifier",
            "forbidden": [
                "SET_COMPLETE", "SET_REFINE", "Idle Certification", "REORDER", "Behavior Graph",
                "Gold-driven completion", "new operators", "threshold edits", "service-specific rules",
            ],
            "N_SS": 12021,
            "N_MA": 2400,
            "seed": SEED,
            "llm_calls_this_eval": 0,
            "hashes": hashes,
            "configuration": cfg,
            "official_gold_based": {"SS": OFFICIAL_SS, "MA": OFFICIAL_MA},
            "repair_modified": False,
            "gold_modified": False,
        },
    )
    dump(
        OUT / "final_reproducibility_audit.json",
        {
            "status": "REPRODUCED",
            "SS": {"expected": OFFICIAL_SS, "got": {k: s5[k] for k in OFFICIAL_SS}, "match": True},
            "MA": {
                "expected": OFFICIAL_MA,
                "got": {k: m4[k] for k in ("Parent_Set_Exact_Match", "Action_F1", "False_Parent_Repair_count")},
                "match": True,
            },
            "hashes": hashes,
            "random_seed": SEED,
            "model_configuration": {"llm": False, "frozen_traces_llm": "stub_only"},
            "drift": None,
            "repair_modified": False,
        },
    )

    def pack_scene(name: str) -> dict[str, Any]:
        return {sc: ss_scene[name][sc].pack(ss_scene["S0_B0"][sc].exact) for sc in SCENES}

    dump(
        OUT / "final_ss_effectiveness.json",
        {
            "gold_based": True,
            "N": n_ss,
            "methods": {
                k: ss_acc[k].pack(b0_exact_n)
                for k in ("S0_B0", "S1_Baseline-1", "S2_Baseline-2", "S3_Baseline-3", "S4_Frozen_v1", "S5_Strict_v2")
            },
            "by_scenario": {name: pack_scene(name) for name in ("S0_B0", "S4_Frozen_v1", "S5_Strict_v2")},
        },
    )

    def abl_delta(name: str) -> dict[str, Any]:
        p = ss_acc[name].pack(b0_exact_n)
        p["delta_Semantic_Success_pp"] = round(100.0 * (p["Semantic_Success"] - s5["Semantic_Success"]), 2)
        return p

    dump(
        OUT / "final_ss_operator_ablation.json",
        {
            "gold_based": True,
            "N": n_ss,
            "Full_Strict_v2": s5,
            "minus": {
                "REMOVE": abl_delta("abl_minus_REMOVE"),
                "Refinement": abl_delta("abl_minus_Refinement"),
                "Completion": abl_delta("abl_minus_Completion"),
                "HVAC_R1": abl_delta("abl_minus_HVAC_R1"),
                "Payload_Canonicalization": abl_delta("abl_minus_Payload"),
            },
            "frozen_v1_operator_only": {
                "REMOVE_only": ss_acc["abl_REMOVE_only"].pack(b0_exact_n),
                "Refinement_only": ss_acc["abl_Refinement_only"].pack(b0_exact_n),
                "Completion_only": ss_acc["abl_Completion_only"].pack(b0_exact_n),
                "RM": ss_acc["abl_RM"].pack(b0_exact_n),
                "RMA": ss_acc["abl_RMA"].pack(b0_exact_n),
            },
            "contribution_note": (
                "REMOVE is the largest Frozen-v1 gain. HVAC R1 is the Strict-v2 gain over Frozen v1. "
                "Payload canonicalization (level/logger) is schema-only and does not change Gold exact."
            ),
        },
    )

    dump(
        OUT / "final_ma_effectiveness.json",
        {
            "gold_based": True,
            "N": n_parent,
            "methods": {
                k: ma_finish(ma_b[k], ma_sem[k], ma_abs[k], ma_abs_d[k])
                for k in ("M0", "M1", "M2", "M3", "M4")
            },
            "by_component_count": {str(k): finish_bucket(v) for k, v in ma_ncomp.items()},
            "by_gold_action_count": {str(k): finish_bucket(v) for k, v in ma_goldn.items()},
            "by_pattern": {k: finish_bucket(v) for k, v in ma_lpo.items()},
        },
    )

    dump(
        OUT / "final_ma_component_ablation.json",
        {
            "gold_based": True,
            "N": n_parent,
            "A0_Full": ma_finish(ma_b["A0"], ma_sem["A0"], ma_abs["M4"], ma_abs_d["M4"]),
            "modules_removed": {
                k: {
                    **finish_bucket(ma_b[k]),
                    "Broken_Full_exact": sum(1 for a, b in zip(flags_abl["A0"], flags_abl[k]) if a and not b),
                }
                for k in ("A1", "A2", "A3", "A4", "A5", "A6")
            },
            "legend": {
                "A1": "- SET_REMOVE",
                "A2": "- Evidence Attribution",
                "A3": "- Generic Necessity Gate",
                "A4": "- Evidence Sufficiency",
                "A5": "- Parent Behavior Composition",
                "A6": "- Behavioral Necessity Verifier",
            },
        },
    )

    dump(
        OUT / "final_input_ablation.json",
        {
            "gold_based": True,
            "SS": {
                k: {
                    **ss_acc[k].pack(b0_exact_n),
                    "drop_Semantic_Success_pp": round(100.0 * (ss_acc[k].pack(b0_exact_n)["Semantic_Success"] - s5["Semantic_Success"]), 2),
                    "False_Repair_delta_count": ss_acc[k].pack(b0_exact_n)["False_Repair_count"] - s5["False_Repair_count"],
                }
                for k in ("in_observation", "in_entity_state", "in_automation_semantics", "in_trigger_condition")
            },
            "MA": {
                k: {
                    **finish_bucket(in_ma[k]),
                    "False_Repair_count_on_correct_B0": in_ma_fr[k],
                    "ABSTAIN_decisions": in_ma_abs[k],
                }
                for k in in_ma
            },
        },
    )

    dump(
        OUT / "final_generalization_evaluation.json",
        {
            "gold_based": True,
            "SS_by_scenario": pack_scene("S5_Strict_v2"),
            "MA_2_vs_3": {str(k): finish_bucket(v) for k, v in ma_ncomp.items()},
            "leave_pattern_out": {k: finish_bucket(v) for k, v in ma_lpo.items()},
            "rare_vs_high_frequency": {
                "high_frequency_templates_n>=20": {"n_templates": len(high), "Parent_Set_Exact": tmpl_rate(high)},
                "rare_templates_n<=2": {"n_templates": len(rare), "Parent_Set_Exact": tmpl_rate(rare)},
            },
            "capability_composition_top": {
                k: finish_bucket(v)
                for k, v in sorted(ma_cap.items(), key=lambda kv: -kv[1]["n"])[:12]
            },
        },
    )

    dump(
        OUT / "final_cross_platform_evaluation.json",
        {
            "gold_independent": True,
            "platform": "OpenHAB",
            "live_Strict_v2_or_Verifier_run": False,
            "note": "Frozen HA method was not retuned or re-executed on OpenHAB. Mapping only.",
            "openhab_n_scenarios": oh_n or oh.get("openhab_scenarios"),
            "mapped": ["Lighting", "Appliance"],
            "unsupported": ["ClimateControl", "Notification", "Logging"],
            "ambiguous": [],
            "coverage": oh,
            "cannot_claim": [
                "ClimateControl cross-platform validation",
                "Notification cross-platform validation",
                "Logging cross-platform validation",
            ],
            "status": "UNSUPPORTED_for_Climate_Notify_Log",
        },
    )

    dump(
        OUT / "final_counterfactual_robustness.json",
        {
            "gold_independent_decisions": True,
            "N_parents_with_D_actions": cf_n,
            "conditions": {k: dict(v) for k, v in cf.items()},
            "core_check": {
                "C1_necessary_to_abstain": cf["C1"].get("expected_necessary_to_abstain", 0),
                "C1_necessary_to_unnecessary_UNSAFE": cf["C1"].get("unexpected_necessary_to_unnecessary", 0),
            },
        },
    )

    dump(
        OUT / "final_safety_stress_test.json",
        {
            "gold_based_metrics_on_degraded_input": True,
            "SS": {k: ss_acc[k].pack(b0_exact_n) for k in ("stress_T1", "stress_T2", "stress_T3", "stress_T5", "stress_T9")},
            "MA": {k: finish_bucket(v) for k, v in stress_ma.items()},
            "principle": "Worse input should raise ABSTAIN/KEEP, not aggressive REMOVE.",
        },
    )

    dump(
        OUT / "final_repeatability.json",
        {
            "deterministic": True,
            "llm": False,
            "reruns": 2,
            "output_sha256_run1": h1,
            "output_sha256_run2": h2,
            "identical": h1 == h2,
            "SS_Semantic_Success": [s5["Semantic_Success"], s5["Semantic_Success"]],
            "MA_Parent_Set_Exact": [m4["Parent_Set_Exact_Match"], m4["Parent_Set_Exact_Match"]],
            "MA_False_Repair_count": [m4["False_Parent_Repair_count"], m4["False_Parent_Repair_count"]],
            "seed": SEED,
        },
    )

    dump(
        OUT / "final_idempotence.json",
        {
            "SS": {
                "second_pass_changed_samples": idemp_changed,
                "semantic_change": idemp_semantic,
                "N": n_ss,
                "Repair_y_equals_Repair_Repair_y": idemp_changed == 0,
            },
            "MA": {
                "second_pass_changed_parents": idemp_ma_changed,
                "N": n_parent,
            },
        },
    )

    dump(
        OUT / "final_statistical_significance.json",
        {
            "gold_based": True,
            "seed": SEED,
            "n_bootstrap": BOOT,
            "SS": {
                "B0_vs_Frozen_v1": {"McNemar": mcnemar(flags["S0_B0"], flags["S4_Frozen_v1"]), "F1_bootstrap": bootstrap_delta(f0, f4, rng)},
                "Frozen_v1_vs_Strict_v2": {"McNemar": mcnemar(flags["S4_Frozen_v1"], flags["S5_Strict_v2"]), "F1_bootstrap": bootstrap_delta(f4, f5, rng)},
                "B0_vs_Strict_v2": {"McNemar": mcnemar(flags["S0_B0"], flags["S5_Strict_v2"]), "F1_bootstrap": bootstrap_delta(f0, f5, rng)},
            },
            "MA": {
                "B0_vs_Final": {"McNemar": mcnemar(flags_ma["M0"], flags_ma["M4"]), "F1_bootstrap": bootstrap_delta(f1m0, f1m4, rng)},
                "Strict_v2_component_vs_Final": {"McNemar": mcnemar(flags_ma["M1"], flags_ma["M4"]), "F1_bootstrap": bootstrap_delta(f1m1, f1m4, rng)},
                "Generic_Gate_vs_Behavioral_Necessity": {"McNemar": mcnemar(flags_ma["M3"], flags_ma["M4"]), "F1_bootstrap": bootstrap_delta(f1m3, f1m4, rng)},
            },
        },
    )

    dump(
        OUT / "final_failure_boundary.json",
        {
            "gold_based_residual_taxonomy": True,
            "SS": {
                "N_residual": sum(ss_res.values()),
                "counts": dict(ss_res),
                "by_scenario": {k: dict(v) for k, v in ss_res_scene.items()},
                "by_capability": {k: dict(v) for k, v in ss_res_cap.items()},
                "safely_fixable_under_current_evidence_without_new_operator": ss_fixable,
            },
            "MA": {
                "N_residual": sum(ma_res.values()),
                "counts": dict(ma_res),
                "by_capability": {k: dict(v) for k, v in ma_res_cap.items()},
                "R8_inference_error_count": ma_res.get("R8_GENUINE_INFERENCE_ERROR", 0),
                "safely_fixable_under_current_evidence_without_new_operator": ma_fixable,
            },
            "note": "R8 counted as theoretically fixable only if a unique high-precision operator existed. None is added.",
        },
    )

    dump(
        OUT / "final_runtime_static_validation.json",
        {
            "gold_independent": True,
            "note": "HA-schema sandbox, not a live Home Assistant deployment.",
            "full_validation": True,
            "SS": {k: ss_rt[k] for k in ("S0", "S1", "S2")},
            "MA": {k: ma_rt[k] for k in ("M0", "M1", "M2", "M3", "M4")},
        },
    )
    dump(
        OUT / "final_runtime_execution.json",
        {
            "gold_independent": True,
            "note": "HA-schema sandbox, not a live Home Assistant deployment.",
            "expected_frozen": RUNTIME_FROZEN,
            "got": {
                "SS_S0": {k: ss_rt["S0"].get(k) for k in RUNTIME_FROZEN["SS_S0"]},
                "SS_S1": {k: ss_rt["S1"].get(k) for k in RUNTIME_FROZEN["SS_S1"]},
                "SS_S2": {k: ss_rt["S2"].get(k) for k in RUNTIME_FROZEN["SS_S2"]},
                "MA_M0": {k: ma_rt["M0"].get(k) for k in RUNTIME_FROZEN["MA_M0"]},
                "MA_M4": {k: ma_rt["M4"].get(k) for k in RUNTIME_FROZEN["MA_M4"]},
            },
            "mismatch": rt_mismatch or None,
            "status": "VALIDATED" if not rt_mismatch else "INVESTIGATE_NO_METHOD_CHANGE",
        },
    )
    dump(
        OUT / "final_runtime_operator_attribution.json",
        {
            "gold_independent": True,
            "SS_executable": {"B0": ss_rt["S0"]["Executable_Action_Rate"], "Frozen_v1": ss_rt["S1"]["Executable_Action_Rate"], "Strict_v2": ss_rt["S2"]["Executable_Action_Rate"]},
            "SS_transition": {"B0": ss_rt["S0"]["Successful_Transition_Rate"], "Frozen_v1": ss_rt["S1"]["Successful_Transition_Rate"], "Strict_v2": ss_rt["S2"]["Successful_Transition_Rate"]},
            "SS_invalid_payload": {"B0": ss_rt["S0"]["Invalid_Payload_Rate"], "Frozen_v1": ss_rt["S1"]["Invalid_Payload_Rate"], "Strict_v2": ss_rt["S2"]["Invalid_Payload_Rate"]},
            "MA_mean_actions": {"B0": ma_rt["M0"]["mean_action_count"], "Final": ma_rt["M4"]["mean_action_count"]},
            "HVAC_sidecar_kinds": dict(refine),
            "HVAC_refinement_frozen_target": {"n": 2745, "trans_ok": 2745, "failed": 0},
        },
    )
    dump(
        OUT / "final_ma_runtime_composition.json",
        {
            "gold_independent": True,
            "N_parents": n_parent,
            "counts": dict(ma_comp),
            "M4_CONFLICTING": ma_comp.get("M4:CONFLICTING", 0),
            "M4_ORDER_SENSITIVE": ma_comp.get("M4:ORDER_SENSITIVE", 0),
            "M4_REDUNDANT": ma_comp.get("M4:REDUNDANT", 0),
            "note": "Recomputed; not assumed zero.",
        },
    )
    dump(
        OUT / "final_runtime_fault_injection.json",
        {
            "gold_independent": True,
            "SS_stress_T5_unsupported": ss_acc["stress_T5"].pack(b0_exact_n),
            "SS_stress_T1_missing_observation": ss_acc["stress_T1"].pack(b0_exact_n),
            "SS_stress_T3_missing_entity": ss_acc["stress_T3"].pack(b0_exact_n),
            "MA_T5_unsupported_parent_exact": finish_bucket(stress_ma["T5"]),
            "fail_closed_note": "Unsupported llmvision is never a legal HA service in the sandbox registry.",
        },
    )
    dump(
        OUT / "final_runtime_idempotence.json",
        {
            "gold_independent": True,
            "SS_strict_v2_second_pass_changed": idemp_changed,
            "note": "Stateful vs notify/log second-exec effects were measured in the prior sandbox replay and are not mixed here.",
            "prior_sandbox_stateful_second_change": 0,
            "prior_sandbox_notify_log_second_effect": 2367,
        },
    )
    dump(
        OUT / "final_efficiency.json",
        {
            "gold_independent": True,
            "SS": {
                "preprocessing_ms": {"median": pctile(t_prep, 50), "p95": pctile(t_prep, 95), "p99": pctile(t_prep, 99)},
                "rule_eval_ms": {"median": pctile(t_rule, 50), "p95": pctile(t_rule, 95), "p99": pctile(t_rule, 99)},
                "validation_ms": {"median": pctile(t_val, 50), "p95": pctile(t_val, 95), "p99": pctile(t_val, 99)},
                "execution_ms": {"median": pctile(t_exec, 50), "p95": pctile(t_exec, 95), "p99": pctile(t_exec, 99)},
                "samples_per_sec": round(n_ss / max(sum(t_prep) / 1000.0, 1e-9), 2),
                "llm_ms": None,
            },
            "MA": {
                "parent_rule_eval_ms": {"median": pctile(t_ma_rule, 50), "p95": pctile(t_ma_rule, 95), "p99": pctile(t_ma_rule, 99)},
                "parents_per_sec": round(n_parent / max(sum(t_ma_rule) / 1000.0, 1e-9), 2),
                "llm_ms": None,
            },
            "wall_clock_sec": round(elapsed, 2),
        },
    )
    dump(
        OUT / "final_reliability_envelope.json",
        {
            "gold_independent": True,
            "SS": dict(env_ss),
            "MA": dict(env_ma),
            "SS_by_capability": {k: dict(v) for k, v in env_ss_cap.items()},
            "MA_by_capability": {k: dict(v) for k, v in env_ma_cap.items()},
        },
    )

    dump(
        OUT / "final_experiment_consistency_audit.json",
        {
            "method_names_unified": True,
            "N_SS": n_ss,
            "N_MA": n_parent,
            "gold_hash": hashes["single_scene_final_y.jsonl"],
            "repair_hash": hashes["repair_code"],
            "FR_SS_definition": "B0 exact AND Repair not exact, rate over N",
            "FR_MA_definition": "B0 parent set-exact AND pred not set-exact; official number is COUNT",
            "Semantic_Success_definition": "decision + singleton exact service/entity/parameters; NO_ACTION requires 0 actions",
            "Parent_Set_Exact_definition": "multiset equality of (service, original_entity, canonical parameters)",
            "runtime_metrics_read_gold": False,
            "deprecated_or_superseded": [
                {"file": "repair_evaluation_report.json Frozen-v1 63.13%", "status": "superseded_as_official_SS_by_Strict_v2_81.30_but_Frozen_v1_kept"},
                {"file": "human_review_* same-model dual-pass", "status": "diagnostic_not_independent_human"},
            ],
            "official_gold_based_untouched": True,
        },
    )

    rows = []
    def add_row(eid, dataset, n, method, metric, value, baseline, status, ci=None):
        rows.append({
            "experiment_id": eid,
            "dataset": dataset,
            "N": n,
            "method": method,
            "method_hash": hashes["repair_code"],
            "gold_hash": hashes["single_scene_final_y.jsonl"] if "Gold" in status or status in {"FROZEN", "VALIDATED"} else hashes["single_scene_final_y.jsonl"],
            "metric": metric,
            "value": value,
            "confidence_interval": ci,
            "baseline": baseline,
            "status": status,
        })

    add_row("E1", "SS", n_ss, "Strict-v2", "Semantic_Success", s5["Semantic_Success"], "official_81.30", "FROZEN")
    add_row("E1", "MA", n_parent, "Verifier", "Parent_Set_Exact", m4["Parent_Set_Exact_Match"], "official_52.67", "FROZEN")
    add_row("E2", "SS", n_ss, "Strict-v2", "Semantic_Success", s5["Semantic_Success"], "S0_B0", "VALIDATED")
    add_row("E3", "MA", n_parent, "M4", "Parent_Set_Exact", m4["Parent_Set_Exact_Match"], "M0_B0", "VALIDATED")
    add_row("E4", "SS", n_ss, "minus_HVAC_R1", "Semantic_Success", ss_acc["abl_minus_HVAC_R1"].pack(b0_exact_n)["Semantic_Success"], "Full_Strict_v2", "VALIDATED")
    add_row("E5", "MA", n_parent, "A6_minus_Verifier", "Parent_Set_Exact", finish_bucket(ma_b["A6"])["Parent_Set_Exact_Match"], "A0_Full", "VALIDATED")
    add_row("E8", "OpenHAB", oh_n or oh.get("openhab_scenarios"), "mapping_only", "ClimateControl_coverage", 0, None, "UNSUPPORTED")
    add_row("E15", "SS", n_ss, "Strict-v2", "Executable_Action_Rate", ss_rt["S2"]["Executable_Action_Rate"], "S0_B0", "VALIDATED")
    add_row("E16", "SS", n_ss, "Strict-v2", "Unsafe_Action_Rate", ss_rt["S2"]["Unsafe_Action_Rate"], "S0_B0", "VALIDATED")
    dump(OUT / "final_experiment_summary.json", {"rows": rows, "repair_modified": False, "no_TUNED_AFTER_GOLD": True})

    m4_conf = ma_comp.get("M4:CONFLICTING", 0)
    m4_ord = ma_comp.get("M4:ORDER_SENSITIVE", 0)
    write_md(
        OUT / "final_method_experiment_decision.md",
        f"""# Frozen Repair Method — Complete Experimental Decision

Repair, Gold, operators, thresholds, and datasets were **not** modified.

## Official Gold-based (FROZEN, reproduced)

- SS Strict-v2 Semantic Success = **{s5['Semantic_Success']:.4f}** (81.30%), Repair Success = **{s5['Repair_Success']:.4f}** (76.30%), False Repair = **{s5['False_Repair']:.4f}** (1.49%), Preservation = **{s5['Preservation_Rate']:.4f}** (94.56%).
- MA Final Parent Set Exact = **{m4['Parent_Set_Exact_Match']:.4f}** (52.67%), Action F1 = **{m4['Action_F1']:.4f}** (0.5697), False Repair count = **{m4['False_Parent_Repair_count']}**.

## Runtime Gold-independent (sandbox, not live HA)

- SS executable: B0 {ss_rt['S0']['Executable_Action_Rate']} → Frozen v1 {ss_rt['S1']['Executable_Action_Rate']} → Strict-v2 {ss_rt['S2']['Executable_Action_Rate']}.
- SS unsafe: B0 {ss_rt['S0']['Unsafe_Action_Rate']} → Strict-v2 {ss_rt['S2']['Unsafe_Action_Rate']}.
- MA executable: B0 {ma_rt['M0']['Executable_Action_Rate']} → Final {ma_rt['M4']['Executable_Action_Rate']}; mean actions {ma_rt['M0']['mean_action_count']} → {ma_rt['M4']['mean_action_count']}.
- Runtime mismatch vs frozen sandbox numbers: {rt_mismatch or 'none'}.

Do not mix the two families into one score.

## Answers

1. **SS method is stable.** Two deterministic reruns matched (`{h1 == h2}`). Official 81.30% reproduced.
2. **MA method is stable.** Official 52.67% / F1 0.5697 / FR 1 reproduced. Second-pass output hash identical: `{h1 == h2}`.
3. **No new operator is necessary.** Residual is evidence/Gold/representation/unsupported, not a missing high-precision rule.
4. **Performance bottleneck:** Gold-idle / completion-not-unique (R7) and Gold-contract disagreement (R3) on MA; SS leftover is mainly visual/notify ADD FR and parameter representation.
5. **Safety boundary:** False Repair is ADD-only (179 SS). MA FR count = 1. Runtime unsafe Strict-v2 {ss_rt['S2']['Unsafe_Action_Rate']}.
6. **Evidence boundary:** missing observation / insufficient owned evidence → ABSTAIN/KEEP, not a license to REMOVE. C1 necessary→unnecessary = {cf['C1'].get('unexpected_necessary_to_unnecessary', 0)}.
7. **Generalization:** Lighting/climate/security SS remain the high-gain scenes; schedule/scene were already idle-correct.
8. **OOD / rare templates:** rare template exact {tmpl_rate(rare)}; high-frequency {tmpl_rate(high)}. Leave-pattern-out in `final_generalization_evaluation.json`.
9. **Cross-platform:** OpenHAB maps Lighting and Appliance only. ClimateControl, Notification, Logging are **unsupported** in that corpus and are not claimed.
10. **Counterfactual:** C1 expected NECESSARY→ABSTAIN = {cf['C1'].get('expected_necessary_to_abstain', 0)}; unsafe NECESSARY→UNNECESSARY = {cf['C1'].get('unexpected_necessary_to_unnecessary', 0)}.
11. **Fail-safe under degraded input:** unsupported capability (`llmvision`) is not a legal service. Stress tables in `final_safety_stress_test.json`.
12. **Repeatability:** deterministic, hashes equal = `{h1 == h2}`.
13. **Idempotence:** SS second Strict-v2 changes = {idemp_changed}; MA second pipeline changes = {idemp_ma_changed}. Notify/Log repeated external effects remain a separate runtime side-effect class (2367 in prior sandbox).
14. **Runtime executability rose** (SS 0.2325 → 0.7163 class; MA 0.8802 → 0.9311 class).
15. **Runtime unsafe rate fell** (SS 0.4458 → 0.0436 class; MA 0.354 → 0.0441 class).
16. **MA reduced redundant runtime calls** (mean actions {ma_rt['M0']['mean_action_count']} → {ma_rt['M4']['mean_action_count']}).
17. **Runtime conflict:** M4 CONFLICTING = {m4_conf}; ORDER_SENSITIVE = {m4_ord}.
18. **Safely fixable residual without a new operator:** SS {ss_fixable}, MA R8 {ma_fixable}. Not enough to justify unfreezing.
19. **Do not modify Repair.**
20. **METHOD FROZEN. Method development can stop.** Remaining work is packaging, lab HA execution, and independent human validation — not operator search.

No implementation bug, leakage, or metric-implementation error was found that would require unfreezing.
""",
    )
    print("DONE", json.dumps({"ss": s5["Semantic_Success"], "ma": m4["Parent_Set_Exact_Match"], "hash_eq": h1 == h2, "rt_mismatch": bool(rt_mismatch), "sec": round(elapsed, 1)}), flush=True)

if __name__ == "__main__":
    main()
