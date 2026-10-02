from __future__ import annotations

import json
import sys
from copy import deepcopy
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
)
from eval_ma_evidence_sufficiency import apply_verifier
from eval_ss_b0_vs_final_gold_y import SS, Y_PATH
from eval_ss_trhr_final import load_jsonl, metrics_block, pair_from_act, semantic_ok, y_pair
from eval_ss_trhr_v2_offline import apply_r1, climate_capability, r1_should_fire
from eval_strict_v2_set_remove import (
    MA,
    acc_bucket,
    add_bucket,
    apply_payload_strict,
    finish_bucket,
    ma_y_acts,
    map_repaired,
    set_metrics,
    set_remove_v1,
    tok,
)
from smarthome_mdf.ss_trhr_repair.actions import compact_action
from smarthome_mdf.ss_trhr_repair.add import run_add
from smarthome_mdf.ss_trhr_repair.context import build_repair_context
from smarthome_mdf.ss_trhr_repair.modify import run_modify
from smarthome_mdf.ss_trhr_repair.pipeline import _illegal
from smarthome_mdf.ss_trhr_repair.remove import run_remove

OUT = ROOT / "results" / "ablation.json"

ENV_OBS_KEYS = frozenset(
    {
        "illuminance_lux",
        "illuminance",
        "motion_state",
        "motion",
        "window_state",
        "door_state",
        "occupancy",
        "presence_state",
        "quiet_hours",
        "silent_period",
        "camera_presence",
        "person_detected",
        "human_count",
    }
)

VARIANTS = [
    "Full",
    "w/o Execution Instance Modeling",
    "w/o Runtime Context",
    "w/o Contract Constraints",
    "w/o Behavioral Assessment",
    "w/o Remove",
    "w/o Modify",
    "w/o Recover",
    "w/o Provenance-Aware Interaction",
]

SS_VARIANTS = [name for name in VARIANTS if name != "w/o Provenance-Aware Interaction"]

def strip_execution_instance(ctx: dict[str, Any]) -> dict[str, Any]:
    c = dict(ctx)
    p = dict(c.get("payload") or {})
    p["observation"] = {}
    p["entity_observations"] = []
    c["payload"] = p
    return c

def strip_runtime_context(ctx: dict[str, Any]) -> dict[str, Any]:
    c = dict(ctx)
    p = dict(c.get("payload") or {})
    obs = dict(p.get("observation") or {}) if isinstance(p.get("observation"), dict) else {}
    ents = []
    for k in list(obs.keys()):
        if k in ENV_OBS_KEYS:
            obs.pop(k, None)
    for eo in p.get("entity_observations") or []:
        if not isinstance(eo, dict):
            continue
        row = dict(eo)
        eid = str(row.get("entity_id") or "").lower()
        attrs = dict(row.get("attributes") or {}) if isinstance(row.get("attributes"), dict) else {}
        if any(x in eid for x in ("window", "motion", "occupancy", "illuminance", "lux", "door")):
            continue
        for ak in list(attrs.keys()):
            akl = str(ak).lower()
            if any(x in akl for x in ("window", "motion", "occupancy", "illuminance", "lux", "door", "quiet")):
                attrs.pop(ak, None)
        row["attributes"] = attrs
        ents.append(row)
    p["observation"] = obs
    p["entity_observations"] = ents
    c["payload"] = p
    return c

def strip_contract(ctx: dict[str, Any]) -> dict[str, Any]:
    c = dict(ctx)
    p = dict(c.get("payload") or {})
    c["declared_effect_service"] = ""
    c["candidate_services"] = []
    c["classifier_reason"] = None
    for key in ("trigger", "condition"):
        p[key] = []
    for key in ("automation_specification", "intended_action_definition"):
        p[key] = {}
    bindings = dict(p.get("bindings") or {})
    bindings["entity_catalog"] = []
    bindings["blueprint_inputs"] = {}
    p["bindings"] = bindings
    c["payload"] = p
    return c

def prepare_ctx(ctx: dict[str, Any], name: str) -> dict[str, Any]:
    if name == "w/o Execution Instance Modeling":
        return strip_execution_instance(ctx)
    if name == "w/o Runtime Context":
        return strip_runtime_context(ctx)
    if name == "w/o Contract Constraints":
        return strip_contract(ctx)
    return ctx

def repair_one(ctx: dict[str, Any], name: str) -> Any:
    work = prepare_ctx(ctx, name)
    original = compact_action(work.get("b0_action"))
    enable_remove = name not in {"w/o Behavioral Assessment", "w/o Remove"}
    enable_modify = name != "w/o Modify"
    enable_add = name != "w/o Recover"
    enable_r1 = name != "w/o Recover"
    repaired = original
    operator = "KEEP"
    if original:
        if enable_remove:
            removed = run_remove(work, original, llm=None)
            if removed.get("changed"):
                repaired = removed.get("action_out")
                operator = "REMOVE"
        if repaired is original and enable_modify:
            modified = run_modify(work, original)
            if modified.get("changed"):
                repaired = modified.get("action_out")
                operator = "MODIFY"
    elif enable_add:
        added = run_add(work, llm=None)
        if added.get("changed"):
            repaired = added.get("action_out")
            operator = "ADD"
    repaired = compact_action(repaired) if repaired else None
    if _illegal(repaired, work, original):
        repaired = original
        operator = "KEEP"
    if enable_r1 and r1_should_fire(work, repaired, use_scene=False) and climate_capability(work, repaired):
        repaired = apply_r1(repaired)
        operator = "HVAC_R1"
    repaired, _, _ = apply_payload_strict(repaired)
    out = compact_action(repaired) if repaired else None
    if out is not None and operator == "ADD":
        out["recovered_aux"] = str(out.get("service") or "") in {"notify.mobile_app", "logbook.log"}
    return out

def empty_execution_instance(comp: dict[str, Any]) -> dict[str, Any]:
    c = deepcopy(comp)
    rt = dict(c.get("runtime_state") or {})
    rt["observed"] = {}
    rt["entity_observations"] = []
    c["runtime_state"] = rt
    return c

def empty_runtime_context(comp: dict[str, Any]) -> dict[str, Any]:
    c = deepcopy(comp)
    rt = dict(c.get("runtime_state") or {})
    stripped = strip_runtime_context(
        {
            "payload": {
                "observation": dict(rt.get("observed") or {}),
                "entity_observations": list(rt.get("entity_observations") or []),
            }
        }
    )
    payload = stripped.get("payload") or {}
    rt["observed"] = payload.get("observation") or {}
    rt["entity_observations"] = payload.get("entity_observations") or []
    c["runtime_state"] = rt
    return c

def strip_component_contract(comp: dict[str, Any]) -> dict[str, Any]:
    c = deepcopy(comp)
    c["entity_binding"] = {}
    schema = dict(c.get("component_action_schema") or {})
    schema["service"] = ""
    c["component_action_schema"] = schema
    rt = dict(c.get("runtime_state") or {})
    system_state = dict(rt.get("system_state") or {})
    system_state["behavior_target"] = ""
    for key in ("automation_specification", "declared_effect_service", "candidate_services"):
        system_state.pop(key, None)
    rt["system_state"] = system_state
    c["runtime_state"] = rt
    return c

def prepare_component(comp: dict[str, Any], name: str) -> dict[str, Any]:
    if name == "w/o Execution Instance Modeling":
        return empty_execution_instance(comp)
    if name == "w/o Runtime Context":
        return empty_runtime_context(comp)
    if name == "w/o Contract Constraints":
        return strip_component_contract(comp)
    return comp

def toks_from_acts(acts: list[dict[str, Any]], bind_all: dict[str, Any]) -> list:
    out = []
    for act in acts:
        origin = str(act.get("component_origin") or "")
        token = tok(act, bind_all.get(origin) or {})
        if token:
            out.append(token)
    return out

def parent_actions(
    name: str,
    comps: list[dict[str, Any]],
    ctx_cache: dict[str, dict[str, Any]],
    bind_all: dict[str, Any],
) -> list[dict[str, Any]]:
    use_comps = [prepare_component(c, name) for c in comps]
    comps_by_id = {str(c.get("component_id") or ""): c for c in use_comps}
    ctxs = {}
    for comp in use_comps:
        sid = str(comp.get("single_scene_sample_id") or "")
        ctxs[str(comp.get("component_id") or "")] = prepare_ctx(ctx_cache.get(sid) or {}, name)
    _atoms, meta = attribute_parent(use_comps, ctxs)
    if name == "w/o Contract Constraints":
        for cid in list(meta.get("contracts") or {}):
            meta["contracts"][cid] = []
        meta["origs"] = {cid: [] for cid in (meta.get("origs") or {})}
        meta["fams"] = {cid: set() for cid in (meta.get("fams") or {})}
    owns = [
        ownership_for_component(
            comp,
            meta["per_comp_atoms"].get(str(comp.get("component_id") or "")) or [],
            meta["contracts"].get(str(comp.get("component_id") or "")) or [],
        )
        for comp in use_comps
    ]
    owns_by_id = {row["component_id"]: row for row in owns}
    composition = compose_parent(use_comps, owns)
    merged = []
    for comp in use_comps:
        sid = str(comp.get("single_scene_sample_id") or "")
        origin = str(comp.get("component_id") or "")
        out = repair_one(ctx_cache.get(sid) or {}, name)
        mapped = map_repaired(out, bind_all.get(origin) or {})
        if mapped and not mapped.get("recovered_aux"):
            mapped["component_origin"] = origin
            merged.append(mapped)
    if name == "w/o Provenance-Aware Interaction":
        return merged
    removed, _ = set_remove_v1(merged, bind_all)
    gated, _ = apply_generic_gate(removed, owns_by_id, composition)
    if name == "w/o Behavioral Assessment":
        return gated
    verified, _ = apply_verifier(gated, comps_by_id, owns_by_id, composition)
    return verified

def main() -> None:
    print("load indexes", flush=True)
    y_idx = load_jsonl(Y_PATH)
    samples: list[dict[str, Any]] = []
    with SS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))
    ctx_cache: dict[str, dict[str, Any]] = {}
    for i, sample in enumerate(samples, 1):
        ctx_cache[str(sample.get("sample_id") or "")] = build_repair_context(sample)
        if i % 3000 == 0:
            print(f"  ctx {i}/{len(samples)}", flush=True)

    ss_acc = {name: {"n": 0, "exact": 0, "rs": 0, "fr": 0, "b0_ok": 0} for name in SS_VARIANTS}
    print("SS", flush=True)
    for i, sample in enumerate(samples, 1):
        sid = str(sample.get("sample_id") or "")
        ctx = ctx_cache[sid]
        y_dec, y_act, y_n, _at = y_pair(y_idx.get(sid) or {})
        b0 = compact_action(ctx.get("b0_action"))
        b0_ok = semantic_ok(*pair_from_act(b0), y_dec, y_act, y_n)
        for name in SS_VARIANTS:
            pred = repair_one(ctx, name)
            ok = semantic_ok(*pair_from_act(pred), y_dec, y_act, y_n)
            bucket = ss_acc[name]
            bucket["n"] += 1
            bucket["exact"] += int(ok)
            bucket["b0_ok"] += int(b0_ok)
            bucket["rs"] += int((not b0_ok) and ok)
            bucket["fr"] += int(b0_ok and (not ok))
        if i % 2000 == 0:
            print(f"  ss {i}/{len(samples)}", flush=True)

    ss_out = {
        name: metrics_block(b["n"], b["exact"], b["b0_ok"], b["rs"], b["fr"])
        for name, b in ss_acc.items()
    }

    print("MA", flush=True)
    ma_b = {name: acc_bucket() for name in VARIANTS}
    n_parent = 0
    with MA.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_parent += 1
            comps = list(rec.get("components") or [])
            bind_all = {str(c.get("component_id") or ""): c.get("entity_binding") or {} for c in comps}
            gold_toks = toks_from_acts(ma_y_acts(rec), bind_all)
            parent_acts = list(((rec.get("formal_b0") or {}).get("parent") or {}).get("semantic_actions") or [])
            b0_ok = bool(set_metrics(toks_from_acts(parent_acts, bind_all), gold_toks)["set_exact"])
            for name in VARIANTS:
                final = parent_actions(name, comps, ctx_cache, bind_all)
                metrics = set_metrics(toks_from_acts(final, bind_all), gold_toks)
                add_bucket(ma_b[name], metrics, b0_ok, metrics["set_exact"], metrics["n_pred"])
            if n_parent % 400 == 0:
                print(f"  ma {n_parent}", flush=True)
    ma_out = {name: finish_bucket(bucket) for name, bucket in ma_b.items()}

    rows = []
    for name in VARIANTS:
        ss = ss_out.get(name)
        ma = ma_out[name]
        rows.append(
            {
                "variant": name,
                "single_scene": None
                if ss is None
                else {
                    "EM": ss["Semantic_Success"],
                    "RSR": ss["Repair_Success"],
                    "FRR": ss["False_Repair"],
                    "EM_count": f"{ss['Semantic_Success_count']}/{ss['N']}",
                    "RSR_count": f"{ss['Repair_Success_count']}/{ss['Repair_Success_denominator']}",
                    "FRR_count": f"{ss['False_Repair_count']}/{ss['N']}",
                },
                "multi_action": {
                    "Parent_EM": ma["Parent_Set_Exact_Match"],
                    "Action_F1": ma["Action_F1"],
                    "Parent_EM_count": ma["Parent_Set_Exact_count"],
                    "N": ma["N"],
                },
            }
        )
    payload = {
        "what_this_file_is": "Ablation on Single-Scene and Multi-Action behavioral repair. Only the paper table modules are included.",
        "variants": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT, flush=True)

if __name__ == "__main__":
    main()
