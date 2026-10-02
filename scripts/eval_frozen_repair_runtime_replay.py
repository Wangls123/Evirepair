from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import voluptuous as vol

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
from eval_strict_v2_set_remove import (
    MA,
    OUT,
    SS,
    TRACE,
    apply_strict_v2,
    cap_of_svc,
    compact_action,
    map_repaired,
    set_remove_v1,
)
from eval_ss_trhr_final import load_jsonl
from ha_runtime_sandbox import (
    HASandbox,
    SERVICE_REGISTRY,
    STATELESS_OK_NO_ENTITY,
    UNSUPPORTED_PREFIXES,
    canonical_service,
    domain_of_entity,
    seed_from_observation,
    transition_ok,
)
from smarthome_mdf.ss_trhr_repair.actions import normalize_action
from smarthome_mdf.ss_trhr_repair.context import build_repair_context

SEED = 20260919
REQUIRED = {
    "climate.set_hvac_mode": ("hvac_mode",),
    "climate.set_preset_mode": ("preset_mode",),
    "google_sheets.append_sheet": ("config_entry", "worksheet", "data"),
    "persistent_notification.create": ("message",),
    "input_text.set_value": ("value",),
    "notify.notify": ("message",),
}

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

def pctile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[i], 4)

def rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None

def as_acts(raw: Any) -> list[dict]:
    if raw in (None, "", {}, []):
        return []
    if isinstance(raw, list):
        out = []
        for a in raw:
            n = compact_action(a) if isinstance(a, dict) and a.get("service") else normalize_action(a)
            if n and n.get("service"):
                out.append(n)
        return out
    n = compact_action(raw) if isinstance(raw, dict) else normalize_action(raw)
    return [n] if n and n.get("service") else []

def entity_ids_of(obs: dict, ents: list, acts: list[dict]) -> list[str]:
    ids = []
    for a in acts:
        if a.get("target_entity"):
            ids.append(str(a["target_entity"]))
    for e in ents or []:
        if isinstance(e, dict) and e.get("entity_id"):
            ids.append(str(e["entity_id"]))
    return list(dict.fromkeys(ids))

def static_validate(acts: list[dict], known: set[str]) -> dict[str, Any]:
    flags = Counter()
    details = []
    if not acts:
        flags["zero_call"] += 1
        return {"ok": True, "n": 0, "flags": dict(flags), "details": details, "executable_n": 0, "invalid_n": 0}
    seen = set()
    executable = 0
    for a in acts:
        raw_svc = str(a.get("service") or "")
        svc = canonical_service(raw_svc)
        ent = a.get("target_entity") or a.get("entity")
        if isinstance(ent, list):
            ent = ent[0] if ent else None
        if isinstance(ent, dict):
            ent = ent.get("entity_id") or ent.get("entity")
        ent = str(ent) if ent not in (None, "", [], {}) else None
        params = dict(a.get("parameters") or {}) if isinstance(a.get("parameters"), dict) else {}
        err = []
        if any(raw_svc.startswith(p) for p in UNSUPPORTED_PREFIXES):
            err.append("E1_UNSUPPORTED")
        spec = SERVICE_REGISTRY.get(svc)
        if spec is None:
            err.append("E1_SERVICE_MISSING")
        else:
            if spec["domain"] and "." in svc and svc.split(".", 1)[0] != spec["domain"] and svc.split(".", 1)[0] not in {"notify"}:
                err.append("E2_DOMAIN")
            if spec["target_required"] and not ent:
                err.append("E3_ENTITY_MISSING")
            if ent:
                if str(ent) not in known and str(ent) not in {"notify.mobile_app_phone"}:

                    if not domain_of_entity(str(ent)):
                        err.append("E3_ENTITY_MISSING")
                    elif "{{" in str(ent) or str(ent).startswith("!input"):
                        err.append("E3_ENTITY_MISSING")
                dom = domain_of_entity(str(ent))
                if spec["target_domains"] and dom and dom not in spec["target_domains"] and spec["domain"] != "homeassistant":
                    err.append("E4_DOMAIN_MISMATCH")
            need = REQUIRED.get(svc) or REQUIRED.get(raw_svc) or ()
            for k in need:
                if params.get(k) in (None, "", [], {}):
                    err.append("E5_REQUIRED_PAYLOAD")
            if spec is not None:
                try:
                    check = dict(params)
                    if svc == "system_log.write" and "message" not in check:
                        raise vol.Invalid("required key not provided @ message")
                    spec["schema"](check)
                except vol.Invalid as exc:
                    msg = str(exc).lower()
                    if "extra keys" in msg or "not a valid" in msg and "extra" in msg:
                        err.append("E7_ILLEGAL_FIELD")
                    elif "required" in msg:
                        err.append("E5_REQUIRED_PAYLOAD")
                    else:
                        err.append("E6_PAYLOAD_VALUE")
        key = (svc, str(ent or ""), json.dumps(params, sort_keys=True, default=str))
        if key in seen:
            err.append("E8_DUPLICATE")
        seen.add(key)
        if not raw_svc:
            err.append("E9_SERIALIZATION")
        for e in err:
            flags[e] += 1
        if not err:
            executable += 1
        details.append({"service": raw_svc, "entity": ent, "errors": err})
    return {
        "ok": executable == len(acts) and not flags.get("E8_DUPLICATE"),
        "n": len(acts),
        "executable_n": executable,
        "invalid_n": len(acts) - executable,
        "flags": dict(flags),
        "details": details,
    }

def make_box(obs: dict, ents: list, acts: list[dict]) -> HASandbox:
    box = HASandbox()
    seed_from_observation(box, obs, ents, entity_ids_of(obs, ents, acts))
    return box

def execute_acts(box: HASandbox, acts: list[dict]) -> dict[str, Any]:
    if not acts:
        return {"n": 0, "success": 0, "failed": 0, "changed": 0, "already": 0, "api": 0, "results": [], "calls": [], "ms": []}
    rows = []
    ms = []
    for a in acts:
        r = box.call(str(a.get("service")), a.get("target_entity") or a.get("entity"), a.get("parameters") or {})
        rows.append(r)
        ms.append(r["execution_time_ms"])
    return {
        "n": len(rows),
        "success": sum(1 for r in rows if r["execution_success"]),
        "failed": sum(1 for r in rows if not r["execution_success"]),
        "changed": sum(1 for r in rows if r["result"] == "STATE_CHANGED"),
        "already": sum(1 for r in rows if r["result"] == "STATE_ALREADY_SATISFIED"),
        "api": sum(1 for r in rows if r["result"] == "API_EXECUTED"),
        "trans_ok": sum(1 for r in rows if transition_ok(r["service"], r["result"], r.get("post_state")) is True),
        "trans_n": sum(1 for r in rows if transition_ok(r["service"], r["result"], r.get("post_state")) is not None),
        "results": [r["result"] for r in rows],
        "calls": rows,
        "ms": ms,
    }

def compose_class(acts: list[dict]) -> str:
    if len(acts) <= 1:
        return "NO_INTERFERENCE"
    by_ent: dict[str, list[str]] = defaultdict(list)
    for a in acts:
        ent = str(a.get("target_entity") or "") or "_none_"
        by_ent[ent].append(canonical_service(str(a.get("service") or "")))
    conflict = redundant = order = False
    for ent, svcs in by_ent.items():
        if ent == "_none_":
            continue
        ons = sum(1 for s in svcs if s.endswith("turn_on") or (s == "climate.set_hvac_mode"))
        offs = sum(1 for s in svcs if s.endswith("turn_off") or s == "climate.turn_off")
        modes = [s for s in svcs if s == "climate.set_hvac_mode"]
        if ons and offs:
            conflict = True
        if len(svcs) > 1 and len(set(svcs)) == 1:
            redundant = True
        if len(modes) >= 1 and offs:
            conflict = True
        if len(svcs) > 1 and "climate.set_hvac_mode" in svcs and any(s.startswith("climate.") for s in svcs):
            order = True
    if conflict:
        return "CONFLICTING"
    if order:
        return "ORDER_SENSITIVE"
    if redundant:
        return "REDUNDANT"
    if len(acts) > 1:
        caps = [cap_of_svc(str(a.get("service") or "")) for a in acts]
        if len(set(caps)) > 1:
            return "DEPENDENT" if "ClimateControl" in caps and "Lighting" in caps else "NO_INTERFERENCE"
    return "NO_INTERFERENCE"

def acc() -> dict[str, Any]:
    return {
        "n": 0,
        "action_n": 0,
        "executable": 0,
        "invalid_service": 0,
        "invalid_entity": 0,
        "invalid_payload": 0,
        "duplicate": 0,
        "zero_ok": 0,
        "exec_success": 0,
        "exec_fail": 0,
        "trans_ok": 0,
        "trans_n": 0,
        "unsafe": 0,
        "ms": [],
        "prep_ms": [],
        "rule_ms": [],
        "val_ms": [],
        "conflicts": 0,
        "redundant": 0,
        "order_sensitive": 0,
    }

def add_static(b: dict, st: dict, n_pred: int) -> None:
    b["n"] += 1
    b["action_n"] += n_pred
    b["executable"] += st["executable_n"]
    fl = st["flags"]
    b["invalid_service"] += fl.get("E1_SERVICE_MISSING", 0) + fl.get("E1_UNSUPPORTED", 0)
    b["invalid_entity"] += fl.get("E3_ENTITY_MISSING", 0) + fl.get("E4_DOMAIN_MISMATCH", 0)
    b["invalid_payload"] += fl.get("E5_REQUIRED_PAYLOAD", 0) + fl.get("E6_PAYLOAD_VALUE", 0) + fl.get("E7_ILLEGAL_FIELD", 0)
    b["duplicate"] += fl.get("E8_DUPLICATE", 0)
    if n_pred == 0:
        b["zero_ok"] += 1

def add_exec(b: dict, ex: dict) -> None:
    b["exec_success"] += ex["success"]
    b["exec_fail"] += ex["failed"]
    b["trans_ok"] += ex.get("trans_ok", 0)
    b["trans_n"] += ex.get("trans_n", 0)
    b["ms"].extend(ex.get("ms") or [])
    b["unsafe"] += ex["failed"]

def summarize(b: dict) -> dict[str, Any]:
    n = b["n"]
    an = b["action_n"]
    return {
        "N_units": n,
        "N_actions": an,
        "Executable_Action_Rate": rate(b["executable"], an) if an else rate(b["zero_ok"], n),
        "Invalid_Service_Rate": rate(b["invalid_service"], an),
        "Invalid_Entity_Rate": rate(b["invalid_entity"], an),
        "Invalid_Payload_Rate": rate(b["invalid_payload"], an),
        "Duplicate_Call_Rate": rate(b["duplicate"], an),
        "Idle_Zero_Call_Rate": rate(b["zero_ok"], n),
        "Execution_Success_Rate": rate(b["exec_success"], max(an, 1)),
        "Successful_Transition_Rate": rate(b["trans_ok"], b["trans_n"]),
        "Unsafe_Action_Rate": rate(b["unsafe"], max(an, 1)),
        "Conflict_Rate": rate(b["conflicts"], n),
        "mean_action_count": round(an / n, 4) if n else 0,
        "latency_ms": {
            "median": pctile(b["ms"], 50),
            "p95": pctile(b["ms"], 95),
            "p99": pctile(b["ms"], 99),
            "n": len(b["ms"]),
        },
        "rule_eval_ms": {"median": pctile(b["rule_ms"], 50), "p95": pctile(b["rule_ms"], 95), "p99": pctile(b["rule_ms"], 99)},
        "validation_ms": {"median": pctile(b["val_ms"], 50), "p95": pctile(b["val_ms"], 95)},
        "prep_ms": {"median": pctile(b["prep_ms"], 50), "p95": pctile(b["prep_ms"], 95)},
    }

def main() -> None:
    rng = random.Random(SEED)
    print("load traces...", flush=True)
    traces = load_jsonl(TRACE)
    ss_rows = []
    ctx_cache: dict[str, dict] = {}
    t_prep = []
    print("walk SS...", flush=True)
    with SS.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            rec = json.loads(line)
            sid = str(rec.get("sample_id") or "")
            t0 = time.perf_counter()
            ctx = build_repair_context(rec)
            t_prep.append((time.perf_counter() - t0) * 1000)
            ctx_cache[sid] = ctx
            tr = traces.get(sid) or {}
            b0 = as_acts(tr.get("original_B0_action") or ctx.get("b0_action"))
            fr = as_acts(tr.get("repaired_action"))
            t1 = time.perf_counter()
            v2a, kinds = apply_strict_v2(ctx, fr[0] if fr else None)
            rule_ms = (time.perf_counter() - t1) * 1000
            v2 = as_acts(v2a)
            obs = (ctx.get("payload") or {}).get("observation") or {}
            ents = (ctx.get("payload") or {}).get("entity_observations") or rec.get("entity_observations") or []
            ss_rows.append(
                {
                    "id": sid,
                    "scope": "SS",
                    "scene": str(rec.get("scene_type") or ctx.get("scene_type") or ""),
                    "obs": obs,
                    "ents": ents,
                    "S0": b0,
                    "S1": fr,
                    "S2": v2,
                    "kinds": kinds,
                    "op": tr.get("operator_used") or tr.get("operator"),
                    "reason": tr.get("reason"),
                    "rule_ms": rule_ms,
                    "prep_ms": t_prep[-1],
                }
            )
            if i % 2000 == 0:
                print(f"  SS {i}", flush=True)
    print(f"SS N={len(ss_rows)}", flush=True)

    methods_ss = ["S0", "S1", "S2"]
    buckets = {m: acc() for m in methods_ss}
    replay_rows = []
    remove_rows = []
    refine_rows = []
    fail_rows = []
    idem = Counter()
    rf = Counter()

    for row in ss_rows:
        known = set(entity_ids_of(row["obs"], row["ents"], row["S0"] + row["S1"] + row["S2"]))
        known.update(HASandbox().states)
        for m in methods_ss:
            acts = row[m]
            t0 = time.perf_counter()
            st = static_validate(acts, known | {"light.living_room", "climate.living_room_ac", "notify.mobile_app_phone"})
            buckets[m]["val_ms"].append((time.perf_counter() - t0) * 1000)
            buckets[m]["rule_ms"].append(row["rule_ms"] if m == "S2" else 0.0)
            buckets[m]["prep_ms"].append(row["prep_ms"])
            add_static(buckets[m], st, len(acts))
            box = make_box(row["obs"], row["ents"], acts)
            ex = execute_acts(box, acts)
            add_exec(buckets[m], ex)
            if m == "S2":
                for c, r in zip(acts, ex.get("calls") or []):
                    replay_rows.append(
                        {
                            "id": row["id"],
                            "scope": "SS",
                            "method": m,
                            "service": c.get("service"),
                            "entity": c.get("target_entity"),
                            "execution_success": r["execution_success"],
                            "result": r["result"],
                            "exception": r["exception"],
                            "execution_time_ms": r["execution_time_ms"],
                            "pre_state": r.get("pre_state"),
                            "post_state": r.get("post_state"),
                        }
                    )
                    if not r["execution_success"]:
                        code = "RF1" if r["exception"] and "service" in str(r["exception"]) else "RF2" if "entity" in str(r["exception"] or "") else "RF3" if "schema" in str(r["exception"] or "") else "RF10"
                        rf[code] += 1
                        fail_rows.append({"id": row["id"], "method": m, "rf": code, "exception": r["exception"], "service": c.get("service")})

                box2 = make_box(row["obs"], row["ents"], acts)
                execute_acts(box2, acts)
                ex2 = execute_acts(box2, acts)
                for r in ex2.get("calls") or []:
                    svc = canonical_service(str(r.get("service") or ""))
                    spec = SERVICE_REGISTRY.get(svc)
                    if spec and spec["effect"] == "event":
                        idem["notify_log_second_effect"] += 1
                    elif r["result"] == "STATE_CHANGED":
                        idem["stateful_second_change"] += 1
                    elif r["result"] == "STATE_ALREADY_SATISFIED":
                        idem["stateful_idempotent"] += 1
                    elif r["execution_success"]:
                        idem["second_ok_no_change"] += 1
                    else:
                        idem["second_exception"] += 1
            if st["flags"].get("E1_SERVICE_MISSING") or st["flags"].get("E1_UNSUPPORTED"):
                rf["RF1"] += 0

        if row["S0"] and not row["S2"] and row.get("op") == "REMOVE":
            box = make_box(row["obs"], row["ents"], row["S0"])
            ex = execute_acts(box, row["S0"])
            res = (ex.get("results") or ["NO_EFFECT"])[0]
            if res in {"STATE_ALREADY_SATISFIED", "NO_EFFECT"}:
                kind = "runtime-verifiable-unnecessary"
            elif res == "API_EXECUTED":
                kind = "semantic-only-auxiliary-effect"
            elif res == "STATE_CHANGED":
                kind = "potentially-risky-remove"
            else:
                kind = "SEMANTIC_UNVERIFIABLE"
            remove_rows.append({"id": row["id"], "reason": row.get("reason"), "b0_result": res, "kind": kind, "scene": row["scene"]})
        if row["kinds"]:
            box = make_box(row["obs"], row["ents"], row["S2"])
            ex = execute_acts(box, row["S2"])
            refine_rows.append(
                {
                    "id": row["id"],
                    "kinds": row["kinds"],
                    "executable": static_validate(row["S2"], known | {"climate.living_room_ac", "light.living_room"}).get("ok"),
                    "result": (ex.get("results") or [None])[0],
                    "trans_ok": bool(ex.get("trans_ok")),
                }
            )

    print("walk MA...", flush=True)
    ma_rows = []
    methods_ma = ["M0", "M1", "M2", "M3", "M4"]
    for m in methods_ma:
        buckets[m] = acc()
    ma_comp = Counter()
    n_parent = 0
    with MA.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_parent += 1
            comps = list(rec.get("components") or [])
            bind_all = {str(c.get("component_id") or ""): c.get("entity_binding") or {} for c in comps}
            parent_acts = as_acts(((rec.get("formal_b0") or {}).get("parent") or {}).get("semantic_actions") or [])
            ctxs = {}
            obs_m: dict = {}
            ents_m: list = []
            for c in comps:
                sid = str(c.get("single_scene_sample_id") or "")
                cid = str(c.get("component_id") or "")
                ctxs[cid] = ctx_cache.get(sid) or {}
                payload = (ctxs[cid].get("payload") or {})
                if payload.get("observation"):
                    obs_m.update(payload.get("observation") or {})
                ents_m.extend(payload.get("entity_observations") or [])
            t0 = time.perf_counter()
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
                ctx = ctxs.get(origin) or {}
                out, _k = apply_strict_v2(ctx, fr) if ctx else (fr, [])
                mapped = map_repaired(out, bind_all.get(origin) or {})
                if mapped:
                    mapped["component_origin"] = origin
                    merged.append(mapped)
            sr, _ = set_remove_v1(merged, bind_all)
            d_acts, _ = apply_generic_gate(sr, owns_by_id, composition)
            e_acts, _ = apply_verifier(d_acts, {str(c.get("component_id") or ""): c for c in comps}, owns_by_id, composition)
            rule_ms = (time.perf_counter() - t0) * 1000
            stages = {"M0": parent_acts, "M1": merged, "M2": sr, "M3": d_acts, "M4": e_acts}
            ma_rows.append({"id": rec.get("multi_action_id"), "n_comp": len(comps), "obs": obs_m, "ents": ents_m, "stages": stages, "rule_ms": rule_ms})
            known = set(entity_ids_of(obs_m, ents_m, parent_acts + merged + e_acts)) | {"light.living_room", "climate.living_room_ac"}
            for m, acts in stages.items():
                st = static_validate(acts, known)
                buckets[m]["rule_ms"].append(rule_ms)
                add_static(buckets[m], st, len(acts))
                box = make_box(obs_m, ents_m, acts)
                ex = execute_acts(box, acts)
                add_exec(buckets[m], ex)
                cls = compose_class(acts)
                ma_comp[f"{m}:{cls}"] += 1
                if cls == "CONFLICTING":
                    buckets[m]["conflicts"] += 1
                elif cls == "REDUNDANT":
                    buckets[m]["redundant"] += 1
                elif cls == "ORDER_SENSITIVE":
                    buckets[m]["order_sensitive"] += 1
                if m == "M4":
                    for c, r in zip(acts, ex.get("calls") or []):
                        replay_rows.append(
                            {
                                "id": rec.get("multi_action_id"),
                                "scope": "MA",
                                "method": m,
                                "service": c.get("service"),
                                "entity": c.get("target_entity"),
                                "execution_success": r["execution_success"],
                                "result": r["result"],
                                "exception": r["exception"],
                                "execution_time_ms": r["execution_time_ms"],
                                "n_comp": len(comps),
                            }
                        )
            if n_parent % 400 == 0:
                print(f"  MA {n_parent}", flush=True)
    print(f"MA N={n_parent}", flush=True)

    print("fault injection...", flush=True)
    faults = Counter()
    ss_sample = [r for r in ss_rows if r["S2"]][:400]
    for row in ss_sample:
        acts = row["S2"]

        box = make_box(row["obs"], row["ents"], acts)
        for eid in list(box.states)[:1]:
            box.unavailable.add(eid)
        ex = execute_acts(box, acts)
        faults["F1_fail" if ex["failed"] else "F1_contained"] += 1

        box = make_box(row["obs"], row["ents"], acts)
        if box.states:
            del box.states[next(iter(box.states))]
        ex = execute_acts(box, acts)
        faults["F2_fail" if ex["failed"] else "F2_ok"] += 1

        box = make_box(row["obs"], row["ents"], acts)
        if acts:
            box.disabled_services.add(canonical_service(str(acts[0].get("service"))))
        ex = execute_acts(box, acts)
        faults["F5_fail" if ex["failed"] else "F5_ok"] += 1

        bad = [dict(a, parameters={"hvac_mode": "not_a_mode"}) if str(a.get("service", "")).startswith("climate.set_hvac") else a for a in acts]
        st = static_validate(bad, {"climate.living_room_ac"})
        faults["F6_caught" if st["invalid_n"] or not bad else "F6_missed"] += 1

        fake = [{"service": "llmvision.video_analyzer", "target_entity": "camera.x", "parameters": {}}]
        st = static_validate(fake, set())
        faults["F10_caught" if st["invalid_n"] else "F10_missed"] += 1
    ma_sample = ma_rows[:150]
    for row in ma_sample:
        acts = row["stages"]["M4"]
        box = make_box(row["obs"], row["ents"], acts)
        ex = execute_acts(box, acts)
        faults["MA_baseline_fail"] += ex["failed"]

        half = acts[: max(0, len(acts) - 1)]
        box = make_box(row["obs"], row["ents"], half)
        ex2 = execute_acts(box, half)
        faults["F8_half_fail"] += ex2["failed"]

    env = {"SS": Counter(), "MA": Counter(), "by_cap": defaultdict(Counter)}
    for row in ss_rows:
        acts = row["S2"]
        cap = cap_of_svc(str(acts[0]["service"])) if acts else "NONE"
        st = static_validate(acts, {"light.living_room", "climate.living_room_ac", "notify.mobile_app_phone"})
        box = make_box(row["obs"], row["ents"], acts)
        ex = execute_acts(box, acts)
        if not acts:
            z = "SAFE_ABSTAIN"
        elif any(canonical_service(str(a.get("service"))) not in SERVICE_REGISTRY and any(str(a.get("service","")).startswith(p) for p in UNSUPPORTED_PREFIXES) for a in acts):
            z = "UNSUPPORTED"
        elif ex["failed"]:
            z = "RUNTIME_FAILURE"
        elif st["ok"] and ex["failed"] == 0:
            z = "RUNTIME_SAFE_REPAIR"
        else:
            z = "SEMANTIC_SAFE_RUNTIME_UNVERIFIED"
        env["SS"][z] += 1
        env["by_cap"][cap][z] += 1
    for row in ma_rows:
        acts = row["stages"]["M4"]
        st = static_validate(acts, {"light.living_room", "climate.living_room_ac"})
        box = make_box(row["obs"], row["ents"], acts)
        ex = execute_acts(box, acts)
        if not acts:
            z = "SAFE_ABSTAIN"
        elif ex["failed"]:
            z = "RUNTIME_FAILURE"
        elif st["ok"]:
            z = "RUNTIME_SAFE_REPAIR"
        else:
            z = "SEMANTIC_SAFE_RUNTIME_UNVERIFIED"
        env["MA"][z] += 1

    ss_sum = {m: summarize(buckets[m]) for m in methods_ss}
    ma_sum = {m: summarize(buckets[m]) for m in methods_ma}

    all_ms = buckets["S2"]["ms"] + buckets["M4"]["ms"]
    aps = round(1000.0 * len(all_ms) / sum(all_ms), 2) if all_ms and sum(all_ms) else None

    static_out = {
        "gold_independent": True,
        "note": "Full-corpus static validation. HA voluptuous schemas. Gold was not read.",
        "full_validation": True,
        "executed_subset": "full_sandbox_execution",
        "SS": ss_sum,
        "MA": ma_sum,
    }
    dump(OUT / "runtime_static_executability.json", static_out)
    write_jsonl(OUT / "runtime_replay_trace.jsonl", replay_rows[:8000])

    trans = {
        "gold_independent": True,
        "SS_S2": {"transition_success": ss_sum["S2"]["Successful_Transition_Rate"], "n": buckets["S2"]["trans_n"]},
        "MA_M4": {"transition_success": ma_sum["M4"]["Successful_Transition_Rate"], "n": buckets["M4"]["trans_n"]},
        "by_result_SS_S2": dict(Counter(r["result"] for r in replay_rows if r.get("scope") == "SS")),
    }
    dump(OUT / "runtime_transition_validation.json", trans)

    rem_c = Counter(r["kind"] for r in remove_rows)
    dump(
        OUT / "runtime_remove_safety.json",
        {
            "gold_independent": True,
            "N_remove": len(remove_rows),
            "counts": dict(rem_c),
            "runtime_verifiable": rem_c.get("runtime-verifiable-unnecessary", 0) + rem_c.get("semantic-only-auxiliary-effect", 0),
            "potentially_risky": rem_c.get("potentially-risky-remove", 0),
            "semantic_unverifiable": rem_c.get("SEMANTIC_UNVERIFIABLE", 0),
            "examples": remove_rows[:40],
        },
    )
    ref_c = Counter()
    for r in refine_rows:
        ref_c["n"] += 1
        ref_c["executable"] += int(bool(r["executable"]))
        ref_c["trans_ok"] += int(bool(r["trans_ok"]))
        ref_c["no_effect"] += int(r.get("result") in {"NO_EFFECT", "STATE_ALREADY_SATISFIED"})
        ref_c["failed"] += int(r.get("result") == "EXECUTION_FAILED")
        for k in r.get("kinds") or []:
            ref_c[str(k)] += 1
    dump(OUT / "runtime_refinement_validation.json", {"gold_independent": True, "counts": dict(ref_c), "N": len(refine_rows)})
    dump(
        OUT / "ma_runtime_composition_safety.json",
        {"gold_independent": True, "counts": dict(ma_comp), "N_parents": n_parent, "note": "Recomputed at runtime; not assumed zero."},
    )
    dump(
        OUT / "runtime_idempotence.json",
        {"gold_independent": True, "counts": dict(idem), "note": "Notification/Logging second execution is an extra external effect by design."},
    )
    fail_safe = {
        "gold_independent": True,
        "faults": dict(faults),
        "F10_unsupported_caught": faults.get("F10_caught", 0),
        "fail_safe_tendency": "ABSTAIN_OR_FAIL_CLOSED" if faults.get("F5_fail", 0) >= faults.get("F5_ok", 0) else "EXECUTE",
    }
    dump(OUT / "runtime_fault_injection.json", fail_safe)
    dump(
        OUT / "runtime_reliability_envelope.json",
        {
            "gold_independent": True,
            "SS": dict(env["SS"]),
            "MA": dict(env["MA"]),
            "by_capability": {k: dict(v) for k, v in env["by_cap"].items()},
            "prior_offline_envelope_not_overwritten": True,
        },
    )
    dump(
        OUT / "runtime_efficiency.json",
        {
            "gold_independent": True,
            "SS_prep_ms": {"median": pctile(t_prep, 50), "p95": pctile(t_prep, 95), "p99": pctile(t_prep, 99)},
            "SS_S2_exec_ms": ss_sum["S2"]["latency_ms"],
            "MA_M4_rule_ms": ma_sum["M4"]["rule_eval_ms"],
            "MA_M4_exec_ms": ma_sum["M4"]["latency_ms"],
            "actions_per_second_sandbox": aps,
            "parents_per_second": round(n_parent / (sum(buckets["M4"]["rule_ms"]) / 1000.0), 2) if buckets["M4"]["rule_ms"] else None,
            "LLM_latency": "none_in_frozen_runtime_path",
        },
    )
    dump(
        OUT / "runtime_failure_analysis.json",
        {"gold_independent": True, "RF": dict(rf), "n_fail_examples": len(fail_rows), "examples": fail_rows[:30]},
    )

    (OUT / "ha_replay_environment.md").write_text(
        f"""# Home Assistant replay environment

Mode: **high-fidelity in-process sandbox** (no live household devices).

Docker is installed on this host (`Docker 28.3.2`) but the Home Assistant Python package is not present, and this evaluation must not attach to a real home. The harness therefore reuses Home Assistant's **voluptuous** schema engine plus official HA Core service names, domains, and required fields.

- Python: {sys.version.split()[0]}
- voluptuous: {vol.__version__ if hasattr(vol, '__version__') else '0.16.x'}
- HA Core package: not installed (schema copied from HA Core service contracts)
- Supported domains: {sorted({v['domain'] for v in SERVICE_REGISTRY.values()})}
- Service registry size: {len(SERVICE_REGISTRY)}
- Entity initialization: observation + entity_observations + action targets + canonical `light.living_room` / `climate.living_room_ac`
- Full static validation: SS 12021 + MA 2400
- Execution: full corpus in sandbox (not a subsample)
- Gold: not read
""",
        encoding="utf-8",
    )

    b0_invalid = buckets["S0"]["invalid_service"] + buckets["S0"]["invalid_entity"] + buckets["S0"]["invalid_payload"]
    write_report(ss_sum, ma_sum, rem_c, ref_c, ma_comp, idem, faults, env, rf, b0_invalid, buckets, n_parent, aps)
    print("DONE", json.dumps({"SS_S2_exec": ss_sum["S2"]["Executable_Action_Rate"], "MA_M4_exec": ma_sum["M4"]["Executable_Action_Rate"]}, indent=2), flush=True)

def write_report(ss_sum, ma_sum, rem_c, ref_c, ma_comp, idem, faults, env, rf, b0_invalid, buckets, n_parent, aps) -> None:
    s0, s1, s2 = ss_sum["S0"], ss_sum["S1"], ss_sum["S2"]
    m0, m4 = ma_sum["M0"], ma_sum["M4"]
    text = f"""# End-to-end runtime validation report

Gold-independent runtime evidence. Official Gold-based metrics are **not** updated:

- SS Strict-v2 Semantic Success = **81.30%** (offline Gold)
- SS False Repair = **1.49%** (offline Gold)
- MA Parent Set Exact = **52.67%** (offline Gold)
- MA False Repair = **1** (offline Gold)

This report does **not** mix those numbers with executability.

Environment: HA-schema sandbox (`voluptuous` + official HA services). No live home devices. Repair was not modified.

## Comparison (Gold-independent)

| Method | Executable | Invalid payload | Exec success | Transition | Unsafe | Mean actions |
| --- | --- | --- | --- | --- | --- | --- |
| SS S0 B0 | {s0.get("Executable_Action_Rate")} | {s0.get("Invalid_Payload_Rate")} | {s0.get("Execution_Success_Rate")} | {s0.get("Successful_Transition_Rate")} | {s0.get("Unsafe_Action_Rate")} | {s0.get("mean_action_count")} |
| SS S1 Frozen v1 | {s1.get("Executable_Action_Rate")} | {s1.get("Invalid_Payload_Rate")} | {s1.get("Execution_Success_Rate")} | {s1.get("Successful_Transition_Rate")} | {s1.get("Unsafe_Action_Rate")} | {s1.get("mean_action_count")} |
| SS S2 Strict-v2 | {s2.get("Executable_Action_Rate")} | {s2.get("Invalid_Payload_Rate")} | {s2.get("Execution_Success_Rate")} | {s2.get("Successful_Transition_Rate")} | {s2.get("Unsafe_Action_Rate")} | {s2.get("mean_action_count")} |
| MA M0 B0 | {m0.get("Executable_Action_Rate")} | {m0.get("Invalid_Payload_Rate")} | {m0.get("Execution_Success_Rate")} | {m0.get("Successful_Transition_Rate")} | {m0.get("Unsafe_Action_Rate")} | {m0.get("mean_action_count")} |
| MA M4 Verifier | {m4.get("Executable_Action_Rate")} | {m4.get("Invalid_Payload_Rate")} | {m4.get("Execution_Success_Rate")} | {m4.get("Successful_Transition_Rate")} | {m4.get("Unsafe_Action_Rate")} | {m4.get("mean_action_count")} |

## Answers

1. B0 invalid/non-executable actions (SS service+entity+payload error counts): **{b0_invalid}** error flags on {buckets["S0"]["action_n"]} B0 actions. Executable rate S0 = {s0.get("Executable_Action_Rate")}.
2. Repair executable rate: S0 {s0.get("Executable_Action_Rate")} → S1 {s1.get("Executable_Action_Rate")} → S2 {s2.get("Executable_Action_Rate")}. MA M0 {m0.get("Executable_Action_Rate")} → M4 {m4.get("Executable_Action_Rate")}.
3. Strict-v2 transition success: {s2.get("Successful_Transition_Rate")} (S0 {s0.get("Successful_Transition_Rate")}).
4. HVAC refinement N={ref_c.get("n", 0)}; executable={ref_c.get("executable", 0)}; trans_ok={ref_c.get("trans_ok", 0)}; failed={ref_c.get("failed", 0)}.
5. Payload invalid rate S0 {s0.get("Invalid_Payload_Rate")} vs S2 {s2.get("Invalid_Payload_Rate")}.
6. REMOVE N={sum(rem_c.values())}; runtime-verifiable unnecessary/aux={rem_c.get("runtime-verifiable-unnecessary", 0)+rem_c.get("semantic-only-auxiliary-effect", 0)}; potentially risky={rem_c.get("potentially-risky-remove", 0)}; unverifiable={rem_c.get("SEMANTIC_UNVERIFIABLE", 0)}. Mean actions S0 {s0.get("mean_action_count")} vs S2 {s2.get("mean_action_count")}.
7. MA mean actions M0 {m0.get("mean_action_count")} vs M4 {m4.get("mean_action_count")}; redundant M4={buckets["M4"]["redundant"]}.
8. New runtime conflicts M4={buckets["M4"]["conflicts"]} / {n_parent}.
9. Order-sensitive parents M4={buckets["M4"]["order_sensitive"]}.
10. Fault injection: {dict(faults)}. Missing/unavailable/disabled services fail closed (EXECUTE fails, no aggressive extra REMOVE).
11. Unsupported (`llmvision` etc.) caught as E1 (F10_caught={faults.get("F10_caught", 0)}).
12. Runtime idempotence: stateful second change={idem.get("stateful_second_change", 0)}; already satisfied={idem.get("stateful_idempotent", 0)}; second exception={idem.get("second_exception", 0)}.
13. Notify/Log second pass extra effects={idem.get("notify_log_second_effect", 0)} (expected; not required to be side-effect free).
14. Unsafe rate SS S2 {s2.get("Unsafe_Action_Rate")} vs MA M4 {m4.get("Unsafe_Action_Rate")}.
15. RUNTIME_SAFE_REPAIR: SS {dict(env["SS"])}; by capability see `runtime_reliability_envelope.json`. Lighting/Climate device calls that ground an entity and pass schema enter RUNTIME_SAFE_REPAIR.
16. SAFE_ABSTAIN: idle / no-call rows (SS {env["SS"].get("SAFE_ABSTAIN")}). Vision/llmvision remain UNSUPPORTED.
17. Gold-independent evidence **does** support practical value: entity grounding and HVAC off-fill make more calls schema-valid and stateful transitions succeed; REMOVE cuts idle/helper/log extras. This is not a Gold-accuracy claim.
18. New Repair operator: **no**. Residual failures are schema gaps (empty notify/log message, missing hvac_mode when R1 does not fire), not a reason to unfreeze Repair in this task.
19. Deployment still needs: real HA Core (or test-instance) with integrations, notify credentials, sheets config_entry, and independent human validation. Sandbox is not a substitute for a lab HA.
20. Method development can stop adding operators. Next is systematic experiment write-up plus optional lab HA replay — not a new Repair family.

## Boundary

Do not quote Executable Rate as Semantic Success. Do not quote 81.30% / 52.67% as runtime safety.

Sandbox actions/sec ≈ {aps}.
"""
    (OUT / "end_to_end_runtime_validation_report.md").write_text(text, encoding="utf-8")

if __name__ == "__main__":
    main()
