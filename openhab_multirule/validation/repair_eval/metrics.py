from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

def _rate(n: float, d: float) -> float:
    return round(float(n) / max(float(d), 1.0), 4)

def _state_satisfies(states: dict[str, str], ob: dict[str, Any]) -> bool:
    ent = str(ob.get("entity", ""))
    item = ent.split(".")[-1]
    op = str(ob.get("operation", "")).lower()
    params = ob.get("parameters") or {}
    cmd = str(params.get("command", "")).upper()
    st = str(states.get(item, "")).upper()
    if not st or st in ("NULL", "UNDEF"):
        return False
    if op in ("turn_on", "on") or cmd == "ON":
        return st == "ON"
    if op in ("turn_off", "off") or cmd == "OFF":
        return st == "OFF"
    if op == "open" or cmd == "OPEN":
        return st == "OPEN"
    if op == "close" or cmd == "CLOSED":
        return st == "CLOSED"
    return st == cmd if cmd else True

def obligation_satisfaction(states: dict[str, str], reference: dict[str, Any]) -> dict[str, Any]:

    obs = reference.get("expected_obligations") or []
    if not obs:
        return {"n": 0, "satisfied": 0, "rate": 1.0 if not reference.get("repair_required") else 0.0}
    by_ent: dict[str, list[dict[str, Any]]] = {}
    for ob in obs:
        ent = str(ob.get("entity", ""))
        by_ent.setdefault(ent, []).append(ob)
    n = len(by_ent)
    satisfied = 0
    for _ent, group in by_ent.items():
        if any(_state_satisfies(states, ob) for ob in group):
            satisfied += 1
    return {"n": n, "satisfied": satisfied, "rate": _rate(satisfied, n)}

def semantic_success(states: dict[str, str], reference: dict[str, Any]) -> bool:
    if reference.get("ambiguity"):
        return True
    if not reference.get("repair_required"):
        return True
    sat = obligation_satisfaction(states, reference)
    if sat["n"] == 0:

        acc = reference.get("acceptable_final_behaviors") or []
        if not acc:
            return False
        return any(_state_satisfies(states, a) for a in acc)

    return sat["rate"] >= 1.0 and sat["satisfied"] >= 1

def conflict_actions_from_reference(reference: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for c in reference.get("conflict_relations") or []:
        out.append(dict(c))
    if not out and reference.get("repair_required"):
        for ob in reference.get("expected_obligations") or []:
            out.append({"type": "MISSING", "entity": ob.get("entity"), "operation": ob.get("operation")})
    return out

def repaired_conflict_actions(
    before_states: dict[str, str],
    after_states: dict[str, str],
    reference: dict[str, Any],
) -> dict[str, Any]:

    conflicts = conflict_actions_from_reference(reference)
    total = len(conflicts)
    repaired = 0
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "repaired": 0})
    for c in conflicts:
        ctype = str(c.get("type", "UNKNOWN")).upper()
        by_type[ctype]["total"] += 1
        entity = str(c.get("entity", "")).split(".")[-1]

        if ctype == "MISSING":
            obs = [o for o in (reference.get("expected_obligations") or []) if str(o.get("entity", "")).endswith(entity) or entity in str(o.get("entity", ""))]
            ok = any(_state_satisfies(after_states, o) for o in obs) if obs else False
            if not obs:

                ok = after_states.get(entity, "NULL") not in ("NULL", "UNDEF", "")
        elif ctype in ("OPPOSING", "REPLACE"):

            acc = reference.get("acceptable_final_behaviors") or []
            ok = any(
                _state_satisfies(after_states, {"entity": a.get("entity", entity), "operation": a.get("operation", ""), "parameters": {}})
                for a in acc
            ) if acc else after_states.get(entity) in ("ON", "OFF")
        elif ctype == "EXTRA":

            ok = semantic_success(after_states, reference)
        else:
            ok = semantic_success(after_states, reference) and not semantic_success(before_states, reference)
        if ok:
            repaired += 1
            by_type[ctype]["repaired"] += 1
    carr = _rate(repaired, total) if total else (1.0 if not reference.get("repair_required") else 0.0)
    return {
        "total_conflict_actions": total,
        "repaired_conflict_actions": repaired,
        "carr": carr,
        "by_type": {k: {**v, "repair_rate": _rate(v["repaired"], v["total"])} for k, v in by_type.items()},
    }

def partial_repair_bucket(carr: float) -> str:
    if carr >= 1.0:
        return "FULLY_REPAIRED"
    if carr >= 0.75:
        return "MOSTLY_REPAIRED"
    if carr >= 0.25:
        return "PARTIALLY_REPAIRED"
    if carr > 0:
        return "MINIMALLY_REPAIRED"
    return "UNREPAIRED"

def false_repair_flags(
    *,
    before_states: dict[str, str],
    after_states: dict[str, str],
    reference: dict[str, Any],
    repaired_tokens: list[str],
    b0_tokens: list[str],
) -> dict[str, Any]:
    flags = []
    if not reference.get("repair_required") and repaired_tokens and repaired_tokens != b0_tokens:
        flags.append("UnnecessaryAction")
    for fb in reference.get("forbidden_behaviors") or []:
        ent = str(fb.get("entity", "")).split(".")[-1]
        if after_states.get(ent) in ("ON", "OFF") and reference.get("ambiguity"):
            flags.append("AmbiguousForcedAction")
            break

    b = obligation_satisfaction(before_states, reference)["satisfied"]
    a = obligation_satisfaction(after_states, reference)["satisfied"]
    if a < b:
        flags.append("StateRegression")
    if reference.get("ambiguity") and repaired_tokens and repaired_tokens != b0_tokens:
        flags.append("AmbiguousNonAbstain")
    return {"false_repair": bool(flags), "flags": flags}

def classify_failure(row: dict[str, Any]) -> str:
    if row.get("failure_class") == "INFRASTRUCTURE_FAILURE":
        return "INFRASTRUCTURE_FAILURE"
    if row.get("failure_class") == "GROUNDING_FAILURE":
        return "GROUNDING_FAILURE"
    if not row.get("runtime_success"):
        return "RUNTIME_EXECUTION_FAILURE"
    if row.get("false_repair"):
        return "FALSE_REPAIR"
    if row.get("carr", 0) <= 0 and row.get("repair_required"):
        return "CONFLICT_UNRESOLVED"
    if row.get("complete_repair"):
        return "NONE"
    if 0 < row.get("carr", 0) < 1:
        return "CONFLICT_UNRESOLVED"
    return "NONE"

def aggregate_track(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    infra = [r for r in rows if r.get("failure_class") == "INFRASTRUCTURE_FAILURE"]
    ready = [r for r in rows if r.get("failure_class") != "INFRASTRUCTURE_FAILURE"]
    conflicted = [r for r in ready if r.get("repair_required") and not r.get("ambiguity")]
    complete = [r for r in conflicted if r.get("complete_repair")]
    mostly = [r for r in conflicted if r.get("partial_bucket") in ("FULLY_REPAIRED", "MOSTLY_REPAIRED")]

    def mean(key: str, subset: list[dict[str, Any]]) -> float:
        if not subset:
            return 0.0
        return _rate(sum(float(r.get(key) or 0) for r in subset), len(subset))

    return {
        "n_intent": len(rows),
        "n_infrastructure_failure": len(infra),
        "n_repair_ready": len(ready),
        "n_conflicted": len(conflicted),
        "runtime_success_intent": _rate(sum(1 for r in rows if r.get("runtime_success")), len(rows)),
        "runtime_success_ready": _rate(sum(1 for r in ready if r.get("runtime_success")), len(ready)),
        "before_semantic_success": mean("before_semantic_success", ready),
        "after_semantic_success": mean("after_semantic_success", ready),
        "semantic_repair_gain": mean("semantic_repair_gain", ready),
        "complete_repair_rate": _rate(len(complete), len(conflicted)),
        "mostly_plus_complete": _rate(len(mostly), len(conflicted)),
        "micro_carr": mean("carr", conflicted),
        "macro_carr": mean("carr", conflicted),
        "conflict_resolution_rate": mean("conflict_resolution_rate", conflicted),
        "false_repair_rate": _rate(sum(1 for r in ready if r.get("false_repair")), len(ready)),
        "new_conflict_rate": _rate(sum(1 for r in ready if r.get("new_conflict")), len(ready)),
        "grounding_success_rate": _rate(sum(1 for r in ready if r.get("grounding_success")), len(ready)),
        "total_conflict_actions": sum(int(r.get("total_conflict_actions") or 0) for r in conflicted),
        "repaired_conflict_actions": sum(int(r.get("repaired_conflict_actions") or 0) for r in conflicted),
        "partial_buckets": dict(Counter(r.get("partial_bucket") for r in conflicted)),
    }
