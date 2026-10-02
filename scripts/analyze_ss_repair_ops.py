from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ss_b0_vs_final_gold_y import (
    EXAMPLE_CAP,
    EVENT_FOCUS,
    OUT,
    SCENARIO_MAP,
    SS,
    TEMPORAL_SCENES,
    Y_PATH,
    _lux,
    _obs_flag,
    _pct,
    _rate,
    already_satisfied,
    b0_from_sample,
    compare_actions,
    first_action,
    semantic_class,
    service_kind,
    y_from_rec,
)

A4_PREFIXES = (
    "notify.",
    "persistent_notification.",
    "system_log.",
    "logbook.",
    "google_sheets.",
)
A4_EXACT = {
    "system_log.write",
    "logbook.log",
    "google_sheets.append_sheet",
    "persistent_notification.create",
    "persistent_notification.dismiss",
}
A2_PREFIXES = ("input_boolean.", "input_text.")
A2_EXACT = {"schedule.activate", "scene.turn_on", "scene.turn_off"}

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def add_example(bucket: dict[str, list], key: str, row: dict[str, Any]) -> None:
    lst = bucket.setdefault(key, [])
    if len(lst) < EXAMPLE_CAP:
        lst.append(row)

def is_a4_event(svc: str) -> bool:
    s = str(svc or "")
    return s in A4_EXACT or s.startswith(A4_PREFIXES)

def is_a2_redundant(svc: str) -> bool:
    s = str(svc or "")
    return s in A2_EXACT or s.startswith(A2_PREFIXES)

def param_mismatch_kind(b0: dict[str, Any], y: dict[str, Any]) -> str:
    bp = dict(b0.get("parameters") or {})
    yp = dict(y.get("parameters") or {})
    if not yp and bp:
        return "gold_empty_b0_filled"
    if yp and not bp:
        return "gold_filled_b0_empty"
    if yp and bp:
        return "both_filled_different"
    return "both_empty_but_flagged"

def classify_case2(sample: dict[str, Any], b0_act: dict[str, Any] | None) -> str:
    svc = (b0_act or {}).get("service") or ""
    if already_satisfied(sample, b0_act):
        return "A1_already_satisfied_state"
    if is_a4_event(svc):
        return "A4_event_effect_unnecessary"
    if is_a2_redundant(svc):
        return "A2_redundant_action"
    return "A3_condition_context_misunderstanding"

def classify_case1(b0_act: dict[str, Any] | None, y_act: dict[str, Any] | None) -> str:
    cmp = compare_actions(b0_act, y_act)
    if cmp["exact_action_match"]:
        return "exact_match"
    if not cmp["service_match"]:
        return "B3_service_mismatch"
    b0_ent = (b0_act or {}).get("entity")
    y_ent = (y_act or {}).get("entity")
    if (b0_ent or None) != (y_ent or None):
        if b0_ent in (None, "") and y_ent:
            return "B4_action_target_ambiguity"
        return "B1_entity_mismatch"
    return "B2_parameter_mismatch"

def classify_case3(sample: dict[str, Any], y_act: dict[str, Any] | None, action_type: str) -> str:
    scene = str(sample.get("scene_type") or "")
    svc = (y_act or {}).get("service") or ""
    kind = service_kind(svc) if svc else action_type
    if scene in TEMPORAL_SCENES or svc.startswith("schedule."):
        return "C3_missing_temporal_action"
    if kind == "EVENT_ACTION" or (not svc and action_type == "EVENT_ACTION"):
        return "C2_missing_event_action"
    if kind == "STATE_ACTION" or (not svc and action_type == "STATE_ACTION"):
        return "C1_missing_state_action"
    return "C4_missing_context_action"

def obs_summary(sample: dict[str, Any]) -> dict[str, Any]:
    obs = sample.get("observed") or {}
    return {
        "motion": _obs_flag(sample, "motion_state", "motion"),
        "occupancy": _obs_flag(sample, "occupancy"),
        "window": _obs_flag(sample, "window_state"),
        "door": _obs_flag(sample, "door_state"),
        "lux": _lux(sample),
        "obs_keys": sorted(str(k) for k in list(obs)[:12]),
    }

def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)

def main() -> None:
    y_idx: dict[str, dict[str, Any]] = {}
    with Y_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = str(rec.get("sample_id") or "")
            if sid:
                y_idx[sid] = rec

    case2 = Counter()
    case1 = Counter()
    case3 = Counter()
    c2_svc: dict[str, Counter] = defaultdict(Counter)
    c2_scene: dict[str, Counter] = defaultdict(Counter)
    c1_svc: dict[str, Counter] = defaultdict(Counter)
    c1_scene: dict[str, Counter] = defaultdict(Counter)
    c3_svc: dict[str, Counter] = defaultdict(Counter)
    c3_scene: dict[str, Counter] = defaultdict(Counter)
    c2_motion = Counter()
    c2_lux_bin = Counter()
    param_kind = Counter()
    param_keys_b0 = Counter()
    param_keys_y = Counter()
    svc_pairs = Counter()
    entity_pairs = Counter()
    entity_missing_by_svc = Counter()
    case1_by_svc = Counter()
    event_focus_miss = Counter()
    event_focus_gold = Counter()
    examples: dict[str, list] = {}
    a3_obs_examples: list[dict[str, Any]] = []

    by_type = {
        "STATE_ACTION": Counter(),
        "EVENT_ACTION": Counter(),
    }
    by_scene: dict[str, Counter] = defaultdict(Counter)

    n = 0
    c1_n = c2_n = c3_n = c4_n = 0
    missing_y = 0

    with SS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            sid = str(s.get("sample_id") or "")
            yrec = y_idx.get(sid)
            if not yrec:
                missing_y += 1
                continue
            n += 1
            scene_raw = str(s.get("scene_type") or "unknown")
            scenario = SCENARIO_MAP.get(scene_raw, scene_raw)
            b0_dec, b0_act = b0_from_sample(s)
            y_dec, y_act, action_type = y_from_rec(yrec)
            if action_type not in by_type:
                action_type = service_kind((y_act or {}).get("service") or "") or "STATE_ACTION"
            sc = by_scene[scenario]
            sc["sample_count"] += 1
            atc = by_type[action_type]
            atc["sample_count"] += 1

            compact = {
                "sample_id": sid,
                "scenario": scenario,
                "scene_type": scene_raw,
                "action_type": action_type,
                "b0_service": (b0_act or {}).get("service"),
                "b0_entity": (b0_act or {}).get("entity"),
                "b0_parameters": (b0_act or {}).get("parameters") or {},
                "y_service": (y_act or {}).get("service"),
                "y_entity": (y_act or {}).get("entity"),
                "y_parameters": (y_act or {}).get("parameters") or {},
            }

            ysvc = (y_act or {}).get("service") or ""
            if y_dec == "ACTION" and (ysvc in EVENT_FOCUS or ysvc.startswith("notify.")):
                event_focus_gold[ysvc] += 1

            if b0_dec == "ACTION" and y_dec == "NO_ACTION":
                c2_n += 1
                tag = classify_case2(s, b0_act)
                case2[tag] += 1
                svc = (b0_act or {}).get("service") or "(none)"
                c2_svc[tag][svc] += 1
                c2_scene[tag][scenario] += 1
                sc["REMOVE"] += 1
                atc["REMOVE"] += 1
                add_example(examples, tag, {k: compact[k] for k in compact if k not in {"b0_parameters", "y_parameters"}})
                if tag == "A3_condition_context_misunderstanding":
                    obs = obs_summary(s)
                    mot = obs.get("motion") or "(none)"
                    c2_motion[mot] += 1
                    lux = obs.get("lux")
                    if lux is None:
                        c2_lux_bin["unknown"] += 1
                    elif lux < 50:
                        c2_lux_bin["<50"] += 1
                    elif lux < 80:
                        c2_lux_bin["50-79"] += 1
                    else:
                        c2_lux_bin[">=80"] += 1
                    if len(a3_obs_examples) < EXAMPLE_CAP:
                        a3_obs_examples.append(
                            {
                                **{k: compact[k] for k in compact if k not in {"b0_parameters", "y_parameters"}},
                                "observation": obs,
                            }
                        )
            elif b0_dec == "ACTION" and y_dec == "ACTION":
                c1_n += 1
                tag = classify_case1(b0_act, y_act)
                case1[tag] += 1
                svc = (b0_act or {}).get("service") or "(none)"
                ysvc2 = (y_act or {}).get("service") or "(none)"
                c1_svc[tag][f"{svc} -> {ysvc2}"] += 1
                c1_scene[tag][scenario] += 1
                case1_by_svc[svc] += 1
                if tag != "exact_match":
                    sc["MODIFY"] += 1
                    atc["MODIFY"] += 1
                else:
                    sc["KEEP"] += 1
                    atc["KEEP"] += 1
                add_example(examples, tag, compact)
                if tag == "B3_service_mismatch":
                    svc_pairs[(svc, ysvc2)] += 1
                if tag in {"B1_entity_mismatch", "B4_action_target_ambiguity"}:
                    entity_pairs[((b0_act or {}).get("entity"), (y_act or {}).get("entity"))] += 1
                    if (b0_act or {}).get("entity") in (None, "") and (y_act or {}).get("entity"):
                        entity_missing_by_svc[ysvc2] += 1
                if tag == "B2_parameter_mismatch":
                    pk = param_mismatch_kind(b0_act or {}, y_act or {})
                    param_kind[pk] += 1
                    for k in (b0_act or {}).get("parameters") or {}:
                        param_keys_b0[str(k)] += 1
                    for k in (y_act or {}).get("parameters") or {}:
                        param_keys_y[str(k)] += 1
            elif b0_dec == "NO_ACTION" and y_dec == "ACTION":
                c3_n += 1
                tag = classify_case3(s, y_act, action_type)
                case3[tag] += 1
                svc = (y_act or {}).get("service") or "(none)"
                c3_svc[tag][svc] += 1
                c3_scene[tag][scenario] += 1
                sc["ADD"] += 1
                atc["ADD"] += 1
                add_example(examples, tag, {k: compact[k] for k in compact if k not in {"b0_parameters", "y_parameters"}})
                if svc in EVENT_FOCUS or str(svc).startswith("notify."):
                    event_focus_miss[svc] += 1
            else:
                c4_n += 1
                sc["KEEP"] += 1
                atc["KEEP"] += 1

    if missing_y:
        raise SystemExit(f"Gold Y missing {missing_y} freeze ids; abort")
    if n != 12021:
        print(f"warning: n={n}", flush=True)

    over = {
        "case": "Case 2 B0 ACTION / Gold Y NO_ACTION",
        "n": c2_n,
        "ratio_of_corpus": _rate(c2_n, n),
        "repair_operator": "REMOVE",
        "taxonomy": {
            "A1_already_satisfied_state": {
                "count": case2["A1_already_satisfied_state"],
                "ratio": _rate(case2["A1_already_satisfied_state"], c2_n),
                "meaning": "Target device/helper already in the B0 target state",
                "by_service": c2_svc["A1_already_satisfied_state"].most_common(15),
                "by_scenario": c2_scene["A1_already_satisfied_state"].most_common(),
            },
            "A2_redundant_action": {
                "count": case2["A2_redundant_action"],
                "ratio": _rate(case2["A2_redundant_action"], c2_n),
                "meaning": "Helper/schedule/scene action with no required Gold device or event effect",
                "by_service": c2_svc["A2_redundant_action"].most_common(15),
                "by_scenario": c2_scene["A2_redundant_action"].most_common(),
            },
            "A3_condition_context_misunderstanding": {
                "count": case2["A3_condition_context_misunderstanding"],
                "ratio": _rate(case2["A3_condition_context_misunderstanding"], c2_n),
                "meaning": "Trigger likely matched; Gold says full observation/condition semantics do not warrant an action",
                "by_service": c2_svc["A3_condition_context_misunderstanding"].most_common(15),
                "by_scenario": c2_scene["A3_condition_context_misunderstanding"].most_common(),
                "motion_state": dict(c2_motion),
                "illuminance_bins": dict(c2_lux_bin),
                "observation_examples": a3_obs_examples,
            },
            "A4_event_effect_unnecessary": {
                "count": case2["A4_event_effect_unnecessary"],
                "ratio": _rate(case2["A4_event_effect_unnecessary"], c2_n),
                "meaning": "Gold Y is NO_ACTION on notify/log/sheet/notification; B0 still produced an event effect",
                "by_service": c2_svc["A4_event_effect_unnecessary"].most_common(15),
                "by_scenario": c2_scene["A4_event_effect_unnecessary"].most_common(),
            },
        },
        "assignment_priority": [
            "detectable already-satisfied target -> A1",
            "notify/log/sheet/persistent_notification -> A4",
            "input_boolean/input_text/schedule.activate/scene.* -> A2",
            "remaining device state fires (light/climate) -> A3",
        ],
        "already_satisfied_note": (
            "A1 is 0 on this corpus: freeze observations almost never include the target "
            "light/climate entity state, so already-off/already-on cannot be proven."
        ),
        "examples": {
            "A1_already_satisfied_state": examples.get("A1_already_satisfied_state", []),
            "A2_redundant_action": examples.get("A2_redundant_action", []),
            "A3_condition_context_misunderstanding": examples.get(
                "A3_condition_context_misunderstanding", []
            ),
            "A4_event_effect_unnecessary": examples.get("A4_event_effect_unnecessary", []),
        },
    }
    dump(OUT / "over_execution_taxonomy.json", over)

    missing_target = case1["B4_action_target_ambiguity"]
    entity_missing_ratio = {
        svc: {"missing_target": c, "case1_for_b0_or_gold_service_proxy": c, "note": "Gold entity present, B0 entity null"}
        for svc, c in entity_missing_by_svc.most_common()
    }
    correction = {
        "case": "Case 1 B0 ACTION / Gold Y ACTION",
        "n": c1_n,
        "ratio_of_corpus": _rate(c1_n, n),
        "exact_match": case1["exact_match"],
        "repair_operator": "MODIFY",
        "taxonomy": {
            "B1_entity_mismatch": {
                "count": case1["B1_entity_mismatch"],
                "ratio": _rate(case1["B1_entity_mismatch"], c1_n),
                "meaning": "Both sides name a target and the targets differ",
                "by_service_pair": c1_svc["B1_entity_mismatch"].most_common(10),
                "by_scenario": c1_scene["B1_entity_mismatch"].most_common(),
            },
            "B2_parameter_mismatch": {
                "count": case1["B2_parameter_mismatch"],
                "ratio": _rate(case1["B2_parameter_mismatch"], c1_n),
                "meaning": "Same service and entity; data/parameters differ",
                "parameter_error_types": dict(param_kind),
                "b0_parameter_keys": param_keys_b0.most_common(15),
                "gold_parameter_keys": param_keys_y.most_common(15),
                "by_service_pair": c1_svc["B2_parameter_mismatch"].most_common(10),
                "by_scenario": c1_scene["B2_parameter_mismatch"].most_common(),
                "caution": (
                    "Dominant type is gold_empty_b0_filled. Gold prompt examples used data:{}. "
                    "MODIFY must not blindly empty B0 log/sheet payloads."
                ),
            },
            "B3_service_mismatch": {
                "count": case1["B3_service_mismatch"],
                "ratio": _rate(case1["B3_service_mismatch"], c1_n),
                "meaning": "Different service (turn_on vs turn_off, or different domain)",
                "top_pairs": [{"b0": a, "y": b, "count": c} for (a, b), c in svc_pairs.most_common(15)],
                "by_scenario": c1_scene["B3_service_mismatch"].most_common(),
            },
            "B4_action_target_ambiguity": {
                "count": case1["B4_action_target_ambiguity"],
                "ratio": _rate(case1["B4_action_target_ambiguity"], c1_n),
                "meaning": "Same service; B0 omitted entity while Gold names a device",
                "entity_missing_count": missing_target,
                "entity_missing_ratio_of_case1": _rate(missing_target, c1_n),
                "by_gold_service": entity_missing_by_svc.most_common(10),
                "top_entity_pairs": [
                    {"b0": a, "y": b, "count": c} for (a, b), c in entity_pairs.most_common(8)
                ],
                "by_scenario": c1_scene["B4_action_target_ambiguity"].most_common(),
            },
        },
        "entity_missing_by_gold_service": entity_missing_ratio,
        "assignment_priority": [
            "service differs -> B3",
            "B0 entity null and Gold entity present -> B4",
            "both entities non-null and different -> B1",
            "else parameter/data differs -> B2",
        ],
        "examples": {
            "B1_entity_mismatch": examples.get("B1_entity_mismatch", []),
            "B2_parameter_mismatch": examples.get("B2_parameter_mismatch", []),
            "B3_service_mismatch": examples.get("B3_service_mismatch", []),
            "B4_action_target_ambiguity": examples.get("B4_action_target_ambiguity", []),
        },
    }
    dump(OUT / "action_correction_taxonomy.json", correction)

    missing = {
        "case": "Case 3 B0 NO_ACTION / Gold Y ACTION",
        "n": c3_n,
        "ratio_of_corpus": _rate(c3_n, n),
        "repair_operator": "ADD",
        "taxonomy": {
            "C1_missing_state_action": {
                "count": case3["C1_missing_state_action"],
                "ratio": _rate(case3["C1_missing_state_action"], c3_n),
                "meaning": "Gold requires a device state change; B0 stayed idle",
                "by_gold_service": c3_svc["C1_missing_state_action"].most_common(10),
                "by_scenario": c3_scene["C1_missing_state_action"].most_common(),
                "context_note": (
                    "All C1 rows in this corpus are climate.set_hvac_mode. They also need "
                    "window+HVAC joint reasoning, but the missed action itself is a state change."
                ),
            },
            "C2_missing_event_action": {
                "count": case3["C2_missing_event_action"],
                "ratio": _rate(case3["C2_missing_event_action"], c3_n),
                "meaning": "Gold requires notify/log/sheet/logbook; B0 produced nothing",
                "by_gold_service": c3_svc["C2_missing_event_action"].most_common(10),
                "by_scenario": c3_scene["C2_missing_event_action"].most_common(),
            },
            "C3_missing_temporal_action": {
                "count": case3["C3_missing_temporal_action"],
                "ratio": _rate(case3["C3_missing_temporal_action"], c3_n),
                "meaning": "Periodic/schedule scene; Gold requires the timed effect",
                "by_gold_service": c3_svc["C3_missing_temporal_action"].most_common(10),
                "by_scenario": c3_scene["C3_missing_temporal_action"].most_common(),
            },
            "C4_missing_context_action": {
                "count": case3["C4_missing_context_action"],
                "ratio": _rate(case3["C4_missing_context_action"], c3_n),
                "meaning": "Residual multi-state misses that are not a state service, event service, or temporal scene",
                "by_gold_service": c3_svc["C4_missing_context_action"].most_common(10),
                "by_scenario": c3_scene["C4_missing_context_action"].most_common(),
            },
        },
        "event_focus": {
            svc: {
                "gold_ACTION": event_focus_gold[svc],
                "b0_missed": event_focus_miss[svc],
                "miss_rate_among_gold_action": _rate(event_focus_miss[svc], event_focus_gold[svc]),
            }
            for svc in sorted(set(event_focus_gold) | set(event_focus_miss))
        },
        "assignment_priority": [
            "periodic/schedule scene -> C3",
            "Gold event service -> C2",
            "Gold state service -> C1",
            "else -> C4",
        ],
        "examples": {
            "C1_missing_state_action": examples.get("C1_missing_state_action", []),
            "C2_missing_event_action": examples.get("C2_missing_event_action", []),
            "C3_missing_temporal_action": examples.get("C3_missing_temporal_action", []),
            "C4_missing_context_action": examples.get("C4_missing_context_action", []),
        },
    }
    dump(OUT / "missing_action_taxonomy.json", missing)

    def type_block(name: str) -> dict[str, Any]:
        c = by_type[name]
        sn = c["sample_count"]
        err = c["REMOVE"] + c["MODIFY"] + c["ADD"]
        ops = {"REMOVE": c["REMOVE"], "MODIFY": c["MODIFY"], "ADD": c["ADD"]}
        primary = max(ops, key=ops.get) if err else "KEEP"
        return {
            "sample_count": sn,
            "error_count": err,
            "error_rate": _rate(err, sn),
            "KEEP": c["KEEP"],
            "operators": {
                "REMOVE": {"count": c["REMOVE"], "ratio_of_type": _rate(c["REMOVE"], sn), "ratio_of_errors": _rate(c["REMOVE"], err)},
                "MODIFY": {"count": c["MODIFY"], "ratio_of_type": _rate(c["MODIFY"], sn), "ratio_of_errors": _rate(c["MODIFY"], err)},
                "ADD": {"count": c["ADD"], "ratio_of_type": _rate(c["ADD"], sn), "ratio_of_errors": _rate(c["ADD"], err)},
            },
            "primary_repair_operator": primary,
        }

    type_report = {
        "STATE_ACTION": type_block("STATE_ACTION"),
        "EVENT_ACTION": type_block("EVENT_ACTION"),
        "notes": {
            "STATE_ACTION": (
                "Primary operator is REMOVE (lighting over-fire + helper/log side effects "
                "on state-typed samples), then MODIFY (climate missing target), then ADD "
                "(climate.set_hvac_mode misses)."
            ),
            "EVENT_ACTION": (
                "REMOVE and MODIFY are both large: Gold NO_ACTION on extra system_log/"
                "sheets vs Gold ACTION with empty data. ADD is required for notify.mobile_app "
                "(109/109 missed)."
            ),
        },
    }
    dump(OUT / "repair_action_type_analysis.json", type_report)

    scenario_out: dict[str, Any] = {}
    for name in ["lighting", "climate", "security", "visual", "periodic", "schedule", "appliance", "scene"]:
        c = by_scene[name]
        sn = c["sample_count"]
        err = c["REMOVE"] + c["MODIFY"] + c["ADD"]
        ops = {"REMOVE": c["REMOVE"], "MODIFY": c["MODIFY"], "ADD": c["ADD"]}
        main = max(ops, key=ops.get) if err else "none"
        if err and ops[main] == 0:
            main = "none"
        scenario_out[name] = {
            "sample_count": sn,
            "error_count": err,
            "main_error_type": main if err else "none",
            "KEEP": c["KEEP"],
            "REMOVE_ratio": _rate(c["REMOVE"], sn),
            "MODIFY_ratio": _rate(c["MODIFY"], sn),
            "ADD_ratio": _rate(c["ADD"], sn),
            "counts": {
                "REMOVE": c["REMOVE"],
                "MODIFY": c["MODIFY"],
                "ADD": c["ADD"],
                "KEEP": c["KEEP"],
            },
        }
    dump(OUT / "repair_scenario_analysis.json", scenario_out)

    p1_cov = c2_n
    p2_cov = c1_n - case1["exact_match"]
    p3_cov = c3_n
    lines = [
        "# Repair design recommendation (B0 vs Final Gold Y)",
        "",
        "Ground truth = `single_scene_final_y.jsonl`. Freeze, B0, and Gold Y were not modified.",
        "Repair is **not implemented** in this step.",
        "",
        f"- N = {n}",
        f"- Case 2 over-execution (REMOVE) = **{c2_n}** ({_pct(_rate(c2_n, n))})",
        f"- Case 1 semantic mismatch (MODIFY) = **{p2_cov}** ({_pct(_rate(p2_cov, n))})",
        f"- Case 3 missed ACTION (ADD) = **{c3_n}** ({_pct(_rate(c3_n, n))})",
        f"- Case 4 keep idle = {c4_n}",
        "",
        "## 1. Does Repair need REMOVE?",
        "",
        "**Yes.** REMOVE is the largest operator, covering "
        f"{c2_n} samples ({_pct(_rate(c2_n, n))} of the corpus).",
        "",
        md_table(
            ["type", "count", "share of Case 2"],
            [
                ["A1 already satisfied", case2["A1_already_satisfied_state"], _pct(_rate(case2["A1_already_satisfied_state"], c2_n))],
                ["A2 redundant helper/schedule", case2["A2_redundant_action"], _pct(_rate(case2["A2_redundant_action"], c2_n))],
                ["A3 trigger/context misunderstanding", case2["A3_condition_context_misunderstanding"], _pct(_rate(case2["A3_condition_context_misunderstanding"], c2_n))],
                ["A4 event/effect unnecessary", case2["A4_event_effect_unnecessary"], _pct(_rate(case2["A4_event_effect_unnecessary"], c2_n))],
            ],
        ),
        "",
        "Highest-confidence REMOVE slices vs Gold: **A4** (extra notify/log/sheet) and **A2** "
        "(input_boolean / schedule.activate with no Gold effect). **A3** is lighting "
        "`light.turn_on/off` after motion/illuminance; Gold says NO_ACTION, so REMOVE is "
        "the correct operator even though trigger likely fired. A1 is 0 because the freeze "
        "rarely exposes current light/HVAC entity state.",
        "",
        "## 2. Does Repair need MODIFY?",
        "",
        f"**Yes.** All {c1_n} Case-1 rows fail exact service+entity+data match.",
        "",
        md_table(
            ["type", "count", "share of Case 1"],
            [
                ["B1 entity mismatch (wrong named target)", case1["B1_entity_mismatch"], _pct(_rate(case1["B1_entity_mismatch"], c1_n))],
                ["B2 parameter mismatch", case1["B2_parameter_mismatch"], _pct(_rate(case1["B2_parameter_mismatch"], c1_n))],
                ["B3 service mismatch", case1["B3_service_mismatch"], _pct(_rate(case1["B3_service_mismatch"], c1_n))],
                ["B4 target omitted (B0 entity null)", case1["B4_action_target_ambiguity"], _pct(_rate(case1["B4_action_target_ambiguity"], c1_n))],
            ],
        ),
        "",
        "MODIFY is mostly **fill the omitted climate/light target** (B4), not swap a wrong "
        "device (B1 is near-zero). Parameter errors are dominated by Gold empty `data` vs "
        "B0 filled log/sheet payload — do not implement MODIFY as wiping parameters.",
        "",
        "## 3. Does Repair need ADD?",
        "",
        f"**Yes, but it is the smallest decision-error cell:** {c3_n} "
        f"({_pct(_rate(c3_n, n))}).",
        "",
        md_table(
            ["type", "count", "share of Case 3"],
            [
                ["C1 missing state", case3["C1_missing_state_action"], _pct(_rate(case3["C1_missing_state_action"], c3_n))],
                ["C2 missing event", case3["C2_missing_event_action"], _pct(_rate(case3["C2_missing_event_action"], c3_n))],
                ["C3 missing temporal", case3["C3_missing_temporal_action"], _pct(_rate(case3["C3_missing_temporal_action"], c3_n))],
                ["C4 missing context (residual)", case3["C4_missing_context_action"], _pct(_rate(case3["C4_missing_context_action"], c3_n))],
            ],
        ),
        "",
        "ADD is required for `notify.mobile_app` (Gold ACTION 109, B0 missed 109) and for "
        "`climate.set_hvac_mode` when B0 stayed idle (C1). Lighting has **0** Case-3 rows, "
        "so ADD-to-turn-on-lights is not a corpus-level need.",
        "",
        "## 4. Operator priority",
        "",
        f"### Priority 1 — REMOVE ({p1_cov} errors, {_pct(_rate(p1_cov, n))} of N)",
        "",
        "Largest disagreement. First slices: A4 extra event effects, A2 helper/schedule "
        "side effects, then A3 lighting motion over-fire. Covers lighting (2680) and "
        "most security over-exec.",
        "",
        f"### Priority 2 — MODIFY ({p2_cov} errors, {_pct(_rate(p2_cov, n))} of N)",
        "",
        "Bind omitted `climate.living_room_ac` / `light.living_room` (B4). Service rewrite "
        "only for the smaller B3 set (persistent_notification.create → logbook.log, "
        "system_log.write → climate.set_preset_mode). Do not treat Gold empty data as a "
        "payload-delete rule.",
        "",
        f"### Priority 3 — ADD ({p3_cov} errors, {_pct(_rate(p3_cov, n))} of N)",
        "",
        "Insert Gold event/state when B0 is idle: climate.set_hvac_mode (601), "
        "notify.mobile_app (109), sheets (52), logbook (15). ADD must stay conservative; "
        "empty B0 is usually correct (Case 4 = 3290).",
        "",
        "## STATE vs EVENT",
        "",
        md_table(
            ["action_type", "N", "REMOVE", "MODIFY", "ADD", "primary"],
            [
                [
                    "STATE_ACTION",
                    type_report["STATE_ACTION"]["sample_count"],
                    type_report["STATE_ACTION"]["operators"]["REMOVE"]["count"],
                    type_report["STATE_ACTION"]["operators"]["MODIFY"]["count"],
                    type_report["STATE_ACTION"]["operators"]["ADD"]["count"],
                    type_report["STATE_ACTION"]["primary_repair_operator"],
                ],
                [
                    "EVENT_ACTION",
                    type_report["EVENT_ACTION"]["sample_count"],
                    type_report["EVENT_ACTION"]["operators"]["REMOVE"]["count"],
                    type_report["EVENT_ACTION"]["operators"]["MODIFY"]["count"],
                    type_report["EVENT_ACTION"]["operators"]["ADD"]["count"],
                    type_report["EVENT_ACTION"]["primary_repair_operator"],
                ],
            ],
        ),
        "",
        "STATE_ACTION mainly needs **REMOVE**, then MODIFY, then ADD.",
        "EVENT_ACTION needs **REMOVE and MODIFY together**, plus a small but complete ADD "
        "path for notify.",
        "",
        "## By scenario (operator / N)",
        "",
        md_table(
            ["scenario", "N", "REMOVE", "MODIFY", "ADD", "main"],
            [
                [
                    name,
                    scenario_out[name]["sample_count"],
                    _pct(scenario_out[name]["REMOVE_ratio"]),
                    _pct(scenario_out[name]["MODIFY_ratio"]),
                    _pct(scenario_out[name]["ADD_ratio"]),
                    scenario_out[name]["main_error_type"],
                ]
                for name in [
                    "lighting",
                    "climate",
                    "security",
                    "visual",
                    "periodic",
                    "schedule",
                    "appliance",
                    "scene",
                ]
            ],
        ),
        "",
        "This document stops before Repair implementation.",
        "",
    ]
    (OUT / "repair_design_recommendation.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "n": n,
                "REMOVE": c2_n,
                "MODIFY": p2_cov,
                "ADD": c3_n,
                "KEEP": c4_n,
                "case2": dict(case2),
                "case1": dict(case1),
                "case3": dict(case3),
                "state_primary": type_report["STATE_ACTION"]["primary_repair_operator"],
                "event_primary": type_report["EVENT_ACTION"]["primary_repair_operator"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

if __name__ == "__main__":
    main()
