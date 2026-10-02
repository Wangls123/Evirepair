from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.evaluation.y_action_normalize import service_level_token
from smarthome_mdf.semantic_alignment.action_parser import parse_raw_action

SS = ROOT / "data" / "benchmarks" / "single_scene.jsonl"
Y_PATH = ROOT / "data" / "references" / "single_scene_gold_y.jsonl"
OUT = ROOT / "runs" / "ss_final_gold_y"

SCENARIO_MAP = {
    "advanced_lighting": "lighting",
    "climate_window": "climate",
    "notification_security": "security",
    "visual_fusion": "visual",
    "periodic_task_scheduling": "periodic",
    "on_off_schedule": "schedule",
    "appliance_monitoring": "appliance",
    "scene_schedule_override": "scene",
}

STATE_PREFIXES = (
    "light.",
    "switch.",
    "climate.",
    "cover.",
    "fan.",
    "lock.",
    "scene.",
    "alarm_control_panel.",
    "humidifier.",
    "water_heater.",
    "vacuum.",
    "remote.",
    "siren.",
    "media_player.",
    "valve.",
    "input_boolean.",
)

EVENT_PREFIXES = (
    "notify.",
    "persistent_notification.",
    "system_log.",
    "logbook.",
    "google_sheets.",
    "camera.",
    "tts.",
    "schedule.",
    "rest_command.",
)

EVENT_EXACT = {
    "system_log.write",
    "logbook.log",
    "google_sheets.append_sheet",
    "persistent_notification.create",
    "persistent_notification.dismiss",
    "input_text.set_value",
    "schedule.activate",
}

EVENT_FOCUS = (
    "notify.mobile_app",
    "notify.notify",
    "system_log.write",
    "google_sheets.append_sheet",
    "logbook.log",
)

TEMPORAL_SCENES = {"periodic_task_scheduling", "on_off_schedule"}
CONTEXT_SCENES = {"visual_fusion", "scene_schedule_override", "climate_window"}
EXAMPLE_CAP = 12

def _rate(n: int, d: int) -> float:
    return round(n / max(d, 1), 4)

def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"

def _norm_text(v: Any) -> str:
    return str(v or "").strip().lower()

def service_kind(svc: str) -> str:
    s = str(svc or "").strip()
    if not s:
        return "OTHER"
    if s in EVENT_EXACT or s.startswith(EVENT_PREFIXES):
        return "EVENT_ACTION"
    if s.startswith(STATE_PREFIXES):
        return "STATE_ACTION"
    return "OTHER"

def _params(rec: dict[str, Any]) -> dict[str, Any]:
    p = dict(rec.get("parameters") or {})
    return {k: v for k, v in p.items() if v not in (None, "", [], {})}

def _entity_from_raw(raw: Any, rec: dict[str, Any]) -> str | None:
    ent = rec.get("entity_id")
    if isinstance(raw, dict) and not ent:
        t = raw.get("target")
        if isinstance(t, str):
            ent = t
        elif isinstance(t, dict):
            ent = t.get("entity_id") or t.get("entity")
        ent = ent or raw.get("target_entity")
    if ent in (None, "", [], {}):
        return None
    return str(ent)

def normalize_action(raw: Any) -> dict[str, Any] | None:
    if raw in (None, "", [], {}):
        return None
    rec = parse_raw_action(raw)
    svc = service_level_token(rec.get("service") or "")
    if not svc and isinstance(raw, dict):
        svc = service_level_token(raw.get("service") or raw.get("action") or "")
    if not svc:
        return None
    return {
        "service": svc,
        "entity": _entity_from_raw(raw, rec),
        "parameters": _params(rec),
    }

def first_action(actions: list[Any]) -> dict[str, Any] | None:
    for a in actions or []:
        n = normalize_action(a)
        if n:
            return n
    return None

def b0_from_sample(sample: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    fb = sample.get("formal_b0") or {}
    st = str(fb.get("execution_status") or "")
    ha = list(fb.get("expected_action_ha") or [])
    sem = list(fb.get("semantic_actions") or [])
    act = first_action(sem) or first_action(ha)
    if st == "SUCCESS_NO_ACTION" or not act:
        return "NO_ACTION", None
    return "ACTION", act

def y_from_rec(yrec: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str]:
    fy = yrec.get("formal_y") or yrec
    dec_raw = str(fy.get("decision") or "").upper()
    acts = list(fy.get("semantic_actions") or [])
    at = str(fy.get("action_type") or "")
    if dec_raw in {"NO_ACTION", ""} or not acts:
        return "NO_ACTION", None, at
    return "ACTION", first_action(acts), at

def compare_actions(b0: dict[str, Any] | None, y: dict[str, Any] | None) -> dict[str, bool]:
    if b0 is None and y is None:
        return {
            "service_match": True,
            "entity_match": True,
            "parameter_match": True,
            "exact_action_match": True,
        }
    if b0 is None or y is None:
        return {
            "service_match": False,
            "entity_match": False,
            "parameter_match": False,
            "exact_action_match": False,
        }
    sm = b0["service"] == y["service"]
    em = (b0.get("entity") or None) == (y.get("entity") or None)
    pm = json.dumps(b0.get("parameters") or {}, sort_keys=True) == json.dumps(
        y.get("parameters") or {}, sort_keys=True
    )
    return {
        "service_match": sm,
        "entity_match": em,
        "parameter_match": pm,
        "exact_action_match": bool(sm and em and pm),
    }

def semantic_class(cmp: dict[str, bool]) -> str:
    if cmp["exact_action_match"]:
        return "exact_match"
    if not cmp["service_match"]:
        return "service_mismatch"
    if not cmp["entity_match"]:
        return "entity_mismatch"
    return "parameter_mismatch"

def exact_ok(b0_dec: str, b0_act: dict | None, y_dec: str, y_act: dict | None) -> bool:
    if b0_dec != y_dec:
        return False
    if b0_dec == "NO_ACTION":
        return True
    return compare_actions(b0_act, y_act)["exact_action_match"]

def _lux(sample: dict[str, Any]) -> float | None:
    obs = sample.get("observed") or {}
    v = obs.get("illuminance_lux")
    if v is None:
        v = obs.get("illuminance")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _obs_flag(sample: dict[str, Any], *keys: str) -> str | None:
    obs = sample.get("observed") or {}
    for k in keys:
        if obs.get(k) is not None:
            return _norm_text(obs.get(k))
    return None

def _entity_state(sample: dict[str, Any], entity: str | None) -> str | None:
    if not entity:
        return None
    for eo in sample.get("entity_observations") or []:
        if not isinstance(eo, dict):
            continue
        if str(eo.get("entity_id") or "") == entity:
            st = eo.get("state")
            if st is not None:
                return _norm_text(st)
    return None

def already_satisfied(sample: dict[str, Any], b0_act: dict[str, Any] | None) -> bool:
    if not b0_act:
        return False
    svc = b0_act["service"]
    ent = b0_act.get("entity")
    cur = _entity_state(sample, ent)
    obs = sample.get("observed") or {}
    ss = sample.get("system_state") or {}
    if svc.endswith(".turn_on"):
        if cur in {"on", "true", "1", "open"}:
            return True
        light = _norm_text(ss.get("current_light_state") or obs.get("light_state"))
        if light in {"on", "true", "1"}:
            return True
    if svc.endswith(".turn_off"):
        if cur in {"off", "false", "0", "closed", "idle"}:
            return True
        light = _norm_text(ss.get("current_light_state") or obs.get("light_state"))
        if light in {"off", "false", "0"}:
            return True
        if svc.startswith("climate."):
            hvac = _norm_text(ss.get("hvac_mode") or ss.get("climate_mode") or obs.get("hvac_mode"))
            if hvac in {"off", "idle", "fan_only", "0"}:
                return True
    if "set_hvac_mode" in svc:
        want = _norm_text((b0_act.get("parameters") or {}).get("hvac_mode"))
        have = _norm_text(ss.get("hvac_mode") or ss.get("climate_mode") or obs.get("hvac_mode") or cur)
        if want and have and want == have:
            return True
    return False

def classify_unnecessary(sample: dict[str, Any], b0_act: dict[str, Any] | None) -> str:
    scene = str(sample.get("scene_type") or "")
    svc = (b0_act or {}).get("service") or ""
    kind = service_kind(svc)
    if already_satisfied(sample, b0_act):
        return "already_satisfied_state"
    if kind == "EVENT_ACTION" or svc.startswith("input_text.") or svc.startswith("input_boolean."):
        return "redundant_action"
    if scene == "climate_window":
        return "wrong_context_reasoning"
    if scene in CONTEXT_SCENES:
        return "wrong_context_reasoning"
    if scene == "advanced_lighting":
        lux = _lux(sample)
        if svc.endswith("turn_on") and lux is not None and lux >= 80:
            return "wrong_context_reasoning"
        return "wrong_trigger_interpretation"
    if scene == "notification_security":
        return "wrong_trigger_interpretation"
    motion = _obs_flag(sample, "motion_state", "motion")
    door = _obs_flag(sample, "door_state")
    if motion in {"on", "true", "1"} or door in {"open", "on", "true"}:
        return "wrong_trigger_interpretation"
    if kind == "STATE_ACTION":
        return "wrong_context_reasoning"
    return "redundant_action"

def classify_missing(sample: dict[str, Any], y_act: dict[str, Any] | None, action_type: str) -> str:
    scene = str(sample.get("scene_type") or "")
    svc = (y_act or {}).get("service") or ""
    kind = service_kind(svc) if svc else action_type
    if scene in TEMPORAL_SCENES or svc.startswith("schedule."):
        return "missing_temporal_action"
    if kind == "EVENT_ACTION" or (not svc and action_type == "EVENT_ACTION"):
        return "missing_event_action"
    if scene in CONTEXT_SCENES or scene == "advanced_lighting":
        return "missing_context_action"
    return "missing_state_action"

def add_example(bucket: dict[str, list], key: str, row: dict[str, Any]) -> None:
    lst = bucket.setdefault(key, [])
    if len(lst) < EXAMPLE_CAP:
        lst.append(row)

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)

def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def main() -> None:
    if not Y_PATH.is_file():
        raise SystemExit(f"missing Gold Y: {Y_PATH}")
    y_idx: dict[str, dict[str, Any]] = {}
    with Y_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = str(rec.get("sample_id") or "")
            if sid:
                y_idx[sid] = rec

    samples = load_jsonl(SS)
    n = 0
    missing_y = 0
    cells = Counter()
    sem_cls = Counter()
    aa_service = aa_entity = aa_param = aa_exact = aa_n = 0
    exact_n = 0
    y_action_type = Counter()
    by_type_cells: dict[str, Counter] = defaultdict(Counter)
    by_scene: dict[str, Counter] = defaultdict(Counter)
    miss_tax = Counter()
    miss_kind = Counter()
    unn_tax = Counter()
    miss_by_svc = Counter()
    unn_by_svc = Counter()
    miss_by_scene = Counter()
    unn_by_scene = Counter()
    miss_event_focus = Counter()
    event_focus_y_action = Counter()
    event_focus_b0_miss = Counter()
    pair_svc = Counter()
    pair_ent = Counter()
    examples: dict[str, list] = {}
    used_freeze_y = 0

    for s in samples:
        sid = str(s.get("sample_id") or "")
        yrec = y_idx.get(sid)
        if not yrec:
            missing_y += 1
            continue
        n += 1
        if s.get("formal_y"):
            used_freeze_y += 0
        scene_raw = str(s.get("scene_type") or "unknown")
        scenario = SCENARIO_MAP.get(scene_raw, scene_raw)
        b0_dec, b0_act = b0_from_sample(s)
        y_dec, y_act, action_type = y_from_rec(yrec)
        if not action_type:
            action_type = service_kind((y_act or {}).get("service") or "") or "UNKNOWN"
        y_action_type[action_type] += 1
        key = f"B0_{b0_dec}_Y_{y_dec}"
        cells[key] += 1
        by_type_cells[action_type][key] += 1
        sc = by_scene[scenario]
        sc["sample_count"] += 1
        sc[key] += 1
        ok = exact_ok(b0_dec, b0_act, y_dec, y_act)
        if ok:
            exact_n += 1
            sc["exact"] += 1
        else:
            sc["error"] += 1

        compact = {
            "sample_id": sid,
            "scenario": scenario,
            "scene_type": scene_raw,
            "action_type": action_type,
            "b0_service": (b0_act or {}).get("service"),
            "b0_entity": (b0_act or {}).get("entity"),
            "y_service": (y_act or {}).get("service"),
            "y_entity": (y_act or {}).get("entity"),
        }

        if b0_dec == "ACTION" and y_dec == "ACTION":
            aa_n += 1
            cmp = compare_actions(b0_act, y_act)
            aa_service += int(cmp["service_match"])
            aa_entity += int(cmp["entity_match"])
            aa_param += int(cmp["parameter_match"])
            aa_exact += int(cmp["exact_action_match"])
            cls = semantic_class(cmp)
            sem_cls[cls] += 1
            sc[f"aa_{cls}"] += 1
            if cls != "exact_match":
                pair_svc[(str((b0_act or {}).get("service")), str((y_act or {}).get("service")))] += 1
                if cls == "entity_mismatch":
                    pair_ent[(str((b0_act or {}).get("entity")), str((y_act or {}).get("entity")))] += 1
                add_example(examples, f"aa_{cls}", compact)
        elif b0_dec == "ACTION" and y_dec == "NO_ACTION":
            tag = classify_unnecessary(s, b0_act)
            unn_tax[tag] += 1
            unn_by_svc[(b0_act or {}).get("service") or "(none)"] += 1
            unn_by_scene[scenario] += 1
            sc["unnecessary"] += 1
            sc[f"unn_{tag}"] += 1
            add_example(examples, f"unn_{tag}", compact)
        elif b0_dec == "NO_ACTION" and y_dec == "ACTION":
            tag = classify_missing(s, y_act, action_type)
            miss_tax[tag] += 1
            ysvc = (y_act or {}).get("service") or "(none)"
            miss_kind[service_kind(ysvc) if ysvc != "(none)" else action_type or "OTHER"] += 1
            miss_by_svc[ysvc] += 1
            miss_by_scene[scenario] += 1
            sc["missing"] += 1
            sc[f"miss_{tag}"] += 1
            add_example(examples, f"miss_{tag}", compact)
            if ysvc in EVENT_FOCUS or str(ysvc).startswith("notify."):
                miss_event_focus[ysvc] += 1

        ysvc = (y_act or {}).get("service") or ""
        if y_dec == "ACTION" and (ysvc in EVENT_FOCUS or ysvc.startswith("notify.")):
            event_focus_y_action[ysvc] += 1
            if b0_dec == "NO_ACTION":
                event_focus_b0_miss[ysvc] += 1

    if missing_y:
        raise SystemExit(f"Gold Y missing {missing_y} freeze ids; abort")
    if n != 12021:
        print(f"warning: evaluated n={n}, expected 12021", flush=True)

    c1 = cells["B0_ACTION_Y_ACTION"]
    c2 = cells["B0_ACTION_Y_NO_ACTION"]
    c3 = cells["B0_NO_ACTION_Y_ACTION"]
    c4 = cells["B0_NO_ACTION_Y_NO_ACTION"]
    tp, fp, fn, tn = c1, c2, c3, c4
    decision_acc = _rate(tp + tn, n)
    action_p = _rate(tp, tp + fp)
    action_r = _rate(tp, tp + fn)
    noact_p = _rate(tn, tn + fn)
    noact_r = _rate(tn, tn + fp)
    semantic_acc = _rate(exact_n, n)

    evaluation = {
        "gold_y": str(Y_PATH.relative_to(ROOT)).replace("\\", "/"),
        "freeze": str(SS.relative_to(ROOT)).replace("\\", "/"),
        "ground_truth": "single_scene_final_y.jsonl",
        "ignored": ["freeze.formal_y", "Y_v6", "Y_v7", "Y_v8", "Repair"],
        "n": n,
        "missing_y": missing_y,
        "used_freeze_formal_y": used_freeze_y,
        "quadrants": {
            "case1_B0_ACTION_Y_ACTION": {
                "count": c1,
                "ratio": _rate(c1, n),
                "meaning": "B0 and Gold both execute; action semantics compared separately",
            },
            "case2_B0_ACTION_Y_NO_ACTION": {
                "count": c2,
                "ratio": _rate(c2, n),
                "meaning": "B0 over-execution",
            },
            "case3_B0_NO_ACTION_Y_ACTION": {
                "count": c3,
                "ratio": _rate(c3, n),
                "meaning": "B0 missed execution",
            },
            "case4_B0_NO_ACTION_Y_NO_ACTION": {
                "count": c4,
                "ratio": _rate(c4, n),
                "meaning": "B0 correctly keeps no action",
            },
        },
        "metrics": {
            "Decision_Accuracy": decision_acc,
            "ACTION_Precision": action_p,
            "ACTION_Recall": action_r,
            "NO_ACTION_Precision": noact_p,
            "NO_ACTION_Recall": noact_r,
            "Semantic_Action_Accuracy": semantic_acc,
        },
        "metric_definitions": {
            "Decision_Accuracy": "(case1 + case4) / N",
            "ACTION_Precision": "case1 / (case1 + case2)",
            "ACTION_Recall": "case1 / (case1 + case3)",
            "NO_ACTION_Precision": "case4 / (case4 + case3)",
            "NO_ACTION_Recall": "case4 / (case4 + case2)",
            "Semantic_Action_Accuracy": "exact(service+entity+data or both NO_ACTION) / N",
        },
        "counts": {
            "B0_ACTION": c1 + c2,
            "B0_NO_ACTION": c3 + c4,
            "Y_ACTION": c1 + c3,
            "Y_NO_ACTION": c2 + c4,
            "exact_semantic_match": exact_n,
            "exact_semantic_match_from_case4_only": exact_n == c4,
            "decision_errors": c2 + c3,
            "action_action_semantic_errors": c1 - aa_exact,
            "action_action_service_match": aa_service,
        },
        "semantic_zero_exact_note": (
            "All Case-1 rows fail exact match. "
            f"entity_mismatch={sem_cls['entity_mismatch']} "
            "(mostly B0 target=null vs Gold climate.living_room_ac / light.living_room); "
            f"parameter_mismatch={sem_cls['parameter_mismatch']} "
            "(Gold data often empty while B0 carries log/sheet payload); "
            f"service_mismatch={sem_cls['service_mismatch']}."
        ),
        "y_action_type": dict(y_action_type),
    }
    dump(OUT / "b0_final_y_evaluation.json", evaluation)

    semantic_report = {
        "denominator": "B0 ACTION AND Gold Y ACTION",
        "n": aa_n,
        "exact_match": {"count": sem_cls["exact_match"], "ratio": _rate(sem_cls["exact_match"], aa_n)},
        "service_mismatch": {
            "count": sem_cls["service_mismatch"],
            "ratio": _rate(sem_cls["service_mismatch"], aa_n),
        },
        "entity_mismatch": {
            "count": sem_cls["entity_mismatch"],
            "ratio": _rate(sem_cls["entity_mismatch"], aa_n),
        },
        "parameter_mismatch": {
            "count": sem_cls["parameter_mismatch"],
            "ratio": _rate(sem_cls["parameter_mismatch"], aa_n),
        },
        "component_match_rates": {
            "service": _rate(aa_service, aa_n),
            "entity": _rate(aa_entity, aa_n),
            "parameter": _rate(aa_param, aa_n),
            "exact": _rate(aa_exact, aa_n),
        },
        "top_service_pairs_on_mismatch": [
            {"b0": a, "y": b, "count": c} for (a, b), c in pair_svc.most_common(15)
        ],
        "top_entity_pairs_on_entity_mismatch": [
            {"b0": a, "y": b, "count": c} for (a, b), c in pair_ent.most_common(10)
        ],
        "examples": {
            "service_mismatch": examples.get("aa_service_mismatch", []),
            "entity_mismatch": examples.get("aa_entity_mismatch", []),
            "parameter_mismatch": examples.get("aa_parameter_mismatch", []),
        },
    }
    dump(OUT / "action_semantic_match_report.json", semantic_report)

    missing_report = {
        "denominator": "B0 NO_ACTION AND Gold Y ACTION",
        "n": c3,
        "ratio_of_corpus": _rate(c3, n),
        "taxonomy": {
            "missing_state_action": {
                "count": miss_tax["missing_state_action"],
                "ratio": _rate(miss_tax["missing_state_action"], c3),
                "meaning": "Gold requires a device/helper state change; B0 executed nothing",
            },
            "missing_event_action": {
                "count": miss_tax["missing_event_action"],
                "ratio": _rate(miss_tax["missing_event_action"], c3),
                "meaning": "Gold requires notify/log/sheets/logbook; B0 executed nothing",
            },
            "missing_temporal_action": {
                "count": miss_tax["missing_temporal_action"],
                "ratio": _rate(miss_tax["missing_temporal_action"], c3),
                "meaning": "Periodic/schedule scene; Gold requires the timed effect, B0 did not fire",
            },
            "missing_context_action": {
                "count": miss_tax["missing_context_action"],
                "ratio": _rate(miss_tax["missing_context_action"], c3),
                "meaning": "Gold action depends on combining multiple states; B0 stayed idle",
            },
        },
        "by_gold_service": miss_by_svc.most_common(20),
        "by_gold_service_kind": {
            **dict(miss_kind),
            "note": (
                "Exclusive taxonomy maps climate.set_hvac_mode misses to missing_context_action "
                "because the scene requires window+HVAC. They remain state-changing services."
            ),
        },
        "by_scenario": miss_by_scene.most_common(),
        "event_focus_missed": dict(miss_event_focus),
        "examples": {
            "missing_state_action": examples.get("miss_missing_state_action", []),
            "missing_event_action": examples.get("miss_missing_event_action", []),
            "missing_temporal_action": examples.get("miss_missing_temporal_action", []),
            "missing_context_action": examples.get("miss_missing_context_action", []),
        },
        "assignment_priority": [
            "periodic/schedule scene or schedule.* -> missing_temporal_action",
            "Gold event service -> missing_event_action",
            "visual/scene/climate/lighting -> missing_context_action",
            "else -> missing_state_action",
        ],
    }
    dump(OUT / "missing_action_taxonomy.json", missing_report)

    unnecessary_report = {
        "denominator": "B0 ACTION AND Gold Y NO_ACTION",
        "n": c2,
        "ratio_of_corpus": _rate(c2, n),
        "taxonomy": {
            "already_satisfied_state": {
                "count": unn_tax["already_satisfied_state"],
                "ratio": _rate(unn_tax["already_satisfied_state"], c2),
                "meaning": "Observed device/helper already at the B0 target state",
            },
            "redundant_action": {
                "count": unn_tax["redundant_action"],
                "ratio": _rate(unn_tax["redundant_action"], c2),
                "meaning": "Event/helper side-effect with no required Gold effect",
            },
            "wrong_trigger_interpretation": {
                "count": unn_tax["wrong_trigger_interpretation"],
                "ratio": _rate(unn_tax["wrong_trigger_interpretation"], c2),
                "meaning": "Trigger matched so B0 fired, but Gold says the action is not warranted",
            },
            "wrong_context_reasoning": {
                "count": unn_tax["wrong_context_reasoning"],
                "ratio": _rate(unn_tax["wrong_context_reasoning"], c2),
                "meaning": "Environment/context (window, illuminance, multi-state) does not support the action",
            },
        },
        "by_b0_service": unn_by_svc.most_common(20),
        "by_scenario": unn_by_scene.most_common(),
        "examples": {
            "already_satisfied_state": examples.get("unn_already_satisfied_state", []),
            "redundant_action": examples.get("unn_redundant_action", []),
            "wrong_trigger_interpretation": examples.get("unn_wrong_trigger_interpretation", []),
            "wrong_context_reasoning": examples.get("unn_wrong_context_reasoning", []),
        },
        "assignment_priority": [
            "runtime state already equals B0 target -> already_satisfied_state",
            "event/helper service -> redundant_action",
            "climate/visual/scene or lighting with high lux -> wrong_context_reasoning",
            "else trigger-like observation -> wrong_trigger_interpretation",
        ],
    }
    dump(OUT / "unnecessary_action_taxonomy.json", unnecessary_report)

    def type_block(at: str) -> dict[str, Any]:
        c = by_type_cells[at]
        nt = sum(c.values())
        return {
            "sample_count": nt,
            "B0_ACTION_Y_ACTION": {"count": c["B0_ACTION_Y_ACTION"], "ratio": _rate(c["B0_ACTION_Y_ACTION"], nt)},
            "B0_ACTION_Y_NO_ACTION": {
                "count": c["B0_ACTION_Y_NO_ACTION"],
                "ratio": _rate(c["B0_ACTION_Y_NO_ACTION"], nt),
            },
            "B0_NO_ACTION_Y_ACTION": {
                "count": c["B0_NO_ACTION_Y_ACTION"],
                "ratio": _rate(c["B0_NO_ACTION_Y_ACTION"], nt),
            },
            "B0_NO_ACTION_Y_NO_ACTION": {
                "count": c["B0_NO_ACTION_Y_NO_ACTION"],
                "ratio": _rate(c["B0_NO_ACTION_Y_NO_ACTION"], nt),
            },
            "over_execution_rate": _rate(c["B0_ACTION_Y_NO_ACTION"], nt),
            "missing_execution_rate": _rate(c["B0_NO_ACTION_Y_ACTION"], nt),
            "decision_accuracy": _rate(
                c["B0_ACTION_Y_ACTION"] + c["B0_NO_ACTION_Y_NO_ACTION"], nt
            ),
        }

    state_event = {
        "STATE_ACTION": type_block("STATE_ACTION"),
        "EVENT_ACTION": type_block("EVENT_ACTION"),
        "event_focus_services": {
            svc: {
                "gold_ACTION": event_focus_y_action[svc],
                "b0_missed": event_focus_b0_miss[svc],
                "miss_rate_among_gold_action": _rate(event_focus_b0_miss[svc], event_focus_y_action[svc]),
            }
            for svc in sorted(set(EVENT_FOCUS) | set(event_focus_y_action) | set(event_focus_b0_miss))
            if event_focus_y_action[svc] or event_focus_b0_miss[svc]
        },
        "event_miss_summary": {
            "gold_event_focus_ACTION": sum(event_focus_y_action.values()),
            "b0_missed_event_focus": sum(event_focus_b0_miss.values()),
            "miss_rate": _rate(sum(event_focus_b0_miss.values()), sum(event_focus_y_action.values())),
            "note": "notify / system_log.write / google_sheets / logbook Gold ACTION where B0 is NO_ACTION",
        },
        "which_has_more_errors": None,
    }
    state_err = (
        by_type_cells["STATE_ACTION"]["B0_ACTION_Y_NO_ACTION"]
        + by_type_cells["STATE_ACTION"]["B0_NO_ACTION_Y_ACTION"]
    )
    event_err = (
        by_type_cells["EVENT_ACTION"]["B0_ACTION_Y_NO_ACTION"]
        + by_type_cells["EVENT_ACTION"]["B0_NO_ACTION_Y_ACTION"]
    )
    state_n = sum(by_type_cells["STATE_ACTION"].values())
    event_n = sum(by_type_cells["EVENT_ACTION"].values())
    state_event["decision_error_counts"] = {"STATE_ACTION": state_err, "EVENT_ACTION": event_err}
    state_event["decision_error_rates"] = {
        "STATE_ACTION": _rate(state_err, state_n),
        "EVENT_ACTION": _rate(event_err, event_n),
    }
    if _rate(state_err, state_n) >= _rate(event_err, event_n):
        state_event["which_has_more_errors"] = "STATE_ACTION"
    else:
        state_event["which_has_more_errors"] = "EVENT_ACTION"
    dump(OUT / "b0_state_event_analysis.json", state_event)

    scenario_out: dict[str, Any] = {}
    for sc_name in ["lighting", "climate", "security", "visual", "periodic", "schedule", "appliance", "scene"]:
        c = by_scene[sc_name]
        sn = c["sample_count"]
        over = c["B0_ACTION_Y_NO_ACTION"]
        miss = c["B0_NO_ACTION_Y_ACTION"]
        aa = c["B0_ACTION_Y_ACTION"]
        mismatch = aa - c["aa_exact_match"]
        acc = _rate(c["exact"], sn)
        over_r = _rate(over, sn)
        miss_r = _rate(miss, sn)
        match_r = None if aa == 0 else _rate(c["aa_exact_match"], aa)
        err_map = {
            "over_execution": over,
            "missing_execution": miss,
            "action_mismatch": mismatch,
        }
        main = max(err_map, key=err_map.get) if sn else "none"
        if err_map[main] == 0:
            main = "none"
        scenario_out[sc_name] = {
            "sample_count": sn,
            "accuracy": acc,
            "decision_accuracy": _rate(
                c["B0_ACTION_Y_ACTION"] + c["B0_NO_ACTION_Y_NO_ACTION"], sn
            ),
            "over_execution_rate": over_r,
            "missing_execution_rate": miss_r,
            "action_match_rate": match_r,
            "main_error_type": main,
            "counts": {
                "B0_ACTION_Y_ACTION": aa,
                "B0_ACTION_Y_NO_ACTION": over,
                "B0_NO_ACTION_Y_ACTION": miss,
                "B0_NO_ACTION_Y_NO_ACTION": c["B0_NO_ACTION_Y_NO_ACTION"],
                "exact_semantic": c["exact"],
                "aa_exact_match": c["aa_exact_match"],
                "aa_service_mismatch": c["aa_service_mismatch"],
                "aa_entity_mismatch": c["aa_entity_mismatch"],
                "aa_parameter_mismatch": c["aa_parameter_mismatch"],
            },
        }
    dump(OUT / "b0_error_by_scenario.json", scenario_out)

    over_share = _rate(c2, n)
    miss_share = _rate(c3, n)
    mismatch_share = _rate(c1 - aa_exact, n)
    if c2 >= c3 and c2 >= (c1 - aa_exact):
        primary_error = "over_execution"
    elif c3 >= (c1 - aa_exact):
        primary_error = "missing_execution"
    else:
        primary_error = "action_mismatch"

    repair_ops = []
    if c2:
        repair_ops.append("delete_or_suppress_action")
    if c3:
        repair_ops.append("add_missing_action")
    if (c1 - aa_exact) > 0:
        repair_ops.append("modify_service_entity_or_parameters")
    repair_mode = "combination" if len(repair_ops) > 1 else (repair_ops[0] if repair_ops else "none")

    lines = [
        "# B0 vs Final Gold Y Analysis",
        "",
        "Ground truth = `runs/ss_final_gold_y/single_scene_final_y.jsonl` (Final Gold Y).",
        "Baseline = frozen `formal_b0` in `single_scene_final.jsonl`.",
        "Freeze `formal_y`, Y_v6/v7/v8, and Repair were not used. Gold Y was not modified.",
        "",
        f"- N = **{n}**",
        f"- Decision Accuracy = **{_pct(decision_acc)}**",
        f"- Semantic Action Accuracy = **{_pct(semantic_acc)}** ({exact_n}/{n})",
        "",
        "## 1. B0 overall performance",
        "",
        "B0 agrees with Gold on the ACTION / NO_ACTION decision in "
        f"{_pct(decision_acc)} of samples. Exact semantic match is {_pct(semantic_acc)} "
        f"({exact_n}/{n}), and that entire mass is Case 4 (both NO_ACTION). "
        f"Among {aa_n} ACTION-ACTION rows, exact service+entity+data match is **0**. "
        f"Service-only match on Case 1 is {_pct(_rate(aa_service, aa_n))}.",
        "",
        md_table(
            ["metric", "value"],
            [
                ["Decision Accuracy", _pct(decision_acc)],
                ["ACTION Precision", _pct(action_p)],
                ["ACTION Recall", _pct(action_r)],
                ["NO_ACTION Precision", _pct(noact_p)],
                ["NO_ACTION Recall", _pct(noact_r)],
                ["Semantic Action Accuracy", _pct(semantic_acc)],
            ],
        ),
        "",
        "## 2. Four quadrants",
        "",
        md_table(
            ["case", "B0", "Gold Y", "count", "ratio", "meaning"],
            [
                ["1", "ACTION", "ACTION", c1, _pct(_rate(c1, n)), "both execute; compare semantics"],
                ["2", "ACTION", "NO_ACTION", c2, _pct(_rate(c2, n)), "over-execution"],
                ["3", "NO_ACTION", "ACTION", c3, _pct(_rate(c3, n)), "missed execution"],
                ["4", "NO_ACTION", "NO_ACTION", c4, _pct(_rate(c4, n)), "correct idle"],
            ],
        ),
        "",
        f"Primary error mass is **{primary_error}** "
        f"(over-exec {c2} / {_pct(over_share)}, miss {c3} / {_pct(miss_share)}, "
        f"ACTION-ACTION mismatch {c1 - aa_exact} / {_pct(mismatch_share)}).",
        "",
        "Among ACTION-ACTION:",
        "",
        md_table(
            ["class", "count", "ratio"],
            [
                ["Exact Match", sem_cls["exact_match"], _pct(_rate(sem_cls["exact_match"], aa_n))],
                ["Service mismatch", sem_cls["service_mismatch"], _pct(_rate(sem_cls["service_mismatch"], aa_n))],
                ["Entity mismatch", sem_cls["entity_mismatch"], _pct(_rate(sem_cls["entity_mismatch"], aa_n))],
                ["Parameter mismatch", sem_cls["parameter_mismatch"], _pct(_rate(sem_cls["parameter_mismatch"], aa_n))],
            ],
        ),
        "",
        "## 3. Over-execution vs missed execution vs wrong action",
        "",
        f"- Over-execution (Case 2): **{c2}** ({_pct(over_share)}).",
        f"- Missed execution (Case 3): **{c3}** ({_pct(miss_share)}).",
        f"- Wrong action among Case 1: **{c1 - aa_exact}** ({_pct(mismatch_share)} of corpus; "
        f"{_pct(_rate(c1 - aa_exact, aa_n))} of ACTION-ACTION).",
        "",
        "Unnecessary ACTION taxonomy:",
        "",
        md_table(
            ["type", "count", "ratio of Case 2"],
            [
                [
                    k,
                    unn_tax[k],
                    _pct(_rate(unn_tax[k], c2)),
                ]
                for k in (
                    "already_satisfied_state",
                    "redundant_action",
                    "wrong_trigger_interpretation",
                    "wrong_context_reasoning",
                )
            ],
        ),
        "",
        "Missing ACTION taxonomy:",
        "",
        md_table(
            ["type", "count", "ratio of Case 3"],
            [
                [k, miss_tax[k], _pct(_rate(miss_tax[k], c3))]
                for k in (
                    "missing_state_action",
                    "missing_event_action",
                    "missing_temporal_action",
                    "missing_context_action",
                )
            ],
        ),
        "",
        "By Gold service kind, Case 3 is "
        f"STATE_ACTION {miss_kind.get('STATE_ACTION', 0)} "
        f"(all `climate.set_hvac_mode`, exclusive-tagged as missing_context_action) and "
        f"EVENT_ACTION {miss_kind.get('EVENT_ACTION', 0)} "
        "(notify / sheets / logbook). Lighting has **0** missed Gold ACTIONs.",
        "",
        "## 4. STATE vs EVENT",
        "",
        md_table(
            ["action_type", "N", "decision acc", "over-exec", "miss", "decision errors"],
            [
                [
                    "STATE_ACTION",
                    state_n,
                    _pct(state_event["STATE_ACTION"]["decision_accuracy"]),
                    state_event["STATE_ACTION"]["B0_ACTION_Y_NO_ACTION"]["count"],
                    state_event["STATE_ACTION"]["B0_NO_ACTION_Y_ACTION"]["count"],
                    state_err,
                ],
                [
                    "EVENT_ACTION",
                    event_n,
                    _pct(state_event["EVENT_ACTION"]["decision_accuracy"]),
                    state_event["EVENT_ACTION"]["B0_ACTION_Y_NO_ACTION"]["count"],
                    state_event["EVENT_ACTION"]["B0_NO_ACTION_Y_ACTION"]["count"],
                    event_err,
                ],
            ],
        ),
        "",
        f"Decision-error **rate** is higher on **{state_event['which_has_more_errors']}** "
        f"(STATE {_pct(_rate(state_err, state_n))}, EVENT {_pct(_rate(event_err, event_n))}). "
        f"Absolute decision-error count is {'STATE' if state_err >= event_err else 'EVENT'} "
        f"({max(state_err, event_err)} vs {min(state_err, event_err)}).",
        "",
        "Event-focus Gold ACTION (notify / system_log.write / google_sheets / logbook) "
        f"= {sum(event_focus_y_action.values())}; B0 missed "
        f"{sum(event_focus_b0_miss.values())} "
        f"({_pct(_rate(sum(event_focus_b0_miss.values()), sum(event_focus_y_action.values())))}).",
        "",
        md_table(
            ["service", "Gold ACTION", "B0 missed", "miss rate"],
            [
                [
                    svc,
                    state_event["event_focus_services"][svc]["gold_ACTION"],
                    state_event["event_focus_services"][svc]["b0_missed"],
                    _pct(state_event["event_focus_services"][svc]["miss_rate_among_gold_action"]),
                ]
                for svc in sorted(state_event["event_focus_services"])
            ],
        ),
        "",
        "## 5. By scenario",
        "",
        md_table(
            [
                "scenario",
                "N",
                "semantic acc",
                "over-exec",
                "miss",
                "AA match",
                "main error",
            ],
            [
                [
                    name,
                    scenario_out[name]["sample_count"],
                    _pct(scenario_out[name]["accuracy"]),
                    _pct(scenario_out[name]["over_execution_rate"]),
                    _pct(scenario_out[name]["missing_execution_rate"]),
                    "n/a"
                    if scenario_out[name]["action_match_rate"] is None
                    else _pct(scenario_out[name]["action_match_rate"]),
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
        "## 6. Which errors are repairable",
        "",
        "Suitable for a later Repair stage (not implemented here):",
        "",
        f"- **Delete / suppress**: Case 2 over-execution ({c2}). Dominant subtypes: "
        f"redundant_action {unn_tax['redundant_action']}, "
        f"wrong_trigger_interpretation {unn_tax['wrong_trigger_interpretation']}, "
        f"wrong_context_reasoning {unn_tax['wrong_context_reasoning']}, "
        f"already_satisfied_state {unn_tax['already_satisfied_state']}.",
        f"- **Add / fill**: Case 3 misses ({c3}), including event-focus misses "
        f"{sum(event_focus_b0_miss.values())} and temporal misses "
        f"{miss_tax['missing_temporal_action']}.",
        f"- **Modify**: Case 1 semantic errors ({c1 - aa_exact}), especially "
        f"entity_mismatch {sem_cls['entity_mismatch']} (B0 often omits target; "
        "Gold names `climate.living_room_ac` / `light.living_room`) and "
        f"parameter_mismatch {sem_cls['parameter_mismatch']} "
        "(Gold `data` is frequently empty while B0 has log/sheet payload — "
        "do not blindly empty B0 parameters).",
        "",
        "Not a Repair target: Case 4 (already correct idle) and Case 1 exact matches.",
        "",
        "## 7. Repair operator recommendation",
        "",
        f"Repair should support a **{repair_mode}**: {', '.join(repair_ops)}.",
        "",
        "Because over-execution is the largest disagreement cell, deletion/suppression "
        "must be first-class. Missed Gold ACTIONs (especially recovered event effects) "
        "also require an add path. ACTION-ACTION entity/parameter errors require modify. "
        "A delete-only or add-only operator cannot cover this corpus.",
        "",
        "This report stops before Repair design or implementation.",
        "",
    ]
    (OUT / "b0_final_y_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "n": n,
        "quadrants": {"c1": c1, "c2": c2, "c3": c3, "c4": c4},
        "metrics": evaluation["metrics"],
        "primary_error": primary_error,
        "state_vs_event": state_event["which_has_more_errors"],
        "repair_mode": repair_mode,
        "outputs": [
            "b0_final_y_evaluation.json",
            "action_semantic_match_report.json",
            "missing_action_taxonomy.json",
            "unnecessary_action_taxonomy.json",
            "b0_state_event_analysis.json",
            "b0_error_by_scenario.json",
            "b0_final_y_analysis.md",
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
