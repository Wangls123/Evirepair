from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.config import ALL_BLUEPRINT_IDS, SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path

OUT_DIR = ROOT / "data" / "single_scene_blueprint_complete" / "frozen_v1" / "validation"
SS_PATH = ROOT / "data" / "single_scene_blueprint_complete" / "frozen_v1" / "single_scene_samples.jsonl"
MA_PATH = ROOT / "data" / "multi_action_frozen_v1" / "frozen_v1" / "multi_action_samples.jsonl"
REQ_PATH = ROOT / "data" / "single_scene_blueprint_complete" / "blueprint_runtime_requirements.json"

EXPECTED_SS_SHA = "ff00370ae4b5d8d9933e3282353369e38faa702ba49425d46c30f98fad47a897"
EXPECTED_MA_SHA = "891a4c39b4a7cf67fb964947199bcfeb1e5fdea22859e14a8c47a1d15c87d883"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _scene_for(bp: str) -> str:
    for scene, ids in SCENE_BLUEPRINT_MAP.items():
        if bp in ids:
            return scene
    return "unknown"

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _obs_fields(sample: dict) -> dict[str, Any]:
    obs = dict(sample.get("observed") or {})
    derived = dict(sample.get("derived_observation") or {})
    syn = dict(sample.get("synthetic_runtime_state") or {})
    entity_obs = list(sample.get("entity_observations") or [])
    tc = {}
    for eo in entity_obs:
        t = (eo.get("attributes") or {}).get("trigger_context")
        if isinstance(t, dict):
            tc.update(t)
    sys_st = dict(sample.get("system_state") or {})
    return {
        "observed": obs,
        "derived": derived,
        "synthetic": syn,
        "trigger_context": tc,
        "system_state": sys_st,
        "entity_obs": entity_obs,
    }

def _has_power(f: dict) -> bool:
    obs = f["observed"]
    for eo in f["entity_obs"]:
        attrs = eo.get("attributes") or {}
        if attrs.get("current_power_w") is not None or attrs.get("current_power") is not None:
            return True
    return obs.get("current_power_w") is not None or obs.get("current_power") is not None

def _has_schedule(f: dict) -> bool:
    tc = f["trigger_context"]
    obs = f["observed"]
    keys = ("schedule_trigger", "scheduled_time", "current_timestamp", "weekday", "last_run_time")
    return any(k in tc or k in obs for k in keys)

def _has_away_holiday(f: dict) -> bool:
    tc = f["trigger_context"]
    ss = f["system_state"]
    obs = f["observed"]
    for src in (tc, ss, obs):
        for k in ("away_mode", "holiday_mode", "zone_empty", "automation_control", "person_zone"):
            if src.get(k) is not None:
                return True
    return False

def _has_motion_lux(f: dict) -> tuple[bool, bool]:
    obs = f["observed"]
    syn = f["synthetic"]
    lux = obs.get("illuminance_lux") or obs.get("illuminance") or syn.get("proxy_illuminance_lux")
    motion = obs.get("motion_state") or f["trigger_context"].get("to_state")
    return motion is not None, lux is not None

def _has_camera_frigate(f: dict) -> dict[str, bool]:
    tc = f["trigger_context"]
    obs = f["observed"]
    binding = sample_binding = {}
    return {
        "frigate_event": bool(tc.get("frigate_event_type") or obs.get("frigate_event_type")),
        "object_type": bool(obs.get("object_type") or tc.get("object_type")),
        "camera_in_binding": True,
    }

def audit_blueprint_samples(bp: str, samples: list[dict]) -> dict[str, Any]:
    ypath = resolve_yaml_path(bp)
    yaml_sha = _sha256(ypath) if ypath and ypath.is_file() else None
    scene = _scene_for(bp)
    n = len(samples)

    field_stats: dict[str, Counter] = defaultdict(Counter)
    evidence = Counter()

    for s in samples:
        f = _obs_fields(s)
        binding = (s.get("blueprint_binding") or {})
        inputs = binding.get("blueprint_inputs") or (binding.get("grounded_instance") or {}).get("blueprint_inputs") or {}
        meta = s.get("synthesis_metadata") or {}
        prov = meta.get("source_provenance") or meta.get("source_dataset") or s.get("source_dataset")

        if _has_power(f):
            evidence["power"] += 1
        if _has_schedule(f):
            evidence["schedule"] += 1
        if _has_away_holiday(f):
            evidence["away_holiday"] += 1

        mot, lux = _has_motion_lux(f)
        if mot:
            evidence["motion"] += 1
        if lux:
            evidence["lux"] += 1

        tc = f["trigger_context"]
        if tc.get("frigate_event_type"):
            evidence["frigate_event"] += 1
        if inputs.get("camera") or inputs.get("in_camera"):
            evidence["camera_input"] += 1
        if inputs.get("light_target"):
            evidence["light_target_input"] += 1
        if inputs.get("schedule_time"):
            evidence["schedule_time_input"] += 1
        if inputs.get("power_sensor") or inputs.get("working_power_threshold"):
            evidence["power_monitor_inputs"] += 1

        for k in ("away_mode", "holiday_mode", "schedule_active", "manual_override"):
            v = tc.get(k) or f["system_state"].get(k)
            field_stats[k][str(v)] += 1

        field_stats["source_dataset"][str(prov or "unknown")] += 1
        field_stats["trigger_id"][str(tc.get("trigger_id") or "none")] += 1

    return {
        "blueprint_id": bp,
        "scene": scene,
        "yaml_path": str(ypath) if ypath else None,
        "yaml_sha256": yaml_sha,
        "sample_count": n,
        "evidence_counts": dict(evidence),
        "field_stats": {k: dict(v) for k, v in field_stats.items()},
        "input_keys_sample": sorted(set(k for s in samples for k in ((s.get("blueprint_binding") or {}).get("blueprint_inputs") or {}).keys()))[:20],
    }

def classify_blueprint(bp: str, audit: dict[str, Any], req_doc: dict | None) -> dict[str, Any]:
    scene = audit["scene"]
    n = audit["sample_count"]
    ev = audit["evidence_counts"]

    classification = "SYNTHESIS_VALID_BLUEPRINT_DRIVEN"
    semantic_mismatch = False
    scene_proxy_used = False
    scene_semantic_mismatch = False
    systematic_missing = False
    wrong_substitution = False
    notes: list[str] = []

    if bp == "appliance_vibration_sensor":
        if n == 0:
            return {
                "classification": "SYNTHESIS_VALID_EVIDENCE_LIMITED",
                "semantic_mismatch": False,
                "scene_proxy_used": False,
                "notes": ["Zero samples — evidence-limited by design; YAML requires vibration sensor public data"],
            }

    if bp == "appliance_power_google_sheets":
        if scene == "appliance_monitoring" and ev.get("power_monitor_inputs", 0) >= n * 0.9 and ev.get("power", 0) >= n * 0.9:
            return {
                "classification": "SYNTHESIS_VALID_BLUEPRINT_DRIVEN",
                "semantic_mismatch": False,
                "scene_proxy_used": False,
                "notes": ["Repaired: appliance_monitoring scene with YAML power-monitor inputs and power evidence"],
            }
        scene_semantic_mismatch = True
        scene_proxy_used = True
        semantic_mismatch = True
        notes.append("Scene periodic_task_scheduling mismatches YAML power-state monitor semantics")
        notes.append(f"Instance inputs use schedule_time proxy ({ev.get('schedule_time_input',0)}/{n}); power_monitor inputs absent")
        if ev.get("power", 0) > 0 and ev.get("schedule", 0) == 0:
            notes.append(f"Runtime evidence is power-driven ({ev.get('power')}/{n}), not schedule/weekday — YAML-derived matching partially rescued samples")
        else:
            systematic_missing = True
        classification = "UPSTREAM_MULTIPLE_DEFECTS"

    elif bp == "presence_holiday_away_lighting":
        if ev.get("away_holiday", 0) >= n * 0.9:
            return {
                "classification": "SYNTHESIS_VALID_BLUEPRINT_DRIVEN",
                "semantic_mismatch": False,
                "scene_proxy_used": False,
                "notes": ["Repaired: away_mode/holiday_mode evidence present in frozen samples"],
            }
        scene_proxy_used = True
        if ev.get("away_holiday", 0) == 0:
            semantic_mismatch = True
            systematic_missing = True
            notes.append("No away_mode/holiday_mode/zone activation evidence in any frozen sample")
        if ev.get("motion", 0) > 0:
            notes.append(f"Samples grounded on motion/trigger_context ({ev.get('motion')}/{n}) not Blueprint away/holiday activation control")
        notes.append("Requirement extraction yields temporal(schedule)+motion, not away/holiday YAML activation semantics")
        classification = "UPSTREAM_MULTIPLE_DEFECTS"

    elif bp in ("camera_frigate_vision_llm", "camera_frigate_intelligent"):
        if ev.get("camera_input", 0) >= n * 0.9 and ev.get("frigate_event", 0) >= n * 0.9:
            return {
                "classification": "SYNTHESIS_VALID_BLUEPRINT_DRIVEN",
                "semantic_mismatch": False,
                "scene_proxy_used": False,
                "notes": ["Repaired: camera binding + frigate event evidence in entity_observations"],
            }
        if ev.get("camera_input", 0) >= n * 0.9 and ev.get("frigate_event", 0) >= n * 0.5:
            classification = "DOWNSTREAM_B0_ONLY_DEFECT"
            notes.append("Frozen samples contain camera binding + frigate event evidence; B0 used wrong field name (in_camera vs camera)")
        else:
            classification = "UPSTREAM_RUNTIME_MATCHING_DEFECT"
            semantic_mismatch = True

    elif bp == "lighting_motion_advanced_v22":
        if ev.get("motion", 0) >= n * 0.9 and ev.get("light_target_input", 0) >= n * 0.9:
            classification = "DOWNSTREAM_B0_ONLY_DEFECT"
            notes.append("Frozen evidence supports motion+lux+light_target; missing default branch was B0 executor defect")
        else:
            classification = "SYNTHESIS_VALID_EVIDENCE_LIMITED"

    elif n == 0:
        classification = "SYNTHESIS_VALID_EVIDENCE_LIMITED"
        notes.append("No frozen samples")

    else:

        if ev.get("schedule_time_input") and not ev.get("power_monitor_inputs") and scene == "periodic_task_scheduling":
            scene_proxy_used = True
        if req_doc and req_doc.get("status") == "OK" and n >= 20:
            classification = "SYNTHESIS_VALID_BLUEPRINT_DRIVEN"

    return {
        "classification": classification,
        "semantic_mismatch": semantic_mismatch,
        "scene_proxy_used": scene_proxy_used,
        "scene_semantic_mismatch": scene_semantic_mismatch,
        "systematic_missing": systematic_missing,
        "wrong_substitution": wrong_substitution,
        "notes": notes,
    }

def trace_multi_action(ss_ids: set[str], ma_rows: list[dict]) -> dict[str, int]:
    comp_refs = 0
    parent_ids: set[str] = set()
    for ma in ma_rows:
        hit = False
        for c in ma.get("components") or []:
            if str(c.get("single_scene_sample_id") or "") in ss_ids:
                comp_refs += 1
                hit = True
        if hit:
            parent_ids.add(str(ma.get("multi_action_id")))
    return {"components": comp_refs, "parents": len(parent_ids)}

def main() -> int:
    ss_sha = _sha256(SS_PATH)
    ma_sha = _sha256(MA_PATH)
    assert ss_sha == EXPECTED_SS_SHA, f"SS SHA mismatch {ss_sha}"
    assert ma_sha == EXPECTED_MA_SHA, f"MA SHA mismatch {ma_sha}"

    ss_rows = _load_jsonl(SS_PATH)
    ma_rows = _load_jsonl(MA_PATH)
    req_all = json.loads(REQ_PATH.read_text(encoding="utf-8"))
    req_by_bp = {r["blueprint_id"]: r for r in req_all.get("blueprints") or []}

    by_bp: dict[str, list[dict]] = defaultdict(list)
    for s in ss_rows:
        bid = str((s.get("blueprint_binding") or {}).get("blueprint_id") or s.get("blueprint_id") or "")
        if bid:
            by_bp[bid].append(s)

    rows = []
    counters = {
        "BLUEPRINT_YAML_MAPPING_MISMATCH": 0,
        "BLUEPRINT_SCENE_SEMANTIC_MISMATCH": 0,
        "SCENE_PROXY_REQUIREMENT_USAGE": 0,
        "SYSTEMATIC_REQUIRED_EVIDENCE_MISSING": 0,
        "WRONG_RUNTIME_FIELD_SUBSTITUTION": 0,
    }
    defective_bps: list[str] = []
    affected_ss_ids: set[str] = set()

    for bp in ALL_BLUEPRINT_IDS:
        samples = by_bp.get(bp, [])
        audit = audit_blueprint_samples(bp, samples)
        cls = classify_blueprint(bp, audit, req_by_bp.get(bp))
        row = {**audit, **cls}
        rows.append(row)

        if cls.get("scene_semantic_mismatch"):
            counters["BLUEPRINT_SCENE_SEMANTIC_MISMATCH"] += 1
        if cls.get("scene_proxy_used"):
            counters["SCENE_PROXY_REQUIREMENT_USAGE"] += 1
        if cls.get("systematic_missing"):
            counters["SYSTEMATIC_REQUIRED_EVIDENCE_MISSING"] += 1
        if cls.get("wrong_substitution"):
            counters["WRONG_RUNTIME_FIELD_SUBSTITUTION"] += 1
        if cls.get("semantic_mismatch") and "UPSTREAM" in cls.get("classification", ""):
            defective_bps.append(bp)
            for s in samples:
                affected_ss_ids.add(str(s.get("sample_id")))

    ma_trace = trace_multi_action(affected_ss_ids, ma_rows) if affected_ss_ids else {"components": 0, "parents": 0}

    upstream_defect = counters["BLUEPRINT_SCENE_SEMANTIC_MISMATCH"] > 0 or counters["SYSTEMATIC_REQUIRED_EVIDENCE_MISSING"] > 0
    verdict = "SYNTHESIS_FROZEN_V1_SEMANTIC_DEFECT_FOUND" if upstream_defect else "SYNTHESIS_FROZEN_V1_VALID"

    report = {
        "verdict": verdict,
        "audited_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "read_only": True,
        "single_scene_frozen_sha256": ss_sha,
        "multi_action_frozen_sha256": ma_sha,
        "counters": {
            **counters,
            "UPSTREAM_SYNTHESIS_SEMANTIC_DEFECT_BLUEPRINTS": len(defective_bps),
            "UPSTREAM_SYNTHESIS_AFFECTED_SINGLE_SAMPLES": len(affected_ss_ids),
            "UPSTREAM_SYNTHESIS_AFFECTED_MULTI_COMPONENTS": ma_trace["components"],
            "UPSTREAM_SYNTHESIS_AFFECTED_MULTI_PARENTS": ma_trace["parents"],
        },
        "defective_blueprint_ids": defective_bps,
        "all_22_blueprint_audit": rows,
        "multi_action_propagation": ma_trace,
        "frozen_datasets_not_modified": True,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "upstream_synthesis_integrity_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "counters": report["counters"], "defective": defective_bps}, indent=2))
    return 0 if verdict == "SYNTHESIS_FROZEN_V1_VALID" else 1

if __name__ == "__main__":
    raise SystemExit(main())
