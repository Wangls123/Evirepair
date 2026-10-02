from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.composer import _resolve_shared_runtime
from smarthome_mdf.multi_action_frozen_v1.config import (
    MULTI_ACTION_FROZEN_SAMPLES,
    MULTI_ACTION_FROZEN_V1_DIR,
    OUTPUT_DIR,
    SCHEMA_VERSION,
)
from smarthome_mdf.multi_action_frozen_v1.construction_validator import validate_construction
from smarthome_mdf.multi_action_frozen_v1.entity_policy import bind_entities_independent, bind_entities_shared
from smarthome_mdf.multi_action_frozen_v1.parent_signature import parent_signature
from smarthome_mdf.multi_action_frozen_v1.provenance_policy import component_provenance, parent_provenance
from smarthome_mdf.multi_action_frozen_v1.schema import _action_schema, _component_runtime
from smarthome_mdf.multi_action_frozen_v1.temporal_policy import align_temporal_components
from smarthome_mdf.paths import DATA_DIR
from smarthome_mdf.single_scene_blueprint_complete.blueprint_instance_defaults import REPAIR_BLUEPRINT_IDS
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

SS_REPAIR_DIR = DATA_DIR / "single_scene_blueprint_complete" / "frozen_v1_ss_repair"
SS_SOURCE_DIR = DATA_DIR / "single_scene_blueprint_complete" / "frozen_v1"
DEFAULT_SS_REPAIR_JSONL = SS_REPAIR_DIR / "single_scene_samples.jsonl"
DEFAULT_SS_SOURCE_JSONL = SS_SOURCE_DIR / "single_scene_samples.jsonl"
DEFAULT_MA_SOURCE_JSONL = MULTI_ACTION_FROZEN_SAMPLES
DEFAULT_MA_OUT_DIR = OUTPUT_DIR / "frozen_v1_ss_repair"

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def build_ss_id_remap(
    *,
    source_jsonl: Path = DEFAULT_SS_SOURCE_JSONL,
    repair_jsonl: Path = DEFAULT_SS_REPAIR_JSONL,
) -> dict[str, str]:
    source = read_jsonl(source_jsonl)
    repair = read_jsonl(repair_jsonl)
    if len(source) != len(repair):
        raise RuntimeError(f"SS length mismatch: {len(source)} vs {len(repair)}")
    old_to_new: dict[str, str] = {}
    for old, new in zip(source, repair):
        bp = str((old.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if bp in REPAIR_BLUEPRINT_IDS:
            old_to_new[str(old["sample_id"])] = str(new["sample_id"])
    return old_to_new

def _parent_needs_sync(parent: dict, old_to_new: dict[str, str]) -> bool:
    for comp in parent.get("components") or []:
        sid = str(comp.get("single_scene_sample_id") or "")
        if sid in old_to_new:
            return True
    return False

def _composition_type(component_samples: list[dict]) -> str:
    scenes = {s.get("scene_type") for s in component_samples}
    return "SAME_SCENE" if len(scenes) == 1 else "CROSS_SCENE"

def sync_parent_record(
    parent: dict,
    *,
    ss_index: dict[str, dict],
    old_to_new: dict[str, str],
) -> tuple[dict, bool]:
    if not _parent_needs_sync(parent, old_to_new):
        return parent, False

    shared_meta = parent.get("shared_entities") or {}
    use_shared = bool(shared_meta.get("shared_entity"))
    shared_map: dict[str, str] = dict(shared_meta.get("shared_entity_mapping") or {})
    force_synthetic = parent.get("source_mode") == "CROSS_DATASET_SYNTHETIC"

    zipped: list[tuple[str, dict]] = []
    for comp in parent.get("components") or []:
        cid = str(comp.get("component_id") or "")
        sid = str(comp.get("single_scene_sample_id") or "")
        new_sid = old_to_new.get(sid, sid)
        sample = ss_index.get(new_sid)
        if sample is None:
            raise KeyError(f"Missing repaired SS sample: {new_sid}")
        zipped.append((cid, sample))

    entity_bindings: list[tuple[str, list, dict]] = []
    for cid, sample in zipped:
        ents = list((sample.get("blueprint_binding") or {}).get("entities") or [])
        comp_shared = {e: shared_map[e] for e in ents if e in shared_map} if use_shared else {}
        if comp_shared:
            recs, meta = bind_entities_shared(
                cid,
                ents,
                comp_shared,
                reason=str((parent.get("components") or [{}])[0].get("entity_binding", {}).get("entity_binding_reason") or "SEMANTICALLY_COMPATIBLE_SHARED"),
            )
        else:
            recs, meta = bind_entities_independent(cid, ents)
        entity_bindings.append((cid, recs, meta))

    temporal = align_temporal_components(zipped, force_synthetic=force_synthetic)
    shared_runtime, _ = _resolve_shared_runtime(zipped, shared_map if use_shared else {})

    components: list[dict] = []
    prov_components: list[dict] = []
    runtime_states: list[dict] = []
    blueprint_ids: list[str] = []
    behavior_sigs: list[str] = []
    source_ids: list[str] = []
    shared_entity = False
    shared_map_out: dict[str, str] = {}

    for (cid, sample), (_, records, bind_meta) in zip(zipped, entity_bindings):
        bb = sample.get("blueprint_binding") or {}
        prov = component_provenance(sample, component_id=cid)
        prov_components.append(prov)
        blueprint_ids.append(str(bb.get("blueprint_id") or ""))
        behavior_sigs.append(str(sample.get("canonical_behavior_signature") or ""))
        source_ids.append(f"{prov.get('source_dataset')}:{prov.get('source_observation_id')}")
        if bind_meta.get("shared_entity"):
            shared_entity = True
            shared_map_out.update(bind_meta.get("shared_entity_mapping") or {})

        comp_temp = next(
            (t for t in temporal.get("composition_timeline", {}).get("components", []) if t["component_id"] == cid),
            {},
        )
        components.append(
            {
                "component_id": cid,
                "blueprint_id": bb.get("blueprint_id"),
                "scene": sample.get("scene_type"),
                "single_scene_sample_id": sample.get("sample_id"),
                "grounded_instance": bb.get("automation_instance_id"),
                "behavior_id": (sample.get("synthesis_metadata") or {}).get("stable_action_path_id"),
                "runtime_state": _component_runtime(sample, cid),
                "source_provenance": prov,
                "entity_binding": {
                    "records": [r.to_dict() for r in records],
                    **bind_meta,
                },
                "temporal_alignment": comp_temp,
                "component_action_schema": _action_schema(sample),
            }
        )
        runtime_states.append(_component_runtime(sample, cid))

    alignment_mode = temporal.get("composition_timeline", {}).get("alignment_mode", "UNKNOWN")
    sig = parent_signature(
        blueprint_ids=blueprint_ids,
        behavior_signatures=behavior_sigs,
        source_identities=source_ids,
        shared_entity=shared_entity,
        shared_entity_mapping=shared_map_out,
        temporal_alignment_class=alignment_mode,
        component_count=len(zipped),
    )

    synced = {
        **parent,
        "schema_version": SCHEMA_VERSION,
        "multi_action_id": parent.get("multi_action_id"),
        "component_count": len(components),
        "components": components,
        "parent_runtime_state": {
            "component_runtime_states": runtime_states,
            "shared_runtime_projection": shared_runtime or {},
        },
        "shared_entities": {
            "shared_entity": shared_entity,
            "shared_entity_mapping": shared_map_out,
        },
        "composition_timeline": temporal.get("composition_timeline"),
        "composition_type": _composition_type([s for _, s in zipped]),
        "source_mode": temporal.get("source_mode"),
        "parent_provenance": parent_provenance(prov_components),
        "canonical_parent_signature": sig,
        "synthesis_metadata": {
            **dict(parent.get("synthesis_metadata") or {}),
            "construction_only": True,
            "ss_repair_sync": "partial_ma_sync_v1",
            "synced_at": _utc_now(),
        },
    }
    return synced, True

def run_partial_ma_sync(
    *,
    ma_source_jsonl: Path | None = None,
    ss_repair_jsonl: Path | None = None,
    ss_source_jsonl: Path | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    ma_source_jsonl = ma_source_jsonl or DEFAULT_MA_SOURCE_JSONL
    ss_repair_jsonl = ss_repair_jsonl or DEFAULT_SS_REPAIR_JSONL
    ss_source_jsonl = ss_source_jsonl or DEFAULT_SS_SOURCE_JSONL
    out_dir = out_dir or DEFAULT_MA_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ss_repair_jsonl.is_file():
        raise FileNotFoundError(f"Missing repaired SS corpus: {ss_repair_jsonl}")

    old_to_new = build_ss_id_remap(source_jsonl=ss_source_jsonl, repair_jsonl=ss_repair_jsonl)
    ss_repair_rows = read_jsonl(ss_repair_jsonl)
    ss_index = {str(s["sample_id"]): s for s in ss_repair_rows}

    ma_source_sha = _sha256(ma_source_jsonl)
    ma_rows = read_jsonl(ma_source_jsonl)

    synced_rows: list[dict] = []
    sync_count = 0
    reject_reasons: Counter[str] = Counter()
    seen_signatures: set[str] = set()
    seen_component_tuples: set[frozenset[str]] = set()

    for parent in ma_rows:
        updated, changed = sync_parent_record(parent, ss_index=ss_index, old_to_new=old_to_new)
        if changed:
            sync_count += 1
        val = validate_construction(
            updated,
            frozen_index=ss_index,
            seen_signatures=seen_signatures,
            seen_component_tuples=seen_component_tuples,
        )
        if val["verdict"] != "CONSTRUCTION_ACCEPT":
            for reason in val.get("reasons") or []:
                reject_reasons[reason.split(":")[0]] += 1
            raise RuntimeError(f"Construction reject for {updated.get('multi_action_id')}: {val.get('reasons')}")
        updated["construction_validator_result"] = val
        synced_rows.append(updated)

    out_jsonl = out_dir / "multi_action_samples.jsonl"
    _write_jsonl(out_jsonl, synced_rows)
    out_sha = _sha256(out_jsonl)

    validation_dir = out_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "verdict": "MA_PARTIAL_SYNC_COMPLETE",
        "synced_at": _utc_now(),
        "source_ma_jsonl": str(ma_source_jsonl),
        "source_ma_sha256": ma_source_sha,
        "upstream_ss_repair_jsonl": str(ss_repair_jsonl),
        "upstream_ss_repair_sha256": _sha256(ss_repair_jsonl),
        "output_ma_jsonl": str(out_jsonl),
        "output_ma_sha256": out_sha,
        "total_parents": len(synced_rows),
        "synced_parents": sync_count,
        "unchanged_parents": len(synced_rows) - sync_count,
        "ss_id_remap_count": len(old_to_new),
        "repair_blueprints": sorted(REPAIR_BLUEPRINT_IDS),
        "reject_reasons": dict(reject_reasons),
    }
    report_path = validation_dir / "partial_ma_sync_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    upstream_dir = out_dir / "upstream"
    upstream_dir.mkdir(parents=True, exist_ok=True)
    upstream_manifest = {
        "single_scene_samples.jsonl": report["upstream_ss_repair_sha256"],
        "repair_source": str(ss_repair_jsonl),
        "original_frozen_v1_sha256": _sha256(ss_source_jsonl),
    }
    (upstream_dir / "single_scene_FREEZE_SHA256_MANIFEST.json").write_text(
        json.dumps(upstream_manifest, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print(json.dumps({"verdict": report["verdict"], "synced_parents": sync_count, "output_sha256": out_sha}, indent=2), flush=True)
    return report
