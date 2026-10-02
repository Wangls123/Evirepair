from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from smarthome_mdf.multi_action_vnext.source_adapters import SourcePool
from smarthome_mdf.paths import DATA_DIR, SAMPLES_V3_DIR
from smarthome_mdf.single_scene_blueprint_complete.config import ALL_BLUEPRINT_IDS, PROJECT_SCENES
from smarthome_mdf.single_scene_blueprint_complete.grounded_instances import (
    GROUNDED_INSTANCE_GENERATOR_VERSION,
    GroundedInstanceCandidate,
    enumerate_grounded_instances,
    instances_to_templates,
)
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import build_instance_templates
from smarthome_mdf.single_scene_blueprint_complete.public_observation_pool import (
    CANONICAL_PATH,
    EVIDENCE_WINDOWS_PATH,
)
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path, yaml_hash

CACHE_MANIFEST_VERSION = "grounded_instance_cache_manifest_v1"
BEHAVIOR_SPEC_EXTRACTOR_VERSION = "behavior_specs_requirement_extractor_v1"
SOURCE_ADAPTER_VERSION = "source_adapters_semantic_capabilities_v1"
INSTANCE_CONFIG_SCHEMA_VERSION = "22_blueprint_instances_v1"

GroundedCacheStatus = Literal["HIT", "MISS", "INVALID"]

CANDIDATES_FILENAME = "grounded_instance_candidates.json"
MANIFEST_FILENAME = "grounded_instance_cache_manifest.json"

def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return f"missing:{path.name}"
    stat = path.stat()
    return hashlib.sha256(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]

def compute_blueprint_yaml_hashes() -> dict[str, str]:
    return {bp_id: yaml_hash(resolve_yaml_path(bp_id) or Path("missing")) for bp_id in ALL_BLUEPRINT_IDS}

def compute_source_index_fingerprint(scenes: list[str] | None = None) -> str:
    scenes = scenes or PROJECT_SCENES
    parts: list[str] = []
    for scene in scenes:
        parts.append(_file_fingerprint(SAMPLES_V3_DIR / "full" / f"{scene}.jsonl"))
        parts.append(_file_fingerprint(SAMPLES_V3_DIR / "valid_labeled" / f"{scene}.jsonl"))
        parts.append(_file_fingerprint(DATA_DIR / "samples" / f"{scene}.jsonl"))
    parts.append(_file_fingerprint(CANONICAL_PATH))
    parts.append(_file_fingerprint(EVIDENCE_WINDOWS_PATH))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

def compute_cache_dependencies(*, scenes: list[str] | None = None) -> dict[str, Any]:
    inst_meta = build_instance_templates()
    return {
        "cache_manifest_version": CACHE_MANIFEST_VERSION,
        "blueprint_yaml_hashes": compute_blueprint_yaml_hashes(),
        "behavior_spec_extractor_version": BEHAVIOR_SPEC_EXTRACTOR_VERSION,
        "source_adapter_version": SOURCE_ADAPTER_VERSION,
        "grounded_instance_generator_version": GROUNDED_INSTANCE_GENERATOR_VERSION,
        "source_index_fingerprint": compute_source_index_fingerprint(scenes),
        "instance_config_schema_version": inst_meta.get("version") or INSTANCE_CONFIG_SCHEMA_VERSION,
    }

def _candidate_count(raw: dict[str, Any]) -> int:
    return sum(len(v) for v in raw.values() if isinstance(v, list))

def _load_candidates_raw(path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("grounded_instance_candidates.json must be a blueprint->list mapping")
    return raw

def candidates_to_grounded(raw: dict[str, list[dict[str, Any]]]) -> dict[str, list[GroundedInstanceCandidate]]:
    return {
        bp: [
            GroundedInstanceCandidate(
                automation_instance_id=c["automation_instance_id"],
                blueprint_id=c["blueprint_id"],
                scene=c["scene"],
                input_bindings=c.get("input_bindings") or {},
                entity_bindings=c.get("entity_bindings") or [],
                parameter_bindings=c.get("parameter_bindings") or {},
                configuration_signature=c.get("configuration_signature") or "",
                display_name=c.get("display_name") or "",
                matcher_viable=bool(c.get("matcher_viable", True)),
                supporting_source_count=int(c.get("supporting_source_count") or 0),
            )
            for c in cands
        ]
        for bp, cands in raw.items()
    }

def load_grounded_from_candidates(path: Path) -> dict[str, list[GroundedInstanceCandidate]]:
    return candidates_to_grounded(_load_candidates_raw(path))

def save_grounded_cache(
    out_dir: Path,
    grounded: dict[str, list[GroundedInstanceCandidate]],
    *,
    deps: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    deps = deps or compute_cache_dependencies()
    candidates_path = out_dir / CANDIDATES_FILENAME
    candidates_path.write_text(
        json.dumps({k: [c.__dict__ for c in v] for k, v in grounded.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "cache_manifest_version": CACHE_MANIFEST_VERSION,
        "dependencies": deps,
        "grounded_instance_count": sum(len(v) for v in grounded.values()),
        "blueprint_count": len(grounded),
        "candidates_path": CANDIDATES_FILENAME,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

def validate_cache_manifest(manifest: dict[str, Any], current_deps: dict[str, Any]) -> tuple[bool, str]:
    if manifest.get("cache_manifest_version") != CACHE_MANIFEST_VERSION:
        return False, "cache_manifest_version mismatch"
    stored = manifest.get("dependencies") or {}
    for key in (
        "behavior_spec_extractor_version",
        "source_adapter_version",
        "grounded_instance_generator_version",
        "source_index_fingerprint",
        "instance_config_schema_version",
    ):
        if stored.get(key) != current_deps.get(key):
            return False, f"{key} mismatch"
    stored_hashes = stored.get("blueprint_yaml_hashes") or {}
    current_hashes = current_deps.get("blueprint_yaml_hashes") or {}
    if stored_hashes != current_hashes:
        return False, "blueprint_yaml_hashes mismatch"
    return True, "ok"

@dataclass
class GroundedCacheResolution:
    status: GroundedCacheStatus
    grounded: dict[str, list[GroundedInstanceCandidate]] | None = None
    instance_templates: dict[str, list[dict]] | None = None
    cache_exists: bool = False
    cache_valid: bool = False
    resume_requested: bool = False
    enumerated: bool = False
    load_seconds: float = 0.0
    invalidate_reason: str = ""

def resolve_grounded_instances(
    out_dir: Path,
    pool: SourcePool,
    reg: dict,
    *,
    clean_run: bool,
    resume_requested: bool,
    verbose: bool = True,
) -> GroundedCacheResolution:

    if not clean_run:
        templates = build_instance_templates().get("instances") or {}
        return GroundedCacheResolution(
            status="MISS",
            instance_templates=templates,
            cache_exists=False,
            cache_valid=False,
            resume_requested=resume_requested,
        )

    t0 = time.perf_counter()
    candidates_path = out_dir / CANDIDATES_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME
    cache_exists = candidates_path.is_file()
    current_deps = compute_cache_dependencies()
    cache_valid = False
    invalidate_reason = ""

    if cache_exists and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache_valid, invalidate_reason = validate_cache_manifest(manifest, current_deps)
    elif cache_exists:

        try:
            raw = _load_candidates_raw(candidates_path)
            if _candidate_count(raw) > 0:
                save_grounded_cache(out_dir, candidates_to_grounded(raw), deps=current_deps)
                cache_valid = True
                invalidate_reason = "manifest bootstrapped from legacy cache"
            else:
                invalidate_reason = "empty candidates file"
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            invalidate_reason = f"corrupt candidates: {exc}"

    if cache_exists and cache_valid:
        grounded = load_grounded_from_candidates(candidates_path)
        status: GroundedCacheStatus = "HIT"
        enumerated = False
        if verbose:
            print(f"Grounded instance cache: HIT ({invalidate_reason or 'manifest valid'})", flush=True)
    elif cache_exists and not cache_valid:
        status = "INVALID"
        if verbose:
            print(f"Grounded instance cache: INVALID ({invalidate_reason})", flush=True)
            print("Enumerating grounded instances (cache invalidated)...", flush=True)
        grounded = enumerate_grounded_instances(pool, reg=reg)
        save_grounded_cache(out_dir, grounded, deps=current_deps)
        enumerated = True
    else:
        status = "MISS"
        if verbose:
            print("Grounded instance cache: MISS", flush=True)
            print("Enumerating grounded instances (may take 20-40 min)...", flush=True)
        grounded = enumerate_grounded_instances(pool, reg=reg)
        save_grounded_cache(out_dir, grounded, deps=current_deps)
        enumerated = True

    load_seconds = time.perf_counter() - t0
    inst_idx = instances_to_templates(grounded)
    return GroundedCacheResolution(
        status=status,
        grounded=grounded,
        instance_templates=inst_idx,
        cache_exists=cache_exists,
        cache_valid=cache_valid or (status == "HIT"),
        resume_requested=resume_requested,
        enumerated=enumerated,
        load_seconds=load_seconds,
        invalidate_reason=invalidate_reason,
    )
