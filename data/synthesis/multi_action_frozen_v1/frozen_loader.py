from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.config import (
    EXPECTED_MULTI_ACTION_FROZEN_SHA256,
    FROZEN_MANIFEST,
    FROZEN_SAMPLES,
    MULTI_ACTION_FROZEN_SAMPLES,
)

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_frozen_integrity() -> dict[str, Any]:
    if not FROZEN_SAMPLES.is_file():
        raise FileNotFoundError(f"Missing frozen corpus: {FROZEN_SAMPLES}")
    actual = _sha256(FROZEN_SAMPLES)
    expected = None
    if FROZEN_MANIFEST.is_file():
        manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
        expected = manifest.get("sample_sha256") or (manifest.get("artifacts") or {}).get(
            "single_scene_samples.jsonl"
        )
    if expected and actual != expected:
        raise RuntimeError(f"Frozen sample hash mismatch: expected {expected}, got {actual}")
    return {"sample_sha256": actual, "verified": bool(expected), "path": str(FROZEN_SAMPLES)}

def load_frozen_samples(*, verify: bool = True) -> tuple[list[dict], dict[str, Any]]:
    meta = verify_frozen_integrity() if verify else {"verified": False}
    return load_corpus_from_path(FROZEN_SAMPLES, meta=meta, expected_count=12000)

def load_corpus_from_path(
    path: Path,
    *,
    meta: dict[str, Any] | None = None,
    expected_count: int | None = None,
    scene_filter: set[str] | frozenset[str] | None = None,
    verify_sha: str | None = None,
) -> tuple[list[dict], dict[str, Any]]:

    if not path.is_file():
        raise FileNotFoundError(f"Missing single-scene corpus: {path}")
    out_meta = dict(meta or {})
    actual = _sha256(path)
    if verify_sha and actual != verify_sha:
        raise RuntimeError(f"Corpus SHA256 mismatch: expected {verify_sha}, got {actual}")
    out_meta.setdefault("sample_sha256", actual)
    out_meta["path"] = str(path)

    samples: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if expected_count is not None and len(samples) != expected_count:
        raise RuntimeError(f"Expected {expected_count} samples, got {len(samples)}")
    if scene_filter:
        allowed = set(scene_filter)
        samples = [s for s in samples if s.get("scene_type") in allowed]
        out_meta["scene_filter"] = sorted(allowed)
        out_meta["filtered_sample_count"] = len(samples)
        if not samples:
            raise RuntimeError(f"No samples remain after scene_filter={sorted(allowed)}")
    out_meta["sample_count"] = len(samples)
    return samples, out_meta

def verify_multi_action_frozen_sha(path: Path | None = None) -> str:
    corpus = path or MULTI_ACTION_FROZEN_SAMPLES
    if not corpus.is_file():
        raise FileNotFoundError(f"Missing frozen Multi-action corpus: {corpus}")
    actual = _sha256(corpus)
    if actual != EXPECTED_MULTI_ACTION_FROZEN_SHA256:
        raise RuntimeError(
            f"Frozen Multi-action SHA mismatch: expected {EXPECTED_MULTI_ACTION_FROZEN_SHA256}, got {actual}"
        )
    return actual

def load_multi_action_frozen(*, verify: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {}
    if verify:
        meta["multi_action_frozen_corpus_sha256"] = verify_multi_action_frozen_sha()
    samples: list[dict[str, Any]] = []
    with MULTI_ACTION_FROZEN_SAMPLES.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    meta["sample_count"] = len(samples)
    return samples, meta
