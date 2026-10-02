from __future__ import annotations

from pathlib import Path

from smarthome_mdf.multi_action_vnext.source_adapters import SourcePool
from smarthome_mdf.paths import DATA_DIR, SAMPLES_V3_DIR
from smarthome_mdf.single_scene_blueprint_complete.public_observation_pool import merge_public_into_pool

def build_legacy_source_pool(scenes: list[str], *, include_public: bool = True) -> SourcePool:

    pool = SourcePool(full_dir=SAMPLES_V3_DIR / "full")
    pool.load(scenes)
    fallbacks = [
        SAMPLES_V3_DIR / "valid_labeled",
        DATA_DIR / "samples",
    ]
    for scene in scenes:
        if pool._pools.get(scene):
            continue
        for base in fallbacks:
            path = base / f"{scene}.jsonl"
            if path.is_file():
                from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

                pool._pools[scene] = read_jsonl(path)
                break

    for scene in scenes:
        path = SAMPLES_V3_DIR / "full" / f"{scene}.jsonl"
        if path.is_file():
            from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

            all_rows = read_jsonl(path)
            if len(all_rows) > len(pool._pools.get(scene, [])):
                pool._pools[scene] = all_rows
    if include_public:
        merge_public_into_pool(pool, scenes)
    return pool
