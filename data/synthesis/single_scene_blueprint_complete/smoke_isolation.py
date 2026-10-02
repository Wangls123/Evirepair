from __future__ import annotations

import shutil
from pathlib import Path

from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR

def isolate_smoke_samples(out_dir: Path | None = None) -> Path | None:
    out = out_dir or OUTPUT_DIR
    formal = out / "single_scene_samples.jsonl"
    if not formal.is_file():
        return None
    smoke_dir = out / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    dest = smoke_dir / "single_scene_samples_SMOKE_TEST.jsonl"
    shutil.move(str(formal), str(dest))
    note = smoke_dir / "README.txt"
    note.write_text(
        "Samples moved here are SMOKE_TEST only.\n"
        "Generated before 22/22 YAML + Parser gate passed.\n"
        "Do NOT use as formal 22-blueprint-complete corpus.\n",
        encoding="utf-8",
    )
    return dest
