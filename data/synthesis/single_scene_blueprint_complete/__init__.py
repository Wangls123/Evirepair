from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smarthome_mdf.single_scene_blueprint_complete.generator import run_single_scene_synthesis

__all__ = ["run_single_scene_synthesis"]

def __getattr__(name: str) -> Any:
    if name == "run_single_scene_synthesis":
        from smarthome_mdf.single_scene_blueprint_complete.generator import run_single_scene_synthesis

        return run_single_scene_synthesis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
