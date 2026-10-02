from __future__ import annotations

from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.config import SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (
    BASE_MIN_PER_BLUEPRINT,
    SCENE_BUDGETS,
    TRUE_EVIDENCE_LIMITED_BLUEPRINTS,
    generated_blueprints_for_scene,
)

def build_scene_budget() -> dict[str, Any]:
    rows = []
    for scene, budget in SCENE_BUDGETS.items():
        inv = len(SCENE_BLUEPRINT_MAP.get(scene, []))
        gen = len(generated_blueprints_for_scene(scene))
        rows.append(
            {
                "scene": scene,
                "inventory_blueprints": inv,
                "generated_blueprints": gen,
                "true_evidence_limited_blueprints": inv - gen,
                **budget,
            }
        )
    return {"scenes": rows, "target_total": sum(b["target"] for b in SCENE_BUDGETS.values())}

def build_blueprint_budget(complexity_doc: dict[str, Any]) -> dict[str, Any]:
    comp = {r["blueprint_id"]: r for r in complexity_doc.get("blueprints") or []}
    rows: list[dict] = []
    for scene, budget in SCENE_BUDGETS.items():
        gen_bps = generated_blueprints_for_scene(scene)
        if not gen_bps:
            continue
        scene_target = budget["target"]
        base_total = min(scene_target, BASE_MIN_PER_BLUEPRINT * len(gen_bps))
        remaining = max(0, scene_target - base_total)
        weights = {bp: comp.get(bp, {}).get("budget_weight", 1.0 / len(gen_bps)) for bp in gen_bps}
        wsum = sum(weights.values()) or 1.0
        for bp in gen_bps:
            extra = int(round(remaining * (weights[bp] / wsum)))
            target = BASE_MIN_PER_BLUEPRINT + extra
            rows.append(
                {
                    "blueprint_id": bp,
                    "scene": scene,
                    "base_min": BASE_MIN_PER_BLUEPRINT,
                    "allocated_target": target,
                    "complexity_score": comp.get(bp, {}).get("complexity_score", 0),
                    "budget_weight": comp.get(bp, {}).get("budget_weight", 0),
                    "attempt_budget": max(800, target * 5),
                }
            )
        for bp in SCENE_BLUEPRINT_MAP.get(scene, []):
            if bp in TRUE_EVIDENCE_LIMITED_BLUEPRINTS:
                rows.append(
                    {
                        "blueprint_id": bp,
                        "scene": scene,
                        "base_min": 0,
                        "allocated_target": 0,
                        "complexity_score": comp.get(bp, {}).get("complexity_score", 0),
                        "budget_weight": 0,
                        "attempt_budget": 0,
                        "status": "TRUE_EVIDENCE_LIMITED",
                    }
                )
    return {"blueprints": rows}
