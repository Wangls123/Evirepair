from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.config import SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.blueprint_instance_defaults import default_inputs_for
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import yaml_exists

def _default_inputs(scene: str, blueprint_id: str) -> dict[str, Any]:
    bp_inputs = default_inputs_for(blueprint_id, scene)
    if bp_inputs:
        return bp_inputs
    if scene == "climate_window":
        return {
            "window_sensor": "binary_sensor.window_living_room",
            "climate_entity": "climate.living_room_ac",
            "open_delay_sec": 30,
            "close_delay_sec": 60,
            "actions_on_open": [{"service": "climate.set_hvac_mode", "target": {"entity_id": "climate.living_room_ac"}, "data": {"hvac_mode": "off"}}],
            "actions_on_close": [{"service": "climate.set_hvac_mode", "target": {"entity_id": "climate.living_room_ac"}, "data": {"hvac_mode": "heat_cool"}}],
        }
    if scene == "appliance_monitoring":
        return {
            "power_entity": "sensor.power_dishwasher",
            "energy_entity": "sensor.energy_dishwasher",
            "start_threshold": 10,
            "finish_threshold": 5,
            "elapsed_time_energy_data": "input_text.dishwasher_runtime",
            "on_start": [{"service": "notify.mobile_app_phone", "data": {"message": "started"}}],
            "on_running": [],
            "on_stop": [{"service": "notify.mobile_app_phone", "data": {"message": "finished"}}],
        }
    if scene == "advanced_lighting":
        return {
            "motion_entity": "binary_sensor.motion_living_room",
            "lightsensor_entity": "sensor.illuminance_living_room",
            "illuminace_level": 50,
            "light_target": {"entity_id": "light.living_room"},
            "no_motion_wait": 120,
        }
    if scene == "notification_security":
        return {
            "start_trigger_settings": [{"entity_id": "binary_sensor.door_front", "from": "off", "to": "on"}],
            "start_notify_settings": [{"service": "notify.mobile_app_phone", "data": {"message": "door opened"}}],
            "start_auto_action_settings": [],
        }
    if scene == "visual_fusion":
        return {
            "camera": "camera.living_room",
            "notify_device": "notify.mobile_app_phone",
        }
    if scene == "on_off_schedule":
        return {"entity": "light.living_room", "on_time": "07:00:00", "off_time": "22:00:00"}
    if scene == "periodic_task_scheduling":
        return {"schedule_time": "08:00:00", "action": [{"service": "script.water_plants"}]}
    if scene == "scene_schedule_override":
        return {"schedule_entity": "input_boolean.schedule_active", "override_entity": "input_boolean.manual_override"}
    return {}

def build_instance_templates() -> dict[str, Any]:
    reg = build_registry()
    instances: dict[str, list[dict]] = {}
    for scene, bp_ids in SCENE_BLUEPRINT_MAP.items():
        for bp_id in bp_ids:
            iid = f"{bp_id}__inst_default"
            inputs = _default_inputs(scene, bp_id)
            instances[bp_id] = [
                {
                    "automation_instance_id": iid,
                    "blueprint_id": bp_id,
                    "scene": scene,
                    "description": f"default instance for {bp_id}",
                    "blueprint_inputs": inputs,
                    "bound_entities": sorted(
                        {
                            v
                            for v in inputs.values()
                            if isinstance(v, str) and "." in v
                        }
                    ),
                    "yaml_available": yaml_exists(bp_id),
                }
            ]

            if yaml_exists(bp_id) and scene in ("climate_window", "advanced_lighting", "appliance_monitoring"):
                alt = dict(inputs)
                if "open_delay_sec" in alt:
                    alt["open_delay_sec"] = 45
                if "illuminace_level" in alt:
                    alt["illuminace_level"] = 80
                if "start_threshold" in alt:
                    alt["start_threshold"] = 15
                instances[bp_id].append(
                    {
                        "automation_instance_id": f"{bp_id}__inst_alt",
                        "blueprint_id": bp_id,
                        "scene": scene,
                        "description": f"alternate instance for {bp_id}",
                        "blueprint_inputs": alt,
                        "bound_entities": instances[bp_id][0]["bound_entities"],
                        "yaml_available": True,
                    }
                )
    return {
        "version": "22_blueprint_instances_v1",
        "total_blueprints": len(instances),
        "instances": instances,
    }

def write_instance_templates(out_dir: Path | None = None) -> Path:
    from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR

    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    doc = build_instance_templates()
    path = out / "automation_instance_templates.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
