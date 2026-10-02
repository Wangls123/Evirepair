from __future__ import annotations

from typing import Any

REPAIR_BLUEPRINT_IDS = frozenset(
    {
        "appliance_notifications_actions",
        "appliance_power_google_sheets",
        "appliance_power_state_detect",
        "camera_frigate_intelligent",
        "camera_frigate_vision_llm",
        "climate_smarter_thermostat",
        "climate_window_restore",
        "lighting_motion_advanced_v22",
        "lighting_motion_maestro_48",
        "lighting_sensor_comprehensive",
        "presence_automation_state",
        "presence_better_thermostat",
        "presence_holiday_away_lighting",
        "security_osam_sensor_alert",
    }
)

BLUEPRINT_DEFAULT_INPUTS: dict[str, dict[str, Any]] = {
    "appliance_notifications_actions": {
        "power_entity": "sensor.power_dishwasher",
        "power_sensor": "sensor.power_dishwasher",
        "energy_entity": "sensor.energy_dishwasher",
        "power_consumption_sensor": "sensor.energy_dishwasher",
        "start_threshold": 10,
        "start_appliance_power": 10,
        "finish_threshold": 5,
        "end_appliance_power": 5,
        "elapsed_time_energy_data": "input_text.dishwasher_runtime",
        "start_power_consumption": "input_number.dishwasher_start_power",
        "end_power_consumption": "input_number.dishwasher_end_power",
        "include_power_tracking": "enable_start_end_helpers",
        "include_runtime_tracking": "enable_runtime_tracking",
        "include_cycle_counter": "enable_cycle_counter",
        "cycle_counter_helper": "counter.dishwasher_cycle_count",
        "include_service_reminder": "enable_service_reminder",
        "service_reminder_counter_helper": "counter.dishwasher_service_reminder",
        "service_reminder_cycles": 50,
        "include_time_service_reminder": "enable_time_service_reminder",
        "service_reminder_time_helper": "input_text.dishwasher_service_time",
        "runtime_tracking_helper": "input_text.dishwasher_runtime_tracking",
        "include_run_status": "run_status_enabled",
        "run_status_entity": "input_select.dishwasher_run_status",
        "run_status_start": "Washing",
        "run_status_end": "Finished",
        "on_start": [{"service": "notify.mobile_app_phone", "data": {"message": "started"}}],
        "on_running": [],
        "on_stop": [{"service": "notify.mobile_app_phone", "data": {"message": "finished"}}],
    },
    "lighting_motion_maestro_48": {
        "motion_sensors": ["binary_sensor.motion_living_room"],
        "main_lights": ["light.living_room"],
        "day_lights": ["light.living_room"],
        "night_lights": ["light.living_room"],
        "scene_day": "scene.living_room_day",
        "lux_threshold": 50,
        "no_motion_wait": 120,
    },
    "appliance_power_google_sheets": {
        "power_sensor": "sensor.power_dishwasher",
        "working_power_threshold": 200,
        "idle_power_threshold": 5,
        "extra_power_min": 30,
        "extra_power_max": 70,
        "is_working_boolean": "input_boolean.dishwasher_is_working",
        "working_started_datetime": "input_datetime.dishwasher_working_started",
        "extra_state_boolean": "input_boolean.dishwasher_extra_state",
        "extra_state_started_datetime": "input_datetime.dishwasher_extra_started",
        "enable_extra_state": True,
        "worksheet_name": "dishwasher_power_log",
        "config_entry_id": "google_sheets_entry_1",
        "notify_web_ui": True,
    },
    "presence_holiday_away_lighting": {
        "automation_control": {
            "automation_control": "enable_zone",
            "automation_control_zone": "zone.home",
        },
        "holiday_scene": "scene.away",
        "light_entities": ["light.living_room", "light.hallway"],
        "include_entity": ["light.living_room", "light.hallway"],
        "trigger": {"trigger_selection": "time", "time_value": "19:30:00"},
        "weekday_options": {"weekday_boolean": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]},
    },
    "camera_frigate_vision_llm": {
        "camera": "camera.living_room",
        "notify_device": "notify.mobile_app_phone",
    },
    "camera_frigate_intelligent": {
        "camera": "camera.living_room",
        "notify_device": "notify.mobile_app_phone",
    },
}

def default_inputs_for(blueprint_id: str, scene: str) -> dict[str, Any]:
    if blueprint_id in BLUEPRINT_DEFAULT_INPUTS:
        return dict(BLUEPRINT_DEFAULT_INPUTS[blueprint_id])
    return {}
