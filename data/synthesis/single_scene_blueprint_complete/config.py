from __future__ import annotations

from pathlib import Path

from smarthome_mdf.paths import DATA_DIR, MDF_ROOT, REPO_ROOT

POLICY_VERSION = "single_scene_blueprint_complete_v1"
SYNTHESIS_METHOD = "blueprint_driven_single_scene_v1"

OUTPUT_DIR = DATA_DIR / "single_scene_blueprint_complete"
LEGACY_SINGLE_SCENE_DIR = DATA_DIR / "samples_v3" / "full"
LEGACY_LABEL = "legacy scene-driven grounded corpus"

BLUEPRINT_COLLECTION_PATH = REPO_ROOT / "dataset" / "advanced_blueprints" / "blueprint_collection.json"
CONTRACTS_PATH = MDF_ROOT / "configs" / "community_blueprint_contracts.json"
RAW_BLUEPRINTS_ROOT = REPO_ROOT / "dataset" / "raw_blueprints"
VISION_BLUEPRINTS_ROOT = REPO_ROOT / "dataset" / "vision_blueprints"

PROJECT_SCENES = [
    "advanced_lighting",
    "appliance_monitoring",
    "climate_window",
    "notification_security",
    "on_off_schedule",
    "periodic_task_scheduling",
    "scene_schedule_override",
    "visual_fusion",
]

SCENE_BLUEPRINT_MAP: dict[str, list[str]] = {
    "advanced_lighting": [
        "lighting_motion_advanced_v22",
        "lighting_sensor_comprehensive",
        "lighting_motion_extended",
        "lighting_motion_sensor_advanced",
    ],
    "climate_window": [
        "climate_window_restore",
        "climate_smarter_thermostat",
        "climate_hvac_auto_adjust",
        "climate_production_grade",
        "presence_better_thermostat",
    ],
    "appliance_monitoring": [
        "appliance_notifications_actions",
        "appliance_power_state_detect",
        "appliance_power_google_sheets",
        "appliance_vibration_sensor",
    ],
    "notification_security": [
        "security_osam_sensor_alert",
        "security_contact_left_open",
        "security_window_dynamic_wait",
        "security_ikea_myggbett",
    ],
    "visual_fusion": [
        "camera_frigate_vision_llm",
        "camera_frigate_intelligent",
    ],
    "scene_schedule_override": [
        "presence_holiday_away_lighting",
        "presence_automation_state",
    ],
    "on_off_schedule": [
        "lighting_motion_maestro_48",
    ],
    "periodic_task_scheduling": [],
}

CANONICAL_YAML_REL: dict[str, str] = {
    "climate_window_restore": "advanced/climate_window_control/window_climate_flexible.yaml",
    "climate_smarter_thermostat": "advanced/climate_window_control/smarter_thermostat.yaml",
    "climate_hvac_auto_adjust": "advanced/climate_window_control/windows_climate_off_foroxon.yaml",
    "climate_production_grade": "advanced/climate_window_control/windowguard_climate_control.yaml",
    "appliance_notifications_actions": "advanced/appliance_power_monitoring/appliance_notifications_actions.yaml",
    "appliance_power_state_detect": "advanced/appliance_power_monitoring/appliance_power_monitor.yaml",
    "lighting_motion_maestro_48": "advanced/advanced_motion_lighting/motion_controlled_lighting_48.yaml",
    "lighting_motion_advanced_v22": "advanced/advanced_motion_lighting/motion_activated_programmable.yaml",
    "lighting_sensor_comprehensive": "advanced/advanced_motion_lighting/sensor_light_comprehensive.yaml",
    "lighting_motion_extended": "advanced/advanced_motion_lighting/motion_light_lux_check.yaml",
    "lighting_motion_sensor_advanced": "advanced/advanced_motion_lighting/motion_aware_lights.yaml",
}

CANONICAL_YAML_ABS: dict[str, Path] = {
    "camera_frigate_vision_llm": VISION_BLUEPRINTS_ROOT / "frigate_llm_notification_v07.yaml",
    "camera_frigate_intelligent": VISION_BLUEPRINTS_ROOT / "sgtbatten_frigate_stable.yaml",
}

PENDING_YAML_REL: dict[str, str] = {
    "appliance_power_google_sheets": "advanced/appliance_power_monitoring/appliance_power_google_sheets.yaml",
    "appliance_vibration_sensor": "advanced/appliance_power_monitoring/appliance_vibration_sensor.yaml",
    "security_osam_sensor_alert": "advanced/door_window_security/osam_open_sensor_alert_manager.yaml",
    "security_contact_left_open": "advanced/door_window_security/contact_sensor_left_open.yaml",
    "security_window_dynamic_wait": "advanced/door_window_security/open_window_dynamic_wait.yaml",
    "security_ikea_myggbett": "advanced/door_window_security/ikea_myggbett_door_window.yaml",
    "presence_holiday_away_lighting": "advanced/presence_mode/holiday_away_lighting.yaml",
    "presence_automation_state": "advanced/presence_mode/automation_state_presence.yaml",
    "presence_better_thermostat": "advanced/presence_mode/better_thermostat_presence_presets.yaml",
}

MIN_BLUEPRINT_SAMPLES = 20
MAX_ATTEMPTS_PER_BLUEPRINT = 80
MAX_TOTAL_ATTEMPTS = 12000
PROGRESS_INTERVAL = 25

FORBIDDEN_SYNTHESIS_FIELDS = (
    "parent_b0",
    "expected_actions",
    "expected_state",
    "ground_truth",
    "conflict",
    "repair",
    "operator",
    "ars",
    "decision_output",
)

ALL_BLUEPRINT_IDS: list[str] = []
for _scene, _bps in SCENE_BLUEPRINT_MAP.items():
    ALL_BLUEPRINT_IDS.extend(_bps)
assert len(ALL_BLUEPRINT_IDS) == 22, f"expected 22 blueprints, got {len(ALL_BLUEPRINT_IDS)}"
assert len(PROJECT_SCENES) == 8
