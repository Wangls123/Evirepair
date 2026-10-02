from __future__ import annotations

from typing import Any

OPEN_TRUE = {"open", "on", "true", "1", "detected", "home", "occupied"}

AUTOTAP_CHANNELS = {
    "thermostat.ac",
    "window_liv.open",
    "hue_light.power",
    "camera.motion",
    "location.appear",
    "door.lock",
}

def _flag(obs: dict[str, Any], *keys: str) -> bool | None:
    for k in keys:
        if k not in obs:
            continue
        s = str(obs.get(k)).strip().lower()
        if s in OPEN_TRUE:
            return True
        if s in {"closed", "off", "false", "0", "clear", "away", "idle"}:
            return False
    return None

def compile_ss_property(ir: dict[str, Any]) -> dict[str, Any]:
    if ir.get("unsupported_input"):
        return {
            "applicable": False,
            "status": "UNSUPPORTED_CAPABILITY",
            "properties": [],
            "reason": "visual_fusion_or_llmvision_not_in_autotap_evaluation_template",
        }
    obs = ir.get("observations") or {}
    caps = set(ir.get("capabilities") or [])
    scene = str(ir.get("scene_type") or "")
    window_open = _flag(obs, "window_state")
    motion = _flag(obs, "motion_state", "occupancy")

    props: list[dict[str, Any]] = []

    if "ClimateControl" in caps and window_open is not None:
        props.append(
            {
                "template": "never_ac_on_and_window_open",
                "ltl": "G(!(window_liv.open=true & thermostat.ac=true))",
                "init_value_dict": {
                    "window_liv.open": bool(window_open),
                },
                "channel_map": {
                    "window_liv.open": "window_state",
                    "thermostat.ac": "climate.on_or_not_off",
                },
                "source": "official_autotap_evaluation_template_plus_explicit_window_and_climate",
            }
        )

    if "Lighting" in caps and motion is not None:
        props.append(
            {
                "template": "motion_light_safety",
                "ltl": "G(!(camera.motion=true) | hue_light.power=true)",
                "init_value_dict": {"camera.motion": bool(motion)},
                "channel_map": {
                    "camera.motion": "motion_state",
                    "hue_light.power": "light_state",
                },
                "source": "official_autotap_evaluation_template_plus_explicit_motion_and_lighting",
            }
        )

    if not props:
        return {
            "applicable": False,
            "status": "PROPERTY_UNAVAILABLE",
            "properties": [],
            "reason": (
                "no_unique_mapping_onto_autotap_Evaluation_device_list "
                f"scene={scene} caps={sorted(caps)}"
            ),
        }

    return {
        "applicable": True,
        "status": "APPLICABLE",
        "properties": props,
        "reason": "mapped_onto_official_autotap_evaluation_channels",
        "property_count": len(props),
        "allowed_channels": sorted(AUTOTAP_CHANNELS),
    }
