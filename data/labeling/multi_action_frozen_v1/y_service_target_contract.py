from __future__ import annotations

TARGET_REQUIRED_EXACT: frozenset[str] = frozenset(
    {
        "light.turn_on",
        "light.turn_off",
        "light.toggle",
        "switch.turn_on",
        "switch.turn_off",
        "switch.toggle",
        "climate.turn_on",
        "climate.turn_off",
        "climate.set_hvac_mode",
        "climate.set_preset_mode",
        "climate.set_temperature",
        "scene.turn_on",
        "input_number.set_value",
        "input_select.select_option",
        "input_boolean.turn_on",
        "input_boolean.turn_off",
        "input_text.set_value",
        "counter.increment",
        "counter.decrement",
        "counter.reset",
        "automation.turn_on",
        "automation.turn_off",
        "script.turn_on",
        "homeassistant.turn_on",
        "homeassistant.turn_off",
        "timer.start",
        "timer.cancel",
        "number.set_value",
    }
)

TARGET_REQUIRED_PREFIXES: tuple[str, ...] = (
    "light.",
    "switch.",
    "climate.",
    "input_number.",
    "input_select.",
    "input_boolean.",
    "input_text.",
    "counter.",
    "automation.",
    "script.",
    "timer.",
    "number.",
)

TARGETLESS_EXACT: frozenset[str] = frozenset(
    {
        "notify.mobile_app",
        "notify.notify",
        "notify.persistent_notification",
        "notify.appliance_started",
        "notify.appliance_finished",
        "notify.close_window",
        "logbook.log",
        "persistent_notification.create",
        "persistent_notification.dismiss",
        "google_sheets.append_sheet",
        "system_log.write",
        "llmvision.analyze_image",
        "llmvision.video_analyzer",
        "telegram_bot.send_photo",
        "telegram_bot.send_video",
        "downloader.download_file",
        "scene.create",
        "state.snapshot",
        "state.restore",
        "schedule.activate",
        "valid.trigger",
        "homeassistant.update_entity",
    }
)

TARGETLESS_PREFIXES: tuple[str, ...] = (
    "notify.",
    "logbook.",
    "persistent_notification.",
    "google_sheets.",
    "system_log.",
    "telegram_bot.",
    "downloader.",
    "llmvision.",
)

SERVICE_ALIASES: dict[str, str] = {
    "notify.mobile_app_phone": "notify.mobile_app",
    "notify.mobile_app_notify": "notify.mobile_app",
}

def canonicalize_service(service: str) -> str:

    svc = str(service or "").strip()
    if not svc:
        return svc
    if svc in SERVICE_ALIASES:
        return SERVICE_ALIASES[svc]
    if svc.startswith("notify.mobile_app_") and svc != "notify.mobile_app":
        return "notify.mobile_app"
    return svc

def service_requires_target(service: str) -> bool:

    if not service or "." not in service:
        return False
    svc = service.strip()
    if svc in TARGETLESS_EXACT:
        return False
    if svc in TARGET_REQUIRED_EXACT:
        return True
    for prefix in TARGETLESS_PREFIXES:
        if svc.startswith(prefix):
            return False
    for prefix in TARGET_REQUIRED_PREFIXES:
        if svc.startswith(prefix):
            return True

    if svc.startswith("scene.") and svc != "scene.create":
        return True
    return False

def classify_action_role(service: str) -> str:

    if not service:
        return "business_device_action"
    if service.startswith(("input_number.", "input_select.", "input_boolean.", "input_text.", "counter.")):
        return "automation_helper_action"
    if service.startswith(("notify.", "persistent_notification.", "telegram_bot.")):
        return "notification_side_effect"
    if service.startswith(("logbook.", "system_log.", "google_sheets.")):
        return "logging_side_effect"
    return "business_device_action"
