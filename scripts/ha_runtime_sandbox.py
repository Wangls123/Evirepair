from __future__ import annotations

import copy
import time
from typing import Any

import voluptuous as vol

HVAC_MODES = ("off", "heat", "cool", "heat_cool", "auto", "dry", "fan_only")
LOG_LEVELS = ("debug", "info", "warning", "error", "critical")

def _schema(required: dict, optional: dict, extra=vol.PREVENT_EXTRA) -> vol.Schema:
    spec: dict[Any, Any] = {}
    spec.update(required)
    spec.update(optional)
    return vol.Schema(spec, extra=extra)

SERVICE_REGISTRY: dict[str, dict[str, Any]] = {
    "light.turn_on": {
        "domain": "light",
        "target_domains": {"light"},
        "target_required": True,
        "effect": "state",
        "schema": _schema({}, {vol.Optional("brightness"): vol.All(int, vol.Range(min=0, max=255)), vol.Optional("transition"): vol.Coerce(float), vol.Optional("kelvin"): vol.Coerce(int), vol.Optional("color_temp"): vol.Coerce(int), vol.Optional("rgb_color"): list, vol.Optional("brightness_pct"): vol.All(vol.Coerce(float), vol.Range(min=0, max=100))}, extra=vol.ALLOW_EXTRA),
    },
    "light.turn_off": {
        "domain": "light",
        "target_domains": {"light"},
        "target_required": True,
        "effect": "state",
        "schema": _schema({}, {vol.Optional("transition"): vol.Coerce(float)}, extra=vol.ALLOW_EXTRA),
    },
    "switch.turn_on": {"domain": "switch", "target_domains": {"switch"}, "target_required": True, "effect": "state", "schema": _schema({}, {}, extra=vol.ALLOW_EXTRA)},
    "switch.turn_off": {"domain": "switch", "target_domains": {"switch"}, "target_required": True, "effect": "state", "schema": _schema({}, {}, extra=vol.ALLOW_EXTRA)},
    "climate.set_hvac_mode": {
        "domain": "climate",
        "target_domains": {"climate"},
        "target_required": True,
        "effect": "state",
        "schema": _schema({vol.Required("hvac_mode"): vol.In(HVAC_MODES)}, {}, extra=vol.PREVENT_EXTRA),
    },
    "climate.turn_off": {"domain": "climate", "target_domains": {"climate"}, "target_required": True, "effect": "state", "schema": _schema({}, {}, extra=vol.ALLOW_EXTRA)},
    "climate.turn_on": {"domain": "climate", "target_domains": {"climate"}, "target_required": True, "effect": "state", "schema": _schema({}, {}, extra=vol.ALLOW_EXTRA)},
    "climate.set_preset_mode": {
        "domain": "climate",
        "target_domains": {"climate"},
        "target_required": True,
        "effect": "state",
        "schema": _schema({vol.Required("preset_mode"): str}, {}, extra=vol.ALLOW_EXTRA),
    },
    "climate.set_temperature": {
        "domain": "climate",
        "target_domains": {"climate"},
        "target_required": True,
        "effect": "state",
        "schema": _schema({}, {vol.Optional("temperature"): vol.Coerce(float), vol.Optional("hvac_mode"): vol.In(HVAC_MODES)}, extra=vol.ALLOW_EXTRA),
    },
    "input_boolean.turn_on": {"domain": "input_boolean", "target_domains": {"input_boolean"}, "target_required": True, "effect": "state", "schema": _schema({}, {}, extra=vol.PREVENT_EXTRA)},
    "input_boolean.turn_off": {"domain": "input_boolean", "target_domains": {"input_boolean"}, "target_required": True, "effect": "state", "schema": _schema({}, {}, extra=vol.PREVENT_EXTRA)},
    "input_text.set_value": {
        "domain": "input_text",
        "target_domains": {"input_text"},
        "target_required": True,
        "effect": "state",
        "schema": _schema({vol.Required("value"): vol.Any(str, int, float)}, {}, extra=vol.PREVENT_EXTRA),
    },
    "notify.mobile_app": {
        "domain": "notify",
        "target_domains": set(),
        "target_required": False,
        "effect": "event",
        "schema": _schema({}, {vol.Optional("message"): str, vol.Optional("title"): str, vol.Optional("data"): dict}, extra=vol.ALLOW_EXTRA),
    },
    "notify.notify": {
        "domain": "notify",
        "target_domains": set(),
        "target_required": False,
        "effect": "event",
        "schema": _schema({vol.Required("message"): str}, {vol.Optional("title"): str, vol.Optional("data"): dict}, extra=vol.ALLOW_EXTRA),
    },
    "persistent_notification.create": {
        "domain": "persistent_notification",
        "target_domains": set(),
        "target_required": False,
        "effect": "event",
        "schema": _schema({vol.Required("message"): str}, {vol.Optional("title"): str, vol.Optional("notification_id"): str}, extra=vol.ALLOW_EXTRA),
    },
    "persistent_notification.dismiss": {
        "domain": "persistent_notification",
        "target_domains": set(),
        "target_required": False,
        "effect": "event",
        "schema": _schema({}, {vol.Optional("notification_id"): str}, extra=vol.ALLOW_EXTRA),
    },
    "logbook.log": {
        "domain": "logbook",
        "target_domains": set(),
        "target_required": False,
        "effect": "event",
        "schema": _schema({}, {vol.Optional("name"): str, vol.Optional("message"): str, vol.Optional("entity_id"): str, vol.Optional("domain"): str}, extra=vol.ALLOW_EXTRA),
    },
    "system_log.write": {
        "domain": "system_log",
        "target_domains": set(),
        "target_required": False,
        "effect": "event",
        "schema": _schema({vol.Required("message"): vol.Any(str, None)}, {vol.Optional("level"): vol.In(LOG_LEVELS), vol.Optional("logger"): str}, extra=vol.ALLOW_EXTRA),
    },
    "google_sheets.append_sheet": {
        "domain": "google_sheets",
        "target_domains": set(),
        "target_required": False,
        "effect": "event",
        "schema": _schema({vol.Required("config_entry"): str, vol.Required("worksheet"): str, vol.Required("data"): dict}, {}, extra=vol.PREVENT_EXTRA),
    },
    "scene.turn_on": {"domain": "scene", "target_domains": {"scene"}, "target_required": True, "effect": "state", "schema": _schema({}, {}, extra=vol.ALLOW_EXTRA)},
    "scene.turn_off": {"domain": "scene", "target_domains": {"scene"}, "target_required": True, "effect": "state", "schema": _schema({}, {}, extra=vol.ALLOW_EXTRA)},
    "schedule.activate": {"domain": "schedule", "target_domains": {"schedule", "scene"}, "target_required": False, "effect": "event", "schema": _schema({}, {}, extra=vol.ALLOW_EXTRA)},
    "homeassistant.turn_on": {"domain": "homeassistant", "target_domains": {"light", "switch", "input_boolean", "automation", "script", "climate", "fan"}, "target_required": True, "effect": "state", "schema": _schema({}, {}, extra=vol.ALLOW_EXTRA)},
    "homeassistant.turn_off": {"domain": "homeassistant", "target_domains": {"light", "switch", "input_boolean", "automation", "script", "climate", "fan"}, "target_required": True, "effect": "state", "schema": _schema({}, {}, extra=vol.ALLOW_EXTRA)},
    "automation.turn_on": {"domain": "automation", "target_domains": {"automation"}, "target_required": True, "effect": "state", "schema": _schema({}, {vol.Optional("stop_actions"): bool}, extra=vol.ALLOW_EXTRA)},
    "automation.turn_off": {"domain": "automation", "target_domains": {"automation"}, "target_required": True, "effect": "state", "schema": _schema({}, {vol.Optional("stop_actions"): bool}, extra=vol.ALLOW_EXTRA)},
}

ALIASES = {
    "notify.mobile_app_phone": "notify.mobile_app",
    "logbook.write": "logbook.log",
}

UNSUPPORTED_PREFIXES = ("llmvision.", "camera.", "downloader.", "tts.", "rest_command.", "telegram_bot.")

STATELESS_OK_NO_ENTITY = {
    "notify.mobile_app",
    "notify.notify",
    "persistent_notification.create",
    "persistent_notification.dismiss",
    "logbook.log",
    "system_log.write",
    "google_sheets.append_sheet",
    "schedule.activate",
}

def canonical_service(svc: str) -> str:
    s = str(svc or "").strip()
    if s in ALIASES:
        return ALIASES[s]
    if s.startswith("notify.") and s not in SERVICE_REGISTRY:
        return "notify.mobile_app"
    return s

def domain_of_entity(eid: str | None) -> str | None:
    if not eid or "." not in str(eid):
        return None
    return str(eid).split(".", 1)[0]

class HASandbox:

    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {}
        self.effects: list[dict[str, Any]] = []
        self.disabled_services: set[str] = set()
        self.unavailable: set[str] = set()

    def clone(self) -> "HASandbox":
        other = HASandbox()
        other.states = copy.deepcopy(self.states)
        other.effects = list(self.effects)
        other.disabled_services = set(self.disabled_services)
        other.unavailable = set(self.unavailable)
        return other

    def register(self, entity_id: str, state: str = "unknown", attrs: dict | None = None) -> None:
        if not entity_id:
            return
        eid = str(entity_id)
        if eid.startswith("!input") or "{{" in eid:
            return
        dom = domain_of_entity(eid) or "unknown"
        prev = self.states.get(eid) or {}
        self.states[eid] = {
            "entity_id": eid,
            "domain": dom,
            "state": state if prev.get("state") in (None, "unknown") else prev.get("state", state),
            "attributes": {**(prev.get("attributes") or {}), **(attrs or {})},
            "available": True,
        }
        if "hvac_mode" in (attrs or {}) and dom == "climate":
            self.states[eid]["state"] = str(attrs["hvac_mode"])
            self.states[eid]["attributes"]["hvac_mode"] = attrs["hvac_mode"]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self.states)

    def call(self, service: str, entity: str | None, params: dict | None) -> dict[str, Any]:
        t0 = time.perf_counter()
        entity = entity[0] if isinstance(entity, list) and entity else entity
        if isinstance(entity, dict):
            entity = entity.get("entity_id") or entity.get("entity")
        entity = str(entity) if entity not in (None, "", [], {}) else None
        svc = canonical_service(service)
        data = dict(params or {})
        pre = self.snapshot()
        outcome = {
            "service": service,
            "canonical_service": svc,
            "entity": entity,
            "parameters": data,
            "execution_success": False,
            "exception": None,
            "pre_state": pre.get(entity) if entity else None,
            "post_state": None,
            "side_effects": [],
            "result": "EXECUTION_FAILED",
            "execution_time_ms": 0.0,
        }
        try:
            if svc in self.disabled_services or any(svc.startswith(p) for p in UNSUPPORTED_PREFIXES):
                outcome["exception"] = "service_unavailable"
                outcome["result"] = "EXECUTION_FAILED"
                return outcome
            spec = SERVICE_REGISTRY.get(svc)
            if spec is None:
                outcome["exception"] = "unknown_service"
                return outcome
            if spec["target_required"] and not entity:
                outcome["exception"] = "missing_target"
                return outcome
            if entity and entity in self.unavailable:
                outcome["exception"] = "entity_unavailable"
                return outcome
            if entity and entity not in self.states and spec["target_required"]:
                outcome["exception"] = "entity_missing"
                return outcome
            if entity and spec["target_domains"]:
                dom = domain_of_entity(entity)
                if dom and dom not in spec["target_domains"] and spec["domain"] != "homeassistant":
                    outcome["exception"] = "domain_mismatch"
                    return outcome

            check = dict(data)
            if svc == "system_log.write" and "message" not in check:
                check["message"] = data.get("logger") or "sandbox"
            spec["schema"](check)
            result = self._apply(svc, entity, data, spec)
            outcome["execution_success"] = True
            outcome["result"] = result
            outcome["post_state"] = self.states.get(entity) if entity else None
            outcome["side_effects"] = list(self.effects[-1:]) if spec["effect"] == "event" else []
        except vol.Invalid as exc:
            outcome["exception"] = f"schema:{exc}"
            outcome["result"] = "EXECUTION_FAILED"
        except Exception as exc:
            outcome["exception"] = f"contained:{type(exc).__name__}:{exc}"
            outcome["result"] = "EXECUTION_FAILED"
        finally:
            outcome["execution_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        return outcome

    def _apply(self, svc: str, entity: str | None, data: dict, spec: dict) -> str:
        if spec["effect"] == "event":
            self.effects.append({"service": svc, "entity": entity, "data": data})
            return "API_EXECUTED"
        if not entity:
            return "NO_EFFECT"
        st = self.states.setdefault(
            entity,
            {"entity_id": entity, "domain": domain_of_entity(entity), "state": "unknown", "attributes": {}, "available": True},
        )
        before = st.get("state")
        attrs = st.setdefault("attributes", {})
        if svc.endswith("turn_on") or svc == "light.turn_on" or svc == "homeassistant.turn_on":
            st["state"] = "on"
            if "brightness" in data:
                attrs["brightness"] = data["brightness"]
        elif svc.endswith("turn_off") or svc == "light.turn_off" or svc == "homeassistant.turn_off":
            if st.get("domain") == "climate" or svc.startswith("climate."):
                st["state"] = "off"
                attrs["hvac_mode"] = "off"
            else:
                st["state"] = "off"
        elif svc == "climate.set_hvac_mode":
            mode = str(data.get("hvac_mode"))
            st["state"] = mode
            attrs["hvac_mode"] = mode
        elif svc == "climate.set_preset_mode":
            attrs["preset_mode"] = data.get("preset_mode")
        elif svc == "climate.set_temperature":
            if "temperature" in data:
                attrs["temperature"] = data["temperature"]
            if data.get("hvac_mode"):
                st["state"] = data["hvac_mode"]
                attrs["hvac_mode"] = data["hvac_mode"]
        elif svc == "input_text.set_value":
            st["state"] = str(data.get("value"))
        elif svc.startswith("scene."):
            st["state"] = "on" if svc.endswith("turn_on") else "off"
        if st.get("state") == before and spec["effect"] == "state":
            return "STATE_ALREADY_SATISFIED"
        if st.get("state") != before:
            return "STATE_CHANGED"
        return "NO_EFFECT"

def seed_from_observation(box: HASandbox, observation: dict | None, entities: list | None, extra_ids: list[str] | None = None) -> None:
    obs = observation if isinstance(observation, dict) else {}
    light_st = obs.get("light_state") or obs.get("light")
    if light_st is not None:
        box.register("light.living_room", "on" if str(light_st).lower() in {"on", "true", "1"} else "off")
    hvac = obs.get("hvac_mode")
    if hvac:
        box.register("climate.living_room_ac", str(hvac).lower(), {"hvac_mode": str(hvac).lower()})
    else:
        box.register("climate.living_room_ac", "heat", {"hvac_mode": "heat"})
    if obs.get("illuminance_lux") is not None or obs.get("illuminance") is not None:
        box.register("sensor.illuminance_living_room", str(obs.get("illuminance_lux") or obs.get("illuminance")))
    motion = obs.get("motion")
    if motion is True or str(obs.get("motion_state") or "").lower() == "on":
        box.register("binary_sensor.motion", "on")
    elif motion is False or str(obs.get("motion_state") or "").lower() == "off":
        box.register("binary_sensor.motion", "off")
    ws = str(obs.get("window_state") or "").lower()
    if ws in {"open", "on"}:
        box.register("binary_sensor.window_living_room", "on", {"window_state": "open"})
    elif ws in {"closed", "off"}:
        box.register("binary_sensor.window_living_room", "off", {"window_state": "closed"})
    for e in entities or []:
        if not isinstance(e, dict):
            continue
        eid = e.get("entity_id")
        if not eid:
            continue
        attrs = e.get("attributes") or {}
        st = str(e.get("state") or "unknown")
        box.register(str(eid), st, attrs if isinstance(attrs, dict) else {})
    for eid in extra_ids or []:
        if eid:
            box.register(str(eid), "unknown")
    box.register("notify.mobile_app_phone", "unknown")
    box.register("light.living_room", box.states.get("light.living_room", {}).get("state") or "off")

def transition_ok(svc: str, result: str, post: dict | None) -> bool | None:

    cs = canonical_service(svc)
    spec = SERVICE_REGISTRY.get(cs)
    if spec and spec["effect"] == "event":
        return result in {"API_EXECUTED"}
    if result == "EXECUTION_FAILED":
        return False
    if not post:
        return False
    st = str(post.get("state") or "").lower()
    mode = str((post.get("attributes") or {}).get("hvac_mode") or st).lower()
    if cs in {"light.turn_on", "switch.turn_on", "input_boolean.turn_on", "homeassistant.turn_on"}:
        return st in {"on", "heat", "cool", "auto", "heat_cool"}
    if cs in {"light.turn_off", "switch.turn_off", "input_boolean.turn_off", "homeassistant.turn_off", "climate.turn_off"}:
        return st in {"off"} or mode == "off"
    if cs == "climate.set_hvac_mode":
        return mode in HVAC_MODES and st == mode
    if result in {"STATE_CHANGED", "STATE_ALREADY_SATISFIED", "API_EXECUTED"}:
        return True
    return False
