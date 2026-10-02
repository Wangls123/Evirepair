from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import urllib.request

from smarthome_mdf.paths import MDF_ROOT
from smarthome_mdf.single_scene_blueprint_complete.config import (
    ALL_BLUEPRINT_IDS,
    CANONICAL_YAML_ABS,
    CANONICAL_YAML_REL,
    OUTPUT_DIR,
    PENDING_YAML_REL,
    RAW_BLUEPRINTS_ROOT,
    SCENE_BLUEPRINT_MAP,
)

CONTRACTS_PATH = MDF_ROOT / "configs" / "community_blueprint_contracts.json"

DOWNLOAD_URLS: dict[str, str] = {
    "security_osam_sensor_alert": "https://raw.githubusercontent.com/thezeus123/OSAM/main/osam_v3.12.1_final.yaml",
    "security_contact_left_open": "https://gist.githubusercontent.com/MaxAnderson95/f404cf57d4aa7c91cbe2d27738dfaf0a/raw/contact-sensor-left-open-notification.yaml",
    "security_window_dynamic_wait": "https://github.com/Flo-R1der/My_Smart-Home_stuff/raw/main/window-notifications/open-window-notifications.yaml",
    "security_ikea_myggbett": "https://github.com/aledziko/HA-blueprints/raw/main/IKEA/Matter/ikea-myggbett-e2492/ikea-myggbett-e2492-matter-door-sensor.yaml",
    "presence_holiday_away_lighting": "https://gist.githubusercontent.com/Blackshome/0a34870755762bcb9fab159d5b94fd25/raw/holiday-away-lighting.yaml",
    "appliance_power_google_sheets": "https://gist.githubusercontent.com/marcelfarres/41aa967977aae9f593caf28f7618c639/raw/power_state_monitor.yaml",
    "appliance_vibration_sensor": "https://raw.githubusercontent.com/axeldavid/VibrationSensorApplianceFilterBlueprint/master/VibrationSensorApplianceFilter.yaml",
    "presence_better_thermostat": "https://raw.githubusercontent.com/n3roGit/MyHomeAssistantMods/main/automation/BetterThermostatControl/BetterThermostat_RoomHeatControl_Lean.yaml",
}

EMBEDDED_YAML: dict[str, str] = {
    "presence_automation_state": """blueprint:
  name: Presence Automations
  description: >
    Disable automation(s) when someone arrives. Re-enable them when they leave.
  domain: automation
  input:
    person:
      name: Person
      description: Select the person to be home.
      selector:
        entity:
          domain: person
    automation:
      name: Automation
      description: Select the automation(s) that need to be turned off.
      selector:
        target:
          entity:
            domain: automation
    inverted:
      name: Invert Blueprint
      description: >
        Enable automations when the person comes home instead of disabling them.
      default: false
      selector:
        boolean: {}
trigger:
  - platform: state
    entity_id: !input person
action:
  - choose:
      - conditions:
          - condition: state
            entity_id: !input person
            state: home
        sequence:
          - choose:
              - conditions: "{{ inverted }}"
                sequence:
                  - service: automation.turn_on
                    target: !input automation
            default:
              - service: automation.turn_off
                target: !input automation
      - conditions:
          - condition: state
            entity_id: !input person
            state: not_home
        sequence:
          - choose:
              - conditions: "{{ inverted }}"
                sequence:
                  - service: automation.turn_off
                    target: !input automation
            default:
              - service: automation.turn_on
                target: !input automation
    default: []
""",
}

def _canonical_rel_path(blueprint_id: str) -> str:
    if blueprint_id in CANONICAL_YAML_REL:
        return CANONICAL_YAML_REL[blueprint_id]
    return PENDING_YAML_REL[blueprint_id]

def resolve_canonical_path(blueprint_id: str) -> Path:
    if blueprint_id in CANONICAL_YAML_ABS:
        return CANONICAL_YAML_ABS[blueprint_id]
    return RAW_BLUEPRINTS_ROOT / _canonical_rel_path(blueprint_id)

def yaml_hash(path: Path) -> str:
    if not path.is_file():
        return "missing"
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()

def _load_contracts() -> dict:
    return json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))

def _contract_url(blueprint_id: str) -> str | None:
    c = (_load_contracts().get("contracts") or {}).get(blueprint_id) or {}
    return c.get("blueprint_url")

def blueprint_to_scene(bp_id: str) -> str | None:
    for scene, ids in SCENE_BLUEPRINT_MAP.items():
        if bp_id in ids:
            return scene
    return None

def _github_blob_to_raw(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url

def download_missing_yamls() -> dict[str, Any]:
    results: list[dict] = []
    ok = 0
    fail = 0
    for bp_id in ALL_BLUEPRINT_IDS:
        path = resolve_canonical_path(bp_id)
        if path.is_file():
            results.append({"blueprint_id": bp_id, "status": "EXISTS", "yaml_path": str(path)})
            ok += 1
            continue
        if bp_id in EMBEDDED_YAML:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(EMBEDDED_YAML[bp_id], encoding="utf-8")
            results.append({"blueprint_id": bp_id, "status": "EMBEDDED", "yaml_path": str(path), "source": "community_topic_264157"})
            ok += 1
            continue
        url = DOWNLOAD_URLS.get(bp_id)
        if not url:
            results.append({"blueprint_id": bp_id, "status": "NO_DOWNLOAD_URL", "yaml_path": str(path)})
            fail += 1
            continue
        url = _github_blob_to_raw(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EviRepair/1.0"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            if "blueprint:" not in text and "domain: automation" not in text:
                raise ValueError("response does not look like HA blueprint YAML")
            path.write_text(text, encoding="utf-8")
            results.append({"blueprint_id": bp_id, "status": "DOWNLOADED", "yaml_path": str(path), "url": url})
            ok += 1
        except Exception as exc:
            results.append({"blueprint_id": bp_id, "status": "DOWNLOAD_FAILED", "error": str(exc), "url": url})
            fail += 1
    audit = {"downloaded_or_present": ok, "failed": fail, "total": 22, "rows": results}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "yaml_download_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit

def build_one_to_one_mapping() -> dict[str, Any]:

    rows: list[dict] = []
    path_to_bp: dict[str, str] = {}
    for bp_id in ALL_BLUEPRINT_IDS:
        path = resolve_canonical_path(bp_id)
        exists = path.is_file()
        rel = str(path)
        mapping_status = "MAPPED" if exists else "YAML_MISSING"
        if exists:
            other = path_to_bp.get(str(path.resolve()))
            if other and other != bp_id:
                mapping_status = "PROXY_YAML"
            else:
                path_to_bp[str(path.resolve())] = bp_id
        rows.append(
            {
                "scene": blueprint_to_scene(bp_id),
                "blueprint_id": bp_id,
                "yaml_path": rel,
                "yaml_exists": exists,
                "yaml_hash": yaml_hash(path) if exists else None,
                "source": "canonical_asset",
                "source_reference": _contract_url(bp_id),
                "mapping_status": mapping_status,
            }
        )
    complete = all(r["yaml_exists"] and r["mapping_status"] == "MAPPED" for r in rows)
    unique_paths = len({r["yaml_path"] for r in rows if r["yaml_exists"]})
    return {
        "total_blueprints": 22,
        "yaml_present": sum(1 for r in rows if r["yaml_exists"]),
        "unique_yaml_paths": unique_paths,
        "one_to_one": unique_paths == sum(1 for r in rows if r["yaml_exists"]),
        "complete": complete,
        "rows": rows,
    }
