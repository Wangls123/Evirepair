from __future__ import annotations

from smarthome_mdf.openhab_validation.platform_semantic_model import ItemCapability

OPENHAB_ITEM_TYPE_CAPABILITIES: dict[str, ItemCapability] = {
    "Switch": ItemCapability(
        name="binary_control",
        operations=("ACTIVATE", "DEACTIVATE", "ENABLE", "DISABLE", "TOGGLE"),
        state_values=("ON", "OFF"),
        value_type="enum",
    ),
    "Dimmer": ItemCapability(
        name="dimmer_control",
        operations=("ACTIVATE", "DEACTIVATE", "SET", "INCREASE", "DECREASE"),
        state_values=("ON", "OFF"),
        value_type="percent",
    ),
    "Contact": ItemCapability(
        name="contact_sensor",
        operations=("OPEN", "CLOSE"),
        state_values=("OPEN", "CLOSED"),
        value_type="enum",
    ),
    "Number": ItemCapability(
        name="numeric_value",
        operations=("SET", "INCREASE", "DECREASE"),
        state_values=(),
        value_type="number",
    ),
    "Rollershutter": ItemCapability(
        name="rollershutter_control",
        operations=("UP", "DOWN", "STOP", "SET"),
        state_values=("UP", "DOWN", "STOP"),
        value_type="enum",
    ),
    "String": ItemCapability(
        name="string_value",
        operations=("SET",),
        state_values=(),
        value_type="string",
    ),
    "DateTime": ItemCapability(
        name="datetime_value",
        operations=("SET",),
        state_values=(),
        value_type="datetime",
    ),

    "Trigger": ItemCapability(
        name="trigger_channel",
        operations=("TRIGGER",),
        state_values=(),
        value_type="trigger",
    ),
}

DEFAULT_ITEM_TYPE = "Switch"

def extract_capability_for_item_type(item_type: str) -> ItemCapability:

    return OPENHAB_ITEM_TYPE_CAPABILITIES.get(item_type, OPENHAB_ITEM_TYPE_CAPABILITIES[DEFAULT_ITEM_TYPE])

def supported_commands_for_item_type(item_type: str) -> list[str]:

    cap = extract_capability_for_item_type(item_type)
    op_to_cmd = {
        "ACTIVATE": "ON",
        "DEACTIVATE": "OFF",
        "ENABLE": "ON",
        "DISABLE": "OFF",
        "TOGGLE": "TOGGLE",
        "OPEN": "OPEN",
        "CLOSE": "CLOSED",
        "UP": "UP",
        "DOWN": "DOWN",
        "STOP": "STOP",
        "SET": "SET",
        "INCREASE": "INCREASE",
        "DECREASE": "DECREASE",
        "TRIGGER": "TRIGGER",
    }
    cmds: list[str] = []
    for op in cap.operations:
        mapped = op_to_cmd.get(op)
        if mapped:
            cmds.append(mapped)
    for s in cap.state_values:
        if s not in cmds:
            cmds.append(s)
    return sorted(set(cmds))

def audit_no_entity_hardcoding(source_text: str) -> list[str]:

    import re

    violations: list[str] = []
    patterns = [
        (r'if\s+item\s*==\s*"', "entity_name_equality_check"),
        (r'if\s+item_name\s*==\s*"', "entity_name_equality_check"),
        (r'"light"\s+in\s+item', "entity_substring_light"),
        (r'"Light"\s+in\s+item', "entity_substring_Light"),
    ]
    for pat, label in patterns:
        if re.search(pat, source_text):
            violations.append(label)
    return violations
