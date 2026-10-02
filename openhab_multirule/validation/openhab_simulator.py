from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from openhab_multirule.validation.action_adapter import native_action_record
from openhab_multirule.validation.types import OpenHABScenario

@dataclass
class RuleExecutionEvent:
    rule_uid: str
    component_id: str
    item: str
    command: str
    timestamp_ms: int

@dataclass
class SimulationResult:
    scenario_id: str
    item_states: dict[str, str]
    executed_actions: list[dict[str, Any]]
    event_stream: list[RuleExecutionEvent] = field(default_factory=list)
    mode: str = "baseline"

    equipment_effects: dict[str, dict[str, Any]] = field(default_factory=dict)
    verification: str = "in_process_simulator"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "item_states": self.item_states,
            "executed_actions": self.executed_actions,
            "event_stream": [
                {
                    "rule_uid": e.rule_uid,
                    "component_id": e.component_id,
                    "item": e.item,
                    "command": e.command,
                    "timestamp_ms": e.timestamp_ms,
                }
                for e in self.event_stream
            ],
            "mode": self.mode,
            "equipment_effects": self.equipment_effects,
            "verification": self.verification,
        }

class OpenHABSimulator:

    def __init__(self, scenario: OpenHABScenario) -> None:
        self.scenario = scenario
        self.item_states: dict[str, str] = {}
        self.equipment_effects: dict[str, dict[str, Any]] = {}
        self.item_types = {item.name: item.item_type for item in scenario.items}
        for item in scenario.items:
            init = item.initial_state
            if init is None:
                if item.item_type == "HVAC":
                    init = "UNKNOWN"
                else:
                    init = "OFF" if item.item_type in ("Switch", "Dimmer") else "CLOSED"
            self.item_states[item.name] = str(init).upper()
            if item.item_type == "HVAC":
                self._set_hvac_mode(item.name, str(init))

    def reset(self) -> None:
        self.__init__(self.scenario)

    def apply_triggers(self, trigger_sequence: list[dict[str, Any]] | None = None) -> None:
        seq = trigger_sequence if trigger_sequence is not None else self.scenario.trigger_sequence
        for trig in seq:
            item = str(trig["item"])
            state = str(trig["state"]).upper()
            self.item_states[item] = state

    def run_baseline(self, *, trigger_sequence: list[dict[str, Any]] | None = None) -> SimulationResult:
        self.reset()
        self.apply_triggers(trigger_sequence)
        events: list[RuleExecutionEvent] = []
        actions: list[dict[str, Any]] = []
        ts = 0
        enabled_rules = [r for r in self.scenario.rules if r.enabled]
        for rule in enabled_rules:
            trig_state = self.item_states.get(rule.trigger_item, "")
            if trig_state != str(rule.trigger_state).upper():
                continue
            for act in rule.actions:
                if act.get("type") != "sendCommand":
                    continue
                item = str(act["item"])
                cmd = str(act["command"]).upper()
                ts += 1
                events.append(
                    RuleExecutionEvent(
                        rule_uid=rule.uid,
                        component_id=rule.component_id,
                        item=item,
                        command=cmd,
                        timestamp_ms=ts,
                    )
                )
                self._apply_command(item, cmd)
                actions.append(
                    native_action_record(item, cmd, rule_uid=rule.uid, component_id=rule.component_id)
                )
        return SimulationResult(
            scenario_id=self.scenario.scenario_id,
            item_states=dict(self.item_states),
            executed_actions=actions,
            event_stream=events,
            mode="baseline",
            equipment_effects=dict(self.equipment_effects),
            verification="in_process_simulator",
        )

    def run_with_commands(
        self,
        commands: list[tuple[str, str]],
        *,
        trigger_sequence: list[dict[str, Any]] | None = None,
    ) -> SimulationResult:

        self.reset()
        self.apply_triggers(trigger_sequence)
        events: list[RuleExecutionEvent] = []
        actions: list[dict[str, Any]] = []
        ts = 0
        for item, cmd in commands:
            ts += 1
            events.append(
                RuleExecutionEvent(
                    rule_uid="repair_plan",
                    component_id="repair",
                    item=item,
                    command=cmd.upper(),
                    timestamp_ms=ts,
                )
            )
            self._apply_command(item, cmd.upper())
            actions.append(
                native_action_record(item, cmd.upper(), rule_uid="repair_plan", component_id="repair")
            )
        return SimulationResult(
            scenario_id=self.scenario.scenario_id,
            item_states=dict(self.item_states),
            executed_actions=actions,
            event_stream=events,
            mode="repair",
            equipment_effects=dict(self.equipment_effects),
            verification="in_process_simulator",
        )

    def _set_hvac_mode(self, item: str, mode_text: str) -> None:

        mode = str(mode_text or "").strip().lower()
        if mode.startswith("mode_"):
            mode = mode.split("_", 1)[1]
        if mode in {"", "unknown"}:
            self.item_states[item] = "UNKNOWN"
            self.equipment_effects[item] = {
                "platform": "openhab",
                "openhab_version": "4.2.0",
                "channel": "hvacMode",
                "item_type": "String",
                "binding": "thermostat hvacMode channel, simulated; no live Thing",
                "mode": None,
                "conditioning": None,
            }
            return
        self.item_states[item] = mode.upper()
        self.equipment_effects[item] = {
            "platform": "openhab",
            "openhab_version": "4.2.0",
            "channel": "hvacMode",
            "item_type": "String",
            "binding": "thermostat hvacMode channel, simulated; no live Thing",
            "mode": mode,
            "conditioning": mode not in {"off", "idle"},
        }

    def _apply_command(self, item: str, command: str) -> None:
        cmd = command.upper()
        if self.item_types.get(item) == "HVAC" and cmd.startswith("MODE_"):
            self._set_hvac_mode(item, cmd)
            return
        if cmd in ("ON", "OFF"):
            self.item_states[item] = cmd
        elif cmd in ("OPEN", "CLOSED"):
            self.item_states[item] = cmd
        else:
            self.item_states[item] = cmd

    def snapshot_items(self) -> dict[str, str]:
        return deepcopy(self.item_states)
