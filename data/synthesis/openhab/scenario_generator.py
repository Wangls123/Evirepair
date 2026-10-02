from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from smarthome_mdf.openhab_validation.config import OPENHAB_SCENARIOS_DIR
from smarthome_mdf.openhab_validation.types import OpenHABItem, OpenHABRule, OpenHABScenario

CONFLICT_TEMPLATES = (
    "missing_action",
    "remove_conflict",
    "extra_action",
    "opposing_light",
    "sequence",
    "mixed_conflict",
)

OPERATOR_COVERAGE = {
    "missing_action": "ADD",
    "remove_conflict": "REMOVE",
    "extra_action": "REMOVE",
    "opposing_light": "REPLACE",
    "sequence": "SEQUENCE",
    "mixed_conflict": "MIXED",
    "clean_no_conflict": "NONE",
    "clean_no_conflict_v2": "NONE",
    "unfixable_opposing": "UNFIXABLE",
    "adversarial_equal_priority": "ADVERSARIAL",
    "ambiguous_no_priority": "AMBIGUOUS",
    "blind_injection": "BLIND",
    "blind_clean": "NONE",
    "blind_ambiguous": "AMBIGUOUS",
}

AUDIT_TEMPLATES = (
    "clean_no_conflict",
    "ambiguous_no_priority",
    "unfixable_opposing",
    "adversarial_equal_priority",
)

SAFETY_TEMPLATES = (
    "clean_no_conflict",
    "ambiguous_no_priority",
)

class OpenHABScenarioGenerator:

    def __init__(self, *, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def generate_batch(
        self,
        n: int,
        *,
        conflict_types: tuple[str, ...] | None = None,
        prefix: str = "oh_sc",
    ) -> list[OpenHABScenario]:
        types = conflict_types or CONFLICT_TEMPLATES
        out: list[OpenHABScenario] = []
        for i in range(n):
            ctype = types[i % len(types)]
            sid = f"{prefix}_{i:04d}_{ctype}"
            out.append(self.generate_one(sid, conflict_type=ctype, seed=self._rng.randint(0, 2**31 - 1)))
        return out

    def generate_one(
        self,
        scenario_id: str,
        *,
        conflict_type: str = "opposing_light",
        seed: int | None = None,
    ) -> OpenHABScenario:
        rng = random.Random(seed if seed is not None else self._rng.randint(0, 2**31 - 1))
        builders = {
            "opposing_light": self._build_opposing_light,
            "missing_action": self._build_missing_action,
            "remove_conflict": self._build_remove_conflict,
            "extra_action": self._build_extra_action,
            "sequence": self._build_sequence,
            "mixed_conflict": self._build_mixed_conflict,
            "clean_no_conflict": self._build_clean_no_conflict,
            "clean_no_conflict_v2": self._build_clean_no_conflict_v2,
            "unfixable_opposing": self._build_unfixable_opposing,
            "adversarial_equal_priority": self._build_adversarial_equal_priority,
            "ambiguous_no_priority": self._build_ambiguous_no_priority,
        }
        fn = builders.get(conflict_type, self._build_opposing_light)
        return fn(scenario_id, rng, conflict_type)

    def save_scenarios(
        self,
        scenarios: list[OpenHABScenario],
        out_dir: Path | None = None,
        *,
        corpus_id: str = "openhab_v3",
    ) -> Path:
        root = out_dir or OPENHAB_SCENARIOS_DIR
        root.mkdir(parents=True, exist_ok=True)
        index: list[dict[str, Any]] = []
        for sc in scenarios:
            record = sc.to_dict()
            path = root / f"{sc.scenario_id}.json"
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            index.append(
                {
                    "scenario_id": sc.scenario_id,
                    "conflict_type": sc.conflict_type,
                    "repair_operator": (sc.metadata or {}).get("repair_operator"),
                    "n_items": len(sc.items),
                    "n_rules": len(sc.rules),
                    "path": str(path.name),
                }
            )
        manifest = root / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as f:
            for row in index:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        corpus_meta = {
            "corpus_id": corpus_id,
            "n_scenarios": len(scenarios),
            "conflict_types": list({sc.conflict_type for sc in scenarios}),
            "operator_coverage": OPERATOR_COVERAGE,
            "scenarios": index,
        }
        (root / "corpus_index.json").write_text(
            json.dumps(corpus_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return root

    def load_corpus(self, corpus_dir: Path) -> list[OpenHABScenario]:
        manifest = corpus_dir / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"missing manifest: {manifest}")
        out: list[OpenHABScenario] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["scenario_id"]
            data = json.loads((corpus_dir / f"{sid}.json").read_text(encoding="utf-8"))
            out.append(OpenHABScenario.from_dict(data))
        return out

    def _base_items(self, rng: random.Random, room: str, *, with_aux: bool = False) -> list[OpenHABItem]:
        light = f"{room}_Light"
        switch = f"{room}_Switch"
        contact = f"{room}_WindowContact"
        items = [
            OpenHABItem(name=light, item_type="Switch", label=f"{room} Light", group="Lights", initial_state="OFF"),
            OpenHABItem(name=switch, item_type="Switch", label=f"{room} Switch", group="Switches", initial_state="OFF"),
            OpenHABItem(name=contact, item_type="Contact", label=f"{room} Window", group="Sensors", initial_state="CLOSED"),
        ]
        if with_aux:
            aux = f"{room}_AuxPlug"
            items.append(
                OpenHABItem(name=aux, item_type="Switch", label=f"{room} Aux", group="Appliances", initial_state="OFF")
            )
        return items

    def _rule(
        self,
        *,
        uid: str,
        name: str,
        component_id: str,
        scene_type: str,
        trigger_item: str,
        trigger_state: str,
        item: str,
        command: str,
        description: str = "",
    ) -> OpenHABRule:
        return OpenHABRule(
            uid=uid,
            name=name,
            component_id=component_id,
            scene_type=scene_type,
            trigger_item=trigger_item,
            trigger_state=trigger_state,
            actions=[{"type": "sendCommand", "item": item, "command": command}],
            description=description,
        )

    def _build_opposing_light(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:
        room = rng.choice(["LivingRoom", "Bedroom", "Kitchen"])
        items = self._base_items(rng, room)
        light = f"{room}_Light"
        contact = f"{room}_WindowContact"
        rules = [
            self._rule(
                uid=f"{scenario_id}_r_motion_on",
                name="Motion turn on light",
                component_id="comp_motion",
                scene_type="advanced_lighting",
                trigger_item=contact,
                trigger_state="OPEN",
                item=light,
                command="ON",
            ),
            self._rule(
                uid=f"{scenario_id}_r_security_off",
                name="Security turn off light",
                component_id="comp_security",
                scene_type="notification_security",
                trigger_item=contact,
                trigger_state="OPEN",
                item=light,
                command="OFF",
            ),
        ]
        trigger = [{"item": contact, "state": "OPEN", "delay_ms": 0}]
        hidden_gt = {
            "desired_item_states": {light: "ON"},
            "rationale": "Motion lighting should win over security off when occupant present.",
            "conflict_items": [light],
            "expected_repair_operator": "REPLACE",
        }
        return OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger,
            hidden_ground_truth=hidden_gt,
            metadata={"template": "opposing_light", "room": room, "repair_operator": "REPLACE"},
        )

    def _build_missing_action(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:
        room = rng.choice(["Hallway", "Garage", "Office"])
        items = self._base_items(rng, room)
        light = f"{room}_Light"
        switch = f"{room}_Switch"
        rules = [
            self._rule(
                uid=f"{scenario_id}_r_switch_on",
                name="Switch should turn on light (intent)",
                component_id="comp_switch",
                scene_type="on_off_schedule",
                trigger_item=switch,
                trigger_state="ON",
                item=switch,
                command="ON",
                description="Bug: command targets switch, intent is light ON",
            ),
        ]
        trigger = [{"item": switch, "state": "ON", "delay_ms": 0}]
        hidden_gt = {
            "desired_item_states": {light: "ON", switch: "ON"},
            "rationale": "Switch ON should turn on light; buggy rule omits light command.",
            "missing_commands": [{"item": light, "command": "ON"}],
            "expected_repair_operator": "ADD",
        }
        scenario = OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger,
            hidden_ground_truth=hidden_gt,
            metadata={"template": "missing_action", "room": room, "repair_operator": "ADD"},
        )
        scenario.rules[0].actions = [{"type": "sendCommand", "item": switch, "command": "ON"}]
        scenario.rules[0].intent_actions = [
            {"type": "sendCommand", "item": light, "command": "ON"},
            {"type": "sendCommand", "item": switch, "command": "ON"},
        ]
        return scenario

    def _build_extra_action(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:
        room = rng.choice(["Bedroom", "Bathroom", "Kitchen"])
        items = self._base_items(rng, room)
        light = f"{room}_Light"
        switch = f"{room}_Switch"
        rules = [
            self._rule(
                uid=f"{scenario_id}_r_night_off",
                name="Night mode off",
                component_id="comp_night",
                scene_type="on_off_schedule",
                trigger_item=switch,
                trigger_state="OFF",
                item=light,
                command="OFF",
            ),
            self._rule(
                uid=f"{scenario_id}_r_spurious_on",
                name="Spurious on (extra)",
                component_id="comp_spurious",
                scene_type="periodic_task_scheduling",
                trigger_item=switch,
                trigger_state="OFF",
                item=light,
                command="ON",
            ),
        ]
        rules[1].intent_actions = []
        trigger = [{"item": switch, "state": "OFF", "delay_ms": 0}]
        hidden_gt = {
            "desired_item_states": {light: "OFF"},
            "rationale": "Night mode should OFF light; spurious ON must be removed.",
            "extra_commands": [{"item": light, "command": "ON"}],
            "expected_repair_operator": "REMOVE",
        }
        return OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger,
            hidden_ground_truth=hidden_gt,
            metadata={"template": "extra_action", "room": room, "repair_operator": "REMOVE"},
        )

    def _build_sequence(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:
        room = rng.choice(["LivingRoom", "Office"])
        items = self._base_items(rng, room)
        light = f"{room}_Light"
        contact = f"{room}_WindowContact"
        rules = [
            self._rule(
                uid=f"{scenario_id}_r_open_on",
                name="Window open turns on light",
                component_id="comp_window",
                scene_type="climate_window",
                trigger_item=contact,
                trigger_state="OPEN",
                item=light,
                command="ON",
            ),
            self._rule(
                uid=f"{scenario_id}_r_open_off",
                name="Window open turns off light (order conflict)",
                component_id="comp_window_off",
                scene_type="climate_window",
                trigger_item=contact,
                trigger_state="OPEN",
                item=light,
                command="OFF",
            ),
        ]
        trigger = [{"item": contact, "state": "OPEN", "delay_ms": 0}]
        hidden_gt = {
            "desired_item_states": {light: "ON"},
            "rationale": "First rule intent (ON) should prevail over conflicting OFF.",
            "execution_order": ["comp_window", "comp_window_off"],
            "expected_repair_operator": "REPLACE",
        }
        return OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger,
            hidden_ground_truth=hidden_gt,
            metadata={"template": "sequence", "room": room, "repair_operator": "REPLACE",
                      "component_execution_order": ["comp_window", "comp_window_off"]},
        )

    def _build_remove_conflict(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:

        room = rng.choice(["Kitchen", "Garage", "Office"])
        items = self._base_items(rng, room, with_aux=True)
        light = f"{room}_Light"
        switch = f"{room}_Switch"
        aux = f"{room}_AuxPlug"
        rules = [
            self._rule(
                uid=f"{scenario_id}_r_light_off",
                name="Switch off turns light off",
                component_id="comp_light",
                scene_type="on_off_schedule",
                trigger_item=switch,
                trigger_state="OFF",
                item=light,
                command="OFF",
            ),
            self._rule(
                uid=f"{scenario_id}_r_spurious_aux",
                name="Spurious aux on",
                component_id="comp_spurious_aux",
                scene_type="appliance_monitoring",
                trigger_item=switch,
                trigger_state="OFF",
                item=aux,
                command="ON",
            ),
        ]
        rules[1].intent_actions = []
        trigger = [{"item": switch, "state": "OFF", "delay_ms": 0}]
        hidden_gt = {
            "desired_item_states": {light: "OFF", aux: "OFF"},
            "rationale": "Light OFF is intended; aux ON is spurious and must be removed.",
            "expected_repair_operator": "REMOVE",
        }
        return OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger,
            hidden_ground_truth=hidden_gt,
            metadata={
                "template": "remove_conflict",
                "room": room,
                "repair_operator": "REMOVE",
                "component_execution_order": ["comp_light", "comp_spurious_aux"],
            },
        )

    def _build_mixed_conflict(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:

        room = rng.choice(["LivingRoom", "Bedroom", "Hallway"])
        items = self._base_items(rng, room, with_aux=True)
        light = f"{room}_Light"
        switch = f"{room}_Switch"
        contact = f"{room}_WindowContact"
        aux = f"{room}_AuxPlug"
        rules = [
            self._rule(
                uid=f"{scenario_id}_r_motion_on",
                name="Motion on",
                component_id="comp_motion",
                scene_type="advanced_lighting",
                trigger_item=contact,
                trigger_state="OPEN",
                item=light,
                command="ON",
            ),
            self._rule(
                uid=f"{scenario_id}_r_security_off",
                name="Security off",
                component_id="comp_security",
                scene_type="notification_security",
                trigger_item=contact,
                trigger_state="OPEN",
                item=light,
                command="OFF",
            ),
            self._rule(
                uid=f"{scenario_id}_r_bug_switch",
                name="Buggy switch handler",
                component_id="comp_switch_bug",
                scene_type="on_off_schedule",
                trigger_item=contact,
                trigger_state="OPEN",
                item=switch,
                command="ON",
            ),
            self._rule(
                uid=f"{scenario_id}_r_spurious_aux",
                name="Spurious aux",
                component_id="comp_aux",
                scene_type="appliance_monitoring",
                trigger_item=contact,
                trigger_state="OPEN",
                item=aux,
                command="ON",
            ),
        ]
        rules[2].intent_actions = [
            {"type": "sendCommand", "item": light, "command": "ON"},
            {"type": "sendCommand", "item": switch, "command": "ON"},
        ]
        rules[3].intent_actions = []
        trigger = [{"item": contact, "state": "OPEN", "delay_ms": 0}]
        hidden_gt = {
            "desired_item_states": {light: "ON", switch: "ON", aux: "OFF"},
            "rationale": "Motion ON + switch bug fixed via ADD; aux spurious REMOVE; light REPLACE.",
            "expected_repair_operator": "MIXED",
        }
        return OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger,
            hidden_ground_truth=hidden_gt,
            metadata={
                "template": "mixed_conflict",
                "room": room,
                "repair_operator": "MIXED",
                "component_execution_order": [
                    "comp_motion",
                    "comp_security",
                    "comp_switch_bug",
                    "comp_aux",
                ],
            },
        )

    def _build_clean_no_conflict(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:
        room = rng.choice(["Office", "Studio", "Den"])
        items = self._base_items(rng, room)
        light = f"{room}_Light"
        switch = f"{room}_Switch"
        rules = [
            self._rule(
                uid=f"{scenario_id}_r_ok",
                name="Correct switch-light binding",
                component_id="comp_ok",
                scene_type="on_off_schedule",
                trigger_item=switch,
                trigger_state="ON",
                item=light,
                command="ON",
            ),
        ]
        trigger = [{"item": switch, "state": "ON", "delay_ms": 0}]
        hidden_gt = {
            "desired_item_states": {light: "ON"},
            "rationale": "Baseline execution already satisfies GT.",
            "expected_repair_operator": "NONE",
            "repair_expected": False,
        }
        return OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger,
            hidden_ground_truth=hidden_gt,
            metadata={"template": "clean_no_conflict", "room": room, "repair_operator": "NONE", "baseline_correct": True},
        )

    def _build_ambiguous_no_priority(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:

        room = rng.choice(["Atrium", "Lobby", "Forum"])
        items = self._base_items(rng, room)
        light = f"{room}_Light"
        contact = f"{room}_WindowContact"
        rules = [
            self._rule(
                uid=f"{scenario_id}_r_policy_a",
                name="Policy A ON",
                component_id="comp_policy_a",
                scene_type="advanced_lighting",
                trigger_item=contact,
                trigger_state="OPEN",
                item=light,
                command="ON",
            ),
            self._rule(
                uid=f"{scenario_id}_r_policy_b",
                name="Policy B OFF",
                component_id="comp_policy_b",
                scene_type="notification_security",
                trigger_item=contact,
                trigger_state="OPEN",
                item=light,
                command="OFF",
            ),
        ]
        rules[0].intent_actions = [{"type": "sendCommand", "item": light, "command": "ON"}]
        rules[1].intent_actions = [{"type": "sendCommand", "item": light, "command": "OFF"}]
        trigger = [{"item": contact, "state": "OPEN", "delay_ms": 0}]
        hidden_gt = {
            "resolution_status": "unresolved",
            "desired_item_states": {},
            "rationale": "Equal-priority opposing policies — GT is unresolved; repair must not force a choice.",
            "repair_expected": False,
            "ambiguous": True,
        }
        return OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger,
            hidden_ground_truth=hidden_gt,
            metadata={
                "template": "ambiguous_no_priority",
                "room": room,
                "repair_operator": "AMBIGUOUS",
                "ambiguous": True,
                "equal_priority": True,
                "component_execution_order": ["comp_policy_a", "comp_policy_b"],
            },
        )

    def _build_unfixable_opposing(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:
        room = rng.choice(["Atrium", "Lobby"])
        items = self._base_items(rng, room)
        light = f"{room}_Light"
        contact = f"{room}_WindowContact"
        rules = [
            self._rule(
                uid=f"{scenario_id}_r_a_on",
                name="Policy A on",
                component_id="comp_policy_a",
                scene_type="advanced_lighting",
                trigger_item=contact,
                trigger_state="OPEN",
                item=light,
                command="ON",
            ),
            self._rule(
                uid=f"{scenario_id}_r_b_off",
                name="Policy B off",
                component_id="comp_policy_b",
                scene_type="notification_security",
                trigger_item=contact,
                trigger_state="OPEN",
                item=light,
                command="OFF",
            ),
        ]
        trigger = [{"item": contact, "state": "OPEN", "delay_ms": 0}]
        hidden_gt = {
            "desired_item_states": {light: "ON"},
            "rationale": "Equal-priority opposing policies — ambiguous without external priority.",
            "expected_repair_operator": "UNFIXABLE",
            "unfixable": True,
        }
        return OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger,
            hidden_ground_truth=hidden_gt,
            metadata={
                "template": "unfixable_opposing",
                "room": room,
                "repair_operator": "UNFIXABLE",
                "component_execution_order": ["comp_policy_a", "comp_policy_b"],
                "equal_priority": True,
            },
        )

    def _build_adversarial_equal_priority(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:
        room = rng.choice(["Lab", "TestRoom"])
        items = self._base_items(rng, room)
        light = f"{room}_Light"
        switch = f"{room}_Switch"
        rules = [
            self._rule(
                uid=f"{scenario_id}_r1",
                name="Rule 1 ON",
                component_id="comp_r1",
                scene_type="on_off_schedule",
                trigger_item=switch,
                trigger_state="ON",
                item=light,
                command="ON",
            ),
            self._rule(
                uid=f"{scenario_id}_r2",
                name="Rule 2 OFF",
                component_id="comp_r2",
                scene_type="on_off_schedule",
                trigger_item=switch,
                trigger_state="ON",
                item=light,
                command="OFF",
            ),
            self._rule(
                uid=f"{scenario_id}_r3",
                name="Rule 3 ON again",
                component_id="comp_r3",
                scene_type="periodic_task_scheduling",
                trigger_item=switch,
                trigger_state="ON",
                item=light,
                command="ON",
            ),
        ]
        trigger = [{"item": switch, "state": "ON", "delay_ms": 0}]
        hidden_gt = {
            "desired_item_states": {light: "ON"},
            "rationale": "Adversarial equal-rank cycle.",
            "expected_repair_operator": "ADVERSARIAL",
            "adversarial": True,
        }
        return OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger,
            hidden_ground_truth=hidden_gt,
            metadata={
                "template": "adversarial_equal_priority",
                "room": room,
                "repair_operator": "ADVERSARIAL",
                "component_execution_order": ["comp_r1", "comp_r2", "comp_r3"],
                "equal_priority": True,
            },
        )

    def generate_clean_corpus(self, n: int = 100, *, prefix: str = "clean") -> list[OpenHABScenario]:

        return [
            self.generate_one(
                f"{prefix}_{i:04d}",
                conflict_type="clean_no_conflict",
                seed=self._rng.randint(0, 2**31 - 1),
            )
            for i in range(n)
        ]

    def generate_clean_no_conflict_v2_corpus(self, n: int = 50, *, prefix: str = "clean_v2") -> list[OpenHABScenario]:

        return [
            self.generate_one(
                f"{prefix}_{i:04d}",
                conflict_type="clean_no_conflict_v2",
                seed=self._rng.randint(0, 2**31 - 1),
            )
            for i in range(n)
        ]

    def _extended_items(self, rng: random.Random, room: str) -> list[OpenHABItem]:

        return [
            OpenHABItem(name=f"{room}_Light", item_type="Switch", label=f"{room} Light", group="Lights", initial_state="OFF"),
            OpenHABItem(name=f"{room}_Dimmer", item_type="Dimmer", label=f"{room} Dimmer", group="Lights", initial_state="OFF"),
            OpenHABItem(name=f"{room}_Temp", item_type="Number", label=f"{room} Temp", group="Sensors", initial_state=21),
            OpenHABItem(name=f"{room}_Contact", item_type="Contact", label=f"{room} Door", group="Sensors", initial_state="CLOSED"),
            OpenHABItem(name=f"{room}_Trigger", item_type="Trigger", label=f"{room} Remote", group="Triggers", initial_state=""),
        ]

    def _build_clean_no_conflict_v2(self, scenario_id: str, rng: random.Random, conflict_type: str) -> OpenHABScenario:
        room = rng.choice(["Hall", "Patio", "Garage", "Nursery"])
        items = self._extended_items(rng, room)
        light = f"{room}_Light"
        dimmer = f"{room}_Dimmer"
        temp = f"{room}_Temp"
        contact = f"{room}_Contact"
        trigger = f"{room}_Trigger"

        rule_style = rng.choice(["simple", "multi_action", "chained", "priority"])
        rules: list[OpenHABRule] = []
        comp_order: list[str] = []

        if rule_style == "simple":
            rules.append(
                OpenHABRule(
                    uid=f"{scenario_id}_r_simple",
                    name="Simple switch binding",
                    component_id="comp_simple",
                    scene_type="on_off_schedule",
                    trigger_item=contact,
                    trigger_state="OPEN",
                    actions=[{"type": "sendCommand", "item": light, "command": "ON"}],
                )
            )
            comp_order = ["comp_simple"]
        elif rule_style == "multi_action":
            rules.append(
                OpenHABRule(
                    uid=f"{scenario_id}_r_multi",
                    name="Multi action rule",
                    component_id="comp_multi",
                    scene_type="advanced_lighting",
                    trigger_item=contact,
                    trigger_state="OPEN",
                    actions=[
                        {"type": "sendCommand", "item": light, "command": "ON"},
                        {"type": "sendCommand", "item": dimmer, "command": "ON"},
                        {"type": "postUpdate", "item": temp, "state": "22"},
                    ],
                )
            )
            comp_order = ["comp_multi"]
        elif rule_style == "chained":
            rules.extend([
                OpenHABRule(
                    uid=f"{scenario_id}_r_chain1",
                    name="Chain step 1",
                    component_id="comp_chain1",
                    scene_type="periodic_task_scheduling",
                    trigger_item=contact,
                    trigger_state="OPEN",
                    actions=[{"type": "sendCommand", "item": light, "command": "ON"}],
                ),
                OpenHABRule(
                    uid=f"{scenario_id}_r_chain2",
                    name="Chain step 2",
                    component_id="comp_chain2",
                    scene_type="periodic_task_scheduling",
                    trigger_item=light,
                    trigger_state="ON",
                    actions=[{"type": "sendCommand", "item": dimmer, "command": "ON"}],
                ),
            ])
            comp_order = ["comp_chain1", "comp_chain2"]
        else:
            rules.extend([
                OpenHABRule(
                    uid=f"{scenario_id}_r_pri_a",
                    name="Priority A",
                    component_id="comp_pri_a",
                    scene_type="notification_security",
                    trigger_item=trigger,
                    trigger_state="TRIGGER",
                    actions=[{"type": "sendCommand", "item": light, "command": "ON"}],
                ),
                OpenHABRule(
                    uid=f"{scenario_id}_r_pri_b",
                    name="Priority B",
                    component_id="comp_pri_b",
                    scene_type="on_off_schedule",
                    trigger_item=trigger,
                    trigger_state="TRIGGER",
                    actions=[{"type": "sendCommand", "item": dimmer, "command": "ON"}],
                ),
            ])
            comp_order = ["comp_pri_a", "comp_pri_b"]

        trig_item = contact if rule_style != "priority" else trigger
        trig_state = "OPEN" if rule_style != "priority" else "TRIGGER"
        trigger_seq = [{"item": trig_item, "state": trig_state, "delay_ms": 0}]

        desired = {light: "ON"}
        if rule_style in ("multi_action", "chained", "priority"):
            desired[dimmer] = "ON"

        hidden_gt = {
            "desired_item_states": desired,
            "rationale": "Clean v2 — repair must not alter correct behavior.",
            "expected_repair_operator": "NONE",
            "repair_expected": False,
        }
        return OpenHABScenario(
            scenario_id=scenario_id,
            seed=rng.randint(0, 999999),
            conflict_type=conflict_type,
            items=items,
            rules=rules,
            trigger_sequence=trigger_seq,
            hidden_ground_truth=hidden_gt,
            metadata={
                "template": "clean_no_conflict_v2",
                "room": room,
                "repair_operator": "NONE",
                "baseline_correct": True,
                "rule_structure": rule_style,
                "component_execution_order": comp_order,
            },
        )

    def generate_ambiguous_batch(self, n: int, *, prefix: str = "ambig") -> list[OpenHABScenario]:
        return [
            self.generate_one(
                f"{prefix}_{i:04d}",
                conflict_type="ambiguous_no_priority",
                seed=self._rng.randint(0, 2**31 - 1),
            )
            for i in range(n)
        ]

    def generate_audit_batch(self, n: int, *, prefix: str = "oh_audit") -> list[OpenHABScenario]:
        out: list[OpenHABScenario] = []
        for i in range(n):
            ctype = AUDIT_TEMPLATES[i % len(AUDIT_TEMPLATES)]
            sid = f"{prefix}_{i:04d}_{ctype}"
            out.append(self.generate_one(sid, conflict_type=ctype, seed=self._rng.randint(0, 2**31 - 1)))
        return out
