from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.openhab_validation.platform_semantic_extractor import scenario_to_openhab_files
from smarthome_mdf.openhab_validation.scenario_generator import OpenHABScenarioGenerator
from smarthome_mdf.openhab_validation.types import OpenHABItem, OpenHABRule, OpenHABScenario
from smarthome_mdf.paths import MDF_ROOT

GENERATOR_VERSION = "openhab_final_benchmark_v2"
GENERATOR_SEED = 20260909
DEFAULT_OUT = MDF_ROOT / "runs" / "final_openhab_validation" / "dataset"

SCENARIO_TYPE_QUOTAS = {
    "repairable": 720,
    "clean": 180,
    "ambiguous": 180,
    "unsupported": 120,
}

COMPLEXITY_QUOTAS = {
    1: 300,
    2: 360,
    3: 300,
    4: 240,
}

FAMILIES = (
    "lighting",
    "climate",
    "security",
    "appliance",
    "notification",
    "schedule",
)

TYPE_TO_CONFLICT = {
    "repairable": (
        "opposing_light",
        "missing_action",
        "remove_conflict",
        "extra_action",
        "sequence",
        "mixed_conflict",
    ),
    "clean": ("clean_no_conflict", "clean_no_conflict_v2"),
    "ambiguous": ("ambiguous_no_priority", "adversarial_equal_priority"),
    "unsupported": ("unfixable_opposing",),
}

FAMILY_SCENE_TYPES = {
    "lighting": "advanced_lighting",
    "climate": "thermostat_control",
    "security": "notification_security",
    "appliance": "appliance_power",
    "notification": "notification_push",
    "schedule": "on_off_schedule",
}

@dataclass
class ScenarioSpec:
    scenario_type: str
    complexity: int
    family: str
    index: int

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _complexity_from_scenario(sc: OpenHABScenario) -> int:
    n_rules = len([r for r in sc.rules if r.enabled])
    n_actions = sum(len(r.actions) for r in sc.rules if r.enabled)
    base = max(n_rules, min(n_actions, 4))
    if base >= 4:
        return 4
    return max(1, base)

def _assign_family(sc: OpenHABScenario, family: str) -> OpenHABScenario:
    scene = FAMILY_SCENE_TYPES.get(family, "advanced_lighting")
    for rule in sc.rules:
        rule.scene_type = scene
    meta = dict(sc.metadata or {})
    meta["scenario_family"] = family
    meta.pop("repair_operator", None)
    sc.metadata = meta
    return sc

def _structural_tags(sc: OpenHABScenario) -> list[str]:
    tags: list[str] = []
    items = {i.name for i in sc.items}
    shared = set()
    for r in sc.rules:
        for act in r.actions:
            item = act.get("item")
            if item in shared:
                tags.append("shared_entity")
            shared.add(item)
    if len(sc.rules) > 1:
        tags.append("multi_rule")
    if any(len(r.actions) > 1 for r in sc.rules):
        tags.append("parameterized")
    if sc.conflict_type in ("sequence", "mixed_conflict"):
        tags.append("temporal")
    ct = sc.conflict_type
    if ct == "opposing_light":
        tags.append("opposing_operation")
    elif ct == "missing_action":
        tags.append("missing_obligation")
    elif ct == "extra_action":
        tags.append("extra_behavior")
    elif ct in ("remove_conflict",):
        tags.append("replace_behavior")
    elif "duplicate" in ct:
        tags.append("duplicate_behavior")
    return sorted(set(tags))

def _build_specs(seed: int = GENERATOR_SEED) -> list[ScenarioSpec]:
    specs: list[ScenarioSpec] = []
    idx = 0
    for stype, quota in SCENARIO_TYPE_QUOTAS.items():
        per_family = quota // len(FAMILIES)
        remainder = quota % len(FAMILIES)
        for fi, family in enumerate(FAMILIES):
            n = per_family + (1 if fi < remainder else 0)
            for _ in range(n):
                specs.append(ScenarioSpec(stype, 1, family, idx))
                idx += 1
    rng = random.Random(seed)
    rng.shuffle(specs)
    complexity_pool: list[int] = []
    for c, q in COMPLEXITY_QUOTAS.items():
        complexity_pool.extend([c] * q)
    rng.shuffle(complexity_pool)
    for i, spec in enumerate(specs):
        spec.complexity = complexity_pool[i]
    return specs

def _generate_for_spec(gen: OpenHABScenarioGenerator, spec: ScenarioSpec) -> OpenHABScenario:
    conflicts = TYPE_TO_CONFLICT[spec.scenario_type]
    ctype = conflicts[spec.index % len(conflicts)]
    sid = f"oh_final_{spec.index:04d}_{spec.family}_{ctype}"
    sc = gen.generate_one(sid, conflict_type=ctype, seed=GENERATOR_SEED + spec.index)
    sc = _assign_family(sc, spec.family)
    while _complexity_from_scenario(sc) < spec.complexity and len(sc.rules) < spec.complexity:
        sc = _augment_complexity(sc, spec)
    return sc

def _augment_complexity(sc: OpenHABScenario, spec: ScenarioSpec) -> OpenHABScenario:
    room = (sc.metadata or {}).get("room", "Room")
    n = len(sc.rules)
    light = next((i.name for i in sc.items if i.item_type == "Switch"), f"{room}_Light")
    aux = f"{room}_Aux{n}"
    sc.items.append(OpenHABItem(name=aux, item_type="Switch", label=f"{room} Aux {n}", group="Appliances", initial_state="OFF"))
    sc.rules.append(
        OpenHABRule(
            uid=f"{sc.scenario_id}_r_extra_{n}",
            name=f"Extra rule {n}",
            component_id=f"comp_extra_{n}",
            scene_type=FAMILY_SCENE_TYPES.get(spec.family, "on_off_schedule"),
            trigger_item=sc.trigger_sequence[0]["item"] if sc.trigger_sequence else aux,
            trigger_state=sc.trigger_sequence[0].get("state", "ON") if sc.trigger_sequence else "ON",
            actions=[{"type": "sendCommand", "item": aux, "command": "ON"}],
        )
    )
    return sc

def _hidden_eval_payload(sc: OpenHABScenario, spec: ScenarioSpec) -> dict[str, Any]:
    gt = dict(sc.hidden_ground_truth or {})
    return {
        "scenario_id": sc.scenario_id,
        "expected_obligations": gt.get("desired_item_states", {}),
        "conflict_relations": {
            "conflict_type": sc.conflict_type,
            "conflict_items": gt.get("conflict_items", []),
            "missing_commands": gt.get("missing_commands", []),
        },
        "acceptable_final_behaviors": gt.get("desired_item_states", {}),
        "scenario_intent": {
            "rationale": gt.get("rationale", ""),
            "scenario_type": spec.scenario_type,
            "scenario_family": spec.family,
            "complexity": spec.complexity,
            "unfixable": gt.get("unfixable", False),
            "adversarial": gt.get("adversarial", False),
        },
    }

def _write_public_artifacts(sc: OpenHABScenario, public_dir: Path) -> None:
    root = public_dir / sc.scenario_id / "public"
    scenario_to_openhab_files(sc, root)
    runtime_dir = root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    initial = {i.name: str(i.initial_state or "OFF").upper() for i in sc.items}
    (runtime_dir / "initial_state.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    events_dir = root / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    (events_dir / "event_sequence.json").write_text(
        json.dumps(sc.trigger_sequence, indent=2), encoding="utf-8"
    )

def _load_scenario_from_public(public_dir: Path, scenario_id: str) -> OpenHABScenario:
    manifest_path = public_dir.parent.parent / "manifests" / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("scenarios", []):
            if row["scenario_id"] == scenario_id:
                return _reconstruct_from_manifest_row(row, public_dir, scenario_id)
    raise FileNotFoundError(scenario_id)

def _reconstruct_from_manifest_row(row: dict, public_dir: Path, scenario_id: str) -> OpenHABScenario:
    from smarthome_mdf.openhab_validation.openhab_dsl_parser import parse_openhab_directory

    pub = public_dir / scenario_id / "public"
    model = parse_openhab_directory(pub)
    events = json.loads((pub / "events" / "event_sequence.json").read_text(encoding="utf-8"))
    initial = json.loads((pub / "runtime" / "initial_state.json").read_text(encoding="utf-8"))
    items = []
    for name, sem in model.items.items():
        items.append(
            OpenHABItem(
                name=name,
                item_type=sem.get("item_type", "Switch"),
                label=sem.get("label", ""),
                group=sem.get("group", ""),
                initial_state=initial.get(name),
            )
        )
    rules = []
    for uid, rule in model.rules.items():
        actions = rule.actions
        rules.append(
            OpenHABRule(
                uid=uid,
                name=rule.name,
                component_id=row.get("scenario_id", scenario_id).split("_")[2] if False else f"comp_{uid}",
                scene_type=row.get("scenario_family", "advanced_lighting"),
                trigger_item=rule.trigger.get("item", ""),
                trigger_state=rule.trigger.get("state", ""),
                actions=actions,
            )
        )
    return OpenHABScenario(
        scenario_id=scenario_id,
        seed=int(row.get("seed", 0)),
        conflict_type=row.get("conflict_type", "opposing_light"),
        items=items,
        rules=rules,
        trigger_sequence=events,
        hidden_ground_truth={},
        metadata={"scenario_family": row.get("scenario_family"), "scenario_type": row.get("scenario_type")},
    )

class FinalBenchmarkGenerator:
    def __init__(self, *, seed: int = GENERATOR_SEED, out_dir: Path | None = None) -> None:
        self.seed = seed
        self.out_dir = out_dir or DEFAULT_OUT
        self.gen = OpenHABScenarioGenerator(seed=seed)

    def generate(self, n: int = 1200) -> list[OpenHABScenario]:
        specs = _build_specs(self.seed)[:n]
        scenarios: list[OpenHABScenario] = []
        for spec in specs:
            sc = _generate_for_spec(self.gen, spec)
            sc.metadata = dict(sc.metadata or {})
            sc.metadata.update(
                {
                    "scenario_type": spec.scenario_type,
                    "complexity": spec.complexity,
                    "structural_tags": _structural_tags(sc),
                    "generator_version": GENERATOR_VERSION,
                }
            )
            scenarios.append(sc)
        return scenarios

    def save(self, scenarios: list[OpenHABScenario], *, out_dir: Path | None = None) -> Path:
        root = out_dir or self.out_dir
        public_root = root / "public"
        hidden_root = root / "hidden_eval"
        manifests = root / "manifests"
        splits = root / "splits"
        for d in (public_root, hidden_root, manifests, splits):
            d.mkdir(parents=True, exist_ok=True)

        index: list[dict[str, Any]] = []
        for i, sc in enumerate(scenarios):
            spec_type = (sc.metadata or {}).get("scenario_type", "repairable")
            spec = ScenarioSpec(spec_type, (sc.metadata or {}).get("complexity", 1), (sc.metadata or {}).get("scenario_family", "lighting"), i)
            _write_public_artifacts(sc, public_root)
            hidden = _hidden_eval_payload(sc, spec)
            hid_dir = hidden_root / sc.scenario_id
            hid_dir.mkdir(parents=True, exist_ok=True)
            (hid_dir / "expected_obligations.json").write_text(
                json.dumps({"obligations": hidden["expected_obligations"]}, indent=2), encoding="utf-8"
            )
            (hid_dir / "conflict_relations.json").write_text(json.dumps(hidden["conflict_relations"], indent=2), encoding="utf-8")
            (hid_dir / "acceptable_final_behaviors.json").write_text(
                json.dumps(hidden["acceptable_final_behaviors"], indent=2), encoding="utf-8"
            )
            (hid_dir / "scenario_intent.json").write_text(json.dumps(hidden["scenario_intent"], indent=2), encoding="utf-8")
            index.append(
                {
                    "scenario_id": sc.scenario_id,
                    "generator_version": GENERATOR_VERSION,
                    "generator_seed": self.seed,
                    "scenario_family": (sc.metadata or {}).get("scenario_family"),
                    "scenario_type": spec_type,
                    "conflict_type": sc.conflict_type,
                    "complexity": (sc.metadata or {}).get("complexity"),
                    "rule_count": len(sc.rules),
                    "item_count": len(sc.items),
                    "structural_tags": (sc.metadata or {}).get("structural_tags", []),
                    "seed": sc.seed,
                }
            )

        manifest = {
            "generator_version": GENERATOR_VERSION,
            "generator_seed": self.seed,
            "n_scenarios": len(scenarios),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scenarios": index,
        }
        (manifests / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        stats_rows = index
        with (manifests / "dataset_statistics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            w.writeheader()
            w.writerows(stats_rows)

        hashes: dict[str, str] = {}
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".items", ".rules", ".csv"):
                rel = str(p.relative_to(root))
                hashes[rel] = _sha256_file(p)
        (manifests / "dataset_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

        self._write_live_split(index, splits)
        return root

    def _write_live_split(self, index: list[dict], splits: Path) -> None:
        rng = random.Random(self.seed + 999)
        live_ids: list[str] = []
        for stype, quota in {"repairable": 288, "clean": 72, "ambiguous": 72, "unsupported": 48}.items():
            pool = [r for r in index if r["scenario_type"] == stype]
            rng.shuffle(pool)
            live_ids.extend(r["scenario_id"] for r in pool[:quota])
        for c, quota in {1: 120, 2: 144, 3: 120, 4: 96}.items():
            pool = [r for r in index if r["complexity"] == c and r["scenario_id"] not in live_ids]
            rng.shuffle(pool)
            for r in pool[: max(0, quota - sum(1 for i in index if i["scenario_id"] in live_ids and i["complexity"] == c))]:
                if r["scenario_id"] not in live_ids:
                    live_ids.append(r["scenario_id"])
        live_ids = sorted(set(live_ids))[:480]
        payload = {"seed": self.seed + 999, "n_live": len(live_ids), "scenario_ids": live_ids}
        (splits / "live_split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (splits / "live_split.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scenario_id"])
            for sid in live_ids:
                w.writerow([sid])

    @staticmethod
    def load_corpus(root: Path) -> list[OpenHABScenario]:
        manifest = json.loads((root / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
        gen = OpenHABScenarioGenerator(seed=int(manifest.get("generator_seed", GENERATOR_SEED)))
        out: list[OpenHABScenario] = []
        for row in manifest["scenarios"]:
            ctype = row["conflict_type"]
            sc = gen.generate_one(row["scenario_id"], conflict_type=ctype, seed=int(row.get("seed", 0)))
            sc.metadata = {
                "scenario_type": row["scenario_type"],
                "scenario_family": row["scenario_family"],
                "complexity": row["complexity"],
                "structural_tags": row.get("structural_tags", []),
            }
            out.append(sc)
        return out

def generate_final_benchmark(out_dir: Path | None = None, n: int = 1200) -> Path:
    g = FinalBenchmarkGenerator(out_dir=out_dir)
    scenarios = g.generate(n)
    return g.save(scenarios, out_dir=out_dir or DEFAULT_OUT)
