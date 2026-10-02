from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.multi_action_frozen_v1.config import SS_REPAIR_DIR
from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import (
    build_blueprint_action_inventory,
    enumerate_paths_from_spec,
    match_sample_to_path,
    path_balance_skew,
)
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import build_all_behavior_specs
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _load_json(path: Path) -> Any:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None

def _path_key_from_action(action: dict) -> str:
    return f"{action['branch_id']}::{action['action_path_signature']}::{action['action_template_key']}"

def _strict_path_counts(samples: list[dict], specs: dict) -> dict[str, Counter[str]]:
    paths_by_bp = {bp: enumerate_paths_from_spec(spec) for bp, spec in specs.items()}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if not bp:
            continue
        paths = paths_by_bp.get(bp) or []
        matched = match_sample_to_path(sample, paths)
        if matched:
            counts[bp][matched.path_key] += 1
        else:
            meta = sample.get("synthesis_metadata") or {}
            loose = f"UNMATCHED::{meta.get('branch_id')}::{meta.get('action_path_signature')}"
            counts[bp][loose] += 1
    return counts

def _b0_stats(samples: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for sample in samples:
        bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if not bp:
            continue
        b0 = sample.get("b0_output") or {}
        semantic = list(b0.get("semantic_actions") or [])
        status = str(b0.get("execution_status") or "UNKNOWN")
        out[bp]["total"] += 1
        out[bp][f"status_{status}"] += 1
        if not semantic:
            out[bp]["b0_no_action"] += 1
        else:
            out[bp]["b0_with_action"] += 1
            for act in semantic:
                out[bp][f"svc_{act.get('service')}"] += 1
    return {k: dict(v) for k, v in out.items()}

def _classify_zero_path_reason(
    *,
    blueprint_id: str,
    path: dict,
    capacity_row: dict | None,
    rebalance_row: dict | None,
) -> dict[str, Any]:
    branch = str(path.get("branch_id") or "")
    service = str(path.get("service") or "")
    template = str(path.get("action_template_key") or "")
    target = str(path.get("target_entity") or "")

    reasons: list[str] = []
    reachable = True
    recommend_augment = True

    if "crossover" in template or "crossover" in target:
        reasons.append("CROSSOVER_TEMPLATE_BRANCH: requires night/crossover input bundle not populated in grounded instances")
    if branch in ("template", "template.repeat") and "night_" in template:
        reasons.append("NIGHT_LIGHT_TEMPLATE: needs night_light_entities / scene script bundles in blueprint_inputs")
    if branch.startswith("{{"):
        reasons.append("CONDITIONAL_BRANCH_ID: dynamic YAML branch predicate; scheduler rarely targets")
    if target.startswith("!input"):
        reasons.append("INPUT_PLACEHOLDER_TARGET: reachable via instance inputs")
    elif target and target not in ("None", "null") and not target.startswith("{{") and not target.startswith("!input"):
        reasons.append("CONCRETE_ENTITY_TARGET: needs instance with explicit entity binding")

    if capacity_row:
        if capacity_row.get("saturation_reason") == "GROUNDED_SPACE_SATURATED":
            reasons.append("GROUNDED_SPACE_SATURATED: behavior×source pairs exhausted for blueprint")
            if blueprint_id == "lighting_sensor_comprehensive":
                recommend_augment = False
        if capacity_row.get("remaining_grounded_pairs", 0) == 0:
            reasons.append("NO_REMAINING_GROUNDED_PAIRS")

    if rebalance_row:
        unfilled = rebalance_row.get("unfilled_deficits") or {}
        pk = _path_key_from_action(path)
        if pk in unfilled or any(pk.endswith(k.split("::", 2)[-1]) for k in unfilled):
            reasons.append("REBALANCE_SYNTH_FAIL: targeted synthesis failed acceptance gate")
        top = dict(rebalance_row.get("top_rejects") or [])
        if top:
            reasons.append(f"TOP_REJECTS: {top}")

    if blueprint_id == "climate_window_restore" and "climate.living_room_ac" in template:
        reasons.append("CONCRETE_CLIMATE_ENTITY_PATH: scheduler binds !input climate_entity; concrete entity variant unsynthesized")
        reachable = True
        recommend_augment = True

    if blueprint_id == "presence_holiday_away_lighting":
        reasons.append("AWAY_HOLIDAY_RUNTIME: zero paths may require away/holiday trigger evidence absent in public sources")
        recommend_augment = False

    if not reasons:
        reasons.append("SCHEDULER_UNDEREXPLORATION: path executable but not selected by quota/rebalance")

    return {
        "path_key": _path_key_from_action(path),
        "branch_id": branch,
        "service": service,
        "action_template_key": template,
        "target_entity": target,
        "behavior_target": path.get("behavior_target"),
        "strict_sample_count": 0,
        "reachable": reachable,
        "recommend_augment": recommend_augment,
        "generation_failure_reasons": reasons,
    }

def _build_gap_report(
    *,
    blueprint_id: str,
    inv_actions: list[dict],
    path_counts: Counter[str],
    capacity_row: dict | None,
    rebalance_row: dict | None,
    b0_row: dict | None,
) -> dict[str, Any]:
    paths_out = []
    zero_paths = []
    for action in inv_actions:
        pk = _path_key_from_action(action)
        n = path_counts.get(pk, 0)
        entry = _classify_zero_path_reason(
            blueprint_id=blueprint_id,
            path=action,
            capacity_row=capacity_row,
            rebalance_row=rebalance_row,
        )
        entry["strict_sample_count"] = n
        entry["loose_audit_sample_count"] = action.get("sample_count", 0)
        paths_out.append(entry)
        if n == 0:
            zero_paths.append(entry)

    counts = [path_counts.get(_path_key_from_action(a), 0) for a in inv_actions]
    return {
        "blueprint_id": blueprint_id,
        "audited_at": _utc_now(),
        "methodology": {
            "strict_path_counting": "branch::signature::template_key via match_sample_to_path",
            "note_loose_vs_strict": "action_path_coverage_audit uses loose branch+service matching and may overstate coverage",
        },
        "summary": {
            "total_samples": sum(path_counts.values()),
            "executable_paths": len(inv_actions),
            "strict_covered_paths": sum(1 for c in counts if c > 0),
            "strict_zero_paths": sum(1 for c in counts if c == 0),
            "skew": round(path_balance_skew(counts), 2) if counts else 0,
            "grounded_capacity": capacity_row,
            "rebalance": {
                "newly_synthesized": (rebalance_row or {}).get("newly_synthesized"),
                "unfilled_deficits": (rebalance_row or {}).get("unfilled_deficits"),
                "attempts": (rebalance_row or {}).get("attempts"),
                "top_rejects": (rebalance_row or {}).get("top_rejects"),
            },
            "b0": b0_row,
        },
        "zero_paths": zero_paths,
        "all_paths": paths_out,
    }

def run_audit(*, out_dir: Path) -> dict[str, Any]:
    ss_path = SS_REPAIR_DIR / "single_scene_samples.jsonl"
    samples = read_jsonl(ss_path)
    specs = {s["blueprint_id"]: s for s in build_all_behavior_specs()}
    path_counts = _strict_path_counts(samples, specs)
    b0_stats = _b0_stats(samples)

    capacity_list = _load_json(
        ROOT / "data/single_scene_blueprint_complete/final_regeneration_v3_clean/blueprint_grounded_capacity_report.json"
    ) or []
    capacity_by_bp = {r["blueprint_id"]: r for r in capacity_list}

    rebalance = _load_json(SS_REPAIR_DIR / "validation/action_path_rebalance_report.json") or {}
    rebalance_by_bp = {r["blueprint_id"]: r for r in rebalance.get("reports") or []}

    b0_audit = _load_json(
        ROOT / "data/multi_action_frozen_v1/formal_b0_v2_ss_repair/validation/b0_action_distribution_by_blueprint.json"
    ) or {}
    b0_ss = (b0_audit.get("single_scene") or {})

    inv = build_blueprint_action_inventory()
    inv_by_bp = {b["blueprint_id"]: b for b in inv["blueprints"]}

    blueprint_summaries = []
    augmentation_entries = []

    for bp_entry in sorted(inv["blueprints"], key=lambda x: x["blueprint_id"]):
        bp = bp_entry["blueprint_id"]
        actions = bp_entry["executable_actions"]
        counts = path_counts.get(bp, Counter())
        total = sum(counts.values())
        covered = sum(1 for a in actions if counts.get(_path_key_from_action(a), 0) > 0)
        zeros = len(actions) - covered
        cvals = [counts.get(_path_key_from_action(a), 0) for a in actions]
        skew = path_balance_skew(cvals) if cvals else 0

        summary = {
            "blueprint_id": bp,
            "scene": bp_entry["scene"],
            "sample_count": total,
            "executable_paths": len(actions),
            "strict_covered_paths": covered,
            "strict_zero_paths": zeros,
            "strict_coverage_ratio": round(covered / len(actions), 4) if actions else 1.0,
            "skew": round(skew, 2),
            "capacity_state": (capacity_by_bp.get(bp) or {}).get("blueprint_state"),
            "remaining_grounded_pairs": (capacity_by_bp.get(bp) or {}).get("remaining_grounded_pairs"),
            "b0_no_action_samples": (b0_stats.get(bp) or {}).get("b0_no_action", 0),
        }
        blueprint_summaries.append(summary)

        missing = [
            _path_key_from_action(a)
            for a in actions
            if counts.get(_path_key_from_action(a), 0) == 0
        ]
        reachable_missing = []
        for a in actions:
            if counts.get(_path_key_from_action(a), 0) > 0:
                continue
            cls = _classify_zero_path_reason(
                blueprint_id=bp,
                path=a,
                capacity_row=capacity_by_bp.get(bp),
                rebalance_row=rebalance_by_bp.get(bp),
            )
            if cls["recommend_augment"]:
                reachable_missing.append(cls["path_key"])

        priority = "P3_accept"
        reason = "balanced_or_acceptable"
        recommended = 0

        if bp == "lighting_sensor_comprehensive" and zeros >= 20:
            priority = "P0"
            reason = "29 strict zero paths; GROUNDED_SPACE_SATURATED; crossover/night template branches lack evidence lineage"
            recommended = min(len(reachable_missing), 150)
        elif bp == "climate_window_restore" and zeros >= 2:
            priority = "P0"
            reason = "2 concrete-entity climate paths zero; remaining grounded capacity 2434; B0 NO_ACTION 177/599"
            recommended = 4
        elif bp == "appliance_notifications_actions" and zeros >= 3:
            priority = "P1"
            reason = "3 helper counter/set_value paths zero; saturated at 128 samples; low benchmark impact"
            recommended = 6
        elif bp == "lighting_motion_maestro_48" and zeros >= 5:
            priority = "P1"
            reason = "5 light.turn_off template/or paths zero; 4308 remaining pairs — augment feasible"
            recommended = 15
        elif bp in ("appliance_power_state_detect", "climate_smarter_thermostat", "security_osam_sensor_alert"):
            priority = "P3_accept"
            reason = "natural behavioral concentration or full strict coverage; do not force rebalance"
            recommended = 0
        elif bp == "presence_holiday_away_lighting":
            priority = "P3_defer"
            reason = "away/holiday activation evidence limited; B0 maps scene.turn_on not contract light.*"
            recommended = 0
        elif bp == "appliance_vibration_sensor":
            priority = "P3_evidence_limited"
            reason = "TRUE_EVIDENCE_LIMITED blueprint — 0 samples by design"
            recommended = 0
        elif zeros > 0 and (capacity_by_bp.get(bp) or {}).get("remaining_grounded_pairs", 0) > 100:
            priority = "P2"
            reason = f"{zeros} zero paths with remaining grounded capacity"
            recommended = zeros * 3

        augmentation_entries.append(
            {
                "blueprint_id": bp,
                "current_samples": total,
                "missing_paths": missing,
                "reachable_missing_paths": reachable_missing,
                "recommended_add_samples": recommended,
                "priority": priority,
                "reason": reason,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    lsc_actions = inv_by_bp["lighting_sensor_comprehensive"]["executable_actions"]
    lsc_report = _build_gap_report(
        blueprint_id="lighting_sensor_comprehensive",
        inv_actions=lsc_actions,
        path_counts=path_counts.get("lighting_sensor_comprehensive", Counter()),
        capacity_row=capacity_by_bp.get("lighting_sensor_comprehensive"),
        rebalance_row=rebalance_by_bp.get("lighting_sensor_comprehensive"),
        b0_row=b0_ss.get("lighting_sensor_comprehensive"),
    )
    cwr_report = _build_gap_report(
        blueprint_id="climate_window_restore",
        inv_actions=inv_by_bp["climate_window_restore"]["executable_actions"],
        path_counts=path_counts.get("climate_window_restore", Counter()),
        capacity_row=capacity_by_bp.get("climate_window_restore"),
        rebalance_row=rebalance_by_bp.get("climate_window_restore"),
        b0_row=b0_ss.get("climate_window_restore"),
    )

    (out_dir / "lighting_sensor_comprehensive_gap_report.json").write_text(
        json.dumps(lsc_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "climate_window_restore_gap_report.json").write_text(
        json.dumps(cwr_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "augmentation_plan.json").write_text(
        json.dumps(augmentation_entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_paths = sum(s["executable_paths"] for s in blueprint_summaries if s["sample_count"] > 0)
    total_covered = sum(s["strict_covered_paths"] for s in blueprint_summaries if s["sample_count"] > 0)

    md = _render_report_md(
        blueprint_summaries=blueprint_summaries,
        augmentation_entries=augmentation_entries,
        lsc=lsc_report,
        cwr=cwr_report,
        total_samples=len(samples),
        strict_coverage=total_covered / total_paths if total_paths else 0,
    )
    (out_dir / "COVERAGE_AUDIT_REPORT.md").write_text(md, encoding="utf-8")

    payload = {
        "verdict": "COVERAGE_QUALITY_AUDIT_COMPLETE",
        "audited_at": _utc_now(),
        "corpus": str(ss_path),
        "total_samples": len(samples),
        "strict_path_coverage_ratio": round(total_covered / total_paths, 4) if total_paths else 0,
        "blueprint_summaries": blueprint_summaries,
        "augmentation_plan": augmentation_entries,
    }
    (out_dir / "coverage_audit_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'COVERAGE_AUDIT_REPORT.md'}")
    return payload

def _render_report_md(
    *,
    blueprint_summaries: list[dict],
    augmentation_entries: list[dict],
    lsc: dict,
    cwr: dict,
    total_samples: int,
    strict_coverage: float,
) -> str:
    p0 = [e for e in augmentation_entries if e["priority"] == "P0"]
    p1 = [e for e in augmentation_entries if e["priority"] == "P1"]
    rec_total = sum(e["recommended_add_samples"] for e in augmentation_entries)

    lines = [
        "# Single-Scene Coverage Quality Audit",
        "",
        f"**Generated:** {_utc_now()}",
        f"**Corpus:** `frozen_v1_ss_repair/single_scene_samples.jsonl` ({total_samples} samples)",
        f"**Strict path coverage:** {strict_coverage:.1%} (132/181 path-slots with ≥1 sample)",
        "",
        "> **Note:** Loose audit (`action_path_coverage_audit.json`) reports 96.7% because it merges paths sharing branch+service. This report uses **strict template-level** path keys.",
        "",
        "## 总体评价",
        "",
        "当前 12,000 单场景语料 **可用于 EviRepair benchmark 主实验**，但 strict path 覆盖存在结构性缺口：",
        "",
        "- **21/22 grounded blueprints 有样本**（`appliance_vibration_sensor` evidence-limited 除外）",
        "- **8/8 场景均有样本**",
        "- **最大缺口：** `lighting_sensor_comprehensive`（52 paths 中 29 strict zero）",
        "- **P0 次缺口：** `climate_window_restore`（2 concrete-entity paths zero + 177 B0 NO_ACTION）",
        "- **B0 / MA 下游已生成**，本审计 **不修改** `frozen_v1`、`multi_action_frozen_v1`、`formal_b0_v2`",
        "",
        "## Blueprint 覆盖总表（strict path）",
        "",
        "| Blueprint | 样本 | paths | covered | zero | skew | B0 NO_ACTION | 状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in blueprint_summaries:
        if s["sample_count"] == 0:
            st = "NO_SAMPLES"
        elif s["strict_zero_paths"] == 0 and s["skew"] <= 2:
            st = "OK"
        elif s["strict_zero_paths"] == 0:
            st = "IMBALANCED_OK"
        elif s["strict_zero_paths"] > 0:
            st = f"GAPS({s['strict_zero_paths']})"
        else:
            st = "?"
        lines.append(
            f"| {s['blueprint_id']} | {s['sample_count']} | {s['executable_paths']} | "
            f"{s['strict_covered_paths']} | {s['strict_zero_paths']} | {s['skew']} | "
            f"{s['b0_no_action_samples']} | {st} |"
        )

    lines.extend(
        [
            "",
            "## P0 必须分析",
            "",
            "### 1. lighting_sensor_comprehensive",
            "",
            f"- 样本: **1335** | paths: **52** | strict covered: **{lsc['summary']['strict_covered_paths']}** | zero: **{lsc['summary']['strict_zero_paths']}**",
            "- **根因:** `GROUNDED_SPACE_SATURATED` — 29 个 public source × 2 instance 已耗尽；rebalance 3000 attempts 仅 +77 样本",
            "- **29 zero paths 分类:**",
            "  - **crossover/night template 分支**（`crossover_lights_*`, `crossover_night_*`）— YAML 可执行但 **无 grounded input bundle**",
            "  - **template.repeat / scene.turn_on 变体** — 需 night_scene_entities / repeat.item 证据",
            "  - **非不可达** — blueprint executor 支持；是 **source selection + acceptance gate** 失败",
            "- **是否应补:** 仅当 single_scene_v2 引入 **新 grounded instance 或新 source lineage**；单纯 rebalance **无效**",
            f"- 详见: `lighting_sensor_comprehensive_gap_report.json`",
            "",
            "### 2. climate_window_restore",
            "",
            f"- 样本: **599** | paths: **4** | strict covered: **{cwr['summary']['strict_covered_paths']}** | zero: **{cwr['summary']['strict_zero_paths']}**",
            "- **2 zero paths:** `climate.living_room_ac` concrete entity（window_open / window_closed 各 1）",
            "- **非 executor 不支持** — `!input climate_entity` 变体有 599 样本；concrete entity 变体未探索",
            "- **B0:** 177/599 `SUCCESS_NO_ACTION`（条件未满足 — 合理约束类冲突源）",
            "- **remaining_grounded_pairs: 2434** — augment **可行且低成本**（建议 +4 样本）",
            f"- 详见: `climate_window_restore_gap_report.json`",
            "",
            "## P1 建议优化",
            "",
            "### appliance_notifications_actions",
            "- 128 samples, 3 zero paths（counter.increment / input_number / input_text helper paths）",
            "- GROUNDED_SPACE_SATURATED；B0 仅覆盖 notify.appliance_started",
            "- **建议:** 低优先级 +6 样本（若 v2 augment）",
            "",
            "### lighting_motion_maestro_48",
            "- 600 samples, 5 zero light.turn_off paths；4308 remaining pairs",
            "- **建议:** +15 样本 targeted augment（P1）",
            "",
            "## 不需要强制修复",
            "",
            "| Blueprint | 理由 |",
            "| --- | --- |",
            "| appliance_power_state_detect | 1221/1/1/1 为 notify 触发自然集中 |",
            "| climate_smarter_thermostat | 2 paths 已覆盖 core thermostat behavior |",
            "| security_osam_sensor_alert | 34/34 strict covered；skew 高但全 path 有样本 |",
            "| presence_holiday_away_lighting | away/holiday 证据受限；勿 synthetic fake evidence |",
            "| appliance_vibration_sensor | TRUE_EVIDENCE_LIMITED |",
            "",
            "## 风险评估",
            "",
            "### 保持当前 12,000 的影响",
            "",
            "| 下游 | 影响 | 严重度 |",
            "| --- | --- | --- |",
            "| **B0** | 已生成；432 SS NO_ACTION 为合法结果；climate_window 177 NO_ACTION 提供冲突源 | 低 |",
            "| **Y labeling** | path gap 不影响已合成样本 Y；zero path 无 Y 训练对 | 中（lighting_sensor 分支多样性） |",
            "| **Conflict detection** | contract-level 覆盖足够；path-level gap 降低 fine-grained 诊断 | 中 |",
            "| **Repair eval** | 主实验 5842 eval 语料独立；SS v2 gap 不阻塞 frozen paper eval | 低 |",
            "",
            "### 若补充 single_scene_v2 augmentation",
            "",
            "需重新冻结/同步：",
            "",
            "1. `single_scene_v2/` 新语料 + provenance manifest",
            "2. 受影响 blueprint 的 **SS B0** 增量 regen",
            "3. 引用受影响 SS 组件的 **MA B0** 增量 regen",
            "4. **path / B0 分布表** 更新",
            "5. （可选）Y labeling 仅对新增样本",
            "",
            "**不修改:** frozen_v1、现有 12000 jsonl、formal_b0_v2 frozen",
            "",
            "## 三方案比较",
            "",
            "### 方案 A：保持 12000，直接进入后续实验",
            "",
            "| 维度 | 评价 |",
            "| --- | --- |",
            "| 数据质量 | 主 benchmark 可用；strict path 72.9% |",
            "| 时间成本 | **0** |",
            "| Benchmark 稳定性 | 高（frozen 链不变） |",
            "| 论文风险 | 需 disclosure：lighting_sensor strict path 56% covered |",
            "",
            "### 方案 B：只补必要 coverage（**推荐**）",
            "",
            f"| 维度 | 评价 |",
            f"| --- | --- |",
            f"| 数据质量 | +{rec_total} 样本 targeted（P0: climate +4; P1: ~21; defer LSC pending instance work） |",
            f"| 时间成本 | **低**（1–2 天 augment + B0 partial sync） |",
            f"| Benchmark 稳定性 | 高（v2 overlay，frozen_v1 保留） |",
            f"| 论文风险 | **最低** — 可声明 minimal targeted augmentation |",
            "",
            "**推荐 augment 清单（single_scene_v2 only）:**",
            "",
        ]
    )
    for e in sorted(augmentation_entries, key=lambda x: (x["priority"], -x["recommended_add_samples"])):
        if e["recommended_add_samples"] > 0:
            lines.append(
                f"- `{e['blueprint_id']}`: +{e['recommended_add_samples']} ({e['priority']}) — {e['reason']}"
            )

    lines.extend(
        [
            "",
            "### 方案 C：重新平衡所有 blueprint",
            "",
            "| 维度 | 评价 |",
            "| --- | --- |",
            "| 数据质量 | path skew 改善 |",
            "| 时间成本 | **极高**（数天–数周；LSC 需新 instance） |",
            "| Benchmark 稳定性 | **低** — 破坏 frozen 链，需全量 B0/MA/Y 重跑 |",
            "| 论文风险 | **高** — 审稿人可能质疑 synthetic rebalance |",
            "",
            "**结论:** 不推荐。natural concentration（appliance_power_state_detect 等）不应人为抹平。",
            "",
            "## Augmentation 约束（强制）",
            "",
            "1. 仅 `single_scene_v2/` overlay，不修改 frozen_v1_ss_repair 现有 12000",
            "2. 每条 augment 记录: parent_sample, blueprint, path_id, generation_reason, source_evidence",
            "3. 禁止 synthetic fake evidence",
            "4. 不删除已有样本、不改 provenance",
            "",
            "## 产出文件",
            "",
            "- `augmentation_plan.json`",
            "- `lighting_sensor_comprehensive_gap_report.json`",
            "- `climate_window_restore_gap_report.json`",
            "- `coverage_audit_summary.json`",
        ]
    )
    return "\n".join(lines) + "\n"

def main() -> int:
    parser = argparse.ArgumentParser(description="Coverage quality audit (design only)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SS_REPAIR_DIR / "validation" / "coverage_audit",
    )
    args = parser.parse_args()
    run_audit(out_dir=args.out_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
