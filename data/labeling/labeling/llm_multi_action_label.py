from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from smarthome_mdf.decision_contract import (
    DecisionContractError,
    build_decision_output,
    normalize_decision_output,
    validate_decision_output_contract,
)
from smarthome_mdf.labeling.llm_auto_label import (
    LLM_API_KEY,
    LLM_API_URL,
    LLM_MAX_PER_MINUTE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REPETITION_PENALTY,
    LLM_SAMPLE_MAX_ATTEMPTS,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
    LlmClient,
    _retry_sleep_seconds,
    print_failed_sample_ids,
)
from smarthome_mdf.labeling.blueprint_constraints import has_out_of_contract_actions
from smarthome_mdf.paths import MULTI_ACTION_SAMPLES, MULTI_QUALITY_REPORT_DIR
from smarthome_mdf.rules.blueprint_action_contract import get_contract, is_action_allowed, validate_y_annotation

from smarthome_mdf.synthesis.multi_action.vocab import load_vocab

MULTI_FILE = MULTI_ACTION_SAMPLES

SYSTEM_PROMPT = """You validate multi-action smart home decision labels (rule-generated).
Evidence is from public datasets only. Do not invent sensor values or new actions.

## Action name format (critical)
- Output expected_action and blocked_action using ONLY Home Assistant service names from candidate_actions_ha
  (e.g. light.turn_on, notify.mobile_app, llmvision.analyze_image).
- candidate_actions in the user JSON are abstract internal names (e.g. turn_on_light) for reference only.
- NEVER reject because HA names do not appear in candidate_actions. Compare ONLY against candidate_actions_ha.

## Action Contract (multi_action_composite)
- Output expected_action and blocked_action using ONLY actions from contract_allowed_actions in user JSON.
- contract_allowed_actions is the whitelist; candidate_actions_ha must be filtered to this set.
- If an action is in candidate_actions_ha but NOT in contract_allowed_actions, do NOT output it.

## Valid multi-action (approve when all hold)
1. expected_action has >= 2 actions, each in contract_allowed_actions (and candidate_actions_ha).
2. Those actions map to >= 2 distinct device groups (use ha_device_group in user JSON;
   lighting, climate, appliance, notification, visual are distinct groups).
3. No mutually exclusive pair both in expected_action (e.g. turn_on_ac + turn_off_ac).
4. Each expected action is supported by evidence and/or action_support rules (plausible given evidence).
5. record_event / llmvision.analyze_image: if listed in candidate_actions_ha and device_group "visual"
   is distinct from other expected actions' groups, it MAY stay in expected_action (do not reject solely for a third action).

## blocked_action (legal partial conflict)
- blocked_action is valid when subset of candidate_actions_ha and disjoint from expected_action.
- Example: expected [light.turn_on, notify.mobile_app] + blocked [climate.turn_on] with conflict_type
  partial_constraint_conflict is APPROVED if evidence supports window open blocking AC while light+notify execute.
- Do not reject the whole sample only because some candidates are blocked; blocked actions do not count toward the >=2 expected count.

## Reject only when
- expected_action has < 2 actions after restricting to candidate_actions_ha, OR
- < 2 device groups among expected actions, OR
- mutual exclusion inside expected_action, OR
- an expected HA action is NOT in candidate_actions_ha, OR
- expected actions clearly contradict evidence with no rule support.

Output ONLY JSON:
{
  "valid_multi_action": true,
  "expected_action": [],
  "blocked_action": [],
  "conflict_type": "none",
  "multi_action_type": "",
  "label_reason": "",
  "reject_reason": ""
}
If invalid: valid_multi_action=false, reject_reason (one short English sentence)."""

def _ha_device_group_map(candidate_ha: List[str]) -> Dict[str, str]:
    vocab, _, _ = load_vocab()
    ha_to_group: Dict[str, str] = {}
    for _abs, meta in vocab.items():
        ha = meta.get("ha_action")
        if ha and ha in candidate_ha:
            ha_to_group[ha] = meta.get("device_group", "")
    return ha_to_group

def _contract_allowed_actions() -> List[str]:
    contract = get_contract("multi_action_composite") or {}
    return list(contract.get("allowed_actions") or [])

def _filter_to_contract(actions: List[str], contract: dict) -> List[str]:
    return [a for a in actions if is_action_allowed(a, contract)]

def _build_user_prompt(sample: dict) -> str:
    dec = sample.get("decision_output", {})
    cand_ha = sample.get("candidate_actions_ha") or []
    contract = get_contract("multi_action_composite") or {}
    contract_allowed = _contract_allowed_actions()
    contract_domains = list(contract.get("action_domains") or [])
    cand_ha_contract = [a for a in cand_ha if is_action_allowed(a, contract)]
    return json.dumps(
        {
            "combo_id": sample.get("metadata", {}).get("combo_id", ""),
            "multi_action_type": sample.get("multi_action_type", ""),
            "has_conflict": sample.get("has_conflict", False),
            "evidence": sample.get("evidence", {}),
            "contract_allowed_actions": contract_allowed,
            "contract_action_domains": contract_domains,
            "candidate_actions_ha": cand_ha_contract,
            "candidate_actions_ha_full": cand_ha,
            "ha_device_group": _ha_device_group_map(cand_ha_contract),
            "candidate_actions_abstract_reference_only": sample.get("candidate_actions", []),
            "action_device_group": sample.get("action_device_group", {}),
            "rule_evidence_map": sample.get("rule_evidence_map", {}),
            "action_support": sample.get("action_support", {}),
            "current_expected_action_ha": dec.get("expected_action", []),
            "current_blocked_action_ha": dec.get("blocked_action", []),
            "current_conflict_type": dec.get("conflict_type", "none"),
            "validation_note": (
                "Validate current_*_ha fields; output expected_action/blocked_action as HA names "
                "subset of contract_allowed_actions only."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )

def _parse_llm_json(text: str) -> Optional[dict]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None

def _apply_llm_result(sample: dict, parsed: dict) -> str:

    dec = sample.setdefault("decision_output", {})
    contract = get_contract("multi_action_composite") or {}
    cand_ha = set(a for a in (sample.get("candidate_actions_ha") or []) if is_action_allowed(a, contract))
    if parsed.get("valid_multi_action"):
        exp = _filter_to_contract(
            [a for a in parsed.get("expected_action", []) if a in cand_ha],
            contract,
        )
        blk = _filter_to_contract(
            [a for a in parsed.get("blocked_action", []) if a in cand_ha],
            contract,
        )
        ann = {
            "expected_action": exp,
            "blocked_action": blk,
            "conflict_type": parsed.get("conflict_type", dec.get("conflict_type", "none")),
        }
        violations, ann = validate_y_annotation(ann, contract, sanitize=False)
        if has_out_of_contract_actions(violations):
            dec["label_status"] = "rejected"
            dec["reject_reason"] = f"outside_contract_allowed_actions: {violations}"
            return "rejected"
        if len(exp) >= 2:
            merged = build_decision_output(
                exp,
                blk,
                conflict_type=ann.get("conflict_type", "none"),
                reason=parsed.get("label_reason", dec.get("label_reason", "")),
                label_status="llm_checked",
            )
            dec.update(merged)
            dec["label_reason"] = parsed.get("label_reason", "")
            dec["labeling_policy"] = "blueprint_action_contract_v1"
            dec["action_domains"] = list(contract.get("action_domains") or [])
            dec.pop("llm_error", None)
            if parsed.get("multi_action_type"):
                sample["multi_action_type"] = parsed["multi_action_type"]
            return "llm_checked"
        dec["label_status"] = "rejected"
        dec["reject_reason"] = parsed.get("reject_reason", "LLM returned <2 valid actions")
        return "rejected"
    dec["label_status"] = "rejected"
    dec["reject_reason"] = parsed.get("reject_reason", "invalid_multi_action")
    return "rejected"

def _label_one_multi_sample(sample: dict, client: LlmClient) -> str:

    sid = sample.get("sample_id", "unknown")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(sample)},
    ]
    last_err: Optional[Exception] = None
    for attempt in range(1, LLM_SAMPLE_MAX_ATTEMPTS + 1):
        try:
            raw = client.chat(messages)
            parsed = _parse_llm_json(raw)
            if not parsed:
                raise ValueError("LLM response is not valid JSON")
            return _apply_llm_result(sample, parsed)
        except Exception as e:
            last_err = e
            if attempt >= LLM_SAMPLE_MAX_ATTEMPTS:
                break
            wait = _retry_sleep_seconds(e, attempt)
            print(
                f"    [{sid}] attempt {attempt}/{LLM_SAMPLE_MAX_ATTEMPTS} failed: {e} "
                f"→ sleep {wait:.0f}s retry",
                flush=True,
            )
            time.sleep(wait)

    dec = sample.setdefault("decision_output", {})
    dec["label_status"] = "rule"
    dec["llm_error"] = str(last_err)[:300]
    raise RuntimeError(f"sample {sid} failed after {LLM_SAMPLE_MAX_ATTEMPTS} attempts: {last_err}")

def _write_samples(path: Path, samples: List[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + ("\n" if samples else ""),
        encoding="utf-8",
    )

def label_file(
    *,
    path: Path = MULTI_FILE,
    max_samples: int = 0,
    dry_run: bool = False,
    only_pending: bool = True,
    only_rejected: bool = False,
    retry_ids: Optional[Set[str]] = None,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    mode = "only-rejected" if only_rejected else ("retry-ids" if retry_ids else ("pending" if only_pending else "all"))
    print(
        f"LLM multi-action labeling: {path} (mode={mode}, max_samples={max_samples or 'all'})",
        flush=True,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    samples = [json.loads(ln) for ln in lines if ln.strip()]
    print(f"Loaded {len(samples)} samples", flush=True)

    client: Optional[LlmClient] = None
    if LLM_API_KEY and not dry_run:
        client = LlmClient(
            LLM_API_KEY,
            LLM_MODEL,
            api_url=LLM_API_URL,
            max_per_minute=LLM_MAX_PER_MINUTE,
            sleep_sec=60.0 / LLM_MAX_PER_MINUTE,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            top_k=LLM_TOP_K,
            presence_penalty=0.0,
            repetition_penalty=LLM_REPETITION_PENALTY,
            json_mode=True,
        )
        print(f"Rate limit: max {LLM_MAX_PER_MINUTE} requests/minute", flush=True)

    stats: Dict[str, Any] = {
        "checked": 0,
        "llm_checked": 0,
        "rejected": 0,
        "skipped": 0,
        "failed": 0,
        "failed_sample_ids": [],
        "errors": [],
    }

    for i, sample in enumerate(samples):
        sid = sample.get("sample_id", "")
        if max_samples and stats["checked"] >= max_samples:
            continue

        dec = sample.setdefault("decision_output", {})
        status = dec.get("label_status", "rule")
        if retry_ids is not None:
            if sid not in retry_ids:
                stats["skipped"] += 1
                continue
        elif only_rejected:
            if status != "rejected":
                stats["skipped"] += 1
                continue
        elif only_pending and status not in ("rule", "pending"):
            stats["skipped"] += 1
            continue

        stats["checked"] += 1
        if stats["checked"] % 25 == 0:
            print(
                f"  [{stats['checked']}/{len(samples)}] llm_checked={stats['llm_checked']} "
                f"rejected={stats['rejected']} failed={stats['failed']}",
                flush=True,
            )

        if dry_run or client is None:
            stats["skipped"] += 1
            continue

        print(f"  [{i+1}/{len(samples)}] {sid} calling LLM...", flush=True)
        try:
            outcome = _label_one_multi_sample(sample, client)
            if outcome == "llm_checked":
                stats["llm_checked"] += 1
            else:
                stats["rejected"] += 1
            print(f"  [{i+1}/{len(samples)}] {sid} OK ({outcome})", flush=True)
        except Exception as e:
            stats["failed"] += 1
            if sid:
                stats["failed_sample_ids"].append(sid)
            stats["errors"].append({"sample_id": sid, "error": str(e)})
            print(f"  [{i+1}/{len(samples)}] SKIP {sid}: {e}", flush=True)

        if not dry_run and stats["checked"] % 10 == 0:
            _write_samples(path, samples)

    if not dry_run:
        _write_samples(path, samples)

    print_failed_sample_ids(
        stats["failed_sample_ids"],
        title="multi_action failed sample_ids (re-run: --retry-ids)",
    )

    report_path = MULTI_QUALITY_REPORT_DIR / "llm_multi_action_label.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(stats, rf, indent=2, ensure_ascii=False)
    return stats

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--all-status", action="store_true", help="Re-check non-pending too")
    parser.add_argument(
        "--only-rejected",
        action="store_true",
        help="Re-label only label_status=rejected; skip llm_checked and rule",
    )
    parser.add_argument(
        "--retry-ids",
        nargs="*",
        default=None,
        help="Only label these sample_id values (for re-run after failure)",
    )
    args = parser.parse_args()

    if args.only_rejected and args.all_status:
        parser.error("--only-rejected cannot be used with --all-status")
    if args.only_rejected and args.retry_ids:
        parser.error("--only-rejected cannot be used with --retry-ids")

    retry_set = set(args.retry_ids) if args.retry_ids else None

    stats = label_file(
        max_samples=args.max_samples,
        dry_run=args.dry_run,
        only_pending=not args.all_status and retry_set is None and not args.only_rejected,
        only_rejected=args.only_rejected,
        retry_ids=retry_set,
    )
    print(stats)
    if not LLM_API_KEY and not args.dry_run:
        print("请设置环境变量 LLM_API_KEY（或 V4_LLM_API_KEY / DASHSCOPE_API_KEY / NVIDIA_API_KEY）")

if __name__ == "__main__":
    main()
