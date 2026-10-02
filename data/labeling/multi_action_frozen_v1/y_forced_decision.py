from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FORMAL_Y_DECISIONS = frozenset({"ACTIONS", "NO_ACTION"})

INVALID_FORMAL_Y_DECISIONS = frozenset(
    {
        "UNCERTAIN",
        "UNKNOWN",
        "ABSTAIN",
        "NEEDS_REVIEW",
        "REJECT",
        "SEMANTIC_REJECT",
        "LLM_SEMANTIC_REJECT",
    }
)

LOW_CONFIDENCE_THRESHOLD = 0.60

Y_SEMANTIC_RETRYABLE_REASONS = frozenset(
    {
        "SERVICE_OUT_OF_COMPONENT_DOMAIN",
        "TARGET_OUT_OF_ENTITY_CATALOG",
        "missing_component_origin",
        "ambiguous_component_origin",
        "missing_target_entity",
        "multiple_entity_candidates",
        "invalid_json",
        "EMPTY_ACTIONS_FOR_ACTIONS",
        "NONEMPTY_ACTIONS_FOR_NO_ACTION",
        "INVALID_FORMAL_Y_DECISION",
        "ARGMAX_INCONSISTENCY",
        "missing_decision",
        "missing_service",
    }
)

FORCED_DECISION_RETRY_INSTRUCTION = (
    "You must make a final forced choice. Select the most likely valid ACTIONS or NO_ACTION outcome. "
    "Do not return uncertainty as the decision. Missing evidence does NOT automatically imply NO_ACTION."
)

@dataclass
class FormalYNormalizationResult:
    structured: list[dict[str, Any]]
    normalization_status: str
    ambiguities: list[dict[str, str]]
    label_decision: str | None
    label_confidence: float | None = None
    low_confidence: bool = False
    alternative_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_reason: str | None = None
    formal_y_valid: bool = False

    def to_legacy_tuple(self) -> tuple[list[dict], str, list[dict[str, str]], str | None]:
        return self.structured, self.normalization_status, self.ambiguities, self.label_decision

def _clamp_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        c = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, c))

def _hypothesis_key(h: dict[str, Any]) -> str:
    decision = str(h.get("decision") or "").upper()
    if decision == "ACTIONS":
        services = sorted(
            str(a.get("service") or "")
            for a in (h.get("actions") or [])
            if isinstance(a, dict) and a.get("service")
        )
        return f"ACTIONS:{','.join(services)}"
    return decision or "UNKNOWN"

def audit_confidence_metadata(parsed: dict[str, Any], decision: str) -> tuple[float | None, bool, list[dict], str | None, list[dict[str, str]]]:

    violations: list[dict[str, str]] = []
    confidence = _clamp_confidence(parsed.get("confidence"))
    low_confidence = confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD

    alts_raw = list(parsed.get("alternative_hypotheses") or [])
    alternatives: list[dict[str, Any]] = []
    for alt in alts_raw:
        if not isinstance(alt, dict):
            continue
        prob = _clamp_confidence(alt.get("probability"))
        if prob is not None and (prob < 0.0 or prob > 1.0):
            violations.append({"reason": "INVALID_PROBABILITY", "value": str(alt.get("probability"))})
        alternatives.append(
            {
                "decision": str(alt.get("decision") or "").upper(),
                "probability": prob,
                "actions": alt.get("actions") or [],
            }
        )

    if alternatives:
        scored = [(a, a.get("probability")) for a in alternatives if a.get("probability") is not None]
        if scored:
            best_alt, best_prob = max(scored, key=lambda x: x[1])
            chosen_prob = confidence
            best_decision = str(best_alt.get("decision") or "").upper()
            if best_decision and best_decision != decision.upper():
                if chosen_prob is None or (best_prob is not None and best_prob > chosen_prob):
                    violations.append(
                        {
                            "reason": "ARGMAX_INCONSISTENCY",
                            "decision": decision,
                            "best_alternative": best_decision,
                            "best_probability": str(best_prob),
                            "confidence": str(chosen_prob),
                        }
                    )
            elif best_decision == decision.upper() and chosen_prob is not None and best_prob is not None:
                if best_prob > chosen_prob + 0.01:
                    violations.append(
                        {
                            "reason": "ARGMAX_INCONSISTENCY",
                            "decision": decision,
                            "best_alternative": best_decision,
                            "best_probability": str(best_prob),
                            "confidence": str(chosen_prob),
                        }
                    )

    uncertainty_reason = str(parsed.get("uncertainty_reason") or parsed.get("explanation") or "") or None
    return confidence, low_confidence, alternatives, uncertainty_reason, violations
