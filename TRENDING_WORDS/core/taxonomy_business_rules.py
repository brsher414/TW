from __future__ import annotations
from typing import Any

FACT_BASED_MARKET_EXPRESSIONS = {
    "SHELF_LIFE": {"短保", "短保质期", "超短保", "长保", "长期保存"},
}


def candidate_by_code(evidence: dict[str, Any], code: str | None) -> dict[str, Any] | None:
    if not code:
        return None
    for item in evidence.get("taxonomy_candidates", []) or []:
        if isinstance(item, dict) and item.get("attribute_code") == code:
            return item
    return None


def existing_labels(evidence: dict[str, Any], code: str | None) -> set[str]:
    candidate = candidate_by_code(evidence, code)
    if not candidate:
        return set()
    return {
        str(item.get("label")).strip()
        for item in candidate.get("candidate_labels", []) or []
        if isinstance(item, dict) and str(item.get("label") or "").strip()
    }


def proposed_label_error(code: str | None, label: Any) -> str | None:
    text = str(label or "").strip()
    if code in FACT_BASED_MARKET_EXPRESSIONS and text in FACT_BASED_MARKET_EXPRESSIONS[code]:
        return "MARKET_EXPRESSION_CANNOT_BE_CREATED_AS_RANGE_LABEL"
    return None


def new_label_eligibility(mapping: dict[str, Any], evidence: dict[str, Any]) -> tuple[bool, str | None]:
    code = mapping.get("attribute_code")
    label = mapping.get("proposed_new_label")
    if not code or not label:
        return True, None
    candidate = candidate_by_code(evidence, code)
    if candidate is None:
        return False, "EXISTING_ATTRIBUTE_LABELS_NOT_VERIFIED"
    if candidate.get("label_evidence_mode") != "all_valid_labels" or candidate.get("label_list_complete") is not True:
        return False, "EXISTING_ATTRIBUTE_LABELS_NOT_VERIFIED"
    labels = existing_labels(evidence, code)
    if str(label).strip() in labels:
        return False, "PROPOSED_LABEL_ALREADY_EXISTS"
    error = proposed_label_error(code, label)
    if error:
        return False, error
    return True, None
