"""Noise Term LLM output parsing and business validation (Schema v1)."""
from __future__ import annotations

import math
from typing import Any

from core.cluster_schema import ValidationResult, extract_json_object
from core.mapping_validation import build_mapping_context, validate_mapping

ALLOWED_TERM_QUALITY = {"valid", "ambiguous", "invalid"}
ALLOWED_NOISE_MAPPING_TYPES = {
    "existing_attribute_existing_label",
    "existing_attribute_new_label",
    "new_attribute_new_label",
    "uncertain",
    "invalid_term",
}
ROOT_FIELDS = {
    "query_key", "analysis_unit", "term", "term_quality", "mapping_type",
    "primary_mapping", "proposed_new_attribute", "proposed_new_label",
    "observed_evidence", "trend_summary", "confidence", "review_required",
    "review_reason", "external_research_recommended",
    "external_search_queries",
}
PRIMARY_FIELDS = {
    "attribute_code", "attribute_name", "existing_label",
    "proposed_new_label",
}


def parse_and_validate_noise_term(
    output_text: str,
    evidence: dict[str, Any],
    *,
    reject_unknown_fields: bool = True,
) -> ValidationResult:
    obj, error = extract_json_object(output_text)
    if obj is None:
        return ValidationResult(
            False,
            "internal_noise",
            None,
            [error],
            [],
            error.split(":", 1)[0],
        )
    return validate_noise_term_insight(
        obj,
        evidence,
        reject_unknown_fields=reject_unknown_fields,
    )


def validate_noise_term_insight(
    result: Any,
    evidence: dict[str, Any],
    *,
    reject_unknown_fields: bool = True,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(result, dict):
        return ValidationResult(
            False,
            "internal_noise",
            None,
            ["ROOT_NOT_OBJECT"],
            [],
            "SCHEMA_VALIDATION_ERROR",
        )

    if reject_unknown_fields:
        unknown = sorted(set(result) - ROOT_FIELDS)
        if unknown:
            errors.append(f"root 含未定义字段：{unknown}")

    if result.get("query_key") != evidence.get("query_key"):
        errors.append("QUERY_KEY_MISMATCH")
    if result.get("analysis_unit") != "noise_term":
        errors.append("analysis_unit必须为noise_term")
    expected_term = (
        evidence.get("term_evidence", {}).get("ngram")
        or evidence.get("term")
    )
    if result.get("term") != expected_term:
        errors.append("TERM_MISMATCH")

    quality = result.get("term_quality")
    mapping_type = result.get("mapping_type")
    if quality not in ALLOWED_TERM_QUALITY:
        errors.append("term_quality非法")
    if mapping_type not in ALLOWED_NOISE_MAPPING_TYPES:
        errors.append("mapping_type非法")

    primary = result.get("primary_mapping")
    if not isinstance(primary, dict):
        errors.append("primary_mapping必须是对象")
        primary = {}
    elif reject_unknown_fields:
        unknown = sorted(set(primary) - PRIMARY_FIELDS)
        if unknown:
            errors.append(f"primary_mapping含未定义字段：{unknown}")

    directory, candidates = build_mapping_context(evidence)
    validate_mapping(
        primary,
        path="primary_mapping",
        directory=directory,
        candidates=candidates,
        result=result,
        errors=errors,
        warnings=warnings,
        allow_new_attribute=False,
    )

    proposed_attribute = result.get("proposed_new_attribute")
    proposed_label = result.get("proposed_new_label")
    evidence_items = result.get("observed_evidence")
    queries = result.get("external_search_queries")

    if proposed_attribute is not None and (
        not isinstance(proposed_attribute, str)
        or not proposed_attribute.strip()
    ):
        errors.append("proposed_new_attribute必须为非空字符串或null")
    if proposed_label is not None and (
        not isinstance(proposed_label, str) or not proposed_label.strip()
    ):
        errors.append("proposed_new_label必须为非空字符串或null")

    if not isinstance(evidence_items, list) or not evidence_items:
        errors.append("observed_evidence不能为空")
    elif not all(isinstance(item, str) and item.strip() for item in evidence_items):
        errors.append("observed_evidence必须为非空字符串数组")

    if not isinstance(result.get("trend_summary"), str) or not result.get(
        "trend_summary", ""
    ).strip():
        errors.append("trend_summary不能为空")
    confidence = result.get("confidence")
    if not _number(confidence) or not 0 <= float(confidence) <= 1:
        errors.append("confidence必须在0到1之间")
    if not isinstance(result.get("review_required"), bool):
        errors.append("review_required必须是布尔值")
    if not isinstance(result.get("review_reason"), str):
        errors.append("review_reason必须是字符串")
    if not isinstance(result.get("external_research_recommended"), bool):
        errors.append("external_research_recommended必须是布尔值")
    if not isinstance(queries, list) or len(queries) > 3:
        errors.append("external_search_queries必须是最多3项的数组")
    elif not all(isinstance(item, str) and item.strip() for item in queries):
        errors.append("external_search_queries必须为非空字符串数组")

    code = primary.get("attribute_code")
    existing = primary.get("existing_label")
    primary_new_label = primary.get("proposed_new_label")

    if mapping_type == "existing_attribute_existing_label":
        if code is None or existing is None or primary_new_label is not None:
            errors.append("已有属性已有标签的字段组合无效")
        if proposed_attribute is not None or proposed_label is not None:
            errors.append("已有属性已有标签不得填写Root新属性/新标签")
    elif mapping_type == "existing_attribute_new_label":
        if code is None or existing is not None or primary_new_label is None:
            errors.append("已有属性候选新标签的字段组合无效")
        if proposed_attribute is not None or proposed_label is not None:
            errors.append("已有属性候选新标签不得填写Root新属性/新标签")
    elif mapping_type == "new_attribute_new_label":
        if any(primary.get(field) is not None for field in PRIMARY_FIELDS):
            errors.append("候选新属性时primary_mapping所有字段必须为null")
        if proposed_attribute is None or proposed_label is None:
            errors.append("候选新属性必须填写Root新属性和初始标签")
    elif mapping_type in {"uncertain", "invalid_term"}:
        if proposed_attribute is not None or proposed_label is not None:
            errors.append(f"{mapping_type}不得填写候选新属性或新标签")

    if quality == "invalid":
        if mapping_type != "invalid_term":
            errors.append("term_quality=invalid时mapping_type必须为invalid_term")
        if result.get("external_research_recommended") is not False:
            errors.append("无效词不得建议外部研究")
    elif mapping_type == "invalid_term":
        errors.append("mapping_type=invalid_term时term_quality必须为invalid")

    if result.get("external_research_recommended") is False and queries:
        errors.append("未建议外部研究时external_search_queries必须为空")
    if result.get("review_required") is True and not result.get(
        "review_reason", ""
    ).strip():
        errors.append("review_required=true时review_reason不能为空")

    return ValidationResult(
        not errors,
        "internal_noise",
        result,
        errors,
        warnings,
        _error_code(errors),
    )


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _error_code(errors: list[str]) -> str | None:
    if not errors:
        return None
    for code in ("QUERY_KEY_MISMATCH", "TERM_MISMATCH"):
        if any(code in item for item in errors):
            return code
    return "SCHEMA_VALIDATION_ERROR"
