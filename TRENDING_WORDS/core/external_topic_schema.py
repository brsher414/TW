"""Validation for category-specific external label-opportunity research JSON."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlparse

from core.config import ALLOWED_EXTERNAL_TOPIC_TYPES
from core.json_repair_adapter import parse_llm_json_with_repair

ROOT_FIELDS = {
    "research_topic_id",
    "category_code",
    "category_name",
    "analysis_unit",
    "source_query_key",
    "cluster_id",
    "topic_name",
    "topic_type",
    "external_research_status",
    "external_findings",
    "trend_hypothesis",
    "hypothesis_limitations",
    "commercial_opportunity",
    "risks_and_limitations",
    "review_required",
}
ALLOWED_STATUS = {"completed", "insufficient_evidence", "error"}
ALLOWED_RECOMMENDATION = {
    "direct_addition", "derived_label", "new_attribute", "taxonomy_restructure",
    "continue_validation", "not_supported", "insufficient_evidence"
}
ALLOWED_OPPORTUNITY = {
    "new_label_under_existing_attribute",
    "derived_label_from_existing_attribute",
    "new_attribute_with_initial_label",
    "taxonomy_restructure",
    "existing_label_trend_validation",
    "insufficient_support",
}
ALLOWED_STRENGTH = {"strong", "moderate", "weak", "insufficient"}
ALLOWED_BREADTH = {"multi_source", "single_source", "unknown"}
ALLOWED_TAXONOMY_CHECK = {
    "not_found", "existing_label_match", "possible_synonym_match", "taxonomy_evidence_unavailable"
}


@dataclass(slots=True)
class TopicValidationResult:
    valid: bool
    parsed: dict[str, Any] | None
    errors: list[str]
    warnings: list[str]
    error_code: str | None = None

    @property
    def error_text(self) -> str:
        return " | ".join(self.errors)

    @property
    def warning_text(self) -> str:
        return " | ".join(self.warnings)


def _iso(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _relation(value: Any, period: dict[str, Any]) -> str:
    source = _iso(value)
    if source is None:
        return "date_unknown"
    try:
        base_start = date.fromisoformat(period["base_period_start"])
        base_end = date.fromisoformat(period["base_period_end"])
        current_start = date.fromisoformat(period["current_period_start"])
        current_end = date.fromisoformat(period["current_period_end"])
    except (KeyError, TypeError, ValueError):
        return "date_unknown"
    if source < base_start:
        return "before_base_period"
    if source <= base_end:
        return "during_base_period"
    if source < current_start:
        return "between_periods"
    if source <= current_end:
        return "during_current_period"
    return "after_current_period"


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_source_url(value: Any) -> str | None:
    """Normalize only explicit http(s) URLs; never invent a protocol."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    markdown = re.fullmatch(r"\[[^\]]*\]\((https?://[^)\s]+)\)", text)
    if markdown:
        text = markdown.group(1)
    else:
        match = re.search(r"https?://[^\s<>\"]+", text)
        if not match:
            return None
        text = match.group(0)
    text = text.rstrip(".,;:!?，。；：！？)]}>'\"")
    try:
        parts = urlparse(text)
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    return text


def _source_title_from_url(url: str) -> str:
    """Return a truthful display fallback when the model omits a page title."""
    try:
        hostname = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return hostname.removeprefix("www.").strip()


SOURCE_TYPES = {
    "brand_product_page",
    "industry_media",
    "retailer_listing",
    "market_report",
    "social_content",
    "other",
}


def _looks_internal_finding(finding: dict[str, Any]) -> bool:
    title = str(finding.get("source_title") or "").casefold()
    claim = str(finding.get("claim") or finding.get("finding") or "").casefold()
    text = f"{title} {claim}"
    tokens = (
        "内部标签体系", "内部 taxonomy", "内部taxonomy", "内部 evidence",
        "内部数据", "internal taxonomy", "internal evidence",
    )
    return any(token in text for token in tokens)


def validate_external_topic(
    output_text: str,
    topic: dict[str, Any],
    *,
    degraded: bool = False,
) -> TopicValidationResult:
    if degraded:
        return TopicValidationResult(
            False, None, ["WEB_RESEARCH_DEGRADED_WITHOUT_TOOLS"], [],
            "WEB_RESEARCH_DEGRADED_WITHOUT_TOOLS",
        )

    repair = parse_llm_json_with_repair(output_text)
    if repair.value is None or not isinstance(repair.value, dict):
        return TopicValidationResult(
            False, None, [repair.error or "JSON_PARSE_ERROR"],
            list(repair.warnings), "JSON_PARSE_ERROR",
        )

    parsed = repair.value
    errors: list[str] = []
    warnings = list(repair.warnings)

    if parsed.get("external_research_status") in {
        "sufficient_evidence", "complete", "success"
    }:
        parsed["external_research_status"] = "completed"
        warnings.append("STATUS_NORMALIZED")

    opportunity = parsed.get("commercial_opportunity")
    if isinstance(opportunity, dict):
        if "recommendation_reason" not in opportunity and "label_decision_reason" in opportunity:
            opportunity["recommendation_reason"] = opportunity.pop("label_decision_reason")
            warnings.append("LEGACY_RECOMMENDATION_REASON_NORMALIZED")
        if "implementation_assessment" in opportunity:
            opportunity.pop("implementation_assessment", None)
            warnings.append("DEPRECATED_IMPLEMENTATION_ASSESSMENT_IGNORED")
        legacy_recommendation = {
            "recommend": "direct_addition",
            "recommend_new_attribute": "new_attribute",
            "recommend_new_label": "direct_addition",
            "direct_addition": "direct_addition",
            "watchlist": "continue_validation",
            "uncertain": "continue_validation",
            "not_recommended": "not_recommended",
            "not_supported": "not_recommended",
        }
        legacy_opportunity = {
            "new_label": "new_label_under_existing_attribute",
            "new_attribute": "new_attribute_with_initial_label",
            "existing_trend": "existing_label_trend_validation",
            "taxonomy_refinement": "insufficient_support",
            "none": "insufficient_support",
        }
        old_rec = opportunity.get("recommendation")
        old_type = opportunity.get("opportunity_type")
        if old_rec in legacy_recommendation:
            opportunity["recommendation"] = legacy_recommendation[old_rec]
            warnings.append("LEGACY_RECOMMENDATION_NORMALIZED")
        if old_type in legacy_opportunity:
            opportunity["opportunity_type"] = legacy_opportunity[old_type]
            warnings.append("LEGACY_OPPORTUNITY_NORMALIZED")
        opportunity["market_breadth"] = {
            "limited": "unknown", "broad": "multi_source"
        }.get(opportunity.get("market_breadth"), opportunity.get("market_breadth"))

    parsed["review_required"] = True
    missing = ROOT_FIELDS - set(parsed)
    if missing:
        errors.append(f"缺少字段:{sorted(missing)}")

    for field, code in (
        ("research_topic_id", "RESEARCH_TOPIC_ID_MISMATCH"),
        ("category_code", "CATEGORY_CODE_MISMATCH"),
        ("category_name", "CATEGORY_NAME_MISMATCH"),
        ("analysis_unit", "ANALYSIS_UNIT_MISMATCH"),
        ("source_query_key", "SOURCE_QUERY_KEY_MISMATCH"),
        ("cluster_id", "CLUSTER_ID_MISMATCH"),
        ("topic_type", "TOPIC_TYPE_MISMATCH"),
    ):
        if parsed.get(field) != topic.get(field):
            parsed[field] = topic.get(field)
            warnings.append(f"{code}_NORMALIZED")

    if parsed.get("analysis_unit") not in {"cluster", "noise_term"}:
        errors.append("analysis_unit非法")
    if parsed.get("topic_type") not in ALLOWED_EXTERNAL_TOPIC_TYPES:
        errors.append("topic_type非法")
    if parsed.get("external_research_status") not in ALLOWED_STATUS:
        errors.append("external_research_status非法")

    # External findings boundary: internal Taxonomy evidence belongs only in
    # commercial_opportunity.existing_taxonomy_check.
    raw_findings = parsed.get("external_findings", [])
    valid_findings: list[dict[str, Any]] = []
    unique_urls: set[str] = set()
    period = topic.get("cluster_evidence", {}).get("analysis_period", {})

    if not isinstance(raw_findings, list):
        errors.append("external_findings必须为数组")
        raw_findings = []
    if len(raw_findings) > 5:
        errors.append("external_findings最多5条")

    for index, finding in enumerate(raw_findings):
        if not isinstance(finding, dict):
            errors.append(f"external_findings[{index}]必须为对象")
            continue
        if "finding" in finding and "claim" not in finding:
            finding["claim"] = finding.pop("finding")
        if not _nonempty_text(finding.get("claim")):
            errors.append(f"external_findings[{index}].claim不能为空")
            continue

        normalized_url = _normalize_source_url(finding.get("source_url"))
        if normalized_url is None and _looks_internal_finding(finding):
            warnings.append(
                f"external_findings[{index}]已移除：内部Taxonomy Evidence不属于外部证据"
            )
            continue
        if normalized_url is None:
            errors.append(f"external_findings[{index}].source_url无效")
            continue

        if normalized_url != finding.get("source_url"):
            warnings.append(f"external_findings[{index}].source_url已规范化")
        if normalized_url in unique_urls:
            warnings.append(f"external_findings[{index}].source_url重复")
        unique_urls.add(normalized_url)
        finding["source_url"] = normalized_url

        source_title = str(finding.get("source_title") or "").strip()
        if not source_title:
            source_title = _source_title_from_url(normalized_url)
            finding["source_title"] = source_title
            warnings.append(
                f"external_findings[{index}].source_title缺失，已使用来源域名"
            )

        source_type = str(finding.get("source_type") or "").strip()
        if source_type not in SOURCE_TYPES:
            finding["source_type"] = "other"
            warnings.append(
                f"external_findings[{index}].source_type缺失或无效，已设为other"
            )

        finding["temporal_relation"] = _relation(finding.get("source_date"), period)
        valid_findings.append(finding)

    parsed["external_findings"] = valid_findings
    parsed["unique_source_count"] = len(unique_urls)
    if parsed.get("external_research_status") == "completed" and not valid_findings:
        errors.append("completed状态至少需要一个有效外部来源")

    if not isinstance(opportunity, dict):
        errors.append("commercial_opportunity必须为对象")
    else:
        recommendation = opportunity.get("recommendation")
        opportunity_type = opportunity.get("opportunity_type")
        label = opportunity.get("proposed_client_facing_label")
        reason = opportunity.get("recommendation_reason")
        taxonomy_check = opportunity.get("existing_taxonomy_check")
        taxonomy_status = (
            taxonomy_check.get("status") if isinstance(taxonomy_check, dict) else None
        )

        if recommendation not in ALLOWED_RECOMMENDATION:
            errors.append("recommendation非法")
        if opportunity_type not in ALLOWED_OPPORTUNITY:
            errors.append("opportunity_type非法")
        if opportunity.get("evidence_strength") not in ALLOWED_STRENGTH:
            errors.append("evidence_strength非法")
        if opportunity.get("market_breadth") not in ALLOWED_BREADTH:
            errors.append("market_breadth非法")
        if "recommendation_reason" not in opportunity:
            errors.append("commercial_opportunity缺少recommendation_reason")
        if not isinstance(taxonomy_check, dict):
            errors.append("commercial_opportunity.existing_taxonomy_check必须为对象")
        else:
            if taxonomy_status not in ALLOWED_TAXONOMY_CHECK:
                errors.append("existing_taxonomy_check.status非法")
            if not _nonempty_text(taxonomy_check.get("check_basis")):
                errors.append("existing_taxonomy_check.check_basis不能为空")

        if recommendation == "direct_addition":
            if taxonomy_status != "not_found":
                errors.append("仅现有标签体系检查为not_found时可直接新增候选标签")
            if not _nonempty_text(label):
                errors.append("direct_addition必须提供proposed_client_facing_label")
            if not _nonempty_text(reason):
                errors.append("direct_addition必须提供recommendation_reason")

        if taxonomy_status == "existing_label_match":
            if recommendation == "direct_addition":
                errors.append("现有标签匹配不得形成新增候选标签")
            if opportunity_type != "existing_label_trend_validation":
                errors.append("现有标签匹配应归为existing_label_trend_validation")
        if (
            taxonomy_status in {"possible_synonym_match", "taxonomy_evidence_unavailable"}
            and recommendation == "direct_addition"
        ):
            errors.append("同义词待核对或Taxonomy证据不可用时不得形成候选标签")

        if (
            opportunity_type in {
                "new_label_under_existing_attribute",
                "new_attribute_with_initial_label",
            }
            and not _nonempty_text(label)
        ):
            errors.append("新增标签或新属性机会必须提供具体候选标签")
        if (
            opportunity_type == "new_attribute_with_initial_label"
            and not _nonempty_text(topic.get("proposed_new_attribute"))
        ):
            errors.append("new_attribute_with_initial_label必须来源于内部新属性机会")
        if recommendation in {"not_recommended", "insufficient_evidence"}:
            if _nonempty_text(label):
                errors.append("不支持或证据不足时不得提供候选标签")
            if not _nonempty_text(reason):
                errors.append("不支持或证据不足时必须说明recommendation_reason")
        if (
            recommendation == "continue_validation"
            and not _nonempty_text(label)
            and not _nonempty_text(reason)
        ):
            errors.append("继续验证但未提出标签时必须说明recommendation_reason")
        if opportunity_type == "insufficient_support" and recommendation == "direct_addition":
            errors.append("insufficient_support不得direct_addition")
        if (
            topic.get("topic_type") == "existing_label_trend"
            and opportunity_type != "existing_label_trend_validation"
        ):
            errors.append("已有标签趋势只能标记为existing_label_trend_validation")
        if topic.get("analysis_unit") == "noise_term" and recommendation == "direct_addition":
            if opportunity.get("evidence_strength") != "strong":
                errors.append("Noise Term仅strong证据可直接新增候选标签")
            if opportunity.get("market_breadth") != "multi_source":
                errors.append("Noise Term形成候选标签必须有多来源支持")

    return TopicValidationResult(
        not errors,
        parsed,
        errors,
        warnings,
        "SCHEMA_VALIDATION_ERROR" if errors else None,
    )
