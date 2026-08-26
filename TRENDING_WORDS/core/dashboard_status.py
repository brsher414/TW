"""Derive business-friendly Cluster and Research Topic states for the dashboard.

The existing build_topics() remains the execution source of truth. This module does
not alter 04/05 logic, Topic IDs, prompts, schemas, or cache signatures.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.dashboard_config import EXCLUSION_DETAILS, EXCLUSION_LABELS
from core.external_topic import build_topics


def _mapping_candidates(insight: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    mappings: list[tuple[str, dict[str, Any]]] = []
    primary = insight.get("primary_mapping")
    if isinstance(primary, dict) and any(
        primary.get(key)
        for key in ("attribute_code", "existing_label", "proposed_new_label")
    ):
        mappings.append(("primary_mapping", primary))
    for index, secondary in enumerate(insight.get("secondary_mappings") or []):
        if isinstance(secondary, dict):
            mappings.append((f"secondary_mappings[{index}]", secondary))
    if insight.get("proposed_new_attribute") or insight.get("proposed_new_label"):
        mappings.append((
            "root_new_opportunity",
            {
                "attribute_code": None,
                "attribute_name": None,
                "existing_label": None,
                "proposed_new_attribute": insight.get("proposed_new_attribute"),
                "proposed_new_label": insight.get("proposed_new_label"),
                "reason": insight.get("review_reason", ""),
            },
        ))
    return mappings


def _excluded_row(
    *,
    cluster_id: int,
    topic_name: str,
    source_mapping: str | None,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "research_topic_id": None,
        "topic_name": topic_name,
        "topic_type": None,
        "source_mapping": source_mapping,
        "external_eligibility": "NOT_ELIGIBLE",
        "external_status": "NOT_APPLICABLE",
        "exclusion_reason_code": reason_code,
        "exclusion_reason_label": EXCLUSION_LABELS[reason_code],
        "exclusion_reason_detail": EXCLUSION_DETAILS[reason_code],
        "external_record": None,
        "external_error": None,
    }


def assess_topics_for_dashboard(
    *,
    cluster_id: int,
    evidence: dict[str, Any],
    internal_record: dict[str, Any],
    external_records: dict[str, dict[str, Any]],
    external_errors: dict[str, dict[str, Any]],
    include_existing_label_trends: bool = False,
) -> list[dict[str, Any]]:
    insight = internal_record.get("parsed_result") or {}
    eligible_topics = build_topics(
        cluster_id=cluster_id,
        evidence=evidence,
        internal_record=internal_record,
        include_existing_label_trends=include_existing_label_trends,
    )
    eligible_sources = {topic.source_mapping for topic in eligible_topics}
    rows: list[dict[str, Any]] = []

    for topic in eligible_topics:
        external = external_records.get(topic.research_topic_id)
        error = external_errors.get(topic.research_topic_id)
        if external:
            status = "COMPLETED"
        elif error:
            status = "ERROR"
        else:
            status = "PENDING"
        rows.append({
            "cluster_id": cluster_id,
            "research_topic_id": topic.research_topic_id,
            "topic_name": topic.topic_name,
            "topic_type": topic.topic_type,
            "source_mapping": topic.source_mapping,
            "external_eligibility": "ELIGIBLE",
            "external_status": status,
            "exclusion_reason_code": None,
            "exclusion_reason_label": None,
            "exclusion_reason_detail": None,
            "external_record": external,
            "external_error": error,
        })

    if insight.get("mapping_type") == "mixed_or_invalid_cluster":
        return [
            _excluded_row(
                cluster_id=cluster_id,
                topic_name=insight.get("cluster_name") or f"Cluster {cluster_id}",
                source_mapping=None,
                reason_code="MIXED_OR_INVALID_CLUSTER",
            )
        ]

    attributes = {
        item.get("attribute_code"): item
        for item in evidence.get("all_existing_attributes", [])
        if isinstance(item, dict) and item.get("attribute_code")
    }
    for source_mapping, mapping in _mapping_candidates(insight):
        if source_mapping in eligible_sources:
            continue
        code = mapping.get("attribute_code")
        name = mapping.get("attribute_name")
        existing_label = mapping.get("existing_label")
        new_attribute = mapping.get("proposed_new_attribute")
        new_label = mapping.get("proposed_new_label")
        meta = attributes.get(code, {}) if code else {}

        if code and (
            meta.get("label_evidence_mode") == "withheld"
            or meta.get("label_evidence_available") is False
        ):
            reason_code = "LABEL_EVIDENCE_WITHHELD"
        elif new_attribute and str(new_attribute).strip() in {"规格", "包装规格"}:
            reason_code = "KNOWN_PACKSIZE_ATTRIBUTE"
        elif existing_label and not include_existing_label_trends:
            reason_code = "EXISTING_LABEL_TREND_DISABLED"
        elif code and not existing_label and not new_label:
            reason_code = "EXISTING_ATTRIBUTE_LABEL_UNKNOWN"
        else:
            reason_code = "NO_RESEARCHABLE_MAPPING"

        topic_name = (
            f"{name} - {existing_label or new_label}"
            if name and (existing_label or new_label)
            else str(
                new_attribute
                or name
                or insight.get("cluster_name")
                or f"Cluster {cluster_id}"
            )
        )
        rows.append(_excluded_row(
            cluster_id=cluster_id,
            topic_name=topic_name,
            source_mapping=source_mapping,
            reason_code=reason_code,
        ))

    if not rows:
        rows.append(_excluded_row(
            cluster_id=cluster_id,
            topic_name=insight.get("cluster_name") or f"Cluster {cluster_id}",
            source_mapping=None,
            reason_code="NO_RESEARCHABLE_MAPPING",
        ))
    return rows


def build_topic_table(
    sources: dict[str, Any],
    *,
    include_existing_label_trends: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_key, internal_record in sources["internal_records"].items():
        rows.extend(assess_topics_for_dashboard(
            cluster_id=int(cluster_key),
            evidence=sources["evidence_by_cluster"].get(cluster_key, {}),
            internal_record=internal_record,
            external_records=sources["external_records"],
            external_errors=sources["external_errors_by_topic"],
            include_existing_label_trends=include_existing_label_trends,
        ))
    return pd.DataFrame(rows)


def cluster_status_table(sources: dict[str, Any], topics: pd.DataFrame) -> pd.DataFrame:
    summary = sources["cluster_summary"].copy()
    internal_ids = {int(key) for key in sources["internal_records"]}
    summary["internal_status"] = summary["cluster_id"].map(
        lambda value: "COMPLETED" if int(value) in internal_ids else "PENDING"
    )
    if topics.empty:
        for column in (
            "topic_count", "eligible_topic_count", "external_completed_count",
            "external_error_count", "excluded_topic_count",
        ):
            summary[column] = 0
        return summary

    grouped = topics.groupby("cluster_id", dropna=False).agg(
        topic_count=("topic_name", "count"),
        eligible_topic_count=(
            "external_eligibility",
            lambda values: sum(value == "ELIGIBLE" for value in values),
        ),
        external_completed_count=(
            "external_status",
            lambda values: sum(value == "COMPLETED" for value in values),
        ),
        external_error_count=(
            "external_status",
            lambda values: sum(value == "ERROR" for value in values),
        ),
        excluded_topic_count=(
            "external_eligibility",
            lambda values: sum(value == "NOT_ELIGIBLE" for value in values),
        ),
    ).reset_index()
    return summary.merge(grouped, on="cluster_id", how="left").fillna({
        "topic_count": 0,
        "eligible_topic_count": 0,
        "external_completed_count": 0,
        "external_error_count": 0,
        "excluded_topic_count": 0,
    })
