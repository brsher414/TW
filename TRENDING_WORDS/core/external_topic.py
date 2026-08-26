"""Build external-research topics for Cluster and Noise-Term analysis units.

The internal field external_research_recommended is advisory metadata only.
It never blocks a valid ACTIVE internal result from being offered for manual research.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

from core.cluster_cache import canonical_json
from core.config import EXTERNAL_TOOLS
from core.external_topic_prompt import (
    EXTERNAL_TOPIC_PROMPT_VERSION,
    EXTERNAL_TOPIC_SYSTEM_PROMPT,
    build_external_topic_prompt,
)

EXTERNAL_TOPIC_SCHEMA_VERSION = "external_topic_schema_v8_direction_alignment"


@dataclass(slots=True)
class ExternalResearchTopic:
    research_topic_id: str
    category_code: str
    category_name: str
    analysis_unit: str
    source_query_key: str
    cluster_id: int
    term: str | None
    ai_research_recommended: bool
    topic_name: str
    topic_type: str
    attribute_code: str | None
    attribute_name: str | None
    existing_label: str | None
    proposed_new_attribute: str | None
    proposed_new_label: str | None
    reason: str
    direction_recommendation: str
    direction_recommendation_reason: str
    source_mapping: str
    cluster_evidence: dict[str, Any]
    internal_insight: dict[str, Any]
    internal_signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def api_item(self, signature: str) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "query_key": self.research_topic_id,
            "research_topic_id": self.research_topic_id,
            "analysis_unit": self.analysis_unit,
            "source_query_key": self.source_query_key,
            "signature": signature,
            "__user_input__": build_external_topic_prompt(self.to_dict()),
        }


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value or "topic")).strip("_").lower()
    if token and token != "topic":
        return token[:48]
    return hashlib.sha256(str(value or "topic").encode("utf-8")).hexdigest()[:12]


def _active(record: dict[str, Any]) -> bool:
    return str(record.get("record_status", "ACTIVE")).upper() != "INVALIDATED"


def _attribute_metadata(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("attribute_code")): item
        for item in evidence.get("all_existing_attributes", [])
        if isinstance(item, dict) and item.get("attribute_code")
    }


def build_topics(
    *,
    category_code: str,
    category_name: str,
    query_key: str | None = None,
    cluster_id: int | None = None,
    evidence: dict[str, Any],
    internal_record: dict[str, Any],
    include_existing_label_trends: bool = False,
) -> list[ExternalResearchTopic]:
    """Create selectable external topics from every valid ACTIVE internal result."""
    if not _active(internal_record) or not internal_record.get("schema_valid", True):
        return []
    insight = internal_record.get("parsed_result") or {}
    if insight.get("mapping_type") in {"mixed_or_invalid_cluster", "invalid_term"}:
        return []

    analysis_unit = str(
        evidence.get("analysis_unit")
        or internal_record.get("analysis_unit")
        or "cluster"
    )
    if analysis_unit not in {"cluster", "noise_term"}:
        return []
    source_query_key = str(
        query_key or evidence.get("query_key") or internal_record.get("query_key") or ""
    )
    if not source_query_key:
        return []
    source_cluster_id = int(
        cluster_id if cluster_id is not None
        else evidence.get("cluster_id", internal_record.get("cluster_id", -1))
    )
    term = evidence.get("term")
    if not term and isinstance(evidence.get("term_evidence"), dict):
        term = evidence["term_evidence"].get("ngram")
    ai_recommended = bool(insight.get("external_research_recommended", False))

    attribute_meta = _attribute_metadata(evidence)
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
                "direction_recommendation": "new_attribute",
                "direction_recommendation_reason": insight.get("review_reason", ""),
            },
        ))

    # Keep uncertain/non-recommended results selectable instead of silently dropping them.
    if not mappings:
        fallback_name = (
            str(term) if analysis_unit == "noise_term" and term
            else str(insight.get("cluster_name") or source_query_key)
        )
        mappings.append((
            "advisory_review_topic",
            {
                "attribute_code": None,
                "attribute_name": None,
                "existing_label": None,
                "proposed_new_attribute": None,
                "proposed_new_label": None,
                "reason": str(
                    insight.get("review_reason")
                    or "内部模型未建议外部研究；保留供人工决定是否验证。"
                ),
                "fallback_topic_name": fallback_name,
                "direction_recommendation": "continue_validation",
                "direction_recommendation_reason": str(insight.get("review_reason") or "内部模型尚未形成明确方向建议。"),
            },
        ))

    topics: list[ExternalResearchTopic] = []
    used_ids: set[str] = set()
    for source_mapping, mapping in mappings:
        code = mapping.get("attribute_code")
        name = mapping.get("attribute_name")
        existing_label = mapping.get("existing_label")
        new_attribute = mapping.get("proposed_new_attribute")
        new_label = mapping.get("proposed_new_label")
        meta = attribute_meta.get(str(code), {}) if code else {}
        if code and (
            meta.get("label_evidence_mode") == "withheld"
            or meta.get("label_evidence_available") is False
        ):
            continue
        if new_attribute and str(new_attribute).strip() in {"包装规格", "规格"}:
            continue

        direction_recommendation = str(
            mapping.get("direction_recommendation") or "continue_validation"
        )
        direction_reason = str(
            mapping.get("direction_recommendation_reason")
            or mapping.get("reason")
            or ""
        )
        if source_mapping == "advisory_review_topic":
            topic_type = "uncertain_opportunity"
            topic_name = str(mapping.get("fallback_topic_name") or source_query_key)
        elif direction_recommendation == "use_existing_label" or existing_label:
            if not include_existing_label_trends:
                continue
            topic_type = "existing_label_trend"
            topic_name = f"{name} - {existing_label or new_label}"
        elif direction_recommendation == "new_attribute" or new_attribute:
            topic_type = "new_attribute_opportunity"
            topic_name = f"{new_attribute or name}{' - ' + str(new_label) if new_label else ''}"
        elif direction_recommendation == "direct_addition" and code and new_label:
            topic_type = "new_label_opportunity"
            topic_name = f"{name} - {new_label}"
        elif direction_recommendation in {
            "derived_label", "taxonomy_restructure", "continue_validation", "not_recommended"
        }:
            topic_type = "uncertain_opportunity"
            topic_name = f"{name or source_query_key}{' - ' + str(new_label) if new_label else ''}"
        elif code and new_label:
            topic_type = "new_label_opportunity"
            topic_name = f"{name} - {new_label}"
        elif code:
            topic_type = "uncertain_opportunity"
            topic_name = str(name or source_query_key)
        else:
            continue

        token_seed = "|".join(
            str(x or "")
            for x in (source_mapping, code, new_attribute, new_label, existing_label)
        )
        topic_id = f"{source_query_key}:topic:{_safe_token(token_seed)}"
        if topic_id in used_ids:
            topic_id += ":" + hashlib.sha256(token_seed.encode("utf-8")).hexdigest()[:8]
        used_ids.add(topic_id)
        topics.append(ExternalResearchTopic(
            research_topic_id=topic_id,
            category_code=str(category_code),
            category_name=str(category_name),
            analysis_unit=analysis_unit,
            source_query_key=source_query_key,
            cluster_id=source_cluster_id,
            term=str(term) if term is not None else None,
            ai_research_recommended=ai_recommended,
            topic_name=topic_name,
            topic_type=topic_type,
            attribute_code=code,
            attribute_name=name,
            existing_label=existing_label,
            proposed_new_attribute=new_attribute,
            proposed_new_label=new_label,
            reason=str(mapping.get("reason", "")),
            direction_recommendation=direction_recommendation,
            direction_recommendation_reason=direction_reason,
            source_mapping=source_mapping,
            cluster_evidence=evidence,
            internal_insight=insight,
            internal_signature=str(internal_record.get("signature", "")),
        ))
    return topics


def build_topic_signature(topic: ExternalResearchTopic, model: str) -> str:
    payload = {
        "model": model,
        "prompt_version": EXTERNAL_TOPIC_PROMPT_VERSION,
        "schema_version": EXTERNAL_TOPIC_SCHEMA_VERSION,
        "tools": EXTERNAL_TOOLS,
        "system_prompt": EXTERNAL_TOPIC_SYSTEM_PROMPT,
        "topic": topic.to_dict(),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
