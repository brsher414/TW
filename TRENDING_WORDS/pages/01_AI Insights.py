"""AI Insights workbench for stable Clusters and Noise Terms."""
from __future__ import annotations

import html
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core.cluster_cache import ClusterCache
from core.cluster_loader import (
    evidence_summary_rows,
    load_cluster_evidence,
    prepare_internal_tasks,
)
from core.cluster_schema import parse_and_validate_internal
from core.config import (
    BASE_URL,
    BASE_URL_DEEPSEEK,
    BASE_URL_OPTIONS,
    DEFAULT_MODEL,
    INTERNAL_DEFAULT_WORKERS,
    INTERNAL_ENABLE_THINKING,
    INTERNAL_MAX_OUTPUT_TOKENS,
    INTERNAL_TEMPERATURE,
    MODEL_OPTIONS,
)
from core.noise_term_schema import parse_and_validate_noise_term
from core.workspace_selector import render_workspace_selector

from core.ui_state import render_ai_sidebar
from core.ai_run_ui import (
    inject_ai_run_styles,
    render_live_status,
    render_run_notice,
    render_persistent_call_history,
)
from core.call_history import append_call_history, read_call_history

st.set_page_config(page_title="AI Insights", page_icon="🧠", layout="wide")
inject_ai_run_styles()
st.session_state.setdefault("internal_last_run_rows", [])
st.markdown(
    """
    <style>


    .summary-overview {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0;
        padding: 11px 15px;
        margin: 6px 0 20px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, .025);
    }
    .summary-group {
        display: inline-flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0;
    }
    .summary-group + .summary-group {
        margin-left: 18px;
        padding-left: 18px;
        border-left: 1px solid #CBD5E1;
    }
    .summary-item {
        display: inline-flex;
        align-items: baseline;
        gap: 4px;
        color: #64748B;
        font-size: 13px;
        font-weight: 560;
        white-space: nowrap;
    }
    .summary-value {
        font-size: 19px;
        line-height: 1;
        font-weight: 760;
        letter-spacing: -.02em;
    }
    .summary-total .summary-value { color: #1E3A8A; }
    .summary-stable .summary-value { color: #2563EB; }
    .summary-explore .summary-value { color: #0891B2; }
    .summary-topic .summary-value { color: #6B7280; }
    .summary-done .summary-value { color: #2563EB; }
    .summary-pending .summary-value { color: #64748B; }
    .summary-operator {
        margin: 0 11px;
        color: #94A3B8;
        font-size: 14px;
        font-weight: 600;
    }
    @media (max-width: 850px) {
        .summary-group + .summary-group {
            width: 100%;
            margin: 10px 0 0;
            padding: 10px 0 0;
            border-left: 0;
            border-top: 1px solid #E2E8F0;
        }
    }
</style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #E2E8F0;
    }
    .overview-strip {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px 22px;
        padding: 11px 16px;
        margin: 4px 0 20px;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        background: #F8FAFC;
        color: #475569;
        font-size: 14px;
    }
    .overview-strip strong {
        color: #0F172A;
        font-size: 18px;
        margin-right: 4px;
    }
    .selection-strip {
        padding: 10px 14px;
        margin: 8px 0 14px;
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        background: linear-gradient(90deg, #EFF6FF, #F8FAFC);
        color: #334155;
        font-size: 14px;
    }
    .result-hero {
        padding: 18px 20px;
        margin: 8px 0 16px;
        border: 1px solid #DDE7F5;
        border-radius: 16px;
        background: linear-gradient(125deg, #F8FAFC, #EFF6FF);
        box-shadow: 0 8px 24px rgba(15, 23, 42, .04);
    }
    .result-eyebrow {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .result-title {
        margin: 5px 0 9px;
        color: #0F172A;
        font-size: 28px;
        font-weight: 750;
    }
    .result-meta {
        color: #475569;
        font-size: 14px;
    }
    .result-source {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid rgba(148, 163, 184, .28);
    }
    .result-source-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 9px;
        border: 1px solid #BFDBFE;
        border-radius: 999px;
        background: rgba(255, 255, 255, .72);
        color: #1D4ED8;
        font-size: 12px;
        font-weight: 700;
    }
    .result-query-key {
        color: #64748B;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
    }
    .result-terms {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 10px;
    }
    .result-terms-label {
        margin-right: 2px;
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
    }
    .result-term-chip {
        display: inline-flex;
        align-items: center;
        padding: 4px 9px;
        border: 1px solid #DBEAFE;
        border-radius: 999px;
        background: #FFFFFF;
        color: #334155;
        font-size: 12px;
        line-height: 1.35;
        box-shadow: 0 2px 6px rgba(15, 23, 42, .025);
    }
    .decision-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 18px;
    }
    .decision-card {
        padding: 14px 15px;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        background: #FFFFFF;
    }
    .decision-label {
        color: #64748B;
        font-size: 12px;
    }
    .decision-value {
        margin-top: 5px;
        color: #0F172A;
        font-size: 18px;
        font-weight: 720;
        word-break: break-word;
    }
    .section-note {
        margin-top: -5px;
        margin-bottom: 10px;
        color: #64748B;
        font-size: 13px;
    }

    .business-conclusion {
        border: 1px solid #DCE6F4;
        border-left: 4px solid #2563EB;
        border-radius: 12px;
        padding: 15px 17px;
        background: linear-gradient(100deg, #F8FAFC 0%, #FFFFFF 100%);
        margin-bottom: 14px;
    }
    .business-label, .mapping-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .04em;
    }
    .business-title {
        color: #0F172A;
        font-size: 23px;
        font-weight: 750;
        margin: 3px 0 8px;
    }
    .business-action {color:#334155;font-size:14px;}
    .mapping-card {
        min-height: 92px;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 15px 16px;
        background: #FFFFFF;
        box-shadow: 0 6px 18px rgba(15,23,42,.04);
    }
    .mapping-value {
        color: #0F172A;
        font-size: 20px;
        font-weight: 720;
        margin-top: 8px;
    }
    .reason-summary {
        border-radius: 10px;
        padding: 12px 14px;
        background: #F8FAFC;
        color: #334155;
        margin-bottom: 10px;
        line-height: 1.65;
    }
    .direction-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 18px;
    }
    .direction-card {
        border: 1px solid #E2E8F0;
        border-radius: 13px;
        padding: 15px 16px;
        background: #FFFFFF;
        box-shadow: 0 6px 18px rgba(15,23,42,.035);
    }
    .direction-card.primary,
    .direction-card.secondary {
        border-top: 3px solid #94A3B8;
    }
    .direction-kicker {
        color: #64748B;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: .08em;
    }
    .direction-title {
        color: #0F172A;
        font-size: 19px;
        font-weight: 740;
        margin: 5px 0 9px;
    }
    .direction-row {
        color: #475569;
        font-size: 13px;
        line-height: 1.65;
    }
    .direction-status {
        display: inline-block;
        margin: 0 0 10px;
        padding: 3px 8px;
        border-radius: 999px;
        background: #EFF6FF;
        color: #1D4ED8;
        font-size: 12px;
        font-weight: 650;
    }
    .direction-status.pending {
        background: #FFF7ED;
        color: #C2410C;
    }
    .direction-evidence {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #E2E8F0;
        color: #475569;
        font-size: 13px;
        line-height: 1.6;
    }
    .driver-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin: 8px 0 14px;
    }
    .driver-chip {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        background: #EFF6FF;
        color: #1D4ED8;
        font-size: 12px;
    }
    .evidence-list {
        margin: 8px 0 14px;
        padding: 12px 16px 12px 34px;
        border: 1px solid #E2E8F0;
        border-radius: 11px;
        background: #FFFFFF;
        color: #334155;
    }
    .evidence-list li {margin: 6px 0; line-height: 1.55;}
    .review-box {
        padding: 12px 14px;
        border-left: 4px solid #F59E0B;
        border-radius: 0 10px 10px 0;
        background: #FFFBEB;
        color: #78350F;
        margin-top: 10px;
    }
    @media (max-width: 950px) {
        .decision-grid,.direction-grid {grid-template-columns: 1fr;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe_text(value: Any, fallback: str = "-") -> str:
    text = str(value or "").strip()
    return text or fallback


MAPPING_TYPE_LABELS = {
    "existing_attribute_existing_label": "匹配现有标签",
    "existing_attribute_new_label": "现有属性下的新标签机会",
    "new_attribute_new_label": "新属性机会",
    "multi_attribute_cluster": "多属性混合主题",
    "mixed_or_invalid_cluster": "混合或无效主题",
    "uncertain": "暂无法确定",
    "invalid_term": "无效或不完整表达",
}


def _mapping_label(value: Any) -> str:
    raw = str(value or "").strip()
    return MAPPING_TYPE_LABELS.get(raw, "待人工判断")


def _representative_terms(record: Any, limit: int | None = None) -> list[str]:
    evidence = getattr(record, "evidence", None) or {}
    items = evidence.get("representative_terms") or []
    terms = []
    for item in items:
        term = item.get("ngram") if isinstance(item, dict) else item
        text = str(term or "").strip()
        if text and text not in terms:
            terms.append(text)
    if not terms:
        term_evidence = evidence.get("term_evidence") or {}
        text = str(term_evidence.get("ngram") or evidence.get("term") or "").strip()
        if text:
            terms.append(text)
    return terms[:limit] if limit else terms


def _join_terms(record: Any, limit: int = 10) -> str:
    terms = _representative_terms(record, limit=limit)
    return "、".join(terms) if terms else "-"


def _extract_reason_items(parsed: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("observed_evidence", "key_driver_terms"):
        raw = parsed.get(key) or []
        if isinstance(raw, list):
            for item in raw:
                text = str(item or "").strip()
                if text and text not in values:
                    values.append(text)
    return values


def _response_text(response: Any) -> str:
    return response.choices[0].message.content or ""


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def _call(
    task: Any,
    api_key: str,
    base_url: str,
    model: str,
    thinking: bool,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.user_input},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking:
        kwargs["extra_body"] = {"enable_thinking": True}
    response = client.chat.completions.create(**kwargs)
    return _response_text(response), {
        **_usage(response),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _validate(task: Any, text: str) -> Any:
    if task.analysis_unit == "cluster":
        return parse_and_validate_internal(text, task.evidence)
    return parse_and_validate_noise_term(text, task.evidence)


def _result_rows(cache: ClusterCache) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in cache.list_records(
        phase="internal",
        schema_valid_only=True,
        active_only=True,
    ):
        parsed = record.get("parsed_result") or {}
        primary = parsed.get("primary_mapping") or {}
        mapping_type = str(parsed.get("mapping_type") or "")
        existing_label = primary.get("existing_label")
        proposed_label = (
            primary.get("proposed_new_label")
            or parsed.get("proposed_new_label")
        )
        suggested_label = existing_label or proposed_label
        rows.append(
            {
                "query_key": record.get("query_key")
                or f"cluster:{record.get('cluster_id')}",
                "对象": parsed.get("cluster_name")
                or parsed.get("term")
                or record.get("query_key"),
                "对象类型": "稳定主题"
                if record.get("analysis_unit", "cluster") == "cluster"
                else "待探索热词",
                "AI 判断": _mapping_label(mapping_type),
                "建议归属属性": primary.get("attribute_name"),
                "建议标签": suggested_label,
                "标签性质": (
                    "现有标签" if existing_label
                    else "候选新标签" if proposed_label
                    else "-"
                ),
                "候选新属性": parsed.get("proposed_new_attribute"),
                "置信度": parsed.get("confidence"),
                "人工复核": "需要" if parsed.get("review_required") else "不需要",
                "外部研究建议": "建议"
                if parsed.get("external_research_recommended") else "未建议",
                "更新时间": record.get("updated_at_utc"),
            }
        )
    return pd.DataFrame(rows)


def _xlsx(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="AI Insights")
    return buffer.getvalue()


def _table_rows(records: list[Any], completed: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        evidence = getattr(record, "evidence", None) or {}
        candidates = evidence.get("taxonomy_candidates") or []
        top = candidates[0] if candidates else {}
        exact = bool(
            any(
                item.get("normalized_exact_match")
                for candidate in candidates
                for item in (candidate.get("candidate_labels") or [])
                if isinstance(item, dict)
            )
        )
        analysis_unit = getattr(record, "analysis_unit", "cluster")
        cluster_id = getattr(record, "cluster_id", -1)
        term = str((evidence.get("term_evidence") or {}).get("ngram") or evidence.get("term") or "").strip()
        display_name = (
            f"Cluster {int(cluster_id)}"
            if analysis_unit == "cluster"
            else term or "待探索热词"
        )
        rows.append(
            {
                "选择": False,
                "query_key": getattr(record, "query_key", ""),
                "Cluster ID": int(cluster_id) if analysis_unit == "cluster" else None,
                "分析对象": display_name,
                "对象类型": "稳定主题" if analysis_unit == "cluster" else "待探索热词",
                "代表热词": _join_terms(record, limit=10),
                "Top 候选属性": top.get("attribute_name") or top.get("attribute_code"),
                "属性相似度": top.get("similarity"),
                "标签匹配": "有" if exact else "无",
                "状态": "已完成" if str(getattr(record, "query_key", "")) in completed else "待分析",
                "_unit_order": 0 if analysis_unit == "cluster" else 1,
                "_cluster_sort": int(cluster_id) if analysis_unit == "cluster" else 10**9,
                "_trend_sort": float((evidence.get("term_evidence") or {}).get("trend_score") or 0),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["_unit_order", "_cluster_sort", "_trend_sort"],
        ascending=[True, True, False],
        kind="stable",
    ).drop(columns=["_unit_order", "_cluster_sort", "_trend_sort"])
    return frame.reset_index(drop=True)


def _result_search_text(
    query_key: str,
    result_row: dict[str, Any],
    source_record: Any | None,
) -> str:
    """Build the searchable text for one saved AI result."""
    parts = [
        query_key,
        str(result_row.get("对象") or ""),
        str(result_row.get("对象类型") or ""),
    ]
    if source_record is not None:
        parts.extend(_representative_terms(source_record))
        analysis_unit = getattr(source_record, "analysis_unit", "cluster")
        cluster_id = getattr(source_record, "cluster_id", None)
        if analysis_unit == "cluster" and cluster_id is not None:
            parts.extend(
                [
                    str(cluster_id),
                    f"Cluster {cluster_id}",
                    f"Cluster:{cluster_id}",
                    f"Cluster{cluster_id}",
                ]
            )
        else:
            term = _join_terms(source_record, limit=1)
            parts.extend([term, f"Noise Term {term}", f"NoiseTerm {term}"])
    return " ".join(part for part in parts if part).casefold()


def _result_option_label(
    query_key: str,
    result_row: dict[str, Any],
    source_record: Any | None,
) -> str:
    """Create a compact but identifiable result option label."""
    title = _safe_text(result_row.get("对象"), query_key)
    if source_record is None:
        return f"{title} | {query_key}"

    analysis_unit = getattr(source_record, "analysis_unit", "cluster")
    cluster_id = getattr(source_record, "cluster_id", None)
    if analysis_unit == "cluster":
        identity = f"Cluster {cluster_id}"
        terms = _join_terms(source_record, limit=5)
        return f"{title} | {identity} | {terms} | {query_key}"

    term = _join_terms(source_record, limit=1)
    return f"{title} | Noise Term: {term} | {query_key}"


def _render_result_detail(
    record: dict[str, Any],
    query_key: str,
    source_record: Any | None,
) -> None:
    parsed = record.get("parsed_result") or {}
    primary = parsed.get("primary_mapping") or {}
    secondary = [item for item in (parsed.get("secondary_mappings") or []) if isinstance(item, dict)]
    title = parsed.get("cluster_name") or parsed.get("term") or record.get("query_key")
    analysis_unit = record.get("analysis_unit", "cluster")
    unit_label = "稳定主题" if analysis_unit == "cluster" else "待探索热词"
    mapping_type = str(parsed.get("mapping_type") or "")
    mapping_label = _mapping_label(mapping_type)
    confidence = parsed.get("confidence")
    confidence_text = f"{float(confidence):.0%}" if confidence is not None else "-"
    review_required = bool(parsed.get("review_required"))
    research_recommended = bool(parsed.get("external_research_recommended"))

    source_badge = unit_label
    source_terms: list[str] = []
    if source_record is not None:
        source_unit = getattr(source_record, "analysis_unit", analysis_unit)
        cluster_id = getattr(source_record, "cluster_id", None)
        if source_unit == "cluster":
            source_badge = f"Cluster {cluster_id}"
            source_terms = _representative_terms(source_record, limit=15)
        else:
            source_term = _join_terms(source_record, limit=1)
            source_badge = f"Noise Term · {source_term}"

    source_terms_html = ""
    if source_terms:
        chips = "".join(
            f'<span class="result-term-chip">{html.escape(term)}</span>'
            for term in source_terms
        )
        source_terms_html = (
            '<div class="result-terms">'
            '<span class="result-terms-label">TOP 热词</span>'
            f'{chips}'
            '</div>'
        )

    st.markdown(
        f"""
        <div class="result-hero">
            <div class="result-eyebrow">AI INSIGHT</div>
            <div class="result-title">{html.escape(_safe_text(title))}</div>
            <div class="result-meta">{unit_label} · {html.escape(mapping_label)} · 置信度 {confidence_text}</div>
            <div class="result-source">
                <span class="result-source-badge">{html.escape(source_badge)}</span>
                <span class="result-query-key">{html.escape(query_key)}</span>
            </div>
            {source_terms_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_map = {
        "existing_attribute_existing_label": "当前表达已被现有标签体系覆盖，无需新增标签。",
        "existing_attribute_new_label": "将该表达作为现有属性下的候选新标签。",
        "new_attribute_new_label": "将该方向作为候选新属性及其初始标签。",
        "multi_attribute_cluster": "该主题包含多个业务维度，应同时评估各映射方向，不以单一主属性作为最终结论。",
        "mixed_or_invalid_cluster": "不进入新增属性或标签流程，优先清洗或排除。",
        "invalid_term": "不进入新增属性或标签流程，优先清洗或排除。",
        "uncertain": "当前证据不足以确定归属，建议补充证据后再判断。",
    }
    action_text = action_map.get(mapping_type, "进入人工判断。")
    st.subheader("业务结论")
    st.markdown(
        f"""
        <div class="business-conclusion">
            <div class="business-label">AI 判断</div>
            <div class="business-title">{html.escape(mapping_label)}</div>
            <div class="business-action"><b>建议动作：</b>{html.escape(action_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    observed_evidence = [
        str(item).strip()
        for item in (parsed.get("observed_evidence") or [])
        if str(item or "").strip()
    ]

    def matched_evidence(item: dict[str, Any]) -> str:
        tokens = [
            item.get("attribute_name"),
            item.get("attribute_code"),
            item.get("proposed_new_attribute"),
            item.get("existing_label"),
            item.get("proposed_new_label"),
        ]
        normalized = [
            str(token).strip().casefold()
            for token in tokens
            if str(token or "").strip()
        ]
        for sentence in observed_evidence:
            folded = sentence.casefold()
            if any(token in folded for token in normalized):
                return sentence
        return ""

    def direction_payload(item: dict[str, Any]) -> dict[str, str]:
        attribute = (
            item.get("attribute_name")
            or item.get("proposed_new_attribute")
            or item.get("attribute_code")
            or "待确认属性"
        )
        existing_label = str(item.get("existing_label") or "").strip()
        proposed_label = str(item.get("proposed_new_label") or "").strip()
        recommendation = str(
            item.get("direction_recommendation") or "continue_validation"
        ).strip()
        recommendation_reason = str(
            item.get("direction_recommendation_reason") or ""
        ).strip()

        status_map = {
            "use_existing_label": ("匹配现有标签", "标签", "existing"),
            "direct_addition": ("候选新标签", "候选标签", "candidate"),
            "derived_label": ("建议作为派生标签", "候选派生标签", "derived"),
            "new_attribute": ("候选新属性", "首个标签", "new-attribute"),
            "taxonomy_restructure": ("需要调整标签体系", "候选表达", "review"),
            "continue_validation": ("继续验证", "待验证表达", "review"),
            "not_recommended": ("暂不建议", "评估表达", "neutral"),
        }
        label_status, label_caption, status_key = status_map.get(
            recommendation,
            status_map["continue_validation"],
        )
        label = existing_label if recommendation == "use_existing_label" else proposed_label
        if not label:
            label = existing_label or proposed_label

        reason = str(
            item.get("reason") or item.get("mapping_reason") or ""
        ).strip()
        if not reason:
            reason = matched_evidence(item)
        if not reason:
            reason = "当前结果未提供该方向的单独证据说明。"
        if not recommendation_reason:
            recommendation_reason = "当前结果未提供单独的落地建议理由。"

        return {
            "attribute": str(attribute),
            "label": label,
            "label_status": label_status,
            "label_caption": label_caption,
            "status_key": status_key,
            "reason": reason,
            "recommendation": recommendation,
            "recommendation_reason": recommendation_reason,
        }

    directions: list[dict[str, str]] = []
    root_new_attribute = str(parsed.get("proposed_new_attribute") or "").strip()
    root_new_label = str(parsed.get("proposed_new_label") or "").strip()

    # Primary new-attribute output is split across Root identity fields and
    # primary_mapping reasoning fields. Combine them into one display direction.
    if primary:
        primary_for_display = dict(primary)
        primary_has_identity = bool(
            primary_for_display.get("attribute_name")
            or primary_for_display.get("proposed_new_attribute")
            or primary_for_display.get("attribute_code")
        )
        if root_new_attribute and not primary_has_identity:
            primary_for_display["proposed_new_attribute"] = root_new_attribute
            primary_for_display["proposed_new_label"] = root_new_label or None
            primary_for_display.setdefault(
                "direction_recommendation", "new_attribute"
            )
        directions.append(direction_payload(primary_for_display))

    directions.extend(direction_payload(item) for item in secondary)

    # A separate Root card is allowed only when Primary already represents a
    # genuinely different, identified direction.
    primary_has_separate_identity = bool(
        primary
        and (
            primary.get("attribute_name")
            or primary.get("proposed_new_attribute")
            or primary.get("attribute_code")
        )
    )
    if root_new_attribute and primary_has_separate_identity:
        directions.append(
            direction_payload(
                {
                    "proposed_new_attribute": root_new_attribute,
                    "proposed_new_label": root_new_label or None,
                    "reason": parsed.get("review_reason") or "",
                    "direction_recommendation": "new_attribute",
                    "direction_recommendation_reason": (
                        parsed.get("review_reason")
                        or "根级候选新属性机会，需确认属性名称、首个标签及赋值规则。"
                    ),
                }
            )
        )

    unique_directions: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for direction in directions:
        key = (
            direction["attribute"].casefold(),
            direction["label"].casefold(),
        )
        if key not in seen:
            seen.add(key)
            unique_directions.append(direction)

    if unique_directions:
        st.subheader("映射方向")
        palette = {
            "existing": ("#167854", "#EEF9F4", "#CDEBDB"),
            "candidate": ("#315F9C", "#EEF5FF", "#CCDDF4"),
            "derived": ("#4A67A1", "#F1F5FF", "#D6E0F5"),
            "new-attribute": ("#6850A5", "#F5F1FF", "#DDD3F4"),
            "review": ("#A35B22", "#FFF7EE", "#F1DCC6"),
            "neutral": ("#667085", "#F7F8FA", "#E2E6EC"),
        }
        cards = []
        for index, direction in enumerate(unique_directions, 1):
            label_value = direction["label"] or "尚无具体表达"
            foreground, background, border = palette.get(
                direction["status_key"], palette["review"]
            )
            cards.append(
                f'<div class="direction-card" style="border-color:{border}">'
                f'<div class="direction-kicker">方向 {index:02d}</div>'
                f'<div class="direction-title">{html.escape(direction["attribute"])}</div>'
                f'<div class="direction-status" style="color:{foreground};background:{background};border:1px solid {border}">'
                f'{html.escape(direction["label_status"])}</div>'
                f'<div class="direction-row"><b>{html.escape(direction["label_caption"])}：</b>'
                f'{html.escape(label_value)}</div>'
                f'<div class="direction-evidence"><b>方向依据：</b>'
                f'{html.escape(direction["reason"])}</div>'
                f'<div style="margin-top:12px;padding:11px 12px;border-radius:10px;background:{background};border-left:3px solid {foreground};font-size:13px;color:#53657A">'
                f'<b style="display:block;color:{foreground};margin-bottom:3px">建议理由</b>'
                f'{html.escape(direction["recommendation_reason"])}</div>'
                f'</div>'
            )
        st.markdown(
            '<div class="direction-grid">' + ''.join(cards) + '</div>',
            unsafe_allow_html=True,
        )

    st.subheader("判断依据")
    summary = parsed.get("trend_summary") or parsed.get("reasoning_summary")
    if summary:
        st.markdown(
            f'<div class="reason-summary">{html.escape(str(summary))}</div>',
            unsafe_allow_html=True,
        )


    drivers = parsed.get("key_driver_terms") or []
    drivers = [str(item).strip() for item in drivers if str(item or "").strip()]
    if drivers:
        chips = ''.join(f'<span class="driver-chip">{html.escape(item)}</span>' for item in drivers[:15])
        st.markdown('<div class="section-note">核心驱动热词</div><div class="driver-wrap">' + chips + '</div>', unsafe_allow_html=True)

    if not summary and not drivers:
        st.info("当前结果没有提供可单独展示的判断依据。")

    review_reason = str(parsed.get("review_reason") or "").strip()
    st.caption(
        f"人工复核：{'需要' if review_required else '不需要'} · "
        f"外部研究：{'建议' if research_recommended else '未建议'}"
    )
    if review_reason:
        st.markdown(
            f'<div class="review-box"><b>复核重点：</b>{html.escape(review_reason)}</div>',
            unsafe_allow_html=True,
        )




context = render_workspace_selector(key_prefix="insights")
OUT = context.insights_dir
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cluster_internal_cache.jsonl"
ERROR = OUT / "cluster_internal_errors.jsonl"
CALL_HISTORY = OUT / "internal_call_history.jsonl"

ai_config = render_ai_sidebar(
    model_options=MODEL_OPTIONS,
    base_url_options=BASE_URL_OPTIONS,
    default_model=DEFAULT_MODEL,
    default_base_url=BASE_URL,
    thinking_default=INTERNAL_ENABLE_THINKING,
)
api_key = ai_config["api_key"]
model = ai_config["model"]
base_url = ai_config["base_url"]
workers = ai_config["workers"]
thinking = ai_config["thinking"]

# Persist non-widget values so the same AI configuration survives page changes.
st.session_state["app_api_key"] = api_key
st.session_state["app_model_name"] = model
st.session_state["app_base_url"] = base_url
st.session_state["app_max_workers"] = int(workers)

st.title(f"🧠 {context.category_name} AI Insights")
st.caption(
    "基于热词与标签体系证据，判断现有属性映射、新属性机会和新标签机会。"
)

internal_notice = st.session_state.pop("internal_combined_notice", None)
if internal_notice:
    render_run_notice(
        success=internal_notice.get("success", 0), failed=internal_notice.get("failed", 0),
        unprocessed=internal_notice.get("unprocessed", 0), input_tokens=internal_notice.get("input_tokens", 0),
        output_tokens=internal_notice.get("output_tokens", 0), total_tokens=internal_notice.get("total_tokens", 0),
        runtime_error=internal_notice.get("runtime_error", ""), label="AI Insights · 本轮运行",
    )

path = context.cluster_evidence_file
if not path.exists():
    st.error(
        "当前品类 / Run 尚未生成标签体系 Evidence，请先完成标签体系对比。"
    )
    st.stop()

try:
    records, _ = load_cluster_evidence(path, strict=True)
except Exception as exc:
    st.error(f"Evidence 加载失败：{exc}")
    st.stop()

cache = ClusterCache(CACHE, ERROR)
record_by_key = {record.query_key: record for record in records}
active = cache.list_records(
    phase="internal",
    schema_valid_only=True,
    active_only=True,
)
completed = {
    str(record.get("query_key") or f"cluster:{record.get('cluster_id')}")
    for record in active
}
cluster_count = sum(record.analysis_unit == "cluster" for record in records)
noise_count = sum(record.analysis_unit == "noise_term" for record in records)

st.markdown(
    f"""
    <div class="overview-strip summary-overview">
        <span class="summary-group">
            <span class="summary-item summary-total"><span class="summary-value">{len(records):,}</span>分析对象</span>
            <span class="summary-operator">=</span>
            <span class="summary-item summary-stable"><span class="summary-value">{cluster_count:,}</span>稳定主题</span>
            <span class="summary-operator">+</span>
            <span class="summary-item summary-explore"><span class="summary-value">{noise_count:,}</span>待探索热词</span>
        </span>
        <span class="summary-group">
            <span class="summary-item summary-done"><span class="summary-value">{len(completed):,}</span>已完成</span>
            <span class="summary-operator">+</span>
            <span class="summary-item summary-pending"><span class="summary-value">{max(0, len(records)-len(completed)):,}</span>待分析</span>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("选择分析对象")
filter_columns = st.columns([2, 1, 1])
with filter_columns[0]:
    search_text = st.text_input(
        "搜索",
        placeholder="搜索主题名称、热词或 Query Key",
    )
with filter_columns[1]:
    unit_filter = st.selectbox(
        "对象类型",
        ["全部", "稳定主题", "待探索热词"],
    )
with filter_columns[2]:
    status_filter = st.selectbox(
        "处理状态",
        ["待分析", "全部", "已完成"],
    )

filtered = records
if unit_filter == "稳定主题":
    filtered = [record for record in filtered if record.analysis_unit == "cluster"]
elif unit_filter == "待探索热词":
    filtered = [record for record in filtered if record.analysis_unit == "noise_term"]
if status_filter == "待分析":
    filtered = [record for record in filtered if record.query_key not in completed]
elif status_filter == "已完成":
    filtered = [record for record in filtered if record.query_key in completed]
query = search_text.strip().casefold()
if query:
    filtered = [
        record
        for record in filtered
        if query in record.query_key.casefold()
        or query in str(record.cluster_id).casefold()
        or query in " ".join(record.representative_term_names).casefold()
    ]

selection_frame = _table_rows(filtered, completed)
if selection_frame.empty:
    st.info("当前筛选条件下没有可选对象。")
    selected_keys: list[str] = []
else:
    select_all = st.checkbox("选择当前筛选结果中的全部对象", value=False)
    if select_all:
        selection_frame["选择"] = True
    edited = st.data_editor(
        selection_frame,
        hide_index=True,
        use_container_width=True,
        height=390,
        disabled=[
            column
            for column in selection_frame.columns
            if column != "选择"
        ],
        column_config={
            "选择": st.column_config.CheckboxColumn("选择", width="small"),
            "query_key": None,
            "Cluster ID": st.column_config.NumberColumn("Cluster ID", width="small", format="%d"),
            "分析对象": st.column_config.TextColumn("分析对象", width="medium"),
            "对象类型": st.column_config.TextColumn("对象类型", width="small"),
            "代表热词": st.column_config.TextColumn("代表热词", width="large"),
            "Top 候选属性": st.column_config.TextColumn("Top 候选属性", width="medium"),
            "属性相似度": st.column_config.NumberColumn("属性相似度", width="small", format="%.3f"),
            "标签匹配": st.column_config.TextColumn("标签匹配", width="small"),
            "状态": st.column_config.TextColumn("状态", width="small"),
        },
        key=f"insights_selection_{context.category_code}_{context.run_id}",
    )
    selected_keys = edited.loc[edited["选择"], "query_key"].astype(str).tolist()

selected_records = [record_by_key[key] for key in selected_keys]
estimated_calls = len(selected_keys)
st.markdown(
    f"""
    <div class="selection-strip">
        本次选择 <strong>{len(selected_keys)}</strong> 个对象 ·
        将重新分析 <strong>{estimated_calls}</strong> 个 ·
        模型 <strong>{model}</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

start = st.button(
    "🚀 生成 AI Insights",
    type="primary",
    disabled=not api_key or not selected_records,
    use_container_width=True,
)
if start:
    try:
        pending, cached = prepare_internal_tasks(
            selected_records, cache=cache, model=model, force_rerun=True,
            category_code=context.category_code, category_name=context.category_name,
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()
    if not pending:
        st.warning("没有生成待调用任务，请检查选择状态和任务构建逻辑。")
        st.stop()

    st.info(f"已开始分析 {len(pending)} 个对象。")
    progress = st.progress(0.0)
    live_status = st.empty()
    run_log = st.container(height=320)
    done = success_count = failure_count = 0
    run_rows: list[dict[str, Any]] = []
    run_tokens = {"input": 0, "output": 0, "total": 0}

    def work(task: Any) -> tuple[Any, tuple[str, dict[str, Any]]]:
        return task, _call(task, api_key, base_url, model, thinking, INTERNAL_MAX_OUTPUT_TOKENS, INTERNAL_TEMPERATURE)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, task) for task in pending]
        for future in as_completed(futures):
            task = None
            text = ""
            metadata: dict[str, Any] = {}
            validation = None
            status = "runtime_error"
            error_message = ""
            try:
                task, (text, metadata) = future.result()
                validation = _validate(task, text)
                if validation.valid:
                    status = "success"
                    cache.put_success(
                        phase="internal", query_key=task.query_key, cluster_id=task.cluster_id,
                        signature=task.signature, parsed_result=validation.parsed, evidence_hash=task.evidence_hash,
                        analysis_unit=task.analysis_unit, raw_output_text=text, model=model,
                        prompt_version=task.prompt_version, schema_version=task.schema_version,
                        validation_warnings=validation.warnings, api_result=metadata,
                    )
                    success_count += 1
                else:
                    status = "schema_error"
                    error_message = validation.error_text
                    cache.put_error(
                        phase="internal", query_key=task.query_key, cluster_id=task.cluster_id,
                        signature=task.signature, error_code=validation.error_code or "SCHEMA_VALIDATION_ERROR",
                        error_message=validation.error_text, analysis_unit=task.analysis_unit,
                        raw_output_text=text, validation_errors=validation.errors,
                        validation_warnings=validation.warnings, **metadata,
                    )
                    failure_count += 1
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                cache.put_error(
                    phase="internal", query_key=getattr(task, "query_key", "unknown"),
                    cluster_id=getattr(task, "cluster_id", -1), error_code="RUNTIME_ERROR",
                    error_message=error_message, analysis_unit=getattr(task, "analysis_unit", "cluster"), **metadata,
                )
                failure_count += 1

            for target, source in (("input", "input_tokens"), ("output", "output_tokens"), ("total", "total_tokens")):
                run_tokens[target] += int(metadata.get(source, 0) or 0)
            run_rows.append({
                "query_key": getattr(task, "query_key", "unknown"),
                "analysis_unit": getattr(task, "analysis_unit", None), "status": status,
                "error_message": error_message, "prompt_version": getattr(task, "prompt_version", ""),
                "schema_version": getattr(task, "schema_version", ""),
                "input_tokens": int(metadata.get("input_tokens", 0) or 0),
                "output_tokens": int(metadata.get("output_tokens", 0) or 0),
                "total_tokens": int(metadata.get("total_tokens", 0) or 0),
                "elapsed_seconds": metadata.get("elapsed_seconds", 0),
                "system_prompt": getattr(task, "system_prompt", ""),
                "user_prompt": getattr(task, "user_input", ""), "raw_output_text": text,
                "validation_warnings": getattr(validation, "warnings", []) if validation else [],
                "validation_errors": getattr(validation, "errors", []) if validation else [],
            })
            # PERSISTENT_CALL_HISTORY_V1
            history_row = run_rows[-1]
            append_call_history(
                CALL_HISTORY,
                phase="internal",
                business_key=str(history_row.get("query_key") or "unknown"),
                query_key=str(history_row.get("query_key") or "unknown"),
                category_code=context.category_code,
                category_name=context.category_name,
                run_id=context.run_id,
                cluster_id=getattr(task, "cluster_id", -1),
                analysis_unit=str(history_row.get("analysis_unit") or "cluster"),
                status=str(history_row.get("status") or "runtime_error"),
                model=model,
                prompt_version=str(history_row.get("prompt_version") or ""),
                schema_version=str(history_row.get("schema_version") or ""),
                signature=str(getattr(task, "signature", "") or ""),
                system_prompt=str(history_row.get("system_prompt") or ""),
                user_prompt=str(history_row.get("user_prompt") or ""),
                raw_output_text=str(history_row.get("raw_output_text") or ""),
                parsed_result=(
                    getattr(validation, "parsed", None)
                    if validation is not None and getattr(validation, "valid", False)
                    else None
                ),
                validation_errors=history_row.get("validation_errors") or [],
                validation_warnings=history_row.get("validation_warnings") or [],
                error_code=(
                    getattr(validation, "error_code", "")
                    if history_row.get("status") == "schema_error"
                    else "RUNTIME_ERROR" if history_row.get("status") == "runtime_error" else ""
                ),
                error_message=str(history_row.get("error_message") or ""),
                input_tokens=history_row.get("input_tokens", 0),
                output_tokens=history_row.get("output_tokens", 0),
                total_tokens=history_row.get("total_tokens", 0),
                elapsed_seconds=history_row.get("elapsed_seconds", 0),
            )
            done += 1
            progress.progress(done / max(len(pending), 1), text=f"{done}/{len(pending)}")
            with live_status.container():
                render_live_status(completed=done, total=len(pending), success=success_count, failed=failure_count, total_tokens=run_tokens["total"])
            with run_log:
                icon = "✅" if status == "success" else "❌"
                detail = f" · {error_message}" if error_message else ""
                st.write(f"{icon} {getattr(task, 'query_key', 'unknown')} — {status}{detail}")

    st.session_state.internal_last_run_rows = run_rows
    st.session_state.internal_combined_notice = {
        "success": success_count, "failed": failure_count,
        "unprocessed": max(0, len(pending) - len(run_rows)),
        "input_tokens": run_tokens["input"], "output_tokens": run_tokens["output"],
        "total_tokens": run_tokens["total"], "runtime_error": "",
    }
    st.rerun()

st.divider()
cache.reload()
results = _result_rows(cache)

st.subheader("结果详情")
if results.empty:
    st.info("尚无有效结果，请先选择分析对象并生成 AI Insights。")
else:
    result_rows_by_key = {
        str(row["query_key"]): row.to_dict()
        for _, row in results.iterrows()
    }
    result_search = st.text_input(
        "搜索分析结果",
        placeholder="输入主题名称、热词或 Query Key",
        key=f"insights_result_search_{context.category_code}_{context.run_id}",
    )
    result_query = result_search.strip().casefold()
    filtered_result_keys = [
        query_key
        for query_key, row in result_rows_by_key.items()
        if not result_query
        or result_query in _result_search_text(
            query_key,
            row,
            record_by_key.get(query_key),
        )
    ]

    if not filtered_result_keys:
        st.info("没有找到匹配的分析结果，请尝试主题名称、热词或 Query Key。")
    else:
        detail_key = st.selectbox(
            "选择分析结果",
            filtered_result_keys,
            format_func=lambda query_key: _result_option_label(
                query_key,
                result_rows_by_key[query_key],
                record_by_key.get(query_key),
            ),
            key=f"insights_result_select_{context.category_code}_{context.run_id}",
        )
        source_record = record_by_key.get(detail_key)
        detail_record = cache.latest_for_query_key(
            phase="internal",
            query_key=detail_key,
            active_only=False,
        )
        if detail_record:
            _render_result_detail(detail_record, detail_key, source_record)

# Keep diagnostics visually separated from the selected result and review focus.
st.markdown(
    '<div style="height:22px;border-bottom:1px solid #E8EEF6;margin:0 0 18px 0;"></div>',
    unsafe_allow_html=True,
)

# PERSISTENT_CALL_HISTORY_V1_UI
with st.expander("AI 调用记录", expanded=False):
    render_persistent_call_history(
        read_call_history(CALL_HISTORY),
        title="AI Insights 调用记录",
        key_prefix=f"internal_history_{context.category_code}_{context.run_id}",
        phase="internal",
        empty_message=(
            "当前 Run 还没有可查看的 AI 调用记录。旧版本结果可能只有分析缓存；"
            "重新分析后，会从本次调用开始完整记录。"
        ),
    )
