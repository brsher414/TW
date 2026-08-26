"""Shared modern run-status, token and AI-call-history UI."""
from __future__ import annotations

import html
import json
from typing import Any, Iterable

import pandas as pd
import streamlit as st


def _number(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def token_totals(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in totals:
            totals[key] += _number(row.get(key))
    return totals


def render_run_notice(
    *,
    success: int,
    failed: int,
    unprocessed: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    runtime_error: str = "",
    label: str = "本轮运行",
) -> None:
    success = _number(success)
    failed = _number(failed)
    unprocessed = _number(unprocessed)
    input_tokens = _number(input_tokens)
    output_tokens = _number(output_tokens)
    total_tokens = _number(total_tokens)
    runtime_error = str(runtime_error or "").strip()

    if runtime_error or unprocessed:
        tone, icon, title = "danger", "!", "运行未完整结束"
    elif failed:
        tone, icon, title = "warning", "!", "运行完成，存在失败任务"
    else:
        tone, icon, title = "success", "✓", "运行完成"

    detail = runtime_error or (
        f"成功 {success} · 失败 {failed} · 未处理 {unprocessed}"
        if unprocessed
        else f"成功 {success} · 失败 {failed}"
    )
    st.markdown(
        f"""
        <div class="ai-run-panel {tone}">
          <div class="ai-run-head">
            <div class="ai-run-icon">{icon}</div>
            <div>
              <div class="ai-run-kicker">{html.escape(label)}</div>
              <div class="ai-run-title">{html.escape(title)}</div>
              <div class="ai-run-detail">{html.escape(detail)}</div>
            </div>
          </div>
          <div class="ai-token-grid">
            <div class="ai-token-card total"><div class="ai-token-label">TOTAL TOKENS</div><div class="ai-token-value">{total_tokens:,}</div><div class="ai-token-note">本轮模型调用总量</div></div>
            <div class="ai-token-card"><div class="ai-token-label">INPUT</div><div class="ai-token-value">{input_tokens:,}</div><div class="ai-token-note">Prompt 与 Evidence</div></div>
            <div class="ai-token-card"><div class="ai-token-label">OUTPUT</div><div class="ai-token-value">{output_tokens:,}</div><div class="ai-token-note">模型返回内容</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_status(
    *, completed: int, total: int, success: int, failed: int, total_tokens: int
) -> None:
    st.markdown(
        f"""
        <div class="ai-live-strip">
          <div><b>{_number(completed):,}/{_number(total):,}</b><span>已处理</span></div>
          <div><b>{_number(success):,}</b><span>成功</span></div>
          <div><b>{_number(failed):,}</b><span>失败</span></div>
          <div><b>{_number(total_tokens):,}</b><span>Tokens</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_ai_run_styles() -> None:
    st.markdown(
        """
        <style>
        .ai-run-panel{margin:12px 0 22px;padding:18px;border:1px solid #e5eaf2;border-radius:18px;background:linear-gradient(145deg,#fff 0%,#f8fbff 100%);box-shadow:0 10px 30px rgba(31,45,61,.07)}
        .ai-run-panel.success{border-color:#cdebdc}.ai-run-panel.warning{border-color:#f4dfb0}.ai-run-panel.danger{border-color:#f1c7cc}
        .ai-run-head{display:flex;align-items:center;gap:12px;margin-bottom:16px}.ai-run-icon{width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-weight:800;background:#e9f8f1;color:#15835b}
        .warning .ai-run-icon{background:#fff6df;color:#a86b00}.danger .ai-run-icon{background:#fff0f1;color:#c23b48}
        .ai-run-kicker{font-size:11px;font-weight:700;letter-spacing:.08em;color:#8390a3}.ai-run-title{font-size:18px;font-weight:760;color:#182230;margin-top:1px}.ai-run-detail{font-size:13px;color:#667085;margin-top:2px}
        .ai-token-grid{display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:10px}.ai-token-card{padding:14px 15px;border-radius:14px;background:rgba(255,255,255,.9);border:1px solid #e8edf4}.ai-token-card.total{background:linear-gradient(135deg,#edf5ff,#f5f0ff);border-color:#d9e3f5}
        .ai-token-label{font-size:10px;font-weight:760;letter-spacing:.08em;color:#7c899c}.ai-token-value{font-size:25px;line-height:1.15;font-weight:790;color:#182230;margin:4px 0}.ai-token-note{font-size:11px;color:#8a96a8}
        .ai-live-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:8px 0 12px}.ai-live-strip>div{display:flex;align-items:baseline;justify-content:center;gap:6px;padding:10px 12px;background:#f8fafc;border:1px solid #e7ebf1;border-radius:12px}.ai-live-strip b{font-size:17px;color:#1e293b}.ai-live-strip span{font-size:11px;color:#7b8798}
        @media(max-width:760px){.ai-token-grid{grid-template-columns:1fr}.ai-live-strip{grid-template-columns:1fr 1fr}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_persistent_call_history(
    records: list[dict[str, Any]],
    *,
    title: str,
    key_prefix: str,
    phase: str,
    empty_message: str = "当前 Run 还没有可查看的 AI 调用记录。",
) -> None:
    """Render append-only call history with a business-friendly visual summary."""
    valid_records = [row for row in records if isinstance(row, dict)]
    total_calls = len(valid_records)
    success_calls = sum(str(row.get("status") or "") == "success" for row in valid_records)
    exception_calls = total_calls - success_calls
    total_tokens = token_totals(valid_records)["total_tokens"]

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;padding:18px 20px;margin:2px 0 14px;border:1px solid #D9E6F5;border-radius:16px;background:linear-gradient(135deg,#F5FAFF 0%,#EEF5FF 55%,#F6F2FF 100%);box-shadow:0 8px 24px rgba(38,76,120,.08);">
          <div>
            <div style="font-size:10px;font-weight:800;letter-spacing:.14em;color:#7187A6;">AI ACTIVITY</div>
            <div style="font-size:20px;font-weight:780;color:#17345A;margin-top:3px;">{html.escape(title)}</div>
            <div style="font-size:12px;color:#667A94;margin-top:4px;">查看每次真实模型调用的输入、输出、Token 与异常信息</div>
          </div>
          <div style="padding:6px 10px;border-radius:999px;background:#DFEBFF;color:#315F9C;border:1px solid #C8DAF5;font-size:11px;font-weight:760;white-space:nowrap;">当前 Run</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not valid_records:
        st.info(empty_message)
        return

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin:0 0 16px;">
          <div style="padding:11px;text-align:center;background:#F8FAFC;border:1px solid #E4EAF2;border-radius:12px;"><b style="font-size:18px;color:#203A5F;">{total_calls:,}</b><div style="font-size:11px;color:#74849A;">调用次数</div></div>
          <div style="padding:11px;text-align:center;background:#F1FAF6;border:1px solid #D7EDDF;border-radius:12px;"><b style="font-size:18px;color:#167854;">{success_calls:,}</b><div style="font-size:11px;color:#74849A;">成功</div></div>
          <div style="padding:11px;text-align:center;background:#FFF8F3;border:1px solid #F1E0D2;border-radius:12px;"><b style="font-size:18px;color:#B45A2A;">{exception_calls:,}</b><div style="font-size:11px;color:#74849A;">异常</div></div>
          <div style="padding:11px;text-align:center;background:#F3F5FF;border:1px solid #DDE2F5;border-radius:12px;"><b style="font-size:18px;color:#465AA5;">{total_tokens:,}</b><div style="font-size:11px;color:#74849A;">Total Tokens</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    search_col, status_col = st.columns([2, 1])
    with search_col:
        query = st.text_input("搜索调用记录", placeholder="输入主题、热词、Cluster 或 Topic ID", key=f"{key_prefix}_search").strip().casefold()
    statuses = sorted({str(row.get("status") or "unknown") for row in valid_records})
    with status_col:
        selected_status = st.selectbox("状态", ["全部", *statuses], key=f"{key_prefix}_status")

    filtered = []
    for row in valid_records:
        if str(row.get("phase") or "") not in {"", phase}:
            continue
        if selected_status != "全部" and str(row.get("status")) != selected_status:
            continue
        haystack = " ".join(str(row.get(key) or "") for key in ("business_key", "query_key", "research_topic_id", "source_query_key", "cluster_id", "analysis_unit", "model")).casefold()
        if query and query not in haystack:
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: str(row.get("created_at_utc") or ""), reverse=True)
    if not filtered:
        st.info("没有找到符合当前筛选条件的调用记录。")
        return

    summary = pd.DataFrame([{
        "时间": row.get("created_at_utc") or "-",
        "对象": row.get("research_topic_id") or row.get("query_key") or row.get("business_key"),
        "调用次序": row.get("attempt_no", 1),
        "状态": row.get("status") or "unknown",
        "模型": row.get("model") or "-",
        "Total Tokens": _number(row.get("total_tokens")),
        "耗时(秒)": row.get("elapsed_seconds", 0),
        "错误摘要": row.get("error_message") or "-",
    } for row in filtered])
    st.dataframe(summary, hide_index=True, use_container_width=True, height=min(360, 42 + 35 * len(summary)))

    call_ids = [str(row.get("call_id") or f"row-{i}") for i, row in enumerate(filtered)]
    by_id = {call_id: row for call_id, row in zip(call_ids, filtered)}
    selected_call_id = st.selectbox(
        "选择一条记录查看详情",
        call_ids,
        format_func=lambda call_id: (
            f"{by_id[call_id].get('created_at_utc') or '-'} | "
            f"{by_id[call_id].get('research_topic_id') or by_id[call_id].get('query_key') or by_id[call_id].get('business_key')} | "
            f"{by_id[call_id].get('status') or 'unknown'}"
        ),
        key=f"{key_prefix}_detail",
    )
    row = by_id[selected_call_id]
    st.caption(f"第 {row.get('attempt_no', 1)} 次调用 · Prompt {row.get('prompt_version') or '-'} · Schema {row.get('schema_version') or '-'} · {_number(row.get('total_tokens')):,} Tokens")

    prompt_tab, output_tab, validation_tab, metadata_tab = st.tabs(["输入内容", "模型返回", "校验结果", "技术信息"])
    with prompt_tab:
        st.markdown("**System Prompt**")
        st.code(row.get("system_prompt") or "", language="text")
        st.markdown("**User Prompt / Evidence**")
        st.code(row.get("user_prompt") or "", language="json")
        if phase == "external" and row.get("tools"):
            st.markdown("**Tools**")
            st.code(json.dumps(row.get("tools"), ensure_ascii=False, indent=2), language="json")
    with output_tab:
        st.code(row.get("raw_output_text") or "", language="json")
    with validation_tab:
        if row.get("error_code") or row.get("error_message"):
            st.error(f"{row.get('error_code') or 'ERROR'} · {row.get('error_message') or '-'}")
        if row.get("validation_errors"):
            st.error(" | ".join(map(str, row.get("validation_errors") or [])))
        if row.get("validation_warnings"):
            st.warning(" | ".join(map(str, row.get("validation_warnings") or [])))
        if not row.get("error_message") and not row.get("validation_errors") and not row.get("validation_warnings"):
            st.success("本次调用无校验错误或警告。")
    with metadata_tab:
        keys = ("call_id", "attempt_no", "created_at_utc", "category_code", "category_name", "run_id", "query_key", "research_topic_id", "source_query_key", "cluster_id", "analysis_unit", "status", "model", "prompt_version", "schema_version", "signature", "input_tokens", "output_tokens", "total_tokens", "elapsed_seconds", "degraded")
        st.code(json.dumps({key: row.get(key) for key in keys}, ensure_ascii=False, indent=2, default=str), language="json")
