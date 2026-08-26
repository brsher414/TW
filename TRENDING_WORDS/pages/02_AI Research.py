"""External research page. AI recommendation is advisory, user decides what to research."""
from __future__ import annotations
import html
import json, sys
from io import BytesIO
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.ui_state import render_ai_sidebar
from core.ai_run_ui import inject_ai_run_styles, render_live_status, render_run_notice, render_persistent_call_history
from core.call_history import append_call_history, read_call_history
from core.api_caller import ApiCaller
from core.cluster_cache import ClusterCache,canonical_json
from core.cluster_loader import load_cluster_evidence
from core.config import BASE_URL,BASE_URL_DEEPSEEK,BASE_URL_OPTIONS,DEFAULT_MODEL,EXTERNAL_DEFAULT_WORKERS,EXTERNAL_ENABLE_FALLBACK,EXTERNAL_ENABLE_THINKING,EXTERNAL_MAX_OUTPUT_TOKENS,EXTERNAL_PROMPT_VERSION,EXTERNAL_SCHEMA_VERSION,EXTERNAL_TEMPERATURE,EXTERNAL_TOOLS,MODEL_OPTIONS
from core.external_topic import EXTERNAL_TOPIC_SCHEMA_VERSION, build_topic_signature,build_topics
from core.external_topic_prompt import EXTERNAL_TOPIC_PROMPT_VERSION, EXTERNAL_TOPIC_SYSTEM_PROMPT
from core.external_topic_schema import validate_external_topic
from core.workspace_selector import render_workspace_selector
st.set_page_config(page_title="外部趋势研究", page_icon="🌐", layout="wide")
inject_ai_run_styles()
st.markdown(
    """
    <style>
    .block-container {padding-top:1.25rem; padding-bottom:3rem; max-width:1500px;}
    [data-testid="stSidebar"] {border-right:1px solid #E2E8F0;}
    .research-summary {display:flex; align-items:center; flex-wrap:wrap; padding:11px 15px; margin:6px 0 20px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; box-shadow:0 3px 12px rgba(15,23,42,.025);}
    .summary-group {display:inline-flex; align-items:center; flex-wrap:wrap;}
    .summary-group + .summary-group {margin-left:18px; padding-left:18px; border-left:1px solid #CBD5E1;}
    .summary-item {display:inline-flex; align-items:baseline; gap:4px; color:#64748B; font-size:13px; font-weight:560; white-space:nowrap;}
    .summary-value {font-size:19px; line-height:1; font-weight:760; color:#2563EB;}
    .summary-op {margin:0 11px; color:#94A3B8; font-weight:600;}
    .selection-strip {padding:10px 14px; margin:8px 0 14px; border:1px solid #DBEAFE; border-radius:10px; background:linear-gradient(90deg,#EFF6FF,#F8FAFC); color:#334155; font-size:14px;}
    .research-hero {padding:18px 20px; margin:8px 0 16px; border:1px solid #DDE7F5; border-radius:16px; background:linear-gradient(125deg,#F8FAFC,#EFF6FF); box-shadow:0 8px 24px rgba(15,23,42,.04);}
    .hero-kicker {color:#64748B; font-size:12px; font-weight:700; letter-spacing:.08em;}
    .hero-title {margin:5px 0 10px; color:#0F172A; font-size:27px; font-weight:760;}
    .hero-source {display:flex; align-items:center; flex-wrap:wrap; gap:8px;}
    .source-badge {display:inline-flex; padding:4px 9px; border:1px solid #BFDBFE; border-radius:999px; background:#FFF; color:#1D4ED8; font-size:12px; font-weight:700;}
    .source-key {color:#64748B; font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; font-size:12px;}
    .term-row {display:flex; align-items:center; flex-wrap:wrap; gap:7px; margin-top:11px;}
    .term-label {color:#64748B; font-size:12px; font-weight:700; margin-right:2px;}
    .term-chip {display:inline-flex; padding:4px 9px; border:1px solid #DBEAFE; border-radius:999px; background:#FFF; color:#334155; font-size:12px;}
    .decision-card {padding:17px 19px; margin:8px 0 16px; border:1px solid #DBEAFE; border-left:4px solid #2563EB; border-radius:13px; background:#F8FBFF;}
    .decision-kicker {color:#64748B; font-size:12px; font-weight:700; letter-spacing:.08em;}
    .decision-title {color:#0F172A; font-size:24px; font-weight:760; margin:4px 0 10px;}
    .decision-summary {color:#334155; line-height:1.75; margin-top:12px;}
    .fact-row {display:flex; flex-wrap:wrap; gap:8px; margin-top:12px;}
    .fact-pill {display:inline-flex; gap:5px; padding:5px 10px; border:1px solid #E2E8F0; border-radius:999px; background:#FFF; color:#475569; font-size:12px;}
    .fact-pill b {color:#0F172A;}
    .section-card {padding:16px 18px; margin:8px 0 16px; border:1px solid #E2E8F0; border-radius:13px; background:#FFF; box-shadow:0 4px 14px rgba(15,23,42,.025);}
    .section-title {color:#0F172A; font-size:18px; font-weight:740; margin-bottom:8px;}
    .section-help {color:#64748B; font-size:13px; margin-bottom:10px;}
    .value-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px;}
    .value-card {display:flex; gap:11px; padding:14px; border:1px solid #E2E8F0; border-radius:11px; background:#F8FAFC; color:#334155; line-height:1.65;}
    .value-index {display:flex; align-items:center; justify-content:center; flex:0 0 26px; height:26px; border-radius:50%; background:#DBEAFE; color:#1D4ED8; font-size:12px; font-weight:760;}
    .boundary-box {padding:12px 14px; margin-top:12px; border:1px solid #FED7AA; border-radius:10px; background:#FFF7ED; color:#9A3412; line-height:1.65;}
    .risk-item {padding:9px 0 9px 18px; border-bottom:1px solid #F1F5F9; color:#475569; line-height:1.65; position:relative;}
    .risk-item:last-child {border-bottom:0;}
    .risk-item:before {content:'•'; position:absolute; left:3px; color:#F59E0B; font-weight:800;}
    .evidence-meta {display:flex; flex-wrap:wrap; gap:7px; margin:8px 0 10px;}
    .evidence-tag {display:inline-flex; padding:3px 8px; border-radius:999px; background:#F1F5F9; color:#475569; font-size:12px;}
    @media(max-width:850px){.summary-group + .summary-group{width:100%; margin:10px 0 0; padding:10px 0 0; border-left:0; border-top:1px solid #E2E8F0;} .value-grid{grid-template-columns:1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)

context=render_workspace_selector(key_prefix="research")
INTERNAL_DIR=context.insights_dir;OUT=context.research_dir;OUT.mkdir(parents=True,exist_ok=True)
INTERNAL_CACHE=INTERNAL_DIR/"cluster_internal_cache.jsonl";CACHE=OUT/"cluster_external_json_v2_cache.jsonl";ERRORS=OUT/"cluster_external_json_v2_errors.jsonl";EXPORT_JSONL=OUT/"cluster_external_json_v2_research.jsonl";EXPORT_XLSX=OUT/"cluster_external_json_v2_summary.xlsx";CALL_HISTORY=OUT/"external_call_history.jsonl"
st.session_state.setdefault("external_last_run_rows", [])
def active(r):return str(r.get("record_status","ACTIVE")).upper()!="INVALIDATED"
def latest_internal(cache):
    latest={}
    for r in cache.list_records(phase="internal",schema_valid_only=True):
        if not active(r):continue
        q=str(r.get("query_key") or f"cluster:{int(r.get('cluster_id',-1))}")
        if q not in latest or str(r.get("updated_at_utc",""))>=str(latest[q].get("updated_at_utc","")):latest[q]=r
    return latest
def current_records(cache):
    latest={}
    for r in cache.list_records(phase="external",schema_valid_only=True):
        if not active(r):continue
        p=r.get("parsed_result")
        if not isinstance(p,dict) or "commercial_opportunity" not in p:continue
        q=str(p.get("research_topic_id") or r.get("query_key") or "")
        if q and (q not in latest or str(r.get("updated_at_utc",""))>=str(latest[q].get("updated_at_utc",""))):latest[q]=r
    return sorted(latest.values(),key=lambda r:str(r.get("query_key","")))
def result_row(r):
    p=r.get("parsed_result") or {};o=p.get("commercial_opportunity") or {};f=p.get("external_findings") or [];urls=[x.get("source_url") for x in f if isinstance(x,dict) and x.get("source_url")]
    return {"research_topic_id":p.get("research_topic_id") or r.get("query_key"),"analysis_unit":p.get("analysis_unit",r.get("analysis_unit","cluster")),"source_query_key":p.get("source_query_key",r.get("source_query_key")),"cluster_id":p.get("cluster_id",r.get("cluster_id")),"topic_name":p.get("topic_name"),"topic_type":p.get("topic_type"),"external_research_status":p.get("external_research_status"),"recommendation":o.get("recommendation"),"opportunity_type":o.get("opportunity_type"),"proposed_client_facing_label":o.get("proposed_client_facing_label"),"evidence_strength":o.get("evidence_strength"),"market_breadth":o.get("market_breadth"),"source_count":len(urls),"source_urls":"; ".join(urls),"summary":o.get("summary"),"trend_hypothesis":p.get("trend_hypothesis"),"hypothesis_limitations":p.get("hypothesis_limitations"),"risks_and_limitations":" | ".join(p.get("risks_and_limitations") or []),"review_required":p.get("review_required"),"validation_warnings":" | ".join(r.get("validation_warnings") or []),"total_tokens":r.get("total_tokens",0),"updated_at_utc":r.get("updated_at_utc","")}
def save_exports(cache):
    records=current_records(cache);df=pd.DataFrame([result_row(r) for r in records])
    with EXPORT_JSONL.open("w",encoding="utf-8",newline="\n") as h:
        for r in records:h.write(canonical_json(r["parsed_result"])+"\n")
    df.to_excel(EXPORT_XLSX,index=False);return df
def cache_lookup(cache,topic,signature):
    if hasattr(cache,"lookup_by_query_key"):return cache.lookup_by_query_key(phase="external",query_key=topic.research_topic_id,signature=signature)
    return cache.lookup(phase="external",cluster_id=topic.cluster_id,signature=signature)
TOPIC_TYPE_LABELS = {
    "existing_label_trend": "匹配现有标签（趋势验证）",
    "existing_attribute_existing_label": "匹配现有标签",
    "new_label_opportunity": "现有属性下的新标签机会",
    "existing_attribute_new_label": "现有属性下的新标签机会",
    "new_attribute_opportunity": "新属性机会",
    "new_attribute_new_label": "新属性机会",
    "multi_attribute_topic": "多属性混合主题",
    "multi_attribute_cluster": "多属性混合主题",
    "noise_term_opportunity": "待探索热词",
    "noise_term": "待探索热词",
    "taxonomy_refinement": "标签体系优化",
    "uncertain_opportunity": "暂无法确定的机会",
    "uncertain": "暂无法确定的机会",
}
RECOMMENDATION_LABELS = {
    "use_existing_label": "使用现有标签",
    "direct_addition": "可直接新增",
    "derived_label": "建议作为派生标签",
    "new_attribute": "建议新建属性",
    "taxonomy_restructure": "需要调整现有标签体系",
    "release_candidate": "可直接新增",
    "continue_validation": "继续验证标签机会",
    "not_supported": "暂不支持形成标签",
    "insufficient_evidence": "外部证据不足",
    "recommend": "建议推进",
    "proceed": "建议推进",
    "promote": "建议推进",
    "watchlist": "持续观察",
    "watch": "持续观察",
    "hold": "暂缓推进",
    "not_recommended": "暂不建议",
    "reject": "暂不建议",
}
OPPORTUNITY_TYPE_LABELS = {
    "new_label_under_existing_attribute": "现有属性下的新标签机会",
    "derived_label_from_existing_attribute": "现有属性的派生标签机会",
    "taxonomy_restructure": "标签体系调整机会",
    "new_attribute_with_initial_label": "新属性及其首个标签机会",
    "existing_label_trend_validation": "现有标签趋势验证",
    "insufficient_support": "暂未形成可落地标签机会",
    "taxonomy_refinement": "标签体系优化",
    "new_label": "新标签机会",
    "new_attribute": "新属性机会",
    "existing_label_validation": "现有标签趋势验证",
    "market_signal": "市场趋势信号",
}
EVIDENCE_LABELS = {"strong": "较强", "high": "较强", "moderate": "中等", "medium": "中等", "weak": "较弱", "low": "较弱"}
BREADTH_LABELS = {"multi_source": "多来源支持", "broad": "覆盖较广", "narrow": "覆盖有限", "single_source": "单一来源"}


def business_label(value: Any, mapping: dict[str, str], fallback: str = "待判断") -> str:
    raw = str(value or "").strip()
    return mapping.get(raw, mapping.get(raw.casefold(), raw.replace("_", " ") if raw else fallback))


def clean_display_hotword(value: Any) -> str:
    text = str(value or "").strip()
    while text.startswith("+"):
        text = text[1:].strip()
    return text


def representative_terms(evidence_record: Any, limit: int = 15) -> list[str]:
    evidence = getattr(evidence_record, "evidence", None) or {}
    values: list[str] = []
    for item in evidence.get("representative_terms") or []:
        term = item.get("ngram") if isinstance(item, dict) else item
        text = clean_display_hotword(term)
        if text and text not in values:
            values.append(text)
    if not values:
        text = str((evidence.get("term_evidence") or {}).get("ngram") or evidence.get("term") or "").strip()
        if text:
            values.append(text)
    return values[:limit]


def compact_search(value: Any) -> str:
    text = str(value or "").casefold()
    return "".join(ch for ch in text if ch not in " :：|｜_-\t\r\n")


def searchable(parts: list[Any], query: str) -> bool:
    raw = " ".join(str(part or "") for part in parts).casefold()
    return query in raw or compact_search(query) in compact_search(raw)


def source_label(row: dict[str, Any]) -> str:
    if row.get("analysis_unit") == "cluster":
        return f"Cluster {row.get('cluster_id')}"
    return f"Noise Term · {row.get('term') or '-'}"


def topic_parts(row: dict[str, Any], evidence_record: Any | None) -> list[Any]:
    parts = list(row.values())
    if evidence_record is not None:
        parts.extend(representative_terms(evidence_record, limit=30))
    cluster_id = row.get("cluster_id")
    if row.get("analysis_unit") == "cluster" and cluster_id is not None:
        parts.extend([cluster_id, f"Cluster {cluster_id}", f"Cluster:{cluster_id}", f"Cluster{cluster_id}"])
    return parts


def render_summary_strip(internal_count: int, topic_count: int, noise_count: int, recommended_count: int, completed_count: int) -> None:
    pending = max(0, topic_count - completed_count)
    st.markdown(
        f"""
        <div class="research-summary">
          <div class="summary-group"><span class="summary-item"><span class="summary-value">{internal_count:,}</span> 内部对象</span><span class="summary-op">→</span><span class="summary-item"><span class="summary-value">{topic_count:,}</span> 研究主题</span></div>
          <div class="summary-group"><span class="summary-item"><span class="summary-value">{noise_count:,}</span> 待探索热词</span><span class="summary-op">·</span><span class="summary-item"><span class="summary-value">{recommended_count:,}</span> AI 建议研究</span></div>
          <div class="summary-group"><span class="summary-item"><span class="summary-value">{completed_count:,}</span> 已完成</span><span class="summary-op">+</span><span class="summary-item"><span class="summary-value">{pending:,}</span> 待处理</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar
ai_config = render_ai_sidebar(
    model_options=MODEL_OPTIONS,
    base_url_options=BASE_URL_OPTIONS,
    default_model=DEFAULT_MODEL,
    default_base_url=BASE_URL,
    thinking_default=EXTERNAL_ENABLE_THINKING,
    heading="Qwen 联网配置",
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

st.title(f"🌐 {context.category_name} 外部趋势研究")
st.caption("内部 AI 的是否研究建议仅用于筛选，不是准入门槛。最终由用户手工选择研究主题。")
notice = st.session_state.pop("external_combined_notice", None)
if notice:
    render_run_notice(
        success=notice.get("success", 0), failed=notice.get("failed", 0),
        unprocessed=notice.get("unprocessed", 0), input_tokens=notice.get("input_tokens", 0),
        output_tokens=notice.get("output_tokens", 0), total_tokens=notice.get("total_tokens", 0),
        runtime_error=notice.get("runtime_error", ""), label="AI Research · 本轮运行",
    )

if not INTERNAL_CACHE.exists():st.error("未找到内部解释缓存，请先完成 AI Insights。");st.stop()
try:evidence_records,_=load_cluster_evidence(context.cluster_evidence_file,strict=True)
except Exception as exc:st.error(f"Evidence 加载失败：{exc}");st.stop()
evidence_by_key={r.query_key:r for r in evidence_records};internal_cache=ClusterCache(INTERNAL_CACHE,INTERNAL_DIR/"cluster_internal_errors.jsonl");external_cache=ClusterCache(CACHE,ERRORS);internal=latest_internal(internal_cache)
with st.expander("研究主题生成范围", expanded=False):
    include_existing = st.checkbox("同时研究已有标签趋势", value=False)
    st.caption("默认聚焦潜在新属性、新标签及需要外部验证的趋势。启用后，也会纳入已被现有标签覆盖的趋势。")

topics = []
for q, r in internal.items():
    evidence_record = evidence_by_key.get(q)
    if evidence_record is not None:
        topics.extend(
            build_topics(
                category_code=context.category_code,
                category_name=context.category_name,
                query_key=q,
                cluster_id=evidence_record.cluster_id,
                evidence=evidence_record.evidence,
                internal_record=r,
                include_existing_label_trends=include_existing,
            )
        )
completed = {str(r.get("parsed_result", {}).get("research_topic_id") or r.get("query_key")) for r in current_records(external_cache)}
topic_df = pd.DataFrame([{
    "research_topic_id": t.research_topic_id,
    "analysis_unit": t.analysis_unit,
    "source_query_key": t.source_query_key,
    "cluster_id": t.cluster_id,
    "term": t.term,
    "ai_research_recommended": t.ai_research_recommended,
    "topic_name": t.topic_name,
    "topic_type": t.topic_type,
    "direction_recommendation": t.direction_recommendation,
    "direction_recommendation_reason": t.direction_recommendation_reason,
    "proposed_new_attribute": t.proposed_new_attribute,
    "proposed_new_label": t.proposed_new_label,
} for t in topics])
noise_count = int(topic_df.get("analysis_unit", pd.Series(dtype=str)).eq("noise_term").sum()) if not topic_df.empty else 0
recommended_count = int(topic_df.get("ai_research_recommended", pd.Series(dtype=bool)).eq(True).sum()) if not topic_df.empty else 0
render_summary_strip(len(internal), len(topics), noise_count, recommended_count, len(completed))
if topic_df.empty:
    st.info("当前没有可研究主题。")
    st.stop()

st.markdown("### 1. 选择并执行研究")
search_text = st.text_input("搜索研究主题", placeholder="输入主题名称、Cluster、热词、属性、标签或 Topic ID")
f1, f2, f3, f4 = st.columns(4)
with f1:
    execution_scope = st.selectbox("处理状态", ["仅未处理", "全部", "仅已完成"])
with f2:
    unit_scope = st.selectbox("分析对象", ["全部", "稳定 Cluster", "待探索热词"])
with f3:
    recommendation_scope = st.selectbox("内部 AI 建议", ["全部", "AI 建议研究", "AI 未建议研究"])
with f4:
    raw_recommendations = sorted(
        topic_df.direction_recommendation.dropna().unique().tolist()
    )
    selected_recommendation = st.selectbox(
        "AI Insight 建议",
        ["全部", *raw_recommendations],
        format_func=lambda value: (
            "全部"
            if value == "全部"
            else business_label(value, RECOMMENDATION_LABELS)
        ),
    )

filtered = topic_df.copy()
if selected_recommendation != "全部":
    filtered = filtered[
        filtered.direction_recommendation.eq(selected_recommendation)
    ]
if unit_scope == "稳定 Cluster":
    filtered = filtered[filtered.analysis_unit.eq("cluster")]
elif unit_scope == "待探索热词":
    filtered = filtered[filtered.analysis_unit.eq("noise_term")]
if recommendation_scope == "AI 建议研究":
    filtered = filtered[filtered.ai_research_recommended.eq(True)]
elif recommendation_scope == "AI 未建议研究":
    filtered = filtered[filtered.ai_research_recommended.eq(False)]
if execution_scope == "仅未处理":
    filtered = filtered[~filtered.research_topic_id.isin(completed)]
elif execution_scope == "仅已完成":
    filtered = filtered[filtered.research_topic_id.isin(completed)]
query = search_text.strip().casefold()
if query:
    filtered = filtered[filtered.apply(lambda row: searchable(topic_parts(row.to_dict(), evidence_by_key.get(str(row["source_query_key"]))), query), axis=1)]

selection_rows = []
for _, row in filtered.iterrows():
    evidence_record = evidence_by_key.get(str(row["source_query_key"]))
    selection_rows.append({
        "选择": False,
        "research_topic_id": row["research_topic_id"],
        "研究主题": row["topic_name"],
        "来源对象": source_label(row.to_dict()),
        "代表热词": "、".join(representative_terms(evidence_record, limit=10)) if evidence_record else "-",
        "AI Insight 建议": business_label(
            row["direction_recommendation"], RECOMMENDATION_LABELS
        ),
        "建议理由": row["direction_recommendation_reason"] or "-",
        "候选方向": row["proposed_new_attribute"] or row["proposed_new_label"] or "-",
        "AI 建议": "建议" if bool(row["ai_research_recommended"]) else "未建议",
        "状态": "已完成" if row["research_topic_id"] in completed else "待处理",
    })
selection_frame = pd.DataFrame(selection_rows)
selected_ids = []
if selection_frame.empty:
    st.info("当前筛选条件下没有研究主题。")
else:
    select_all = st.checkbox("选择当前筛选结果中的全部主题", value=False)
    if select_all:
        selection_frame["选择"] = True
    edited = st.data_editor(
        selection_frame,
        hide_index=True,
        use_container_width=True,
        height=390,
        disabled=[c for c in selection_frame.columns if c != "选择"],
        column_config={
            "选择": st.column_config.CheckboxColumn("选择", width="small"),
            "research_topic_id": None,
            "研究主题": st.column_config.TextColumn("研究主题", width="large"),
            "来源对象": st.column_config.TextColumn("来源对象", width="medium"),
            "代表热词": st.column_config.TextColumn("代表热词", width="large"),
            "AI Insight 结论": st.column_config.TextColumn("AI Insight 结论", width="medium"),
            "候选方向": st.column_config.TextColumn("候选方向", width="medium"),
            "AI 建议": st.column_config.TextColumn("AI 建议", width="small"),
            "状态": st.column_config.TextColumn("状态", width="small"),
        },
        key=f"research_selection_{context.category_code}_{context.run_id}",
    )
    selected_ids = edited.loc[edited["选择"], "research_topic_id"].astype(str).tolist()
st.markdown(f'<div class="selection-strip">本次选择 <strong>{len(selected_ids)}</strong> 个主题 · 已完成主题再次选择时将重新研究 · 模型 <strong>{html.escape(model)}</strong></div>', unsafe_allow_html=True)

def stop_call():
    caller=st.session_state.get("external_caller")
    if caller:caller.stop()
b1,b2=st.columns([3,1]);start=b1.button("🚀 开始外部研究",type="primary",use_container_width=True,disabled=not selected_ids or not api_key);b2.button("⏹ 请求停止",on_click=stop_call,use_container_width=True)
if start:
    selected_topics = [
        topic
        for topic in topics
        if topic.research_topic_id in selected_ids
    ]
    pending: list[dict[str, Any]] = []
    task_map: dict[str, tuple[Any, str]] = {}

    for topic in selected_topics:
        signature = build_topic_signature(topic, model)
        pending.append(topic.api_item(signature))
        task_map[topic.research_topic_id] = (topic, signature)

    st.info(
        f"已提交 {len(pending)} 个外部研究任务，"
        "每个任务对应一个选中的研究主题。"
    )

    if not pending:
        st.error("没有生成待调用研究任务，请检查主题选择和 Topic 构建逻辑。")
        st.stop()

    caller = ApiCaller(api_key, base_url, pool_size=workers)
    caller.reset()
    st.session_state.external_caller = caller

    progress = st.progress(0.0)
    live = st.empty()
    log = st.container(height=350)
    rows: list[dict[str, Any]] = []
    tokens = {"input": 0, "output": 0, "total": 0}

    def callback(done, total, item_text, api_result):
        query_key = str(api_result.get("query_key") or "")
        status = "callback_error"
        error_message = ""
        error_code = ""
        topic = None
        signature = ""
        validation = None

        try:
            if not query_key:
                raise KeyError("API 结果缺少 query_key")
            if query_key not in task_map:
                raise KeyError(f"未知 research_topic_id: {query_key}")

            topic, signature = task_map[query_key]
            common = dict(
                phase="external",
                cluster_id=topic.cluster_id,
                query_key=query_key,
                signature=signature,
                model=model,
                prompt_version=EXTERNAL_TOPIC_PROMPT_VERSION,
                schema_version=EXTERNAL_TOPIC_SCHEMA_VERSION,
                evidence_hash=build_topic_signature(topic, "evidence"),
                internal_signature=topic.internal_signature,
            )

            if api_result.get("error"):
                status = "api_error"
                error_code = "API_ERROR"
                error_message = str(api_result["error"])
                external_cache.put_error(
                    **common,
                    error_code="API_ERROR",
                    error_message=str(api_result["error"]),
                    analysis_unit=topic.analysis_unit if topic is not None else "cluster",
                    raw_output_text=api_result.get("output_text", ""),
                    api_error=str(api_result["error"]),
                    input_tokens=api_result.get("input_tokens", 0),
                    output_tokens=api_result.get("output_tokens", 0),
                    total_tokens=api_result.get("total_tokens", 0),
                    elapsed_seconds=api_result.get("elapsed_seconds", 0),
                    degraded=bool(api_result.get("degraded", False)),
                )
            else:
                validation = validate_external_topic(
                    api_result.get("output_text", ""),
                    topic.to_dict(),
                    degraded=bool(api_result.get("degraded", False)),
                )
                if validation.valid:
                    status = "success"
                    external_cache.put_success(
                        **common,
                        parsed_result=validation.parsed,
                        raw_output_text=api_result.get("output_text", ""),
                        api_result=api_result,
                        validation_warnings=validation.warnings,
                        review_status="PENDING",
                        tools=EXTERNAL_TOOLS,
                    )
                else:
                    status = "schema_error"
                    error_code = validation.error_code or "SCHEMA_VALIDATION_ERROR"
                    error_message = validation.error_text
                    external_cache.put_error(
                    **common,
                    error_code=validation.error_code or "SCHEMA_VALIDATION_ERROR",
                    error_message=validation.error_text,
                    analysis_unit=topic.analysis_unit if topic is not None else "cluster",
                    raw_output_text=api_result.get("output_text", ""),
                    validation_errors=validation.errors,
                    validation_warnings=validation.warnings,
                    input_tokens=api_result.get("input_tokens", 0),
                    output_tokens=api_result.get("output_tokens", 0),
                    total_tokens=api_result.get("total_tokens", 0),
                    elapsed_seconds=api_result.get("elapsed_seconds", 0),
                    degraded=bool(api_result.get("degraded", False)),
                )
        except Exception as exc:
            status = "callback_error"
            error_code = "CALLBACK_ERROR"
            error_message = f"{type(exc).__name__}: {str(exc)[:500]}"
        finally:
            for key, source in (
                ("input", "input_tokens"),
                ("output", "output_tokens"),
                ("total", "total_tokens"),
            ):
                tokens[key] += int(api_result.get(source, 0) or 0)

            rows.append(
                {
                    "research_topic_id": query_key,
                    "analysis_unit": (
                        topic.analysis_unit if topic is not None else None
                    ),
                    "source_query_key": (
                        topic.source_query_key if topic is not None else None
                    ),
                    "status": status,
                    "error_message": error_message,
                    "input_tokens": api_result.get("input_tokens", 0),
                    "output_tokens": api_result.get("output_tokens", 0),
                    "total_tokens": api_result.get("total_tokens", 0),
                }
            )

            # PERSISTENT_CALL_HISTORY_V1_EXTERNAL
            try:
                append_call_history(
                    CALL_HISTORY, phase="external", business_key=query_key or "unknown",
                    query_key=query_key or "unknown", research_topic_id=query_key,
                    source_query_key=topic.source_query_key if topic is not None else "",
                    category_code=context.category_code, category_name=context.category_name,
                    run_id=context.run_id, cluster_id=topic.cluster_id if topic is not None else -1,
                    analysis_unit=topic.analysis_unit if topic is not None else "cluster",
                    status=status, model=model, prompt_version=EXTERNAL_TOPIC_PROMPT_VERSION,
                    schema_version=EXTERNAL_TOPIC_SCHEMA_VERSION, signature=signature,
                    evidence_hash=build_topic_signature(topic, "evidence") if topic is not None else "",
                    system_prompt=EXTERNAL_TOPIC_SYSTEM_PROMPT, user_prompt=str(item_text or ""),
                    tools=EXTERNAL_TOOLS, degraded=bool(api_result.get("degraded", False)),
                    raw_output_text=str(api_result.get("output_text", "") or ""),
                    parsed_result=getattr(validation, "parsed", None) if validation is not None and getattr(validation, "valid", False) else None,
                    validation_errors=getattr(validation, "errors", []) if validation is not None else [],
                    validation_warnings=getattr(validation, "warnings", []) if validation is not None else [],
                    error_code=error_code, error_message=error_message,
                    input_tokens=api_result.get("input_tokens", 0), output_tokens=api_result.get("output_tokens", 0),
                    total_tokens=api_result.get("total_tokens", 0), elapsed_seconds=api_result.get("elapsed_seconds", 0),
                )
            except Exception as history_exc:
                rows[-1]["history_error"] = f"{type(history_exc).__name__}: {str(history_exc)[:500]}"

            progress.progress(
                done / total if total else 1.0,
                text=f"{done}/{total}",
            )
            with live.container():
                render_live_status(
                    completed=done, total=total,
                    success=sum(row.get("status") == "success" for row in rows),
                    failed=sum(row.get("status") != "success" for row in rows),
                    total_tokens=tokens["total"],
                )
            with log:
                icon = "✅" if status == "success" else "❌"
                detail = f" · {error_message}" if error_message else ""
                st.write(f"{icon} {query_key or '未知任务'} — {status}{detail}")

    try:
        batch_results = caller.call_batch(
            items=pending,
            model=model,
            system_prompt=EXTERNAL_TOPIC_SYSTEM_PROMPT,
            text_column="__user_input__",
            tools=EXTERNAL_TOOLS,
            enable_thinking=thinking,
            temperature=EXTERNAL_TEMPERATURE,
            max_output_tokens=EXTERNAL_MAX_OUTPUT_TOKENS,
            enable_fallback=EXTERNAL_ENABLE_FALLBACK,
            max_workers=workers,
            progress_callback=callback,
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

    callback_errors = [
        str(result.get("_progress_callback_error"))
        for result in batch_results
        if result.get("_progress_callback_error")
    ]
    expected_count = len(pending)
    processed_count = len(rows)
    unprocessed_count = max(0, expected_count - processed_count)

    success_count = sum(row["status"] == "success" for row in rows)
    failed_rows = [row for row in rows if row["status"] != "success"]
    failed_count = len(failed_rows)

    runtime_errors: list[str] = []
    if processed_count != expected_count:
        runtime_errors.append(
            f"提交 {expected_count} 个研究任务，"
            f"但只收到 {processed_count} 个执行结果。"
        )
    if callback_errors:
        runtime_errors.append(
            "进度回调异常：" + " | ".join(callback_errors[:3])
        )

    st.session_state.external_last_run_rows = rows
    st.session_state.external_combined_notice = {
        "success": success_count,
        "failed": failed_count,
        "unprocessed": unprocessed_count,
        "cache_hits": 0,
        "input_tokens": tokens["input"],
        "output_tokens": tokens["output"],
        "total_tokens": tokens["total"],
        "runtime_error": " ".join(runtime_errors),
    }

    external_cache.reload()
    if success_count > 0:
        save_exports(external_cache)

    if failed_rows or runtime_errors:
        st.error(
            f"本轮研究存在异常：成功 {success_count}，"
            f"失败 {failed_count}，未处理 {unprocessed_count}。"
        )
        with st.expander("查看失败原因", expanded=True):
            st.dataframe(
                pd.DataFrame(failed_rows),
                hide_index=True,
                use_container_width=True,
            )
        st.stop()

    st.rerun()
st.divider()
st.markdown("### 2. 查看研究结论")
external_cache.reload()
records_now = current_records(external_cache)
df = pd.DataFrame([result_row(r) for r in records_now])
if df.empty:
    st.info("暂无有效结果。")
else:
    result_search = st.text_input("搜索研究结果", placeholder="输入主题名称、Cluster、热词、来源 Query Key 或 Topic ID")
    rows_by_id = {str(row["research_topic_id"]): row.to_dict() for _, row in df.iterrows()}
    result_query = result_search.strip().casefold()
    result_ids = []
    for topic_id, row in rows_by_id.items():
        source_key = str(row.get("source_query_key") or "")
        evidence_record = evidence_by_key.get(source_key)
        original_topic = next((t for t in topics if str(t.research_topic_id) == topic_id), None)
        parts = topic_parts(row, evidence_record)
        if original_topic is not None:
            parts.extend([original_topic.term, original_topic.proposed_new_attribute, original_topic.proposed_new_label])
        if not result_query or searchable(parts, result_query):
            result_ids.append(topic_id)
    if not result_ids:
        st.info("没有找到匹配的研究结果，请尝试主题名称、热词、Cluster 或 Query Key。")
    else:
        inspect = st.selectbox(
            "选择研究主题",
            result_ids,
            format_func=lambda q: f"{rows_by_id[q].get('topic_name') or q} | {source_label(rows_by_id[q])} | {q}",
            key=f"research_result_select_{context.category_code}_{context.run_id}",
        )
        r = next(x for x in records_now if str(x.get("parsed_result", {}).get("research_topic_id") or x.get("query_key")) == inspect)
        parsed = r["parsed_result"]
        opportunity = parsed.get("commercial_opportunity") or {}
        source_query_key = str(parsed.get("source_query_key") or "")
        evidence_record = evidence_by_key.get(source_query_key)
        unit = parsed.get("analysis_unit") or "cluster"
        if unit == "cluster":
            source_badge = f"Cluster {parsed.get('cluster_id')}"
            top_terms = representative_terms(evidence_record, limit=15) if evidence_record else []
        else:
            term = (representative_terms(evidence_record, limit=1) or ["-"])[0] if evidence_record else "-"
            source_badge = f"Noise Term · {term}"
            top_terms = []
        terms_html = ""
        if top_terms:
            chips = "".join(f'<span class="term-chip">{html.escape(term)}</span>' for term in top_terms)
            terms_html = f'<div class="term-row"><span class="term-label">TOP 热词</span>{chips}</div>'
        st.markdown(
            f"""
            <div class="research-hero">
              <div class="hero-kicker">研究主题</div>
              <div class="hero-title">{html.escape(str(parsed.get('topic_name') or inspect))}</div>
              <div class="hero-source"><span class="source-badge">{html.escape(source_badge)}</span><span class="source-key">{html.escape(source_query_key)}</span></div>
              {terms_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if unit == "noise_term":
            st.info("这是单词级早期信号研究，外部资料不能直接证明内部词频变化的原因。")

        recommendation = business_label(opportunity.get("recommendation"), RECOMMENDATION_LABELS)
        opportunity_type = business_label(opportunity.get("opportunity_type"), OPPORTUNITY_TYPE_LABELS)
        client_label = str(opportunity.get("proposed_client_facing_label") or "暂未形成候选客户标签")
        evidence_strength = business_label(opportunity.get("evidence_strength"), EVIDENCE_LABELS, "待评估")
        market_breadth = business_label(opportunity.get("market_breadth"), BREADTH_LABELS, "待评估")
        summary = str(opportunity.get("summary") or "当前结果未提供商业机会摘要。")
        recommendation_reason = str(opportunity.get("recommendation_reason") or opportunity.get("label_decision_reason") or "当前结果未提供建议理由。")
        st.markdown(
            f"""
            <div class="decision-card">
              <div class="decision-kicker">商业机会判断</div>
              <div class="decision-title">{html.escape(recommendation)}</div>
              <div><b>机会方向：</b>{html.escape(opportunity_type)}</div>
              <div><b>候选客户标签：</b>{html.escape(client_label)}</div>
              <div class="fact-row">
                <span class="fact-pill"><b>证据强度</b>{html.escape(evidence_strength)}</span>
                <span class="fact-pill"><b>市场覆盖</b>{html.escape(market_breadth)}</span>
                <span class="fact-pill"><b>有效来源</b>{len(parsed.get('external_findings') or [])}</span>
              </div>
              <div class="decision-summary">{html.escape(summary)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        client_values = [str(v).strip() for v in (opportunity.get("client_value") or []) if str(v or "").strip()]
        if client_values:
            value_cards = "".join(f'<div class="value-card"><span class="value-index">{i:02d}</span><span>{html.escape(value)}</span></div>' for i, value in enumerate(client_values, 1))
            st.markdown(f'<div class="section-card"><div class="section-title">对客户意味着什么</div><div class="section-help">将研究结论转化为可用于沟通、标签体系判断或后续验证的业务价值。</div><div class="value-grid">{value_cards}</div></div>', unsafe_allow_html=True)

        hypothesis = str(parsed.get("trend_hypothesis") or "").strip()
        limitation = str(parsed.get("hypothesis_limitations") or "").strip()
        hypothesis_body = html.escape(hypothesis) if hypothesis else "当前结果未形成明确的市场解释。"
        boundary_html = f'<div class="boundary-box"><b>解释边界：</b>{html.escape(limitation)}</div>' if limitation else ""
        st.markdown(f'<div class="section-card"><div class="section-title">市场信号解读</div><div class="section-help">基于外部资料对该机会可能代表的市场方向进行解释，不等同于已经证明内部趋势增长的原因。</div><div>{hypothesis_body}</div>{boundary_html}</div>', unsafe_allow_html=True)

        risks = [str(v).strip() for v in (parsed.get("risks_and_limitations") or []) if str(v or "").strip()]
        if risks:
            risk_html = "".join(f'<div class="risk-item">{html.escape(risk)}</div>' for risk in risks)
            st.markdown(f'<div class="section-card"><div class="section-title">风险与限制</div><div class="section-help">在向客户推荐或调整标签体系前，需要保留的证据边界与验证事项。</div>{risk_html}</div>', unsafe_allow_html=True)

        st.markdown("#### 外部证据")
        st.caption("以下资料用于支撑前述市场信号解读和商业机会判断。")
        findings = parsed.get("external_findings") or []
        if not findings:
            st.info("当前结果没有可展示的外部证据。")
        for i, finding in enumerate(findings, 1):
            title = finding.get("source_title") or "未命名来源"
            with st.expander(f"证据 {i:02d}｜{title}", expanded=i == 1):
                meta = "".join([
                    f'<span class="evidence-tag">{html.escape(str(finding.get("source_type") or "来源类型未知"))}</span>',
                    f'<span class="evidence-tag">{html.escape(str(finding.get("source_date") or "日期未知"))}</span>',
                    f'<span class="evidence-tag">{html.escape(str(finding.get("temporal_relation") or "时间关系未知"))}</span>',
                ])
                st.markdown(f'<div class="evidence-meta">{meta}</div>', unsafe_allow_html=True)
                st.write(finding.get("claim") or "")
                url = str(finding.get("source_url") or "").strip()
                if url.startswith(("https://", "http://")):
                    st.markdown(f"[打开来源]({url})")

# PERSISTENT_CALL_HISTORY_V1_EXTERNAL_UI
with st.expander("AI 调用记录", expanded=False):
    render_persistent_call_history(
        read_call_history(CALL_HISTORY),
        title="AI Research 调用记录",
        key_prefix=f"external_history_{context.category_code}_{context.run_id}",
        phase="external",
        empty_message=(
            "当前 Run 还没有可查看的 AI 调用记录。旧版本研究结果可能只有结果缓存；"
            "完成新的外部研究后，会从本次调用开始完整记录。"
        ),
    )
