"""Hot-word Dashboard: discovery, Cluster comparison and taxonomy alignment."""
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dashboard_charts import (
    cluster_absolute_growth_bar,
    cluster_growth_rate_bar,
    cluster_terms_period_chart,
    single_term_period_chart,
    umap_diagnostic_chart,
)
from core.dashboard_loader import load_dashboard_sources
from core.workspace_selector import render_workspace_selector

st.set_page_config(page_title="热词研究看板", page_icon="📈", layout="wide")
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
st.markdown("""
<style>
.block-container{padding-top:1.25rem;padding-bottom:3rem;max-width:1500px}
[data-testid="stSidebar"]{border-right:1px solid #E2E8F0}
.hero{padding:18px 20px;border:1px solid #DDE7F5;border-radius:16px;background:linear-gradient(125deg,#F8FAFC 0%,#EFF6FF 100%);margin:8px 0 16px;box-shadow:0 8px 24px rgba(15,23,42,.045)}
.hero .eyebrow{font-size:12px;letter-spacing:.12em;color:#64748B;font-weight:700}.hero .term{font-size:34px;font-weight:750;line-height:1.15;color:#0F172A;margin:5px 0 7px}.hero .meta{font-size:14px;color:#475569}
.overview{display:flex;gap:12px 24px;flex-wrap:wrap;padding:10px 15px;margin:4px 0 20px;border:1px solid #E2E8F0;border-radius:12px;background:#F8FAFC;color:#475569;font-size:14px}.overview strong{font-size:18px;color:#0F172A;margin-right:4px}
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:8px 0 10px}.compare-card{border:1px solid #DDE7F5;border-radius:14px;background:#fff;padding:16px 18px;box-shadow:0 8px 20px rgba(15,23,42,.035)}.compare-card.term{border-top:4px solid #3B82F6}.compare-card.cluster{border-top:4px solid #0F766E}.compare-title{font-size:13px;font-weight:700;color:#64748B;margin-bottom:12px}.compare-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.compare-value{font-size:25px;font-weight:750;color:#0F172A}.compare-label{font-size:12px;color:#64748B;margin-top:4px}.insight-strip{padding:10px 14px;border-radius:10px;background:#F8FAFC;border:1px solid #E2E8F0;color:#334155;font-size:14px;margin:8px 0 18px}
.tax-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:8px 0 6px}.tax-card{border:1px solid #E2E8F0;border-radius:14px;background:#fff;padding:15px 16px;box-shadow:0 7px 20px rgba(15,23,42,.035)}.tax-rank{font-size:11px;font-weight:800;letter-spacing:.12em;color:#94A3B8}.tax-name{font-size:18px;font-weight:720;color:#0F172A;margin:5px 0 12px}.tax-score-row{display:flex;justify-content:space-between;font-size:12px;color:#64748B;margin-bottom:6px}.tax-track{height:7px;background:#EAF0F7;border-radius:999px;overflow:hidden}.tax-fill{height:100%;background:linear-gradient(90deg,#93C5FD,#2563EB);border-radius:999px}.tax-badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}.tax-badge{display:inline-block;padding:3px 8px;border-radius:999px;background:#EFF6FF;color:#1D4ED8;font-size:12px}.tax-empty{display:inline-block;padding:3px 8px;border-radius:999px;background:#F1F5F9;color:#64748B;font-size:12px}
.section-note{font-size:13px;color:#64748B;margin-top:-5px;margin-bottom:10px}@media(max-width:950px){.compare-grid,.tax-grid{grid-template-columns:1fr}}
</style>
""", unsafe_allow_html=True)


def fnum(value: Any, default: float = 0.0) -> float:
    try: value = float(value)
    except (TypeError, ValueError): return default
    return value if pd.notna(value) else default


def inum(value: Any, default: int = 0) -> int:
    return int(fnum(value, default))


def cluster_label(cluster_id: int, top_terms: Any) -> str:
    terms = str(top_terms or "").replace(";", "、").strip("、 ")
    return f"Cluster {cluster_id}" + (f" · {terms}" if terms else "")


def status(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"已生成 EVIDENCE", "已完成", "COMPLETED", "SUCCESS", "ACTIVE"}: return "✓ 已对比"
    if text in {"失败", "FAILED", "ERROR"}: return "! 失败"
    return "○ 待对比"


def rank_by_score(frame: pd.DataFrame, value: float) -> tuple[int, int]:
    scores = pd.to_numeric(frame["trend_score"], errors="coerce").dropna()
    return int((scores > value).sum()) + 1, len(scores)


def taxonomy_cards(query_key: str, evidence_by_key: dict[str, dict[str, Any]]) -> None:
    evidence = evidence_by_key.get(query_key) or {}
    candidates = evidence.get("taxonomy_candidates") or []
    st.subheader("标签体系对比")
    st.markdown('<div class="section-note">候选属性按语义相似度排序；相似度仅用于候选间比较，不等于业务置信度。</div>', unsafe_allow_html=True)
    if not candidates:
        st.info("当前对象暂无可展示的候选属性。")
        return
    maximum = max([fnum(item.get("similarity")) for item in candidates[:3]] + [1e-9])
    cards=[]
    for index, item in enumerate(candidates[:3], 1):
        name=html.escape(str(item.get("attribute_name") or item.get("attribute_code") or "未知属性"))
        similarity=fnum(item.get("similarity")); width=max(0.0,min(100.0,100.0*similarity/maximum))
        matches=[html.escape(str(label.get("label"))) for label in (item.get("candidate_labels") or []) if isinstance(label,dict) and label.get("normalized_exact_match")]
        badges="".join(f'<span class="tax-badge">{label}</span>' for label in matches) if matches else '<span class="tax-empty">暂无精确匹配</span>'
        cards.append(f'<div class="tax-card"><div class="tax-rank">候选 {index:02d}</div><div class="tax-name">{name}</div><div class="tax-score-row"><span>语义相似度</span><strong>{similarity:.3f}</strong></div><div class="tax-track"><div class="tax-fill" style="width:{width:.1f}%"></div></div><div class="tax-badges">{badges}</div></div>')
    st.markdown('<div class="tax-grid">'+''.join(cards)+'</div>',unsafe_allow_html=True)


context=render_workspace_selector(key_prefix="dashboard")
st.title(f"📈 {context.category_name}热词研究")
st.caption(f"{context.category_code} · {context.period['base_quarter']} vs {context.period['current_quarter']} · Run {context.run_id}")
try: data=load_dashboard_sources("raw",context)
except Exception as exc: st.error(f"看板加载失败：{exc}");st.stop()
all_terms=data["trend_clustered"].copy();normal=data["normal_terms"].copy();noise=data["noise_terms"].copy();summary=data["cluster_summary"].copy()
if not summary.empty:
    summary["cluster_id"]=pd.to_numeric(summary["cluster_id"],errors="coerce");summary=summary[summary["cluster_id"].ge(0)].copy()
labels={int(row.cluster_id):cluster_label(int(row.cluster_id),getattr(row,"top_terms","")) for row in summary.itertuples(index=False)}
st.markdown(
    f"""
    <div class="overview summary-overview">
        <span class="summary-group">
            <span class="summary-item summary-total"><span class="summary-value">{len(all_terms):,}</span>热词</span>
            <span class="summary-operator">=</span>
            <span class="summary-item summary-stable"><span class="summary-value">{len(normal):,}</span>已稳定归簇</span>
            <span class="summary-operator">+</span>
            <span class="summary-item summary-explore"><span class="summary-value">{len(noise):,}</span>待探索</span>
        </span>
        <span class="summary-group">
            <span class="summary-item summary-topic"><span class="summary-value">{len(summary):,}</span>语义主题</span>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("热词")
a,b,c=st.columns([2.2,1,1])
with a: keyword=st.text_input("搜索",placeholder="输入热词、Cluster ID 或主题代表词")
with b: sort_label=st.selectbox("排序方式",["增长率","增长量","当前规模","热词排序分"])
with c: scope=st.selectbox("归簇状态",["全部","已归簇","待探索"])
view=normal.copy() if scope=="已归簇" else noise.copy() if scope=="待探索" else all_terms.copy()
view["cluster_id"]=pd.to_numeric(view["cluster_id"],errors="coerce").fillna(-1).astype(int);view["cluster_status"]=view["cluster_id"].map(lambda x:"待探索" if x==-1 else "已归簇");view["cluster_display"]=view["cluster_id"].map(lambda x:"待探索" if x==-1 else labels.get(x,f"Cluster {x}"));view["growth_pct"]=pd.to_numeric(view["growth_rate"],errors="coerce")*100;view["taxonomy_display"]=(view["taxonomy_status"] if "taxonomy_status" in view else pd.Series("",index=view.index)).map(status)
if keyword.strip():
    search=view["ngram"].astype(str)+" "+view["cluster_display"].astype(str);view=view[search.str.contains(keyword.strip(),case=False,regex=False,na=False)]
sort_col={"增长率":"growth_rate","增长量":"absolute_growth","当前规模":"current_count","热词排序分":"trend_score"}[sort_label];view=view.sort_values(sort_col,ascending=False,kind="stable")
if view.empty: st.warning("当前筛选条件下没有热词。");st.stop()
cols=["ngram","growth_pct","absolute_growth","current_count","base_count","trend_score","cluster_status","cluster_display","source_type","taxonomy_display"]
event=st.dataframe(view[[x for x in cols if x in view]],hide_index=True,use_container_width=True,height=540,on_select="rerun",selection_mode="single-row",key=f"hotword_table_{context.category_code}_{context.run_id}",column_config={"ngram":st.column_config.TextColumn("热词",width="medium"),"growth_pct":st.column_config.NumberColumn("增长率",format="%.1f%%",width="small"),"absolute_growth":st.column_config.NumberColumn("增长量",format="%+d",width="small"),"current_count":st.column_config.NumberColumn("当前规模",format="%,d",width="small"),"base_count":st.column_config.NumberColumn("基准规模",format="%,d",width="small"),"trend_score":st.column_config.NumberColumn("热词排序分",format="%.3f",help="仅用于同一 Run 内相对排序。"),"cluster_status":"归簇状态","cluster_display":st.column_config.TextColumn("语义主题",width="large"),"source_type":"来源类型","taxonomy_display":"标签体系"})
rows=list(getattr(event.selection,"rows",[]) or [])
if not rows: st.info("请点击上方热词表中的任意一行，查看详细分析。");st.stop()
selected=view.iloc[int(rows[0])];term=str(selected["ngram"]);cid=int(selected["cluster_id"]);is_noise=cid==-1;query_key=str(selected.get("query_key") or "")
identity="待探索 · 尚未形成稳定语义主题" if is_noise else labels.get(cid,f"Cluster {cid}")
st.markdown(f'<div class="hero"><div class="eyebrow">当前热词</div><div class="term">{html.escape(term)}</div><div class="meta">{html.escape(identity)}</div></div>',unsafe_allow_html=True)
period=data.get("term_period_docfreq",pd.DataFrame()).copy();evidence=data.get("evidence_by_query_key",{}) or {}
if is_noise:
    rank,total=rank_by_score(all_terms,fnum(selected.get("trend_score")))
    st.markdown(f'<div class="compare-card term"><div class="compare-title">当前热词</div><div class="compare-stats"><div><div class="compare-value">{fnum(selected.get("growth_rate")):.1%}</div><div class="compare-label">增长率</div></div><div><div class="compare-value">{inum(selected.get("absolute_growth")):+,}</div><div class="compare-label">增长量</div></div><div><div class="compare-value">{inum(selected.get("current_count")):,}</div><div class="compare-label">当前规模</div></div></div></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="insight-strip">按热词排序分：全部热词中第 <b>{rank} / {total}</b></div>',unsafe_allow_html=True)
    st.subheader("月度走势");found=period[period["ngram"].astype(str).eq(term)] if not period.empty and "ngram" in period else pd.DataFrame()
    if found.empty: st.warning("月度缓存中未找到当前热词。")
    else: st.plotly_chart(single_term_period_chart(period,term),use_container_width=True)
    taxonomy_cards(query_key,evidence)
else:
    cluster_row=summary[pd.to_numeric(summary["cluster_id"],errors="coerce").eq(cid)].iloc[0];cluster_terms=normal[pd.to_numeric(normal["cluster_id"],errors="coerce").eq(cid)].copy();wg=fnum(selected.get("growth_rate"));cg=fnum(cluster_row.get("mean_growth_rate"));wa=inum(selected.get("absolute_growth"));ca=inum(cluster_terms["absolute_growth"].sum());wc=inum(selected.get("current_count"));cc=inum(cluster_terms["current_count"].sum());rank,total=rank_by_score(cluster_terms,fnum(selected.get("trend_score")));delta=(wg-cg)*100;contrib=100*wa/ca if ca else 0
    st.markdown(f'<div class="compare-grid"><div class="compare-card term"><div class="compare-title">当前热词</div><div class="compare-stats"><div><div class="compare-value">{wg:.1%}</div><div class="compare-label">增长率</div></div><div><div class="compare-value">{wa:+,}</div><div class="compare-label">增长量</div></div><div><div class="compare-value">{wc:,}</div><div class="compare-label">当前规模</div></div></div></div><div class="compare-card cluster"><div class="compare-title">所属语义主题</div><div class="compare-stats"><div><div class="compare-value">{cg:.1%}</div><div class="compare-label">平均增长率</div></div><div><div class="compare-value">{ca:+,}</div><div class="compare-label">主题总增长量</div></div><div><div class="compare-value">{cc:,}</div><div class="compare-label">主题当前总规模</div></div></div></div></div>',unsafe_allow_html=True)
    relation="高" if delta>=0 else "低";st.markdown(f'<div class="insight-strip">当前热词增长率比主题平均<b>{relation} {abs(delta):.1f} 个百分点</b> · 贡献主题增长量的 <b>{contrib:.1f}%</b> · 按热词排序分：主题内第 <b>{rank} / {total}</b></div>',unsafe_allow_html=True)
    title_col,control_col=st.columns([3,1]);title_col.subheader("主题内热词对比");choices=[x for x in [8,10,12,15,20] if x<=max(1,len(cluster_terms))] or [max(1,len(cluster_terms))];top_n=control_col.selectbox("显示词数",choices,index=min(2,len(choices)-1))
    left,right=st.columns(2,gap="medium")
    with left: st.plotly_chart(cluster_growth_rate_bar(cluster_terms,top_n=top_n,selected_term=term),use_container_width=True)
    with right: st.plotly_chart(cluster_absolute_growth_bar(cluster_terms,top_n=top_n,selected_term=term),use_container_width=True)
    st.subheader("月度走势")
    if data.get("period_cache_available",False):
        available=cluster_terms["ngram"].astype(str).drop_duplicates().tolist();peers=cluster_terms[~cluster_terms["ngram"].astype(str).eq(term)].sort_values("trend_score",ascending=False)["ngram"].astype(str).head(4).tolist();compare=st.multiselect("选择同主题热词",available,default=list(dict.fromkeys([term]+peers)))
        if compare: st.plotly_chart(cluster_terms_period_chart(period,compare,selected_term=term,title=f"Cluster {cid} 热词月度走势"),use_container_width=True)
    taxonomy_cards(query_key,evidence)

st.divider();st.subheader("语义空间")
st.plotly_chart(umap_diagnostic_chart(all_terms,summary,show_noise=is_noise,selected_cluster=None if is_noise else cid,selected_term=term),use_container_width=True)
