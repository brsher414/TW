"""Plotly charts for the hot-word Dashboard."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
BLUE="#2563EB";BLUE_LIGHT="#BFDBFE";TEAL="#0F766E";TEAL_LIGHT="#CCFBF1";ORANGE="#F59E0B";GRID="#E2E8F0";TEXT="#0F172A"
def _top(frame,metric,n):
    s=frame.copy()
    for c in ["base_count","current_count","growth_rate","absolute_growth","cluster_probability",metric]:s[c]=pd.to_numeric(s[c],errors="coerce")
    return s.dropna(subset=["ngram",metric]).sort_values(metric,ascending=False,kind="stable").head(max(1,int(n))).copy()
def _bar(frame,metric,n,selected_term,title,scale,y_title,pct=False):
    s=_top(frame,metric,n);y=s[metric]*100 if pct else s[metric];custom=s[["growth_rate","absolute_growth","base_count","current_count","cluster_probability"]].to_numpy();fig=go.Figure(go.Bar(x=s.ngram,y=y,customdata=custom,marker=dict(color=y,colorscale=scale,showscale=False,line=dict(width=0)),text=[f"{v:.1f}%" if pct else f"+{v:,.0f}" for v in y],textposition="outside",cliponaxis=False,hovertemplate="<b>%{x}</b><br>增长率：%{customdata[0]:.1%}<br>增长量：%{customdata[1]:,.0f}<br>基准规模：%{customdata[2]:,.0f}<br>当前规模：%{customdata[3]:,.0f}<br>归簇置信度：%{customdata[4]:.3f}<extra></extra>"));focus=s[s.ngram.astype(str).eq(str(selected_term))]
    if not focus.empty:
        fy=float(focus.iloc[0][metric])*(100 if pct else 1);fig.add_trace(go.Scatter(x=[focus.iloc[0].ngram],y=[fy],mode="markers",marker=dict(size=11,color=scale[-1][1],symbol="diamond",line=dict(width=1.5,color="#FFFFFF")),hoverinfo="skip",showlegend=False))
    upper=float(y.max()) if len(y) else 0;fig.update_layout(title=dict(text=title,x=.01,xanchor="left",font=dict(size=18,color=TEXT)),xaxis_title=None,yaxis_title=y_title,height=455,showlegend=False,bargap=.3,margin=dict(l=25,r=15,t=60,b=95),paper_bgcolor="#FFF",plot_bgcolor="#FFF");fig.update_yaxes(range=[0,upper*1.18 if upper>0 else 1],gridcolor=GRID,zeroline=False,ticksuffix="%" if pct else None,tickformat=None if pct else ",");fig.update_xaxes(tickangle=-28,showgrid=False);return fig
def cluster_growth_rate_bar(frame,*,top_n=12,selected_term=None,title="增长率"):return _bar(frame,"growth_rate",top_n,selected_term,title,[[0,BLUE_LIGHT],[.55,"#3B82F6"],[1,"#1D4ED8"]],"增长率",True)
def cluster_absolute_growth_bar(frame,*,top_n=12,selected_term=None,title="增长量"):return _bar(frame,"absolute_growth",top_n,selected_term,title,[[0,TEAL_LIGHT],[.55,"#14B8A6"],[1,TEAL]],"增长量",False)
def single_term_period_chart(frame,term):
    s=frame[frame.ngram.astype(str).eq(str(term))].sort_values("period_code");fig=px.line(s,x="period_label",y="docfreq",markers=True,title=f"“{term}”月度走势",color_discrete_sequence=[BLUE]);fig.update_traces(line=dict(width=3),marker=dict(size=7,line=dict(width=1,color="#FFF")));fig.update_layout(xaxis_title="月份",yaxis_title="文档频次",height=420,margin=dict(l=20,r=20,t=58,b=20),hovermode="x unified",showlegend=False,paper_bgcolor="#FFF",plot_bgcolor="#FFF");fig.update_yaxes(gridcolor=GRID,zeroline=False);return fig
def cluster_terms_period_chart(frame,terms,*,selected_term,title):
    s=frame[frame.ngram.astype(str).isin([str(x) for x in terms])];palette=["#94A3B8","#0F766E","#7C3AED","#D97706","#64748B"];fig=go.Figure()
    for i,term in enumerate(terms):
        t=s[s.ngram.astype(str).eq(str(term))].sort_values("period_code");focus=str(term)==str(selected_term)
        if t.empty:continue
        fig.add_trace(go.Scatter(x=t.period_label,y=t.docfreq,mode="lines+markers",name=str(term),line=dict(width=3.2 if focus else 1.8,color=BLUE if focus else palette[i%len(palette)]),marker=dict(size=7 if focus else 4),opacity=1 if focus else .48,hovertemplate=f"<b>{term}</b><br>月份：%{{x}}<br>文档频次：%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(title=dict(text=title,x=.01,xanchor="left",font=dict(size=18,color=TEXT)),xaxis_title="月份",yaxis_title="文档频次",height=455,margin=dict(l=20,r=20,t=58,b=25),hovermode="x unified",legend_title="主题内热词",paper_bgcolor="#FFF",plot_bgcolor="#FFF");fig.update_yaxes(gridcolor=GRID,zeroline=False);return fig
def umap_diagnostic_chart(frame,cluster_summary,*,show_noise,selected_cluster,selected_term=None):
    s=frame.copy();s["cluster_id"]=pd.to_numeric(s["cluster_id"],errors="coerce");s=s.dropna(subset=["umap_x","umap_y","cluster_id"]);is_noise=s.get("is_noise",s.cluster_id.eq(-1));normal=s[(~is_noise.fillna(False).astype(bool))&s.cluster_id.ge(0)];noise=s[is_noise.fillna(False).astype(bool)|s.cluster_id.eq(-1)];fig=go.Figure()
    if not normal.empty:
        custom=normal[["ngram","cluster_id","growth_rate","absolute_growth"]].to_numpy();fig.add_trace(go.Scattergl(x=normal.umap_x,y=normal.umap_y,mode="markers",customdata=custom,marker=dict(size=6.5,opacity=.52,color=normal.cluster_id,colorscale="Turbo",colorbar=dict(title="Cluster",thickness=10,len=.66,outlinewidth=0)),hovertemplate="<b>%{customdata[0]}</b><br>Cluster %{customdata[1]}<br>增长率：%{customdata[2]:.1%}<br>增长量：%{customdata[3]:,.0f}<extra></extra>"))
    if show_noise and not noise.empty:
        custom=noise[["ngram","growth_rate","absolute_growth"]].to_numpy();fig.add_trace(go.Scattergl(x=noise.umap_x,y=noise.umap_y,mode="markers",customdata=custom,marker=dict(size=6,color=ORANGE,opacity=.48,symbol="diamond"),hovertemplate="<b>%{customdata[0]}</b><br>待探索<br>增长率：%{customdata[1]:.1%}<br>增长量：%{customdata[2]:,.0f}<extra></extra>"))
    if selected_cluster is not None:
        focus=normal[normal.cluster_id.eq(int(selected_cluster))]
        if not focus.empty:fig.add_trace(go.Scattergl(x=focus.umap_x,y=focus.umap_y,mode="markers",marker=dict(size=15,color="rgba(37,99,235,.12)",line=dict(width=2,color="rgba(37,99,235,.55)")),hoverinfo="skip"))
    if selected_term:
        focus=s[s.ngram.astype(str).eq(str(selected_term))]
        if not focus.empty:
            custom=focus[["ngram","cluster_id","growth_rate","absolute_growth"]].to_numpy();fig.add_trace(go.Scattergl(x=focus.umap_x,y=focus.umap_y,mode="markers",marker=dict(size=26,color="rgba(37,99,235,.14)",line=dict(width=2.5,color="rgba(37,99,235,.65)")),hoverinfo="skip"));fig.add_trace(go.Scattergl(x=focus.umap_x,y=focus.umap_y,mode="markers",customdata=custom,marker=dict(size=11,color="#1D4ED8",symbol="diamond",line=dict(width=1.5,color="#FFF")),hovertemplate="<b>%{customdata[0]}</b><br>当前热词<br>Cluster %{customdata[1]}<br>增长率：%{customdata[2]:.1%}<br>增长量：%{customdata[3]:,.0f}<extra></extra>"))
    fig.update_layout(title_text="",xaxis_title=None,yaxis_title=None,height=600,showlegend=False,paper_bgcolor="#FFF",plot_bgcolor="#F8FAFC",margin=dict(l=22,r=88,t=8,b=22));fig.layout.title.text="";fig.update_xaxes(showgrid=True,gridcolor=GRID,zeroline=False,showticklabels=False);fig.update_yaxes(showgrid=True,gridcolor=GRID,zeroline=False,showticklabels=False);return fig
