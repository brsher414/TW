"""Presentation constants and context-aware dashboard cache paths."""
from pathlib import Path
from core.project_context import ProjectContext
DEFAULT_CLUSTER_TOP_N=10
DEFAULT_TERM_TOP_N=10
MAX_CLUSTER_LINES=64
SCOPE_OPTIONS={"当前完整结果":"full","展示至 04 内部解释":"internal","仅展示 LLM 前数据":"raw"}
def dashboard_cache_paths(context:ProjectContext)->dict[str,Path]:
    root=context.dashboard_cache_dir
    return {"term_period_docfreq":root/"term_period_docfreq.parquet","cluster_period_ngram_sum":root/"cluster_period_ngram_sum.parquet","cluster_period_product_coverage":root/"cluster_period_product_coverage.parquet","period_cache_summary":root/"period_cache_summary.json","period_validation_detail":root/"period_validation_detail.parquet"}
EXCLUSION_LABELS={"MIXED_OR_INVALID_CLUSTER":"Cluster 主题混杂或无效","LABEL_EVIDENCE_WITHHELD":"需先确认现有标签覆盖","KNOWN_PACKSIZE_ATTRIBUTE":"已知现有属性，不作为新属性研究","EXISTING_LABEL_TREND_DISABLED":"已有标签趋势研究未开启","EXISTING_ATTRIBUTE_LABEL_UNKNOWN":"标签新旧状态无法确认","NO_RESEARCHABLE_MAPPING":"未形成可研究机会"}
EXCLUSION_DETAILS={"MIXED_OR_INVALID_CLUSTER":"04 判断该 Cluster 同时包含多个无关业务主题或噪声，无法形成可靠的单一外部研究主题。","LABEL_EVIDENCE_WITHHELD":"该方向已匹配到现有属性，但当前 Taxonomy Retrieval 未提供标签明细。","KNOWN_PACKSIZE_ATTRIBUTE":"规格已属于 PACKSIZE 现有属性，不应再次包装为新属性。","EXISTING_LABEL_TREND_DISABLED":"04 已确认该主题属于现有标签，当前未启用已有标签趋势研究。","EXISTING_ATTRIBUTE_LABEL_UNKNOWN":"标签新旧状态仍需 Taxonomy Review。","NO_RESEARCHABLE_MAPPING":"04 没有形成可独立研究的方向。"}
