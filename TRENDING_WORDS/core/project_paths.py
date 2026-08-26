from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
BASE_QUARTER = "2025Q1"
CURRENT_QUARTER = "2026Q1"
TREND_OUTPUT_DIR = DATA_DIR / "final_trend_outputs_2025Q1_vs_2026Q1_n2_5_len120_min800_base200_coh1.0"
CLUSTER_OUTPUT_DIR = TREND_OUTPUT_DIR / "bge_m3_cluster_outputs_growth_gt_0.4"
TAXONOMY_SOURCE = DATA_DIR / "YD_SEGMENT_LABEL.xlsx"
TAXONOMY_REFERENCE_DIR = CLUSTER_OUTPUT_DIR / "taxonomy_reference"
TAXONOMY_REFERENCE = TAXONOMY_REFERENCE_DIR / "taxonomy_reference.parquet"
TAXONOMY_SOURCE_NORMALIZED = TAXONOMY_REFERENCE_DIR / "taxonomy_source_normalized.parquet"
TAXONOMY_EMBEDDING_INDEX = TAXONOMY_REFERENCE_DIR / "taxonomy_embedding_index.parquet"
TAXONOMY_EMBEDDINGS = TAXONOMY_REFERENCE_DIR / "taxonomy_attribute_embeddings.npy"
CATEGORY_CONTEXT = TAXONOMY_REFERENCE_DIR / "category_context.json"
CLUSTER_SUMMARY = CLUSTER_OUTPUT_DIR / "cluster_summary.parquet"
TREND_CLUSTERED = CLUSTER_OUTPUT_DIR / "trend_clustered.parquet"
TAXONOMY_RETRIEVAL_DIR = CLUSTER_OUTPUT_DIR / "taxonomy_retrieval"
TAXONOMY_CANDIDATE_REVIEW = TAXONOMY_RETRIEVAL_DIR / "taxonomy_candidate_review.parquet"
CLUSTER_LLM_EVIDENCE = TAXONOMY_RETRIEVAL_DIR / "cluster_llm_evidence.jsonl"
BGE_MODEL_NAME = "BAAI/bge-m3"
TOP_K_ATTRIBUTES = 5
TOP_REPRESENTATIVE_TERMS = 20
