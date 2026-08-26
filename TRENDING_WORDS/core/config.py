"""TRENDING_WORDS MVP unified configuration."""

# -----------------------------------------------------------------------------
# API and model configuration
# -----------------------------------------------------------------------------
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BASE_URL_DEEPSEEK = "https://api.deepseek.com"
BASE_URL_OPTIONS = [BASE_URL, BASE_URL_DEEPSEEK]

MODEL_OPTIONS = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "DeepSeek-V4-Flash-0731",
    "DeepSeek-V4-Pro-0813"
    "qwen3.8-max",
    "qwen3.7-plus",
    "qwen3.7-plus-2026-05-26",
    "qwen3.7-max-2026-05-17",
    "qwen3.7-max-preview",
    "qwen3.7-flash",
    "qwen3.7-flash-2026-07-15",
]
DEFAULT_MODEL = "qwen3.8-max"

DEFAULT_WORKERS = 5
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------
OUTPUT_DIR = "output"
INTERNAL_LLM_OUTPUT_DIR = "output/internal_llm"
EXTERNAL_RESEARCH_OUTPUT_DIR = "output/external_research"

# -----------------------------------------------------------------------------
# AI Insights configuration
# -----------------------------------------------------------------------------
INTERNAL_TEMPERATURE = 0.1
INTERNAL_MAX_OUTPUT_TOKENS = 30000
INTERNAL_ENABLE_THINKING = True
INTERNAL_ENABLE_FALLBACK = False
INTERNAL_DEFAULT_WORKERS = 5

# -----------------------------------------------------------------------------
# AI Research configuration
# -----------------------------------------------------------------------------
EXTERNAL_TEMPERATURE = 0.1
EXTERNAL_MAX_OUTPUT_TOKENS = 30000
EXTERNAL_ENABLE_THINKING = True

# External Research must be supported by verifiable web-tool execution.
# Do not fall back to a no-tool response that would later fail evidence checks.
EXTERNAL_ENABLE_FALLBACK = False

EXTERNAL_DEFAULT_WORKERS = 3
EXTERNAL_TOOLS = [
    {"type": "web_search"},
    {"type": "web_extractor"},
]

# -----------------------------------------------------------------------------
# Prompt and schema versions
# -----------------------------------------------------------------------------
# Generic version constants retained for backward compatibility with prompt and
# schema modules that still import these names directly.
INTERNAL_PROMPT_VERSION = "internal_v14_direction_contract_consolidated"
INTERNAL_SCHEMA_VERSION = "internal_schema_v10_direction_contract_consolidated"
EXTERNAL_PROMPT_VERSION = "external_topic_v10_direction_alignment"
EXTERNAL_SCHEMA_VERSION = "external_topic_schema_v8_direction_alignment"

# Dedicated contracts imported by core.cluster_loader.
# Cluster aliases preserve the current consolidated contract version.
INTERNAL_CLUSTER_PROMPT_VERSION = INTERNAL_PROMPT_VERSION
INTERNAL_CLUSTER_SCHEMA_VERSION = INTERNAL_SCHEMA_VERSION

# Noise Term contracts use distinct identifiers so Cluster and Noise metadata
# remain distinguishable even though both currently inherit the consolidated
# internal contract generation.
INTERNAL_NOISE_PROMPT_VERSION = f"{INTERNAL_PROMPT_VERSION}_noise_term"
INTERNAL_NOISE_SCHEMA_VERSION = f"{INTERNAL_SCHEMA_VERSION}_noise_term"

# -----------------------------------------------------------------------------
# Legacy/default output paths
# -----------------------------------------------------------------------------
TREND_RESULT_RELATIVE_DIR = (
    "data/final_trend_outputs_2025Q1_vs_2026Q1_n2_5_len120_min800_base200_coh1.0/"
    "bge_m3_cluster_outputs_growth_gt_0.4"
)
TAXONOMY_RETRIEVAL_RELATIVE_DIR = (
    f"{TREND_RESULT_RELATIVE_DIR}/taxonomy_retrieval"
)
CLUSTER_EVIDENCE_JSONL = (
    f"{TAXONOMY_RETRIEVAL_RELATIVE_DIR}/cluster_llm_evidence.jsonl"
)
CLUSTER_EVIDENCE_PRETTY_JSON = (
    f"{TAXONOMY_RETRIEVAL_RELATIVE_DIR}/cluster_llm_evidence_pretty.json"
)

# -----------------------------------------------------------------------------
# Business enums
# -----------------------------------------------------------------------------
ALLOWED_MAPPING_TYPES = [
    "existing_attribute_existing_label",
    "existing_attribute_new_label",
    "new_attribute_new_label",
    "multi_attribute_cluster",
    "mixed_or_invalid_cluster",
    "uncertain",
]

ALLOWED_CLUSTER_QUALITY = [
    "high",
    "medium",
    "low",
    "mixed_or_invalid",
]

ALLOWED_REVIEW_STATUS = [
    "PENDING",
    "APPROVED",
    "REVISED",
    "REJECTED",
]

ALLOWED_EXTERNAL_RESEARCH_STATUS = [
    "completed",
    "insufficient_evidence",
    "error",
]

ALLOWED_EXTERNAL_TOPIC_TYPES = [
    "existing_label_trend",
    "new_label_opportunity",
    "new_attribute_opportunity",
    "uncertain_opportunity",
]
