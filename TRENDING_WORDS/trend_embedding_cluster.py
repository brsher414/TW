"""Category-isolated local semantic clustering for TRENDING_WORDS.

Pipeline:
- Load trend outputs from the active category/run.
- Create or reuse local BGE-M3 dense embeddings.
- Reduce with UMAP and cluster with HDBSCAN.
- Keep noise in trend_clustered, but exclude cluster_id=-1 from cluster_summary.
- Write all artifacts to the current run's cluster directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

TRENDING_ROOT = Path(__file__).resolve().parent
if str(TRENDING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRENDING_ROOT))

from core.project_context import ProjectContext
from core.run_manifest import create_manifest, set_active_run, update_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run category-isolated BGE-M3 clustering")
    parser.add_argument("--category", default=None, help="品类代码，例如 YD")
    parser.add_argument("--run-id", default=None, help="可选：覆盖自动生成的 run_id")
    parser.add_argument(
        "--force-rebuild-embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖配置中的 embedding cache 重建开关",
    )
    return parser.parse_args()


ARGS = parse_args()
CONTEXT = (
    ProjectContext.from_category(ARGS.category, project_root=TRENDING_ROOT)
    if ARGS.category
    else ProjectContext.active(project_root=TRENDING_ROOT)
)
if ARGS.run_id:
    CONTEXT = CONTEXT.with_run_id(ARGS.run_id)

CLUSTER_CONFIG = dict(CONTEXT.config.get("cluster", {}))
PROJECT_DIR = TRENDING_ROOT
INPUT_DIR = CONTEXT.trend_dir
OUTPUT_DIR = CONTEXT.cluster_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared model cache across categories and runs. This is model data, not business data.
HF_HOME_DIR = PROJECT_DIR / "models" / "huggingface"
HF_HUB_CACHE_DIR = HF_HOME_DIR / "hub"
HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# These imports must occur after setting Hugging Face environment variables.
import hdbscan
import numpy as np
import polars as pl
import torch
import umap
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

INPUT_FILES = {
    "contained": "trend_contained.parquet",
    "container": "trend_container.parquet",
    "low_cohesion": "trend_low_cohesion.parquet",
}
GROWTH_RATE_THRESHOLD = float(CONTEXT.trend["growth_rate_threshold"])
INCLUDE_LOW_COHESION_IN_CLUSTERING = bool(
    CLUSTER_CONFIG.get("include_low_cohesion", True)
)
BGE_MODEL_PATH = str(CLUSTER_CONFIG.get("embedding_model", "BAAI/bge-m3"))
BGE_MODEL_ID = BGE_MODEL_PATH
BGE_DEVICE = str(CLUSTER_CONFIG.get("device", "cpu"))
BGE_USE_FP16 = bool(CLUSTER_CONFIG.get("use_fp16", False))
BGE_BATCH_SIZE = int(CLUSTER_CONFIG.get("batch_size", 8))
BGE_MAX_LENGTH = int(CLUSTER_CONFIG.get("max_length", 64))
EMBEDDING_DIMENSIONS = int(CLUSTER_CONFIG.get("embedding_dimensions", 1024))
CACHE_SAVE_EVERY_BATCHES = int(CLUSTER_CONFIG.get("cache_save_every_batches", 10))
FORCE_REBUILD_EMBEDDINGS = (
    bool(ARGS.force_rebuild_embeddings)
    if ARGS.force_rebuild_embeddings is not None
    else bool(CLUSTER_CONFIG.get("force_rebuild_embeddings", False))
)
CATEGORY_CONTEXT = str(
    CLUSTER_CONFIG.get("category_context", CONTEXT.category_name)
)
EMBEDDING_TEXT_TEMPLATE = str(
    CLUSTER_CONFIG.get("embedding_text_template", "品类：{category}；商品描述热词：{term}")
)
UMAP_CLUSTER_COMPONENTS = int(CLUSTER_CONFIG.get("umap_cluster_components", 10))
UMAP_N_NEIGHBORS = int(CLUSTER_CONFIG.get("umap_n_neighbors", 15))
UMAP_MIN_DIST_CLUSTER = float(CLUSTER_CONFIG.get("umap_min_dist_cluster", 0.0))
UMAP_METRIC = str(CLUSTER_CONFIG.get("umap_metric", "cosine"))
UMAP_VIS_COMPONENTS = 2
UMAP_MIN_DIST_VIS = float(CLUSTER_CONFIG.get("umap_min_dist_vis", 0.1))
RANDOM_STATE = int(CLUSTER_CONFIG.get("random_state", 42))
HDBSCAN_MIN_CLUSTER_SIZE = int(CLUSTER_CONFIG.get("hdbscan_min_cluster_size", 5))
HDBSCAN_MIN_SAMPLES = int(CLUSTER_CONFIG.get("hdbscan_min_samples", 3))
HDBSCAN_METRIC = str(CLUSTER_CONFIG.get("hdbscan_metric", "euclidean"))
HDBSCAN_CLUSTER_SELECTION_METHOD = str(
    CLUSTER_CONFIG.get("hdbscan_cluster_selection_method", "eom")
)
TOP_TERMS_PER_CLUSTER = int(CLUSTER_CONFIG.get("top_terms_per_cluster", 15))

LOG_PATH = OUTPUT_DIR / "pipeline_run_log.txt"
EMBEDDING_CACHE_PATH = OUTPUT_DIR / "embedding_cache.parquet"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger("bge_cluster")

REQUIRED_COLUMNS = {
    "ngram", "growth_rate", "base_count", "current_count",
    "ngram_len", "total_count", "cohesion",
}
CACHE_SCHEMA = {
    "embedding_text": pl.String,
    "text_hash": pl.String,
    "embedding_model": pl.String,
    "embedding_dimensions": pl.Int64,
    "embedding": pl.List(pl.Float64),
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv_bom(df: pl.DataFrame, path: Path) -> None:
    df.write_csv(path, include_bom=True)


def safe_read_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"输入文件不存在：{path}\n请先运行：uv run main.py --category {CONTEXT.category_code}"
        )
    return pl.read_parquet(path)


def config_dict() -> dict[str, Any]:
    return {
        "category_code": CONTEXT.category_code,
        "category_name": CONTEXT.category_name,
        "run_id": CONTEXT.run_id,
        "project_dir": str(PROJECT_DIR),
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "hf_home": str(HF_HOME_DIR),
        "hf_hub_cache": str(HF_HUB_CACHE_DIR),
        "growth_rate_threshold": GROWTH_RATE_THRESHOLD,
        "include_low_cohesion": INCLUDE_LOW_COHESION_IN_CLUSTERING,
        "bge_model_path": BGE_MODEL_PATH,
        "bge_model_id": BGE_MODEL_ID,
        "bge_device": BGE_DEVICE,
        "bge_batch_size": BGE_BATCH_SIZE,
        "bge_max_length": BGE_MAX_LENGTH,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "cache_save_every_batches": CACHE_SAVE_EVERY_BATCHES,
        "category_context": CATEGORY_CONTEXT,
        "umap_cluster_components": UMAP_CLUSTER_COMPONENTS,
        "umap_n_neighbors": UMAP_N_NEIGHBORS,
        "hdbscan_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
        "hdbscan_min_samples": HDBSCAN_MIN_SAMPLES,
        "random_state": RANDOM_STATE,
        "force_rebuild_embeddings": FORCE_REBUILD_EMBEDDINGS,
    }


def standardize_frame(df: pl.DataFrame, source_type: str, filename: str) -> pl.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{filename} 缺少字段：{sorted(missing)}")
    expressions = [
        pl.lit(source_type).alias("source_type"),
        pl.col("ngram").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("growth_rate").cast(pl.Float64, strict=False),
        pl.col("base_count").cast(pl.Int64, strict=False),
        pl.col("current_count").cast(pl.Int64, strict=False),
        pl.col("ngram_len").cast(pl.Int64, strict=False),
        pl.col("total_count").cast(pl.Int64, strict=False),
        pl.col("cohesion").cast(pl.Float64, strict=False),
        (
            pl.col("parents").cast(pl.String, strict=False).fill_null("")
            if "parents" in df.columns else pl.lit("").alias("parents")
        ),
        (
            pl.col("children").cast(pl.String, strict=False).fill_null("")
            if "children" in df.columns else pl.lit("").alias("children")
        ),
    ]
    return df.with_columns(expressions).filter(
        pl.col("ngram").is_not_null()
        & (pl.col("ngram") != "")
        & pl.col("growth_rate").is_not_null()
        & pl.col("growth_rate").is_finite()
        & (pl.col("growth_rate") > GROWTH_RATE_THRESHOLD)
        & pl.col("base_count").is_not_null()
        & pl.col("current_count").is_not_null()
    )


def load_candidates() -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    frames: list[pl.DataFrame] = []
    counts: dict[str, int] = {}
    for source_type, filename in INPUT_FILES.items():
        frame = standardize_frame(
            safe_read_parquet(INPUT_DIR / filename), source_type, filename
        )
        frames.append(frame)
        counts[source_type] = frame.height
        logger.info("%s: growth_rate > %.3f 后 %d 行", filename, GROWTH_RATE_THRESHOLD, frame.height)

    merged_all = pl.concat(frames, how="diagonal_relaxed")
    low_audit = merged_all.filter(pl.col("source_type") == "low_cohesion")
    merged = (
        merged_all
        if INCLUDE_LOW_COHESION_IN_CLUSTERING
        else merged_all.filter(pl.col("source_type") != "low_cohesion")
    )
    merged = (
        merged.with_columns(
            pl.when(pl.col("source_type") == "container").then(3)
            .when(pl.col("source_type") == "contained").then(2)
            .otherwise(1).alias("_priority")
        )
        .sort(
            ["ngram", "_priority", "current_count", "growth_rate"],
            descending=[False, True, True, True],
        )
        .unique(subset=["ngram"], keep="first", maintain_order=True)
        .drop("_priority")
        .with_columns([
            (pl.col("current_count") - pl.col("base_count")).alias("absolute_growth"),
            (pl.col("growth_rate") * (pl.col("current_count") + 1).log()).alias("trend_score"),
            pl.col("ngram").map_elements(
                lambda term: EMBEDDING_TEXT_TEMPLATE.format(
                    category=CATEGORY_CONTEXT, term=term
                ),
                return_dtype=pl.String,
            ).alias("embedding_text"),
        ])
    )
    if merged.height < max(HDBSCAN_MIN_CLUSTER_SIZE, 3):
        raise ValueError(f"主聚类候选仅 {merged.height} 条，数量不足。")
    if merged["ngram"].n_unique() != merged.height:
        raise RuntimeError("候选词去重失败。")
    logger.info("主聚类唯一候选词：%d", merged.height)
    return merged, low_audit, counts


def empty_cache() -> pl.DataFrame:
    return pl.DataFrame(schema=CACHE_SCHEMA)


def load_embedding_cache() -> pl.DataFrame:
    if FORCE_REBUILD_EMBEDDINGS or not EMBEDDING_CACHE_PATH.exists():
        return empty_cache()
    try:
        cache = pl.read_parquet(EMBEDDING_CACHE_PATH)
    except Exception as exc:
        logger.warning("Embedding cache 读取失败，将重建：%s", exc)
        return empty_cache()
    if not set(CACHE_SCHEMA).issubset(cache.columns):
        logger.warning("Embedding cache 字段不兼容，将重建。")
        return empty_cache()
    return cache.filter(
        (pl.col("embedding_model") == BGE_MODEL_ID)
        & (pl.col("embedding_dimensions") == EMBEDDING_DIMENSIONS)
    )


def save_embedding_cache(cache: pl.DataFrame, rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return cache
    combined = pl.concat(
        [cache, pl.DataFrame(rows, schema=CACHE_SCHEMA)], how="vertical_relaxed"
    ).unique(
        subset=["embedding_text", "embedding_model", "embedding_dimensions"],
        keep="last",
    )
    combined.write_parquet(EMBEDDING_CACHE_PATH)
    return combined


def build_local_embeddings(candidates: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    texts = candidates["embedding_text"].to_list()
    cache = load_embedding_cache()
    cache_map: dict[str, list[float]] = {}
    invalid = 0
    for row in cache.iter_rows(named=True):
        vector = row["embedding"]
        if (
            vector is None
            or len(vector) != EMBEDDING_DIMENSIONS
            or not np.isfinite(np.asarray(vector, dtype=np.float32)).all()
        ):
            invalid += 1
            continue
        cache_map[row["embedding_text"]] = vector

    missing = [text for text in texts if text not in cache_map]
    stats = {
        "cache_hits": len(texts) - len(missing),
        "new_embeddings": len(missing),
        "invalid_cache_rows": invalid,
    }
    logger.info(
        "Embedding cache 命中 %d；本地新增 %d；无效缓存 %d",
        stats["cache_hits"], stats["new_embeddings"], invalid,
    )
    if missing:
        logger.info("模型将下载/读取到 HF_HOME=%s", HF_HOME_DIR)
        model = BGEM3FlagModel(
            BGE_MODEL_PATH, use_fp16=BGE_USE_FP16, device=BGE_DEVICE
        )
        pending: list[dict[str, Any]] = []
        total_batches = math.ceil(len(missing) / BGE_BATCH_SIZE)
        for start in range(0, len(missing), BGE_BATCH_SIZE):
            batch = missing[start:start + BGE_BATCH_SIZE]
            batch_no = start // BGE_BATCH_SIZE + 1
            logger.info("BGE-M3 batch %d/%d；数量 %d", batch_no, total_batches, len(batch))
            result = model.encode(
                batch,
                batch_size=BGE_BATCH_SIZE,
                max_length=BGE_MAX_LENGTH,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = np.asarray(result["dense_vecs"], dtype=np.float32)
            if vectors.shape != (len(batch), EMBEDDING_DIMENSIONS):
                raise RuntimeError(f"BGE-M3 输出 shape 异常：{vectors.shape}")
            if not np.isfinite(vectors).all():
                raise RuntimeError(f"BGE-M3 batch {batch_no} 含 NaN/inf")
            for text, vector in zip(batch, vectors):
                values = vector.astype(float).tolist()
                cache_map[text] = values
                pending.append({
                    "embedding_text": text,
                    "text_hash": stable_hash(text),
                    "embedding_model": BGE_MODEL_ID,
                    "embedding_dimensions": EMBEDDING_DIMENSIONS,
                    "embedding": values,
                })
            if batch_no % CACHE_SAVE_EVERY_BATCHES == 0 or batch_no == total_batches:
                cache = save_embedding_cache(cache, pending)
                logger.info("Embedding cache 已保存：batch %d/%d；新增 %d 条", batch_no, total_batches, len(pending))
                pending.clear()
        del model

    vectors = [cache_map[text] for text in texts]
    return candidates.with_columns(
        pl.Series("embedding", vectors, dtype=pl.List(pl.Float64))
    ), stats


def cluster_candidates(embedded: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    matrix = np.asarray(embedded["embedding"].to_list(), dtype=np.float32)
    if matrix.shape != (embedded.height, EMBEDDING_DIMENSIONS) or not np.isfinite(matrix).all():
        raise ValueError(f"Embedding matrix 异常：{matrix.shape}")
    matrix = normalize(matrix, norm="l2").astype(np.float32)
    count = len(matrix)
    neighbors = min(UMAP_N_NEIGHBORS, max(2, count - 1))
    dimensions = min(UMAP_CLUSTER_COMPONENTS, max(2, count - 2))
    logger.info("UMAP 聚类空间：%d x %d -> %d；neighbors=%d", count, matrix.shape[1], dimensions, neighbors)
    space = umap.UMAP(
        n_components=dimensions,
        n_neighbors=neighbors,
        min_dist=UMAP_MIN_DIST_CLUSTER,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        n_jobs=1,
        low_memory=True,
    ).fit_transform(matrix)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(space).astype(int)
    visual = umap.UMAP(
        n_components=UMAP_VIS_COMPONENTS,
        n_neighbors=neighbors,
        min_dist=UMAP_MIN_DIST_VIS,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        n_jobs=1,
        low_memory=True,
    ).fit_transform(matrix)
    clustered = embedded.drop(["embedding", "embedding_text"]).with_columns([
        pl.Series("cluster_id", labels, dtype=pl.Int64),
        pl.Series("is_noise", labels == -1, dtype=pl.Boolean),
        pl.Series("cluster_probability", clusterer.probabilities_, dtype=pl.Float64),
        pl.Series("outlier_score", clusterer.outlier_scores_, dtype=pl.Float64),
        pl.Series("umap_x", visual[:, 0], dtype=pl.Float64),
        pl.Series("umap_y", visual[:, 1], dtype=pl.Float64),
    ])
    mask = labels >= 0
    unique = set(labels[mask].tolist())
    silhouette = None
    if len(unique) >= 2 and mask.sum() > len(unique):
        try:
            silhouette = float(silhouette_score(space[mask], labels[mask]))
        except Exception as exc:
            logger.warning("Silhouette 计算失败：%s", exc)
    noise = int((labels == -1).sum())
    metrics = {
        "cluster_count": len(set(labels.tolist()) - {-1}),
        "noise_count": noise,
        "noise_ratio": noise / len(labels),
        "silhouette_non_noise_umap_space": silhouette,
        "umap_cluster_components_actual": dimensions,
        "umap_n_neighbors_actual": neighbors,
    }
    logger.info("聚类完成：clusters=%d；noise=%d/%d (%.2f%%)", metrics["cluster_count"], noise, len(labels), metrics["noise_ratio"] * 100)
    return clustered, metrics


def build_cluster_summary(clustered: pl.DataFrame) -> pl.DataFrame:
    """Summarize only stable clusters; cluster_id=-1 remains in separate noise output."""
    stable = clustered.filter(pl.col("cluster_id") >= 0)
    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(stable["cluster_id"].unique().to_list()):
        group = stable.filter(pl.col("cluster_id") == cluster_id)
        ranked = group.sort(
            ["trend_score", "cluster_probability", "current_count"],
            descending=[True, True, True],
        )
        base = int(group["base_count"].sum())
        current = int(group["current_count"].sum())
        distribution = group.group_by("source_type").len().sort("source_type")
        top = ranked.head(TOP_TERMS_PER_CLUSTER)
        rows.append({
            "cluster_id": int(cluster_id),
            "is_noise": False,
            "cluster_size": group.height,
            "sum_base_count": base,
            "sum_current_count": current,
            "aggregate_ngram_docfreq_growth": ((current - base) / base if base else None),
            "mean_growth_rate": float(group["growth_rate"].mean()),
            "median_growth_rate": float(group["growth_rate"].median()),
            "max_growth_rate": float(group["growth_rate"].max()),
            "mean_cluster_probability": float(group["cluster_probability"].mean()),
            "median_outlier_score": float(group["outlier_score"].median()),
            "source_type_distribution": "; ".join(
                f"{row['source_type']}:{row['len']}"
                for row in distribution.iter_rows(named=True)
            ),
            "top_terms": "; ".join(top["ngram"].to_list()),
            "top_growth_rates": "; ".join(
                f"{value:.6f}" for value in top["growth_rate"].to_list()
            ),
        })
    if not rows:
        raise RuntimeError("没有形成任何稳定 Cluster；cluster_summary 无法生成。")
    return pl.DataFrame(rows).sort(
        ["aggregate_ngram_docfreq_growth", "cluster_size"],
        descending=[True, True],
        nulls_last=True,
    )


def validate_outputs(candidates: pl.DataFrame, clustered: pl.DataFrame, summary: pl.DataFrame) -> None:
    if candidates.height != clustered.height:
        raise RuntimeError("行数校验失败")
    if clustered["ngram"].n_unique() != clustered.height:
        raise RuntimeError("ngram 唯一性校验失败")
    if clustered.filter(pl.col("growth_rate") <= GROWTH_RATE_THRESHOLD).height:
        raise RuntimeError("growth threshold 校验失败")
    null_counts = clustered.select([
        pl.col("cluster_id").null_count(),
        pl.col("umap_x").null_count(),
        pl.col("umap_y").null_count(),
    ]).row(0)
    if null_counts != (0, 0, 0):
        raise RuntimeError("聚类字段存在 null")
    if clustered.filter(
        ~pl.col("umap_x").is_finite() | ~pl.col("umap_y").is_finite()
    ).height:
        raise RuntimeError("UMAP 坐标异常")
    if summary.filter(pl.col("cluster_id") < 0).height:
        raise RuntimeError("cluster_summary 不允许包含 cluster_id=-1")
    logger.info("一致性校验通过：行数、唯一性、阈值、聚类标签、二维坐标和 Noise 隔离。")


def write_outputs(
    candidates: pl.DataFrame,
    low_audit: pl.DataFrame,
    clustered: pl.DataFrame,
    summary: pl.DataFrame,
) -> None:
    candidates_output = candidates.drop("embedding_text")
    candidates_output.write_parquet(OUTPUT_DIR / "trend_candidates.parquet")
    write_csv_bom(candidates_output, OUTPUT_DIR / "trend_candidates.csv")
    low_audit.write_parquet(OUTPUT_DIR / "filtered_low_cohesion_audit.parquet")
    write_csv_bom(low_audit, OUTPUT_DIR / "filtered_low_cohesion_audit.csv")
    clustered.write_parquet(OUTPUT_DIR / "trend_clustered.parquet")
    write_csv_bom(clustered, OUTPUT_DIR / "trend_clustered.csv")
    summary.write_parquet(OUTPUT_DIR / "cluster_summary.parquet")
    write_csv_bom(summary, OUTPUT_DIR / "cluster_summary.csv")
    noise = clustered.filter(pl.col("cluster_id") == -1).sort(
        ["trend_score", "outlier_score"], descending=[True, True]
    )
    noise.write_parquet(OUTPUT_DIR / "cluster_noise_terms.parquet")
    write_csv_bom(noise, OUTPUT_DIR / "cluster_noise_terms.csv")


def run_cluster_pipeline() -> dict[str, Any]:
    started = time.time()
    logger.info("=" * 100)
    logger.info(
        "%s (%s) 本地 BGE-M3 热词聚类开始；无 LLM；无 API。",
        CONTEXT.category_code,
        CONTEXT.category_name,
    )
    logger.info("CONFIG=%s", json.dumps(config_dict(), ensure_ascii=False))
    if BGE_DEVICE == "cpu" and BGE_USE_FP16:
        raise ValueError("CPU 模式请设置 use_fp16=false")
    if CACHE_SAVE_EVERY_BATCHES < 1:
        raise ValueError("cache_save_every_batches 必须 >= 1")

    candidates, low_audit, counts = load_candidates()
    embedded, embedding_stats = build_local_embeddings(candidates)
    clustered, cluster_metrics = cluster_candidates(embedded)
    summary = build_cluster_summary(clustered)
    validate_outputs(candidates, clustered, summary)
    write_outputs(candidates, low_audit, clustered, summary)

    run_summary = {
        "status": "success",
        "category_code": CONTEXT.category_code,
        "category_name": CONTEXT.category_name,
        "run_id": CONTEXT.run_id,
        "source_filtered_counts": counts,
        "main_unique_candidates": candidates.height,
        "low_cohesion_audit_rows": low_audit.height,
        "embedding": embedding_stats,
        "clustering": cluster_metrics,
        "runtime_seconds": round(time.time() - started, 3),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "config": config_dict(),
        "metric_note": (
            "aggregate_ngram_docfreq_growth 是簇内 ngram 文档频次聚合变化，"
            "不能解释为去重商品数增长。"
        ),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("SUMMARY=%s", json.dumps(run_summary, ensure_ascii=False))
    logger.info("输出目录：%s", OUTPUT_DIR)
    logger.info("=" * 100)
    return run_summary


def main() -> None:
    CONTEXT.ensure_directories()
    create_manifest(CONTEXT)
    update_stage(CONTEXT, "cluster", "running")
    try:
        run_cluster_pipeline()
        update_stage(
            CONTEXT,
            "cluster",
            "completed",
            artifacts={
                "trend_candidates": OUTPUT_DIR / "trend_candidates.parquet",
                "trend_clustered": OUTPUT_DIR / "trend_clustered.parquet",
                "cluster_summary": OUTPUT_DIR / "cluster_summary.parquet",
                "cluster_noise_terms": OUTPUT_DIR / "cluster_noise_terms.parquet",
                "embedding_cache": EMBEDDING_CACHE_PATH,
                "cluster_run_summary": OUTPUT_DIR / "run_summary.json",
            },
        )
        set_active_run(CONTEXT)
    except Exception:
        update_stage(CONTEXT, "cluster", "failed")
        logger.exception(
            "Cluster stage failed category=%s run_id=%s",
            CONTEXT.category_code,
            CONTEXT.run_id,
        )
        raise


if __name__ == "__main__":
    main()
