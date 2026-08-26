"""Build monthly Dashboard caches for stable Cluster terms and Noise Terms.

Rules
-----
- Term-level cache includes every eligible term from trend_clustered.parquet,
  including cluster_id=-1 Noise Terms.
- Cluster-level aggregate caches include only cluster_id >= 0.
- Eligibility uses the same growth_rate_threshold configured for the category.
- Text normalization and n-gram generation are delegated to main.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as trend_main
from core.analysis_unit import eligible_trend_terms
from core.project_context import ProjectContext
from core.run_manifest import create_manifest, update_stage


def parse_periodcode(value: Any) -> tuple[str, str, str]:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) != 8 or not text.isdigit() or text[4:6] != "14":
        raise ValueError(f"非法 PERIODCODE: {value!r}")
    month = int(text[6:8])
    if not 1 <= month <= 12:
        raise ValueError(f"非法 PERIODCODE 月份: {value!r}")
    label = f"{text[:4]}-{month:02d}"
    quarter = f"{text[:4]}Q{((month - 1) // 3) + 1}"
    return text, label, quarter


def normalize_with_main(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return trend_main.normalize_desc(str(value))


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_trend_terms(context: ProjectContext) -> pd.DataFrame:
    path = context.trend_clustered_file
    if not path.exists():
        raise FileNotFoundError(path)
    trend = pd.read_parquet(path)
    required = {
        "ngram", "cluster_id", "growth_rate", "base_count", "current_count"
    }
    missing = sorted(required.difference(trend.columns))
    if missing:
        raise KeyError(f"trend_clustered 缺少字段: {missing}")

    threshold = float(context.trend["growth_rate_threshold"])
    trend = eligible_trend_terms(
        trend,
        growth_rate_threshold=threshold,
    )
    trend["cluster_id"] = pd.to_numeric(
        trend["cluster_id"], errors="coerce"
    )
    if "is_noise" not in trend.columns:
        trend["is_noise"] = trend["cluster_id"].eq(-1)
    else:
        trend["is_noise"] = (
            trend["is_noise"].fillna(False).astype(bool)
            | trend["cluster_id"].eq(-1)
        )
    trend = trend.dropna(subset=["cluster_id"]).copy()
    trend["cluster_id"] = trend["cluster_id"].astype(int)
    trend["ngram"] = trend["ngram"].astype(str)
    trend = trend.drop_duplicates(subset=["ngram"], keep="first")
    return trend


def source_columns(source_path: Path) -> tuple[str, str]:
    schema_names = set(pq.ParquetFile(source_path).schema.names)
    desc_col = trend_main.DESC_COL
    period_col = trend_main.PERIOD_COL
    missing = [c for c in (desc_col, period_col) if c not in schema_names]
    if missing:
        raise KeyError(f"源文件缺少字段: {missing}")
    return desc_col, period_col


def iter_source_batches(
    source_path: Path,
    *,
    desc_col: str,
    period_col: str,
    batch_rows: int,
) -> Iterable[pd.DataFrame]:
    parquet = pq.ParquetFile(source_path)
    for batch in parquet.iter_batches(
        batch_size=batch_rows,
        columns=[desc_col, period_col],
    ):
        yield batch.to_pandas()


def process_source(
    source_path: Path,
    trend: pd.DataFrame,
    *,
    batch_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    desc_col, period_col = source_columns(source_path)
    candidate_terms = set(trend["ngram"].astype(str))
    term_cluster = dict(zip(trend["ngram"].astype(str), trend["cluster_id"].astype(int)))
    normal_terms = {
        term for term, cluster_id in term_cluster.items() if cluster_id >= 0
    }

    term_counts: Counter[tuple[str, str, int]] = Counter()
    cluster_document_counts: Counter[tuple[str, int]] = Counter()
    period_labels: dict[str, str] = {}
    invalid_periods: Counter[str] = Counter()
    processed_rows = 0
    empty_rows = 0
    matched_documents = 0

    for frame in iter_source_batches(
        source_path,
        desc_col=desc_col,
        period_col=period_col,
        batch_rows=batch_rows,
    ):
        for description, period_value in zip(
            frame[desc_col].tolist(), frame[period_col].tolist()
        ):
            processed_rows += 1
            normalized = normalize_with_main(description)
            if not normalized:
                empty_rows += 1
                continue
            try:
                period_code, period_label, _ = parse_periodcode(period_value)
            except ValueError:
                invalid_periods[str(period_value)] += 1
                continue
            period_labels[period_code] = period_label

            generated = set(trend_main.make_unique_char_ngrams(normalized))
            matched = generated.intersection(candidate_terms)
            if not matched:
                continue
            matched_documents += 1

            # Term-level document frequency includes normal + Noise Terms.
            for term in matched:
                term_counts[(period_code, term, term_cluster[term])] += 1

            # Cluster-level document coverage excludes Noise.
            matched_cluster_ids = {
                term_cluster[term]
                for term in matched.intersection(normal_terms)
                if term_cluster[term] >= 0
            }
            for cluster_id in matched_cluster_ids:
                cluster_document_counts[(period_code, cluster_id)] += 1

    term_rows = [
        {
            "period_code": period_code,
            "period_label": period_labels.get(period_code, period_code),
            "cluster_id": cluster_id,
            "ngram": term,
            "docfreq": count,
        }
        for (period_code, term, cluster_id), count in term_counts.items()
    ]
    term_df = pd.DataFrame(term_rows)
    if term_df.empty:
        term_df = pd.DataFrame(
            columns=["period_code", "period_label", "cluster_id", "ngram", "docfreq"]
        )
    else:
        term_df = term_df.sort_values(
            ["period_code", "cluster_id", "ngram"], kind="stable"
        ).reset_index(drop=True)

    coverage_rows = [
        {
            "period_code": period_code,
            "period_label": period_labels.get(period_code, period_code),
            "cluster_id": cluster_id,
            "unique_product_count": count,
        }
        for (period_code, cluster_id), count in cluster_document_counts.items()
    ]
    coverage_df = pd.DataFrame(coverage_rows)
    if coverage_df.empty:
        coverage_df = pd.DataFrame(
            columns=[
                "period_code", "period_label", "cluster_id",
                "unique_product_count",
            ]
        )
    else:
        coverage_df = coverage_df.sort_values(
            ["period_code", "cluster_id"], kind="stable"
        ).reset_index(drop=True)

    stats = {
        "processed_rows": processed_rows,
        "empty_description_rows": empty_rows,
        "matched_documents": matched_documents,
        "invalid_periods": dict(invalid_periods),
    }
    return term_df, coverage_df, stats


def build_cluster_sum(term_df: pd.DataFrame) -> pd.DataFrame:
    normal = term_df[
        pd.to_numeric(term_df["cluster_id"], errors="coerce").ge(0)
    ].copy()
    if normal.empty:
        return pd.DataFrame(
            columns=["period_code", "period_label", "cluster_id", "ngram_docfreq_sum"]
        )
    return (
        normal.groupby(
            ["period_code", "period_label", "cluster_id"],
            as_index=False,
        )["docfreq"]
        .sum()
        .rename(columns={"docfreq": "ngram_docfreq_sum"})
        .sort_values(["period_code", "cluster_id"], kind="stable")
        .reset_index(drop=True)
    )


def build_validation(term_df: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    monthly = term_df.copy()
    if monthly.empty:
        result = trend[
            ["cluster_id", "ngram", "base_count", "current_count", "is_noise"]
        ].copy()
        result["base_monthly_count"] = 0
        result["current_monthly_count"] = 0
    else:
        monthly["quarter"] = monthly["period_code"].map(
            lambda value: parse_periodcode(value)[2]
        )
        quarter = (
            monthly[
                monthly["quarter"].isin(
                    [trend_main.BASE_QUARTER, trend_main.CURRENT_QUARTER]
                )
            ]
            .groupby(["cluster_id", "ngram", "quarter"], as_index=False)["docfreq"]
            .sum()
            .pivot_table(
                index=["cluster_id", "ngram"],
                columns="quarter",
                values="docfreq",
                fill_value=0,
            )
            .reset_index()
        )
        result = trend[
            ["cluster_id", "ngram", "base_count", "current_count", "is_noise"]
        ].merge(quarter, on=["cluster_id", "ngram"], how="left")
        result["base_monthly_count"] = pd.to_numeric(
            result.get(trend_main.BASE_QUARTER, 0), errors="coerce"
        ).fillna(0)
        result["current_monthly_count"] = pd.to_numeric(
            result.get(trend_main.CURRENT_QUARTER, 0), errors="coerce"
        ).fillna(0)

    result["base_count"] = pd.to_numeric(result["base_count"], errors="coerce").fillna(0)
    result["current_count"] = pd.to_numeric(result["current_count"], errors="coerce").fillna(0)
    result["base_difference"] = result["base_monthly_count"] - result["base_count"]
    result["current_difference"] = (
        result["current_monthly_count"] - result["current_count"]
    )
    result["base_exact_match"] = result["base_difference"].eq(0)
    result["current_exact_match"] = result["current_difference"].eq(0)
    result["base_relative_error"] = (
        result["base_difference"].abs()
        / result["base_count"].replace(0, pd.NA)
    ).fillna(result["base_difference"].abs().gt(0).astype(float))
    result["current_relative_error"] = (
        result["current_difference"].abs()
        / result["current_count"].replace(0, pd.NA)
    ).fillna(result["current_difference"].abs().gt(0).astype(float))
    result["base_within_1pct"] = result["base_relative_error"].le(0.01)
    result["current_within_1pct"] = result["current_relative_error"].le(0.01)
    result["analysis_unit"] = result["is_noise"].map(
        lambda value: "noise_term" if bool(value) else "cluster_term"
    )
    return result.sort_values(
        ["analysis_unit", "cluster_id", "ngram"], kind="stable"
    ).reset_index(drop=True)


def rates(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "base_exact_match_rate": 0.0,
            "current_exact_match_rate": 0.0,
            "base_within_1pct_rate": 0.0,
            "current_within_1pct_rate": 0.0,
        }
    return {
        "base_exact_match_rate": float(frame["base_exact_match"].mean()),
        "current_exact_match_rate": float(frame["current_exact_match"].mean()),
        "base_within_1pct_rate": float(frame["base_within_1pct"].mean()),
        "current_within_1pct_rate": float(frame["current_within_1pct"].mean()),
    }


def run(context: ProjectContext, *, batch_rows: int) -> None:
    started = time.perf_counter()
    source_path = context.sampled_products_file
    output_dir = context.dashboard_cache_dir
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    trend = load_trend_terms(context)
    term_df, coverage_df, source_stats = process_source(
        source_path,
        trend,
        batch_rows=batch_rows,
    )
    cluster_sum_df = build_cluster_sum(term_df)
    validation_df = build_validation(term_df, trend)

    output_dir.mkdir(parents=True, exist_ok=True)
    term_file = output_dir / "term_period_docfreq.parquet"
    cluster_sum_file = output_dir / "cluster_period_ngram_sum.parquet"
    coverage_file = output_dir / "cluster_period_product_coverage.parquet"
    validation_file = output_dir / "period_validation_detail.parquet"
    summary_file = output_dir / "period_cache_summary.json"

    term_df.to_parquet(term_file, index=False)
    cluster_sum_df.to_parquet(cluster_sum_file, index=False)
    coverage_df.to_parquet(coverage_file, index=False)
    validation_df.to_parquet(validation_file, index=False)

    cluster_validation = validation_df[
        validation_df["analysis_unit"].eq("cluster_term")
    ]
    noise_validation = validation_df[
        validation_df["analysis_unit"].eq("noise_term")
    ]
    overall_rates = rates(validation_df)
    cluster_rates = rates(cluster_validation)
    noise_rates = rates(noise_validation)

    summary = {
        "category_code": context.category_code,
        "run_id": context.run_id,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(source_path),
        "source_rows": source_stats["processed_rows"],
        "matched_documents": source_stats["matched_documents"],
        "empty_description_rows": source_stats["empty_description_rows"],
        "invalid_periods": source_stats["invalid_periods"],
        "growth_rate_threshold": float(context.trend["growth_rate_threshold"]),
        "term_count": int(len(trend)),
        "cluster_term_count": int((trend["cluster_id"] >= 0).sum()),
        "noise_term_count": int((trend["cluster_id"] == -1).sum()),
        "monthly_term_rows": int(len(term_df)),
        "normal_cluster_count": int(
            trend.loc[trend["cluster_id"] >= 0, "cluster_id"].nunique()
        ),
        **overall_rates,
        "cluster_term_validation": cluster_rates,
        "noise_term_validation": noise_rates,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "metric_note": (
            "term_period_docfreq 包含正常词和 Noise Term；"
            "Cluster 聚合缓存只包含 cluster_id>=0。"
        ),
    }
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("[OK] Dashboard 月度缓存已重建")
    print(f"  term rows: {len(term_df):,}")
    print(f"  normal terms: {(trend['cluster_id'] >= 0).sum():,}")
    print(f"  noise terms: {(trend['cluster_id'] == -1).sum():,}")
    print(f"  Base exact match: {overall_rates['base_exact_match_rate']:.2%}")
    print(f"  Current exact match: {overall_rates['current_exact_match_rate']:.2%}")
    print(f"  Summary: {summary_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category")
    parser.add_argument("--run-id")
    parser.add_argument("--batch-rows", type=int)
    args = parser.parse_args()

    context = (
        ProjectContext.from_category(args.category, project_root=ROOT)
        if args.category
        else ProjectContext.active(project_root=ROOT)
    )
    if args.run_id:
        context = context.with_run_id(args.run_id)
    context.ensure_directories()
    create_manifest(context)
    batch_rows = args.batch_rows or int(
        context.config.get("dashboard_cache", {}).get("chunk_rows", 200_000)
    )

    update_stage(context, "dashboard_cache", "running")
    try:
        run(context, batch_rows=batch_rows)
        output_dir = context.dashboard_cache_dir
        update_stage(
            context,
            "dashboard_cache",
            "completed",
            artifacts={
                "term_period_docfreq": output_dir / "term_period_docfreq.parquet",
                "cluster_period_ngram_sum": output_dir / "cluster_period_ngram_sum.parquet",
                "cluster_period_product_coverage": output_dir / "cluster_period_product_coverage.parquet",
                "period_validation_detail": output_dir / "period_validation_detail.parquet",
                "period_cache_summary": output_dir / "period_cache_summary.json",
            },
        )
    except Exception:
        update_stage(context, "dashboard_cache", "failed")
        raise


if __name__ == "__main__":
    main()
