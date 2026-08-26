from __future__ import annotations

import argparse
import calendar
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analysis_unit import (
    ANALYSIS_UNIT_CLUSTER,
    ANALYSIS_UNIT_NOISE_TERM,
)
from core.project_context import ProjectContext
from core.run_manifest import create_manifest, set_active_run, update_stage
from core.taxonomy_common import (
    EXCLUDED_DIMENSION_CODES,
    attribute_directory,
    json_load,
)

ALLOWED_MAPPING_TYPES = [
    "existing_attribute_existing_label",
    "existing_attribute_new_label",
    "new_attribute_new_label",
    "multi_attribute_cluster",
    "mixed_or_invalid_cluster",
    "uncertain",
]


def to_json_safe(value: Any) -> Any:
    """Recursively convert pandas/numpy values into strict JSON values."""
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [to_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _quarter_bounds(quarter: str) -> tuple[str, str]:
    """Convert YYYYQ1..YYYYQ4 into natural-quarter date boundaries."""
    text = str(quarter).strip().upper()
    if len(text) != 6 or text[4] != "Q" or not text[:4].isdigit():
        raise ValueError(
            f"无法将季度转换为日期边界：{quarter!r}。"
            "请在 TOML [period] 中配置 *_period_start/end。"
        )
    year = int(text[:4])
    quarter_number = int(text[5])
    if quarter_number not in {1, 2, 3, 4}:
        raise ValueError(f"非法季度：{quarter!r}")
    start_month = (quarter_number - 1) * 3 + 1
    end_month = start_month + 2
    last_day = calendar.monthrange(year, end_month)[1]
    return (
        date(year, start_month, 1).isoformat(),
        date(year, end_month, last_day).isoformat(),
    )


def build_analysis_period(context: ProjectContext) -> dict[str, str]:
    """Build explicit period boundaries for external temporal classification.

    Explicit TOML dates win. If absent, YYYYQn values are converted to natural
    quarter boundaries. This keeps the evidence self-contained and prevents
    external findings with valid source dates from becoming date_unknown.
    """
    period = context.period
    base_quarter = str(period["base_quarter"])
    current_quarter = str(period["current_quarter"])

    default_base_start, default_base_end = _quarter_bounds(base_quarter)
    default_current_start, default_current_end = _quarter_bounds(current_quarter)

    result = {
        "base_period": base_quarter,
        "current_period": current_quarter,
        "base_period_start": str(
            period.get("base_period_start", default_base_start)
        ),
        "base_period_end": str(
            period.get("base_period_end", default_base_end)
        ),
        "current_period_start": str(
            period.get("current_period_start", default_current_start)
        ),
        "current_period_end": str(
            period.get("current_period_end", default_current_end)
        ),
    }

    for key in (
        "base_period_start",
        "base_period_end",
        "current_period_start",
        "current_period_end",
    ):
        try:
            date.fromisoformat(result[key])
        except ValueError as exc:
            raise ValueError(
                f"[period].{key} 必须为 YYYY-MM-DD，当前值={result[key]!r}"
            ) from exc

    if result["base_period_start"] > result["base_period_end"]:
        raise ValueError("base_period_start 不能晚于 base_period_end")
    if result["current_period_start"] > result["current_period_end"]:
        raise ValueError("current_period_start 不能晚于 current_period_end")
    return result


def finite(value: Any) -> Any:
    return (
        value
        if isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
        else None
    )


def term_obj(row: pd.Series) -> dict[str, Any]:
    fields = [
        "ngram",
        "growth_rate",
        "base_count",
        "current_count",
        "absolute_growth",
        "trend_score",
        "source_type",
        "cohesion",
        "cluster_probability",
        "outlier_score",
        "parents",
        "children",
    ]
    output: dict[str, Any] = {}
    for field in fields:
        if field not in row.index:
            continue
        value = row[field]
        if field in {"parents", "children"}:
            value = json_load(value, [])
        output[field] = to_json_safe(value)
    return output


def candidate_diagnostics(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    top1 = candidates[0] if candidates else {}
    top2 = candidates[1] if len(candidates) > 1 else {}
    exact_labels = [
        label["label"]
        for candidate in candidates
        for label in candidate.get("candidate_labels", [])
        if label.get("normalized_exact_match")
    ]
    exact_codes = sorted(
        {
            candidate["attribute_code"]
            for candidate in candidates
            if any(
                label.get("normalized_exact_match")
                for label in candidate.get("candidate_labels", [])
            )
        }
    )
    diagnostics = {
        "top1_similarity": top1.get("similarity"),
        "top2_similarity": top2.get("similarity"),
        "top1_top2_margin": (
            top1.get("similarity", 0) - top2.get("similarity", 0)
            if top2
            else None
        ),
    }
    exact_matches = {
        "any_candidate_has_exact_match": bool(exact_labels),
        "exact_match_attribute_codes": exact_codes,
        "exact_match_labels": exact_labels,
    }
    return to_json_safe(diagnostics), to_json_safe(exact_matches)


def cluster_metrics(row: pd.Series) -> dict[str, Any]:
    fields = [
        "cluster_size",
        "aggregate_ngram_docfreq_growth",
        "mean_growth_rate",
        "median_growth_rate",
        "max_growth_rate",
        "sum_base_count",
        "sum_current_count",
        "mean_cluster_probability",
        "median_outlier_score",
        "source_type_distribution",
        "representative_low_cohesion_count",
        "representative_low_cohesion_ratio",
    ]
    return {
        field: to_json_safe(row[field])
        for field in fields
        if field in row.index
    }


def run(
    review_path: Path,
    summary_path: Path,
    terms_path: Path,
    source_path: Path,
    output_path: Path,
    top_n: int,
    context: ProjectContext,
) -> None:
    review = pd.read_parquet(review_path)
    summary = pd.read_parquet(summary_path)
    terms = pd.read_parquet(terms_path)
    source = pd.read_parquet(source_path)
    directory = attribute_directory(source)
    analysis_period = build_analysis_period(context)

    forbidden = EXCLUDED_DIMENSION_CODES & {
        item["attribute_code"] for item in directory
    }
    if forbidden:
        raise AssertionError(
            f"非属性维度进入 all_existing_attributes: {sorted(forbidden)}"
        )

    summary_map = {
        int(row.cluster_id): row
        for _, row in summary[
            pd.to_numeric(summary.cluster_id, errors="coerce").ge(0)
        ].iterrows()
    }
    records: list[dict[str, Any]] = []

    for _, review_row in review.sort_values(
        ["analysis_unit", "query_key"], kind="stable"
    ).iterrows():
        analysis_unit = str(review_row.analysis_unit)
        query_key = str(review_row.query_key)
        cluster_id = int(review_row.cluster_id)
        source_cluster_id = int(review_row.source_cluster_id)
        term = None if pd.isna(review_row.term) else str(review_row.term)
        candidates = json_load(review_row.taxonomy_candidates_json, [])
        invalid = EXCLUDED_DIMENSION_CODES & {
            str(item.get("attribute_code")) for item in candidates
        }
        if invalid:
            raise AssertionError(
                f"{query_key} 非属性候选: {sorted(invalid)}"
            )

        if analysis_unit == ANALYSIS_UNIT_CLUSTER:
            if cluster_id < 0 or cluster_id not in summary_map:
                raise ValueError(f"Cluster Evidence 缺少 Summary：{query_key}")
            unit_terms = terms[
                pd.to_numeric(terms.cluster_id, errors="coerce").eq(cluster_id)
            ].copy()
            sort_column = (
                "trend_score" if "trend_score" in unit_terms else "growth_rate"
            )
            unit_terms = unit_terms.sort_values(
                sort_column, ascending=False
            ).head(top_n)
            metrics = cluster_metrics(summary_map[cluster_id])
            term_evidence = None
            metric_note = (
                "base_count/current_count 为商品描述文档频次；"
                "Cluster 聚合可重复覆盖同一描述。"
            )
        elif analysis_unit == ANALYSIS_UNIT_NOISE_TERM:
            if cluster_id != -1 or source_cluster_id != -1 or not term:
                raise ValueError(f"Noise Term 身份字段无效：{query_key}")
            matches = terms[
                terms["ngram"].astype(str).eq(term)
                & pd.to_numeric(terms["cluster_id"], errors="coerce").eq(-1)
            ].copy()
            if matches.empty:
                raise ValueError(f"trend_clustered 中找不到 Noise Term：{term}")
            unit_terms = matches.head(1)
            term_evidence = term_obj(unit_terms.iloc[0])
            metrics = {
                "cluster_size": 1,
                "aggregate_ngram_docfreq_growth": term_evidence.get("growth_rate"),
                "mean_growth_rate": term_evidence.get("growth_rate"),
                "median_growth_rate": term_evidence.get("growth_rate"),
                "max_growth_rate": term_evidence.get("growth_rate"),
                "sum_base_count": term_evidence.get("base_count"),
                "sum_current_count": term_evidence.get("current_count"),
                "mean_cluster_probability": term_evidence.get(
                    "cluster_probability"
                ),
                "median_outlier_score": term_evidence.get("outlier_score"),
                "source_type_distribution": str(
                    term_evidence.get("source_type") or ""
                ),
            }
            metric_note = (
                "base_count/current_count 为当前未稳定归簇词的商品描述文档频次；"
                "该对象是单词级分析单元，不代表 Cluster -1 整体。"
            )
        else:
            raise ValueError(f"未知 analysis_unit：{analysis_unit}")

        diagnostics, exact_matches = candidate_diagnostics(candidates)
        record = {
            "analysis_unit": analysis_unit,
            "query_key": query_key,
            "cluster_id": cluster_id,
            "source_cluster_id": source_cluster_id,
            "term": term,
            "analysis_period": analysis_period,
            "cluster_metrics": metrics,
            "representative_terms": [
                term_obj(row) for _, row in unit_terms.iterrows()
            ],
            "attribute_diagnostics": diagnostics,
            "all_existing_attributes": directory,
            "taxonomy_candidates": candidates,
            "exact_matches": exact_matches,
            "allowed_mapping_types": ALLOWED_MAPPING_TYPES,
            "metric_note": metric_note,
        }
        if term_evidence is not None:
            record["term_evidence"] = term_evidence
        records.append(to_json_safe(record))

    from core.cluster_loader import validate_evidence

    errors = []
    for line_number, record in enumerate(records, 1):
        errors.extend(
            issue
            for issue in validate_evidence(record, line_number=line_number)
            if issue.severity == "error"
        )
    if errors:
        raise ValueError(
            "Evidence校验失败: "
            + " | ".join(
                f"{issue.code}:{issue.message}" for issue in errors[:20]
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    pretty_path = output_path.with_name("cluster_llm_evidence_pretty.json")
    pretty_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    counts = pd.Series(
        [record["analysis_unit"] for record in records]
    ).value_counts().to_dict()
    print(f"[OK] evidence={len(records)} {counts} -> {output_path}")
    print(f"[OK] analysis_period={analysis_period}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category")
    parser.add_argument("--run-id")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--cluster-summary", type=Path)
    parser.add_argument("--trend-clustered", type=Path)
    parser.add_argument("--taxonomy-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-representative-terms", type=int)
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

    review = args.review or context.taxonomy_candidate_review_file
    summary = args.cluster_summary or context.cluster_summary_file
    terms = args.trend_clustered or context.trend_clustered_file
    source = args.taxonomy_source or context.taxonomy_source_normalized_file
    output = args.output or context.cluster_evidence_file
    top_n = args.top_representative_terms or int(
        context.taxonomy.get("top_representative_terms", 15)
    )

    update_stage(context, "taxonomy", "building_evidence")
    try:
        run(review, summary, terms, source, output, top_n, context)
        update_stage(
            context,
            "taxonomy",
            "completed",
            artifacts={
                "cluster_llm_evidence": output,
                "cluster_llm_evidence_pretty": output.with_name(
                    "cluster_llm_evidence_pretty.json"
                ),
            },
        )
        set_active_run(context)
    except Exception:
        update_stage(context, "taxonomy", "failed")
        raise


if __name__ == "__main__":
    main()
