"""Context-isolated, read-only loaders for the trend Dashboard.

All source paths are derived from the selected ProjectContext. No legacy global
path constants are used. Cached reads are keyed by absolute path and mtime.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.analysis_unit import (
    build_cluster_query_key,
    build_noise_term_query_key,
    eligible_trend_terms,
)
from core.project_context import ProjectContext


def _abs(path: Path) -> Path:
    return path.expanduser().resolve()


def _mtime(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data(show_spinner=False)
def read_json(path_text: str, modified_ns: int) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


@st.cache_data(show_spinner=False)
def read_jsonl(path_text: str, modified_ns: int) -> list[dict[str, Any]]:
    path = Path(path_text)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                rows.append({
                    "_invalid_jsonl_line": line_number,
                    "_source_path": str(path),
                })
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


@st.cache_data(show_spinner=False)
def read_parquet(path_text: str, modified_ns: int) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _json(path: Path) -> dict[str, Any]:
    path = _abs(path)
    return read_json(str(path), _mtime(path))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    path = _abs(path)
    return read_jsonl(str(path), _mtime(path))


def _parquet(path: Path) -> pd.DataFrame:
    path = _abs(path)
    return read_parquet(str(path), _mtime(path))


def _assert_context_path(path: Path, context: ProjectContext, name: str) -> None:
    resolved = _abs(path)
    expected_root = _abs(context.run_dir)
    try:
        resolved.relative_to(expected_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} 未位于当前 Workspace。\n"
            f"当前品类：{context.category_code}\n"
            f"当前 Run：{context.run_id}\n"
            f"期望根目录：{expected_root}\n"
            f"实际路径：{resolved}"
        ) from exc


def dashboard_paths(context: ProjectContext) -> dict[str, Path]:
    paths = {
        "cluster_summary": context.cluster_dir / "cluster_summary.parquet",
        "trend_clustered": context.cluster_dir / "trend_clustered.parquet",
        "trend_candidates": context.cluster_dir / "trend_candidates.parquet",
        "noise_terms": context.cluster_dir / "cluster_noise_terms.parquet",
        "low_cohesion_audit": context.cluster_dir / "filtered_low_cohesion_audit.parquet",
        "run_summary": context.cluster_dir / "run_summary.json",
        "evidence": context.taxonomy_dir / "cluster_llm_evidence.jsonl",
        "internal_cache": context.insights_dir / "cluster_internal_cache.jsonl",
        "internal_errors": context.insights_dir / "cluster_internal_errors.jsonl",
        "external_cache": context.research_dir / "cluster_external_json_v2_cache.jsonl",
        "external_errors": context.research_dir / "cluster_external_json_v2_errors.jsonl",
        "term_period_docfreq": context.dashboard_cache_dir / "term_period_docfreq.parquet",
        "cluster_period_ngram_sum": context.dashboard_cache_dir / "cluster_period_ngram_sum.parquet",
        "cluster_period_product_coverage": context.dashboard_cache_dir / "cluster_period_product_coverage.parquet",
        "period_cache_summary": context.dashboard_cache_dir / "period_cache_summary.json",
        "period_validation_detail": context.dashboard_cache_dir / "period_validation_detail.parquet",
    }
    for name, path in paths.items():
        _assert_context_path(path, context, name)
    return paths


def _record_query_key(record: dict[str, Any]) -> str | None:
    key = record.get("query_key")
    if key:
        return str(key)
    cluster_id = record.get("cluster_id")
    if cluster_id is None:
        return None
    try:
        cluster_id = int(cluster_id)
    except (TypeError, ValueError):
        return None
    return build_cluster_query_key(cluster_id) if cluster_id >= 0 else None


def _source_query_key(record: dict[str, Any]) -> str | None:
    direct = record.get("source_query_key")
    if direct:
        return str(direct)
    parsed = record.get("parsed_result")
    if isinstance(parsed, dict) and parsed.get("source_query_key"):
        return str(parsed["source_query_key"])
    topic = record.get("topic")
    if isinstance(topic, dict) and topic.get("source_query_key"):
        return str(topic["source_query_key"])
    return None


def _latest_records(
    records: list[dict[str, Any]],
    *,
    key_getter,
    require_valid: bool = True,
    require_active: bool = True,
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if require_valid and record.get("schema_valid") is not True:
            continue
        if require_active and str(record.get("record_status", "ACTIVE")).upper() != "ACTIVE":
            continue
        key = key_getter(record)
        if not key:
            continue
        timestamp = str(
            record.get("updated_at_utc")
            or record.get("created_at_utc")
            or ""
        )
        previous = latest.get(key)
        previous_timestamp = str(
            (previous or {}).get("updated_at_utc")
            or (previous or {}).get("created_at_utc")
            or ""
        )
        if previous is None or timestamp >= previous_timestamp:
            latest[key] = record
    return latest


def _latest_errors(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _latest_records(
        records,
        key_getter=lambda row: str(row.get("query_key")) if row.get("query_key") else None,
        require_valid=False,
        require_active=False,
    )


def _status_text(
    query_key: str,
    evidence_by_key: dict[str, dict[str, Any]],
    internal_by_key: dict[str, dict[str, Any]],
    external_by_source_key: dict[str, dict[str, Any]],
) -> tuple[str, str, str]:
    taxonomy = "已生成 Evidence" if query_key in evidence_by_key else "未进入 Evidence"
    ai = "已完成" if query_key in internal_by_key else "未执行"
    research = "已完成" if query_key in external_by_source_key else "未执行"
    return taxonomy, ai, research


def _prepare_terms(
    raw: pd.DataFrame,
    *,
    context: ProjectContext,
    evidence_by_key: dict[str, dict[str, Any]],
    internal_by_key: dict[str, dict[str, Any]],
    external_by_source_key: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        empty = raw.copy()
        return empty, empty, empty
    required = {"ngram", "cluster_id", "growth_rate"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise KeyError(f"trend_clustered 缺少字段：{missing}")

    threshold = float(context.trend["growth_rate_threshold"])
    terms = eligible_trend_terms(raw, growth_rate_threshold=threshold)
    terms["cluster_id"] = pd.to_numeric(terms["cluster_id"], errors="coerce")
    terms = terms.dropna(subset=["cluster_id"]).copy()
    terms["cluster_id"] = terms["cluster_id"].astype(int)
    if "is_noise" not in terms.columns:
        terms["is_noise"] = terms["cluster_id"].eq(-1)
    else:
        terms["is_noise"] = (
            terms["is_noise"].fillna(False).astype(bool)
            | terms["cluster_id"].eq(-1)
        )

    query_keys: list[str] = []
    for row in terms.itertuples(index=False):
        cluster_id = int(row.cluster_id)
        if cluster_id >= 0:
            key = build_cluster_query_key(cluster_id)
        else:
            key = build_noise_term_query_key(
                category_code=context.category_code,
                run_id=context.run_id,
                ngram=str(row.ngram),
            )
        query_keys.append(key)
    terms["query_key"] = query_keys

    statuses = [
        _status_text(
            key,
            evidence_by_key,
            internal_by_key,
            external_by_source_key,
        )
        for key in terms["query_key"]
    ]
    terms["taxonomy_status"] = [item[0] for item in statuses]
    terms["ai_insights_status"] = [item[1] for item in statuses]
    terms["research_status"] = [item[2] for item in statuses]

    normal = terms[(terms["cluster_id"] >= 0) & (~terms["is_noise"])].copy()
    noise = terms[(terms["cluster_id"] == -1) | terms["is_noise"]].copy()
    if len(normal) + len(noise) != len(terms):
        raise RuntimeError("正常词与 Noise Term 分区不完整。")
    return terms, normal, noise


def load_dashboard_sources(
    data_scope: str,
    context: ProjectContext,
) -> dict[str, Any]:
    if data_scope not in {"raw", "internal", "full"}:
        raise ValueError(f"Unknown data scope: {data_scope}")

    paths = dashboard_paths(context)
    required = ("cluster_summary", "trend_clustered", "run_summary")
    missing = [name for name in required if not paths[name].exists()]
    if missing:
        detail = "\n".join(f"- {name}: {paths[name]}" for name in missing)
        raise FileNotFoundError(
            f"当前品类 {context.category_code} / Run {context.run_id} 缺少看板数据：\n{detail}"
        )

    evidence_rows = _jsonl(paths["evidence"])
    evidence_by_key = {
        str(row["query_key"]): row
        for row in evidence_rows
        if row.get("query_key")
    }

    internal_rows: list[dict[str, Any]] = []
    internal_error_rows: list[dict[str, Any]] = []
    if data_scope in {"internal", "full"}:
        internal_rows = _jsonl(paths["internal_cache"])
        internal_error_rows = _jsonl(paths["internal_errors"])
    internal_by_key = _latest_records(
        internal_rows,
        key_getter=_record_query_key,
    )

    external_rows: list[dict[str, Any]] = []
    external_error_rows: list[dict[str, Any]] = []
    if data_scope == "full":
        external_rows = _jsonl(paths["external_cache"])
        external_error_rows = _jsonl(paths["external_errors"])
    external_by_source_key = _latest_records(
        external_rows,
        key_getter=_source_query_key,
    )

    raw_cluster_summary = _parquet(paths["cluster_summary"])
    if not raw_cluster_summary.empty:
        raw_cluster_summary["cluster_id"] = pd.to_numeric(
            raw_cluster_summary["cluster_id"], errors="coerce"
        )
        cluster_summary = raw_cluster_summary[
            raw_cluster_summary["cluster_id"].ge(0)
        ].copy()
    else:
        cluster_summary = raw_cluster_summary

    raw_terms = _parquet(paths["trend_clustered"])
    all_terms, normal_terms, noise_terms = _prepare_terms(
        raw_terms,
        context=context,
        evidence_by_key=evidence_by_key,
        internal_by_key=internal_by_key,
        external_by_source_key=external_by_source_key,
    )

    cache_names = (
        "term_period_docfreq",
        "cluster_period_ngram_sum",
        "cluster_period_product_coverage",
        "period_cache_summary",
    )

    return {
        "context_key": f"{context.category_code}:{context.run_id}",
        "category_code": context.category_code,
        "run_id": context.run_id,
        "paths": {name: _abs(path) for name, path in paths.items()},
        "source_path_debug": {
            name: str(_abs(path)) for name, path in paths.items()
        },
        "growth_rate_threshold": float(context.trend["growth_rate_threshold"]),
        "run_summary": _json(paths["run_summary"]),
        "cluster_summary": cluster_summary,
        "trend_clustered": all_terms,
        "normal_terms": normal_terms,
        "noise_terms": noise_terms,
        "trend_candidates": _parquet(paths["trend_candidates"]),
        "low_cohesion_audit": _parquet(paths["low_cohesion_audit"]),
        "evidence_records": evidence_rows,
        "evidence_by_query_key": evidence_by_key,
        "internal_records": internal_by_key,
        "external_records": external_by_source_key,
        "internal_errors": internal_error_rows,
        "external_errors": external_error_rows,
        "internal_errors_by_query_key": _latest_errors(internal_error_rows),
        "external_errors_by_topic": _latest_errors(external_error_rows),
        "period_cache_available": all(paths[name].exists() for name in cache_names),
        "term_period_docfreq": _parquet(paths["term_period_docfreq"]),
        "cluster_period_ngram_sum": _parquet(paths["cluster_period_ngram_sum"]),
        "cluster_period_product_coverage": _parquet(paths["cluster_period_product_coverage"]),
        "period_cache_summary": _json(paths["period_cache_summary"]),
        "period_validation_detail": _parquet(paths["period_validation_detail"]),
    }
