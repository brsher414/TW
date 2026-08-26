"""Shared analysis-unit identities, trend eligibility, and stable query keys.

This module is intentionally infrastructure-only. Importing it does not change
current Retrieval, Evidence, Dashboard, cache, or LLM behavior. Later pipeline
stages should call these functions rather than reimplementing filtering or keys.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import Any

import pandas as pd

ANALYSIS_UNIT_CLUSTER = "cluster"
ANALYSIS_UNIT_NOISE_TERM = "noise_term"
VALID_ANALYSIS_UNITS = frozenset(
    {ANALYSIS_UNIT_CLUSTER, ANALYSIS_UNIT_NOISE_TERM}
)


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8-safe JSON used for stable identifiers."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def normalize_ngram(value: Any) -> str:
    """Normalize a term for identity comparison without changing its meaning.

    Rules are deliberately conservative:
    - Unicode NFKC normalizes full-width/half-width variants.
    - Leading/trailing whitespace is removed.
    - Internal whitespace runs are collapsed to one ASCII space.
    - Letter case is preserved because the original term remains the display value.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def build_cluster_query_key(cluster_id: int) -> str:
    """Build the backward-compatible identity for a stable cluster."""
    if not isinstance(cluster_id, int) or isinstance(cluster_id, bool):
        raise TypeError("cluster_id 必须为 int。")
    if cluster_id < 0:
        raise ValueError("稳定 Cluster 的 cluster_id 必须大于等于 0。")
    return f"cluster:{cluster_id}"


def build_noise_term_query_key(
    *,
    category_code: str,
    run_id: str,
    ngram: Any,
    hash_length: int = 16,
) -> str:
    """Build a deterministic identity for one noise term within a category run."""
    category = str(category_code or "").strip().upper()
    run = str(run_id or "").strip()
    term = normalize_ngram(ngram)
    if not category:
        raise ValueError("category_code 不能为空。")
    if not run:
        raise ValueError("run_id 不能为空。")
    if not term:
        raise ValueError("ngram 不能为空。")
    if not isinstance(hash_length, int) or not 12 <= hash_length <= 64:
        raise ValueError("hash_length 必须为 12 到 64 的整数。")

    payload = {
        "analysis_unit": ANALYSIS_UNIT_NOISE_TERM,
        "category_code": category,
        "run_id": run,
        "ngram": term,
    }
    digest = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()[:hash_length]
    return f"noise_term:{digest}"


def analysis_unit_display_name(
    *,
    analysis_unit: str,
    cluster_id: int | None = None,
    ngram: Any = None,
) -> str:
    """Return a concise human-readable label for pages and exports."""
    if analysis_unit == ANALYSIS_UNIT_CLUSTER:
        if cluster_id is None:
            raise ValueError("Cluster display name 需要 cluster_id。")
        return f"Cluster {int(cluster_id)}"
    if analysis_unit == ANALYSIS_UNIT_NOISE_TERM:
        term = normalize_ngram(ngram)
        if not term:
            raise ValueError("Noise Term display name 需要 ngram。")
        return f"未稳定归簇词｜{term}"
    raise ValueError(
        f"analysis_unit 非法：{analysis_unit!r}；允许值={sorted(VALID_ANALYSIS_UNITS)}"
    )


def eligible_trend_terms(
    frame: pd.DataFrame,
    *,
    growth_rate_threshold: float,
) -> pd.DataFrame:
    """Apply the shared technical eligibility contract to trend-term rows.

    Eligibility is intentionally narrow and mirrors the upstream trend rule:
    - non-empty normalized ngram
    - finite numeric growth_rate
    - strict growth_rate > configured threshold
    - one row per normalized ngram

    Cohesion, cluster probability, outlier score, and Top-N ranking are not
    eligibility gates here.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 必须为 pandas.DataFrame。")
    missing = {"ngram", "growth_rate"} - set(frame.columns)
    if missing:
        raise KeyError(f"趋势词数据缺少字段：{sorted(missing)}")

    threshold = float(growth_rate_threshold)
    if not math.isfinite(threshold):
        raise ValueError("growth_rate_threshold 必须是有限数值。")

    result = frame.copy()
    result["_normalized_ngram"] = result["ngram"].map(normalize_ngram)
    result["growth_rate"] = pd.to_numeric(
        result["growth_rate"], errors="coerce"
    )
    finite_growth = result["growth_rate"].map(
        lambda value: isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    result = result[
        result["_normalized_ngram"].ne("")
        & finite_growth
        & result["growth_rate"].gt(threshold)
    ].copy()

    # Stable ordering preserves the upstream priority before deterministic dedupe.
    result = result.drop_duplicates(
        subset=["_normalized_ngram"], keep="first"
    )
    result["ngram"] = result["_normalized_ngram"]
    return result.drop(columns=["_normalized_ngram"]).reset_index(drop=True)
