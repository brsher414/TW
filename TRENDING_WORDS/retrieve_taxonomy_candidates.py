from __future__ import annotations

import argparse
import sys
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
    build_cluster_query_key,
    build_noise_term_query_key,
    eligible_trend_terms,
    normalize_ngram,
)
from core.project_context import ProjectContext
from core.run_manifest import create_manifest, update_stage
from core.taxonomy_common import (
    EXCLUDED_DIMENSION_CODES,
    encode_texts,
    json_dump,
    valid_labels_by_code,
)


def _split(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_ngram(item) for item in value if normalize_ngram(item)]

    text = str(value or "").strip()
    for separator in (";", "、", "|", ","):
        if separator in text:
            return [
                normalize_ngram(item)
                for item in text.split(separator)
                if normalize_ngram(item)
            ]
    return [normalize_ngram(text)] if normalize_ngram(text) else []


def _terms(row: pd.Series) -> list[str]:
    for column in ("top_terms", "representative_terms", "terms"):
        if column in row and str(row[column]).strip():
            return _split(row[column])
    return []


def _labels(labels: list[str], terms: list[str]) -> list[dict[str, Any]]:
    joined = " ".join(terms).casefold()
    output: list[dict[str, Any]] = []

    for rank, label in enumerate(labels, 1):
        exact_terms = [
            term for term in terms if term.casefold() == label.casefold()
        ]
        output.append(
            {
                "rank": rank,
                "label": label,
                "similarity": 1.0 if exact_terms else 0.0,
                "normalized_exact_match": bool(exact_terms),
                "exact_match_terms": exact_terms,
                "string_containment_match": label.casefold() in joined,
            }
        )
    return output


def _units(
    context: ProjectContext,
    clusters_path: Path,
    noise_path: Path,
) -> list[dict[str, Any]]:
    clusters = pd.read_parquet(clusters_path)
    clusters["cluster_id"] = pd.to_numeric(
        clusters["cluster_id"], errors="coerce"
    )
    clusters = clusters[clusters.cluster_id.ge(0)]

    units: list[dict[str, Any]] = []
    for _, row in clusters.iterrows():
        cluster_id = int(row.cluster_id)
        terms = _terms(row)
        if terms:
            units.append(
                {
                    "analysis_unit": ANALYSIS_UNIT_CLUSTER,
                    "query_key": build_cluster_query_key(cluster_id),
                    "cluster_id": cluster_id,
                    "source_cluster_id": cluster_id,
                    "term": None,
                    "terms": terms,
                    "query_text": "趋势主题代表词：" + "、".join(terms),
                }
            )

    noise = eligible_trend_terms(
        pd.read_parquet(noise_path),
        growth_rate_threshold=float(context.trend["growth_rate_threshold"]),
    )
    mask = pd.Series(False, index=noise.index)
    if "cluster_id" in noise:
        mask |= pd.to_numeric(noise.cluster_id, errors="coerce").eq(-1)
    if "is_noise" in noise:
        mask |= noise.is_noise.fillna(False).astype(bool)

    for term in noise.loc[mask, "ngram"].tolist():
        units.append(
            {
                "analysis_unit": ANALYSIS_UNIT_NOISE_TERM,
                "query_key": build_noise_term_query_key(
                    category_code=context.category_code,
                    run_id=context.run_id,
                    ngram=term,
                ),
                "cluster_id": -1,
                "source_cluster_id": -1,
                "term": term,
                "terms": [term],
                "query_text": "趋势词：" + term,
            }
        )

    keys = [item["query_key"] for item in units]
    if len(keys) != len(set(keys)):
        raise ValueError("Analysis Unit query_key 重复")
    return units


def run(
    context: ProjectContext,
    clusters_path: Path,
    noise_path: Path,
    reference_path: Path,
    embeddings_path: Path,
    source_path: Path,
    output_path: Path,
    model_name: str,
) -> None:
    units = _units(context, clusters_path, noise_path)
    if not units:
        raise RuntimeError("没有可检索 Analysis Unit")

    reference = pd.read_parquet(reference_path)
    invalid_codes = EXCLUDED_DIMENSION_CODES & set(reference.attribute_code)
    if invalid_codes:
        raise AssertionError(
            f"非属性维度进入Reference:{sorted(invalid_codes)}"
        )

    vectors = np.load(embeddings_path)
    if len(vectors) != len(reference):
        raise ValueError("Reference与Embedding行数不一致")

    labels_by_code = valid_labels_by_code(pd.read_parquet(source_path))
    query_vectors = encode_texts(
        [item["query_text"] for item in units],
        model_name,
    )
    scores = query_vectors @ vectors.T
    rows: list[dict[str, Any]] = []

    for index, unit in enumerate(units):
        candidates: list[dict[str, Any]] = []

        # All valid Taxonomy attributes are provided to the LLM. Similarity and
        # rank are retained only as relevance signals; no Top-K truncation is
        # applied. Every included attribute carries all of its valid labels.
        for rank, reference_index in enumerate(np.argsort(-scores[index]), 1):
            reference_row = reference.iloc[int(reference_index)]
            code = str(reference_row.attribute_code)
            labels = labels_by_code.get(code, [])
            label_objects = _labels(labels, unit["terms"])

            candidates.append(
                {
                    "rank": rank,
                    "attribute_code": code,
                    "attribute_name": str(reference_row.attribute_name),
                    "similarity": float(scores[index, int(reference_index)]),
                    "label_evidence_mode": (
                        "all_valid_labels" if labels else "none"
                    ),
                    "total_valid_label_count": len(labels),
                    "provided_label_count": len(label_objects),
                    "label_list_complete": bool(labels),
                    "candidate_labels": label_objects,
                }
            )

        rows.append(
            {
                "analysis_unit": unit["analysis_unit"],
                "query_key": unit["query_key"],
                "cluster_id": unit["cluster_id"],
                "source_cluster_id": unit["source_cluster_id"],
                "term": unit["term"],
                "representative_terms_json": json_dump(unit["terms"]),
                "taxonomy_candidates_json": json_dump(candidates),
            }
        )

    output = pd.DataFrame(rows).sort_values(
        ["analysis_unit", "query_key"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    output.to_csv(
        output_path.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print("[OK]", output.analysis_unit.value_counts().to_dict(), "->", output_path)
    print(
        "[OK] 每个分析对象已写入全部有效 Taxonomy 属性及其全部有效标签；"
        "similarity/rank 仅用于排序。"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category")
    parser.add_argument("--run-id")
    parser.add_argument("--clusters", type=Path)
    parser.add_argument("--noise-terms", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--taxonomy-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    # Accepted only so an old command or pipeline does not break. The value is
    # intentionally ignored because candidate attributes are no longer cut off.
    parser.add_argument(
        "--top-k-attributes",
        type=int,
        help=argparse.SUPPRESS,
    )
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
    update_stage(context, "taxonomy", "retrieving_candidates")

    try:
        output = args.output or context.taxonomy_candidate_review_file
        model_name = args.model or str(
            context.taxonomy.get(
                "reference_embedding_model",
                context.cluster.get("embedding_model", "BAAI/bge-m3"),
            )
        )
        run(
            context,
            args.clusters or context.cluster_summary_file,
            args.noise_terms or context.cluster_noise_terms_file,
            args.reference or context.taxonomy_reference_file,
            args.embeddings or context.taxonomy_embeddings_file,
            args.taxonomy_source or context.taxonomy_source_normalized_file,
            output,
            model_name,
        )
        update_stage(
            context,
            "taxonomy",
            "retrieval_completed",
            artifacts={
                "taxonomy_candidate_review": output,
                "taxonomy_candidate_review_csv": output.with_suffix(".csv"),
            },
        )
    except Exception:
        update_stage(context, "taxonomy", "failed")
        raise


if __name__ == "__main__":
    main()
