"""Central category configuration and isolated run paths."""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _token(value: object) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value).strip())
    return text.strip("-") or "unknown"


def _number(value: object) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


@dataclass(frozen=True)
class ProjectContext:
    project_root: Path
    config_path: Path
    config: dict[str, Any]

    @classmethod
    def from_category(
        cls,
        category_code: str,
        *,
        project_root: Path,
    ) -> "ProjectContext":
        root = project_root.resolve()
        registry = _load_toml(root / "configs" / "category_registry.toml")
        code = category_code.strip().upper()
        entry = registry.get("categories", {}).get(code)
        if not isinstance(entry, dict) or not entry.get("enabled", False):
            raise ValueError(f"品类 {code!r} 未注册或未启用")
        config_path = (root / str(entry["config"])).resolve()
        context = cls(root, config_path, _load_toml(config_path))
        context.validate()
        return context

    @classmethod
    def active(cls, *, project_root: Path) -> "ProjectContext":
        root = project_root.resolve()
        registry = _load_toml(root / "configs" / "category_registry.toml")
        code = str(registry["defaults"]["active_category"])
        return cls.from_category(code, project_root=root)

    def with_run_id(self, run_id: str) -> "ProjectContext":
        config = json.loads(json.dumps(self.config, ensure_ascii=False))
        config.setdefault("runtime", {})["run_id_override"] = _token(run_id)
        return ProjectContext(self.project_root, self.config_path, config)

    def validate(self) -> None:
        for section in (
            "category", "source", "period", "trend", "cluster", "taxonomy"
        ):
            if not isinstance(self.config.get(section), dict):
                raise ValueError(f"配置缺少 [{section}] 区块")

    @property
    def category(self) -> dict[str, Any]:
        return self.config["category"]

    @property
    def source(self) -> dict[str, Any]:
        return self.config["source"]

    @property
    def period(self) -> dict[str, Any]:
        return self.config["period"]

    @property
    def trend(self) -> dict[str, Any]:
        return self.config["trend"]

    @property
    def cluster(self) -> dict[str, Any]:
        return self.config["cluster"]

    @property
    def taxonomy(self) -> dict[str, Any]:
        return self.config["taxonomy"]

    @property
    def category_code(self) -> str:
        return str(self.category["code"]).upper()

    @property
    def category_name(self) -> str:
        return str(self.category["name"])

    @property
    def run_id(self) -> str:
        override = self.config.get("runtime", {}).get("run_id_override")
        if override:
            return str(override)
        trend = self.trend
        return "_".join([
            f"{_token(self.period['base_quarter'])}_vs_{_token(self.period['current_quarter'])}",
            f"n{int(trend['min_ngram'])}_{int(trend['max_ngram'])}",
            f"len{int(trend['max_desc_chars'])}",
            f"min{int(trend['min_freq'])}",
            f"base{int(trend['min_base_count'])}",
            f"ctx{int(trend['min_context_diversity'])}",
            f"coh{_number(trend['min_cohesion'])}",
            f"growth{_number(trend['growth_rate_threshold'])}",
        ])

    @property
    def category_root(self) -> Path:
        return self.project_root / "data" / "categories" / self.category_code

    # Long-lived category input. No extra source/ directory.
    @property
    def taxonomy_source_file(self) -> Path:
        configured = Path(str(self.source["taxonomy_file"]))
        return configured if configured.is_absolute() else self.category_root / configured

    @property
    def staging_dir(self) -> Path:
        return self.category_root / "staging"

    @property
    def run_dir(self) -> Path:
        return self.category_root / "runs" / self.run_id

    @property
    def etl_dir(self) -> Path:
        return self.run_dir / "etl"

    @property
    def trend_dir(self) -> Path:
        return self.run_dir / "trend"

    @property
    def cluster_dir(self) -> Path:
        return self.run_dir / "cluster"

    @property
    def taxonomy_dir(self) -> Path:
        return self.run_dir / "taxonomy"

    @property
    def insights_dir(self) -> Path:
        return self.run_dir / "insights"

    @property
    def research_dir(self) -> Path:
        return self.run_dir / "research"

    @property
    def dashboard_cache_dir(self) -> Path:
        return self.run_dir / "dashboard_cache"

    @property
    def latest_file(self) -> Path:
        return self.category_root / "latest.json"

    @property
    def manifest_file(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def sampled_products_file(self) -> Path:
        return self.etl_dir / str(self.source["product_file"])

    @property
    def trend_clustered_file(self) -> Path:
        return self.cluster_dir / "trend_clustered.parquet"

    @property
    def cluster_summary_file(self) -> Path:
        return self.cluster_dir / "cluster_summary.parquet"

    @property
    def cluster_noise_terms_file(self) -> Path:
        """Run-scoped HDBSCAN noise terms retained for term-level analysis."""
        return self.cluster_dir / "cluster_noise_terms.parquet"

    @property
    def taxonomy_source_normalized_file(self) -> Path:
        return self.taxonomy_dir / "taxonomy_source_normalized.parquet"

    @property
    def taxonomy_reference_file(self) -> Path:
        return self.taxonomy_dir / "taxonomy_reference.parquet"

    @property
    def taxonomy_embeddings_file(self) -> Path:
        return self.taxonomy_dir / "taxonomy_attribute_embeddings.npy"

    @property
    def taxonomy_embedding_index_file(self) -> Path:
        return self.taxonomy_dir / "taxonomy_embedding_index.parquet"

    @property
    def category_context_file(self) -> Path:
        return self.taxonomy_dir / "category_context.json"

    @property
    def taxonomy_candidate_review_file(self) -> Path:
        return self.taxonomy_dir / "taxonomy_candidate_review.parquet"

    @property
    def cluster_evidence_file(self) -> Path:
        return self.taxonomy_dir / "cluster_llm_evidence.jsonl"

    def ensure_directories(self) -> None:
        self.category_root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.staging_dir,
            self.etl_dir,
            self.trend_dir,
            self.cluster_dir,
            self.taxonomy_dir,
            self.insights_dir,
            self.research_dir,
            self.dashboard_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
