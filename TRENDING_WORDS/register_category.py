"""Bootstrap a new category config and register it before the main pipeline runs.

Usage:
    uv run register_category.py --category CHEESE --service 03 --template YD

The script:
1. Reads the Chinese category name from Oracle where SEGTYPE=CATEGORY.
2. Creates configs/<CATEGORY>.toml from an existing category template.
3. Adds/updates [categories.<CATEGORY>] in configs/category_registry.toml.

It is intentionally independent of ProjectContext because ProjectContext cannot load a
category until its config and registry entry already exist.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

for root in (WORKSPACE_ROOT, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.connection import create_oracle_connection_pool, execute_query  # noqa: E402

CATEGORY_NAME_SQL = """
SELECT DISTINCT
    t2.CSEGMENT AS CSEGMENT
FROM db_cate_segment t1
JOIN db_dic_segment t2
  ON t1.segno = t2.segno
 AND t1.catcode = t2.catcode
WHERE t1.catcode = :category_code
  AND UPPER(TRIM(t2.SEGTYPE)) = 'CATEGORY'
ORDER BY t2.CSEGMENT
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one category TOML and register it automatically"
    )
    parser.add_argument("--category", required=True, help="新建品类代码，例如 CHEESE")
    parser.add_argument(
        "--service",
        default="03",
        choices=("02", "03"),
        help="Oracle service，默认 03",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="用于继承参数的已有品类代码，例如 YD；默认使用 registry active_category",
    )
    parser.add_argument(
        "--taxonomy-category-code",
        default=None,
        help="Oracle Taxonomy catcode；默认等于 --category",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的 configs/<CATEGORY>.toml",
    )
    return parser.parse_args()


def normalize_code(value: str) -> str:
    code = re.sub(r"[^A-Za-z0-9_-]+", "", str(value or "").strip()).upper()
    if not code:
        raise ValueError("category code 不能为空")
    return code


def query_category_name(*, service: str, taxonomy_category_code: str) -> str:
    pool = create_oracle_connection_pool(service=service)
    try:
        rows, _ = execute_query(
            pool,
            CATEGORY_NAME_SQL,
            params={"category_code": taxonomy_category_code},
            return_header=True,
        )
    finally:
        try:
            pool.close()
        except Exception:
            pass

    names = {str(row[0]).strip() for row in rows if row and str(row[0] or "").strip()}
    if not names:
        raise RuntimeError(
            f"未找到 catcode={taxonomy_category_code} 且 SEGTYPE=CATEGORY 的 CSEGMENT"
        )
    if len(names) != 1:
        raise RuntimeError(
            f"catcode={taxonomy_category_code} 对应多个 CATEGORY 中文名：{sorted(names)}"
        )
    return next(iter(names))


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"不支持的 TOML 值类型：{type(value).__name__}")


def dump_toml(data: dict[str, Any]) -> str:
    """Serialize the nested table shape used by category configs and registry."""
    lines: list[str] = []

    def emit_table(path: list[str], values: dict[str, Any]) -> None:
        scalars = [
            (key, value)
            for key, value in values.items()
            if not isinstance(value, dict)
        ]
        children = [
            (key, value)
            for key, value in values.items()
            if isinstance(value, dict)
        ]
        if path:
            lines.append(f"[{' . '.join(path).replace(' . ', '.')}]")
        for key, value in scalars:
            lines.append(f"{key} = {toml_value(value)}")
        if path or scalars:
            lines.append("")
        for key, child in children:
            emit_table([*path, key], child)

    root_scalars = {
        key: value for key, value in data.items() if not isinstance(value, dict)
    }
    for key, value in root_scalars.items():
        lines.append(f"{key} = {toml_value(value)}")
    if root_scalars:
        lines.append("")
    for key, value in data.items():
        if isinstance(value, dict):
            emit_table([key], value)
    return "\n".join(lines).rstrip() + "\n"


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"defaults": {}, "categories": {}}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_template_code(
    registry: dict[str, Any],
    requested: str | None,
) -> str:
    if requested:
        return normalize_code(requested)
    active = str((registry.get("defaults") or {}).get("active_category") or "").strip()
    if not active:
        raise RuntimeError("registry 没有 active_category，请显式传入 --template")
    return normalize_code(active)


def build_category_config(
    template: dict[str, Any],
    *,
    category_code: str,
    category_name: str,
    taxonomy_category_code: str,
    service: str,
) -> dict[str, Any]:
    # Deep copy using JSON is sufficient because category TOMLs only contain JSON-safe scalars/lists.
    config = json.loads(json.dumps(template, ensure_ascii=False))
    config.setdefault("category", {})
    config["category"]["code"] = category_code
    config["category"]["name"] = category_name

    config.setdefault("source", {})
    config["source"]["bundle"] = category_code
    config["source"]["taxonomy_catcode"] = taxonomy_category_code
    config["source"]["taxonomy_file"] = f"{category_code}_SEGMENT_LABEL.xlsx"

    config.setdefault("etl", {})
    config["etl"]["service"] = service
    return config


def main() -> None:
    args = parse_args()
    category_code = normalize_code(args.category)
    taxonomy_category_code = normalize_code(
        args.taxonomy_category_code or category_code
    )
    configs_dir = PROJECT_ROOT / "configs"
    registry_path = configs_dir / "category_registry.toml"
    target_path = configs_dir / f"{category_code}.toml"

    registry = load_registry(registry_path)
    template_code = resolve_template_code(registry, args.template)
    template_path = configs_dir / f"{template_code}.toml"
    if not template_path.exists():
        raise FileNotFoundError(f"未找到模板配置：{template_path}")
    if target_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"目标配置已存在：{target_path}；如需覆盖请增加 --overwrite"
        )

    category_name = query_category_name(
        service=args.service,
        taxonomy_category_code=taxonomy_category_code,
    )
    with template_path.open("rb") as handle:
        template = tomllib.load(handle)
    category_config = build_category_config(
        template,
        category_code=category_code,
        category_name=category_name,
        taxonomy_category_code=taxonomy_category_code,
        service=args.service,
    )

    categories = registry.setdefault("categories", {})
    categories[category_code] = {
        "name": category_name,
        "config": f"configs/{category_code}.toml",
        "enabled": True,
    }
    registry.setdefault("defaults", {})
    if not registry["defaults"].get("active_category"):
        registry["defaults"]["active_category"] = category_code

    category_text = dump_toml(category_config)
    registry_text = dump_toml(registry)
    # Validate both outputs before writing either file.
    tomllib.loads(category_text)
    tomllib.loads(registry_text)

    configs_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(category_text, encoding="utf-8")
    registry_path.write_text(registry_text, encoding="utf-8")

    print(f"[OK] category={category_code} name={category_name}")
    print(f"Created: {target_path}")
    print(f"Updated: {registry_path}")
    print(
        "Next: uv run extract_taxonomy_source.py "
        f"--category {category_code} --service {args.service}"
    )


if __name__ == "__main__":
    main()
