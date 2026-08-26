"""Shared Taxonomy mapping validation for Cluster and Noise Term schemas."""
from __future__ import annotations

from typing import Any


def build_mapping_context(
    evidence: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    directory = {
        str(item.get("attribute_code")): str(item.get("attribute_name"))
        for item in evidence.get("all_existing_attributes", [])
        if isinstance(item, dict)
        and isinstance(item.get("attribute_code"), str)
        and item.get("attribute_code")
        and isinstance(item.get("attribute_name"), str)
        and item.get("attribute_name")
    }
    candidates = {
        str(item.get("attribute_code")): item
        for item in evidence.get("taxonomy_candidates", [])
        if isinstance(item, dict)
        and isinstance(item.get("attribute_code"), str)
        and item.get("attribute_code")
    }
    return directory, candidates


def validate_mapping(
    mapping: Any,
    *,
    path: str,
    directory: dict[str, str],
    candidates: dict[str, dict[str, Any]],
    result: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    allow_new_attribute: bool,
) -> None:
    """Validate one existing/new-attribute mapping without changing its meaning."""
    if not isinstance(mapping, dict):
        errors.append(f"{path}必须是对象")
        return

    code = mapping.get("attribute_code")
    name = mapping.get("attribute_name")
    existing = mapping.get("existing_label")
    new_attribute = (
        mapping.get("proposed_new_attribute")
        if allow_new_attribute
        else None
    )
    new_label = mapping.get("proposed_new_label")

    if code is not None:
        if not isinstance(code, str) or not code.strip():
            errors.append(f"{path}.attribute_code必须是非空字符串或null")
            return
        if new_attribute is not None:
            errors.append(
                f"{path}使用已有属性时proposed_new_attribute必须为空"
            )
        if code not in directory:
            errors.append(
                f"{path}.attribute_code={code!r}不在all_existing_attributes"
            )
        elif directory[code] != name:
            errors.append(f"{path}.attribute_name与完整目录不一致")

        candidate = candidates.get(code)
        if candidate is None:
            if existing is not None:
                errors.append(
                    f"{path}.existing_label不可使用：该属性未进入详细候选，"
                    "当前Evidence未提供其标签证据"
                )
            if result.get("review_required") is not True:
                result["review_required"] = True
                reason = str(result.get("review_reason") or "").strip()
                addition = (
                    f"选择的已有属性 {code} 未进入Top-K详细候选，"
                    "缺少该属性的详细标签证据，需要人工复核。"
                )
                result["review_reason"] = f"{reason} {addition}".strip()
                warnings.append(
                    f"REVIEW_REQUIRED_AUTO_NORMALIZED:{path}:{code}"
                )
            warnings.append(
                f"EXISTING_ATTRIBUTE_OUTSIDE_RETRIEVAL_CANDIDATES:{code}"
            )
        elif existing is not None:
            labels = {
                item.get("label")
                for item in candidate.get("candidate_labels", [])
                if isinstance(item, dict)
            }
            if existing not in labels:
                errors.append(
                    f"{path}.existing_label={existing!r}"
                    "不在提供的标签证据中"
                )
        return

    if name is not None:
        errors.append(
            f"{path}.attribute_code为空时attribute_name也必须为空；"
            "新属性名称应写入proposed_new_attribute"
        )
    if existing is not None:
        errors.append(f"{path}没有已有属性时existing_label必须为空")
    if allow_new_attribute and new_attribute is not None:
        if not isinstance(new_attribute, str) or not new_attribute.strip():
            errors.append(
                f"{path}.proposed_new_attribute必须是非空字符串或null"
            )
    if allow_new_attribute and new_attribute is None and new_label is not None:
        warnings.append(
            f"{path}提出新标签但未声明所属已有属性或新属性"
        )
