"""Cluster LLM 输出解析与业务校验（Schema v4）。"""
from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from typing import Any
from urllib.parse import urlparse
from core.config import ALLOWED_CLUSTER_QUALITY, ALLOWED_EXTERNAL_RESEARCH_STATUS, ALLOWED_MAPPING_TYPES
from core.mapping_validation import build_mapping_context, validate_mapping

@dataclass(slots=True)
class ValidationResult:
    valid: bool; phase: str; parsed: dict[str, Any] | None; errors: list[str]; warnings: list[str]; error_code: str | None = None
    def to_dict(self): return asdict(self)
    @property
    def error_text(self): return " | ".join(self.errors)
    @property
    def warning_text(self): return " | ".join(self.warnings)

def extract_json_object(text: str):
    if not isinstance(text, str) or not text.strip(): return None, "EMPTY_OUTPUT: 模型未返回文本"
    clean = text.strip().lstrip("\ufeff")
    candidates = [clean]
    if clean.startswith("```") and clean.endswith("```") and "\n" in clean:
        candidates.append(clean[clean.find("\n") + 1:-3].strip())
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict): return obj, None
        except Exception: pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(clean):
        if ch != "{": continue
        try:
            obj, _ = decoder.raw_decode(clean[i:])
            if isinstance(obj, dict): return obj, None
        except json.JSONDecodeError: continue
    return None, "JSON_PARSE_ERROR: 无法提取合法 JSON 对象"

def parse_and_validate_internal(output_text, evidence, *, reject_unknown_fields=True):
    obj, err = extract_json_object(output_text)
    if obj is None: return ValidationResult(False, "internal", None, [err], [], err.split(":", 1)[0])
    return validate_internal_insight(obj, evidence, reject_unknown_fields=reject_unknown_fields)

def validate_internal_insight(result, evidence, *, reject_unknown_fields=True):
    errors, warnings = [], []
    allowed = {"cluster_id","cluster_name","mapping_type","primary_mapping","secondary_mappings","proposed_new_attribute","proposed_new_label","cluster_quality","split_recommended","suspected_outliers","key_driver_terms","observed_evidence","trend_summary","confidence","review_required","review_reason","external_research_recommended","external_search_queries"}
    missing = allowed - set(result)
    if missing: errors.append(f"root 缺少字段：{sorted(missing)}")
    unknown = set(result) - allowed
    if unknown: (errors if reject_unknown_fields else warnings).append(f"root 含未定义字段：{sorted(unknown)}")
    if result.get("cluster_id") != evidence.get("cluster_id"): errors.append("CLUSTER_ID_MISMATCH")
    if result.get("mapping_type") not in ALLOWED_MAPPING_TYPES: errors.append("mapping_type非法")
    if result.get("cluster_quality") not in ALLOWED_CLUSTER_QUALITY: errors.append("cluster_quality非法")
    confidence = result.get("confidence")
    if not _number(confidence) or not 0 <= float(confidence) <= 1: errors.append("confidence必须为0到1")
    for key in ("split_recommended","review_required","external_research_recommended"):
        if not isinstance(result.get(key), bool): errors.append(f"{key}必须为布尔值")
    directory = {x.get("attribute_code"): x.get("attribute_name") for x in evidence.get("all_existing_attributes", []) if isinstance(x, dict)}
    candidates = {x.get("attribute_code"): x for x in evidence.get("taxonomy_candidates", []) if isinstance(x, dict)}
    terms = {x.get("ngram") for x in evidence.get("representative_terms", []) if isinstance(x, dict)}
    primary = result.get("primary_mapping")
    if not isinstance(primary, dict): errors.append("primary_mapping必须是对象"); primary = {}
    _validate_mapping(primary, "primary_mapping", directory, candidates, result, errors, warnings, False)
    _validate_direction_recommendation(primary, "primary_mapping", result, errors)
    secondary = result.get("secondary_mappings")
    if not isinstance(secondary, list): errors.append("secondary_mappings必须是数组"); secondary = []
    allowed_secondary = {"attribute_code","attribute_name","existing_label","proposed_new_attribute","proposed_new_label","reason","direction_recommendation","direction_recommendation_reason"}
    for i, mapping in enumerate(secondary):
        if not isinstance(mapping, dict): errors.append(f"secondary_mappings[{i}]必须是对象"); continue
        extra = set(mapping) - allowed_secondary
        if extra: errors.append(f"secondary_mappings[{i}]含未定义字段：{sorted(extra)}")
        _validate_mapping(mapping, f"secondary_mappings[{i}]", directory, candidates, result, errors, warnings, True)
        _validate_direction_recommendation(mapping, f"secondary_mappings[{i}]", result, errors)

    # A multi-attribute cluster must contain at least two distinct, evidence-backed
    # business dimensions. Multiple labels under the same attribute count as one
    # direction; brand/noise/data-quality observations do not create a direction.
    if result.get("mapping_type") == "multi_attribute_cluster":
        direction_keys: set[tuple[str, str]] = set()
        mappings = [primary, *secondary]
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            existing_code = str(mapping.get("attribute_code") or "").strip()
            proposed_attribute = str(
                mapping.get("proposed_new_attribute") or ""
            ).strip()
            reason = str(mapping.get("reason") or "").strip()
            if existing_code:
                direction_keys.add(("existing", existing_code.casefold()))
            elif proposed_attribute and reason:
                direction_keys.add(("new", proposed_attribute.casefold()))
        if len(direction_keys) < 2:
            errors.append(
                "MULTI_ATTRIBUTE_REQUIRES_AT_LEAST_TWO_DIRECTIONS"
            )

    for key in ("suspected_outliers", "key_driver_terms"):
        values = result.get(key)
        if not isinstance(values, list): errors.append(f"{key}必须是数组")
        else:
            bad = [x for x in values if not isinstance(x, str) or x not in terms]
            if bad: errors.append(f"{key}含非代表词：{bad}")
    if not isinstance(result.get("observed_evidence"), list) or not result.get("observed_evidence"): errors.append("observed_evidence不能为空")
    if not isinstance(result.get("external_search_queries"), list) or len(result.get("external_search_queries", [])) > 3: errors.append("external_search_queries必须为最多3条数组")
    return ValidationResult(not errors, "internal", result, errors, warnings, _code(errors))

def _validate_direction_recommendation(mapping, path, result, errors):
    allowed = {"use_existing_label", "direct_addition", "derived_label", "new_attribute", "taxonomy_restructure", "continue_validation", "not_recommended"}
    recommendation = mapping.get("direction_recommendation")
    reason = mapping.get("direction_recommendation_reason")
    existing_label = mapping.get("existing_label")
    proposed_label = mapping.get("proposed_new_label")
    proposed_attribute = mapping.get("proposed_new_attribute")
    if recommendation not in allowed: errors.append(f"{path}.direction_recommendation非法")
    if not isinstance(reason, str) or not reason.strip(): errors.append(f"{path}.direction_recommendation_reason不能为空")
    if recommendation == "use_existing_label" and (existing_label is None or proposed_label is not None): errors.append(f"{path}:use_existing_label必须填写existing_label且不得填写proposed_new_label")
    if recommendation == "direct_addition" and not proposed_label:
        mapping["direction_recommendation"] = "continue_validation"
        if not str(mapping.get("direction_recommendation_reason") or "").strip():
            mapping["direction_recommendation_reason"] = (
                "模型建议直接新增，但未提供明确、单一的候选标签名称；"
                "已转为继续验证，需确认标准标签命名、层级、边界和赋值规则。"
            )
        warnings.append(f"{path}:DIRECT_ADDITION_WITHOUT_LABEL_NORMALIZED")
        recommendation = "continue_validation"
    if recommendation == "direct_addition" and existing_label:
        errors.append(f"{path}:direct_addition不得同时填写existing_label")
    if recommendation == "derived_label" and not proposed_label:
        errors.append(f"{path}:derived_label必须提供proposed_new_label")
    if recommendation == "new_attribute":
        is_primary_new_attribute = (
            path == "primary_mapping"
            and result.get("mapping_type") == "new_attribute_new_label"
            and bool(result.get("proposed_new_attribute"))
            and bool(result.get("proposed_new_label"))
        )
        is_secondary_new_attribute = (
            path.startswith("secondary_mappings[")
            and bool(proposed_attribute)
            and bool(proposed_label)
        )
        if not (is_primary_new_attribute or is_secondary_new_attribute):
            errors.append(
                f"{path}:建议新建属性时，主方向必须在根级提供新属性和首个标签；"
                "次方向必须在本方向提供新属性和首个标签"
            )
    if recommendation == "taxonomy_restructure" and result.get("review_required") is not True: errors.append(f"{path}:taxonomy_restructure必须review_required=true")
    if recommendation == "not_recommended" and proposed_label is not None: errors.append(f"{path}:not_recommended不得保留候选新标签")

def _validate_mapping(mapping, path, directory, candidates, result, errors, warnings, allow_new_attribute):
    """Backward-compatible wrapper around the shared Mapping validator."""
    validate_mapping(
        mapping,
        path=path,
        directory=directory,
        candidates=candidates,
        result=result,
        errors=errors,
        warnings=warnings,
        allow_new_attribute=allow_new_attribute,
    )

def parse_and_validate_external(output_text, evidence, internal_insight, *, degraded=False, reject_unknown_fields=True):
    if degraded: return ValidationResult(False,"external",None,["WEB_RESEARCH_DEGRADED_WITHOUT_TOOLS"],[],"WEB_RESEARCH_DEGRADED_WITHOUT_TOOLS")
    obj, err = extract_json_object(output_text)
    if obj is None: return ValidationResult(False,"external",None,[err],[],err.split(":",1)[0])
    errors=[]; warnings=[]
    if obj.get("cluster_id") != evidence.get("cluster_id") or obj.get("cluster_id") != internal_insight.get("cluster_id"): errors.append("CLUSTER_ID_MISMATCH")
    if obj.get("external_research_status") not in ALLOWED_EXTERNAL_RESEARCH_STATUS: errors.append("external_research_status非法")
    findings = obj.get("external_findings", []); urls=set()
    if not isinstance(findings, list): errors.append("external_findings必须是数组"); findings=[]
    for i, finding in enumerate(findings):
        if not isinstance(finding, dict): errors.append(f"external_findings[{i}]无效"); continue
        url=finding.get("source_url")
        if not _url(url): errors.append(f"external_findings[{i}].source_url无效")
        else: urls.add(url)
    return ValidationResult(not errors,"external",obj,errors,warnings,_code(errors))

def _number(value): return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(float(value))
def _url(value):
    if not isinstance(value,str): return False
    p=urlparse(value); return p.scheme in {"http","https"} and bool(p.netloc)
def _code(errors):
    if not errors: return None
    for code in ("CLUSTER_ID_MISMATCH","JSON_PARSE_ERROR","EMPTY_OUTPUT","WEB_RESEARCH_DEGRADED_WITHOUT_TOOLS"):
        if any(code in e for e in errors): return code
    return "SCHEMA_VALIDATION_ERROR"
