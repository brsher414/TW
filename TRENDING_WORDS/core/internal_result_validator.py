from __future__ import annotations
from typing import Any
from core.taxonomy_common import FACT_BASED_MARKET_EXPRESSIONS, clean

def validate_mapping(mapping: dict[str,Any], taxonomy_candidates: list[dict[str,Any]]) -> list[str]:
    errors=[];code=clean(mapping.get("attribute_code"));proposal=clean(mapping.get("proposed_new_label"))
    by_code={clean(x.get("attribute_code")):x for x in taxonomy_candidates}
    if code and code not in by_code: errors.append(f"EXISTING_ATTRIBUTE_OUTSIDE_RETRIEVAL_CANDIDATES:{code}")
    labels={clean(x) for x in (by_code.get(code,{}).get("all_labels") or [])}
    if proposal and proposal in labels: errors.append(f"PROPOSED_LABEL_ALREADY_EXISTS:{code}:{proposal}")
    if proposal in FACT_BASED_MARKET_EXPRESSIONS.get(code,set()): errors.append(f"MARKET_EXPRESSION_CANNOT_BE_CREATED_AS_RANGE_LABEL:{code}:{proposal}")
    return errors

def validate_internal_result(parsed_result: dict[str,Any], taxonomy_candidates: list[dict[str,Any]]) -> list[str]:
    mappings=[]
    if isinstance(parsed_result.get("primary_mapping"),dict):mappings.append(parsed_result["primary_mapping"])
    mappings += [x for x in parsed_result.get("secondary_mappings",[]) if isinstance(x,dict)]
    return [e for m in mappings for e in validate_mapping(m,taxonomy_candidates)]
