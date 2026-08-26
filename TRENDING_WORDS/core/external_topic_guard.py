from __future__ import annotations
from typing import Any
from core.taxonomy_business_rules import new_label_eligibility


def assess_mapping_for_external_research(mapping: dict[str, Any], evidence: dict[str, Any]) -> tuple[bool, str | None]:
    if mapping.get("attribute_code") and mapping.get("proposed_new_label"):
        return new_label_eligibility(mapping, evidence)
    return True, None
