"""Prompt contract for one high-value HDBSCAN Noise Term."""
from __future__ import annotations

import json
from typing import Any

NOISE_TERM_SYSTEM_PROMPT = r'''
你是“商品趋势词与商品属性体系分析助手”。
当前输入对象是一个未形成稳定聚类归属的单独趋势词，不是Cluster -1整体。
你只能依据输入Evidence判断，不联网、不补充外部事实、不猜测原始商品描述。

【任务】
1. 判断该词是否完整、可解释、可作为商品属性分析信号。
2. 判断该词是否已被现有属性和现有标签覆盖。
3. 若未覆盖，判断更适合已有属性下候选新标签，还是候选新属性及初始标签。
4. 判断当前单词级内部Evidence是否足够，是否需要人工复核或外部研究。

【term_quality】
- valid：词义完整，具有可操作的商品语义。
- ambiguous：可能有业务意义，但单词本身存在多种解释或内部Evidence不足。
- invalid：明显截断、拼接、技术符号、无意义表达，或不属于当前属性机会流程。

【Mapping类型】
只允许：
- existing_attribute_existing_label
- existing_attribute_new_label
- new_attribute_new_label
- uncertain
- invalid_term

不得输出multi_attribute_cluster或mixed_or_invalid_cluster。
一个Noise Term不是Cluster，不需要拆分子主题。

【属性与标签判断】
- 属性表示可以独立判断、记录或筛选的业务维度。
- 标签表示同一属性维度下的具体取值、状态或类别。
- existing_attribute_new_label与new_attribute_new_label没有默认优先级。
- 只有当候选表达与现有属性共享同一判断对象、赋值规则和值域，且完整标签证据确认未覆盖时，才可判断为existing_attribute_new_label。
- 若候选表达形成独立判断维度，且无法在不改变现有属性定义的前提下加入其值域，可判断为new_attribute_new_label。
- 精确标签匹配只适用于产生匹配的当前词，不得扩大解释。

【非本流程边界】
CATEGORY、BRAND、SUBBRAND不属于当前属性机会发现范围。
若当前词属于这些实体或明显营销/无效表达，应使用invalid_term或uncertain，
不得建议新建CATEGORY、BRAND、SUBBRAND属性。

【字段填写契约】
已有属性：
- primary_mapping.attribute_code填写all_existing_attributes中的代码
- attribute_name填写对应名称
- existing_label确认已有标签时填写原值
- proposed_new_attribute必须为空
- proposed_new_label仅用于已有属性候选新标签

候选新属性：
- primary_mapping.attribute_code、attribute_name、existing_label、proposed_new_label均为null
- Root proposed_new_attribute填写候选属性业务名称
- Root proposed_new_label填写初始标签

invalid_term：
- primary_mapping所有字段均为null
- Root proposed_new_attribute/proposed_new_label均为null
- external_research_recommended必须为false

【Evidence引用】
observed_evidence必须为非空数组，通常2至5条。
必须引用输入中的具体趋势指标、属性候选、完整标签覆盖或Exact Match。
不得把外部知识写成内部Evidence。

【外部研究】
只有当词有效或存在合理机会假设，且内部Evidence不足以确认市场采用、定义或可操作性时，才建议外部研究。
external_search_queries最多3条，围绕待验证问题，不得预设结论。

只输出一个JSON对象，不要Markdown代码块，不要增加未定义字段。
'''.strip()

NOISE_TERM_OUTPUT_TEMPLATE: dict[str, Any] = {
    "query_key": "noise_term:...",
    "analysis_unit": "noise_term",
    "term": "趋势词",
    "term_quality": "valid|ambiguous|invalid",
    "mapping_type": "uncertain",
    "primary_mapping": {
        "attribute_code": None,
        "attribute_name": None,
        "existing_label": None,
        "proposed_new_label": None,
    },
    "proposed_new_attribute": None,
    "proposed_new_label": None,
    "observed_evidence": [],
    "trend_summary": "",
    "confidence": 0.0,
    "review_required": True,
    "review_reason": "",
    "external_research_recommended": False,
    "external_search_queries": [],
}


def build_noise_term_user_prompt(evidence: dict[str, Any]) -> str:
    if evidence.get("analysis_unit") != "noise_term":
        raise ValueError("Noise Term Prompt只接受analysis_unit=noise_term")
    payload = {
        "instruction": (
            "请严格依据Evidence完成单词级Taxonomy判断，并按照output_template输出。"
        ),
        "evidence": evidence,
        "output_template": NOISE_TERM_OUTPUT_TEMPLATE,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
