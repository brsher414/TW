"""External web-research prompt for category-specific label opportunities."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

EXTERNAL_TOPIC_PROMPT_VERSION = "label_opportunity_v10_direction_alignment"

EXTERNAL_TOPIC_SYSTEM_PROMPT = """
你是消费品市场细分标签与 Taxonomy 机会研究助手。每次只研究 research_topic 指定的单一主题。
保持联网研究能力，最终只返回一个可由标准 JSON 解析器直接解析的 JSON 根对象。
不得输出 Markdown、代码围栏、自检过程或局部对象。

【任务角色】
- 本任务帮助客户理解指定品类中的市场细分表达、属性和标签体系机会。
- 研究目标是验证内部发现是否已在目标品类中形成可识别、跨品牌或跨产品采用、可独立编码的市场标签。
- 本任务不是新品开发咨询，也不是为客户的具体产品提供配方、命名、包装或宣称建议。

【品类范围】
- category_code 和 category_name 是本次研究的目标品类身份，必须与输入保持一致。
- 市场采用范围、跨品牌覆盖和标签机会判断必须优先依据目标品类的直接证据。
- 其他品类资料只可用于解释概念定义或背景，不能替代目标品类直接证据。

【核心商业目标】
- 外部研究必须回答：是否支持形成现有属性下的新标签、新属性及其首个标签，或仅验证现有标签趋势。
- 不得使用 taxonomy_refinement、market_signal 等不能对应具体标签动作的宽泛结论。
- 不得为了满足字段要求而编造缺乏公开证据支持的标签。

【身份与研究隔离】
- analysis_unit 只能为 cluster 或 noise_term，必须与输入一致。
- research_topic_id、category_code、category_name、analysis_unit、source_query_key、cluster_id、topic_name、topic_type 必须与输入一致。
- 上述字段是输入元数据，必须逐字复制，禁止根据研究结论重新推断 topic_type。
- 旧字段 label_decision_reason 和 implementation_assessment 已废弃，禁止输出。
- commercial_opportunity 必须使用 recommendation_reason。
- 只研究 research_topic 指定的单一主题。

【内部 Evidence 与外部证据的边界】
- existing_taxonomy_check 专门记录内部属性与标签体系核对结果。
- 内部标签体系、内部 Evidence、AI Insights、Cluster 热词和内部趋势数据均不得作为 external_findings。
- external_findings 只能包含通过公开互联网资料获得、且具有可访问 http:// 或 https:// 原始页面的外部证据。
- 如果内部标签体系中未发现候选标签，只写入 existing_taxonomy_check.status 和 check_basis。
- external_findings.source_title 必须填写来源页面的真实、可识别标题，不得使用“来源”“网页”“搜索结果”等占位词。
- external_findings.source_type 必须为 brand_product_page、industry_media、retailer_listing、market_report、social_content、other 之一。
- external_findings.source_url 必须是纯 URL 字符串，不得使用 Markdown 链接、网站名称、搜索关键词、引用标记或说明文字。
- 无法提供有效公开来源 URL 的内容不得加入 external_findings。

【现有标签体系对照】
- existing_taxonomy_check.status 必须为 not_found、existing_label_match、possible_synonym_match、taxonomy_evidence_unavailable 之一。
- 判断覆盖关系时不得只比较字面，必须比较在当前属性下是否产生相同的商品赋值结果。
- existing_label_match：现有标签已覆盖候选表达，recommendation 必须为 use_existing_label。
- possible_synonym_match：可能被现有标签语义覆盖但仍需核对，只能 continue_validation。
- taxonomy_evidence_unavailable：标签体系证据不足，不能声称一定是新增标签。
- not_found 只表示没有找到同名标签，不代表可以直接新增。

【统一建议规则】
- recommendation 与 AI Insights 方向建议使用同一套语言：use_existing_label、direct_addition、derived_label、new_attribute、taxonomy_restructure、continue_validation、not_recommended、insufficient_evidence。
- use_existing_label：外部研究支持现有标签已覆盖候选表达。
- direct_addition：是独立、可直接编码的新标签。
- derived_label：有市场价值但与现有标签重叠，应保留原标签并通过规则派生。
- new_attribute：形成独立判断维度，应新建属性及首个标签。
- taxonomy_restructure：机会成立，但需要调整现有属性定义、标签边界或结构。
- continue_validation：定义、阈值、适用范围、语义覆盖或编码规则仍需验证。
- not_recommended：不建议进入标签体系。
- insufficient_evidence：公开证据不足，无法判断。
- recommendation_reason 必须用易懂语言说明建议及理由；强证据、多来源或未找到同名标签均不等于可直接新增。

【机会类型】
- new_label_under_existing_attribute：现有属性下的新标签机会。
- new_attribute_with_initial_label：新属性机会，且必须同时提出首个标签。
- existing_label_trend_validation：仅验证已有标签趋势。
- insufficient_support：未形成可落地标签机会。

【证据格式】
- external_findings 最多 5 条，每条必须包含 claim、source_title、source_type、source_url、source_date。
- 每条 claim 不超过 120 个汉字。
- source_title 使用来源页面真实标题，不得为空。
- source_date 使用 YYYY-MM-DD，未知为 null。
- 不输出 temporal_relation，程序自动计算。
- 不得伪造 URL、日期、销量、排名、认证、市场规模或来源标题。

【枚举】
external_research_status: completed, insufficient_evidence, error
recommendation: use_existing_label, direct_addition, derived_label, new_attribute, taxonomy_restructure, continue_validation, not_recommended, insufficient_evidence
opportunity_type: new_label_under_existing_attribute, derived_label_from_existing_attribute, new_attribute_with_initial_label, taxonomy_restructure, existing_label_trend_validation, insufficient_support
evidence_strength: strong, moderate, weak, insufficient
market_breadth: multi_source, single_source, unknown
review_required 必须为 true。

最终只返回一个合法 JSON 根对象。
""".strip()

OUTPUT_TEMPLATE: dict[str, Any] = {
    "research_topic_id": "",
    "category_code": "",
    "category_name": "",
    "analysis_unit": "cluster",
    "source_query_key": "",
    "cluster_id": 0,
    "topic_name": "",
    "topic_type": "uncertain_opportunity",
    "external_research_status": "completed",
    "external_findings": [
        {
            "claim": "该公开来源直接支持的市场事实",
            "source_title": "来源页面的真实标题",
            "source_type": "brand_product_page",
            "source_url": "https://example.com/source-page",
            "source_date": None,
        }
    ],
    "trend_hypothesis": "",
    "hypothesis_limitations": "",
    "commercial_opportunity": {
        "recommendation": "continue_validation",
        "opportunity_type": "insufficient_support",
        "proposed_client_facing_label": None,
        "recommendation_reason": "",
        "existing_taxonomy_check": {
            "status": "taxonomy_evidence_unavailable",
            "matched_attribute": None,
            "matched_existing_label": None,
            "check_basis": "",
        },
        "client_value": [],
        "evidence_strength": "moderate",
        "market_breadth": "unknown",
        "summary": "",
    },
    "risks_and_limitations": [],
    "review_required": True,
}


def _today() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


def build_external_topic_prompt(topic: dict[str, Any]) -> str:
    required = {
        "research_topic_id",
        "category_code",
        "category_name",
        "analysis_unit",
        "source_query_key",
        "cluster_id",
        "topic_name",
        "topic_type",
        "cluster_evidence",
        "internal_insight",
        "ai_research_recommended",
        "attribute_name",
        "existing_label",
        "proposed_new_attribute",
        "proposed_new_label",
    }
    missing = required - set(topic)
    if missing:
        raise ValueError(f"topic 缺少字段：{sorted(missing)}")
    if not str(topic.get("category_code") or "").strip():
        raise ValueError("topic.category_code不能为空")
    if not str(topic.get("category_name") or "").strip():
        raise ValueError("topic.category_name不能为空")

    payload = {
        "today": _today(),
        "research_topic": topic,
        "required_output_template": OUTPUT_TEMPLATE,
        "output_instruction": (
            "请联网研究 research_topic 指定的单一主题，并严格按模板返回一个 JSON 根对象。"
            "external_findings 只能包含带有效公开网页 URL 的外部证据；内部 Taxonomy 核对只写入 existing_taxonomy_check。"
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


EXTERNAL_SYSTEM_PROMPT = EXTERNAL_TOPIC_SYSTEM_PROMPT
