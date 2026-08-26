"""Safe JSON parsing fallback for LLM output.

Parsing order:
1. Standard json.loads on the full text.
2. Standard json.loads after removing a Markdown code fence.
3. Standard raw_decode candidates containing required root identity fields.
4. json_repair.loads fallback, only when standard parsing failed.

Repair never bypasses the external-topic business schema. The caller must still
validate topic identity, enums, URLs, field types, and review requirements.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

try:
    import json_repair
except ImportError:  # pragma: no cover - handled explicitly at runtime
    json_repair = None


DEFAULT_ROOT_IDENTITY_FIELDS = {
    "research_topic_id",
    "cluster_id",
    "topic_type",
    "external_research_status",
}


@dataclass(slots=True)
class RepairedJsonResult:
    value: dict[str, Any] | None
    warnings: list[str]
    error: str | None
    repaired: bool


def strip_markdown_fence(text: str) -> str:
    clean = text.strip().lstrip("\ufeff")
    if not clean.startswith("```"):
        return clean

    lines = clean.splitlines()
    if not lines:
        return clean

    first = lines[0].strip().lower()
    if first in {"```", "```json", "```javascript", "```js"}:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _is_root_object(
    value: Any,
    required_fields: set[str],
) -> bool:
    return isinstance(value, dict) and required_fields.issubset(value.keys())


def _standard_candidates(text: str) -> list[dict[str, Any]]:
    clean = text.strip().lstrip("\ufeff")
    unfenced = strip_markdown_fence(clean)
    output: list[dict[str, Any]] = []

    for candidate in dict.fromkeys((clean, unfenced)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            output.append(value)

    decoder = json.JSONDecoder()
    for candidate in dict.fromkeys((clean, unfenced)):
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                output.append(value)
    return output


def parse_llm_json_with_repair(
    text: str,
    *,
    required_root_fields: Iterable[str] = DEFAULT_ROOT_IDENTITY_FIELDS,
) -> RepairedJsonResult:
    if not isinstance(text, str) or not text.strip():
        return RepairedJsonResult(None, [], "EMPTY_OUTPUT", False)

    required = set(required_root_fields)
    standard = [
        value
        for value in _standard_candidates(text)
        if _is_root_object(value, required)
    ]
    if standard:
        return RepairedJsonResult(
            max(standard, key=len),
            [],
            None,
            False,
        )

    if json_repair is None:
        return RepairedJsonResult(
            None,
            [],
            "JSON_PARSE_ERROR: 标准解析失败，且未安装 json-repair。",
            False,
        )

    clean = strip_markdown_fence(text)
    attempts = [clean]
    # If explanatory prose surrounds the JSON, give the repair library the broadest
    # likely object slice as a second attempt.
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        attempts.append(clean[first_brace:last_brace + 1])

    repair_errors: list[str] = []
    for candidate in dict.fromkeys(attempts):
        try:
            value = json_repair.loads(candidate)
        except Exception as exc:  # library-specific parse failures
            repair_errors.append(f"{type(exc).__name__}: {exc}")
            continue

        if _is_root_object(value, required):
            return RepairedJsonResult(
                value,
                ["JSON_REPAIRED_FROM_MALFORMED_OUTPUT"],
                None,
                True,
            )

        repair_errors.append("修复结果不是包含完整主题身份字段的根对象")

    detail = " | ".join(repair_errors[-2:]) if repair_errors else "未知修复错误"
    return RepairedJsonResult(
        None,
        [],
        f"JSON_REPAIR_FAILED: {detail}",
        True,
    )
