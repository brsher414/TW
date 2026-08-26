"""Append-only persistent call history shared by AI Insights and AI Research."""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

HISTORY_VERSION = "ai_call_history_v1"
_LOCK = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _hash(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _call_id(scope: str, business_key: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha1(
        f"{scope}|{business_key}|{stamp}|{uuid.uuid4().hex}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{stamp}_{digest}"


def read_call_history(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def append_call_history(
    path: str | Path,
    *,
    phase: str,
    business_key: str,
    category_code: str,
    category_name: str,
    run_id: str,
    status: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    system_prompt: str,
    user_prompt: str,
    query_key: str = "",
    research_topic_id: str = "",
    source_query_key: str = "",
    cluster_id: Any = -1,
    analysis_unit: str = "cluster",
    signature: str = "",
    evidence_hash: str = "",
    tools: list[dict[str, Any]] | None = None,
    degraded: bool = False,
    raw_output_text: str = "",
    parsed_result: dict[str, Any] | None = None,
    validation_errors: Iterable[Any] | None = None,
    validation_warnings: Iterable[Any] | None = None,
    error_code: str = "",
    error_message: str = "",
    input_tokens: Any = 0,
    output_tokens: Any = 0,
    total_tokens: Any = 0,
    elapsed_seconds: Any = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = read_call_history(target)
    attempt_no = 1 + sum(
        str(row.get("business_key") or "") == str(business_key)
        for row in previous
    )
    system_prompt = str(system_prompt or "")
    user_prompt = str(user_prompt or "")
    record = {
        "record_version": HISTORY_VERSION,
        "call_id": _call_id(phase, business_key),
        "attempt_no": attempt_no,
        "created_at_utc": _utc_now(),
        "phase": str(phase),
        "category_code": str(category_code),
        "category_name": str(category_name),
        "run_id": str(run_id),
        "business_key": str(business_key),
        "query_key": str(query_key or business_key),
        "research_topic_id": str(research_topic_id or ""),
        "source_query_key": str(source_query_key or ""),
        "cluster_id": _int(cluster_id),
        "analysis_unit": str(analysis_unit or "cluster"),
        "status": str(status),
        "model": str(model),
        "prompt_version": str(prompt_version),
        "schema_version": str(schema_version),
        "signature": str(signature),
        "evidence_hash": str(evidence_hash),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": _hash(system_prompt),
        "user_prompt_sha256": _hash(user_prompt),
        "tools": list(tools or []),
        "degraded": bool(degraded),
        "raw_output_text": str(raw_output_text or ""),
        "parsed_result": parsed_result if isinstance(parsed_result, dict) else None,
        "validation_errors": [str(x) for x in (validation_errors or [])],
        "validation_warnings": [str(x) for x in (validation_warnings or [])],
        "error_code": str(error_code or ""),
        "error_message": str(error_message or ""),
        "input_tokens": _int(input_tokens),
        "output_tokens": _int(output_tokens),
        "total_tokens": _int(total_tokens),
        "elapsed_seconds": round(_float(elapsed_seconds), 3),
        "metadata": dict(metadata or {}),
    }
    with _LOCK, target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_json_line(record) + "\n")
    return record
