"""Latest-only JSONL cache compaction for TRENDING_WORDS.

Internal unique entity: phase=internal + cluster_id
External unique entity: phase=external + query_key (research_topic_id)

The newest successful/current record physically replaces older records for the same
entity. The rewrite is atomic. Error JSONL files are not touched.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _entity_key(record: dict[str, Any]) -> tuple[str, str] | None:
    phase = str(record.get("phase") or "").strip().lower()
    if phase == "internal":
        cluster_id = record.get("cluster_id")
        return ("internal", str(cluster_id)) if cluster_id is not None else None
    if phase == "external":
        query_key = record.get("query_key")
        if query_key:
            return ("external", str(query_key))
        cluster_id = record.get("cluster_id")
        return ("external", f"cluster:{cluster_id}") if cluster_id is not None else None
    return None


def _is_current_candidate(record: dict[str, Any]) -> bool:
    """Keep records that can represent current state.

    Successful records and review updates are candidates. Validation/API failures
    normally live in a separate error file and are never used to overwrite success.
    """
    if record.get("schema_valid") is True:
        return True
    record_type = str(record.get("record_type") or "").lower()
    return record_type in {"success", "review_update", "review"}


def compact_latest_records(
    cache_path: str | Path,
    *,
    phases: Iterable[str] = ("internal", "external"),
) -> dict[str, int]:
    """Atomically keep only the latest current record per business entity.

    Records without a recognized entity key are preserved. If the file does not
    exist or is empty, the function returns zero counts.
    """
    path = Path(cache_path)
    if not path.exists() or path.stat().st_size == 0:
        return {"before": 0, "after": 0, "removed": 0}

    allowed_phases = {str(item).lower() for item in phases}
    parsed_rows: list[dict[str, Any]] = []
    passthrough_lines: list[str] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                # Preserve malformed lines instead of silently deleting data.
                passthrough_lines.append(raw.rstrip("\r\n"))
                continue
            if isinstance(value, dict):
                parsed_rows.append(value)
            else:
                passthrough_lines.append(raw.rstrip("\r\n"))

    latest_index: dict[tuple[str, str], int] = {}
    keep_flags = [True] * len(parsed_rows)

    for index, record in enumerate(parsed_rows):
        key = _entity_key(record)
        if key is None or key[0] not in allowed_phases:
            continue
        if not _is_current_candidate(record):
            # A failure must never replace a previous successful result.
            keep_flags[index] = False
            continue
        previous = latest_index.get(key)
        if previous is not None:
            keep_flags[previous] = False
        latest_index[key] = index

    kept_rows = [row for index, row in enumerate(parsed_rows) if keep_flags[index]]
    output_lines = [
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        for row in kept_rows
    ]
    output_lines.extend(passthrough_lines)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            if output_lines:
                handle.write("\n".join(output_lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

    before = len(parsed_rows) + len(passthrough_lines)
    after = len(output_lines)
    return {"before": before, "after": after, "removed": before - after}


def find_cache_path(cache: Any) -> Path | None:
    """Best-effort discovery of ClusterCache's JSONL path."""
    candidates = (
        "cache_path", "_cache_path", "path", "_path", "jsonl_path",
        "_jsonl_path", "file_path", "_file_path",
    )
    for name in candidates:
        value = getattr(cache, name, None)
        if isinstance(value, (str, Path)):
            return Path(value)
    return None


def compact_cache_instance(cache: Any) -> dict[str, int] | None:
    path = find_cache_path(cache)
    if path is None:
        return None
    result = compact_latest_records(path)
    reload_method = getattr(cache, "reload", None)
    if callable(reload_method):
        reload_method()
    return result
