from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from core.project_context import ProjectContext

STAGES = ("etl", "trend", "cluster", "taxonomy", "insights", "research", "dashboard_cache")
def _now() -> str: return datetime.now(timezone.utc).isoformat(timespec="seconds")
def _write(context: ProjectContext, manifest: dict[str, Any]) -> None:
    context.ensure_directories(); manifest["updated_at_utc"] = _now()
    context.manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
def create_manifest(context: ProjectContext, overwrite: bool = False) -> dict[str, Any]:
    if context.manifest_file.exists() and not overwrite:
        return json.loads(context.manifest_file.read_text(encoding="utf-8"))
    manifest = {"schema_version":"1.0","category_code":context.category_code,"category_name":context.category_name,"run_id":context.run_id,"created_at_utc":_now(),"updated_at_utc":_now(),"status":{stage:"not_started" for stage in STAGES},"artifacts":{}}
    _write(context, manifest); return manifest
def update_stage(context: ProjectContext, stage: str, status: str, *, artifacts: dict[str, Path | str] | None = None) -> dict[str, Any]:
    manifest = create_manifest(context); manifest["status"][stage] = status
    for key, value in (artifacts or {}).items():
        path = Path(value)
        try: stored = str(path.resolve().relative_to(context.run_dir.resolve()))
        except ValueError: stored = str(path)
        manifest["artifacts"][key] = stored
    _write(context, manifest); return manifest
def set_active_run(context: ProjectContext) -> None:
    context.category_root.mkdir(parents=True, exist_ok=True)
    context.latest_file.write_text(json.dumps({"category_code":context.category_code,"active_run_id":context.run_id,"updated_at_utc":_now()}, ensure_ascii=False, indent=2), encoding="utf-8")
