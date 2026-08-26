"""Shared category and Run selector for every Streamlit page."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import streamlit as st

from core.project_context import ProjectContext
from core.ui_state import APP_CATEGORY_KEY, APP_RUN_KEY, initialise_shared_state

ROOT = Path(__file__).resolve().parent.parent


def _registry() -> dict:
    path = ROOT / "configs" / "category_registry.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _latest_run(category_root: Path) -> str | None:
    path = category_root / "latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    for key in ("run_id", "active_run_id", "latest_run_id"):
        if payload.get(key):
            return str(payload[key])
    return None


def render_workspace_selector(*, key_prefix: str = "workspace") -> ProjectContext:
    """Render shared widgets and return the selected ProjectContext.

    ``key_prefix`` is retained for backwards compatibility, but category and
    Run state are intentionally global across Dashboard, AI Insights and
    AI Research.
    """
    registry = _registry()
    entries = registry.get("categories", {})
    codes = [
        str(code).upper()
        for code, entry in entries.items()
        if isinstance(entry, dict) and entry.get("enabled", False)
    ]
    if not codes:
        raise ValueError("category_registry.toml 中没有已启用品类。")

    default = str(
        registry.get("defaults", {}).get("active_category", codes[0])
    ).upper()
    if default not in codes:
        default = codes[0]

    initialise_shared_state(default_category=default)
    current_category = str(st.session_state.get(APP_CATEGORY_KEY) or default).upper()
    if current_category not in codes:
        current_category = default
        st.session_state[APP_CATEGORY_KEY] = current_category

    st.sidebar.markdown("### 查看范围")
    category = st.sidebar.selectbox(
        "品类",
        codes,
        index=codes.index(current_category),
        key="shared_category_widget",
    )
    category = str(category).upper()
    category_changed = category != st.session_state.get(APP_CATEGORY_KEY)
    st.session_state[APP_CATEGORY_KEY] = category

    base = ProjectContext.from_category(category, project_root=ROOT)
    runs_root = base.category_root / "runs"
    run_ids = sorted(
        [path.name for path in runs_root.iterdir() if path.is_dir()],
        reverse=True,
    ) if runs_root.exists() else []
    if base.run_id not in run_ids and base.run_dir.exists():
        run_ids.insert(0, base.run_id)
    if not run_ids:
        run_ids = [base.run_id]

    preferred = _latest_run(base.category_root) or base.run_id
    remembered = None if category_changed else st.session_state.get(APP_RUN_KEY)
    selected_default = str(remembered or preferred)
    if selected_default not in run_ids:
        selected_default = preferred if preferred in run_ids else run_ids[0]

    run_id = st.sidebar.selectbox(
        "Run",
        run_ids,
        index=run_ids.index(selected_default),
        key=f"shared_run_widget_{category}",
    )
    st.session_state[APP_RUN_KEY] = str(run_id)
    context = base.with_run_id(str(run_id))
    st.sidebar.caption(
        f"{context.category_name} · "
        f"{context.period['base_quarter']} vs {context.period['current_quarter']}"
    )
    return context
