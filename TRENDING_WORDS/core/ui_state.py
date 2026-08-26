"""Reliable shared Streamlit state for Workspace and AI configuration.

Permanent ``app_*`` keys are shared across pages. Each AI page receives a
separate temporary widget namespace, while all changes are persisted to the
same permanent values.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterable, Sequence

import streamlit as st

APP_CATEGORY_KEY = "app_category_code"
APP_RUN_KEY = "app_run_id"
APP_API_KEY = "app_api_key"
APP_MODEL_KEY = "app_model_name"
APP_BASE_URL_KEY = "app_base_url"
APP_WORKERS_KEY = "app_max_workers"
APP_THINKING_KEY = "app_enable_thinking"

DEFAULT_MAX_WORKERS = 15
MAX_WORKERS_LIMIT = 20


def _secret(path: tuple[str, ...], default: Any = "") -> Any:
    try:
        value: Any = st.secrets
        for name in path:
            value = value[name]
        return value
    except Exception:
        return default


def _workers(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_WORKERS
    return min(MAX_WORKERS_LIMIT, max(1, parsed))


def valid_choice(value: str, choices: Iterable[str], fallback: str) -> str:
    options = [str(item) for item in choices]
    return str(value) if str(value) in options else str(fallback)


def _namespace(key_prefix: str | None, heading: str) -> str:
    raw = str(key_prefix or heading or "ai").strip().casefold()
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return cleaned or "ai"


def initialise_shared_state(
    *,
    default_category: str | None = None,
    default_model: str = "",
    default_base_url: str = "",
    thinking_default: bool = True,
) -> None:
    if APP_CATEGORY_KEY not in st.session_state and default_category:
        st.session_state[APP_CATEGORY_KEY] = str(default_category).upper()

    secret_key = str(
        _secret(("qwen", "api_key"), os.getenv("QWEN_API_KEY", "")) or ""
    )
    secret_model = str(
        _secret(("defaults", "model"), default_model) or default_model or ""
    )
    secret_url = str(
        _secret(("qwen", "base_url"), default_base_url)
        or default_base_url
        or ""
    )

    st.session_state.setdefault(APP_API_KEY, secret_key)
    st.session_state.setdefault(APP_MODEL_KEY, secret_model)
    st.session_state.setdefault(APP_BASE_URL_KEY, secret_url)
    st.session_state.setdefault(APP_WORKERS_KEY, DEFAULT_MAX_WORKERS)
    st.session_state[APP_WORKERS_KEY] = _workers(st.session_state[APP_WORKERS_KEY])
    st.session_state[APP_THINKING_KEY] = True


def render_ai_sidebar(
    *,
    model_options: Sequence[str],
    base_url_options: Sequence[str],
    default_model: str,
    default_base_url: str,
    thinking_default: bool = True,
    heading: str = "AI 配置",
    key_prefix: str | None = None,
) -> dict[str, Any]:
    """Render a cross-page AI configuration panel.

    ``key_prefix`` is optional for backward compatibility. Existing pages are
    separated by their distinct headings; callers may explicitly pass
    ``insights`` or ``research`` for maximum clarity.
    """
    initialise_shared_state(
        default_model=default_model,
        default_base_url=default_base_url,
        thinking_default=True,
    )

    models = list(dict.fromkeys(
        [str(item) for item in model_options if str(item).strip()]
        + [str(default_model)]
    ))
    urls = list(dict.fromkeys(
        [str(item) for item in base_url_options if str(item).strip()]
        + [str(default_base_url)]
    ))
    if not models:
        raise ValueError("模型列表为空。")
    if not urls:
        raise ValueError("Base URL 列表为空。")

    ns = _namespace(key_prefix, heading)
    keys = {
        "api": f"_{ns}_ai_api_key",
        "model": f"_{ns}_ai_model",
        "url": f"_{ns}_ai_base_url",
        "workers": f"_{ns}_ai_workers",
    }

    current_model = valid_choice(
        str(st.session_state.get(APP_MODEL_KEY) or default_model),
        models,
        models[0],
    )
    current_url = valid_choice(
        str(st.session_state.get(APP_BASE_URL_KEY) or default_base_url),
        urls,
        urls[0],
    )

    # Restore this page's temporary widgets from permanent state. Explicit
    # assignment avoids stale empty password-widget state after navigation.
    st.session_state[keys["api"]] = str(st.session_state.get(APP_API_KEY) or "")
    st.session_state[keys["model"]] = current_model
    st.session_state[keys["url"]] = current_url
    st.session_state[keys["workers"]] = _workers(
        st.session_state.get(APP_WORKERS_KEY, DEFAULT_MAX_WORKERS)
    )

    def persist() -> None:
        api_value = str(st.session_state.get(keys["api"], "") or "")
        # Never let Streamlit's page cleanup erase a non-empty permanent key.
        if api_value or not str(st.session_state.get(APP_API_KEY) or ""):
            st.session_state[APP_API_KEY] = api_value
        st.session_state[APP_MODEL_KEY] = str(
            st.session_state.get(keys["model"], current_model) or current_model
        )
        st.session_state[APP_BASE_URL_KEY] = str(
            st.session_state.get(keys["url"], current_url) or current_url
        )
        st.session_state[APP_WORKERS_KEY] = _workers(
            st.session_state.get(keys["workers"], DEFAULT_MAX_WORKERS)
        )
        st.session_state[APP_THINKING_KEY] = True
        st.session_state["api_key"] = st.session_state[APP_API_KEY]
        st.session_state["model_name"] = st.session_state[APP_MODEL_KEY]
        st.session_state["base_url"] = st.session_state[APP_BASE_URL_KEY]
        st.session_state["max_workers"] = st.session_state[APP_WORKERS_KEY]

    st.sidebar.markdown(f"### {heading}")
    model = st.sidebar.selectbox(
        "模型", models, key=keys["model"], on_change=persist
    )
    with st.sidebar.expander("模型与调用设置", expanded=False):
        api_key = st.text_input(
            "API Key", type="password", key=keys["api"], on_change=persist
        )
        base_url = st.selectbox(
            "Base URL", urls, key=keys["url"], on_change=persist
        )
        workers = st.slider(
            "并发数",
            min_value=1,
            max_value=MAX_WORKERS_LIMIT,
            key=keys["workers"],
            on_change=persist,
        )

    persist()
    return {
        "api_key": str(api_key or st.session_state.get(APP_API_KEY) or ""),
        "model": str(model),
        "base_url": str(base_url),
        "workers": _workers(workers),
        "thinking": True,
    }


# Compatibility aliases.
def init_shared_ai_state(
    *, default_model: str = "", default_base_url: str = ""
) -> None:
    initialise_shared_state(
        default_model=default_model,
        default_base_url=default_base_url,
    )


def capture_shared_ai_state() -> None:
    # Legacy pages are retained, but new pages should use render_ai_sidebar.
    api = str(st.session_state.get("api_key", "") or "")
    if api or not str(st.session_state.get(APP_API_KEY) or ""):
        st.session_state[APP_API_KEY] = api
    if st.session_state.get("model_name"):
        st.session_state[APP_MODEL_KEY] = st.session_state["model_name"]
    if st.session_state.get("base_url"):
        st.session_state[APP_BASE_URL_KEY] = st.session_state["base_url"]
    if st.session_state.get("max_workers") is not None:
        st.session_state[APP_WORKERS_KEY] = _workers(
            st.session_state["max_workers"]
        )
    st.session_state[APP_THINKING_KEY] = True


def sync_api_state() -> None:
    capture_shared_ai_state()
