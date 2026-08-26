"""
通用 API 调用器 - 调用 Qwen Responses API。

改进点：
1. 修复异常路径中 usage 可能重复累计的问题；
2. 批量结果显式携带 _input_index、query_key、cluster_id；
3. 保持批量结果顺序与输入顺序一致；
4. 保留重试、工具降级、停止控制和 Token 累计能力。
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from core.connection_pool import ConnectionPool
from core.config import MAX_RETRIES, RETRY_BASE_DELAY


class ApiCaller:
    """通用 Qwen Responses API 调用器。"""

    def __init__(self, api_key: str, base_url: str, pool_size: int = 10):
        if not api_key or not api_key.strip():
            raise ValueError("api_key 不能为空。")
        if not base_url or not base_url.strip():
            raise ValueError("base_url 不能为空。")
        if pool_size < 1:
            raise ValueError("pool_size 必须 >= 1。")

        self.pool = ConnectionPool(api_key.strip(), base_url.strip(), pool_size)
        self._stop_event = threading.Event()

    def call_single(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        enable_fallback: bool = True,
    ) -> dict[str, Any]:
        """执行单次 API 调用，支持重试和工具降级。

        Returns:
            {
                "output_text": str,
                "thinking_text": str,
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int,
                "elapsed_seconds": float,
                "degraded": bool,
                "error": str | None,
            }
        """
        if not model or not model.strip():
            raise ValueError("model 不能为空。")
        if not messages:
            raise ValueError("messages 不能为空。")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens 必须 >= 1。")

        start = time.time()
        accum_tokens = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        for attempt in range(MAX_RETRIES):
            response = None
            usage_accumulated = False

            if self._stop_event.is_set():
                return self._make_result(
                    "",
                    "",
                    accum_tokens,
                    start,
                    degraded=False,
                    error="用户中断",
                )

            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "input": messages,
                    "extra_body": {"enable_thinking": enable_thinking},
                }
                if tools:
                    kwargs["tools"] = tools
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if max_output_tokens is not None:
                    kwargs["max_output_tokens"] = max_output_tokens

                response = self._create_response(kwargs)
                self._accum_usage(accum_tokens, getattr(response, "usage", None))
                usage_accumulated = True

                has_message = self._has_message_output(response)

                if not has_message and tools and enable_fallback:
                    # 联网/工具调用未产生 message 时，去掉 tools 重试。
                    # 对外部研究任务，调用方必须通过 degraded=True 识别为“无联网降级”。
                    if not getattr(response, "output", None):
                        raise RuntimeError(
                            "API 返回空响应，请检查账户余额或 API 配置"
                        )

                    if self._stop_event.is_set():
                        return self._make_result(
                            "",
                            "",
                            accum_tokens,
                            start,
                            degraded=False,
                            error="用户中断",
                        )

                    fallback_kwargs: dict[str, Any] = {
                        "model": model,
                        "input": messages,
                        "extra_body": {"enable_thinking": enable_thinking},
                    }
                    if temperature is not None:
                        fallback_kwargs["temperature"] = temperature
                    if max_output_tokens is not None:
                        fallback_kwargs["max_output_tokens"] = max_output_tokens

                    fallback_response = self._create_response(fallback_kwargs)
                    self._accum_usage(
                        accum_tokens,
                        getattr(fallback_response, "usage", None),
                    )

                    output_text, thinking_text = self._extract_output(
                        fallback_response
                    )
                    if not output_text.strip():
                        raise RuntimeError(
                            "原始调用和降级调用均未返回有效消息，"
                            "请检查 API 可用性或账户余额"
                        )

                    return self._make_result(
                        output_text,
                        thinking_text,
                        accum_tokens,
                        start,
                        degraded=True,
                    )

                output_text, thinking_text = self._extract_output(response)
                if not output_text.strip():
                    raise RuntimeError(
                        "API 未返回有效文本输出，请检查 API 可用性"
                    )

                return self._make_result(
                    output_text,
                    thinking_text,
                    accum_tokens,
                    start,
                    degraded=False,
                )

            except Exception as exc:
                # 仅当当前 response 的 usage 尚未累计时才补记，避免重复累计。
                if (
                    response is not None
                    and not usage_accumulated
                    and getattr(response, "usage", None) is not None
                ):
                    self._accum_usage(accum_tokens, response.usage)

                if self._stop_event.is_set():
                    return self._make_result(
                        "",
                        "",
                        accum_tokens,
                        start,
                        degraded=False,
                        error="用户中断",
                    )

                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    if self._stop_event.wait(delay):
                        return self._make_result(
                            "",
                            "",
                            accum_tokens,
                            start,
                            degraded=False,
                            error="用户中断",
                        )
                else:
                    return self._make_result(
                        "",
                        "",
                        accum_tokens,
                        start,
                        degraded=False,
                        error=(
                            f"API调用失败({MAX_RETRIES}次重试): "
                            f"{str(exc)[:300]}"
                        ),
                    )

        # 理论上不会执行到这里，保留防御性返回。
        return self._make_result(
            "",
            "",
            accum_tokens,
            start,
            degraded=False,
            error="API调用未产生结果",
        )

    def call_batch(
        self,
        items: list[dict[str, Any]],
        model: str,
        system_prompt: str,
        text_column: str,
        image_column: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        enable_fallback: bool = True,
        max_workers: int = 10,
        progress_callback: Callable[
            [int, int, str, dict[str, Any]], None
        ]
        | None = None,
    ) -> list[dict[str, Any]]:
        """批量并发调用，并保证返回结果顺序与输入顺序一致。

        每个结果会额外携带：
        - _input_index：原始输入索引；
        - query_key：若输入行包含该字段；
        - cluster_id：若输入行包含该字段。
        """
        from core.message_builder import build_messages

        if max_workers < 1:
            raise ValueError("max_workers 必须 >= 1。")
        if not text_column:
            raise ValueError("text_column 不能为空。")
        if not items:
            return []

        total = len(items)
        results: list[dict[str, Any] | None] = [None] * total

        def _process(idx: int, row: dict[str, Any]):
            metadata = self._build_result_metadata(idx, row)

            if self._stop_event.is_set():
                stopped_result = self._empty_result(error="用户中断")
                stopped_result.update(metadata)
                return idx, stopped_result

            user_text = str(row.get(text_column, ""))
            image_urls = None
            if image_column and image_column in row:
                raw = str(row.get(image_column, ""))
                if raw.strip() and raw.strip().lower() != "nan":
                    image_urls = [
                        url.strip()
                        for url in raw.split(":::")
                        if url.strip()
                    ]

            messages = build_messages(
                system_prompt,
                user_text,
                image_urls=image_urls,
            )
            result = self.call_single(
                messages=messages,
                model=model,
                tools=tools,
                enable_thinking=enable_thinking,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                enable_fallback=enable_fallback,
            )
            result.update(metadata)
            return idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process, idx, row): idx
                for idx, row in enumerate(items)
            }
            completed = 0

            for future in as_completed(futures):
                fallback_idx = futures[future]
                try:
                    idx, result = future.result()
                except Exception as exc:
                    # 防止单个 Future 的意外异常终止整个批次。
                    idx = fallback_idx
                    result = self._empty_result(
                        error=f"批处理线程异常: {str(exc)[:300]}"
                    )
                    result.update(
                        self._build_result_metadata(idx, items[idx])
                    )

                results[idx] = result
                completed += 1

                item_text = str(items[idx].get(text_column, ""))
                if progress_callback:
                    try:
                        progress_callback(
                            completed,
                            total,
                            item_text,
                            result,
                        )
                    except Exception as exc:
                        # Preserve the completed API result and expose callback failures.
                        # The page can then report the real cache/schema callback error.
                        result["_progress_callback_error"] = (
                            f"{type(exc).__name__}: {str(exc)[:500]}"
                        )

        # 防御性补齐；正常情况下不存在 None。
        final_results: list[dict[str, Any]] = []
        for idx, result in enumerate(results):
            if result is None:
                result = self._empty_result(error="任务未返回结果")
                result.update(self._build_result_metadata(idx, items[idx]))
            final_results.append(result)

        return final_results

    def _create_response(self, kwargs: dict[str, Any]):
        """从连接池借用 client 执行请求，并确保归还。"""
        client = self.pool.get_connection()
        try:
            return client.responses.create(**kwargs)
        finally:
            self.pool.return_connection(client)

    @staticmethod
    def _has_message_output(response) -> bool:
        return any(
            getattr(item, "type", None) == "message"
            for item in (getattr(response, "output", None) or [])
        )

    @staticmethod
    def _extract_output(response) -> tuple[str, str]:
        """从 response.output 提取最终文本和思考内容。"""
        texts: list[str] = []
        thinking_texts: list[str] = []

        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) != "message":
                continue

            for content in getattr(item, "content", None) or []:
                content_type = getattr(content, "type", None)
                if content_type == "output_text":
                    text = getattr(content, "text", None)
                    if text:
                        texts.append(text)
                elif content_type == "thinking":
                    thinking = getattr(content, "thinking", None)
                    if thinking:
                        thinking_texts.append(thinking)

        return "\n".join(texts), "\n".join(thinking_texts)

    @staticmethod
    def _accum_usage(accum: dict[str, int], usage) -> None:
        if not usage:
            return
        accum["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        accum["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        accum["total_tokens"] += getattr(usage, "total_tokens", 0) or 0

    @staticmethod
    def _make_result(
        output_text: str,
        thinking_text: str,
        tokens: dict[str, int],
        start_time: float,
        degraded: bool,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "output_text": output_text,
            "thinking_text": thinking_text,
            "input_tokens": int(tokens.get("input_tokens", 0)),
            "output_tokens": int(tokens.get("output_tokens", 0)),
            "total_tokens": int(tokens.get("total_tokens", 0)),
            "elapsed_seconds": round(time.time() - start_time, 1),
            "degraded": bool(degraded),
            "error": error,
        }

    @staticmethod
    def _empty_result(error: str | None = None) -> dict[str, Any]:
        return {
            "output_text": "",
            "thinking_text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "elapsed_seconds": 0.0,
            "degraded": False,
            "error": error,
        }

    @staticmethod
    def _build_result_metadata(
        idx: int,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {"_input_index": idx}
        if "query_key" in row:
            metadata["query_key"] = row.get("query_key")
        if "cluster_id" in row:
            metadata["cluster_id"] = row.get("cluster_id")
        return metadata

    def stop(self) -> None:
        """请求停止。已发出的 HTTP 请求可能仍会自然完成。"""
        self._stop_event.set()

    def reset(self) -> None:
        self._stop_event.clear()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()
