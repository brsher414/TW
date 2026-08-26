"""
消息构建器 - 构建纯文本 / 多模态 messages
"""
import base64
from pathlib import Path


def build_messages(
    system_prompt: str,
    user_text: str,
    image_urls: list[str] | None = None,
    image_files: list[tuple[str, bytes]] | None = None,
) -> list[dict]:
    """构建 Responses API 的 input 消息列表

    Args:
        system_prompt: 系统提示词
        user_text: 用户文本输入
        image_urls: 图片 URL 列表（多模态）
        image_files: 上传的图片文件列表 [(filename, bytes), ...]（多模态）

    Returns:
        messages 列表
    """
    messages = []

    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    # 判断是否有图片内容
    has_images = (image_urls and len(image_urls) > 0) or (image_files and len(image_files) > 0)

    if not has_images:
        # 纯文本模式
        messages.append({"role": "user", "content": user_text})
    else:
        # 多模态模式
        content = [{"type": "input_text", "text": user_text}]

        # 添加 URL 图片
        if image_urls:
            for url in image_urls:
                url = url.strip()
                if url:
                    content.append({"type": "input_image", "image_url": url})

        # 添加上传的图片（转 base64）
        if image_files:
            for filename, file_bytes in image_files:
                ext = Path(filename).suffix.lower().lstrip(".")
                mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}
                mime = mime_map.get(ext, "jpeg")
                b64 = base64.b64encode(file_bytes).decode("utf-8")
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/{mime};base64,{b64}",
                })

        messages.append({"role": "user", "content": content})

    return messages
