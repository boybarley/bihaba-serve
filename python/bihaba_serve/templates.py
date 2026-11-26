"""
Modular chat prompt template registry (isolated from engine & HTTP server).
"""
from __future__ import annotations


def _flatten_content(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content or "")


def format_messages(messages: list[dict], template: str = "qwen35", enable_thinking: bool = False) -> tuple[str, str]:
    """
    Returns (system_prefix_str, full_prompt_str).
    `system_prefix_str` is non-empty when messages[0] is a system message so the engine
    can auto-snapshot its GPU DeltaNet + KV state for 0ms multi-user TTFT!
    """
    sys_prefix = ""
    parts: list[str] = []

    if template == "llama3":
        parts.append("<|begin_of_text|>")
        for idx, m in enumerate(messages):
            role = m.get("role", "user")
            text = _flatten_content(m.get("content", ""))
            seg = f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"
            parts.append(seg)
            if idx == 0 and role == "system":
                sys_prefix = "".join(parts)
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return sys_prefix, "".join(parts)

    # ChatML / Qwen2.5 / Qwen3 / Qwen3.5
    for idx, m in enumerate(messages):
        role = m.get("role", "user")
        text = _flatten_content(m.get("content", ""))
        seg = f"<|im_start|>{role}\n{text}<|im_end|>\n"
        parts.append(seg)
        if idx == 0 and role == "system":
            sys_prefix = "".join(parts)

    if template == "qwen35":
        if enable_thinking:
            parts.append("<|im_start|>assistant\n<think>\n")
        else:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    else:
        parts.append("<|im_start|>assistant\n")

    return sys_prefix, "".join(parts)
