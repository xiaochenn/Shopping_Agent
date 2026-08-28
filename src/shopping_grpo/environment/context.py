"""评测和 GRPO 共用的上下文窗口管理。

历史由完整的 assistant tool-call + tool observation 组组成；超长时只删除较旧的
完整组，保留固定 prompt 和最近的环境反馈，避免留下半截工具调用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from urllib.request import Request, urlopen


class ContextBudgetError(RuntimeError):
    """The required prompt and latest environment state do not fit the configured budget."""


RESULT_CLEARING_VERSION = "shopping-result-clearing-v1"
_OBSERVATION_FIELD_PATTERN = re.compile(r"^([a-z_]+):\s*(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class ContextWindowStats:
    original_tokens: int
    final_tokens: int
    removed_tokens: int
    removed_groups: int
    removed_messages: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ToolResultClearingStats:
    """Audit record for deterministic historical tool-result clearing."""

    cleared_tool_results: int
    kept_recent_groups: int

    def to_dict(self) -> dict[str, int | str]:
        return {"version": RESULT_CLEARING_VERSION, **asdict(self)}


def tool_result_placeholder(tool_name, observation):
    """Return a compact, non-actionable memory of an old tool result.

    The original assistant tool call stays in history.  The replacement is a
    valid tool response rather than an arbitrary token splice, and only keeps
    stable facts that can help comparison later.  In particular it never
    reproduces historic buttons or search-result ASIN lists, because neither is
    a legal action target outside the latest observation.
    """

    fields = _observation_fields(observation)
    lines = [
        "[SHOPPING_TOOL_RESULT_CLEARED_V1]",
        f"tool: {tool_name or 'unknown'}",
        "status: historical result cleared to satisfy the context budget.",
        "The original tool call is retained. This record is historical and has no actionable buttons.",
    ]
    for key in ("query", "asin", "title", "brand", "category", "price", "selected_options", "key_attributes"):
        value = fields.get(key)
        if value:
            lines.append(f"{key}: {_short_field(value)}")
    return "\n".join(lines)


def clear_old_tool_results(messages, keep_recent_groups=3):
    """Replace old tool observations while preserving every assistant call.

    This is intentionally milder than dropping old interaction groups.  It is
    usable by message-based API rollouts and SFT replay.  Token-based veRL
    rollouts use the same ``tool_result_placeholder`` payload after locating a
    complete tool-message span.
    """

    keep_recent_groups = int(keep_recent_groups)
    if keep_recent_groups < 1:
        raise ValueError("keep_recent_groups must be positive")
    original = [dict(message) for message in messages]
    anchor, groups = _split_chat_tool_groups(original)
    if len(groups) <= keep_recent_groups:
        return original, ToolResultClearingStats(0, len(groups))

    cleared = 0
    rewritten = []
    for group_index, group in enumerate(groups):
        copied_group = []
        should_clear = group_index < len(groups) - keep_recent_groups
        for message in group:
            copied = dict(message)
            if should_clear and copied.get("role") == "tool":
                copied["content"] = tool_result_placeholder(
                    copied.get("name") or _tool_name_from_group(group),
                    copied.get("content"),
                )
                cleared += 1
            copied_group.append(copied)
        rewritten.append(copied_group)
    return anchor + _flatten(rewritten), ToolResultClearingStats(
        cleared, min(keep_recent_groups, len(groups))
    )


def _tool_name_from_group(group):
    assistant = group[0] if group else {}
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1:
        return "unknown"
    return str(((calls[0].get("function") or {}).get("name")) or "unknown")


def _observation_fields(observation):
    if not isinstance(observation, str):
        return {}
    return {
        key: value.strip()
        for key, value in _OBSERVATION_FIELD_PATTERN.findall(observation)
        if value.strip()
    }


def _short_field(value, limit=240):
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def compact_chat_messages(messages, tools, count_tokens, max_input_tokens):
    """保留固定 prompt，并尽量保留能放入预算的最新完整工具调用组。"""
    if int(max_input_tokens) < 1:
        raise ValueError("max_input_tokens must be positive")
    original = [dict(message) for message in messages]
    original_tokens = int(count_tokens(original, tools))
    if original_tokens <= int(max_input_tokens):
        return original, ContextWindowStats(
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            removed_tokens=0,
            removed_groups=0,
            removed_messages=0,
        )

    # anchor 是 system/user 等固定提示；groups 才是可以整体删除的交互历史。
    anchor, groups = _split_chat_tool_groups(original)
    if not groups:
        raise ContextBudgetError(
            f"required prompt uses {original_tokens} tokens, above input budget {max_input_tokens}"
        )

    latest_only = anchor + groups[-1]
    latest_tokens = int(count_tokens(latest_only, tools))
    if latest_tokens > int(max_input_tokens):
        raise ContextBudgetError(
            "fixed prompt plus latest tool observation uses "
            f"{latest_tokens} tokens, above input budget {max_input_tokens}"
        )

    # 每次只保留消息后缀，二分查找预算内能保留的最大历史长度。
    low, high, kept_groups = 1, len(groups), 1
    kept_tokens = latest_tokens
    while low <= high:
        middle = (low + high) // 2
        candidate = anchor + _flatten(groups[-middle:])
        candidate_tokens = int(count_tokens(candidate, tools))
        if candidate_tokens <= int(max_input_tokens):
            kept_groups = middle
            kept_tokens = candidate_tokens
            low = middle + 1
        else:
            high = middle - 1

    compacted = anchor + _flatten(groups[-kept_groups:])
    removed = groups[:-kept_groups]
    return compacted, ContextWindowStats(
        original_tokens=original_tokens,
        final_tokens=kept_tokens,
        removed_tokens=original_tokens - kept_tokens,
        removed_groups=len(removed),
        removed_messages=sum(len(group) for group in removed),
    )


def compact_token_trajectory(
    prompt_ids,
    response_mask,
    response_logprobs,
    max_input_tokens,
    preserve_recent_groups=1,
):
    """删除旧的完整 assistant/tool token 组，同时保持训练数组对齐。"""
    prompt_ids = list(prompt_ids)
    response_mask = list(response_mask)
    response_logprobs = list(response_logprobs)
    max_input_tokens = int(max_input_tokens)
    preserve_recent_groups = int(preserve_recent_groups)
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive")
    if preserve_recent_groups < 1:
        raise ValueError("preserve_recent_groups must be positive")
    if len(prompt_ids) < len(response_mask):
        raise ValueError("response_mask cannot be longer than prompt_ids")
    if response_logprobs and len(response_logprobs) != len(response_mask):
        raise ValueError("response_logprobs must be empty or aligned with response_mask")

    original_tokens = len(prompt_ids)
    if original_tokens <= max_input_tokens:
        return prompt_ids, response_mask, response_logprobs, ContextWindowStats(
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            removed_tokens=0,
            removed_groups=0,
            removed_messages=0,
        )

    # response_mask 标记 assistant 输出，前面的 token 是不能单独删除的固定 prompt。
    initial_prompt_length = len(prompt_ids) - len(response_mask)
    group_ends = _complete_token_group_ends(response_mask)
    removable_group_ends = group_ends[:-preserve_recent_groups]
    required_removal = original_tokens - max_input_tokens
    drop_count = next(
        (end for end in removable_group_ends if end >= required_removal),
        None,
    )
    if drop_count is None:
        required_tokens = original_tokens - (removable_group_ends[-1] if removable_group_ends else 0)
        raise ContextBudgetError(
            "fixed prompt plus protected recent tool group uses "
            f"{required_tokens} tokens, above input budget {max_input_tokens}"
        )

    # 同步裁剪 prompt、response mask 和 logprobs；任意一个数组错位都会破坏训练。
    compacted_prompt = prompt_ids[:initial_prompt_length] + prompt_ids[
        initial_prompt_length + drop_count :
    ]
    compacted_mask = response_mask[drop_count:]
    compacted_logprobs = response_logprobs[drop_count:] if response_logprobs else []
    removed_groups = sum(end <= drop_count for end in group_ends)
    final_tokens = len(compacted_prompt)
    return compacted_prompt, compacted_mask, compacted_logprobs, ContextWindowStats(
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        removed_tokens=drop_count,
        removed_groups=removed_groups,
        removed_messages=removed_groups * 2,
    )


class VllmChatTokenCounter:
    """Count the exact rendered chat tokens through the serving vLLM tokenizer."""

    def __init__(self, model, base_url, api_key, timeout=60, transport=None):
        self.model = model
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        self.url = f"{base_url}/tokenize"
        self.api_key = api_key
        self.timeout = int(timeout)
        self.transport = transport

    def __call__(self, messages, tools):
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "shopping-grpo-longhorizon/0.1",
        }
        if self.transport is not None:
            response = self.transport(self.url, payload, headers, self.timeout)
        else:
            request = Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=self.timeout) as raw:
                response = json.loads(raw.read().decode("utf-8"))
        count = response.get("count")
        if not isinstance(count, int) or count < 0:
            raise ValueError("vLLM /tokenize response is missing a non-negative integer count")
        return count


class VllmTextTokenCounter:
    """Count plain text with the tokenizer used by the serving vLLM model."""

    def __init__(self, model, base_url, api_key, timeout=60, transport=None):
        self.model = model
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        self.url = f"{base_url}/tokenize"
        self.api_key = api_key
        self.timeout = int(timeout)
        self.transport = transport

    def __call__(self, text):
        payload = {"model": self.model, "prompt": str(text), "add_special_tokens": False}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "shopping-grpo-longhorizon/0.1",
        }
        if self.transport is not None:
            response = self.transport(self.url, payload, headers, self.timeout)
        else:
            request = Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=self.timeout) as raw:
                response = json.loads(raw.read().decode("utf-8"))
        count = response.get("count")
        if not isinstance(count, int) or count < 0:
            raise ValueError("vLLM /tokenize response is missing a non-negative integer count")
        return count


def _split_chat_tool_groups(messages):
    first_assistant = next(
        (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
        len(messages),
    )
    anchor = messages[:first_assistant]
    history = messages[first_assistant:]
    groups = []
    index = 0
    while index < len(history):
        assistant = history[index]
        if assistant.get("role") != "assistant" or not assistant.get("tool_calls"):
            raise ContextBudgetError("chat history is not a sequence of complete assistant/tool groups")
        call_ids = {
            call.get("id")
            for call in assistant.get("tool_calls") or []
            if call.get("id") is not None
        }
        group = [assistant]
        index += 1
        while index < len(history) and history[index].get("role") == "tool":
            tool_message = history[index]
            if tool_message.get("tool_call_id") not in call_ids:
                raise ContextBudgetError("tool message does not match its preceding assistant tool call")
            group.append(tool_message)
            index += 1
        if len(group) == 1:
            raise ContextBudgetError("assistant tool call is missing its tool response")
        groups.append(group)
    return anchor, groups


def _flatten(groups):
    return [message for group in groups for message in group]


def _complete_token_group_ends(response_mask):
    """Return end offsets for every completed 1-valued assistant + 0-valued tool pair."""
    group_ends = []
    index = 0
    while index < len(response_mask):
        if response_mask[index] != 1:
            raise ContextBudgetError("token trajectory does not start with an assistant response")
        while index < len(response_mask) and response_mask[index] == 1:
            index += 1
        if index >= len(response_mask) or response_mask[index] != 0:
            break
        while index < len(response_mask) and response_mask[index] == 0:
            index += 1
        group_ends.append(index)
    return group_ends
