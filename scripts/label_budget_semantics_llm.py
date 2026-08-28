#!/usr/bin/env python3
"""Label the Tier-1 budget queue with an OpenAI-compatible DeepSeek endpoint.

The script is resumable: completed task IDs in --output are skipped.  It reads
OPENCODE_URL/OPENCODE_API_KEY or DEEPSEEK_URL/DEEPSEEK_API_KEY from the
repository .env file unless those variables are already set in the process
environment.  Secrets are never written to output files or logs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scripts.generate_budget_semantics import (
    APPROX_ABSOLUTE_TOLERANCE_FLOOR,
    APPROX_LOWER_TOLERANCE,
    APPROX_UPPER_TOLERANCE,
    SCHEMA_VERSION,
    _approximate_band,
    _approximate_range,
    _normalise,
)


MODEL_DEFAULT = "deepseek-v4-flash"
VALID_TYPES = {
    "hard_upper",
    "approximate_band",
    "range",
    "approximate_range",
    "lower_bound",
    "price_preference",
    "no_explicit_budget",
    "unknown",
}
SYSTEM_PROMPT = """你是中文购物任务中的预算语义标注器。只分析用户原文中的价格要求，不要根据商品、
常识或 gold 猜测。输出严格 JSON object，不要 Markdown。

标签定义：
- hard_upper：明确最高/不超过/以内/以下的价格上限。
- approximate_band：约、大概、左右、上下、这个范围等近似价格；只填写 target。
- range：明确价格区间；填写 lower 与 upper。
- approximate_range：区间后又带“左右/上下”等近似词；填写原文陈述的 lower 与 upper，
  不要自行计算容差。
- lower_bound：起、以上等明确下限；填写 lower。
- price_preference：只说便宜、预算有限等但没有可执行金额。
- no_explicit_budget：没有价格需求。
- unknown：有价格相关信息但无法可靠判定。

字段必须为 budget_type、target、lower、upper、evidence、confidence。evidence 必须是原文中连续的
精确片段；没有证据时为 null。金额统一为人民币数值，不要自行添加 tolerance。confidence 介于 0 和 1。"""

SYSTEM_PROMPT += """

输出示例（只模仿 JSON 格式，不要解释）：
原文：预算35，有适合的吗？
{"budget_type":"hard_upper","target":null,"lower":null,"upper":35,"evidence":"预算35","confidence":0.95}
原文：价格在80-100元左右。
{"budget_type":"approximate_range","target":null,"lower":80,"upper":100,"evidence":"价格在80-100元左右","confidence":0.9}
原文：预算大概40元左右。
{"budget_type":"approximate_band","target":40,"lower":null,"upper":null,"evidence":"预算大概40元左右","confidence":0.95}
"""


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _source_tasks(path: Path) -> dict[int, dict[str, str]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for task_id, item in enumerate(items):
        instruction = item["instructions"][0]["instruction"]
        asin = item["instructions"][0].get("asin") or item["asin"]
        result[task_id] = {"instruction": str(instruction), "asin": str(asin)}
    return result


def _existing_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(row["task_id"]) for row in _read_jsonl(path)}


def _strip_json_fence(content: str) -> str:
    value = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


class ResponseFormatError(ValueError):
    """A model response whose answer channel cannot be parsed as JSON."""

    def __init__(self, message: str, diagnostic: dict[str, Any]):
        super().__init__(message)
        self.diagnostic = diagnostic


def _content_from_parts(value: object) -> str | None:
    """Normalise OpenAI-style string or content-part responses to text."""
    if isinstance(value, str):
        return value.strip() or None
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for part in value:
        if isinstance(part, str):
            text = part
        elif isinstance(part, dict):
            text = part.get("text") or part.get("content") or ""
        else:
            text = ""
        if isinstance(text, str) and text:
            parts.append(text)
    joined = "".join(parts).strip()
    return joined or None


def _response_diagnostic(body: object, message: object, content: object) -> dict[str, Any]:
    """Keep response-shape evidence for debugging without request headers/secrets."""
    body_dict = body if isinstance(body, dict) else {}
    message_dict = message if isinstance(message, dict) else {}
    choices = body_dict.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    usage = body_dict.get("usage")
    usage_dict = usage if isinstance(usage, dict) else {}
    preview = _content_from_parts(content)
    return {
        "response_keys": sorted(str(key) for key in body_dict),
        "message_keys": sorted(str(key) for key in message_dict),
        "content_type": type(content).__name__,
        "content_length": len(preview) if preview is not None else 0,
        "content_preview": preview[:800] if preview else None,
        "has_reasoning_content": bool(message_dict.get("reasoning_content")),
        "finish_reason": first_choice.get("finish_reason"),
        "completion_tokens": usage_dict.get("completion_tokens"),
    }


def _extract_response_content(body: object) -> str:
    """Extract the textual answer across common OpenAI-compatible response shapes."""
    if not isinstance(body, dict):
        raise ResponseFormatError("response body is not a JSON object", {"body_type": type(body).__name__})
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ResponseFormatError("response has no usable choices[0]", _response_diagnostic(body, None, None))
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ResponseFormatError("response choices[0] has no message object", _response_diagnostic(body, message, None))

    for field in ("content", "output_text", "text"):
        text = _content_from_parts(message.get(field))
        if text is not None:
            return text
    text = _content_from_parts(body.get("output_text"))
    if text is not None:
        return text
    raise ResponseFormatError("response message has no textual answer", _response_diagnostic(body, message, message.get("content")))


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 2) if number > 0 else None


def validate_model_label(raw: object, instruction: str) -> dict[str, Any]:
    """Validate model output and apply deterministic approximate-band policy."""
    if not isinstance(raw, dict):
        raise ValueError("model response must be a JSON object")
    kind = str(raw.get("budget_type") or "")
    if kind not in VALID_TYPES:
        raise ValueError(f"invalid budget_type: {kind!r}")
    evidence = raw.get("evidence")
    if evidence is not None:
        evidence = str(evidence).strip()
        if not evidence or _normalise(evidence) not in _normalise(instruction):
            raise ValueError("evidence is not an exact instruction substring")
    confidence = raw.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")

    target = _number(raw.get("target"))
    lower = _number(raw.get("lower"))
    upper = _number(raw.get("upper"))
    if kind == "approximate_band":
        if target is None:
            raise ValueError("approximate_band requires target")
        lower, upper = _approximate_band(target)
    elif kind == "hard_upper":
        if upper is None:
            raise ValueError("hard_upper requires upper")
        target = lower = None
    elif kind == "range":
        if lower is None or upper is None or lower > upper:
            raise ValueError("range requires ordered lower and upper")
        target = None
    elif kind == "approximate_range":
        if lower is None or upper is None or lower > upper:
            raise ValueError("approximate_range requires ordered stated lower and upper")
        lower, upper = _approximate_range(lower, upper)
        target = None
    elif kind == "lower_bound":
        if lower is None:
            raise ValueError("lower_bound requires lower")
        target = upper = None
    else:
        target = lower = upper = None
    return {
        "budget_type": kind,
        "target": target,
        "lower": lower,
        "upper": upper,
        "evidence": evidence,
        "confidence": round(confidence, 4),
    }


def _chat_completion(
    *,
    url: str,
    api_key: str,
    model: str,
    instruction: str,
    max_tokens: int,
    json_mode: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"用户原文：\n{instruction}"},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    def send(request_payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{url.rstrip('/')}/chat/completions",
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                # OpenCode Go's Cloudflare edge rejects urllib's default
                # ``Python-urllib/<version>`` user agent (error 1010).
                "User-Agent": "shopping-grpo-budget-labeler/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        body = send(payload)
    except urllib.error.HTTPError as exc:
        # OpenCode Go's chat-completions endpoint currently rejects response_format,
        # although it remains OpenAI-compatible for ordinary chat requests.  The
        # system prompt and local validation still enforce the output contract.
        if not json_mode or exc.code not in {400, 403, 422}:
            raise
        payload.pop("response_format", None)
        body = send(payload)
    content = _extract_response_content(body)
    try:
        return json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        message = body.get("choices", [{}])[0].get("message") if isinstance(body.get("choices"), list) and body["choices"] else None
        content_value = message.get("content") if isinstance(message, dict) else None
        raise ResponseFormatError(str(exc), _response_diagnostic(body, message, content_value)) from exc


def _write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _label_one(
    queued: dict[str, Any],
    *,
    tasks: dict[int, dict[str, str]],
    url: str,
    api_key: str,
    model: str,
    retries: int,
    max_tokens: int,
    json_mode: bool,
) -> tuple[dict[str, Any], str | None]:
    task_id = int(queued["task_id"])
    task = tasks.get(task_id)
    if task is None:
        raise ValueError(f"task {task_id} is absent from source")
    error = None
    diagnostic = None
    for attempt in range(retries + 1):
        try:
            raw = _chat_completion(
                url=url,
                api_key=api_key,
                model=model,
                instruction=task["instruction"],
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            semantic = validate_model_label(raw, task["instruction"])
            break
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            error = str(exc)
            diagnostic = exc.diagnostic if isinstance(exc, ResponseFormatError) else None
            semantic = None
            if attempt < retries:
                time.sleep(1 + attempt)
    if semantic is None:
        semantic = {
            "budget_type": "unknown",
            "target": None,
            "lower": None,
            "upper": None,
            "evidence": None,
            "confidence": 0.0,
        }
        status = "llm_error"
    else:
        status = "pending_audit"
    row = {
        **semantic,
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "asin": task["asin"],
        "instruction_sha256": hashlib.sha256(task["instruction"].encode("utf-8")).hexdigest(),
        "source": "llm",
        "model": model,
        "review_status": status,
    }
    if error is not None:
        row["error"] = error
    if diagnostic is not None:
        row["response_diagnostic"] = diagnostic
    return row, error


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dotenv", type=Path, default=root / ".env")
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "environments/ShopSimulator/shop_env/data/items_eval_train.json",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_needs_llm.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_llm.jsonl",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=None,
        help="deterministically shuffle pending task IDs before applying --limit",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="completion-token budget; reasoning-capable models need room for a final JSON answer",
    )
    parser.add_argument(
        "--json-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request OpenAI-compatible JSON-object mode (enabled by default)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_dotenv(args.dotenv)
    url = os.environ.get("OPENCODE_URL") or os.environ.get("DEEPSEEK_URL")
    api_key = os.environ.get("OPENCODE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    model = args.model or os.environ.get("OPENCODE_MODEL") or os.environ.get("DEEPSEEK_MODEL") or MODEL_DEFAULT
    if not url or not api_key:
        raise SystemExit("configure OPENCODE_URL/OPENCODE_API_KEY or DEEPSEEK_URL/DEEPSEEK_API_KEY")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    tasks = _source_tasks(args.source)
    queue = _read_jsonl(args.queue)
    completed = _existing_ids(args.output)
    pending = [row for row in queue if int(row["task_id"]) not in completed]
    if args.selection_seed is not None:
        pending.sort(
            key=lambda row: hashlib.sha256(
                f"{args.selection_seed}:{int(row['task_id'])}".encode("utf-8")
            ).hexdigest()
        )
    if args.limit is not None:
        pending = pending[: args.limit]

    counts: dict[str, int] = {"completed": len(completed), "requested": len(pending), "written": 0, "errors": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _label_one,
                queued,
                tasks=tasks,
                url=url,
                api_key=api_key,
                model=model,
                retries=args.retries,
                max_tokens=args.max_tokens,
                json_mode=args.json_mode,
            ): int(queued["task_id"])
            for queued in pending
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task_id = futures[future]
            try:
                row, error = future.result()
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            _write_row(args.output, row)
            counts["written"] += 1
            if error is not None:
                counts["errors"] += 1
            print(json.dumps({"progress": f"{index}/{len(pending)}", "task_id": task_id, "budget_type": row["budget_type"]}, ensure_ascii=False), flush=True)
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
