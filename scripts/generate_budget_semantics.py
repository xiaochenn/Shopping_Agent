#!/usr/bin/env python3
"""Create deterministic Tier-1 budget-semantics labels for ShopSimulator tasks.

This script deliberately favors precision over recall.  It labels only
unambiguous price expressions with regular expressions and emits every other
task to a separate queue for the Tier-2 semantic model / human-review path.
It never changes the embedded ShopSimulator product or task archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "shopping-budget-semantics-v1"
APPROX_LOWER_TOLERANCE = 0.10
APPROX_UPPER_TOLERANCE = 0.05
APPROX_ABSOLUTE_TOLERANCE_FLOOR = 10.0

# A currency token or a multiplier unit is required for single-value rules.
# This prevents a bare number elsewhere in the instruction being mistaken for
# a budget.  Chinese-number expressions (for example, “三千”) intentionally
# fall through to the semantic-model queue.
AMOUNT = r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>万|千|[kK])?\s*(?P<currency>元|块钱|块)?"
HARD_UPPER_PATTERNS = (
    (
        "explicit_upper_budget",
        re.compile(
            rf"(?:预算|价格)(?:控制)?\s*(?:在|为|是)?\s*{AMOUNT}\s*(?:以内|以下|内)(?![\u4e00-\u9fff])"
        ),
    ),
    (
        "explicit_upper_phrase",
        re.compile(rf"(?:不超过|不高于|最高)\s*{AMOUNT}(?![\u4e00-\u9fff])"),
    ),
)
APPROXIMATE_PATTERN = re.compile(
    rf"(?:预算|价格)(?:控制)?\s*(?:在|为|是)?\s*"
    rf"(?P<prefix>大概|大约|约|差不多)?\s*{AMOUNT}\s*"
    rf"(?P<suffix>左右|上下|这个范围|范围)?"
)
RANGE_PATTERN = re.compile(
    r"(?:预算|价格)(?:控制)?\s*(?:在|为|是)?\s*"
    r"(?P<low_value>\d+(?:\.\d+)?)\s*(?P<low_unit>万|千|[kK])?\s*(?:元|块钱|块)?\s*"
    r"(?:-|~|～|至|到)\s*"
    r"(?P<high_value>\d+(?:\.\d+)?)\s*(?P<high_unit>万|千|[kK])?\s*(?:元|块钱|块)?"
)
LOWER_BOUND_PATTERN = re.compile(
    rf"(?:预算|价格)(?:控制)?\s*(?:在|为|是)?\s*{AMOUNT}\s*(?:起|以上|及以上)(?![\u4e00-\u9fff])"
)
PRICE_SIGNAL_PATTERN = re.compile(
    r"预算|价格|价钱|花费|人民币|不超过|不高于|最高|以内|以下|"
    r"\d+(?:\.\d+)?\s*(?:万|千|[kK]|元|块钱|块)"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise(text: object) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).replace(",", "")


def _amount(value: str, unit: str | None, currency: str | None = None) -> float | None:
    """Return a numeric amount only when currency context is explicit."""
    if not unit and not currency:
        return None
    amount = float(value)
    unit = str(unit or "").casefold()
    if unit == "万":
        amount *= 10000
    elif unit in {"千", "k"}:
        amount *= 1000
    return amount


def _round_price(value: float) -> float:
    return round(value, 2)


def _approximate_band(target: float) -> tuple[float, float]:
    """Return the asymmetric band with a minimum ten-yuan tolerance."""
    lower_delta = max(APPROX_ABSOLUTE_TOLERANCE_FLOOR, target * APPROX_LOWER_TOLERANCE)
    upper_delta = max(APPROX_ABSOLUTE_TOLERANCE_FLOOR, target * APPROX_UPPER_TOLERANCE)
    return max(0.0, _round_price(target - lower_delta)), _round_price(target + upper_delta)


def _approximate_range(lower: float, upper: float) -> tuple[float, float]:
    """Expand an approximate range using the asymmetric tolerance policy."""
    lower_delta = max(APPROX_ABSOLUTE_TOLERANCE_FLOOR, lower * APPROX_LOWER_TOLERANCE)
    upper_delta = max(APPROX_ABSOLUTE_TOLERANCE_FLOOR, upper * APPROX_UPPER_TOLERANCE)
    return max(0.0, _round_price(lower - lower_delta)), _round_price(upper + upper_delta)


def _base_label(
    *,
    kind: str,
    rule: str,
    evidence: str | None,
    target: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "budget_type": kind,
        "target": target,
        "lower": lower,
        "upper": upper,
        "evidence": evidence,
        "source": "regex",
        "rule": rule,
        "review_status": "auto_accepted" if kind != "needs_llm" else "pending_llm",
    }


def extract_budget_semantics(instruction: object) -> dict[str, Any]:
    """Return a conservative Tier-1 budget label for one instruction."""
    text = _normalise(instruction)

    # An explicit range has precedence, so “1000-2000” is never parsed as 1000.
    match = RANGE_PATTERN.search(text)
    if match:
        # Optional currency tokens can make a regex engine backtrack and stop
        # before “元左右”.  Inspect the unmatched tail explicitly so an
        # approximate range is never silently accepted as a strict range.
        approximate_tail = re.match(
            r"\s*(?:元|块钱|块)?\s*(?:左右|上下|这个范围|范围)",
            text[match.end() :],
        )
        low = _amount(match.group("low_value"), match.group("low_unit"), "元")
        high = _amount(match.group("high_value"), match.group("high_unit"), "元")
        if approximate_tail is not None and low is not None and high is not None and 0 < low <= high:
            lower, upper = _approximate_range(low, high)
            return _base_label(
                kind="approximate_range",
                rule="approximate_range_10_down_5_up_min_10_yuan",
                evidence=match.group(0) + approximate_tail.group(0),
                lower=lower,
                upper=upper,
            )
        if (
            approximate_tail is None
            and low is not None
            and high is not None
            and 0 < low <= high
        ):
            return _base_label(
                kind="range",
                rule="explicit_range",
                evidence=match.group(0),
                lower=_round_price(low),
                upper=_round_price(high),
            )

    for rule, pattern in HARD_UPPER_PATTERNS:
        match = pattern.search(text)
        if match:
            amount = _amount(match.group("value"), match.group("unit"), match.group("currency"))
            if amount is not None and amount > 0:
                return _base_label(
                    kind="hard_upper",
                    rule=rule,
                    evidence=match.group(0),
                    upper=_round_price(amount),
                )

    # “左右”等近似词不是 hard_upper：它们具有当前项目约定的非对称价格带。
    match = APPROXIMATE_PATTERN.search(text)
    if match and (match.group("prefix") or match.group("suffix")):
        amount = _amount(match.group("value"), match.group("unit"), match.group("currency"))
        if amount is not None and amount > 0:
            lower, upper = _approximate_band(amount)
            return _base_label(
                kind="approximate_band",
                rule="approximate_budget_band_10_down_5_up_min_10_yuan",
                evidence=match.group(0),
                target=_round_price(amount),
                lower=lower,
                upper=upper,
            )

    match = LOWER_BOUND_PATTERN.search(text)
    if match:
        amount = _amount(match.group("value"), match.group("unit"), match.group("currency"))
        if amount is not None and amount > 0:
            return _base_label(
                kind="lower_bound",
                rule="explicit_lower_bound",
                evidence=match.group(0),
                lower=_round_price(amount),
            )

    if not PRICE_SIGNAL_PATTERN.search(text):
        return _base_label(
            kind="no_explicit_budget",
            rule="no_price_signal",
            evidence=None,
        )
    return _base_label(
        kind="needs_llm",
        rule="insufficient_regex_confidence",
        evidence=None,
    )


def iter_tasks(path: Path):
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a JSON array")
    for task_id, item in enumerate(items):
        instructions = item.get("instructions")
        if not isinstance(instructions, list) or len(instructions) != 1:
            raise ValueError(f"task {task_id}: expected exactly one instruction")
        record = instructions[0]
        instruction = str(record.get("instruction") or "")
        asin = str(record.get("asin") or item.get("asin") or "")
        if not instruction or not asin:
            raise ValueError(f"task {task_id}: missing instruction or asin")
        yield task_id, asin, instruction


def build_labels(source: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels = []
    counts: Counter[str] = Counter()
    for task_id, asin, instruction in iter_tasks(source):
        label = extract_budget_semantics(instruction)
        label.update(
            {
                "task_id": task_id,
                "asin": asin,
                "instruction_sha256": _sha256_text(instruction),
            }
        )
        labels.append(label)
        counts[str(label["budget_type"])] += 1
    return labels, dict(sorted(counts.items()))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "environments/ShopSimulator/shop_env/data/items_eval_train.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1.jsonl",
    )
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_needs_llm.jsonl",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1.metadata.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels, counts = build_labels(args.source)
    task_ids = [int(label["task_id"]) for label in labels]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task IDs")
    _write_jsonl(args.output, labels)
    _write_jsonl(
        args.review_queue,
        [label for label in labels if label["budget_type"] == "needs_llm"],
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source": str(args.source),
        "source_sha256": _sha256_file(args.source),
        "task_count": len(labels),
        "counts_by_budget_type": counts,
        "approximate_band": {
            "lower_tolerance": APPROX_LOWER_TOLERANCE,
            "upper_tolerance": APPROX_UPPER_TOLERANCE,
            "absolute_tolerance_floor": APPROX_ABSOLUTE_TOLERANCE_FLOOR,
        },
        "approximate_range": {
            "lower_policy": "subtract max(10 yuan, 10% of stated lower)",
            "upper_policy": "add max(10 yuan, 5% of stated upper)",
        },
        "tier_1_policy": "high_precision_regex_only",
        "tier_2_queue": str(args.review_queue),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **metadata}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
