#!/usr/bin/env python3
"""Interactively review unresolved task-level budget semantics in a terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.generate_budget_semantics import _approximate_band, _approximate_range


LABELS = {
    "1": ("hard_upper", "明确的最高价格 / 预算上限"),
    "2": ("approximate_band", "单个近似价格，如“100 元左右”"),
    "3": ("range", "明确价格区间"),
    "4": ("approximate_range", "带“左右/上下”的价格区间"),
    "5": ("lower_bound", "明确最低价格，如“100 元以上/起”"),
    "6": ("price_preference", "只表达便宜等偏好，没有可执行金额"),
    "7": ("no_explicit_budget", "没有价格要求"),
    "8": ("unknown", "人工确认仍无法可靠判定"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda value: int(value["task_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _ask_number(prompt: str) -> float:
    while True:
        raw = input(prompt).strip().replace("元", "")
        try:
            value = float(raw)
        except ValueError:
            print("请输入正数，例如 108 或 108.5。")
            continue
        if value > 0:
            return round(value, 2)
        print("请输入大于 0 的数值。")


def _ask_label() -> str | None:
    while True:
        print("\n可选标签：")
        for key, (name, description) in LABELS.items():
            print(f"  {key}. {name:<18} {description}")
        choice = input("选择 [1-8]，s=跳过，q=退出：").strip().lower()
        if choice == "s":
            return None
        if choice == "q":
            raise KeyboardInterrupt
        if choice in LABELS:
            return LABELS[choice][0]
        print("请输入 1-8、s 或 q。")


def _collect_constraint(kind: str) -> tuple[float | None, float | None, float | None]:
    """Return target, effective lower, effective upper for a human label."""
    if kind == "hard_upper":
        upper = _ask_number("价格上限 upper：")
        return None, None, upper
    if kind == "approximate_band":
        target = _ask_number("近似中心价 target：")
        lower, upper = _approximate_band(target)
        print(f"按既定容差，实际生效区间为 [{lower:g}, {upper:g}]。")
        return target, lower, upper
    if kind == "range":
        lower = _ask_number("价格下限 lower：")
        upper = _ask_number("价格上限 upper：")
        while upper < lower:
            print("upper 不能小于 lower。")
            upper = _ask_number("价格上限 upper：")
        return None, lower, upper
    if kind == "approximate_range":
        stated_lower = _ask_number("原文区间下限：")
        stated_upper = _ask_number("原文区间上限：")
        while stated_upper < stated_lower:
            print("上限不能小于下限。")
            stated_upper = _ask_number("原文区间上限：")
        lower, upper = _approximate_range(stated_lower, stated_upper)
        print(f"按既定容差，实际生效区间为 [{lower:g}, {upper:g}]。")
        return None, lower, upper
    if kind == "lower_bound":
        lower = _ask_number("价格下限 lower：")
        return None, lower, None
    return None, None, None


def _display(row: dict[str, Any], ordinal: int, total: int) -> None:
    print("\n" + "=" * 88)
    print(f"[{ordinal}/{total}] task_id={row['task_id']}  asin={row['asin']}")
    print("\n原始 prompt：")
    print(row["instruction"])
    print("\n模型参考：")
    print(
        f"  status={row['model_review_status']}  label={row['model_budget_type']}  "
        f"confidence={row.get('model_confidence')}"
    )
    if row.get("model_evidence"):
        print(f"  evidence={row['model_evidence']}")
    if row.get("model_error"):
        print(f"  error={row['model_error']}")


def _decision(row: dict[str, Any], kind: str, reviewer: str) -> dict[str, Any]:
    target, lower, upper = _collect_constraint(kind)
    default_evidence = row.get("model_evidence") or ""
    evidence = input(f"原文证据片段（回车使用 {default_evidence!r}；- 表示无）：").strip()
    if not evidence:
        evidence = default_evidence or None
    elif evidence == "-":
        evidence = None
    notes = input("备注（可留空）：").strip() or None
    return {
        "schema_version": "shopping-budget-semantics-v1",
        "source": "human",
        "task_id": row["task_id"],
        "asin": row["asin"],
        "instruction_sha256": hashlib.sha256(row["instruction"].encode("utf-8")).hexdigest(),
        "budget_type": kind,
        "target": target,
        "lower": lower,
        "upper": upper,
        "evidence": evidence,
        "review_status": "human_reviewed",
        "reviewer": reviewer or None,
        "notes": notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_manual_review.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_manual_decisions.jsonl",
    )
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--task-id", type=int, default=None, help="review one specific pending task")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = _read_jsonl(args.queue)
    existing = {int(row["task_id"]): row for row in _read_jsonl(args.output)}
    pending = [row for row in queue if int(row["task_id"]) not in existing]
    if args.task_id is not None:
        pending = [row for row in pending if int(row["task_id"]) == args.task_id]
    pending.sort(key=lambda row: int(row["task_id"]))
    if not pending:
        print("没有待人工复核的任务。")
        return 0

    print(f"待复核 {len(pending)} 条；每次提交立即保存到 {args.output}。")
    try:
        for ordinal, row in enumerate(pending, 1):
            _display(row, ordinal, len(pending))
            kind = _ask_label()
            if kind is None:
                continue
            existing[int(row["task_id"])] = _decision(row, kind, args.reviewer)
            _atomic_write_jsonl(args.output, list(existing.values()))
            print("已保存。")
    except KeyboardInterrupt:
        print("\n已退出；此前已保存的决策仍会保留。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
