#!/usr/bin/env python3
"""Create a task-level human-review queue for unresolved budget annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _source_tasks(path: Path) -> dict[int, dict[str, str]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return {
        task_id: {
            "instruction": str(item["instructions"][0]["instruction"]),
            "asin": str(item["instructions"][0].get("asin") or item["asin"]),
        }
        for task_id, item in enumerate(items)
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_llm_json_mode_retry_2048.jsonl",
    )
    parser.add_argument(
        "--initial-labels",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_llm.jsonl",
        help="first-pass labels; valid unknown rows not superseded by --labels are also reviewed",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "environments/ShopSimulator/shop_env/data/items_eval_train.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_manual_review.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = _source_tasks(args.source)
    retry_labels = _read_jsonl(args.labels)
    retry_ids = {int(row["task_id"]) for row in retry_labels}
    unresolved_by_id = {
        int(row["task_id"]): row for row in retry_labels if row["budget_type"] == "unknown"
    }
    # The retry only covers first-pass transport failures.  Preserve any valid
    # first-pass semantic unknown that was not part of that retry set.
    for row in _read_jsonl(args.initial_labels):
        task_id = int(row["task_id"])
        if task_id not in retry_ids and row["budget_type"] == "unknown":
            unresolved_by_id[task_id] = row
    unresolved = list(unresolved_by_id.values())
    review_rows: list[dict[str, Any]] = []
    for label in sorted(unresolved, key=lambda row: int(row["task_id"])):
        task_id = int(label["task_id"])
        task = tasks[task_id]
        review_rows.append(
            {
                "task_id": task_id,
                "asin": task["asin"],
                "instruction": task["instruction"],
                "model_budget_type": label["budget_type"],
                "model_target": label.get("target"),
                "model_lower": label.get("lower"),
                "model_upper": label.get("upper"),
                "model_evidence": label.get("evidence"),
                "model_confidence": label.get("confidence"),
                "model_review_status": label["review_status"],
                "model_error": label.get("error"),
                "human_budget_type": None,
                "human_target": None,
                "human_lower": None,
                "human_upper": None,
                "human_evidence": None,
                "human_notes": None,
                "review_status": "pending_manual",
            }
        )
    _write_jsonl(args.output, review_rows)
    print(json.dumps({"manual_review_rows": len(review_rows), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
