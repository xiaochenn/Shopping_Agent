#!/usr/bin/env python3
"""Merge Tier-1, LLM, retry, and human budget annotations into one sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.generate_budget_semantics import SCHEMA_VERSION, _normalise
from scripts.label_budget_semantics_llm import validate_model_label


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _by_task_id(path: Path) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(path)
    result = {int(row["task_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate task_id in {path}")
    return result


def _source_tasks(path: Path) -> dict[int, dict[str, str]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return {
        task_id: {
            "asin": str(item["instructions"][0].get("asin") or item["asin"]),
            "instruction": str(item["instructions"][0]["instruction"]),
        }
        for task_id, item in enumerate(items)
    }


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate(label: dict[str, Any], task: dict[str, str], origin: str) -> dict[str, Any]:
    if str(label.get("asin")) != task["asin"]:
        raise ValueError(f"{origin}: ASIN does not match source for task {label['task_id']}")
    expected_hash = hashlib.sha256(task["instruction"].encode("utf-8")).hexdigest()
    if label.get("instruction_sha256") != expected_hash:
        raise ValueError(f"{origin}: instruction hash does not match source for task {label['task_id']}")
    candidate = {key: label.get(key) for key in ("budget_type", "target", "lower", "upper", "evidence")}
    # Tier-1 and human rows intentionally have no model confidence.  Validation
    # needs the field only to verify the shared semantic shape.
    candidate["confidence"] = label.get("confidence", 1.0)
    if candidate["budget_type"] == "approximate_range":
        # Earlier stages already expanded the stated endpoints into the final
        # effective interval.  Validate its ordered numeric shape as a range,
        # but retain the semantic type without applying the expansion twice.
        lower, upper = candidate["lower"], candidate["upper"]
        if isinstance(lower, bool) or isinstance(upper, bool):
            raise ValueError(f"{origin}: approximate_range endpoints must be numeric")
        try:
            lower, upper = round(float(lower), 2), round(float(upper), 2)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{origin}: approximate_range endpoints must be numeric") from exc
        if lower < 0 or upper <= 0 or lower > upper:
            raise ValueError(f"{origin}: approximate_range endpoints must be ordered and non-negative")
        evidence = candidate["evidence"]
        if evidence is not None:
            evidence = str(evidence).strip()
            if not evidence or _normalise(evidence) not in _normalise(task["instruction"]):
                raise ValueError(f"{origin}: evidence is not an exact instruction substring")
        return {
            "budget_type": "approximate_range",
            "target": None,
            "lower": lower,
            "upper": upper,
            "evidence": evidence,
            "confidence": round(float(candidate["confidence"]), 4),
        }
    return validate_model_label(candidate, task["instruction"])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "environments/ShopSimulator/shop_env/data/items_eval_train.json")
    parser.add_argument("--tier1", type=Path, default=root / "data/annotations/budget_semantics_v1.jsonl")
    parser.add_argument("--initial-llm", type=Path, default=root / "data/annotations/budget_semantics_v1_llm.jsonl")
    parser.add_argument(
        "--retry-llm",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_llm_json_mode_retry_2048.jsonl",
    )
    parser.add_argument("--human", type=Path, default=root / "data/annotations/budget_semantics_v1_manual_decisions.jsonl")
    parser.add_argument("--output", type=Path, default=root / "data/annotations/budget_semantics_v1_merged.jsonl")
    parser.add_argument(
        "--excluded-output",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_excluded_human_unknown.jsonl",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=root / "data/annotations/budget_semantics_v1_merged.metadata.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = _source_tasks(args.source)
    tier1 = _by_task_id(args.tier1)
    initial_llm = _by_task_id(args.initial_llm)
    retry_llm = _by_task_id(args.retry_llm)
    human = _by_task_id(args.human)
    task_ids = set(tasks)
    if set(tier1) != task_ids:
        raise ValueError("Tier-1 labels do not cover exactly the source tasks")

    expected_llm = {task_id for task_id, row in tier1.items() if row["budget_type"] == "needs_llm"}
    if set(initial_llm) != expected_llm:
        raise ValueError("initial LLM labels do not cover exactly the Tier-1 queue")

    retry_unknown = {task_id for task_id, row in retry_llm.items() if row["budget_type"] == "unknown"}
    initial_unknown_not_retried = {
        task_id
        for task_id, row in initial_llm.items()
        if task_id not in retry_llm and row["budget_type"] == "unknown"
    }
    expected_human = retry_unknown | initial_unknown_not_retried
    if set(human) != expected_human:
        raise ValueError(
            f"human labels do not cover unresolved tasks: missing={len(expected_human - set(human))}, "
            f"unexpected={len(set(human) - expected_human)}"
        )

    merged: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    stages: Counter[str] = Counter()
    for task_id in sorted(task_ids):
        task = tasks[task_id]
        tier1_row = tier1[task_id]
        if tier1_row["budget_type"] != "needs_llm":
            chosen, stage = tier1_row, "tier1_regex"
        elif task_id in retry_llm:
            retry_row = retry_llm[task_id]
            if retry_row["budget_type"] == "unknown":
                chosen, stage = human[task_id], "human_review"
            else:
                chosen, stage = retry_row, "llm_retry"
        else:
            initial_row = initial_llm[task_id]
            if initial_row["budget_type"] == "unknown":
                chosen, stage = human[task_id], "human_review"
            else:
                chosen, stage = initial_row, "llm_initial"

        normalized = _validate(chosen, task, stage)
        if normalized["budget_type"] == "unknown" and stage == "human_review":
            excluded.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": task_id,
                    "asin": task["asin"],
                    "instruction_sha256": hashlib.sha256(task["instruction"].encode("utf-8")).hexdigest(),
                    "reason": "human_confirmed_unknown_budget_semantics",
                    "source": "human_review",
                }
            )
            continue
        stages[stage] += 1
        merged.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "asin": task["asin"],
                "instruction_sha256": hashlib.sha256(task["instruction"].encode("utf-8")).hexdigest(),
                **normalized,
                "review_status": chosen["review_status"],
                "source": "merged",
                "provenance": {
                    "stage": stage,
                    "source": chosen.get("source"),
                    "review_status": chosen.get("review_status"),
                },
            }
        )

    _atomic_write_jsonl(args.output, merged)
    _atomic_write_jsonl(args.excluded_output, excluded)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_task_count": len(tasks),
        "task_count": len(merged),
        "excluded_task_count": len(excluded),
        "stage_counts": dict(sorted(stages.items())),
        "budget_type_counts": dict(sorted(Counter(row["budget_type"] for row in merged).items())),
        "review_status_counts": dict(sorted(Counter(row["review_status"] for row in merged).items())),
        "human_confirmed_unknown_count": len(excluded),
    }
    _write_json(args.metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
