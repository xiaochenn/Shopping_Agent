"""Load frozen task-level budget semantics without mutating source tasks."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path


BUDGET_SEMANTICS_VERSION = "shopping-budget-semantics-v1"
EXECUTABLE_TYPES = {
    "hard_upper",
    "approximate_band",
    "range",
    "approximate_range",
    "lower_bound",
    "price_preference",
    "no_explicit_budget",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_labels_path() -> Path:
    value = os.environ.get("SHOP_BUDGET_SEMANTICS")
    return Path(value) if value else _repository_root() / "data/annotations/budget_semantics_v1_merged.jsonl"


def _default_excluded_path() -> Path:
    value = os.environ.get("SHOP_BUDGET_SEMANTICS_EXCLUDED")
    return (
        Path(value)
        if value
        else _repository_root() / "data/annotations/budget_semantics_v1_excluded_human_unknown.jsonl"
    )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _number(value, *, allow_zero: bool) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0 or (number == 0 and not allow_zero):
        return None
    return round(number, 2)


def _constraint_from_row(row: dict) -> dict:
    task_id = row.get("task_id")
    if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
        raise ValueError("budget semantics task_id must be a non-negative integer")
    if row.get("schema_version") != BUDGET_SEMANTICS_VERSION:
        raise ValueError(f"task {task_id}: unsupported budget semantics schema")
    kind = row.get("budget_type")
    if kind not in EXECUTABLE_TYPES:
        raise ValueError(f"task {task_id}: non-executable budget type {kind!r}")
    lower = _number(row.get("lower"), allow_zero=True)
    upper = _number(row.get("upper"), allow_zero=False)
    target = _number(row.get("target"), allow_zero=False)

    if kind == "hard_upper" and upper is None:
        raise ValueError(f"task {task_id}: hard_upper requires upper")
    if kind == "lower_bound" and lower is None:
        raise ValueError(f"task {task_id}: lower_bound requires lower")
    if kind == "approximate_band" and (target is None or lower is None or upper is None):
        raise ValueError(f"task {task_id}: approximate_band requires target/lower/upper")
    if kind in {"range", "approximate_range"} and (lower is None or upper is None):
        raise ValueError(f"task {task_id}: {kind} requires lower/upper")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"task {task_id}: lower exceeds upper")
    if kind in {"price_preference", "no_explicit_budget"}:
        lower = upper = target = None
    elif kind == "hard_upper":
        lower = target = None
    elif kind == "lower_bound":
        upper = target = None
    elif kind in {"range", "approximate_range"}:
        target = None

    return {
        "task_id": task_id,
        "budget_type": kind,
        "price_lower": lower,
        "price_upper": upper,
        "budget_target": target,
        "budget_evidence": row.get("evidence"),
        "budget_annotation_status": "labeled",
        "budget_eligible": True,
        "budget_label_asin": row.get("asin"),
        "budget_instruction_sha256": row.get("instruction_sha256"),
    }


def load_budget_semantics(labels_path: Path | None = None, excluded_path: Path | None = None) -> dict[int, dict]:
    """Return frozen executable constraints keyed by source task ID.

    Missing or malformed labels fail closed: the new harness must never silently
    fall back to the legacy instruction regex.
    """
    labels_path = Path(labels_path or _default_labels_path())
    excluded_path = Path(excluded_path or _default_excluded_path())
    if not labels_path.is_file():
        raise FileNotFoundError(f"budget semantics labels are missing: {labels_path}")
    labels: dict[int, dict] = {}
    for row in _read_jsonl(labels_path):
        constraint = _constraint_from_row(row)
        task_id = constraint["task_id"]
        if task_id in labels:
            raise ValueError(f"duplicate budget semantics task_id: {task_id}")
        labels[task_id] = constraint

    excluded_rows: dict[int, dict] = {}
    if excluded_path.is_file():
        for row in _read_jsonl(excluded_path):
            task_id = row.get("task_id")
            if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
                raise ValueError("excluded budget task_id must be a non-negative integer")
            if task_id in excluded_rows:
                raise ValueError(f"duplicate excluded budget task_id: {task_id}")
            excluded_rows[task_id] = row
    excluded = set(excluded_rows)
    overlap = set(labels) & excluded
    if overlap:
        raise ValueError(f"budget labels and exclusions overlap: {sorted(overlap)[:5]}")
    for task_id in excluded:
        excluded_row = excluded_rows[task_id]
        labels[task_id] = {
            "task_id": task_id,
            "budget_type": "unknown",
            "price_lower": None,
            "price_upper": None,
            "budget_target": None,
            "budget_evidence": None,
            "budget_annotation_status": "excluded_human_unknown",
            "budget_eligible": False,
            "budget_label_asin": excluded_row.get("asin"),
            "budget_instruction_sha256": excluded_row.get("instruction_sha256"),
        }
    return labels


def constraint_for_task(task_id: int, semantics: dict[int, dict]) -> dict:
    try:
        return semantics[int(task_id)]
    except KeyError as exc:
        raise ValueError(f"task {task_id}: missing frozen budget semantics") from exc


def validate_constraint_source(constraint: dict, *, asin: object, instruction: object) -> None:
    """Reject a sidecar label if it no longer belongs to the source task."""
    expected_asin = constraint.get("budget_label_asin")
    expected_hash = constraint.get("budget_instruction_sha256")
    actual_asin = str(asin)
    actual_hash = hashlib.sha256(str(instruction).encode("utf-8")).hexdigest()
    if not isinstance(expected_asin, str) or expected_asin != actual_asin:
        raise ValueError(
            f"task {constraint.get('task_id')}: budget label ASIN does not match source"
        )
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ValueError(
            f"task {constraint.get('task_id')}: budget label instruction hash does not match source"
        )
