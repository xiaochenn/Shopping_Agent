import json
import tempfile
import unittest
from pathlib import Path

from web_agent_site.engine.budget_semantics import (
    constraint_for_task,
    load_budget_semantics,
    validate_constraint_source,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class BudgetSemanticsTest(unittest.TestCase):
    def test_loads_executable_constraints_and_preserves_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            labels = directory / "labels.jsonl"
            excluded = directory / "excluded.jsonl"
            write_jsonl(
                labels,
                [
                    {
                        "schema_version": "shopping-budget-semantics-v1",
                        "task_id": 0,
                        "budget_type": "approximate_band",
                        "target": 40,
                        "lower": 30,
                        "upper": 50,
                        "evidence": "40元左右",
                        "asin": "100",
                        "instruction_sha256": "52bb3cf998e1107ae3bfe4aaff3a9e08cacf5121d3881524153ce0b949ee95f2",
                    },
                    {
                        "schema_version": "shopping-budget-semantics-v1",
                        "task_id": 1,
                        "budget_type": "no_explicit_budget",
                        "target": None,
                        "lower": None,
                        "upper": None,
                        "evidence": None,
                        "asin": "101",
                        "instruction_sha256": "37a458205114f2f05992aee7cd2db0847552d0e7a2252abd7b26e7384992f546",
                    },
                ],
            )
            write_jsonl(
                excluded,
                [{
                    "task_id": 2,
                    "asin": "102",
                    "instruction_sha256": "53c73c578792bd99ddafc42ef7d9498e16b1ecfa28c5bb3c4d85f31136e628fb",
                }],
            )
            semantics = load_budget_semantics(labels, excluded)
            self.assertEqual(constraint_for_task(0, semantics)["price_lower"], 30.0)
            self.assertEqual(constraint_for_task(0, semantics)["price_upper"], 50.0)
            self.assertIsNone(constraint_for_task(1, semantics)["price_upper"])
            self.assertFalse(constraint_for_task(2, semantics)["budget_eligible"])
            validate_constraint_source(
                constraint_for_task(0, semantics), asin="100", instruction="价格在40元左右"
            )

    def test_source_mismatch_fails_closed(self):
        constraint = {
            "task_id": 0,
            "budget_label_asin": "100",
            "budget_instruction_sha256": "not-a-real-hash",
        }
        with self.assertRaisesRegex(ValueError, "instruction hash"):
            validate_constraint_source(constraint, asin="100", instruction="预算100元")

    def test_missing_task_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing frozen budget semantics"):
            constraint_for_task(3, {})
