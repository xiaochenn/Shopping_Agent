import unittest

from scripts.label_budget_semantics_llm import ResponseFormatError, _extract_response_content, validate_model_label


class LabelBudgetSemanticsLlmTests(unittest.TestCase):
    def test_canonicalizes_approximate_band_with_local_policy(self):
        value = validate_model_label(
            {
                "budget_type": "approximate_band",
                "target": 40,
                "lower": None,
                "upper": None,
                "evidence": "预算大概40元左右",
                "confidence": 0.9,
            },
            "预算大概40元左右。",
        )
        self.assertEqual(value["lower"], 30.0)
        self.assertEqual(value["upper"], 50.0)

    def test_rejects_nonliteral_evidence_and_invalid_range(self):
        with self.assertRaisesRegex(ValueError, "exact instruction substring"):
            validate_model_label(
                {"budget_type": "hard_upper", "upper": 100, "evidence": "预算100元", "confidence": 1},
                "预算不超过100元。",
            )
        with self.assertRaisesRegex(ValueError, "ordered lower and upper"):
            validate_model_label(
                {"budget_type": "range", "lower": 200, "upper": 100, "evidence": "100到200元", "confidence": 1},
                "价格在100到200元。",
            )

    def test_canonicalizes_approximate_range_with_local_policy(self):
        value = validate_model_label(
            {
                "budget_type": "approximate_range",
                "target": None,
                "lower": 80,
                "upper": 100,
                "evidence": "价格在80-100元左右",
                "confidence": 0.9,
            },
            "价格在80-100元左右。",
        )
        self.assertEqual((value["lower"], value["upper"]), (70.0, 110.0))

    def test_extracts_string_and_content_part_answer_channels(self):
        self.assertEqual(
            _extract_response_content({"choices": [{"message": {"content": "{\"budget_type\": \"range\"}"}}]}),
            '{"budget_type": "range"}',
        )
        self.assertEqual(
            _extract_response_content(
                {
                    "choices": [
                        {"message": {"content": [{"type": "text", "text": "{\"budget_type\":"}, {"text": " \"range\"}"}]}}
                    ]
                }
            ),
            '{"budget_type": "range"}',
        )

    def test_empty_answer_has_safe_shape_diagnostic(self):
        with self.assertRaises(ResponseFormatError) as caught:
            _extract_response_content(
                {"id": "request-id", "choices": [{"message": {"content": "", "reasoning_content": "hidden"}}]}
            )
        diagnostic = caught.exception.diagnostic
        self.assertEqual(diagnostic["content_length"], 0)
        self.assertTrue(diagnostic["has_reasoning_content"])
        self.assertNotIn("hidden", str(diagnostic))


if __name__ == "__main__":
    unittest.main()
