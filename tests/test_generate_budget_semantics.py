import unittest

from scripts.generate_budget_semantics import extract_budget_semantics


class BudgetSemanticsTests(unittest.TestCase):
    def test_explicit_upper_is_a_hard_upper(self):
        label = extract_budget_semantics("想买耳机，预算控制在1000元以内。")
        self.assertEqual(label["budget_type"], "hard_upper")
        self.assertEqual(label["upper"], 1000.0)

    def test_approximate_budget_uses_asymmetric_band(self):
        label = extract_budget_semantics("大概预算在3600块钱这个范围。")
        self.assertEqual(label["budget_type"], "approximate_band")
        self.assertEqual(label["target"], 3600.0)
        self.assertEqual(label["lower"], 3240.0)
        self.assertEqual(label["upper"], 3780.0)

    def test_approximate_budget_with_around_is_not_a_hard_upper(self):
        label = extract_budget_semantics("预算在3000元左右。")
        self.assertEqual(label["budget_type"], "approximate_band")
        self.assertEqual(label["lower"], 2700.0)
        self.assertEqual(label["upper"], 3150.0)

    def test_low_value_approximate_budget_has_a_ten_yuan_floor(self):
        label = extract_budget_semantics("预算大概40元左右。")
        self.assertEqual(label["budget_type"], "approximate_band")
        self.assertEqual(label["lower"], 30.0)
        self.assertEqual(label["upper"], 50.0)

    def test_range_and_lower_bound_are_distinct(self):
        price_range = extract_budget_semantics("价格在1000到2000元之间。")
        lower_bound = extract_budget_semantics("预算在4k以上。")
        self.assertEqual(price_range["budget_type"], "range")
        self.assertEqual((price_range["lower"], price_range["upper"]), (1000.0, 2000.0))
        self.assertEqual(lower_bound["budget_type"], "lower_bound")
        self.assertEqual(lower_bound["lower"], 4000.0)

    def test_approximate_range_is_expanded_with_local_policy(self):
        label = extract_budget_semantics("价格在80-100元左右。")
        self.assertEqual(label["budget_type"], "approximate_range")
        self.assertEqual((label["lower"], label["upper"]), (70.0, 110.0))

    def test_ambiguous_or_absent_price_goes_to_correct_queue(self):
        ambiguous = extract_budget_semantics("预算不超过三千元，想买国产耳机。")
        absent = extract_budget_semantics("想买一款轻便的国产耳机。")
        self.assertEqual(ambiguous["budget_type"], "needs_llm")
        self.assertEqual(absent["budget_type"], "no_explicit_budget")
        self.assertEqual(
            extract_budget_semantics("适合5岁左右小孩的枕头。 ")["budget_type"],
            "no_explicit_budget",
        )


if __name__ == "__main__":
    unittest.main()
