"""Tests for the deterministic ShoppingState / fixed-K context contract."""

import unittest

from shopping_grpo.environment.shopping_state import (
    augment_current_observation,
    build_context_view,
    empty_shopping_state,
    reduce_shopping_state,
)


def assistant(call_id, name, arguments):
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def tool(call_id, name, content):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


class ShoppingStateTest(unittest.TestCase):
    def test_state_keeps_facts_without_current_page_buttons(self):
        observation = """[SHOPPING_OBSERVATION_V2]
page_type: product_detail
asin: 750684323117
title: 自动浇水器
brand: 园艺店
category: 自动灌溉设备
price: 228.0
key_attributes: 电磁阀, 雾化
selected_options: {\"颜色分类\": \"25米套装\"}
可点击的按钮: [\"buy now\", \"25米套装\"]"""
        state = reduce_shopping_state(
            empty_shopping_state(), "select_option", {"value": "25米套装"}, observation
        )
        rendered = augment_current_observation("current page", state)
        self.assertIn("750684323117", rendered)
        self.assertIn("read_only: true", rendered)
        self.assertNotIn("buy now", rendered.split("[/SHOPPING_STATE_V1]", 1)[0])
        self.assertEqual(state["reviewed_products"][0]["selected_price"], 228.0)

    def test_context_keeps_only_three_full_results_and_never_mutates_audit(self):
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"}]
        for index in range(4):
            call_id = f"call-{index}"
            messages.extend([assistant(call_id, "search_products", "{}"), tool(call_id, "search_products", f"page-{index}")])
        view, meta = build_context_view(messages, keep_recent_groups=3)
        self.assertEqual(meta["cleared_tool_results"], 1)
        self.assertIn("[SHOPPING_TOOL_RESULT_CLEARED_V1]", view[3]["content"])
        self.assertNotIn("page-0", view[3]["content"])
        self.assertEqual(view[-1]["content"], "page-3")
        self.assertEqual(messages[3]["content"], "page-0")


if __name__ == "__main__":
    unittest.main()
