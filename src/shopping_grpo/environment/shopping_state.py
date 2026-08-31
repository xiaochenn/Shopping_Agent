"""Deterministic, read-only shopping memory for the fixed-K agent harness.

The state is deliberately a *fact ledger*, not another copy of observations.
It is built only from successful environment actions and their agent-visible
observations.  In particular it never reads goals, rewards, or gold products.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping

from shopping_grpo.environment.context import clear_old_tool_results


SHOPPING_STATE_VERSION = "shopping-state-v1"
CONTEXT_POLICY_VERSION = "shopping-state-context-v1"
DEFAULT_KEEP_RECENT_GROUPS = 3

_FIELD_RE = re.compile(r"^([a-z_]+):\s*(.*)$", re.MULTILINE)
_TOTAL_RESULTS_RE = re.compile(r"Total results:\s*(\d+)")
_DETAIL_TOOLS = frozenset(
    {"open_product", "view_description", "view_features", "view_reviews", "view_attributes", "select_option"}
)
_SUBPAGE_TOOLS = {
    "view_description": "description",
    "view_features": "features",
    "view_reviews": "reviews",
    "view_attributes": "attributes",
}


def empty_shopping_state() -> dict[str, Any]:
    """Return a fresh canonical state with no task-specific hidden data."""

    return {
        "version": SHOPPING_STATE_VERSION,
        "as_of_step": 0,
        "searched_queries": [],
        "reviewed_products": [],
        "reviewed_product_archive": [],
        "evidence_excerpts": [],
        "current_product_asin": None,
        "terminal": False,
    }


def reduce_shopping_state(
    previous_state: Mapping[str, Any] | None,
    tool_name: str,
    parameters: Mapping[str, Any] | None,
    observation: str,
    *,
    done: bool = False,
    query_cap: int = 12,
    product_cap: int = 8,
    archive_cap: int = 12,
    viewed_pages_cap: int = 3,
) -> dict[str, Any]:
    """Return the next state after one *successful* environment result.

    Callers must not invoke this for parser failures or action-guard rejections.
    ``observation`` may be the raw environment text; only its public fields are
    copied into the returned state.
    """

    state = _normalise_state(previous_state)
    state["as_of_step"] += 1
    tool_name = str(tool_name or "")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    fields = _fields(observation)

    if tool_name == "search_products":
        query = str(parameters.get("query") or fields.get("query") or "").strip()
        if query:
            _upsert_query(
                state["searched_queries"], query, _result_count(observation), state["as_of_step"]
            )
            del state["searched_queries"][:-int(query_cap)]

    if tool_name in _DETAIL_TOOLS:
        asin = str(fields.get("asin") or parameters.get("asin") or "").strip()
        if asin:
            card = _upsert_product(state, asin, fields)
            card["last_access_step"] = state["as_of_step"]
            if tool_name in _SUBPAGE_TOOLS:
                page = _SUBPAGE_TOOLS[tool_name]
                if page not in card["viewed_pages"]:
                    card["viewed_pages"].append(page)
                del card["viewed_pages"][:-int(viewed_pages_cap)]
                _append_evidence_excerpt(
                    state,
                    asin=asin,
                    page=page,
                    text=fields.get("content", ""),
                )
            if tool_name == "select_option":
                selected = _json_object(fields.get("selected_options"))
                if selected:
                    card["selected_options"] = selected
                    card["selected_price"] = _number_or_text(fields.get("price"))
            card["selection_status"] = "current_candidate"
            state["current_product_asin"] = asin

    if tool_name in {"back_to_search", "next_page", "search_products"}:
        current = state.get("current_product_asin")
        if current:
            card = _find_product(state, current)
            if card is not None:
                card["selection_status"] = "reviewed_not_selected"
            state["current_product_asin"] = None

    state["terminal"] = bool(done)
    _enforce_product_caps(state, product_cap=int(product_cap), archive_cap=int(archive_cap))
    return canonical_shopping_state(state)


def canonical_shopping_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a deep-copied deterministic state suitable for JSONL audit traces."""

    value = _normalise_state(state)
    value["searched_queries"].sort(key=lambda row: (int(row.get("last_seen_step", 0)), row["query"]))
    value["reviewed_products"].sort(key=lambda row: (int(row.get("last_access_step", 0)), row["asin"]))
    value["reviewed_product_archive"].sort(
        key=lambda row: (int(row.get("last_access_step", 0)), row["asin"])
    )
    value["evidence_excerpts"].sort(
        key=lambda row: (int(row.get("step", 0)), row.get("asin", ""), row.get("page", ""))
    )
    return value


def shopping_state_json(state: Mapping[str, Any] | None) -> str:
    return json.dumps(canonical_shopping_state(state), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_shopping_state(state: Mapping[str, Any] | None) -> str:
    """Render a non-actionable prefix for the current tool result."""

    canonical = canonical_shopping_state(state)
    return "\n".join(
        (
            "[SHOPPING_STATE_V1]",
            f"as_of_step: {canonical['as_of_step']}",
            "read_only: true",
            "scope: facts from all successful prior tool results, including recent results; this is not a page and has no buttons.",
            "action_boundary: only the current tool result defines legal actions. Do not click identifiers from this state.",
            f"state_json: {shopping_state_json(canonical)}",
            "[/SHOPPING_STATE_V1]",
        )
    )


def augment_current_observation(observation: str, state: Mapping[str, Any] | None) -> str:
    """Attach state to an observation without changing the Action Guard input."""

    return f"{render_shopping_state(state)}\n\n{observation}"


def build_context_view(messages: list[Mapping[str, Any]], keep_recent_groups: int = DEFAULT_KEEP_RECENT_GROUPS):
    """Apply the fixed-K result-clearing view without mutating audit messages."""

    rewritten, stats = clear_old_tool_results(messages, keep_recent_groups=keep_recent_groups)
    return rewritten, {
        "version": CONTEXT_POLICY_VERSION,
        "keep_recent_complete_groups": int(keep_recent_groups),
        "cleared_tool_results": int(stats.cleared_tool_results),
    }


def _normalise_state(value: Mapping[str, Any] | None) -> dict[str, Any]:
    state = deepcopy(dict(value or empty_shopping_state()))
    state.setdefault("version", SHOPPING_STATE_VERSION)
    state.setdefault("as_of_step", 0)
    state.setdefault("searched_queries", [])
    state.setdefault("reviewed_products", [])
    state.setdefault("reviewed_product_archive", [])
    state.setdefault("evidence_excerpts", [])
    state.setdefault("current_product_asin", None)
    state.setdefault("terminal", False)
    return state


def _fields(observation: str) -> dict[str, str]:
    if not isinstance(observation, str):
        return {}
    return {key: value.strip() for key, value in _FIELD_RE.findall(observation) if value.strip()}


def _result_count(observation: str) -> int | None:
    match = _TOTAL_RESULTS_RE.search(observation or "")
    return int(match.group(1)) if match else None


def _upsert_query(
    rows: list[dict[str, Any]], query: str, result_count: int | None, step: int
) -> None:
    for row in rows:
        if row.get("query") == query:
            row["last_seen_step"] = int(step)
            if result_count is not None:
                row["result_count"] = result_count
            return
    rows.append({"query": query, "result_count": result_count, "last_seen_step": int(step)})


def _append_evidence_excerpt(state: dict[str, Any], *, asin: str, page: str, text: str) -> None:
    text = " ".join(str(text).split())
    if not text:
        return
    rows = state["evidence_excerpts"]
    rows.append(
        {
            "asin": asin,
            "page": page,
            "text": text[:160],
            "step": state["as_of_step"],
        }
    )
    del rows[:-6]


def _product_from_fields(asin: str, fields: Mapping[str, str]) -> dict[str, Any]:
    return {
        "asin": asin,
        "title": fields.get("title", ""),
        "brand": fields.get("brand", ""),
        "category": fields.get("category", ""),
        "price": fields.get("price", ""),
        "key_attributes": fields.get("key_attributes", ""),
        "viewed_pages": [],
        "selected_options": _json_object(fields.get("selected_options")),
        "selected_price": None,
        "selection_status": "under_review",
        "last_access_step": 0,
    }


def _upsert_product(state: dict[str, Any], asin: str, fields: Mapping[str, str]) -> dict[str, Any]:
    card = _find_product(state, asin)
    if card is None:
        archived = next((row for row in state["reviewed_product_archive"] if row.get("asin") == asin), None)
        if archived is not None:
            state["reviewed_product_archive"].remove(archived)
            card = _product_from_fields(asin, fields)
            card.update({key: archived[key] for key in ("title", "price", "selection_status", "last_access_step") if key in archived})
        else:
            card = _product_from_fields(asin, fields)
        state["reviewed_products"].append(card)
    for key in ("title", "brand", "category", "price", "key_attributes"):
        if fields.get(key):
            card[key] = fields[key]
    selected = _json_object(fields.get("selected_options"))
    if selected:
        card["selected_options"] = selected
    return card


def _find_product(state: Mapping[str, Any], asin: str) -> dict[str, Any] | None:
    return next((row for row in state.get("reviewed_products", []) if row.get("asin") == asin), None)


def _enforce_product_caps(state: dict[str, Any], *, product_cap: int, archive_cap: int) -> None:
    products = state["reviewed_products"]
    current = state.get("current_product_asin")
    while len(products) > product_cap:
        choices = [row for row in products if row.get("asin") != current]
        if not choices:
            break
        victim = min(choices, key=lambda row: (bool(row.get("selected_options")), int(row.get("last_access_step", 0)), row["asin"]))
        products.remove(victim)
        state["reviewed_product_archive"].append(
            {
                "asin": victim["asin"],
                "title": victim.get("title", ""),
                "price": victim.get("price", ""),
                "selection_status": victim.get("selection_status", "reviewed_not_selected"),
                "last_access_step": victim.get("last_access_step", 0),
            }
        )
    archive = state["reviewed_product_archive"]
    while len(archive) > archive_cap:
        archive.remove(min(archive, key=lambda row: (int(row.get("last_access_step", 0)), row["asin"])))


def _json_object(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return {str(key): str(item) for key, item in parsed.items()} if isinstance(parsed, Mapping) else {}


def _number_or_text(value: str | None) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return str(value)
