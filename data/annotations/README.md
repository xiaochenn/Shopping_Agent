# Budget semantics annotations

`budget_semantics_v1.jsonl` is a task-level sidecar annotation file. It never
modifies the embedded ShopSimulator archive. Each row is keyed by the frozen
runtime `task_id`, gold `asin`, and SHA-256 of the source instruction.

Generate the deterministic first tier with:

```bash
uv run python scripts/generate_budget_semantics.py
```

Tier 1 accepts only explicit price constraints with high-confidence regular
expressions. Approximate budgets become `approximate_band`: the lower delta is
`max(10 yuan, 10% of target)` and the upper delta is `max(10 yuan, 5% of
target)`. A stated range followed by “左右/上下” becomes `approximate_range`:
the same policy expands its stated lower and upper endpoints. Explicit
“超过/以上/起” is a hard `lower_bound`; a bare “预算 + 金额” is a hard upper
bound. Ambiguous price language is emitted to
`budget_semantics_v1_needs_llm.jsonl` for the Tier-2 semantic model and then
human review; it must not silently become a hard budget constraint.

Run Tier 2 against the configured DeepSeek endpoint in a small audited batch:

```bash
python scripts/label_budget_semantics_llm.py --limit 30
```

It reads `OPENCODE_URL`/`OPENCODE_API_KEY` or
`DEEPSEEK_URL`/`DEEPSEEK_API_KEY` from `.env`, writes only the model's
structured labels to `budget_semantics_v1_llm.jsonl`, and can resume without
reprocessing completed task IDs. Add `--selection-seed 20260826` for a
deterministic random audit sample rather than the queue's task-ID order.

For unresolved `unknown` rows, generate a separate human-review queue:

```bash
python scripts/prepare_budget_semantics_manual_review.py
```

The resulting JSONL preserves the source instruction and model result, while
leaving the `human_*` decision fields blank. It is also a sidecar file and does
not modify ShopSimulator task data.

Review that queue interactively with:

```bash
PYTHONPATH=. .venv/bin/python scripts/review_budget_semantics.py --reviewer pyc
```

The terminal program shows one source prompt at a time, offers the label menu,
calculates the effective range for approximate labels, and immediately saves
each decision to `budget_semantics_v1_manual_decisions.jsonl`. Re-running it
skips saved task IDs. Use `--task-id 187` to review just one task.

After all manual decisions are complete, merge the four stages into one
validated task-level sidecar file:

```bash
PYTHONPATH=. .venv/bin/python scripts/merge_budget_semantics.py
```

This writes `budget_semantics_v1_merged.jsonl` and companion metadata. It
rejects incomplete human coverage, mismatched source hashes, invalid evidence,
and invalid numeric constraints; it never changes the source task archive.
Human-confirmed `unknown` budget semantics are excluded from the usable merged
label set and recorded in `budget_semantics_v1_excluded_human_unknown.jsonl`.

## Runtime harness

`start.sh` loads the merged sidecar at startup and makes it part of every
compiled goal. The source task is checked against the sidecar's ASIN and
instruction SHA-256; a missing or mismatched label stops startup/goal loading
instead of falling back to the historical price regex. Reward v3 then treats
`price_lower` and `price_upper` as inclusive hard bounds when they are present.
The two human-confirmed `unknown` tasks retain their source position but are
explicitly marked `budget_eligible=false`; they must not be selected for future
teacher-data collection or training.
