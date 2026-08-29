#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-model}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
CONTEXT_INPUT_BUDGET="${CONTEXT_INPUT_BUDGET:-16384}"

# Both default endpoints are local services.  Explicitly bypass any shell-wide
# proxy so Python's urllib token-counter reaches vLLM instead of the proxy.
LOCAL_NO_PROXY="127.0.0.1,localhost,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --summary "$OUTPUT_DIR/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --context-input-budget "$CONTEXT_INPUT_BUDGET" \
  --result-clearing \
  --result-keep-recent-groups 3

"$ROOT/.venv/bin/python" scripts/build_eval_report.py --run-dir "$OUTPUT_DIR"
