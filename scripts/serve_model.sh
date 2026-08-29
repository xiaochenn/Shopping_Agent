#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-Qwen/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
LLM_PORT="${LLM_PORT:-8000}"

# FlashInfer may JIT-compile a sampler extension while vLLM starts.  The
# project-managed ninja executable therefore must be visible to EngineCore,
# which is spawned by the venv's vLLM entry point.
export PATH="$ROOT/.venv/bin:$PATH"

if [[ ! -x "$ROOT/.venv/bin/vllm" ]]; then
  echo "vLLM is not installed. Run: bash scripts/setup.sh" >&2
  exit 1
fi

exec "$ROOT/.venv/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$LLM_PORT" \
  --max-model-len 24576 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
