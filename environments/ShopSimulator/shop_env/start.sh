#!/usr/bin/env bash
set -euo pipefail

SHOP_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_PATH="${SHOP_SEARCH_INDEX:-${SHOP_ENV_ROOT}/search_engine/products.sqlite3}"
BUDGET_LABELS="${SHOP_BUDGET_SEMANTICS:-${SHOP_ENV_ROOT}/../../../data/annotations/budget_semantics_v1_merged.jsonl}"
BUDGET_EXCLUDED="${SHOP_BUDGET_SEMANTICS_EXCLUDED:-${SHOP_ENV_ROOT}/../../../data/annotations/budget_semantics_v1_excluded_human_unknown.jsonl}"

if [[ ! -f "${INDEX_PATH}" ]]; then
  echo "ShopSimulator index is missing: ${INDEX_PATH}" >&2
  echo "Run from the repository root: bash scripts/setup.sh" >&2
  exit 1
fi

if [[ ! -f "${BUDGET_LABELS}" || ! -f "${BUDGET_EXCLUDED}" ]]; then
  echo "Frozen budget-semantics sidecars are missing." >&2
  echo "Expected labels: ${BUDGET_LABELS}" >&2
  echo "Expected exclusions: ${BUDGET_EXCLUDED}" >&2
  exit 1
fi

export SHOP_ENVIRONMENT_VERSION=shopsimulator-environment-v2.1
export SHOP_ENV_CONFIG="${SHOP_ENV_CONFIG:-${SHOP_ENV_ROOT}/configs/environment.json}"
export SHOP_SEARCH_INDEX="${INDEX_PATH}"
export SHOP_BUDGET_SEMANTICS="${BUDGET_LABELS}"
export SHOP_BUDGET_SEMANTICS_EXCLUDED="${BUDGET_EXCLUDED}"
export SHOP_MAX_STEPS="${SHOP_MAX_STEPS:-35}"
export SHOPSIM_ENV_SLOTS="${SHOPSIM_ENV_SLOTS:-8}"
export SHOPSIM_PORT="${SHOPSIM_PORT:-5700}"

cd "${SHOP_ENV_ROOT}/shop_env"
exec python pack_api.py
