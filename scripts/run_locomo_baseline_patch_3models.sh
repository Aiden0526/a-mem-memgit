#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# =========================
# Editable experiment config
# =========================
# Change these values directly if you want a single file that is ready to run
# after download. Environment variables still override them.
CONFIG_API_KEY="sk-tBM2jk6fMYPNFKsBTcIQV61m2eOxeev1YxP360IpXUhhqSJu"
CONFIG_API_BASE="https://app.ppapi.ai/v1/chat/completions"
CONFIG_BACKEND="openai"
CONFIG_DATASET="data/locomo10.json"
CONFIG_RATIO="0.1"
CONFIG_START_SAMPLE="0"
CONFIG_END_SAMPLE=""
CONFIG_BATCH="1"
CONFIG_RETRIEVE_K="10"
CONFIG_PATCH_TOP_K="2"
CONFIG_PATCH_USAGE="always"
CONFIG_TEMPERATURE_C5="0.5"
CONFIG_RUN_TARGET="patch" # robust, patch, or both
CONFIG_MODELS=(
  "gpt-5.4"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_locomo_baseline_patch_3models.sh [robust|patch|both]

If no argument is passed, CONFIG_RUN_TARGET is used.

Edit the CONFIG_* values at the top of this file for the simplest workflow.
Set CONFIG_RUN_TARGET to `robust`, `patch`, or `both`.
Environment variables can also override those defaults.

Supported overrides:
  OPENAI_API_KEY   API key for the OpenAI-compatible endpoint
  OPENAI_BASE_URL  Base URL or full chat/completions URL
  MODEL_LIST       Comma-separated models; overrides CONFIG_MODELS
  DATASET          Dataset path
  RATIO            Dataset ratio after load
  START_SAMPLE     Start index after ratio filtering
  END_SAMPLE       Optional exclusive end index
  BATCH            Worker count
  RETRIEVE_K       Retrieval top-k for both runners
  PATCH_TOP_K      Patch retrieval top-k
  PATCH_USAGE      Patch mode
  TEMPERATURE_C5   Category-5 temperature override

Examples:
  bash scripts/run_locomo_baseline_patch_3models.sh both
  OPENAI_API_KEY=... OPENAI_BASE_URL=https://app.ppapi.ai/v1/chat/completions   MODEL_LIST=gpt-5.4-mini,gemini-3.1-pro-preview,gpt-4.1-mini   bash scripts/run_locomo_baseline_patch_3models.sh patch
EOF
}

RUN_TARGET="${1:-$CONFIG_RUN_TARGET}"
case "$RUN_TARGET" in
  robust|patch|both) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

API_KEY="${OPENAI_API_KEY:-${API_KEY:-$CONFIG_API_KEY}}"
API_BASE="${OPENAI_BASE_URL:-${API_BASE:-$CONFIG_API_BASE}}"
BACKEND="${BACKEND:-$CONFIG_BACKEND}"
DATASET="${DATASET:-$CONFIG_DATASET}"
RATIO="${RATIO:-$CONFIG_RATIO}"
START_SAMPLE="${START_SAMPLE:-$CONFIG_START_SAMPLE}"
END_SAMPLE="${END_SAMPLE:-$CONFIG_END_SAMPLE}"
BATCH="${BATCH:-$CONFIG_BATCH}"
RETRIEVE_K="${RETRIEVE_K:-$CONFIG_RETRIEVE_K}"
PATCH_TOP_K="${PATCH_TOP_K:-$CONFIG_PATCH_TOP_K}"
PATCH_USAGE="${PATCH_USAGE:-$CONFIG_PATCH_USAGE}"
TEMPERATURE_C5="${TEMPERATURE_C5:-$CONFIG_TEMPERATURE_C5}"

MODELS=("${CONFIG_MODELS[@]}")
if [[ -n "${MODEL_LIST:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODEL_LIST"
fi

if [[ -z "$API_KEY" ]]; then
  echo "Set CONFIG_API_KEY in this script, or export OPENAI_API_KEY/API_KEY." >&2
  exit 1
fi

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "At least one model is required. Edit CONFIG_MODELS or set MODEL_LIST accordingly." >&2
  exit 1
fi

if [[ "${#MODELS[@]}" -eq 1 && -z "${MODELS[0]}" ]]; then
  echo "At least one non-empty model is required. Edit CONFIG_MODELS or set MODEL_LIST accordingly." >&2
  exit 1
fi

mkdir -p robust_results patch_results logs

sanitize() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  value="${value//./_}"
  echo "$value"
}

build_common_args() {
  local model="$1"
  local -n out_ref=$2
  out_ref=(
    --backend "$BACKEND"
    --model "$model"
    --api_key "$API_KEY"
    --api_base "$API_BASE"
    --dataset "$DATASET"
    --ratio "$RATIO"
    --start_sample "$START_SAMPLE"
    --temperature_c5 "$TEMPERATURE_C5"
  )

  if [[ -n "$END_SAMPLE" ]]; then
    out_ref+=(--end_sample "$END_SAMPLE")
  fi

  if [[ -n "$BATCH" ]]; then
    out_ref+=(--batch "$BATCH")
  fi
}

print_resolved_config() {
  echo "Run target: $RUN_TARGET"
  echo "API base: $API_BASE"
  echo "Dataset: $DATASET"
  echo "Ratio: $RATIO | start_sample: $START_SAMPLE | end_sample: ${END_SAMPLE:-NONE} | batch: $BATCH"
  echo "retrieve_k: $RETRIEVE_K | patch_top_k: $PATCH_TOP_K | patch_usage: $PATCH_USAGE | temperature_c5: $TEMPERATURE_C5"
  echo "Models: ${MODELS[*]}"
}

print_resolved_config

for model in "${MODELS[@]}"; do
  safe_model="$(sanitize "$model")"
  common_args=()
  build_common_args "$model" common_args

  echo "============================================================"
  echo "Model: $model"

  if [[ "$RUN_TARGET" == "robust" || "$RUN_TARGET" == "both" ]]; then
    robust_output="robust_results/locomo_robust_${safe_model}.json"
    robust_cmd=(
      python test_advanced_robust.py
      "${common_args[@]}"
      --retrieve_k "$RETRIEVE_K"
      --output "$robust_output"
    )
    echo "Running robust baseline -> $robust_output"
    "${robust_cmd[@]}"
  fi

  if [[ "$RUN_TARGET" == "patch" || "$RUN_TARGET" == "both" ]]; then
    patch_output="patch_results/locomo_patch_${safe_model}.json"
    patch_cmd=(
      python test_advanced_patch.py
      "${common_args[@]}"
      --retrieve_k "$RETRIEVE_K"
      --patch_top_k "$PATCH_TOP_K"
      --patch_usage "$PATCH_USAGE"
      --output "$patch_output"
    )
    echo "Running patch variant -> $patch_output"
    "${patch_cmd[@]}"
  fi
done
