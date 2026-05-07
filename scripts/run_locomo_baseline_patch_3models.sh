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
CONFIG_API_KEY=
CONFIG_API_BASE=https://openrouter.ai/api/v1
CONFIG_BACKEND="openai"
CONFIG_DATASET="data/locomo10.json"
CONFIG_RATIO="1.0"
CONFIG_START_SAMPLE="0"
CONFIG_END_SAMPLE=""
CONFIG_SAMPLE_IDS=""
CONFIG_BATCH="9"
CONFIG_RETRIEVE_K="10"
CONFIG_PATCH_TOP_K="2"
CONFIG_PATCH_USAGE="always"
CONFIG_TEMPERATURE_C5="0.5"
CONFIG_RAW_LLM_LOG="logs/gpt55_robust_raw.log"
CONFIG_RUN_TARGET="patch" # robust, patch, or both
CONFIG_RESUME="1"
CONFIG_PATCH_CACHE_ROOT_BASE=""
CONFIG_MODELS=(
  # "qwen/qwen3.6-27b"
  "moonshotai/kimi-k2.6"
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
  SAMPLE_IDS       Optional comma-separated dataset sample ids
  BATCH            Worker count
  RETRIEVE_K       Retrieval top-k for both runners
  PATCH_TOP_K      Patch retrieval top-k
  PATCH_USAGE      Patch mode
  TEMPERATURE_C5   Category-5 temperature override
  RESUME           1 to automatically resume from existing outputs/caches
  PATCH_CACHE_ROOT_BASE Optional base directory for patch caches; per-model subdirs are derived automatically
  RAW_LLM_LOG      Optional raw prompt/response log file for robust runs

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
SAMPLE_IDS="${SAMPLE_IDS:-$CONFIG_SAMPLE_IDS}"
BATCH="${BATCH:-$CONFIG_BATCH}"
RETRIEVE_K="${RETRIEVE_K:-$CONFIG_RETRIEVE_K}"
PATCH_TOP_K="${PATCH_TOP_K:-$CONFIG_PATCH_TOP_K}"
PATCH_USAGE="${PATCH_USAGE:-$CONFIG_PATCH_USAGE}"
TEMPERATURE_C5="${TEMPERATURE_C5:-$CONFIG_TEMPERATURE_C5}"
RAW_LLM_LOG="${RAW_LLM_LOG:-$CONFIG_RAW_LLM_LOG}"
RESUME="${RESUME:-$CONFIG_RESUME}"
PATCH_CACHE_ROOT_BASE="${PATCH_CACHE_ROOT_BASE:-$CONFIG_PATCH_CACHE_ROOT_BASE}"

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

ensure_resume_ready() {
  local output_path="$1"
  local label="$2"

  if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE" ]]; then
    if [[ -f "$output_path" ]]; then
      echo "Resume enabled: reusing existing $label output $output_path"
    else
      echo "Resume enabled: starting fresh $label output $output_path"
    fi
    return 0
  fi

  if [[ -f "$output_path" ]]; then
    echo "$label output already exists and RESUME=0: $output_path" >&2
    echo "Either set RESUME=1 to continue, or change the model / output target." >&2
    exit 1
  fi
}

sanitize() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  value="${value//./_}"
  echo "$value"
}

set_common_args() {
  local model="$1"
  COMMON_ARGS=(
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
    COMMON_ARGS+=(--end_sample "$END_SAMPLE")
  fi
  if [[ -n "$SAMPLE_IDS" ]]; then
    COMMON_ARGS+=(--sample-ids "$SAMPLE_IDS")
  fi

  if [[ -n "$BATCH" ]]; then
    COMMON_ARGS+=(--batch "$BATCH")
  fi
}

selection_suffix() {
  local suffix=""
  if [[ -n "$SAMPLE_IDS" ]]; then
    local safe_ids="${SAMPLE_IDS//,/ _}"
    safe_ids="${safe_ids// /_}"
    safe_ids="${safe_ids//[^A-Za-z0-9_]/_}"
    safe_ids="${safe_ids//__/_}"
    suffix="_samples_${safe_ids}"
  elif [[ "$RATIO" != "1.0" || "$START_SAMPLE" != "0" || -n "$END_SAMPLE" ]]; then
    local safe_ratio
    safe_ratio="$(sanitize "$RATIO")"
    local safe_start
    safe_start="$(sanitize "$START_SAMPLE")"
    local safe_end="end"
    if [[ -n "$END_SAMPLE" ]]; then
      safe_end="$(sanitize "$END_SAMPLE")"
    fi
    suffix="_ratio${safe_ratio}_range${safe_start}_${safe_end}"
  fi
  printf '%s' "$suffix"
}

print_resolved_config() {
  echo "Run target: $RUN_TARGET"
  echo "API base: $API_BASE"
  echo "Dataset: $DATASET"
  echo "Ratio: $RATIO | start_sample: $START_SAMPLE | end_sample: ${END_SAMPLE:-NONE} | sample_ids: ${SAMPLE_IDS:-ALL} | batch: $BATCH"
  echo "retrieve_k: $RETRIEVE_K | patch_top_k: $PATCH_TOP_K | patch_usage: $PATCH_USAGE | temperature_c5: $TEMPERATURE_C5"
  if [[ -n "$RAW_LLM_LOG" ]]; then
    echo "raw_llm_log: $RAW_LLM_LOG"
  fi
  echo "resume: $RESUME | patch_cache_root_base: ${PATCH_CACHE_ROOT_BASE:-AUTO}"
  echo "Models: ${MODELS[*]}"
}

print_resolved_config

for model in "${MODELS[@]}"; do
  patch_cache_root=""
  safe_model="$(sanitize "$model")"
  output_suffix="$(selection_suffix)"
  COMMON_ARGS=()
  set_common_args "$model"
  if [[ -n "$PATCH_CACHE_ROOT_BASE" ]]; then
    mkdir -p "$PATCH_CACHE_ROOT_BASE"
    patch_cache_root="$PATCH_CACHE_ROOT_BASE/locomo_patch_${safe_model}"
  fi

  echo "============================================================"
  echo "Model: $model"

  if [[ "$RUN_TARGET" == "robust" || "$RUN_TARGET" == "both" ]]; then
    robust_output="robust_results/locomo_robust_${safe_model}${output_suffix}.json"
    robust_cmd=(
      python test_advanced_robust.py
      "${COMMON_ARGS[@]}"
      --retrieve_k "$RETRIEVE_K"
      --output "$robust_output"
    )
    if [[ -n "$RAW_LLM_LOG" ]]; then
      robust_cmd+=(--raw_llm_log "$RAW_LLM_LOG")
    fi
    ensure_resume_ready "$robust_output" "LoCoMo robust"
    echo "Running robust baseline -> $robust_output"
    "${robust_cmd[@]}"
  fi

  if [[ "$RUN_TARGET" == "patch" || "$RUN_TARGET" == "both" ]]; then
    patch_output="patch_results/locomo_patch_${safe_model}${output_suffix}.json"
    patch_cmd=(
      python test_advanced_patch.py
      "${COMMON_ARGS[@]}"
      --retrieve_k "$RETRIEVE_K"
      --patch_top_k "$PATCH_TOP_K"
      --patch_usage "$PATCH_USAGE"
      --output "$patch_output"
    )
    ensure_resume_ready "$patch_output" "LoCoMo patch"
    if [[ -n "$patch_cache_root" ]]; then
      patch_cmd+=(--cache_root "$patch_cache_root")
    fi
    echo "Running patch variant -> $patch_output"
    "${patch_cmd[@]}"
  fi
done
