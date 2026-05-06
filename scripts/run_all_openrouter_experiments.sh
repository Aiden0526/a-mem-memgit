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

# ============================================================
# Collaborator-friendly OpenRouter experiment launcher
# ============================================================
# Edit CONFIG_API_KEY here or export OPENROUTER_API_KEY / OPENAI_API_KEY
# before running. The script bootstraps uv, creates .venv, installs the
# repo requirements, and then runs both LoCoMo and Persona experiments for
# baseline (robust) and patch (EvoMem) methods.

CONFIG_API_KEY=""
CONFIG_API_BASE="https://openrouter.ai/api/v1"
CONFIG_BACKEND="openrouter"
CONFIG_MODELS=(
  "moonshotai/kimi-k2.6"
  "qwen/qwen3.6-27b"
)

CONFIG_SUITE="all"          # all, locomo, persona
CONFIG_METHOD="both"        # robust, patch, both
CONFIG_RESUME="1"

# Python / environment bootstrap
CONFIG_VENV_DIR=".venv"
CONFIG_BOOTSTRAP_PYTHON="python3"
CONFIG_INSTALL_UV_IF_MISSING="1"
CONFIG_INSTALL_REQUIREMENTS="1"
CONFIG_REQUIREMENTS_FILE="requirements.txt"

# LoCoMo defaults
CONFIG_LOCOMO_DATASET="data/locomo10.json"
CONFIG_LOCOMO_RATIO="1.0"
CONFIG_LOCOMO_START_SAMPLE="0"
CONFIG_LOCOMO_END_SAMPLE=""
CONFIG_LOCOMO_BATCH="9"
CONFIG_LOCOMO_RETRIEVE_K="10"
CONFIG_LOCOMO_PATCH_TOP_K="2"
CONFIG_LOCOMO_PATCH_USAGE="always"
CONFIG_LOCOMO_TEMPERATURE_C5="0.5"
CONFIG_LOCOMO_RAW_LLM_LOG=""
CONFIG_LOCOMO_PATCH_CACHE_ROOT_BASE=""

# Persona defaults
CONFIG_PERSONA_BENCHMARK_FILE="data/Persona-release/benchmark_v34/text/benchmark_9p_ood_v34.csv"
CONFIG_PERSONA_ROOT=""
CONFIG_PERSONA_SIZE="32k"
CONFIG_PERSONA_BATCH="9"
CONFIG_PERSONA_RETRIEVE_K="10"
CONFIG_PERSONA_OUTPUT_DIR="results"
CONFIG_PERSONA_LITELLM_LOG="WARNING"
CONFIG_PERSONA_PERSONA_IDS=""
CONFIG_PERSONA_MAX_ITEMS=""
CONFIG_PERSONA_INCLUDE_DEBUG_COLUMNS="0"
CONFIG_PERSONA_CACHE_ROOT=""
CONFIG_PERSONA_PREFERENCE_AWARE="1"
CONFIG_PERSONA_PREFERENCE_AWARE_LEVEL="none"

# Persona patch defaults
CONFIG_PERSONA_PATCH_TOP_K="3"
CONFIG_PERSONA_PATCH_USAGE="always"
CONFIG_PERSONA_MIN_PATCH_SIMILARITY="0.4"
CONFIG_PERSONA_FORCE_REINGEST_PATCHES="0"
CONFIG_PERSONA_EXCLUDE_REVOKE_PATCHES="0"
CONFIG_PERSONA_EXCLUDE_ADD_PATCHES="0"
CONFIG_PERSONA_REQUIRE_PREF_CHANGE="0"
CONFIG_PERSONA_LLM_PATCH_FILTER="0"
CONFIG_PERSONA_GT_PATCH="0"
CONFIG_PERSONA_GT_PATCH_FILE=""
CONFIG_PERSONA_GT_PATCH_TOP_K=""
CONFIG_PERSONA_GT_PATCH_MIN_SIMILARITY=""
CONFIG_PERSONA_GP_PATCH_RETRIEVAL="similarity"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_all_openrouter_experiments.sh [all|locomo|persona] [robust|patch|both]

Defaults:
  suite  = CONFIG_SUITE (all)
  method = CONFIG_METHOD (both)

What this script does:
  1. Ensures uv is available
  2. Creates .venv with uv
  3. Installs requirements.txt with uv
  4. Runs LoCoMo and/or Persona experiments through the existing launchers

Recommended ways to provide the API key:
  1. Edit CONFIG_API_KEY at the top of this file, or
  2. export OPENROUTER_API_KEY=..., or
  3. export OPENAI_API_KEY=...

Useful overrides:
  MODEL_LIST                      Comma-separated models
  API_KEY / OPENROUTER_API_KEY    OpenRouter key
  API_BASE                        Defaults to https://openrouter.ai/api/v1
  RESUME                          1 to resume interrupted runs (default)
  VENV_DIR                        Virtualenv directory (default: .venv)
  INSTALL_REQUIREMENTS            1 to install/update requirements (default)

Examples:
  bash scripts/run_all_openrouter_experiments.sh
  bash scripts/run_all_openrouter_experiments.sh all both
  OPENROUTER_API_KEY=... bash scripts/run_all_openrouter_experiments.sh locomo both
  MODEL_LIST=moonshotai/kimi-k2.6 bash scripts/run_all_openrouter_experiments.sh persona patch
EOF
}

SUITE="${1:-$CONFIG_SUITE}"
METHOD="${2:-$CONFIG_METHOD}"

case "$SUITE" in
  all|locomo|persona) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

case "$METHOD" in
  robust|patch|both) ;;
  *)
    usage >&2
    exit 1
    ;;
esac

API_KEY="${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-${API_KEY:-$CONFIG_API_KEY}}}"
API_BASE="${OPENAI_BASE_URL:-${API_BASE:-$CONFIG_API_BASE}}"
BACKEND="${BACKEND:-$CONFIG_BACKEND}"
RESUME="${RESUME:-$CONFIG_RESUME}"
VENV_DIR="${VENV_DIR:-$CONFIG_VENV_DIR}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-$CONFIG_BOOTSTRAP_PYTHON}"
INSTALL_UV_IF_MISSING="${INSTALL_UV_IF_MISSING:-$CONFIG_INSTALL_UV_IF_MISSING}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-$CONFIG_INSTALL_REQUIREMENTS}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$CONFIG_REQUIREMENTS_FILE}"

MODELS=("${CONFIG_MODELS[@]}")
if [[ -n "${MODEL_LIST:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODEL_LIST"
fi

NORMALIZED_MODELS=()
for model in "${MODELS[@]}"; do
  model="${model#${model%%[![:space:]]*}}"
  model="${model%${model##*[![:space:]]}}"
  model="${model%,}"
  if [[ -n "$model" ]]; then
    NORMALIZED_MODELS+=("$model")
  fi
done
MODELS=("${NORMALIZED_MODELS[@]}")

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "At least one model is required. Edit CONFIG_MODELS or set MODEL_LIST." >&2
  exit 1
fi

if [[ -z "$API_KEY" ]]; then
  echo "OpenRouter API key is empty." >&2
  echo "Edit CONFIG_API_KEY in scripts/run_all_openrouter_experiments.sh, or export OPENROUTER_API_KEY / OPENAI_API_KEY." >&2
  exit 1
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "Requirements file not found: $REQUIREMENTS_FILE" >&2
  exit 1
fi

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$INSTALL_UV_IF_MISSING" != "1" && "$INSTALL_UV_IF_MISSING" != "true" && "$INSTALL_UV_IF_MISSING" != "TRUE" ]]; then
    echo "uv is not installed. Install uv first or set INSTALL_UV_IF_MISSING=1." >&2
    exit 1
  fi
  if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
    echo "Bootstrap python not found: $BOOTSTRAP_PYTHON" >&2
    exit 1
  fi
  echo "uv not found; installing it with $BOOTSTRAP_PYTHON -m pip install --user uv"
  "$BOOTSTRAP_PYTHON" -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to install uv automatically." >&2
    exit 1
  fi
}

bootstrap_venv() {
  ensure_uv
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR"
    uv venv "$VENV_DIR" --python "$BOOTSTRAP_PYTHON"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  if [[ "$INSTALL_REQUIREMENTS" == "1" || "$INSTALL_REQUIREMENTS" == "true" || "$INSTALL_REQUIREMENTS" == "TRUE" ]]; then
    echo "Installing Python requirements from $REQUIREMENTS_FILE"
    uv pip install -r "$REQUIREMENTS_FILE"
  fi
}

join_by_comma() {
  local IFS=','
  echo "$*"
}

print_header() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

print_config() {
  echo "Suite: $SUITE | Method: $METHOD | Resume: $RESUME"
  echo "Backend: $BACKEND"
  echo "API base: $API_BASE"
  echo "Venv: $VENV_DIR | Requirements: $REQUIREMENTS_FILE"
  echo "Models: ${MODELS[*]}"
}

run_locomo() {
  local model_csv
  model_csv="$(join_by_comma "${MODELS[@]}")"
  print_header "Running LoCoMo experiments"
  BACKEND="$BACKEND" \
  OPENAI_API_KEY="$API_KEY" \
  OPENROUTER_API_KEY="$API_KEY" \
  OPENAI_BASE_URL="$API_BASE" \
  MODEL_LIST="$model_csv" \
  DATASET="${DATASET:-$CONFIG_LOCOMO_DATASET}" \
  RATIO="${RATIO:-$CONFIG_LOCOMO_RATIO}" \
  START_SAMPLE="${START_SAMPLE:-$CONFIG_LOCOMO_START_SAMPLE}" \
  END_SAMPLE="${END_SAMPLE:-$CONFIG_LOCOMO_END_SAMPLE}" \
  BATCH="${BATCH:-$CONFIG_LOCOMO_BATCH}" \
  RETRIEVE_K="${RETRIEVE_K:-$CONFIG_LOCOMO_RETRIEVE_K}" \
  PATCH_TOP_K="${PATCH_TOP_K:-$CONFIG_LOCOMO_PATCH_TOP_K}" \
  PATCH_USAGE="${PATCH_USAGE:-$CONFIG_LOCOMO_PATCH_USAGE}" \
  TEMPERATURE_C5="${TEMPERATURE_C5:-$CONFIG_LOCOMO_TEMPERATURE_C5}" \
  RAW_LLM_LOG="${RAW_LLM_LOG:-$CONFIG_LOCOMO_RAW_LLM_LOG}" \
  PATCH_CACHE_ROOT_BASE="${PATCH_CACHE_ROOT_BASE:-$CONFIG_LOCOMO_PATCH_CACHE_ROOT_BASE}" \
  RESUME="$RESUME" \
  bash scripts/run_locomo_baseline_patch_3models.sh "$METHOD"
}

run_persona() {
  local model_csv
  model_csv="$(join_by_comma "${MODELS[@]}")"
  print_header "Running Persona experiments"
  BACKEND="$BACKEND" \
  OPENAI_API_KEY="$API_KEY" \
  OPENROUTER_API_KEY="$API_KEY" \
  OPENAI_BASE_URL="$API_BASE" \
  MODEL_LIST="$model_csv" \
  BENCHMARK_FILE="${BENCHMARK_FILE:-$CONFIG_PERSONA_BENCHMARK_FILE}" \
  PERSONA_ROOT="${PERSONA_ROOT:-$CONFIG_PERSONA_ROOT}" \
  SIZE="${SIZE:-$CONFIG_PERSONA_SIZE}" \
  BATCH="${BATCH:-$CONFIG_PERSONA_BATCH}" \
  RETRIEVE_K="${RETRIEVE_K:-$CONFIG_PERSONA_RETRIEVE_K}" \
  OUTPUT_DIR="${OUTPUT_DIR:-$CONFIG_PERSONA_OUTPUT_DIR}" \
  LITELLM_LOG="${LITELLM_LOG:-$CONFIG_PERSONA_LITELLM_LOG}" \
  PERSONA_IDS="${PERSONA_IDS:-$CONFIG_PERSONA_PERSONA_IDS}" \
  MAX_ITEMS="${MAX_ITEMS:-$CONFIG_PERSONA_MAX_ITEMS}" \
  INCLUDE_DEBUG_COLUMNS="${INCLUDE_DEBUG_COLUMNS:-$CONFIG_PERSONA_INCLUDE_DEBUG_COLUMNS}" \
  CACHE_ROOT="${CACHE_ROOT:-$CONFIG_PERSONA_CACHE_ROOT}" \
  PREFERENCE_AWARE="${PREFERENCE_AWARE:-$CONFIG_PERSONA_PREFERENCE_AWARE}" \
  PREFERENCE_AWARE_LEVEL="${PREFERENCE_AWARE_LEVEL:-$CONFIG_PERSONA_PREFERENCE_AWARE_LEVEL}" \
  PATCH_TOP_K="${PATCH_TOP_K:-$CONFIG_PERSONA_PATCH_TOP_K}" \
  PATCH_USAGE="${PATCH_USAGE:-$CONFIG_PERSONA_PATCH_USAGE}" \
  MIN_PATCH_SIMILARITY="${MIN_PATCH_SIMILARITY:-$CONFIG_PERSONA_MIN_PATCH_SIMILARITY}" \
  FORCE_REINGEST_PATCHES="${FORCE_REINGEST_PATCHES:-$CONFIG_PERSONA_FORCE_REINGEST_PATCHES}" \
  EXCLUDE_REVOKE_PATCHES="${EXCLUDE_REVOKE_PATCHES:-$CONFIG_PERSONA_EXCLUDE_REVOKE_PATCHES}" \
  EXCLUDE_ADD_PATCHES="${EXCLUDE_ADD_PATCHES:-$CONFIG_PERSONA_EXCLUDE_ADD_PATCHES}" \
  REQUIRE_PREF_CHANGE="${REQUIRE_PREF_CHANGE:-$CONFIG_PERSONA_REQUIRE_PREF_CHANGE}" \
  LLM_PATCH_FILTER="${LLM_PATCH_FILTER:-$CONFIG_PERSONA_LLM_PATCH_FILTER}" \
  GT_PATCH="${GT_PATCH:-$CONFIG_PERSONA_GT_PATCH}" \
  GT_PATCH_FILE="${GT_PATCH_FILE:-$CONFIG_PERSONA_GT_PATCH_FILE}" \
  GT_PATCH_TOP_K="${GT_PATCH_TOP_K:-$CONFIG_PERSONA_GT_PATCH_TOP_K}" \
  GT_PATCH_MIN_SIMILARITY="${GT_PATCH_MIN_SIMILARITY:-$CONFIG_PERSONA_GT_PATCH_MIN_SIMILARITY}" \
  GP_PATCH_RETRIEVAL="${GP_PATCH_RETRIEVAL:-$CONFIG_PERSONA_GP_PATCH_RETRIEVAL}" \
  RESUME="$RESUME" \
  bash scripts/run_persona_baseline_patch.sh "$METHOD"
}

bootstrap_venv
print_config

case "$SUITE" in
  all)
    run_locomo
    run_persona
    ;;
  locomo)
    run_locomo
    ;;
  persona)
    run_persona
    ;;
esac
