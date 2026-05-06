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

CONFIG_API_KEY=""
CONFIG_API_BASE="https://openrouter.ai/api/v1"
CONFIG_BACKEND="openrouter"
CONFIG_LOCOMO_MODELS=(
  "moonshotai/kimi-k2.6"
  "qwen/qwen3.6-27b"
)
CONFIG_PERSONA_MODELS=(
  "moonshotai/kimi-k2.6"
  "qwen/qwen3.6-27b"
  "google/gemini-3.1-pro-preview"
)

CONFIG_SUITE="all"
CONFIG_METHOD="both"
CONFIG_RESUME="1"
CONFIG_EXECUTION_MODE="parallel"   # sequential, parallel, or tmux
CONFIG_JOB_NAME_PREFIX=""
CONFIG_JOB_LOG_DIR="logs/parallel_jobs"
CONFIG_JOB_PID_DIR="logs/parallel_pids"

CONFIG_VENV_DIR=".venv"
CONFIG_BOOTSTRAP_PYTHON="python3"
CONFIG_INSTALL_UV_IF_MISSING="1"
CONFIG_INSTALL_REQUIREMENTS="1"
CONFIG_REINSTALL_REQUIREMENTS="0"
CONFIG_REQUIREMENTS_FILE="requirements.txt"

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

CONFIG_PERSONA_DATASET_URL="https://huggingface.co/datasets/Aiden0526/PersonaMem-v2-enhanced-release/resolve/main/PersonaMem-v2-enhanced-release.zip"
CONFIG_PERSONA_DATASET_ARCHIVE="data/PersonaMem-v2-enhanced-release.zip"
CONFIG_PERSONA_DATASET_DIR="data/PersonaMem-v2-enhanced-release"
CONFIG_PERSONA_BENCHMARK_FILE="data/PersonaMem-v2-enhanced-release/benchmark_v34/text/benchmark_9p_ood_v34.csv"
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

Useful overrides:
  MODEL_LIST                      Comma-separated models for both suites
  LOCOMO_MODEL_LIST               Comma-separated models for LoCoMo only
  PERSONA_MODEL_LIST              Comma-separated models for Persona only
  OPENROUTER_API_KEY              OpenRouter key
  OPENAI_API_KEY                  Alternative env var for the same key
  API_BASE                        Defaults to https://openrouter.ai/api/v1
  EXECUTION_MODE                  sequential, parallel, or tmux (default: parallel)
  JOB_NAME_PREFIX                 Optional prefix for background/tmux job names
  JOB_LOG_DIR                     Log directory for background/tmux jobs
  JOB_PID_DIR                     PID directory for background jobs
  REINSTALL_REQUIREMENTS          1 to force reinstall even if .venv already exists

Examples:
  bash scripts/run_all_openrouter_experiments.sh
  OPENROUTER_API_KEY=... bash scripts/run_all_openrouter_experiments.sh locomo both
  EXECUTION_MODE=sequential bash scripts/run_all_openrouter_experiments.sh persona patch
EOF
}

SUITE="${1:-$CONFIG_SUITE}"
METHOD="${2:-$CONFIG_METHOD}"
EXECUTION_MODE="${EXECUTION_MODE:-$CONFIG_EXECUTION_MODE}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-$CONFIG_JOB_NAME_PREFIX}"
JOB_LOG_DIR="${JOB_LOG_DIR:-$CONFIG_JOB_LOG_DIR}"
JOB_PID_DIR="${JOB_PID_DIR:-$CONFIG_JOB_PID_DIR}"

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

case "$EXECUTION_MODE" in
  sequential|parallel|tmux) ;;
  *)
    echo "Unsupported EXECUTION_MODE: $EXECUTION_MODE" >&2
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
REINSTALL_REQUIREMENTS="${REINSTALL_REQUIREMENTS:-$CONFIG_REINSTALL_REQUIREMENTS}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$CONFIG_REQUIREMENTS_FILE}"

LOCOMO_MODELS=("${CONFIG_LOCOMO_MODELS[@]}")
PERSONA_MODELS=("${CONFIG_PERSONA_MODELS[@]}")

trim_model() {
  local model="$1"
  model="${model#${model%%[![:space:]]*}}"
  model="${model%${model##*[![:space:]]}}"
  model="${model%,}"
  printf '%s' "$model"
}

set_models_from_csv() {
  local target="$1"
  local csv="$2"
  local old_ifs="$IFS"
  local parsed=()
  local model
  IFS=','
  read -r -a parsed <<< "$csv"
  IFS="$old_ifs"

  case "$target" in
    LOCOMO_MODELS) LOCOMO_MODELS=() ;;
    PERSONA_MODELS) PERSONA_MODELS=() ;;
    *)
      echo "Unsupported target array: $target" >&2
      exit 1
      ;;
  esac

  for model in "${parsed[@]}"; do
    model="$(trim_model "$model")"
    if [[ -n "$model" ]]; then
      case "$target" in
        LOCOMO_MODELS) LOCOMO_MODELS+=("$model") ;;
        PERSONA_MODELS) PERSONA_MODELS+=("$model") ;;
      esac
    fi
  done
}

if [[ -n "${MODEL_LIST:-}" ]]; then
  set_models_from_csv LOCOMO_MODELS "$MODEL_LIST"
  set_models_from_csv PERSONA_MODELS "$MODEL_LIST"
fi
if [[ -n "${LOCOMO_MODEL_LIST:-}" ]]; then
  set_models_from_csv LOCOMO_MODELS "$LOCOMO_MODEL_LIST"
fi
if [[ -n "${PERSONA_MODEL_LIST:-}" ]]; then
  set_models_from_csv PERSONA_MODELS "$PERSONA_MODEL_LIST"
fi

if [[ "$SUITE" != "persona" && "${#LOCOMO_MODELS[@]}" -eq 0 ]]; then
  echo "At least one LoCoMo model is required." >&2
  exit 1
fi
if [[ "$SUITE" != "locomo" && "${#PERSONA_MODELS[@]}" -eq 0 ]]; then
  echo "At least one Persona model is required." >&2
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
  local created_venv=0
  ensure_uv
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR"
    uv venv "$VENV_DIR" --python "$BOOTSTRAP_PYTHON"
    created_venv=1
  else
    echo "Reusing existing virtual environment at $VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  if [[ "$INSTALL_REQUIREMENTS" != "1" && "$INSTALL_REQUIREMENTS" != "true" && "$INSTALL_REQUIREMENTS" != "TRUE" ]]; then
    return 0
  fi

  if [[ "$created_venv" == "1" || "$REINSTALL_REQUIREMENTS" == "1" || "$REINSTALL_REQUIREMENTS" == "true" || "$REINSTALL_REQUIREMENTS" == "TRUE" ]]; then
    echo "Installing Python requirements from $REQUIREMENTS_FILE"
    uv pip install -r "$REQUIREMENTS_FILE"
  else
    echo "Skipping requirement install because $VENV_DIR already exists"
    echo "Set REINSTALL_REQUIREMENTS=1 to force reinstall requirements"
  fi
}

ensure_persona_dataset() {
  local requested_benchmark="$1"
  local dataset_url="${PERSONA_DATASET_URL:-$CONFIG_PERSONA_DATASET_URL}"
  local archive_path="${PERSONA_DATASET_ARCHIVE:-$CONFIG_PERSONA_DATASET_ARCHIVE}"
  local dataset_dir="${PERSONA_DATASET_DIR:-$CONFIG_PERSONA_DATASET_DIR}"
  local default_benchmark="$dataset_dir/benchmark_v34/text/benchmark_9p_ood_v34.csv"

  if [[ -f "$requested_benchmark" || -f "$default_benchmark" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$archive_path")"

  if [[ ! -f "$archive_path" ]]; then
    echo "PersonaMem dataset not found locally. Downloading to $archive_path"
    if command -v wget >/dev/null 2>&1; then
      if ! wget -O "$archive_path" "$dataset_url"; then
        echo "wget failed; trying curl instead"
        rm -f "$archive_path"
        if ! command -v curl >/dev/null 2>&1; then
          echo "curl is not installed, and wget download failed." >&2
          exit 1
        fi
        curl -L --fail -o "$archive_path" "$dataset_url"
      fi
    else
      echo "wget not found; trying curl instead"
      if ! command -v curl >/dev/null 2>&1; then
        echo "Neither wget nor curl is installed, so PersonaMem cannot be downloaded automatically." >&2
        exit 1
      fi
      curl -L --fail -o "$archive_path" "$dataset_url"
    fi
  fi

  if ! command -v unzip >/dev/null 2>&1; then
    echo "unzip is required to extract PersonaMem but was not found." >&2
    exit 1
  fi

  echo "Extracting PersonaMem dataset archive"
  unzip -oq "$archive_path" -d "$(dirname "$archive_path")"

  if [[ ! -f "$requested_benchmark" && ! -f "$default_benchmark" ]]; then
    echo "PersonaMem dataset extraction completed, but the benchmark file is still missing." >&2
    echo "Checked: $requested_benchmark" >&2
    echo "Checked: $default_benchmark" >&2
    exit 1
  fi
}

prepare_requested_datasets() {
  local benchmark_file
  local default_benchmark
  if [[ "$SUITE" != "all" && "$SUITE" != "persona" ]]; then
    return 0
  fi

  default_benchmark="${PERSONA_DATASET_DIR:-$CONFIG_PERSONA_DATASET_DIR}/benchmark_v34/text/benchmark_9p_ood_v34.csv"
  benchmark_file="${BENCHMARK_FILE:-$CONFIG_PERSONA_BENCHMARK_FILE}"
  echo "Checking PersonaMem dataset before launching jobs"
  ensure_persona_dataset "$benchmark_file"
  if [[ ! -f "$benchmark_file" && -f "$default_benchmark" ]]; then
    export BENCHMARK_FILE="$default_benchmark"
  fi
}

join_by_comma() {
  local old_ifs="$IFS"
  IFS=','
  printf '%s' "$*"
  IFS="$old_ifs"
}

print_header() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

print_config() {
  echo "Suite: $SUITE | Method: $METHOD | Resume: $RESUME"
  echo "Execution mode: $EXECUTION_MODE"
  echo "Backend: $BACKEND"
  echo "API base: $API_BASE"
  echo "Venv: $VENV_DIR | Requirements: $REQUIREMENTS_FILE | reinstall_requirements: $REINSTALL_REQUIREMENTS"
  echo "Persona benchmark default: ${BENCHMARK_FILE:-$CONFIG_PERSONA_BENCHMARK_FILE}"
  if [[ "$EXECUTION_MODE" != "sequential" ]]; then
    echo "job_name_prefix: ${JOB_NAME_PREFIX:-NONE} | job_log_dir: $JOB_LOG_DIR"
    if [[ "$EXECUTION_MODE" == "parallel" ]]; then
      echo "job_pid_dir: $JOB_PID_DIR"
    fi
  fi
  if [[ "$SUITE" == "all" || "$SUITE" == "locomo" ]]; then
    echo "LoCoMo models: ${LOCOMO_MODELS[*]}"
  fi
  if [[ "$SUITE" == "all" || "$SUITE" == "persona" ]]; then
    echo "Persona models: ${PERSONA_MODELS[*]}"
  fi
}

shell_quote() {
  printf '%q' "$1"
}

model_alias() {
  local model="$1"
  local alias
  case "$model" in
    *kimi*) alias="kimi" ;;
    *qwen*) alias="qwen" ;;
    *gemini*) alias="gemini" ;;
    *)
      alias="${model##*/}"
      alias="${alias%%-*}"
      alias="${alias%%.*}"
      alias="$(printf '%s' "$alias" | tr '[:upper:]' '[:lower:]')"
      alias="${alias//[^a-z0-9_]/_}"
      ;;
  esac
  echo "$alias"
}

method_alias() {
  case "$1" in
    robust) echo "base" ;;
    patch) echo "patch" ;;
    *) echo "$1" ;;
  esac
}

job_name_for() {
  local suite="$1"
  local model="$2"
  local method="$3"
  local alias
  local method_tag
  alias="$(model_alias "$model")"
  method_tag="$(method_alias "$method")"
  if [[ -n "$JOB_NAME_PREFIX" ]]; then
    echo "${JOB_NAME_PREFIX}_${suite}_${alias}_${method_tag}"
  else
    echo "${suite}_${alias}_${method_tag}"
  fi
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for EXECUTION_MODE=tmux but was not found." >&2
    exit 1
  fi
}

PARALLEL_PIDS=()
PARALLEL_JOB_NAMES=()
PARALLEL_PID_FILES=()

spawn_parallel_job() {
  local job_name="$1"
  local command_text="$2"
  local log_file="$JOB_LOG_DIR/${job_name}.log"
  local pid_file="$JOB_PID_DIR/${job_name}.pid"
  local existing_pid=""

  mkdir -p "$JOB_LOG_DIR" "$JOB_PID_DIR"

  if [[ -f "$pid_file" ]]; then
    existing_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "parallel job already running, skipping: $job_name (pid=$existing_pid)"
      return 0
    fi
    rm -f "$pid_file"
  fi

  bash -lc "$command_text" >"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_file"
  PARALLEL_PIDS+=("$pid")
  PARALLEL_JOB_NAMES+=("$job_name")
  PARALLEL_PID_FILES+=("$pid_file")
  echo "Started parallel job: $job_name (pid=$pid)"
  echo "  log: $log_file"
  echo "  pid: $pid_file"
}

wait_for_parallel_jobs() {
  local idx
  local pid
  local job_name
  local pid_file
  local status
  local failed=0

  if [[ "${#PARALLEL_PIDS[@]}" -eq 0 ]]; then
    echo "No parallel jobs were launched."
    return 0
  fi

  print_header "Waiting for parallel jobs"
  for idx in "${!PARALLEL_PIDS[@]}"; do
    pid="${PARALLEL_PIDS[$idx]}"
    job_name="${PARALLEL_JOB_NAMES[$idx]}"
    pid_file="${PARALLEL_PID_FILES[$idx]}"
    status=0
    if wait "$pid"; then
      echo "Job finished successfully: $job_name"
    else
      status=$?
      echo "Job failed: $job_name (exit=$status)" >&2
      failed=1
    fi
    rm -f "$pid_file"
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "One or more parallel jobs failed. Check logs under $JOB_LOG_DIR." >&2
    return 1
  fi
  echo "All parallel jobs finished successfully."
}

spawn_tmux_job() {
  local job_name="$1"
  local command_text="$2"
  local log_file="$JOB_LOG_DIR/${job_name}.log"
  local wrapped_command

  mkdir -p "$JOB_LOG_DIR"

  if tmux has-session -t "$job_name" 2>/dev/null; then
    echo "tmux session already exists, skipping: $job_name"
    return 0
  fi

  wrapped_command="${command_text} 2>&1 | tee -a $(shell_quote "$log_file")"
  tmux new-session -d -s "$job_name" "bash -lc $(shell_quote "$wrapped_command")"
  echo "Started tmux session: $job_name"
  echo "  log: $log_file"
}

build_locomo_job_command() {
  local model="$1"
  local runner_method="$2"
  local locomo_batch="${BATCH:-$CONFIG_LOCOMO_BATCH}"
  local locomo_retrieve_k="${RETRIEVE_K:-$CONFIG_LOCOMO_RETRIEVE_K}"
  local locomo_patch_top_k="${PATCH_TOP_K:-$CONFIG_LOCOMO_PATCH_TOP_K}"
  local locomo_patch_usage="${PATCH_USAGE:-$CONFIG_LOCOMO_PATCH_USAGE}"
  local locomo_temperature_c5="${TEMPERATURE_C5:-$CONFIG_LOCOMO_TEMPERATURE_C5}"
  local locomo_raw_llm_log="${RAW_LLM_LOG:-$CONFIG_LOCOMO_RAW_LLM_LOG}"
  local locomo_patch_cache_root_base="${PATCH_CACHE_ROOT_BASE:-$CONFIG_LOCOMO_PATCH_CACHE_ROOT_BASE}"
  local locomo_dataset="${DATASET:-$CONFIG_LOCOMO_DATASET}"
  local locomo_ratio="${RATIO:-$CONFIG_LOCOMO_RATIO}"
  local locomo_start_sample="${START_SAMPLE:-$CONFIG_LOCOMO_START_SAMPLE}"
  local locomo_end_sample="${END_SAMPLE:-$CONFIG_LOCOMO_END_SAMPLE}"
  printf '%s' "cd $(shell_quote "$ROOT_DIR") && source $(shell_quote "$ROOT_DIR/$VENV_DIR/bin/activate") && BACKEND=$(shell_quote "$BACKEND") OPENAI_API_KEY=$(shell_quote "$API_KEY") OPENROUTER_API_KEY=$(shell_quote "$API_KEY") OPENAI_BASE_URL=$(shell_quote "$API_BASE") MODEL_LIST=$(shell_quote "$model") DATASET=$(shell_quote "$locomo_dataset") RATIO=$(shell_quote "$locomo_ratio") START_SAMPLE=$(shell_quote "$locomo_start_sample") END_SAMPLE=$(shell_quote "$locomo_end_sample") BATCH=$(shell_quote "$locomo_batch") RETRIEVE_K=$(shell_quote "$locomo_retrieve_k") PATCH_TOP_K=$(shell_quote "$locomo_patch_top_k") PATCH_USAGE=$(shell_quote "$locomo_patch_usage") TEMPERATURE_C5=$(shell_quote "$locomo_temperature_c5") RAW_LLM_LOG=$(shell_quote "$locomo_raw_llm_log") PATCH_CACHE_ROOT_BASE=$(shell_quote "$locomo_patch_cache_root_base") RESUME=$(shell_quote "$RESUME") bash scripts/run_locomo_baseline_patch_3models.sh $(shell_quote "$runner_method")"
}

build_persona_job_command() {
  local model="$1"
  local runner_method="$2"
  local benchmark_file="$3"
  local persona_batch="${BATCH:-$CONFIG_PERSONA_BATCH}"
  local persona_retrieve_k="${RETRIEVE_K:-$CONFIG_PERSONA_RETRIEVE_K}"
  local persona_output_dir="${OUTPUT_DIR:-$CONFIG_PERSONA_OUTPUT_DIR}"
  local persona_litellm_log="${LITELLM_LOG:-$CONFIG_PERSONA_LITELLM_LOG}"
  local persona_ids="${PERSONA_IDS:-$CONFIG_PERSONA_PERSONA_IDS}"
  local max_items="${MAX_ITEMS:-$CONFIG_PERSONA_MAX_ITEMS}"
  local include_debug_columns="${INCLUDE_DEBUG_COLUMNS:-$CONFIG_PERSONA_INCLUDE_DEBUG_COLUMNS}"
  local cache_root="${CACHE_ROOT:-$CONFIG_PERSONA_CACHE_ROOT}"
  local preference_aware="${PREFERENCE_AWARE:-$CONFIG_PERSONA_PREFERENCE_AWARE}"
  local preference_aware_level="${PREFERENCE_AWARE_LEVEL:-$CONFIG_PERSONA_PREFERENCE_AWARE_LEVEL}"
  local persona_patch_top_k="${PATCH_TOP_K:-$CONFIG_PERSONA_PATCH_TOP_K}"
  local persona_patch_usage="${PATCH_USAGE:-$CONFIG_PERSONA_PATCH_USAGE}"
  local min_patch_similarity="${MIN_PATCH_SIMILARITY:-$CONFIG_PERSONA_MIN_PATCH_SIMILARITY}"
  local force_reingest_patches="${FORCE_REINGEST_PATCHES:-$CONFIG_PERSONA_FORCE_REINGEST_PATCHES}"
  local exclude_revoke_patches="${EXCLUDE_REVOKE_PATCHES:-$CONFIG_PERSONA_EXCLUDE_REVOKE_PATCHES}"
  local exclude_add_patches="${EXCLUDE_ADD_PATCHES:-$CONFIG_PERSONA_EXCLUDE_ADD_PATCHES}"
  local require_pref_change="${REQUIRE_PREF_CHANGE:-$CONFIG_PERSONA_REQUIRE_PREF_CHANGE}"
  local llm_patch_filter="${LLM_PATCH_FILTER:-$CONFIG_PERSONA_LLM_PATCH_FILTER}"
  local gt_patch="${GT_PATCH:-$CONFIG_PERSONA_GT_PATCH}"
  local gt_patch_file="${GT_PATCH_FILE:-$CONFIG_PERSONA_GT_PATCH_FILE}"
  local gt_patch_top_k="${GT_PATCH_TOP_K:-$CONFIG_PERSONA_GT_PATCH_TOP_K}"
  local gt_patch_min_similarity="${GT_PATCH_MIN_SIMILARITY:-$CONFIG_PERSONA_GT_PATCH_MIN_SIMILARITY}"
  local gp_patch_retrieval="${GP_PATCH_RETRIEVAL:-$CONFIG_PERSONA_GP_PATCH_RETRIEVAL}"
  local persona_root="${PERSONA_ROOT:-$CONFIG_PERSONA_ROOT}"
  local persona_size="${SIZE:-$CONFIG_PERSONA_SIZE}"
  printf '%s' "cd $(shell_quote "$ROOT_DIR") && source $(shell_quote "$ROOT_DIR/$VENV_DIR/bin/activate") && BACKEND=$(shell_quote "$BACKEND") OPENAI_API_KEY=$(shell_quote "$API_KEY") OPENROUTER_API_KEY=$(shell_quote "$API_KEY") OPENAI_BASE_URL=$(shell_quote "$API_BASE") MODEL_LIST=$(shell_quote "$model") BENCHMARK_FILE=$(shell_quote "$benchmark_file") PERSONA_ROOT=$(shell_quote "$persona_root") SIZE=$(shell_quote "$persona_size") BATCH=$(shell_quote "$persona_batch") RETRIEVE_K=$(shell_quote "$persona_retrieve_k") OUTPUT_DIR=$(shell_quote "$persona_output_dir") LITELLM_LOG=$(shell_quote "$persona_litellm_log") PERSONA_IDS=$(shell_quote "$persona_ids") MAX_ITEMS=$(shell_quote "$max_items") INCLUDE_DEBUG_COLUMNS=$(shell_quote "$include_debug_columns") CACHE_ROOT=$(shell_quote "$cache_root") PREFERENCE_AWARE=$(shell_quote "$preference_aware") PREFERENCE_AWARE_LEVEL=$(shell_quote "$preference_aware_level") PATCH_TOP_K=$(shell_quote "$persona_patch_top_k") PATCH_USAGE=$(shell_quote "$persona_patch_usage") MIN_PATCH_SIMILARITY=$(shell_quote "$min_patch_similarity") FORCE_REINGEST_PATCHES=$(shell_quote "$force_reingest_patches") EXCLUDE_REVOKE_PATCHES=$(shell_quote "$exclude_revoke_patches") EXCLUDE_ADD_PATCHES=$(shell_quote "$exclude_add_patches") REQUIRE_PREF_CHANGE=$(shell_quote "$require_pref_change") LLM_PATCH_FILTER=$(shell_quote "$llm_patch_filter") GT_PATCH=$(shell_quote "$gt_patch") GT_PATCH_FILE=$(shell_quote "$gt_patch_file") GT_PATCH_TOP_K=$(shell_quote "$gt_patch_top_k") GT_PATCH_MIN_SIMILARITY=$(shell_quote "$gt_patch_min_similarity") GP_PATCH_RETRIEVAL=$(shell_quote "$gp_patch_retrieval") RESUME=$(shell_quote "$RESUME") bash scripts/run_persona_baseline_patch.sh $(shell_quote "$runner_method")"
}

resolve_persona_benchmark() {
  local benchmark_file="${BENCHMARK_FILE:-$CONFIG_PERSONA_BENCHMARK_FILE}"
  local default_benchmark="${PERSONA_DATASET_DIR:-$CONFIG_PERSONA_DATASET_DIR}/benchmark_v34/text/benchmark_9p_ood_v34.csv"
  ensure_persona_dataset "$benchmark_file"
  if [[ ! -f "$benchmark_file" && -f "$default_benchmark" ]]; then
    benchmark_file="$default_benchmark"
  fi
  printf '%s' "$benchmark_file"
}

spawn_locomo_tmux_jobs() {
  local model
  local runner_method
  local job_name
  local cmd
  print_header "Launching LoCoMo tmux jobs"
  for model in "${LOCOMO_MODELS[@]}"; do
    for runner_method in robust patch; do
      if [[ "$METHOD" != "both" && "$METHOD" != "$runner_method" ]]; then
        continue
      fi
      job_name="$(job_name_for locomo "$model" "$runner_method")"
      cmd="$(build_locomo_job_command "$model" "$runner_method")"
      spawn_tmux_job "$job_name" "$cmd"
    done
  done
}

spawn_persona_tmux_jobs() {
  local benchmark_file
  local model
  local runner_method
  local job_name
  local cmd
  benchmark_file="$(resolve_persona_benchmark)"
  print_header "Launching Persona tmux jobs"
  for model in "${PERSONA_MODELS[@]}"; do
    for runner_method in robust patch; do
      if [[ "$METHOD" != "both" && "$METHOD" != "$runner_method" ]]; then
        continue
      fi
      job_name="$(job_name_for persona "$model" "$runner_method")"
      cmd="$(build_persona_job_command "$model" "$runner_method" "$benchmark_file")"
      spawn_tmux_job "$job_name" "$cmd"
    done
  done
}

spawn_locomo_parallel_jobs() {
  local model
  local runner_method
  local job_name
  local cmd
  print_header "Launching LoCoMo parallel jobs"
  for model in "${LOCOMO_MODELS[@]}"; do
    for runner_method in robust patch; do
      if [[ "$METHOD" != "both" && "$METHOD" != "$runner_method" ]]; then
        continue
      fi
      job_name="$(job_name_for locomo "$model" "$runner_method")"
      cmd="$(build_locomo_job_command "$model" "$runner_method")"
      spawn_parallel_job "$job_name" "$cmd"
    done
  done
}

spawn_persona_parallel_jobs() {
  local benchmark_file
  local model
  local runner_method
  local job_name
  local cmd
  benchmark_file="$(resolve_persona_benchmark)"
  print_header "Launching Persona parallel jobs"
  for model in "${PERSONA_MODELS[@]}"; do
    for runner_method in robust patch; do
      if [[ "$METHOD" != "both" && "$METHOD" != "$runner_method" ]]; then
        continue
      fi
      job_name="$(job_name_for persona "$model" "$runner_method")"
      cmd="$(build_persona_job_command "$model" "$runner_method" "$benchmark_file")"
      spawn_parallel_job "$job_name" "$cmd"
    done
  done
}

run_locomo() {
  local model_csv
  model_csv="$(join_by_comma "${LOCOMO_MODELS[@]}")"
  print_header "Running LoCoMo experiments"
  BACKEND="$BACKEND"   OPENAI_API_KEY="$API_KEY"   OPENROUTER_API_KEY="$API_KEY"   OPENAI_BASE_URL="$API_BASE"   MODEL_LIST="$model_csv"   DATASET="${DATASET:-$CONFIG_LOCOMO_DATASET}"   RATIO="${RATIO:-$CONFIG_LOCOMO_RATIO}"   START_SAMPLE="${START_SAMPLE:-$CONFIG_LOCOMO_START_SAMPLE}"   END_SAMPLE="${END_SAMPLE:-$CONFIG_LOCOMO_END_SAMPLE}"   BATCH="${BATCH:-$CONFIG_LOCOMO_BATCH}"   RETRIEVE_K="${RETRIEVE_K:-$CONFIG_LOCOMO_RETRIEVE_K}"   PATCH_TOP_K="${PATCH_TOP_K:-$CONFIG_LOCOMO_PATCH_TOP_K}"   PATCH_USAGE="${PATCH_USAGE:-$CONFIG_LOCOMO_PATCH_USAGE}"   TEMPERATURE_C5="${TEMPERATURE_C5:-$CONFIG_LOCOMO_TEMPERATURE_C5}"   RAW_LLM_LOG="${RAW_LLM_LOG:-$CONFIG_LOCOMO_RAW_LLM_LOG}"   PATCH_CACHE_ROOT_BASE="${PATCH_CACHE_ROOT_BASE:-$CONFIG_LOCOMO_PATCH_CACHE_ROOT_BASE}"   RESUME="$RESUME"   bash scripts/run_locomo_baseline_patch_3models.sh "$METHOD"
}

run_persona() {
  local model_csv
  local benchmark_file
  model_csv="$(join_by_comma "${PERSONA_MODELS[@]}")"
  benchmark_file="$(resolve_persona_benchmark)"
  print_header "Running Persona experiments"
  BACKEND="$BACKEND"   OPENAI_API_KEY="$API_KEY"   OPENROUTER_API_KEY="$API_KEY"   OPENAI_BASE_URL="$API_BASE"   MODEL_LIST="$model_csv"   BENCHMARK_FILE="$benchmark_file"   PERSONA_ROOT="${PERSONA_ROOT:-$CONFIG_PERSONA_ROOT}"   SIZE="${SIZE:-$CONFIG_PERSONA_SIZE}"   BATCH="${BATCH:-$CONFIG_PERSONA_BATCH}"   RETRIEVE_K="${RETRIEVE_K:-$CONFIG_PERSONA_RETRIEVE_K}"   OUTPUT_DIR="${OUTPUT_DIR:-$CONFIG_PERSONA_OUTPUT_DIR}"   LITELLM_LOG="${LITELLM_LOG:-$CONFIG_PERSONA_LITELLM_LOG}"   PERSONA_IDS="${PERSONA_IDS:-$CONFIG_PERSONA_PERSONA_IDS}"   MAX_ITEMS="${MAX_ITEMS:-$CONFIG_PERSONA_MAX_ITEMS}"   INCLUDE_DEBUG_COLUMNS="${INCLUDE_DEBUG_COLUMNS:-$CONFIG_PERSONA_INCLUDE_DEBUG_COLUMNS}"   CACHE_ROOT="${CACHE_ROOT:-$CONFIG_PERSONA_CACHE_ROOT}"   PREFERENCE_AWARE="${PREFERENCE_AWARE:-$CONFIG_PERSONA_PREFERENCE_AWARE}"   PREFERENCE_AWARE_LEVEL="${PREFERENCE_AWARE_LEVEL:-$CONFIG_PERSONA_PREFERENCE_AWARE_LEVEL}"   PATCH_TOP_K="${PATCH_TOP_K:-$CONFIG_PERSONA_PATCH_TOP_K}"   PATCH_USAGE="${PATCH_USAGE:-$CONFIG_PERSONA_PATCH_USAGE}"   MIN_PATCH_SIMILARITY="${MIN_PATCH_SIMILARITY:-$CONFIG_PERSONA_MIN_PATCH_SIMILARITY}"   FORCE_REINGEST_PATCHES="${FORCE_REINGEST_PATCHES:-$CONFIG_PERSONA_FORCE_REINGEST_PATCHES}"   EXCLUDE_REVOKE_PATCHES="${EXCLUDE_REVOKE_PATCHES:-$CONFIG_PERSONA_EXCLUDE_REVOKE_PATCHES}"   EXCLUDE_ADD_PATCHES="${EXCLUDE_ADD_PATCHES:-$CONFIG_PERSONA_EXCLUDE_ADD_PATCHES}"   REQUIRE_PREF_CHANGE="${REQUIRE_PREF_CHANGE:-$CONFIG_PERSONA_REQUIRE_PREF_CHANGE}"   LLM_PATCH_FILTER="${LLM_PATCH_FILTER:-$CONFIG_PERSONA_LLM_PATCH_FILTER}"   GT_PATCH="${GT_PATCH:-$CONFIG_PERSONA_GT_PATCH}"   GT_PATCH_FILE="${GT_PATCH_FILE:-$CONFIG_PERSONA_GT_PATCH_FILE}"   GT_PATCH_TOP_K="${GT_PATCH_TOP_K:-$CONFIG_PERSONA_GT_PATCH_TOP_K}"   GT_PATCH_MIN_SIMILARITY="${GT_PATCH_MIN_SIMILARITY:-$CONFIG_PERSONA_GT_PATCH_MIN_SIMILARITY}"   GP_PATCH_RETRIEVAL="${GP_PATCH_RETRIEVAL:-$CONFIG_PERSONA_GP_PATCH_RETRIEVAL}"   RESUME="$RESUME"   bash scripts/run_persona_baseline_patch.sh "$METHOD"
}

bootstrap_venv
prepare_requested_datasets
print_config

if [[ "$EXECUTION_MODE" == "tmux" ]]; then
  require_tmux
  case "$SUITE" in
    all)
      spawn_locomo_tmux_jobs
      spawn_persona_tmux_jobs
      ;;
    locomo)
      spawn_locomo_tmux_jobs
      ;;
    persona)
      spawn_persona_tmux_jobs
      ;;
  esac
  echo "tmux jobs launched. Use 'tmux ls' to inspect running sessions."
  exit 0
fi

if [[ "$EXECUTION_MODE" == "parallel" ]]; then
  case "$SUITE" in
    all)
      spawn_locomo_parallel_jobs
      spawn_persona_parallel_jobs
      ;;
    locomo)
      spawn_locomo_parallel_jobs
      ;;
    persona)
      spawn_persona_parallel_jobs
      ;;
  esac
  echo "Parallel jobs launched with plain shell child processes. Logs are under $JOB_LOG_DIR."
  wait_for_parallel_jobs
  exit $?
fi

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
