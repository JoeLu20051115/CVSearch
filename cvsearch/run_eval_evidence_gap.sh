#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=""
MODEL_PATH=""
ANNOTATION_PATH=""
SAM_MODEL_PATH=""
NLP_MODEL_PATH=""
CLIP_MODEL_PATH=""
BENCHMARK=""
GPU=""
ANSWERS_FILE=""
LOG_FILE=""
CONFIG=""
MODE=""
SPLIT="all"
SPLIT_SEED="260809"
ORDINALS=""
NUM_CHUNKS="1"
CHUNK_IDX="0"
RESUME=0
FORCE=0

usage() {
    echo "Usage: $0 --root-path PATH --model-path PATH --annotation-path PATH --sam-model-path PATH --nlp-model-path PATH --clip-model-path PATH --benchmark NAME --gpu ID --answers-file FILE --log-file FILE --config NAME_OR_JSON [options]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root-path) ROOT_PATH="${2:?missing value for $1}"; shift 2 ;;
        --model-path) MODEL_PATH="${2:?missing value for $1}"; shift 2 ;;
        --annotation-path) ANNOTATION_PATH="${2:?missing value for $1}"; shift 2 ;;
        --sam-model-path) SAM_MODEL_PATH="${2:?missing value for $1}"; shift 2 ;;
        --nlp-model-path) NLP_MODEL_PATH="${2:?missing value for $1}"; shift 2 ;;
        --clip-model-path) CLIP_MODEL_PATH="${2:?missing value for $1}"; shift 2 ;;
        --benchmark) BENCHMARK="${2:?missing value for $1}"; shift 2 ;;
        --gpu) GPU="${2:?missing value for $1}"; shift 2 ;;
        --answers-file) ANSWERS_FILE="${2:?missing value for $1}"; shift 2 ;;
        --log-file) LOG_FILE="${2:?missing value for $1}"; shift 2 ;;
        --config) CONFIG="${2:?missing value for $1}"; shift 2 ;;
        --mode) MODE="${2:?missing value for $1}"; shift 2 ;;
        --split) SPLIT="${2:?missing value for $1}"; shift 2 ;;
        --split-seed) SPLIT_SEED="${2:?missing value for $1}"; shift 2 ;;
        --ordinals) ORDINALS="${2:?missing value for $1}"; shift 2 ;;
        --num-chunks) NUM_CHUNKS="${2:?missing value for $1}"; shift 2 ;;
        --chunk-idx) CHUNK_IDX="${2:?missing value for $1}"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --force) FORCE=1; shift ;;
        --help|-h) usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

for value in ROOT_PATH MODEL_PATH ANNOTATION_PATH SAM_MODEL_PATH NLP_MODEL_PATH CLIP_MODEL_PATH BENCHMARK GPU ANSWERS_FILE LOG_FILE CONFIG; do
    [[ -n "${!value}" ]] || { echo "Missing required option for ${value}" >&2; usage; }
done
[[ $RESUME -eq 0 || $FORCE -eq 0 ]] || { echo "--resume and --force are mutually exclusive" >&2; exit 2; }
[[ ! -e "$ANSWERS_FILE" || $RESUME -eq 1 || $FORCE -eq 1 ]] || { echo "Refusing to overwrite $ANSWERS_FILE" >&2; exit 2; }
ANSWERS_ABS="$(realpath -m -- "$ANSWERS_FILE")"
LOG_ABS="$(realpath -m -- "$LOG_FILE")"
[[ "$ANSWERS_ABS" != "$LOG_ABS" ]] || { echo "Answers and log paths must differ" >&2; exit 2; }

mkdir -p "$(dirname "$ANSWERS_FILE")" "$(dirname "$LOG_FILE")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EFFECTIVE="${PYTHON_BIN:-python}"
ARGS=(
    "${SCRIPT_DIR}/perform_EGSearch.py"
    --root-path "$ROOT_PATH"
    --model-path "$MODEL_PATH"
    --annotation-path "$ANNOTATION_PATH"
    --sam-model-path "$SAM_MODEL_PATH"
    --nlp-model-path "$NLP_MODEL_PATH"
    --clip-model-path "$CLIP_MODEL_PATH"
    --benchmark "$BENCHMARK"
    --answers-file "$ANSWERS_FILE"
    --config "$CONFIG"
    --split "$SPLIT"
    --split-seed "$SPLIT_SEED"
    --num-chunks "$NUM_CHUNKS"
    --chunk-idx "$CHUNK_IDX"
)
[[ -z "$MODE" ]] || ARGS+=(--mode "$MODE")
[[ -z "$ORDINALS" ]] || ARGS+=(--ordinals "$ORDINALS")
[[ $RESUME -eq 0 ]] || ARGS+=(--resume)
[[ $FORCE -eq 0 ]] || ARGS+=(--force)

{
    echo "runner=evidence-gap-task7-v1"
    printf 'CUDA_VISIBLE_DEVICES=%q\n' "$GPU"
    printf 'python=%q\n' "$PYTHON_EFFECTIVE"
    printf 'resume=%q force=%q\n' "$RESUME" "$FORCE"
    printf 'argv='
    printf '%q ' "$PYTHON_EFFECTIVE" "${ARGS[@]}"
    printf '\n'
} >>"$LOG_FILE"

exec env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_EFFECTIVE" "${ARGS[@]}" >>"$LOG_FILE" 2>&1
