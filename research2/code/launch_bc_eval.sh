#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jaisharma/HW8/research2}"
PYTHON="${PYTHON:-/home/jaisharma/miniconda3/envs/pcb/bin/python}"
CONFIG="${CONFIG:-configs/dataset_128px_v1.yaml}"
MODEL_DIR="${MODEL_DIR:-results/bc_128px_v1/models}"
METRICS_DIR="${METRICS_DIR:-results/bc_128px_v1/metrics}"
MODEL_FAMILY="${MODEL_FAMILY:-scratch_bc_128}"
EVAL_SPLIT="${EVAL_SPLIT:-id}"
EVAL_DIR="${EVAL_DIR:-}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-400}"
SUCCESS_THRESHOLDS="${SUCCESS_THRESHOLDS:-0.005 0.01 0.02 0.05}"
STOP_THRESHOLD="${STOP_THRESHOLD:-0.001}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
BUDGETS="${BUDGETS:-5 20 50}"
SEEDS="${SEEDS:-0 1 2}"
TRAIN_CONFIGS="${TRAIN_CONFIGS:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

cd "$ROOT"

LOG_SAFE_MODEL_FAMILY="${MODEL_FAMILY//[^A-Za-z0-9_]/_}"
LOG_DIR="logs/bc_128px_v1_eval_${LOG_SAFE_MODEL_FAMILY}_${EVAL_SPLIT}_${RUN_ID}"
SUPERVISOR_PID_FILE="logs/bc_128px_v1_eval_${LOG_SAFE_MODEL_FAMILY}_${EVAL_SPLIT}_supervisor.pid"
WORKER_PID_FILE="$LOG_DIR/worker_pids.tsv"
STATUS_FILE="$LOG_DIR/worker_status.tsv"

mkdir -p "$LOG_DIR" "$METRICS_DIR"

if [ "$EVAL_SPLIT" != "id" ] && [ "$EVAL_SPLIT" != "ood" ]; then
  echo "EVAL_SPLIT must be id or ood"
  exit 2
fi

if [ -s "$SUPERVISOR_PID_FILE" ] && kill -0 "$(cat "$SUPERVISOR_PID_FILE")" 2>/dev/null; then
  echo "BC eval supervisor already running pid=$(cat "$SUPERVISOR_PID_FILE")"
  exit 2
fi

if pgrep -af "code/run_bc_eval_matrix.py.*--model-family $MODEL_FAMILY.*--eval-split $EVAL_SPLIT" >/dev/null; then
  echo "BC $MODEL_FAMILY $EVAL_SPLIT eval already running"
  pgrep -af "code/run_bc_eval_matrix.py.*--model-family $MODEL_FAMILY.*--eval-split $EVAL_SPLIT"
  exit 2
fi

echo "$$" > "$SUPERVISOR_PID_FILE"
printf "worker_index\tpid\tgpu_id\n" > "$WORKER_PID_FILE"
printf "worker_index\tpid\tstatus\n" > "$STATUS_FILE"

log_work() {
  printf "%s\n" "- $(date -u '+%Y-%m-%d %H:%M:%S UTC'): $*" >> WORK_LOG.md
}

terminate_workers() {
  if [ -f "$WORKER_PID_FILE" ]; then
    awk 'NR > 1 {print $2}' "$WORKER_PID_FILE" | while read -r pid; do
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    done
  fi
}

trap 'log_work "BC $EVAL_SPLIT eval supervisor $$ received termination signal; stopping workers."; terminate_workers; exit 130' INT TERM

read -r -a GPU_ID_LIST <<< "$GPU_IDS"
read -r -a SUCCESS_THRESHOLD_LIST <<< "$SUCCESS_THRESHOLDS"
read -r -a BUDGET_LIST <<< "$BUDGETS"
read -r -a SEED_LIST <<< "$SEEDS"
TRAIN_CONFIG_ARGS=()
if [ -n "$TRAIN_CONFIGS" ]; then
  read -r -a TRAIN_CONFIG_LIST <<< "$TRAIN_CONFIGS"
  TRAIN_CONFIG_ARGS=(--train-configs "${TRAIN_CONFIG_LIST[@]}")
fi
if [ "${#GPU_ID_LIST[@]}" -eq 0 ]; then
  echo "GPU_IDS is empty"
  exit 2
fi
if [ "${#SUCCESS_THRESHOLD_LIST[@]}" -eq 0 ]; then
  echo "SUCCESS_THRESHOLDS is empty"
  exit 2
fi

log_work "Started BC 128px $EVAL_SPLIT eval supervisor $$ for model_family=$MODEL_FAMILY with $NUM_WORKERS workers on GPU_IDS=$GPU_IDS, budgets=${BUDGET_LIST[*]}, seeds=${SEED_LIST[*]}. Success thresholds=$SUCCESS_THRESHOLDS, stop threshold=$STOP_THRESHOLD. Metrics: $METRICS_DIR. Logs: $LOG_DIR."

for worker_index in $(seq 0 "$((NUM_WORKERS - 1))"); do
  worker_name=$(printf "worker_%02d" "$worker_index")
  gpu_id="${GPU_ID_LIST[$((worker_index % ${#GPU_ID_LIST[@]}))]}"
  worker_log="$LOG_DIR/${worker_name}.log"
  worker_status="$LOG_DIR/${worker_name}_status.jsonl"
  eval_dir_args=()
  if [ -n "$EVAL_DIR" ]; then
    eval_dir_args=(--eval-dir "$EVAL_DIR")
  fi

  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
    TORCH_NUM_THREADS="$TORCH_NUM_THREADS" \
    "$PYTHON" code/run_bc_eval_matrix.py \
      --config "$CONFIG" \
      --model-dir "$MODEL_DIR" \
      --metrics-dir "$METRICS_DIR" \
      --model-family "$MODEL_FAMILY" \
      --eval-split "$EVAL_SPLIT" \
      "${eval_dir_args[@]}" \
      "${TRAIN_CONFIG_ARGS[@]}" \
      --budgets "${BUDGET_LIST[@]}" \
      --seeds "${SEED_LIST[@]}" \
      --worker-index "$worker_index" \
      --num-workers "$NUM_WORKERS" \
      --device auto \
      --steps-per-episode "$STEPS_PER_EPISODE" \
      --success-thresholds "${SUCCESS_THRESHOLD_LIST[@]}" \
      --stop-threshold "$STOP_THRESHOLD" \
      --status-path "$worker_status"
  ) > "$worker_log" 2>&1 &

  pid=$!
  printf "%02d\t%s\t%s\n" "$worker_index" "$pid" "$gpu_id" >> "$WORKER_PID_FILE"
  log_work "Launched BC $EVAL_SPLIT eval worker $worker_name with PID $pid on GPU $gpu_id."
done

failures=0
while IFS=$'\t' read -r worker_index pid gpu_id; do
  if [ "$worker_index" = "worker_index" ]; then
    continue
  fi
  if wait "$pid"; then
    status="ok"
  else
    status="failed"
    failures=$((failures + 1))
  fi
  printf "%s\t%s\t%s\n" "$worker_index" "$pid" "$status" >> "$STATUS_FILE"
  log_work "BC $EVAL_SPLIT eval worker $(printf "worker_%02d" "$((10#$worker_index))") finished with status $status on GPU $gpu_id."
done < "$WORKER_PID_FILE"

if [ "$failures" -eq 0 ]; then
  log_work "BC 128px $EVAL_SPLIT eval matrix completed successfully. Metrics: $METRICS_DIR/$EVAL_SPLIT."
else
  log_work "BC 128px $EVAL_SPLIT eval matrix finished with $failures worker failure(s). See $STATUS_FILE."
fi

exit "$failures"
