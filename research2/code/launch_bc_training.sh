#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jaisharma/HW8/research2}"
PYTHON="${PYTHON:-/home/jaisharma/miniconda3/envs/pcb/bin/python}"
CONFIG="${CONFIG:-configs/dataset_128px_v1.yaml}"
DATASET_DIR="${DATASET_DIR:-results/datasets_128px_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-results/bc_128px_v1}"
MODEL_FAMILY="${MODEL_FAMILY:-scratch_bc_128}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
BACKBONE_LR="${BACKBONE_LR:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
BUDGETS="${BUDGETS:-5 20 50}"
SEEDS="${SEEDS:-0 1 2}"
TRAIN_CONFIGS="${TRAIN_CONFIGS:-}"
PHASE_BALANCE="${PHASE_BALANCE:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

cd "$ROOT"

LOG_DIR="logs/bc_128px_v1_training_${RUN_ID}"
SUPERVISOR_PID_FILE="$LOG_DIR/supervisor.pid"
WORKER_PID_FILE="$LOG_DIR/worker_pids.tsv"
STATUS_FILE="$LOG_DIR/worker_status.tsv"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ -s "$SUPERVISOR_PID_FILE" ] && kill -0 "$(cat "$SUPERVISOR_PID_FILE")" 2>/dev/null; then
  echo "BC training supervisor already running pid=$(cat "$SUPERVISOR_PID_FILE")"
  exit 2
fi

if pgrep -af "code/run_bc_matrix.py.*$OUTPUT_DIR" >/dev/null; then
  echo "BC training already running for $OUTPUT_DIR"
  pgrep -af "code/run_bc_matrix.py.*$OUTPUT_DIR"
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

trap 'log_work "BC training supervisor $$ received termination signal; stopping workers."; terminate_workers; exit 130' INT TERM

read -r -a GPU_ID_LIST <<< "$GPU_IDS"
if [ "${#GPU_ID_LIST[@]}" -eq 0 ]; then
  echo "GPU_IDS is empty"
  exit 2
fi
read -r -a BUDGET_LIST <<< "$BUDGETS"
read -r -a SEED_LIST <<< "$SEEDS"
TRAIN_CONFIG_ARGS=()
if [ -n "$TRAIN_CONFIGS" ]; then
  read -r -a TRAIN_CONFIG_LIST <<< "$TRAIN_CONFIGS"
  TRAIN_CONFIG_ARGS=(--train-configs "${TRAIN_CONFIG_LIST[@]}")
fi
PHASE_BALANCE_ARGS=()
if [ "$PHASE_BALANCE" = "1" ] || [ "$PHASE_BALANCE" = "true" ] || [ "$PHASE_BALANCE" = "TRUE" ]; then
  PHASE_BALANCE_ARGS=(--phase-balance)
fi
BACKBONE_LR_ARGS=()
if [ -n "$BACKBONE_LR" ]; then
  BACKBONE_LR_ARGS=(--backbone-lr "$BACKBONE_LR")
fi

log_work "Started BC 128px training supervisor $$ for model_family=$MODEL_FAMILY with $NUM_WORKERS workers on GPU_IDS=$GPU_IDS, budgets=${BUDGET_LIST[*]}, seeds=${SEED_LIST[*]}, lr=$LR, backbone_lr=${BACKBONE_LR:-default}, weight_decay=$WEIGHT_DECAY, phase_balance=$PHASE_BALANCE. Output: $OUTPUT_DIR. Logs: $LOG_DIR."

for worker_index in $(seq 0 "$((NUM_WORKERS - 1))"); do
  worker_name=$(printf "worker_%02d" "$worker_index")
  gpu_id="${GPU_ID_LIST[$((worker_index % ${#GPU_ID_LIST[@]}))]}"
  worker_log="$LOG_DIR/${worker_name}.log"
  worker_status="$LOG_DIR/${worker_name}_status.jsonl"

  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
    TORCH_NUM_THREADS="$TORCH_NUM_THREADS" \
    "$PYTHON" code/run_bc_matrix.py \
      --config "$CONFIG" \
      --dataset-dir "$DATASET_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --model-family "$MODEL_FAMILY" \
      "${TRAIN_CONFIG_ARGS[@]}" \
      "${PHASE_BALANCE_ARGS[@]}" \
      --budgets "${BUDGET_LIST[@]}" \
      --seeds "${SEED_LIST[@]}" \
      --worker-index "$worker_index" \
      --num-workers "$NUM_WORKERS" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      "${BACKBONE_LR_ARGS[@]}" \
      --weight-decay "$WEIGHT_DECAY" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --device auto \
      --status-path "$worker_status"
  ) > "$worker_log" 2>&1 &

  pid=$!
  printf "%02d\t%s\t%s\n" "$worker_index" "$pid" "$gpu_id" >> "$WORKER_PID_FILE"
  log_work "Launched BC training worker $worker_name with PID $pid on GPU $gpu_id."
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
  log_work "BC training worker $(printf "worker_%02d" "$((10#$worker_index))") finished with status $status on GPU $gpu_id."
done < "$WORKER_PID_FILE"

if [ "$failures" -eq 0 ]; then
  log_work "BC 128px training matrix completed successfully. Output: $OUTPUT_DIR."
else
  log_work "BC 128px training matrix finished with $failures worker failure(s). See $STATUS_FILE."
fi

exit "$failures"
