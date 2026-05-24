#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jaisharma/HW8/research2}"
PYTHON="${PYTHON:-/home/jaisharma/miniconda3/envs/pcb/bin/python}"
CONFIG="${CONFIG:-configs/dataset_128px_v1.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/datasets_128px_v1}"
MAX_STEPS="${MAX_STEPS:-400}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
BUDGETS="${BUDGETS:-5 20 50}"
SEEDS="${SEEDS:-0 1 2}"
WORKER_GROUPS_MODE="${WORKER_GROUPS_MODE:-default_8}"

cd "$ROOT"

LOG_DIR="logs/full_collection_128px_v1_parallel_${RUN_ID}"
SUPERVISOR_PID_FILE="logs/full_collection_128px_v1_parallel_supervisor.pid"
WORKER_PID_FILE="$LOG_DIR/worker_pids.tsv"
STATUS_FILE="$LOG_DIR/worker_status.tsv"

case "$WORKER_GROUPS_MODE" in
  default_8)
    WORKER_GROUPS=(
      "color_red_only color_multi"
      "avoid_color_red_only avoid_color_multi"
      "spatial_narrow spatial_wide"
      "avoid_spatial_narrow avoid_spatial_wide"
      "camera_fixed camera_multi_pose"
      "avoid_camera_fixed avoid_camera_multi_pose"
      "lighting_fixed lighting_diverse"
      "avoid_lighting_fixed avoid_lighting_diverse"
    )
    ;;
  high_budget_6)
    WORKER_GROUPS=(
      "color_red_only color_multi avoid_color_red_only"
      "avoid_color_multi spatial_narrow spatial_wide"
      "avoid_spatial_narrow avoid_spatial_wide camera_fixed"
      "camera_multi_pose avoid_camera_fixed avoid_camera_multi_pose"
      "lighting_fixed lighting_diverse"
      "avoid_lighting_fixed avoid_lighting_diverse"
    )
    ;;
  avoid_8)
    WORKER_GROUPS=(
      "avoid_color_red_only"
      "avoid_color_multi"
      "avoid_spatial_narrow"
      "avoid_spatial_wide"
      "avoid_camera_fixed"
      "avoid_camera_multi_pose"
      "avoid_lighting_fixed"
      "avoid_lighting_diverse"
    )
    ;;
  *)
    echo "unknown WORKER_GROUPS_MODE=$WORKER_GROUPS_MODE"
    exit 2
    ;;
esac

read -r -a BUDGET_LIST <<< "$BUDGETS"
read -r -a SEED_LIST <<< "$SEEDS"
SUCCESS_THRESHOLD_ARGS=()
if [ -n "$SUCCESS_THRESHOLD" ]; then
  SUCCESS_THRESHOLD_ARGS=(--success-threshold "$SUCCESS_THRESHOLD")
fi
AGGREGATE_TRAIN_CONFIGS=()
for group in "${WORKER_GROUPS[@]}"; do
  read -r -a group_train_configs <<< "$group"
  for train_config in "${group_train_configs[@]}"; do
    AGGREGATE_TRAIN_CONFIGS+=("$train_config")
  done
done

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ -s "$SUPERVISOR_PID_FILE" ] && kill -0 "$(cat "$SUPERVISOR_PID_FILE")" 2>/dev/null; then
  echo "parallel supervisor already running pid=$(cat "$SUPERVISOR_PID_FILE")"
  exit 2
fi

if pgrep -af "code/collect_dataset.py.*$OUTPUT_DIR" >/dev/null; then
  echo "collector already running for $OUTPUT_DIR"
  pgrep -af "code/collect_dataset.py.*$OUTPUT_DIR"
  exit 2
fi

echo "$$" > "$SUPERVISOR_PID_FILE"
printf "worker_index\tpid\ttrain_configs\n" > "$WORKER_PID_FILE"
printf "worker_index\tpid\tstatus\n" > "$STATUS_FILE"

log_work() {
  printf "%s\n" "- $(date -u +%F' '%T' UTC'): $*" >> WORK_LOG.md
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

trap 'log_work "Parallel full collection supervisor $$ received termination signal; stopping workers."; terminate_workers; exit 130' INT TERM

log_work "Started parallel 128px collection supervisor $$ with ${#WORKER_GROUPS[@]} workers, worker mode=$WORKER_GROUPS_MODE, budgets=${BUDGET_LIST[*]}, seeds=${SEED_LIST[*]}, max_steps=$MAX_STEPS, success_threshold=${SUCCESS_THRESHOLD:-default}. Output: $OUTPUT_DIR. Logs: $LOG_DIR."

for worker_index in "${!WORKER_GROUPS[@]}"; do
  worker_name=$(printf "worker_%02d" "$worker_index")
  read -r -a train_configs <<< "${WORKER_GROUPS[$worker_index]}"
  worker_log="$LOG_DIR/${worker_name}.log"
  worker_summary="$OUTPUT_DIR/collection_summary_${RUN_ID}_${worker_name}.json"

  (
    set -euo pipefail
    "$PYTHON" code/collect_dataset.py \
      --config "$CONFIG" \
      --output-dir "$OUTPUT_DIR" \
      --train-configs "${train_configs[@]}" \
      --budgets "${BUDGET_LIST[@]}" \
      --seeds "${SEED_LIST[@]}" \
      --max-steps-per-demo "$MAX_STEPS" \
      "${SUCCESS_THRESHOLD_ARGS[@]}" \
      --summary-path "$worker_summary"
  ) > "$worker_log" 2>&1 &

  pid=$!
  printf "%02d\t%s\t%s\n" "$worker_index" "$pid" "${WORKER_GROUPS[$worker_index]}" >> "$WORKER_PID_FILE"
  log_work "Launched parallel collection worker $worker_name with PID $pid for train configs: ${WORKER_GROUPS[$worker_index]}."
done

failures=0
while IFS=$'\t' read -r worker_index pid train_configs; do
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
  log_work "Parallel collection worker $(printf "worker_%02d" "$((10#$worker_index))") finished with status $status. Train configs: $train_configs."
done < "$WORKER_PID_FILE"

if [ "$failures" -eq 0 ]; then
  if "$PYTHON" code/aggregate_collection_summary.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --train-configs "${AGGREGATE_TRAIN_CONFIGS[@]}" \
    --budgets "${BUDGET_LIST[@]}" \
    --seeds "${SEED_LIST[@]}" \
    --summary-path "$OUTPUT_DIR/collection_summary_${RUN_ID}.json" \
    > "$LOG_DIR/aggregate.log" 2>&1; then
    log_work "Parallel 128px collection completed successfully for budgets=${BUDGET_LIST[*]}, seeds=${SEED_LIST[*]}; final summary written to $OUTPUT_DIR/collection_summary_${RUN_ID}.json."
  else
    failures=$((failures + 1))
    log_work "Parallel 128px full collection workers finished, but final aggregation failed. See $LOG_DIR/aggregate.log."
  fi
else
  log_work "Parallel 128px full collection finished with $failures worker failure(s). See $STATUS_FILE."
fi

exit "$failures"
