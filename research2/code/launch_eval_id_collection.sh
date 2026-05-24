#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jaisharma/HW8/research2}"
PYTHON="${PYTHON:-/home/jaisharma/miniconda3/envs/pcb/bin/python}"
CONFIG="${CONFIG:-configs/dataset_128px_v1.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/eval_id_128px_v1}"
MAX_STEPS="${MAX_STEPS:-400}"
BUDGET="${BUDGET:-50}"
SEEDS=(${SEEDS:-200 201 202})
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

cd "$ROOT"

LOG_DIR="logs/eval_id_128px_v1_parallel_${RUN_ID}"
SUPERVISOR_PID_FILE="logs/eval_id_128px_v1_parallel_supervisor.pid"
WORKER_PID_FILE="$LOG_DIR/worker_pids.tsv"
STATUS_FILE="$LOG_DIR/worker_status.tsv"

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

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ -s "$SUPERVISOR_PID_FILE" ] && kill -0 "$(cat "$SUPERVISOR_PID_FILE")" 2>/dev/null; then
  echo "ID eval supervisor already running pid=$(cat "$SUPERVISOR_PID_FILE")"
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

trap 'log_work "ID eval collection supervisor $$ received termination signal; stopping workers."; terminate_workers; exit 130' INT TERM

log_work "Started ID eval 128px collection supervisor $$ with ${#WORKER_GROUPS[@]} workers. Output: $OUTPUT_DIR. Logs: $LOG_DIR."

for worker_index in "${!WORKER_GROUPS[@]}"; do
  worker_name=$(printf "worker_%02d" "$worker_index")
  read -r -a train_configs <<< "${WORKER_GROUPS[$worker_index]}"
  worker_log="$LOG_DIR/${worker_name}.log"
  worker_summary="$OUTPUT_DIR/collection_summary_${worker_name}.json"

  (
    set -euo pipefail
    "$PYTHON" code/collect_dataset.py \
      --config "$CONFIG" \
      --output-dir "$OUTPUT_DIR" \
      --train-configs "${train_configs[@]}" \
      --budgets "$BUDGET" \
      --seeds "${SEEDS[@]}" \
      --max-steps-per-demo "$MAX_STEPS" \
      --summary-path "$worker_summary"
  ) > "$worker_log" 2>&1 &

  pid=$!
  printf "%02d\t%s\t%s\n" "$worker_index" "$pid" "${WORKER_GROUPS[$worker_index]}" >> "$WORKER_PID_FILE"
  log_work "Launched ID eval worker $worker_name with PID $pid for train configs: ${WORKER_GROUPS[$worker_index]}."
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
  log_work "ID eval worker $(printf "worker_%02d" "$((10#$worker_index))") finished with status $status. Train configs: $train_configs."
done < "$WORKER_PID_FILE"

if [ "$failures" -eq 0 ]; then
  if "$PYTHON" code/aggregate_collection_summary.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --summary-path "$OUTPUT_DIR/collection_summary.json" \
    --train-configs \
      color_red_only color_multi \
      avoid_color_red_only avoid_color_multi \
      spatial_narrow spatial_wide \
      avoid_spatial_narrow avoid_spatial_wide \
      camera_fixed camera_multi_pose \
      avoid_camera_fixed avoid_camera_multi_pose \
      lighting_fixed lighting_diverse \
      avoid_lighting_fixed avoid_lighting_diverse \
    --budgets "$BUDGET" \
    --seeds "${SEEDS[@]}" \
    > "$LOG_DIR/aggregate.log" 2>&1; then
    log_work "ID eval 128px collection completed successfully; final summary written to $OUTPUT_DIR/collection_summary.json."
  else
    failures=$((failures + 1))
    log_work "ID eval workers finished, but final aggregation failed. See $LOG_DIR/aggregate.log."
  fi
else
  log_work "ID eval collection finished with $failures worker failure(s). See $STATUS_FILE."
fi

exit "$failures"
