#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jaisharma/HW8/research2}"
PYTHON="${PYTHON:-/home/jaisharma/miniconda3/envs/pcb/bin/python}"
CONFIG="${CONFIG:-configs/dataset_128px_v1.yaml}"
SHARD_OUTPUT_DIR="${SHARD_OUTPUT_DIR:-results/datasets_128px_v1_phase_precise_avoid_shards}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-results/datasets_128px_v1_phase_precise_avoid}"
BUDGET="${BUDGET:-200}"
SEED="${SEED:-0}"
SHARDS_PER_CONFIG="${SHARDS_PER_CONFIG:-4}"
MAX_STEPS="${MAX_STEPS:-600}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.001}"
RUN_ID="${RUN_ID:-sharded_avoid_$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_CONFIGS="${TRAIN_CONFIGS:-avoid_color_red_only avoid_color_multi avoid_spatial_narrow avoid_spatial_wide avoid_camera_fixed avoid_camera_multi_pose avoid_lighting_fixed avoid_lighting_diverse}"

cd "$ROOT"

read -r -a TRAIN_CONFIG_LIST <<< "$TRAIN_CONFIGS"
LOG_DIR="logs/sharded_avoid_collection_${RUN_ID}"
WORKER_PID_FILE="$LOG_DIR/worker_pids.tsv"
STATUS_FILE="$LOG_DIR/worker_status.tsv"
mkdir -p "$LOG_DIR" "$SHARD_OUTPUT_DIR" "$MERGED_OUTPUT_DIR"

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

trap 'log_work "Sharded avoid collection supervisor $$ received termination signal; stopping workers."; terminate_workers; exit 130' INT TERM

printf "worker_index\tpid\ttrain_config\tshard\tepisode_start\tepisode_end\n" > "$WORKER_PID_FILE"
printf "worker_index\tpid\tstatus\n" > "$STATUS_FILE"

if [ "$SHARDS_PER_CONFIG" -lt 1 ]; then
  echo "SHARDS_PER_CONFIG must be >= 1"
  exit 2
fi
if [ "$((BUDGET % SHARDS_PER_CONFIG))" -ne 0 ]; then
  echo "BUDGET must be divisible by SHARDS_PER_CONFIG"
  exit 2
fi

episodes_per_shard="$((BUDGET / SHARDS_PER_CONFIG))"
worker_index=0
log_work "Started sharded precise obstacle-aware collection supervisor $$ with ${#TRAIN_CONFIG_LIST[@]} configs, shards_per_config=$SHARDS_PER_CONFIG, total_workers=$((${#TRAIN_CONFIG_LIST[@]} * SHARDS_PER_CONFIG)), budget=$BUDGET, seed=$SEED, max_steps=$MAX_STEPS, success_threshold=$SUCCESS_THRESHOLD. Shards: $SHARD_OUTPUT_DIR. Merged: $MERGED_OUTPUT_DIR. Logs: $LOG_DIR."

for train_config in "${TRAIN_CONFIG_LIST[@]}"; do
  for shard_index in $(seq 0 "$((SHARDS_PER_CONFIG - 1))"); do
    episode_start="$((shard_index * episodes_per_shard))"
    episode_end="$(((shard_index + 1) * episodes_per_shard))"
    worker_name=$(printf "worker_%03d" "$worker_index")
    shard_suffix=$(printf "__shard%02d_of%02d" "$shard_index" "$SHARDS_PER_CONFIG")
    worker_log="$LOG_DIR/${worker_name}.log"
    worker_summary="$SHARD_OUTPUT_DIR/collection_summary_${RUN_ID}_${worker_name}.json"
    (
      set -euo pipefail
      "$PYTHON" code/collect_dataset.py \
        --config "$CONFIG" \
        --output-dir "$SHARD_OUTPUT_DIR" \
        --train-configs "$train_config" \
        --budgets "$BUDGET" \
        --seeds "$SEED" \
        --max-steps-per-demo "$MAX_STEPS" \
        --success-threshold "$SUCCESS_THRESHOLD" \
        --episode-start "$episode_start" \
        --episode-end "$episode_end" \
        --output-suffix "$shard_suffix" \
        --summary-path "$worker_summary"
    ) > "$worker_log" 2>&1 &
    pid=$!
    printf "%03d\t%s\t%s\t%02d\t%s\t%s\n" "$worker_index" "$pid" "$train_config" "$shard_index" "$episode_start" "$episode_end" >> "$WORKER_PID_FILE"
    log_work "Launched sharded avoid collection $worker_name PID $pid for $train_config shard $shard_index episodes [$episode_start, $episode_end)."
    worker_index="$((worker_index + 1))"
  done
done

failures=0
while IFS=$'\t' read -r idx pid train_config shard_index episode_start episode_end; do
  if [ "$idx" = "worker_index" ]; then
    continue
  fi
  if wait "$pid"; then
    status="ok"
  else
    status="failed"
    failures="$((failures + 1))"
  fi
  printf "%s\t%s\t%s\n" "$idx" "$pid" "$status" >> "$STATUS_FILE"
  log_work "Sharded avoid collection worker $(printf "worker_%03d" "$((10#$idx))") finished with status $status for $train_config shard $shard_index."
done < "$WORKER_PID_FILE"

if [ "$failures" -ne 0 ]; then
  log_work "Sharded avoid collection finished with $failures worker failure(s). See $STATUS_FILE."
  exit "$failures"
fi

"$PYTHON" code/merge_dataset_shards.py \
  --input-dir "$SHARD_OUTPUT_DIR" \
  --output-dir "$MERGED_OUTPUT_DIR" \
  --train-configs "${TRAIN_CONFIG_LIST[@]}" \
  --budget "$BUDGET" \
  --seed "$SEED" \
  --shard-count "$SHARDS_PER_CONFIG" \
  --summary-path "$MERGED_OUTPUT_DIR/shard_merge_${RUN_ID}.json" \
  > "$LOG_DIR/merge.log" 2>&1

log_work "Sharded precise obstacle-aware collection completed successfully and merged canonical datasets. Merged output: $MERGED_OUTPUT_DIR."
