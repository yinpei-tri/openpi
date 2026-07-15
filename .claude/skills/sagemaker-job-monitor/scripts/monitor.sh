#!/usr/bin/env bash
# Monitor a SageMaker AWS-Batch *service* job for openpi multi-node JAX training.
#
# Usage:
#   monitor.sh <job_substring> [status|logs|watch] [--queue Q] [--profile P] [--region R]
#
#   status  (default) -> print the job's current batch status
#   logs              -> print the health-signal summary from both nodes' logs
#   watch             -> poll every INTERVAL sec until the job leaves RUNNABLE/PENDING,
#                        or (once RUNNING) until real training steps advance or it fails
#
# <job_substring> is any unique part of the job name, e.g. the timestamp "17-05-59".
#
# Env knobs: QUEUE, PROFILE (default sagemaker), REGION (default us-west-2),
#   INTERVAL (watch poll seconds, default 120), MAXPOLL (default 120).
#
# These are AWS Batch SERVICE jobs (SubmitServiceJob), NOT classic SageMaker training
# jobs, so `describe-training-job` returns "resource not found" -- use list-service-jobs.
# Ambient AWS_* env keys override the SSO profile in boto3/CLI, so we unset them.
set -uo pipefail
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN 2>/dev/null || true

JOB="${1:?Usage: monitor.sh <job_substring> [status|logs|watch]}"
MODE="${2:-status}"
PROFILE="${PROFILE:-sagemaker}"
REGION="${REGION:-us-west-2}"
INTERVAL="${INTERVAL:-120}"
MAXPOLL="${MAXPOLL:-120}"
LOG_GROUP="/aws/sagemaker/TrainingJobs"
AWSP="--profile $PROFILE --region $REGION"

# --- queue autodetect: scan all queues for the job if QUEUE unset ------------------
find_queue() {
  [ -n "${QUEUE:-}" ] && { echo "$QUEUE"; return; }
  local qs
  qs=$(aws batch describe-job-queues $AWSP --query "jobQueues[].jobQueueName" --output text 2>/dev/null | tr '\t' '\n')
  for q in $qs; do
    for st in RUNNING STARTING RUNNABLE PENDING SUBMITTED SUCCEEDED FAILED; do
      local hit
      hit=$(aws batch list-service-jobs --job-queue "$q" --job-status $st $AWSP \
            --query "jobSummaryList[?contains(jobName,'$JOB')].jobName" --output text 2>/dev/null)
      [ -n "$hit" ] && [ "$hit" != "None" ] && { echo "$q"; return; }
    done
  done
}

get_status() {
  local q="$1" st out
  for st in RUNNING STARTING RUNNABLE PENDING SUBMITTED SUCCEEDED FAILED; do
    out=$(aws batch list-service-jobs --job-queue "$q" --job-status $st $AWSP \
          --query "jobSummaryList[?contains(jobName,'$JOB')].status" --output text 2>/dev/null)
    [ -n "$out" ] && [ "$out" != "None" ] && { echo "$out"; return; }
  done
  echo "UNKNOWN"
}

# --- find the two node log streams (algo-1 = rank 0, algo-2 = rank 1) --------------
# The service job's streams are AWSBatch<sanitized-jobname-truncated><hash>/algo-N-<ts>.
# The job name is TRUNCATED then hashed, so the timestamp substring (e.g. 17-05-59) is NOT
# recoverable from the stream name. Instead: take the newest `/algo-` streams and let the
# caller content-match. STREAM_HASH env can pin the exact AWSBatch<hash> prefix if known
# (derive it once from `logs` output) to avoid scanning other users' jobs.
find_streams() {
  local raw
  raw=$(aws logs describe-log-streams --log-group-name "$LOG_GROUP" --order-by LastEventTime \
        --descending $AWSP --max-items 50 --query "logStreams[].logStreamName" --output text 2>/dev/null \
        | tr '\t' '\n' | grep -aE "/algo-[0-9]")
  if [ -n "${STREAM_HASH:-}" ]; then
    echo "$raw" | grep -a "$STREAM_HASH" | head -4
  else
    echo "$raw" | head -8
  fi
}

analyze_logs() {
  local streams
  streams=$(find_streams)
  if [ -z "$streams" ]; then echo "(no log streams found yet -- job may still be provisioning)"; return; fi
  # Heuristic: pick the two streams whose events are most recent (the running job).
  # Print health signals per stream.
  # MATCH env pins which streams belong to this job by a string present in the LOG BODY
  # (e.g. the exp_name "multinode_full_ema", or the checkpoint_s3_uri containing the job
  # timestamp). Defaults to the current EMA config's exp_name. Without STREAM_HASH this
  # scans recent streams and keeps the first 2 whose body matches MATCH.
  local match="${MATCH:-multinode_full_ema}"
  local shown=0
  for s in $streams; do
    [ $shown -ge 2 ] && break
    local msgs
    msgs=$(aws logs get-log-events --log-group-name "$LOG_GROUP" --log-stream-name "$s" $AWSP \
           --limit 2500 --query "events[].message" --output text 2>/dev/null | tr '\t' '\n')
    # When STREAM_HASH is pinned, the stream already uniquely identifies this job -> take
    # it. Otherwise require the body to mention MATCH (skips other users' recent streams).
    if [ -z "${STREAM_HASH:-}" ]; then
      echo "$msgs" | grep -aq -- "$match" || continue
    fi
    shown=$((shown+1))
    echo "########## $s ##########"
    echo "$msgs" | grep -aE "Running on:|global devices" | tail -1
    echo "--- data sharding fingerprint (proc=0 and proc=1 MUST differ) ---"
    echo "$msgs" | grep -aE "per-shard image means" | tail -1
    echo "--- shard split (want jax_proc=N/2, not N/1) ---"
    echo "$msgs" | grep -aE "data-shard-split" | tail -1
    echo "--- tentative + real training steps ---"
    echo "$msgs" | grep -aE "Tentative run completed" | tail -1
    echo "$msgs" | awk '/Tentative run completed/{f=1} f&&/Step [0-9]+:/{print}' | tail -3
    echo "$msgs" | grep -aE "Step [0-9]+:" | tail -2
    echo "--- checkpoints / sync ---"
    echo "$msgs" | grep -aE "Waiting for checkpoint|Final checkpoint upload|Periodic checkpoint upload|saved checkpoint|save_state" | tail -2
    echo "--- FAILURES (should be empty) ---"
    echo "$msgs" | grep -aE "Traceback|Shutdown barrier|Aborted|core dumped|Error occurred|NaN|non-trivial shardings" | tail -3
    echo ""
  done
}

QUEUE_RESOLVED="$(find_queue)"
[ -z "$QUEUE_RESOLVED" ] && { echo "Job '$JOB' not found in any queue (check name / profile)."; exit 1; }

case "$MODE" in
  status)
    echo "queue=$QUEUE_RESOLVED  status=$(get_status "$QUEUE_RESOLVED")"
    ;;
  logs)
    echo "queue=$QUEUE_RESOLVED  status=$(get_status "$QUEUE_RESOLVED")"
    analyze_logs
    ;;
  watch)
    i=0
    prev=""
    while [ $i -lt "$MAXPOLL" ]; do
      i=$((i+1))
      st=$(get_status "$QUEUE_RESOLVED")
      echo "$(date -u +%H:%M:%S) poll $i status=$st"
      case "$st" in
        FAILED|SUCCEEDED) echo "TERMINAL -> $st"; analyze_logs; exit 0 ;;
        RUNNING)
          # once running, look for real training progress or a failure
          msgs=$(for s in $(find_streams); do
                   aws logs get-log-events --log-group-name "$LOG_GROUP" --log-stream-name "$s" $AWSP \
                     --limit 5000 --query "events[].message" --output text 2>/dev/null | tr '\t' '\n'
                 done)
          if echo "$msgs" | grep -aqE "Traceback|Shutdown barrier|Aborted|core dumped|non-trivial shardings"; then
            echo "FAILURE SIGNAL DETECTED"; analyze_logs; exit 0
          fi
          nsteps=$(echo "$msgs" | awk '/Tentative run completed/{f=1} f&&/Step [0-9]+:/{c++} END{print c+0}')
          if [ "${nsteps:-0}" -ge 2 ]; then
            echo "REAL TRAINING ADVANCING (${nsteps} post-tentative steps)"; analyze_logs; exit 0
          fi
          ;;
      esac
      sleep "$INTERVAL"
    done
    echo "watch timed out after $((MAXPOLL)) polls (still $st)"
    ;;
  *) echo "Unknown mode: $MODE (use status|logs|watch)"; exit 1 ;;
esac
