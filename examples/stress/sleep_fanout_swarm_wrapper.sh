#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${HOME}/.flyte/config-dogfood.yaml"
RUN_NAME=""
REQUESTED_RUN_NAME=""
SWARM_SIZE=2
RUNS_PER_WORKER=1
MAX_RPS=1
N_CHILDREN=5000
SLEEP_DURATION=800
POLL_INTERVAL=2
ABORT_REASON="wrapper interrupted"
PROJECT=""
DOMAIN=""
IMAGE_REGISTRY="${FLYTE_STRESS_IMAGE_REGISTRY:-376129846803.dkr.ecr.us-east-2.amazonaws.com/union}"
IMAGE_NAME="${FLYTE_STRESS_IMAGE_NAME:-dogfood}"
IMAGE_BUILDER="${FLYTE_STRESS_IMAGE_BUILDER:-remote}"
RUN_ENV_ARGS=()

LAUNCH_PID=""
LAUNCH_LOG=""
LAUNCH_DONE=0
LAUNCH_RC=0
RUN_VISIBLE=0
ABORT_SENT=0
STOPPING=0

SCRIPT_START_EPOCH="$(date +%s)"
RUN_VISIBLE_AT=""
FIRST_RUNNING_AT=""
ROOT_TERMINAL_AT=""

PEAK_SEEN=0
PEAK_RUNNING=0
PEAK_ACTIVE=0
LAST_LAUNCH_STAGE=""

EXPECTED_TOP_ACTIONS=0
EXPECTED_CHILD_RUNS=0
EXPECTED_TOTAL_CHILDREN=0
CHILD_RUNS=()
CHILD_RUNS_DISCOVERED_AT=""
AGG_FIRST_RUNNING_AT=""
AGG_ALL_VISIBLE_AT=""
AGG_TERMINAL_AT=""
AGG_PEAK_SEEN=0
AGG_PEAK_RUNNING=0
AGG_PEAK_ACTIVE=0
FINAL_ROOT_ACTIONS_JSON=""

usage() {
  cat <<'EOF'
Usage:
  examples/stress/sleep_fanout_swarm_wrapper.sh [options]

Options:
  --config PATH           Flyte config path. Default: ~/.flyte/config-dogfood.yaml
  --project NAME          Override project for launch/get/abort.
  --domain NAME           Override domain for launch/get/abort.
  --run-name NAME         Use a fixed run name.
  --image-registry VALUE  Registry prefix for the task image. Default: 376129846803.dkr.ecr.us-east-2.amazonaws.com/union
  --image-name VALUE      Repository name for the task image. Default: dogfood
  --image-builder VALUE   Flyte image builder to use. Default: remote
  --run-env KEY=VALUE     Pass through to 'flyte run --env'. Can be specified multiple times.
  --swarm-size INT        Number of submitter tasks. Default: 2
  --runs-per-worker INT   Number of sleep_fanout runs each submitter launches. Default: 1
  --max-rps INT           Max submissions per second per submitter. Default: 1
  --n-children INT        Number of leaves per sleep_fanout run. Default: 5000
  --sleep-duration VALUE  Sleep duration passed to each child run. Default: 800
  --poll-interval SEC     Poll interval in seconds. Default: 2
  --abort-reason TEXT     Reason passed to 'flyte abort run'. Default: wrapper interrupted
  --help                  Show this message.

Notes:
  This wrapper first monitors the swarm root run (`main` + `primer` + submitters).
  After the root finishes submitting, it discovers child run IDs from submitter
  logs and then switches to aggregate leaf-action counts across those child runs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --domain)
      DOMAIN="$2"
      shift 2
      ;;
    --run-name)
      REQUESTED_RUN_NAME="$2"
      shift 2
      ;;
    --image-registry)
      IMAGE_REGISTRY="$2"
      shift 2
      ;;
    --image-name)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --image-builder)
      IMAGE_BUILDER="$2"
      shift 2
      ;;
    --run-env)
      RUN_ENV_ARGS+=("--env" "$2")
      shift 2
      ;;
    --swarm-size)
      SWARM_SIZE="$2"
      shift 2
      ;;
    --runs-per-worker)
      RUNS_PER_WORKER="$2"
      shift 2
      ;;
    --max-rps)
      MAX_RPS="$2"
      shift 2
      ;;
    --n-children)
      N_CHILDREN="$2"
      shift 2
      ;;
    --sleep-duration)
      SLEEP_DURATION="$2"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL="$2"
      shift 2
      ;;
    --abort-reason)
      ABORT_REASON="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

RUN_NAME="${REQUESTED_RUN_NAME}"
EXPECTED_TOP_ACTIONS=$((SWARM_SIZE + 2))
EXPECTED_CHILD_RUNS=$((SWARM_SIZE * RUNS_PER_WORKER))
EXPECTED_TOTAL_CHILDREN=$((EXPECTED_CHILD_RUNS * N_CHILDREN))

if ! command -v flyte >/dev/null 2>&1; then
  echo "flyte is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but was not found in PATH." >&2
  exit 1
fi

CONFIG="${CONFIG/#\~/${HOME}}"
LAUNCH_LOG="$(mktemp "${TMPDIR:-/tmp}/sleep-fanout-swarm-launch.XXXXXX.log")"

cleanup() {
  if [[ -n "${LAUNCH_LOG}" && -f "${LAUNCH_LOG}" ]]; then
    rm -f "${LAUNCH_LOG}"
  fi
}
trap cleanup EXIT

flyte_cmd() {
  _U_USE_ACTIONS="${_U_USE_ACTIONS:-1}" flyte -c "${CONFIG}" --image-builder "${IMAGE_BUILDER}" "$@"
}

flyte_cmd_json() {
  COLUMNS=500 _U_USE_ACTIONS="${_U_USE_ACTIONS:-1}" flyte -c "${CONFIG}" --image-builder "${IMAGE_BUILDER}" -of json-raw "$@" \
    | perl -pe 's/\e\[[0-9;]*[A-Za-z]//g'
}

project_args=()
domain_args=()
run_args=()

if [[ -n "${PROJECT}" ]]; then
  project_args=(-p "${PROJECT}")
  run_args+=(-p "${PROJECT}")
fi

if [[ -n "${DOMAIN}" ]]; then
  domain_args=(-d "${DOMAIN}")
  run_args+=(-d "${DOMAIN}")
fi

abort_remote_run() {
  if [[ "${ABORT_SENT}" -eq 1 || "${RUN_VISIBLE}" -eq 0 || -z "${RUN_NAME}" ]]; then
    return
  fi

  ABORT_SENT=1
  echo
  echo "Requesting abort for swarm root run ${RUN_NAME}..."
  echo "Already-submitted child runs are separate runs and may continue."
  if ! flyte_cmd abort run "${project_args[@]}" "${domain_args[@]}" --reason "${ABORT_REASON}" "${RUN_NAME}"; then
    echo "Abort request failed for run ${RUN_NAME}." >&2
  fi
}

handle_signal() {
  local sig="$1"
  if [[ "${STOPPING}" -eq 1 ]]; then
    echo
    echo "Received ${sig} again, exiting immediately."
    exit 130
  fi
  STOPPING=1

  echo
  echo "Received ${sig}, requesting abort for swarm root run ${RUN_NAME:-<pending>}."
  echo "Continuing to monitor until the current section reaches a terminal phase. Press Ctrl-C again to exit immediately."

  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill "${LAUNCH_PID}" 2>/dev/null || true
  fi

  abort_remote_run
}

trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

format_duration() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo "n/a"
    return
  fi

  local seconds="$value"
  printf '%02dh:%02dm:%02ds' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

elapsed_from_start() {
  local epoch="$1"
  if [[ -z "${epoch}" ]]; then
    echo ""
    return
  fi
  echo "$((epoch - SCRIPT_START_EPOCH))"
}

is_terminal_phase() {
  case "$1" in
    ACTION_PHASE_SUCCEEDED|ACTION_PHASE_FAILED|ACTION_PHASE_ABORTED|ACTION_PHASE_TIMED_OUT)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

fetch_actions_json() {
  flyte_cmd_json get action "${project_args[@]}" "${domain_args[@]}" "${RUN_NAME}"
}

fetch_actions_json_for_run() {
  local run_name="$1"
  flyte_cmd_json get action "${project_args[@]}" "${domain_args[@]}" "${run_name}"
}

fetch_action_logs() {
  local run_name="$1"
  local action_name="$2"
  COLUMNS=500 _U_USE_ACTIONS="${_U_USE_ACTIONS:-1}" flyte -c "${CONFIG}" --image-builder "${IMAGE_BUILDER}" \
    get logs "${project_args[@]}" "${domain_args[@]}" "${run_name}" "${action_name}" 2>/dev/null \
    | perl -pe 's/\e\[[0-9;]*[A-Za-z]//g'
}

sanitize_run_name() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr -d '\r')"
  value="$(printf '%s' "${value}" | sed 's/[[:space:]]*$//')"
  value="$(printf '%s' "${value}" | grep -Eo '[ur][[:alnum:]]{5,}' | head -n 1 || true)"
  printf '%s' "${value}"
}

resolve_run_name_from_log() {
  if [[ -z "${LAUNCH_LOG}" || ! -s "${LAUNCH_LOG}" ]]; then
    return 1
  fi

  local parsed=""
  parsed="$(
    perl -pe 's/\e\[[0-9;]*[A-Za-z]//g' "${LAUNCH_LOG}" \
      | sed -n 's/.*Created Run: //p' \
      | tail -n 1
  )"

  if [[ -z "${parsed}" ]]; then
    parsed="$(
      perl -pe 's/\e\[[0-9;]*[A-Za-z]//g' "${LAUNCH_LOG}" \
        | sed -n 's#.*URL: .*/runs/\([^/?[:space:]]*\).*#\1#p' \
        | tail -n 1
    )"
  fi

  if [[ -z "${parsed}" ]]; then
    return 1
  fi

  parsed="$(sanitize_run_name "${parsed}")"
  if [[ -z "${parsed}" ]]; then
    return 1
  fi

  RUN_NAME="${parsed}"
  return 0
}

launch_stage_from_log() {
  if [[ -z "${LAUNCH_LOG}" || ! -s "${LAUNCH_LOG}" ]]; then
    return 1
  fi

  local lines=""
  local stage=""

  lines="$(
    perl -pe 's/\e\[[0-9;]*[A-Za-z]//g; s/\r/\n/g' "${LAUNCH_LOG}" \
      | sed '/^[[:space:]]*$/d' \
      | tail -n 100
  )"

  if [[ -z "${lines}" ]]; then
    return 1
  fi

  stage="$(
    printf '%s\n' "${lines}" \
      | grep -E '^(Building|Pushing|Image |Created Run:|URL:|Error:|ERROR|Failed|failed|Using |#)' \
      | tail -n 1 || true
  )"

  if [[ -z "${stage}" ]]; then
    stage="$(printf '%s\n' "${lines}" | tail -n 1)"
  fi

  if [[ -n "${stage}" ]]; then
    printf '%s' "${stage}"
    return 0
  fi

  return 1
}

print_row() {
  printf '%-8s %-28s %-12s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n' \
    "$(date '+%H:%M:%S')" \
    "$1" \
    "$2" \
    "$3" \
    "$4" \
    "$5" \
    "$6" \
    "$7" \
    "$8" \
    "$9" \
    "${10}"
}

print_agg_row() {
  printf '%-8s %-12s %-18s %-14s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n' \
    "$(date '+%H:%M:%S')" \
    "$1" \
    "$2" \
    "$3" \
    "$4" \
    "$5" \
    "$6" \
    "$7" \
    "$8" \
    "$9" \
    "${10}"
}

snapshot_tsv() {
  local actions_json="$1"
  jq -r \
    --argjson expected "${EXPECTED_TOP_ACTIONS}" \
    '
      [ .[] ] as $all
      | ($all | map(select(.id.name == "a0")) | .[0]) as $root
      | [ $all[] | select(.id.name != "a0") ] as $kids
      | {
          root_phase: ($root.status.phase // "MISSING"),
          seen: ($kids | length),
          queued: ($kids | map(select(.status.phase == "ACTION_PHASE_QUEUED")) | length),
          waiting: ($kids | map(select(.status.phase == "ACTION_PHASE_WAITING_FOR_RESOURCES")) | length),
          initializing: ($kids | map(select(.status.phase == "ACTION_PHASE_INITIALIZING")) | length),
          running: ($kids | map(select(.status.phase == "ACTION_PHASE_RUNNING")) | length),
          succeeded: ($kids | map(select(.status.phase == "ACTION_PHASE_SUCCEEDED")) | length),
          failed: ($kids | map(select(.status.phase == "ACTION_PHASE_FAILED")) | length),
          aborted: ($kids | map(select(.status.phase == "ACTION_PHASE_ABORTED")) | length),
          timed_out: ($kids | map(select(.status.phase == "ACTION_PHASE_TIMED_OUT")) | length)
        }
      | .not_seen = (if $expected > .seen then ($expected - .seen) else 0 end)
      | .active = (.queued + .waiting + .initializing + .running)
      | [
          .root_phase,
          .seen,
          .not_seen,
          .queued,
          .waiting,
          .initializing,
          .running,
          .active,
          .succeeded,
          .failed,
          .aborted,
          .timed_out
        ]
      | @tsv
    ' <<<"${actions_json}"
}

child_run_known() {
  local target="$1"
  local existing=""
  for existing in "${CHILD_RUNS[@]}"; do
    if [[ "${existing}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

discover_child_runs_from_root() {
  local root_actions_json="$1"
  local action_name=""
  local child_run=""

  while IFS= read -r action_name; do
    [[ -z "${action_name}" ]] && continue
    while IFS= read -r child_run; do
      [[ -z "${child_run}" ]] && continue
      if ! child_run_known "${child_run}"; then
        CHILD_RUNS+=("${child_run}")
      fi
    done < <(
      fetch_action_logs "${RUN_NAME}" "${action_name}" \
        | perl -ne '
            if (/submitted_run idx=\d+ url=.*\/runs\/([^\/?\s]+)/) {
              print "$1\n";
            } elsif (/submitted_run idx=\d+ url=([ur][[:alnum:]]{5,})/) {
              print "$1\n";
            }
          '
    )
  done < <(jq -r '.[] | select(.id.name != "a0") | .id.name' <<<"${root_actions_json}")
}

aggregate_child_runs_tsv() {
  local discovered=0
  local roots_terminal=0
  local seen=0
  local queued=0
  local waiting=0
  local initializing=0
  local running=0
  local succeeded=0
  local failed=0
  local aborted=0
  local timed_out=0
  local run_name=""
  local json=""
  local root_phase=""
  local c_seen=0
  local c_queued=0
  local c_waiting=0
  local c_initializing=0
  local c_running=0
  local c_succeeded=0
  local c_failed=0
  local c_aborted=0
  local c_timed_out=0
  local not_created=0
  local active=0

  discovered="${#CHILD_RUNS[@]}"
  for run_name in "${CHILD_RUNS[@]}"; do
    if ! json="$(fetch_actions_json_for_run "${run_name}" 2>/dev/null)"; then
      continue
    fi
    IFS=$'\t' read -r root_phase c_seen c_queued c_waiting c_initializing c_running c_succeeded c_failed c_aborted c_timed_out \
      <<<"$(jq -r '
        [ .[] ] as $all
        | ($all | map(select(.id.name == "a0")) | .[0]) as $root
        | [ $all[] | select(.id.name != "a0") ] as $kids
        | [
            ($root.status.phase // "MISSING"),
            ($kids | length),
            ($kids | map(select(.status.phase == "ACTION_PHASE_QUEUED")) | length),
            ($kids | map(select(.status.phase == "ACTION_PHASE_WAITING_FOR_RESOURCES")) | length),
            ($kids | map(select(.status.phase == "ACTION_PHASE_INITIALIZING")) | length),
            ($kids | map(select(.status.phase == "ACTION_PHASE_RUNNING")) | length),
            ($kids | map(select(.status.phase == "ACTION_PHASE_SUCCEEDED")) | length),
            ($kids | map(select(.status.phase == "ACTION_PHASE_FAILED")) | length),
            ($kids | map(select(.status.phase == "ACTION_PHASE_ABORTED")) | length),
            ($kids | map(select(.status.phase == "ACTION_PHASE_TIMED_OUT")) | length)
          ]
        | @tsv
      ' <<<"${json}")"

    if is_terminal_phase "${root_phase}"; then
      roots_terminal=$((roots_terminal + 1))
    fi
    seen=$((seen + c_seen))
    queued=$((queued + c_queued))
    waiting=$((waiting + c_waiting))
    initializing=$((initializing + c_initializing))
    running=$((running + c_running))
    succeeded=$((succeeded + c_succeeded))
    failed=$((failed + c_failed))
    aborted=$((aborted + c_aborted))
    timed_out=$((timed_out + c_timed_out))
  done

  if (( EXPECTED_TOTAL_CHILDREN > seen )); then
    not_created=$((EXPECTED_TOTAL_CHILDREN - seen))
  fi
  active=$((queued + waiting + initializing + running))

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${discovered}/${EXPECTED_CHILD_RUNS}" \
    "${seen}/${EXPECTED_TOTAL_CHILDREN}" \
    "${not_created}" \
    "${queued}" \
    "${waiting}" \
    "${initializing}" \
    "${running}" \
    "${active}" \
    "${succeeded}" \
    "${failed}" \
    "${aborted}" \
    "${timed_out}" \
    "${roots_terminal}"
}

print_root_summary() {
  local root_phase="$1"
  local seen="$2"
  local succeeded="$3"
  local failed="$4"
  local aborted="$5"
  local timed_out="$6"
  local total_elapsed="$(( $(date +%s) - SCRIPT_START_EPOCH ))"

  echo
  echo "Root Summary"
  echo "  run_name: ${RUN_NAME:-<unresolved>}"
  echo "  root_phase: ${root_phase}"
  echo "  abort_requested: $(if [[ "${STOPPING}" -eq 1 ]]; then echo yes; else echo no; fi)"
  echo "  top_level_actions_seen: ${seen}/${EXPECTED_TOP_ACTIONS}"
  echo "  succeeded: ${succeeded}"
  echo "  failed: ${failed}"
  echo "  aborted: ${aborted}"
  echo "  timed_out: ${timed_out}"
  echo "  peak_seen: ${PEAK_SEEN}/${EXPECTED_TOP_ACTIONS}"
  echo "  peak_running: ${PEAK_RUNNING}"
  echo "  peak_active: ${PEAK_ACTIVE}"
  echo "  time_to_run_visible: $(format_duration "$(elapsed_from_start "${RUN_VISIBLE_AT}")")"
  echo "  time_to_first_running: $(format_duration "$(elapsed_from_start "${FIRST_RUNNING_AT}")")"
  echo "  time_to_root_terminal: $(format_duration "$(elapsed_from_start "${ROOT_TERMINAL_AT}")")"
  echo "  total_elapsed: $(format_duration "${total_elapsed}")"
}

print_agg_summary() {
  local discovered="$1"
  local roots_terminal="$2"
  local seen="$3"
  local succeeded="$4"
  local failed="$5"
  local aborted="$6"
  local timed_out="$7"
  local total_elapsed="$(( $(date +%s) - SCRIPT_START_EPOCH ))"

  echo
  echo "Aggregate Summary"
  echo "  child_runs_discovered: ${discovered}/${EXPECTED_CHILD_RUNS}"
  echo "  total_expected_children: ${EXPECTED_TOTAL_CHILDREN}"
  echo "  child_run_roots_terminal: ${roots_terminal}/${discovered}"
  echo "  children_seen: ${seen}/${EXPECTED_TOTAL_CHILDREN}"
  echo "  succeeded: ${succeeded}"
  echo "  failed: ${failed}"
  echo "  aborted: ${aborted}"
  echo "  timed_out: ${timed_out}"
  echo "  peak_seen: ${AGG_PEAK_SEEN}/${EXPECTED_TOTAL_CHILDREN}"
  echo "  peak_running: ${AGG_PEAK_RUNNING}"
  echo "  peak_active: ${AGG_PEAK_ACTIVE}"
  echo "  child_runs_discovered_at: $(format_duration "$(elapsed_from_start "${CHILD_RUNS_DISCOVERED_AT}")")"
  echo "  aggregate_first_running: $(format_duration "$(elapsed_from_start "${AGG_FIRST_RUNNING_AT}")")"
  echo "  aggregate_all_visible: $(format_duration "$(elapsed_from_start "${AGG_ALL_VISIBLE_AT}")")"
  echo "  aggregate_terminal: $(format_duration "$(elapsed_from_start "${AGG_TERMINAL_AT}")")"
  echo "  total_elapsed: $(format_duration "${total_elapsed}")"
}

cd "${REPO_ROOT}"

if [[ -n "${RUN_NAME}" ]]; then
  echo "Launching swarm run ${RUN_NAME}"
else
  echo "Launching swarm run with generated actions name"
fi
echo "  config: ${CONFIG}"
echo "  swarm_size: ${SWARM_SIZE}"
echo "  runs_per_worker: ${RUNS_PER_WORKER}"
echo "  max_rps: ${MAX_RPS}"
echo "  n_children_per_run: ${N_CHILDREN}"
echo "  sleep_duration: ${SLEEP_DURATION}"
echo "  poll_interval: ${POLL_INTERVAL}s"
echo "  image target: ${IMAGE_REGISTRY}/${IMAGE_NAME}"
echo "  image builder: ${IMAGE_BUILDER}"
echo "  image platforms: ${FLYTE_STRESS_IMAGE_PLATFORMS:-linux/amd64}"
if [[ -n "${PROJECT}" || -n "${DOMAIN}" ]]; then
  echo "  project/domain override: ${PROJECT:-<config>} / ${DOMAIN:-<config>}"
fi
if [[ "${#RUN_ENV_ARGS[@]}" -gt 0 ]]; then
  echo "  run env overrides: ${RUN_ENV_ARGS[*]}"
fi
echo "  child_runs_expected: ${EXPECTED_CHILD_RUNS}"
echo "  total_children_expected: ${EXPECTED_TOTAL_CHILDREN}"
echo
printf '%-8s %-28s %-12s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n' \
  "time" "root_phase" "seen_top" "not_seen" "queued" "waiting" "init" "running" "active" "ok" "aborted"

(
  export FLYTE_STRESS_IMAGE_REGISTRY="${IMAGE_REGISTRY}"
  export FLYTE_STRESS_IMAGE_NAME="${IMAGE_NAME}"
  if [[ -n "${REQUESTED_RUN_NAME}" ]]; then
    flyte_cmd run "${run_args[@]}" "${RUN_ENV_ARGS[@]}" --name "${REQUESTED_RUN_NAME}" \
      examples/stress/sleep_fanout.py main \
      --swarm_size "${SWARM_SIZE}" \
      --runs_per_worker "${RUNS_PER_WORKER}" \
      --max_rps "${MAX_RPS}" \
      --n_children "${N_CHILDREN}" \
      --sleep_duration "${SLEEP_DURATION}"
  else
    flyte_cmd run "${run_args[@]}" "${RUN_ENV_ARGS[@]}" \
      examples/stress/sleep_fanout.py main \
      --swarm_size "${SWARM_SIZE}" \
      --runs_per_worker "${RUNS_PER_WORKER}" \
      --max_rps "${MAX_RPS}" \
      --n_children "${N_CHILDREN}" \
      --sleep_duration "${SLEEP_DURATION}"
  fi
) >"${LAUNCH_LOG}" 2>&1 &
LAUNCH_PID=$!

FINAL_ROOT_PHASE="UNKNOWN"
FINAL_SEEN=0
FINAL_SUCCEEDED=0
FINAL_FAILED=0
FINAL_ABORTED=0
FINAL_TIMED_OUT=0

while true; do
  if [[ -z "${RUN_NAME}" ]] && resolve_run_name_from_log; then
    echo
    echo "Resolved run name: ${RUN_NAME}"
  fi

  if [[ "${LAUNCH_DONE}" -eq 0 ]] && ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    if wait "${LAUNCH_PID}"; then
      LAUNCH_RC=0
    else
      LAUNCH_RC=$?
    fi
    LAUNCH_DONE=1

    if [[ "${LAUNCH_RC}" -eq 0 ]]; then
      echo
      echo "Launch command completed for run ${RUN_NAME:-<unresolved>}."
    elif [[ "${RUN_VISIBLE}" -eq 0 ]]; then
      echo
      echo "Launch command failed before the run became visible." >&2
      cat "${LAUNCH_LOG}" >&2
      exit "${LAUNCH_RC}"
    else
      echo
      echo "Launch command exited with ${LAUNCH_RC}, but the run is already visible. Continuing to monitor." >&2
      cat "${LAUNCH_LOG}" >&2
    fi
  fi

  if [[ -n "${RUN_NAME}" ]] && actions_json="$(fetch_actions_json 2>/dev/null)"; then
    now_epoch="$(date +%s)"
    RUN_VISIBLE=1
    FINAL_ROOT_ACTIONS_JSON="${actions_json}"
    if [[ -z "${RUN_VISIBLE_AT}" ]]; then
      RUN_VISIBLE_AT="${now_epoch}"
    fi

    if [[ "${STOPPING}" -eq 1 && "${ABORT_SENT}" -eq 0 ]]; then
      abort_remote_run
    fi

    IFS=$'\t' read -r root_phase seen not_seen queued waiting initializing running active succeeded failed aborted timed_out \
      <<<"$(snapshot_tsv "${actions_json}")"

    FINAL_ROOT_PHASE="${root_phase}"
    FINAL_SEEN="${seen}"
    FINAL_SUCCEEDED="${succeeded}"
    FINAL_FAILED="${failed}"
    FINAL_ABORTED="${aborted}"
    FINAL_TIMED_OUT="${timed_out}"

    if (( seen > PEAK_SEEN )); then
      PEAK_SEEN="${seen}"
    fi
    if (( running > PEAK_RUNNING )); then
      PEAK_RUNNING="${running}"
    fi
    if (( active > PEAK_ACTIVE )); then
      PEAK_ACTIVE="${active}"
    fi
    if [[ -z "${FIRST_RUNNING_AT}" && "${running}" -gt 0 ]]; then
      FIRST_RUNNING_AT="${now_epoch}"
    fi

    print_row \
      "${root_phase}" \
      "${seen}/${EXPECTED_TOP_ACTIONS}" \
      "${not_seen}" \
      "${queued}" \
      "${waiting}" \
      "${initializing}" \
      "${running}" \
      "${active}" \
      "${succeeded}" \
      "${aborted}"

    if is_terminal_phase "${root_phase}"; then
      ROOT_TERMINAL_AT="${now_epoch}"
      break
    fi
  else
    if stage="$(launch_stage_from_log)"; then
      if [[ "${stage}" != "${LAST_LAUNCH_STAGE}" ]]; then
        LAST_LAUNCH_STAGE="${stage}"
        echo "launch: ${stage}"
      fi
    fi
    print_row "RESOLVING_RUN" "0/${EXPECTED_TOP_ACTIONS}" "${EXPECTED_TOP_ACTIONS}" 0 0 0 0 0 0 0
  fi

  sleep "${POLL_INTERVAL}"
done

if [[ "${LAUNCH_DONE}" -eq 0 ]]; then
  if wait "${LAUNCH_PID}"; then
    true
  else
    true
  fi
fi

print_root_summary \
  "${FINAL_ROOT_PHASE}" \
  "${FINAL_SEEN}" \
  "${FINAL_SUCCEEDED}" \
  "${FINAL_FAILED}" \
  "${FINAL_ABORTED}" \
  "${FINAL_TIMED_OUT}"

if [[ "${FINAL_ROOT_PHASE}" == "ACTION_PHASE_SUCCEEDED" && -n "${FINAL_ROOT_ACTIONS_JSON}" ]]; then
  discover_child_runs_from_root "${FINAL_ROOT_ACTIONS_JSON}"

  if [[ "${#CHILD_RUNS[@]}" -gt 0 ]]; then
    CHILD_RUNS_DISCOVERED_AT="$(date +%s)"

    echo
    echo "Discovered child runs: ${#CHILD_RUNS[@]}/${EXPECTED_CHILD_RUNS}"
    printf '%s\n' "${CHILD_RUNS[@]}"
    echo
    printf '%-8s %-12s %-18s %-14s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n' \
      "time" "runs" "seen_children" "not_created" "queued" "waiting" "init" "running" "active" "ok" "aborted"

    AGG_FINAL_DISCOVERED=0
    AGG_FINAL_SEEN=0
    AGG_FINAL_SUCCEEDED=0
    AGG_FINAL_FAILED=0
    AGG_FINAL_ABORTED=0
    AGG_FINAL_TIMED_OUT=0
    AGG_FINAL_ROOTS_TERMINAL=0

    while true; do
      IFS=$'\t' read -r runs seen_children not_created queued waiting initializing running active succeeded failed aborted timed_out roots_terminal \
        <<<"$(aggregate_child_runs_tsv)"

      AGG_FINAL_DISCOVERED="${runs%%/*}"
      AGG_FINAL_SEEN="${seen_children%%/*}"
      AGG_FINAL_SUCCEEDED="${succeeded}"
      AGG_FINAL_FAILED="${failed}"
      AGG_FINAL_ABORTED="${aborted}"
      AGG_FINAL_TIMED_OUT="${timed_out}"
      AGG_FINAL_ROOTS_TERMINAL="${roots_terminal}"

      if (( AGG_FINAL_SEEN > AGG_PEAK_SEEN )); then
        AGG_PEAK_SEEN="${AGG_FINAL_SEEN}"
      fi
      if (( running > AGG_PEAK_RUNNING )); then
        AGG_PEAK_RUNNING="${running}"
      fi
      if (( active > AGG_PEAK_ACTIVE )); then
        AGG_PEAK_ACTIVE="${active}"
      fi
      if [[ -z "${AGG_FIRST_RUNNING_AT}" && "${running}" -gt 0 ]]; then
        AGG_FIRST_RUNNING_AT="$(date +%s)"
      fi
      if [[ -z "${AGG_ALL_VISIBLE_AT}" && "${AGG_FINAL_SEEN}" -ge "${EXPECTED_TOTAL_CHILDREN}" ]]; then
        AGG_ALL_VISIBLE_AT="$(date +%s)"
      fi

      print_agg_row \
        "${runs}" \
        "${seen_children}" \
        "${not_created}" \
        "${queued}" \
        "${waiting}" \
        "${initializing}" \
        "${running}" \
        "${active}" \
        "${succeeded}" \
        "${aborted}"

      if (( AGG_FINAL_DISCOVERED == EXPECTED_CHILD_RUNS )) && (( AGG_FINAL_ROOTS_TERMINAL == AGG_FINAL_DISCOVERED )) && (( active == 0 )); then
        AGG_TERMINAL_AT="$(date +%s)"
        break
      fi

      sleep "${POLL_INTERVAL}"
    done

    print_agg_summary \
      "${AGG_FINAL_DISCOVERED}" \
      "${AGG_FINAL_ROOTS_TERMINAL}" \
      "${AGG_FINAL_SEEN}" \
      "${AGG_FINAL_SUCCEEDED}" \
      "${AGG_FINAL_FAILED}" \
      "${AGG_FINAL_ABORTED}" \
      "${AGG_FINAL_TIMED_OUT}"
  else
    echo
    echo "No child runs discovered from submitter logs."
  fi
fi
