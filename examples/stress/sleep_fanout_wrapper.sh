#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${HOME}/.flyte/config-dogfood.yaml"
RUN_NAME=""
REQUESTED_RUN_NAME=""
N_CHILDREN=10
SLEEP_DURATION=10
POLL_INTERVAL=1
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
STOPPING_AT=""
ABORT_NOTE_SHOWN=0
INTERRUPT_GRACE_SEC=15

SCRIPT_START_EPOCH="$(date +%s)"
RUN_VISIBLE_AT=""
ALL_CHILDREN_VISIBLE_AT=""
FIRST_RUNNING_AT=""
FIRST_SUCCESS_AT=""
ROOT_TERMINAL_AT=""

PEAK_SEEN=0
PEAK_RUNNING=0
PEAK_ACTIVE=0
LAST_LAUNCH_STAGE=""

usage() {
  cat <<'EOF'
Usage:
  examples/stress/sleep_fanout_wrapper.sh [options]

Options:
  --config PATH           Flyte config path. Default: ~/.flyte/config-dogfood.yaml
  --project NAME          Override project for launch/get/abort.
  --domain NAME           Override domain for launch/get/abort.
  --run-name NAME         Use a fixed run name. Avoid this on _U_USE_ACTIONS=1 if you want abort to work.
  --image-registry VALUE  Registry prefix for the task image. Default: 376129846803.dkr.ecr.us-east-2.amazonaws.com/union
  --image-name VALUE      Repository name for the task image. Default: dogfood
  --image-builder VALUE   Flyte image builder to use. Default: remote
  --run-env KEY=VALUE     Pass through to 'flyte run --env'. Can be specified multiple times.
  --n-children INT        Number of child actions. Default: 10
  --sleep-duration VALUE  Sleep duration passed to the task. Default: 10
  --poll-interval SEC     Poll interval in seconds. Default: 1
  --abort-reason TEXT     Reason passed to 'flyte abort run'. Default: wrapper interrupted
  --help                  Show this message.

Example:
  examples/stress/sleep_fanout_wrapper.sh \
    --config ~/.flyte/config-dogfood.yaml \
    --n-children 10 \
    --sleep-duration 10
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

if ! command -v flyte >/dev/null 2>&1; then
  echo "flyte is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but was not found in PATH." >&2
  exit 1
fi

CONFIG="${CONFIG/#\~/${HOME}}"
LAUNCH_LOG="$(mktemp "${TMPDIR:-/tmp}/sleep-fanout-launch.XXXXXX.log")"

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
  echo "Requesting abort for run ${RUN_NAME}..."
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
  STOPPING_AT="$(date +%s)"

  echo
  echo "Received ${sig}, requesting abort for run ${RUN_NAME:-<pending>}."
  echo "Continuing to monitor until the run reaches a terminal phase. Press Ctrl-C again to exit immediately."

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
  printf '%-8s %-28s %-14s %-14s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n' \
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
    --argjson expected "${N_CHILDREN}" \
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
      | .not_created = (if $expected > .seen then ($expected - .seen) else 0 end)
      | .active = (.queued + .waiting + .initializing + .running)
      | [
          .root_phase,
          .seen,
          .not_created,
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

print_summary() {
  local root_phase="$1"
  local seen="$2"
  local succeeded="$3"
  local failed="$4"
  local aborted="$5"
  local timed_out="$6"
  local total_elapsed="$(( $(date +%s) - SCRIPT_START_EPOCH ))"

  echo
  echo "Summary"
  echo "  run_name: ${RUN_NAME:-<unresolved>}"
  echo "  root_phase: ${root_phase}"
  echo "  abort_requested: $(if [[ "${STOPPING}" -eq 1 ]]; then echo yes; else echo no; fi)"
  echo "  children_seen: ${seen}/${N_CHILDREN}"
  echo "  succeeded: ${succeeded}"
  echo "  failed: ${failed}"
  echo "  aborted: ${aborted}"
  echo "  timed_out: ${timed_out}"
  echo "  peak_seen: ${PEAK_SEEN}/${N_CHILDREN}"
  echo "  peak_running: ${PEAK_RUNNING}"
  echo "  peak_active: ${PEAK_ACTIVE}"
  echo "  time_to_run_visible: $(format_duration "$(elapsed_from_start "${RUN_VISIBLE_AT}")")"
  echo "  time_to_all_children_visible: $(format_duration "$(elapsed_from_start "${ALL_CHILDREN_VISIBLE_AT}")")"
  echo "  time_to_first_running: $(format_duration "$(elapsed_from_start "${FIRST_RUNNING_AT}")")"
  echo "  time_to_first_success: $(format_duration "$(elapsed_from_start "${FIRST_SUCCESS_AT}")")"
  echo "  time_to_root_terminal: $(format_duration "$(elapsed_from_start "${ROOT_TERMINAL_AT}")")"
  echo "  total_elapsed: $(format_duration "${total_elapsed}")"
}

cd "${REPO_ROOT}"

if [[ -n "${RUN_NAME}" ]]; then
  echo "Launching run ${RUN_NAME}"
else
  echo "Launching run with generated actions name"
fi
echo "  config: ${CONFIG}"
echo "  children: ${N_CHILDREN}"
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
if [[ -n "${REQUESTED_RUN_NAME}" && "${_U_USE_ACTIONS:-1}" == "1" ]]; then
  echo "  warning: custom run names can break abort routing on the actions path"
fi
echo
printf '%-8s %-28s %-14s %-14s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n' \
  "time" "root_phase" "seen" "not_created" "queued" "waiting" "init" "running" "active" "ok" "aborted"

(
  export FLYTE_STRESS_IMAGE_REGISTRY="${IMAGE_REGISTRY}"
  export FLYTE_STRESS_IMAGE_NAME="${IMAGE_NAME}"
  if [[ -n "${REQUESTED_RUN_NAME}" ]]; then
    flyte_cmd run "${run_args[@]}" "${RUN_ENV_ARGS[@]}" --name "${REQUESTED_RUN_NAME}" \
      examples/stress/sleep_fanout.py sleep_fanout \
      --n_children "${N_CHILDREN}" \
      --sleep_duration "${SLEEP_DURATION}"
  else
    flyte_cmd run "${run_args[@]}" "${RUN_ENV_ARGS[@]}" \
      examples/stress/sleep_fanout.py sleep_fanout \
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
    if [[ -z "${RUN_VISIBLE_AT}" ]]; then
      RUN_VISIBLE_AT="${now_epoch}"
    fi

    if [[ "${STOPPING}" -eq 1 && "${ABORT_SENT}" -eq 0 ]]; then
      abort_remote_run
    fi

    IFS=$'\t' read -r root_phase seen not_created queued waiting initializing running active succeeded failed aborted timed_out \
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

    if [[ -z "${ALL_CHILDREN_VISIBLE_AT}" && "${seen}" -eq "${N_CHILDREN}" ]]; then
      ALL_CHILDREN_VISIBLE_AT="${now_epoch}"
    fi
    if [[ -z "${FIRST_RUNNING_AT}" && "${running}" -gt 0 ]]; then
      FIRST_RUNNING_AT="${now_epoch}"
    fi
    if [[ -z "${FIRST_SUCCESS_AT}" && "${succeeded}" -gt 0 ]]; then
      FIRST_SUCCESS_AT="${now_epoch}"
    fi

    print_row \
      "${root_phase}" \
      "${seen}/${N_CHILDREN}" \
      "${not_created}" \
      "${queued}" \
      "${waiting}" \
      "${initializing}" \
      "${running}" \
      "${active}" \
      "${succeeded}" \
      "${aborted}"

    if [[ "${STOPPING}" -eq 1 && "${ABORT_NOTE_SHOWN}" -eq 0 && "${ABORT_SENT}" -eq 1 ]] && ! is_terminal_phase "${root_phase}"; then
      echo "Abort requested. Waiting for root action to become terminal..."
      ABORT_NOTE_SHOWN=1
    fi

    if is_terminal_phase "${root_phase}"; then
      ROOT_TERMINAL_AT="${now_epoch}"
      break
    fi
  else
    if [[ -z "${RUN_NAME}" ]]; then
      launch_stage="$(launch_stage_from_log || true)"
      if [[ -n "${launch_stage}" && "${launch_stage}" != "${LAST_LAUNCH_STAGE}" ]]; then
        echo "launch: ${launch_stage}"
        LAST_LAUNCH_STAGE="${launch_stage}"
      fi
    fi

    if [[ "${LAUNCH_DONE}" -eq 1 && "${LAUNCH_RC}" -ne 0 && "${RUN_VISIBLE}" -eq 0 ]]; then
      echo "Run ${RUN_NAME:-<unresolved>} never became visible after launch failure." >&2
      exit "${LAUNCH_RC}"
    fi

    if [[ -z "${RUN_NAME}" ]]; then
      print_row "RESOLVING_RUN" "0/${N_CHILDREN}" "${N_CHILDREN}" 0 0 0 0 0 0 0
    else
      print_row "NO_ACTION_DATA" "0/${N_CHILDREN}" "${N_CHILDREN}" 0 0 0 0 0 0 0
    fi

    if [[ "${STOPPING}" -eq 1 && "${LAUNCH_DONE}" -eq 1 && "${RUN_VISIBLE}" -eq 0 && -n "${STOPPING_AT}" ]]; then
      if (( $(date +%s) - STOPPING_AT >= INTERRUPT_GRACE_SEC )); then
        echo "Run ${RUN_NAME:-<unresolved>} did not become visible within ${INTERRUPT_GRACE_SEC}s after interrupt. Exiting." >&2
        exit 130
      fi
    fi
  fi

  sleep "${POLL_INTERVAL}"
done

if [[ "${LAUNCH_DONE}" -eq 0 ]]; then
  if wait "${LAUNCH_PID}"; then
    LAUNCH_RC=0
  else
    LAUNCH_RC=$?
  fi
  LAUNCH_DONE=1
fi

print_summary \
  "${FINAL_ROOT_PHASE}" \
  "${FINAL_SEEN}" \
  "${FINAL_SUCCEEDED}" \
  "${FINAL_FAILED}" \
  "${FINAL_ABORTED}" \
  "${FINAL_TIMED_OUT}"

if [[ "${STOPPING}" -eq 1 ]]; then
  exit 130
fi

if [[ "${FINAL_ROOT_PHASE}" != "ACTION_PHASE_SUCCEEDED" ]]; then
  exit 1
fi
