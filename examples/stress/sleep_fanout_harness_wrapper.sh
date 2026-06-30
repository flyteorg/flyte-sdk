#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${HOME}/.flyte/config-dogfood.yaml"
PROJECT=""
DOMAIN=""
IMAGE_REGISTRY="${FLYTE_STRESS_IMAGE_REGISTRY:-376129846803.dkr.ecr.us-east-2.amazonaws.com/union}"
IMAGE_NAME="${FLYTE_STRESS_IMAGE_NAME:-dogfood}"
IMAGE_PLATFORMS="${FLYTE_STRESS_IMAGE_PLATFORMS:-linux/amd64}"
IMAGE_BUILDER="${FLYTE_STRESS_IMAGE_BUILDER:-remote}"
FANOUT_CPU_REQUEST="${FLYTE_STRESS_FANOUT_CPU_REQUEST:-1}"
FANOUT_CPU_LIMIT="${FLYTE_STRESS_FANOUT_CPU_LIMIT:-2}"
FANOUT_MEMORY_REQUEST="${FLYTE_STRESS_FANOUT_MEMORY_REQUEST:-2Gi}"
FANOUT_MEMORY_LIMIT="${FLYTE_STRESS_FANOUT_MEMORY_LIMIT:-4Gi}"
TOTAL_RUNS=20
SUBMIT_CONCURRENCY=100
N_CHILDREN=5000
SLEEP_DURATION=800
POLL_INTERVAL=2
ABORT_REASON="wrapper interrupted"
RUN_ENV_KVS=()

EXPECTED_TOTAL_CHILDREN=0
CHILD_RUNS=()

LAUNCH_PID=""
LAUNCH_LOG=""
LAUNCH_RC_FILE=""
RUNS_FILE=""
LAUNCH_DONE=0
LAUNCH_RC=0
STOPPING=0
ABORT_SENT=0

SCRIPT_START_EPOCH="$(date +%s)"
FIRST_DISCOVERED_AT=""
FIRST_RUNNING_AT=""
ALL_VISIBLE_AT=""
TERMINAL_AT=""

PEAK_SEEN=0
PEAK_RUNNING=0
PEAK_ACTIVE=0
PEAK_CREATE_RPS=0
PEAK_PARENT_LIVE=0
PEAK_PARENT_RUNNING=0
LAST_LAUNCH_STAGE=""
SDK_WHEEL_PATH=""
SDK_SRC_NEWER=0

usage() {
  cat <<'EOF'
Usage:
  examples/stress/sleep_fanout_harness_wrapper.sh [options]

Options:
  --config PATH               Flyte config path. Default: ~/.flyte/config-dogfood.yaml
  --project NAME              Override project for get/abort.
  --domain NAME               Override domain for get/abort.
  --image-registry VALUE      Registry prefix for the task image.
  --image-name VALUE          Repository name for the task image.
  --image-builder VALUE       Flyte image builder to use for lookups. Default: remote
  --run-env KEY=VALUE         Export an env var into the local submit harness and propagate it to remote runs.
  --total-runs INT            Number of top-level sleep_fanout runs to submit. Default: 20
  --submit-concurrency INT    Local submission concurrency. Default: 100
  --n-children INT            Leaves per sleep_fanout run. Default: 5000
  --sleep-duration VALUE      Sleep duration in seconds per leaf. Default: 800
  --poll-interval SEC         Poll interval in seconds. Default: 2
  --abort-reason TEXT         Reason passed to 'flyte abort run'. Default: wrapper interrupted
  --help                      Show this message.
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
      RUN_ENV_KVS+=("$2")
      shift 2
      ;;
    --total-runs)
      TOTAL_RUNS="$2"
      shift 2
      ;;
    --submit-concurrency)
      SUBMIT_CONCURRENCY="$2"
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

EXPECTED_TOTAL_CHILDREN=$((TOTAL_RUNS * N_CHILDREN))

if ! command -v flyte >/dev/null 2>&1; then
  echo "flyte is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is required but was not found in PATH." >&2
  exit 1
fi

CONFIG="${CONFIG/#\~/${HOME}}"
LAUNCH_LOG="$(mktemp "${TMPDIR:-/tmp}/sleep-fanout-harness.XXXXXX.log")"
LAUNCH_RC_FILE="$(mktemp "${TMPDIR:-/tmp}/sleep-fanout-harness.XXXXXX.rc")"
RUNS_FILE="$(mktemp "${TMPDIR:-/tmp}/sleep-fanout-runs.XXXXXX.txt")"

cleanup() {
  if [[ -n "${LAUNCH_LOG}" && -f "${LAUNCH_LOG}" ]]; then
    rm -f "${LAUNCH_LOG}"
  fi
  if [[ -n "${LAUNCH_RC_FILE}" && -f "${LAUNCH_RC_FILE}" ]]; then
    rm -f "${LAUNCH_RC_FILE}"
  fi
  if [[ -n "${RUNS_FILE}" && -f "${RUNS_FILE}" ]]; then
    rm -f "${RUNS_FILE}"
  fi
}
trap cleanup EXIT

project_args=()
domain_args=()

if [[ -n "${PROJECT}" ]]; then
  project_args=(-p "${PROJECT}")
fi

if [[ -n "${DOMAIN}" ]]; then
  domain_args=(-d "${DOMAIN}")
fi

flyte_cmd_json() {
  COLUMNS=500 _U_USE_ACTIONS="${_U_USE_ACTIONS:-1}" flyte -c "${CONFIG}" --image-builder "${IMAGE_BUILDER}" -of json-raw "$@" \
    | perl -pe 's/\e\[[0-9;]*[A-Za-z]//g'
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

format_duration() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo "n/a"
    return
  fi
  printf '%02dh:%02dm:%02ds' "$((value / 3600))" "$(((value % 3600) / 60))" "$((value % 60))"
}

elapsed_from_start() {
  local epoch="$1"
  if [[ -z "${epoch}" ]]; then
    echo ""
    return
  fi
  echo "$((epoch - SCRIPT_START_EPOCH))"
}

print_row() {
  printf '%-8s %-12s %-8s %-8s %-18s %-14s %-8s %-10s %-8s %-10s %-8s %-8s\n' \
    "$(date +%H:%M:%S)" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}"
}

sanitize_run_name() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr -d '\r')"
  value="$(printf '%s' "${value}" | sed 's/[[:space:]]*$//')"
  value="$(printf '%s' "${value}" | grep -Eo '[ur][[:alnum:]]{5,}' | head -n 1 || true)"
  printf '%s' "${value}"
}

detect_sdk_wheel_status() {
  local wheel_path=""
  local newest_src=""

  wheel_path="$(
    find "${REPO_ROOT}/dist" -maxdepth 1 -type f -name 'flyte-*.whl' -print 2>/dev/null \
      | sort \
      | tail -n 1 || true
  )"
  SDK_WHEEL_PATH="${wheel_path}"
  SDK_SRC_NEWER=0

  if [[ -z "${wheel_path}" ]]; then
    return
  fi

  newest_src="$(
    find "${REPO_ROOT}/src/flyte" -type f -newer "${wheel_path}" -print 2>/dev/null \
      | head -n 1 || true
  )"
  if [[ -n "${newest_src}" ]]; then
    SDK_SRC_NEWER=1
  fi
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

discover_child_runs() {
  local child_run=""

  if [[ -s "${RUNS_FILE}" ]]; then
    while IFS= read -r child_run; do
      child_run="$(sanitize_run_name "${child_run}")"
      [[ -z "${child_run}" ]] && continue
      if ! child_run_known "${child_run}"; then
        CHILD_RUNS+=("${child_run}")
      fi
    done < "${RUNS_FILE}"
  fi

  if [[ -s "${LAUNCH_LOG}" ]]; then
    while IFS= read -r child_run; do
      child_run="$(sanitize_run_name "${child_run}")"
      [[ -z "${child_run}" ]] && continue
      if ! child_run_known "${child_run}"; then
        CHILD_RUNS+=("${child_run}")
      fi
    done < <(
      perl -ne '
        s/\e\[[0-9;]*[A-Za-z]//g;
        s/\r/\n/g;
        if (/submitted_run idx=\d+ url=.*\/runs\/([^\/?\s]+)/) {
          print "$1\n";
        } elsif (/submitted_run idx=\d+ url=([ur][[:alnum:]]{5,})/) {
          print "$1\n";
        }
      ' "${LAUNCH_LOG}"
    )
  fi
}

fetch_actions_json_for_run() {
  local run_name="$1"
  flyte_cmd_json get action "${project_args[@]}" "${domain_args[@]}" "${run_name}"
}

aggregate_child_runs_tsv() {
  local discovered=0
  local roots_terminal=0
  local parent_live=0
  local parent_running=0
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
    elif [[ "${root_phase}" != "MISSING" ]]; then
      parent_live=$((parent_live + 1))
      if [[ "${root_phase}" == "ACTION_PHASE_RUNNING" ]]; then
        parent_running=$((parent_running + 1))
      fi
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

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${discovered}/${TOTAL_RUNS}" \
    "${parent_live}" \
    "${parent_running}" \
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

launch_stage_from_log() {
  if [[ ! -s "${LAUNCH_LOG}" ]]; then
    return 1
  fi

  local stage=""
  stage="$(
    perl -pe 's/\e\[[0-9;]*[A-Za-z]//g; s/\r/\n/g' "${LAUNCH_LOG}" \
      | sed '/^[[:space:]]*$/d' \
      | grep -E '^(submitted=|Done\.|submitted_run|Error:|ERROR|Failed|failed)' \
      | tail -n 1 || true
  )"

  if [[ -z "${stage}" ]]; then
    stage="$(perl -pe 's/\e\[[0-9;]*[A-Za-z]//g; s/\r/\n/g' "${LAUNCH_LOG}" | sed '/^[[:space:]]*$/d' | tail -n 1)"
  fi

  [[ -n "${stage}" ]] || return 1
  printf '%s' "${stage}"
}

abort_remote_runs() {
  local run_name=""
  if [[ "${ABORT_SENT}" -eq 1 ]]; then
    return
  fi
  ABORT_SENT=1

  for run_name in "${CHILD_RUNS[@]}"; do
    COLUMNS=500 _U_USE_ACTIONS="${_U_USE_ACTIONS:-1}" flyte -c "${CONFIG}" --image-builder "${IMAGE_BUILDER}" \
      abort run "${project_args[@]}" "${domain_args[@]}" --reason "${ABORT_REASON}" "${run_name}" >/dev/null 2>&1 || true
  done
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
  echo "Received ${sig}, stopping local submissions and aborting discovered runs."
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill "${LAUNCH_PID}" 2>/dev/null || true
  fi
  discover_child_runs
  abort_remote_runs
}

trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

cd "${REPO_ROOT}"
detect_sdk_wheel_status

echo "Launching local multi-run harness"
echo "  config: ${CONFIG}"
echo "  total_runs: ${TOTAL_RUNS}"
echo "  submit_concurrency: ${SUBMIT_CONCURRENCY}"
echo "  n_children_per_run: ${N_CHILDREN}"
echo "  total_children_expected: ${EXPECTED_TOTAL_CHILDREN}"
echo "  sleep_duration: ${SLEEP_DURATION}"
echo "  poll_interval: ${POLL_INTERVAL}s"
echo "  image target: ${IMAGE_REGISTRY}/${IMAGE_NAME}"
echo "  image builder: ${IMAGE_BUILDER}"
echo "  image platforms: ${IMAGE_PLATFORMS}"
if [[ "${FLYTE_HARNESS_FORCE_LOCAL_SDK:-0}" == "1" || "${FLYTE_HARNESS_FORCE_LOCAL_SDK:-}" == "true" ]]; then
  echo "  sdk source: ${REPO_ROOT}/src forced via $(command -v flyte)"
else
  echo "  sdk source: installed flyte via $(command -v flyte)"
fi
if [[ -n "${SDK_WHEEL_PATH}" ]]; then
  echo "  sdk wheel: ${SDK_WHEEL_PATH}"
  if [[ "${SDK_SRC_NEWER}" -eq 1 ]]; then
    echo "  warning: src/flyte is newer than the dist wheel; remote image will not include recent SDK src changes until you rebuild the wheel"
  fi
else
  echo "  sdk wheel: <missing>"
fi
echo "  fanout parent resources: cpu ${FANOUT_CPU_REQUEST}/${FANOUT_CPU_LIMIT}, memory ${FANOUT_MEMORY_REQUEST}/${FANOUT_MEMORY_LIMIT}"
echo "  use_actions: ${_U_USE_ACTIONS:-1}"
if [[ -n "${PROJECT}" || -n "${DOMAIN}" ]]; then
  echo "  project/domain override: ${PROJECT:-<config>} / ${DOMAIN:-<config>}"
fi
if [[ "${#RUN_ENV_KVS[@]}" -gt 0 ]]; then
  echo "  run env overrides: ${RUN_ENV_KVS[*]}"
fi
echo
printf '%-8s %-12s %-8s %-8s %-18s %-14s %-8s %-10s %-8s %-10s %-8s %-8s\n' \
  "time" "runs" "p_live" "p_run" "seen_children" "not_created" "d_seen" "create_rps" "rps/p" "eta_fill" "running" "active"

(
  export _U_USE_ACTIONS="${_U_USE_ACTIONS:-1}"
  export FLYTE_STRESS_IMAGE_REGISTRY="${IMAGE_REGISTRY}"
  export FLYTE_STRESS_IMAGE_NAME="${IMAGE_NAME}"
  export FLYTE_STRESS_IMAGE_PLATFORMS="${IMAGE_PLATFORMS}"
  export FLYTE_HARNESS_CONFIG="${CONFIG}"
  export FLYTE_HARNESS_IMAGE_BUILDER="${IMAGE_BUILDER}"
  export FLYTE_HARNESS_PROJECT="${PROJECT}"
  export FLYTE_HARNESS_DOMAIN="${DOMAIN}"
  export FLYTE_HARNESS_RUNS_FILE="${RUNS_FILE}"
  local_kv=""
  for local_kv in "${RUN_ENV_KVS[@]}"; do
    export "${local_kv}"
  done
  rc=0
  python examples/stress/sleep_fanout_harness.py \
    --total "${TOTAL_RUNS}" \
    --concurrency "${SUBMIT_CONCURRENCY}" \
    --n_children "${N_CHILDREN}" \
    --sleep_seconds "${SLEEP_DURATION}" || rc=$?
  printf '%s\n' "${rc}" > "${LAUNCH_RC_FILE}"
  exit "${rc}"
) >"${LAUNCH_LOG}" 2>&1 &
LAUNCH_PID=$!

FINAL_DISCOVERED=0
FINAL_SEEN=0
FINAL_SUCCEEDED=0
FINAL_FAILED=0
FINAL_ABORTED=0
FINAL_TIMED_OUT=0
FINAL_ROOTS_TERMINAL=0
FINAL_PARENT_LIVE=0
FINAL_PARENT_RUNNING=0
LAST_SAMPLE_TS=""
LAST_SAMPLE_SEEN=""

while true; do
  discover_child_runs

  if [[ -z "${FIRST_DISCOVERED_AT}" && "${#CHILD_RUNS[@]}" -gt 0 ]]; then
    FIRST_DISCOVERED_AT="$(date +%s)"
  fi

  if [[ "${LAUNCH_DONE}" -eq 0 && -s "${LAUNCH_RC_FILE}" ]]; then
    LAUNCH_RC="$(tr -d '\r\n[:space:]' < "${LAUNCH_RC_FILE}")"
    if [[ -z "${LAUNCH_RC}" ]]; then
      LAUNCH_RC=1
    fi
    wait "${LAUNCH_PID}" 2>/dev/null || true
    LAUNCH_DONE=1
  fi

  if [[ "${#CHILD_RUNS[@]}" -gt 0 ]]; then
    IFS=$'\t' read -r runs parent_live parent_running seen_children not_created queued waiting initializing running active succeeded failed aborted timed_out roots_terminal \
      <<<"$(aggregate_child_runs_tsv)"

    FINAL_DISCOVERED="${runs%%/*}"
    FINAL_PARENT_LIVE="${parent_live}"
    FINAL_PARENT_RUNNING="${parent_running}"
    FINAL_SEEN="${seen_children%%/*}"
    FINAL_SUCCEEDED="${succeeded}"
    FINAL_FAILED="${failed}"
    FINAL_ABORTED="${aborted}"
    FINAL_TIMED_OUT="${timed_out}"
    FINAL_ROOTS_TERMINAL="${roots_terminal}"

    if (( FINAL_SEEN > PEAK_SEEN )); then
      PEAK_SEEN="${FINAL_SEEN}"
    fi
    if (( running > PEAK_RUNNING )); then
      PEAK_RUNNING="${running}"
    fi
    if (( active > PEAK_ACTIVE )); then
      PEAK_ACTIVE="${active}"
    fi
    if (( parent_live > PEAK_PARENT_LIVE )); then
      PEAK_PARENT_LIVE="${parent_live}"
    fi
    if (( parent_running > PEAK_PARENT_RUNNING )); then
      PEAK_PARENT_RUNNING="${parent_running}"
    fi
    if [[ -z "${FIRST_RUNNING_AT}" && "${running}" -gt 0 ]]; then
      FIRST_RUNNING_AT="$(date +%s)"
    fi
    if [[ -z "${ALL_VISIBLE_AT}" && "${FINAL_SEEN}" -ge "${EXPECTED_TOTAL_CHILDREN}" ]]; then
      ALL_VISIBLE_AT="$(date +%s)"
    fi

    sample_ts="$(date +%s)"
    delta_seen="0"
    create_rps="n/a"
    create_rps_per_parent="n/a"
    eta_fill="n/a"
    if [[ -n "${LAST_SAMPLE_TS}" && -n "${LAST_SAMPLE_SEEN}" ]]; then
      delta_t=$((sample_ts - LAST_SAMPLE_TS))
      if (( delta_t > 0 )); then
        delta_seen=$((FINAL_SEEN - LAST_SAMPLE_SEEN))
        if (( delta_seen < 0 )); then
          delta_seen=0
        fi
        create_rps="$(python - <<'PY' "${delta_seen}" "${delta_t}"
import sys
delta_seen = int(sys.argv[1])
delta_t = int(sys.argv[2])
print(f"{delta_seen / delta_t:.1f}")
PY
)"
        if (( parent_running > 0 )); then
          create_rps_per_parent="$(python - <<'PY' "${delta_seen}" "${delta_t}" "${parent_running}"
import sys
delta_seen = int(sys.argv[1])
delta_t = int(sys.argv[2])
parent_running = int(sys.argv[3])
print(f"{(delta_seen / delta_t) / parent_running:.1f}")
PY
)"
        fi
        create_rps_int="$(python - <<'PY' "${delta_seen}" "${delta_t}"
import sys
delta_seen = int(sys.argv[1])
delta_t = int(sys.argv[2])
print(int(delta_seen / delta_t))
PY
)"
        if (( create_rps_int > PEAK_CREATE_RPS )); then
          PEAK_CREATE_RPS="${create_rps_int}"
        fi
        if (( FINAL_SEEN < EXPECTED_TOTAL_CHILDREN && delta_seen > 0 )); then
          eta_fill="$(python - <<'PY' "${EXPECTED_TOTAL_CHILDREN}" "${FINAL_SEEN}" "${delta_seen}" "${delta_t}"
import math
import sys
expected = int(sys.argv[1])
seen = int(sys.argv[2])
delta_seen = int(sys.argv[3])
delta_t = int(sys.argv[4])
remaining = expected - seen
seconds = math.ceil(remaining / (delta_seen / delta_t))
h, rem = divmod(seconds, 3600)
m, s = divmod(rem, 60)
print(f"{h:02d}:{m:02d}:{s:02d}")
PY
)"
        elif (( FINAL_SEEN >= EXPECTED_TOTAL_CHILDREN )); then
          eta_fill="00:00:00"
        fi
      fi
    fi
    LAST_SAMPLE_TS="${sample_ts}"
    LAST_SAMPLE_SEEN="${FINAL_SEEN}"

    print_row \
      "${runs}" \
      "${parent_live}" \
      "${parent_running}" \
      "${seen_children}" \
      "${not_created}" \
      "${delta_seen}" \
      "${create_rps}" \
      "${create_rps_per_parent}" \
      "${eta_fill}" \
      "${running}" \
      "${active}"

    if [[ "${LAUNCH_DONE}" -eq 1 ]] && (( FINAL_ROOTS_TERMINAL == FINAL_DISCOVERED )) && (( active == 0 )); then
      TERMINAL_AT="$(date +%s)"
      break
    fi
  else
    if stage="$(launch_stage_from_log)"; then
      if [[ "${stage}" != "${LAST_LAUNCH_STAGE}" ]]; then
        LAST_LAUNCH_STAGE="${stage}"
        echo "launch: ${stage}"
      fi
    fi
    print_row "0/${TOTAL_RUNS}" 0 0 "0/${EXPECTED_TOTAL_CHILDREN}" "${EXPECTED_TOTAL_CHILDREN}" 0 0 0 0 0 0
  fi

  if [[ "${LAUNCH_DONE}" -eq 1 && "${LAUNCH_RC}" -ne 0 && "${#CHILD_RUNS[@]}" -eq 0 ]]; then
    echo
    echo "Local submit harness failed before any runs were discovered." >&2
    cat "${LAUNCH_LOG}" >&2
    exit "${LAUNCH_RC}"
  fi

  sleep "${POLL_INTERVAL}"
done

echo
echo "Aggregate Summary"
echo "  runs_discovered: ${FINAL_DISCOVERED}/${TOTAL_RUNS}"
echo "  total_expected_children: ${EXPECTED_TOTAL_CHILDREN}"
echo "  child_run_roots_terminal: ${FINAL_ROOTS_TERMINAL}/${FINAL_DISCOVERED}"
echo "  peak_parent_live: ${PEAK_PARENT_LIVE}"
echo "  peak_parent_running: ${PEAK_PARENT_RUNNING}"
echo "  children_seen: ${FINAL_SEEN}/${EXPECTED_TOTAL_CHILDREN}"
echo "  succeeded: ${FINAL_SUCCEEDED}"
echo "  failed: ${FINAL_FAILED}"
echo "  aborted: ${FINAL_ABORTED}"
echo "  timed_out: ${FINAL_TIMED_OUT}"
echo "  peak_seen: ${PEAK_SEEN}/${EXPECTED_TOTAL_CHILDREN}"
echo "  peak_running: ${PEAK_RUNNING}"
echo "  peak_active: ${PEAK_ACTIVE}"
echo "  peak_create_rps: ${PEAK_CREATE_RPS}"
echo "  first_run_discovered: $(format_duration "$(elapsed_from_start "${FIRST_DISCOVERED_AT}")")"
echo "  aggregate_first_running: $(format_duration "$(elapsed_from_start "${FIRST_RUNNING_AT}")")"
echo "  aggregate_all_visible: $(format_duration "$(elapsed_from_start "${ALL_VISIBLE_AT}")")"
echo "  aggregate_terminal: $(format_duration "$(elapsed_from_start "${TERMINAL_AT}")")"
echo "  total_elapsed: $(format_duration "$(( $(date +%s) - SCRIPT_START_EPOCH ))")"
