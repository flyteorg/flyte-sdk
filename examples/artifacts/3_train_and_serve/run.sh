#!/usr/bin/env bash
# The README walkthrough as one script: publish data, deploy the model and the app, serve new data, check the
# quality gate, roll back, and deploy the factory that does it all on new data.
#
#   ./run.sh --config PATH [--queue Q] [--project P] [--domain D] [--skip-factory]
#
# --queue is passed to `flyte materialize` and `flyte factory materialize`.
# Steps 2 onwards need the Union lineage service and flyteplugins-union.
# Deploying train.py registers a trigger that retrains on every new day of reviews; deploying factory.py adds
# one that also redeploys the app. Deactivate them when you are done:
#   flyte update trigger retrain-on-new-reviews refresh-sentiment-model.retrain_on_new_reviews --deactivate
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=""
SCOPE=()
QUEUE=()
SKIP_FACTORY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --queue) QUEUE=(--queue "$2"); shift 2 ;;
    --project|--domain) SCOPE+=("$1" "$2"); shift 2 ;;
    --skip-factory) SKIP_FACTORY=1; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${CONFIG}" ]] || { echo "--config PATH is required" >&2; exit 2; }
cd "${HERE}"

FLYTE=(flyte --config "${CONFIG}")
step() { echo; echo "=== $*"; }
deploy() { "${FLYTE[@]}" deploy ${SCOPE[@]+"${SCOPE[@]}"} --root-dir . "$@"; }
land() { "${FLYTE[@]}" run ${SCOPE[@]+"${SCOPE[@]}"} --root-dir . --follow data.py "$@"; }
mat() { "${FLYTE[@]}" materialize "$1" ${SCOPE[@]+"${SCOPE[@]}"} "${@:2}" ${QUEUE[@]+"${QUEUE[@]}"} --wait; }
served() { PYTHONPATH=. python served.py --config "${CONFIG}" sentiment-api; }
# The lineage graph picks up a deploy within about a minute.
wait_for() {
  for _ in $(seq 1 30); do
    if "${FLYTE[@]}" materialize "$1" ${SCOPE[@]+"${SCOPE[@]}"} "$2" --partition date=2026-09-14 --plan >/dev/null 2>&1; then
      return
    fi
    sleep 5
  done
}

step "1. publish two weeks of data, then deploy the model"
deploy data.py env
land land --start 2026-09-01 --end 2026-09-14
deploy train.py env
wait_for artifact sentiment_model

step "2. train the first model, then deploy the app"
mat artifact sentiment_model --partition date=2026-09-14
deploy serve.py api
served
wait_for app sentiment-api

step "3-4. new data arrives; serve the model for it"
land land_reviews --day 2026-09-15
mat app sentiment-api --partition date=2026-09-15
SERVED_15="$(served)"
echo "${SERVED_15}"

step "5. the quality gate: a model below the bar is never published, and the app keeps serving the last good one"
land land_reviews --day 2026-09-16
if mat app sentiment-api --partition date=2026-09-16 --input sentiment-train.train.min_accuracy=0.95; then
  echo "expected the gate to fail the run" >&2
  exit 1
fi
[[ "$(served)" == "${SERVED_15}" ]] || { echo "the app changed after a failed gate" >&2; exit 1; }
mat app sentiment-api --partition date=2026-09-16

step "6. roll back to the 2026-09-15 model"
mat app sentiment-api --partition date=2026-09-15 --version "sentiment_model=${SERVED_15##* }"
[[ "$(served)" == "${SERVED_15}" ]] || { echo "the app was not rolled back" >&2; exit 1; }

if [[ "${SKIP_FACTORY}" == "1" ]]; then
  exit 0
fi

step "7. the factory: new data retrains and redeploys in one run"
"${FLYTE[@]}" factory deploy ${SCOPE[@]+"${SCOPE[@]}"} factory.py --dry-run
"${FLYTE[@]}" factory deploy ${SCOPE[@]+"${SCOPE[@]}"} factory.py
land land_reviews --day 2026-09-17
"${FLYTE[@]}" factory materialize ${SCOPE[@]+"${SCOPE[@]}"} sentiment sentiment-api --partition date=2026-09-17 \
  ${QUEUE[@]+"${QUEUE[@]}"} --wait
served
