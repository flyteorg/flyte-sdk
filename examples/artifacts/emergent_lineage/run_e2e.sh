#!/usr/bin/env bash
# End-to-end: deploy every team's module SEPARATELY, seed raw data, then pull daily_report through the
# graph that emerges from their declarations.
#
#   ./run_e2e.sh [--config PATH] [--date YYYY-MM-DD] [--start YYYY-MM-DD] [--project P] [--domain D]
#                [--queue Q] [--with-legacy] [--skip-materialize] [--no-devbox-shim]
#
# --date is the partition to materialize; --start is the first day of raw data to seed (default: 30 days
# before --date, which the 30-day training window needs).
#
# NOTE: deploying analytics/report.py registers daily_report's refresh policy, a LIVE cron trigger (02:00 daily)
# that keeps materializing daily_report until you deactivate it:
#   flyte update trigger keep-report-fresh refresh-daily-report.keep_report_fresh --deactivate
# triggers/weekly_review.py adds another (Mondays 05:00, churn_model):
#   flyte update trigger for-weekly-review refresh-churn-model.for_weekly_review --deactivate
#
# --config defaults to the union-devbox config of the union-fullstack checkout
# (../../../../local-testing/union-devbox/.flyte/config.yaml relative to this directory).
# --queue is passed to `flyte materialize` (on the local devbox use --queue testcluster).
# `flyte materialize` comes from flyteplugins-union; without it the script stops after seeding.
# The apps are deployed after the first materialize: churn-scoring resolves churn_model@latest at deploy.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${HERE}/../../../../local-testing/union-devbox/.flyte/config.yaml"
DATE="2026-09-08"
START=""
PROJECT_ARGS=()
QUEUE_ARGS=()
WITH_LEGACY=0
SKIP_MATERIALIZE=0
DEVBOX_SHIM=auto

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --date) DATE="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --project) PROJECT_ARGS+=(--project "$2"); shift 2 ;;
    --domain) PROJECT_ARGS+=(--domain "$2"); shift 2 ;;
    --queue) QUEUE_ARGS+=(--queue "$2"); shift 2 ;;
    --with-legacy) WITH_LEGACY=1; shift ;;
    --skip-materialize) SKIP_MATERIALIZE=1; shift ;;
    --no-devbox-shim) DEVBOX_SHIM=0; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# train reads a 30-day window and report a 7-day one, so seed 30 days back from the target date.
if [[ -z "${START}" ]]; then
  START="$(python3 -c "import datetime as d; print((d.date.fromisoformat('${DATE}') - d.timedelta(days=30)).isoformat())")"
fi

# A local devbox's app service needs an x-user-subject header the CLI does not send (hosted tenants derive it
# from auth). Against a localhost endpoint, load ../devbox_shim/sitecustomize.py, which adds it.
if [[ "${DEVBOX_SHIM}" == "auto" ]] && grep -Eq 'endpoint:.*(localhost|127\.0\.0\.1)' "${CONFIG}"; then
  export PYTHONPATH="${HERE}/../devbox_shim${PYTHONPATH:+:${PYTHONPATH}}"
  echo "local devbox endpoint: adding the x-user-subject header for app deploys (--no-devbox-shim to skip)"
fi

cd "${HERE}"
FLYTE=(flyte --config "${CONFIG}")
deploy() {  # deploy <file> <env variable>
  echo
  echo "=== flyte deploy $1 $2"
  "${FLYTE[@]}" deploy ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} --root-dir . "$1" "$2"
}

# 1. Every team deploys on its own. No module imports another team's task, only its handles.
deploy ingest/events.py env
deploy ingest/seed.py env
deploy ml/features.py env
deploy ml/train.py env
deploy analytics/report.py env
deploy analytics/notify.py env
deploy triggers/revalidate.py env
deploy triggers/weekly_review.py for_weekly_review
if [[ "${WITH_LEGACY}" == "1" ]]; then
  deploy legacy/legacy_clean.py env
  deploy legacy/adapter.py env
fi

# 2. Land raw data from "outside" the graph.
echo
echo "=== seeding raw_events ${START} .. ${DATE}"
"${FLYTE[@]}" run ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} --root-dir . --follow ingest/seed.py seed --start "${START}" --end "${DATE}"

if [[ "${SKIP_MATERIALIZE}" == "1" ]]; then
  exit 0
fi
if ! "${FLYTE[@]}" materialize --help >/dev/null 2>&1; then
  echo "flyte materialize is not available (install flyteplugins-union); stopping after the seed." >&2
  echo "(the apps are not deployed: churn-scoring needs a churn_model version to resolve)" >&2
  exit 0
fi

# 3. Plan only: the instance DAG for one partition, every parameter accounted for.
echo
echo "=== flyte materialize daily_report --partition date=${DATE} --plan"
"${FLYTE[@]}" materialize ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} daily_report --partition "date=${DATE}" --plan

# 4. The real pull. The planner walks back to raw_events and builds what is missing. --wait blocks until
#    the run finishes, so the apps below find a published churn_model.
echo
echo "=== flyte materialize daily_report --partition date=${DATE} --wait"
"${FLYTE[@]}" materialize ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} daily_report --partition "date=${DATE}" --wait

# 5. Apps, now that churn_model has a version for churn-scoring to resolve at deploy.
deploy apps/scoring.py scoring
deploy apps/dashboard.py dashboard

# 6. The researcher's view of the same data.
echo
echo "=== notebook/explore.py"
PYTHONPATH=. python notebook/explore.py --config "${CONFIG}" --date "${DATE}"
