#!/usr/bin/env bash
# End-to-end: deploy every team's module SEPARATELY, seed raw data, then pull daily_report through the
# graph that emerges from their declarations, and serve the churn-scoring app from it.
#
#   ./run_e2e.sh --config PATH [--date YYYY-MM-DD] [--start YYYY-MM-DD] [--project P] [--domain D]
#                [--queue Q] [--with-legacy] [--skip-materialize] [--skip-app]
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
# --config (required) is the flyte config of the cluster to deploy to.
# --queue is passed to `flyte materialize`; use a queue with enough CPU for the pipeline's tasks.
# Steps 3-4 and 7 (`flyte materialize artifact|app`), the refresh triggers and the apps' artifact resolution are
# experimental and need the Union lineage service and flyteplugins-union (which provides `flyte materialize`);
# without the plugin the script stops after seeding.
# The apps are deployed after the first materialize: churn-scoring resolves churn_model@latest at deploy.
# --skip-app stops before step 7 (materializing the churn-scoring app, which retrains churn_model) and step 8
# (materializing churn-alerts, which reads three artifacts, one unpartitioned, over a two-day range).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=""
DATE="2026-09-08"
START=""
PROJECT_ARGS=()
QUEUE_ARGS=()
WITH_LEGACY=0
SKIP_MATERIALIZE=0
SKIP_APP=0

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
    --skip-app) SKIP_APP=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "--config PATH is required (the flyte config of the cluster to deploy to)" >&2
  exit 2
fi

# train reads a 30-day window and report a 7-day one, so seed 30 days back from the target date.
if [[ -z "${START}" ]]; then
  START="$(python3 -c "import datetime as d; print((d.date.fromisoformat('${DATE}') - d.timedelta(days=30)).isoformat())")"
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
deploy ml/thresholds.py env
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

# 3-4 are experimental: they need the Union lineage service and flyteplugins-union.
# 3. Plan only: the instance DAG for one partition, every parameter accounted for.
echo
echo "=== flyte materialize artifact daily_report --partition date=${DATE} --plan"
"${FLYTE[@]}" materialize artifact ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} daily_report --partition "date=${DATE}" --plan

# 4. The real pull. The planner walks back to raw_events and builds what is missing. --wait blocks until
#    the run finishes, so the apps below find a published churn_model.
echo
echo "=== flyte materialize artifact daily_report --partition date=${DATE} --wait"
"${FLYTE[@]}" materialize artifact ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} daily_report --partition "date=${DATE}" --wait

# 5. Apps, now that churn_model has a version for churn-scoring to resolve at deploy. churn-alerts also reads
#    churn_thresholds, which nothing above built: materialize it (no partitions, no upstream) first.
echo
echo "=== flyte materialize artifact churn_thresholds --wait"
"${FLYTE[@]}" materialize artifact ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} churn_thresholds --wait
deploy apps/scoring.py scoring
deploy apps/dashboard.py dashboard
deploy apps/alerts.py alerts

# 6. The researcher's view of the same data.
echo
echo "=== notebook/explore.py"
PYTHONPATH=. python notebook/explore.py --config "${CONFIG}" --date "${DATE}"

if [[ "${SKIP_APP}" == "1" ]]; then
  exit 0
fi

# 7. Materialize the app: build what its parameters read (churn_model, back through features and events), then
#    redeploy churn-scoring serving the version the run built. --rebuild ml-train.train retrains, so the app must end
#    up on a churn_model version it did not serve before.
served() { PYTHONPATH=. python apps/served.py --config "${CONFIG}" "${1:-churn-scoring}"; }
echo
echo "=== flyte materialize app churn-scoring --partition date=${DATE} --plan"
"${FLYTE[@]}" materialize app ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} churn-scoring --partition "date=${DATE}" --plan
BEFORE="$(served)"
echo "churn-scoring serves: ${BEFORE}"
echo
echo "=== flyte materialize app churn-scoring --partition date=${DATE} --rebuild ml-train.train --wait"
"${FLYTE[@]}" materialize app ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} churn-scoring --partition "date=${DATE}" --rebuild ml-train.train --wait
AFTER="$(served)"
echo "churn-scoring serves: ${AFTER}"
if [[ "${AFTER}" == "${BEFORE}" || "${AFTER}" != *"churn_model "* ]]; then
  echo "churn-scoring was not redeployed with the retrained churn_model" >&2
  exit 1
fi

# 8. An app reading three artifacts over a range: churn_model and daily_report for two days (the newest is served)
#    and the unpartitioned churn_thresholds, rebuilt with an --input override (a new min_users each time, so it
#    always publishes a new version). The run must not roll churn_model back: after step 7's --rebuild the cache
#    returns an older version of the day's model than the newest, and an app already serving the newest keeps it
#    while churn_thresholds moves to the new version.
PREV="$(python3 -c "import datetime as d; print((d.date.fromisoformat('${DATE}') - d.timedelta(days=1)).isoformat())")"
MIN_USERS="$(date +%s)"
echo
echo "=== flyte materialize app churn-alerts --partition date=${PREV}..${DATE} --input ml-thresholds.calibrate.min_users=${MIN_USERS} --plan"
"${FLYTE[@]}" materialize app ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} churn-alerts --partition "date=${PREV}..${DATE}" --input "ml-thresholds.calibrate.min_users=${MIN_USERS}" --plan
BEFORE="$(served churn-alerts)"
echo "churn-alerts serves:"; echo "${BEFORE}"
echo
echo "=== flyte materialize app churn-alerts --partition date=${PREV}..${DATE} --input ml-thresholds.calibrate.min_users=${MIN_USERS} --wait"
"${FLYTE[@]}" materialize app ${PROJECT_ARGS[@]+"${PROJECT_ARGS[@]}"} ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} churn-alerts --partition "date=${PREV}..${DATE}" --input "ml-thresholds.calibrate.min_users=${MIN_USERS}" --wait
ALERTS="$(served churn-alerts)"
echo "churn-alerts serves:"; echo "${ALERTS}"
if [[ "$(grep '^thresholds ' <<<"${ALERTS}")" == "$(grep '^thresholds ' <<<"${BEFORE}")" ]]; then
  echo "churn-alerts was not redeployed with the new churn_thresholds" >&2
  exit 1
fi
if [[ "$(grep '^model ' <<<"${ALERTS}")" != "$(grep '^model ' <<<"${BEFORE}")" ]]; then
  echo "churn-alerts changed the churn_model it serves (a rollback to the cached version?)" >&2
  exit 1
fi
