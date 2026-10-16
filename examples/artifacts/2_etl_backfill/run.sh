#!/usr/bin/env bash
# The README walkthrough as one script: deploy, land raw data, plan, build, backfill, rerun, change a parameter,
# force a rebuild, and plan a day whose data never landed.
#
#   ./run.sh --config PATH [--queue Q] [--project P] [--domain D]
#
# --queue is passed to `flyte materialize`; use a queue with room for a few 500m tasks at once.
# Steps 2 onwards need the Union lineage service and flyteplugins-union.
# Deploying pipeline.py registers a nightly trigger (03:00); deactivate it when you are done:
#   flyte update trigger nightly-stats refresh-daily-stats.nightly_stats --deactivate
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=""
SCOPE=()
QUEUE=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --queue) QUEUE=(--queue "$2"); shift 2 ;;
    --project|--domain) SCOPE+=("$1" "$2"); shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${CONFIG}" ]] || { echo "--config PATH is required" >&2; exit 2; }
cd "${HERE}"

FLYTE=(flyte --config "${CONFIG}")
step() { echo; echo "=== $*"; }
materialize() { "${FLYTE[@]}" materialize artifact ${SCOPE[@]+"${SCOPE[@]}"} "$@"; }
build() { materialize "$@" ${QUEUE[@]+"${QUEUE[@]}"} --wait; }

step "1. deploy, and land 2026-09-01 .. 2026-09-14"
"${FLYTE[@]}" deploy ${SCOPE[@]+"${SCOPE[@]}"} --root-dir . pipeline.py env
"${FLYTE[@]}" deploy ${SCOPE[@]+"${SCOPE[@]}"} --root-dir . land.py env
"${FLYTE[@]}" run ${SCOPE[@]+"${SCOPE[@]}"} --root-dir . --follow land.py land --start 2026-09-01 --end 2026-09-14

# The lineage graph picks up a deploy within about a minute.
for _ in $(seq 1 30); do
  if materialize weekly_stats --partition date=2026-09-07 --plan >/dev/null 2>&1; then break; fi
  sleep 5
done

step "2. plan one day"
materialize weekly_stats --partition date=2026-09-07 --plan

step "3. build it"
build weekly_stats --partition date=2026-09-07

step "4. backfill a week (the days shared with step 3 are reused), then again (all reused)"
build weekly_stats --partition date=2026-09-08..2026-09-14 --concurrency 8
build weekly_stats --partition date=2026-09-08..2026-09-14 --concurrency 8

step "5. a new min_fare rebuilds what depends on it; --rebuild forces one step and what follows"
build weekly_stats --partition date=2026-09-07 --input trips-etl.clean.min_fare=5
build weekly_stats --partition date=2026-09-14 --rebuild trips-etl.summarize

step "6. a day whose raw files never landed: the plan names them, nothing runs"
if materialize weekly_stats --partition date=2026-09-20 --plan; then
  echo "expected the plan to report missing raw_trips partitions" >&2
  exit 1
fi

echo
echo "Done. Each run's page shows what it built and reused; the Lineage view shows the graph."
