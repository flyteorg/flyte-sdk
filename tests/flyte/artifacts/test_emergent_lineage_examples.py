"""The graph emerges from separately authored modules: import every example module under
examples/artifacts/4_multi_team and run deploy-time extraction over all of them, with no backend."""

import importlib
import json
import pathlib
import sys

import pytest

import flyte
from flyte.app import AppEnvironment
from flyte.artifacts._lineage import BINDINGS_LABEL, app_env_lineage_labels, summarize
from flyte.artifacts._refresh import refresh_envs

EXAMPLE_ROOT = pathlib.Path(__file__).resolve().parents[3] / "examples" / "artifacts" / "4_multi_team"
MODULES = [
    "ingest.events",
    "ingest.seed",
    "ml.features",
    "ml.train",
    "analytics.report",
    "analytics.notify",
    "triggers.revalidate",
    "triggers.weekly_review",
    "apps.scoring",
    "apps.dashboard",
    "legacy.legacy_clean",
    "legacy.adapter",
    "notebook.explore",
]


@pytest.fixture(scope="module")
def loaded():
    sys.path.insert(0, str(EXAMPLE_ROOT))
    try:
        mods = {m: importlib.import_module(m) for m in MODULES}
        yield mods
    finally:
        sys.path.remove(str(EXAMPLE_ROOT))
        for m in list(sys.modules):
            if m.split(".")[0] in {"ingest", "ml", "analytics", "triggers", "apps", "legacy", "notebook"}:
                sys.modules.pop(m, None)


def _envs(mods):
    envs = [v for m in mods.values() for v in vars(m).values() if isinstance(v, flyte.Environment)]
    # What deploying them brings along: the refresh environments of the artifacts they produce.
    return envs + [e for e in refresh_envs(envs) if e not in envs]


def test_graph_emerges_across_modules(loaded):
    envs = _envs(loaded)
    tasks = [t for e in envs if isinstance(e, flyte.TaskEnvironment) for t in e.tasks.values()]
    s = summarize(tasks, root_dir=EXAMPLE_ROOT)
    assert s.line() == "✓ 12 tasks, 5 artifact handles, 5 dependency edges resolved"
    assert s.warnings == []
    assert set(s.nodes) == {"raw_events", "events", "features", "churn_model", "daily_report"}
    assert set(s.edge_set) == {
        ("raw_events", "events"),
        ("events", "features"),
        ("features", "churn_model"),
        ("features", "daily_report"),
        ("churn_model", "daily_report"),
    }
    by_task = {lin.task: lin for lin in s.lineages}
    # Refresh policies: inbound triggers, with a label edge to the target and no typed binding. The owner's
    # (refresh= on daily_report) comes with the analytics deploy; ML Quality's is its own environment.
    nightly = by_task["refresh-daily-report.keep_report_fresh"]
    assert (nightly.consumes, nightly.produces) == (["daily_report"], [])
    assert nightly.bindings["materialize_on"] == {
        "target": "daily_report",
        "event": {"kind": "cron", "cron": "0 2 * * *", "timezone": "UTC"},
        "lag": {"days": 1, "hours": 0},
    }
    assert nightly.bindings["parameters"] == {} and nightly.src_file == "analytics/report.py"
    weekly = by_task["refresh-churn-model.for_weekly_review"]
    assert weekly.bindings["materialize_on"]["event"]["cron"] == "0 5 * * 1"
    assert weekly.src_file == "triggers/weekly_review.py"
    # A typed sink: consumes daily_report by identity, publishes nothing; `date` is bound implicitly.
    send = by_task["analytics-notify.send_report"]
    assert (send.consumes, send.produces, send.pullable) == (["daily_report"], [], True)
    assert send.bindings["parameters"]["date"]["implicit"] is True
    assert send.bindings["parameters"]["to"]["kind"] == "default"
    # Implicit and explicit partition bindings record the same thing as get_partition_value.
    feat = json.loads(by_task["ml.featurize"].labels[BINDINGS_LABEL])["parameters"]["date"]
    assert (feat["kind"], feat["node"], feat["dim"], feat["implicit"]) == ("partition", "features", "date", True)
    rep = json.loads(by_task["analytics.report"].labels[BINDINGS_LABEL])["parameters"]["date"]
    assert (rep["kind"], rep["node"], rep["dim"]) == ("partition", "daily_report", "date") and "implicit" not in rep
    # Seeding publishes at run time only: no edge, raw_events stays a source.
    assert by_task["ingest-seed.publish_raw"].produces == []
    # The adapter is a second producer of events, with the same edge as clean.
    assert by_task["ingest-adapter.clean_adapter"].edges == [("raw_events", "events")]
    clean = json.loads(by_task["ingest.clean"].labels[BINDINGS_LABEL])
    assert clean["src_file"] == "ingest/events.py" and clean["level"] == 5
    assert clean["artifacts"]["raw_events"]["source"] is True
    assert clean["artifacts"]["events"]["expected"] == {"region": ["us", "eu"]}
    train = json.loads(by_task["ml-train.train"].labels[BINDINGS_LABEL])
    assert train["parameters"]["history"]["mapping"] == {"kind": "window", "dim": "date", "days": 30, "hours": 0}
    assert train["artifacts"]["features"]["src_file"] == "ml/features.py"
    assert train["outputs"] == 1
    src = (EXAMPLE_ROOT / "ml" / "train.py").read_text().splitlines()
    lr = train["parameters"]["lr"]
    assert lr["src_file"] == "ml/train.py" and "lr: float" in src[lr["src_line"] - 1]
    assert all(lin.pullable for lin in s.lineages if lin.produces)


def test_each_module_validates_on_its_own(loaded):
    """Every module is deployable separately: its own extraction passes with only the handles it imports."""
    for name, mod in loaded.items():
        tasks = [t for v in vars(mod).values() if isinstance(v, flyte.TaskEnvironment) for t in v.tasks.values()]
        summarize(tasks, root_dir=EXAMPLE_ROOT)


def test_apps_and_triggers(loaded):
    apps = {e.name: e for e in _envs(loaded) if isinstance(e, AppEnvironment)}
    scoring = app_env_lineage_labels(apps["churn-scoring"])
    scoring_bindings = json.loads(scoring.pop(BINDINGS_LABEL))
    assert scoring == {"team": "ml", "lineage.consumes": "churn_model"}
    assert {p: b["node"] for p, b in scoring_bindings["parameters"].items()} == {"model": "churn_model"}
    dashboard = app_env_lineage_labels(apps["churn-dashboard"])
    # Label-only: an empty record still tells a snapshot the app has nothing to bind (rather than "unknown").
    assert json.loads(dashboard.pop(BINDINGS_LABEL)) == {
        "version": 1,
        "app": "churn-dashboard",
        "parameters": {},
        "artifacts": {},
    }
    assert dashboard == {"team": "analytics", "lineage.consumes": "app:churn-scoring,daily_report"}
    revalidate = loaded["triggers.revalidate"].revalidate
    assert revalidate.automation.name == "churn_model"
