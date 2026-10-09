"""The ETL and train-and-serve examples under examples/artifacts deploy cleanly: import their modules and run
deploy-time extraction, with no backend. The graphs their READMEs draw must be the graphs the code declares."""

import importlib
import json
import pathlib
import sys

import pytest

import flyte
from flyte.app import AppEnvironment
from flyte.artifacts._lineage import BINDINGS_LABEL, app_env_lineage_labels, summarize
from flyte.artifacts._refresh import refresh_envs

ROOT = pathlib.Path(__file__).resolve().parents[3] / "examples" / "artifacts"


def _load(example: str, modules):
    """Import ``modules`` from one example directory, then forget them: the examples reuse module names."""
    root = str(ROOT / example)
    sys.path.insert(0, root)
    try:
        return {m: importlib.import_module(m) for m in modules}
    finally:
        sys.path.remove(root)
        for m in modules:
            sys.modules.pop(m, None)


def _envs(mods):
    envs = [v for m in mods.values() for v in vars(m).values() if isinstance(v, flyte.Environment)]
    return envs + [e for e in refresh_envs(envs) if e not in envs]


def _tasks(envs):
    return [t for e in envs if isinstance(e, flyte.TaskEnvironment) for t in e.tasks.values()]


@pytest.fixture(scope="module")
def etl():
    return _load("2_etl_backfill", ["pipeline", "land"])


@pytest.fixture(scope="module")
def train_and_serve():
    return _load("3_train_and_serve", ["data", "train", "serve"])


def test_etl_graph(etl):
    s = summarize(_tasks(_envs(etl)), root_dir=ROOT / "2_etl_backfill")
    assert s.line() == "✓ 6 tasks, 4 artifact handles, 3 dependency edges resolved"
    assert s.warnings == []
    assert set(s.edge_set) == {("raw_trips", "trips"), ("trips", "daily_stats"), ("daily_stats", "weekly_stats")}
    by_task = {lin.task: lin for lin in s.lineages}
    clean = json.loads(by_task["trips-etl.clean"].labels[BINDINGS_LABEL])
    # date and city are bound by their names; min_fare is a plain default the README overrides with --input.
    assert {p: (b["kind"], b.get("implicit")) for p, b in clean["parameters"].items() if p != "raw"} == {
        "date": ("partition", True),
        "city": ("partition", True),
        "min_fare": ("default", None),
    }
    assert clean["artifacts"]["trips"]["expected"] == {"city": ["nyc", "sf"]}
    summarize_ = json.loads(by_task["trips-etl.summarize"].labels[BINDINGS_LABEL])
    assert summarize_["parameters"]["per_city"]["mapping"] == {"kind": "all", "dim": "city"}
    rollup = json.loads(by_task["trips-etl.rollup"].labels[BINDINGS_LABEL])
    assert rollup["parameters"]["week"]["mapping"] == {"kind": "window", "dim": "date", "days": 7, "hours": 0}
    nightly = by_task["refresh-daily-stats.nightly_stats"]
    assert nightly.bindings["materialize_on"]["event"] == {"kind": "cron", "cron": "0 3 * * *", "timezone": "UTC"}
    assert nightly.bindings["materialize_on"]["lag"] == {"days": 1, "hours": 0}
    # Landing publishes at run time only: raw_trips stays a source.
    assert by_task["trips-landing.land_trips"].produces == []


def test_train_and_serve_graph(train_and_serve):
    envs = _envs(train_and_serve)
    s = summarize(_tasks(envs), root_dir=ROOT / "3_train_and_serve")
    assert s.warnings == []
    assert set(s.edge_set) == {("reviews", "sentiment_model")}
    by_task = {lin.task: lin for lin in s.lineages}
    train = json.loads(by_task["sentiment-train.train"].labels[BINDINGS_LABEL])
    assert train["parameters"]["history"]["mapping"] == {"kind": "window", "dim": "date", "days": 7, "hours": 0}
    assert train["parameters"]["min_accuracy"]["kind"] == "default"
    retrain = by_task["refresh-sentiment-model.retrain_on_new_reviews"]
    assert retrain.bindings["materialize_on"]["event"] == {"kind": "source", "source": "reviews", "filter": {}}
    (api,) = [e for e in envs if isinstance(e, AppEnvironment)]
    labels = app_env_lineage_labels(api)
    assert json.loads(labels.pop(BINDINGS_LABEL))["parameters"]["model"]["node"] == "sentiment_model"
    assert labels == {"lineage.consumes": "sentiment_model"}


def test_each_module_deploys_on_its_own(etl, train_and_serve):
    for example, mods in (("2_etl_backfill", etl), ("3_train_and_serve", train_and_serve)):
        for mod in mods.values():
            summarize(
                [t for v in vars(mod).values() if isinstance(v, flyte.TaskEnvironment) for t in v.tasks.values()],
                root_dir=ROOT / example,
            )
