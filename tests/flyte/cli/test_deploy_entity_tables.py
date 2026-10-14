"""Deploy renders one table per entity kind, not one table per deploy."""

import json

from flyte.cli._deploy import _print_entity_rows

TASK_ROW = [("type", "task"), ("name", "env.t1"), ("version", "v1"), ("triggers", "")]
APP_ROW = [
    ("type", "App"),
    ("name", "my-app"),
    ("revision", "0"),
    ("desired state", "DESIRED_STATE_ACTIVE"),
    ("current state", "DEPLOYMENT_STATUS_UNSPECIFIED"),
    ("public_url", ""),
    ("console_url", ""),
]


def _tables(rows, output_format="table"):
    printed = []
    import flyte.cli._deploy as deploy_mod

    original = deploy_mod.common.print_output
    deploy_mod.common.print_output = lambda renderable, of: printed.append(renderable)
    try:
        _print_entity_rows(rows, "Entities", output_format)
    finally:
        deploy_mod.common.print_output = original
    return printed


def test_task_and_app_rows_get_separate_tables():
    # A recursive deploy mixes kinds whose rows carry different columns. Rendered as one
    # table, the app values file under the task headers (revision under "Version", desired
    # state under "Triggers") and the overflow becomes unlabeled columns.
    tables = _tables([TASK_ROW, APP_ROW])

    assert len(tables) == 2
    tasks, apps = tables
    assert tasks.title == "Tasks"
    assert [c.header for c in tasks.columns] == ["Type", "Name", "Version", "Triggers"]
    assert tasks.row_count == 1

    assert apps.title == "Apps"
    assert [c.header for c in apps.columns] == [
        "Type",
        "Name",
        "Revision",
        "Desired state",
        "Current state",
        "Public_url",
        "Console_url",
    ]
    assert apps.row_count == 1


def test_homogeneous_rows_stay_in_one_table():
    tables = _tables([TASK_ROW, TASK_ROW])

    assert len(tables) == 1
    assert tables[0].title == "Tasks"
    assert tables[0].row_count == 2


def test_no_rows_prints_nothing():
    assert _tables([]) == []


def test_json_raw_stays_a_single_flat_document():
    # Splitting JSON per kind would emit two documents; each row already serializes
    # on its own keys, so one flat list is correct.
    out = _tables([TASK_ROW, APP_ROW], output_format="json-raw")

    assert len(out) == 1
    parsed = json.loads(out[0])
    assert [row["type"] for row in parsed] == ["task", "App"]
