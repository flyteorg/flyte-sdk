from flyte.models import ActionID

from flyteplugins.otel import format_trace_id, trace_id_for_run

MAX_TRACE_ID = 2**128 - 1


def action(name="a0", run_name="run-abc", project="proj", domain="dev", org="acme"):
    return ActionID(name=name, run_name=run_name, project=project, domain=domain, org=org)


def test_same_run_gives_same_trace_id_from_any_process():
    """The whole point: two containers derive the id independently and agree."""
    assert trace_id_for_run(action()) == trace_id_for_run(action())


def test_the_action_within_the_run_does_not_change_the_trace_id():
    """Every action in a run has to land in the same trace, whichever one asks."""
    assert trace_id_for_run(action(name="a0")) == trace_id_for_run(action(name="a7"))


def test_runs_are_distinguished_by_full_identity():
    base = trace_id_for_run(action())
    assert trace_id_for_run(action(run_name="run-xyz")) != base
    assert trace_id_for_run(action(project="other")) != base
    assert trace_id_for_run(action(domain="prod")) != base
    assert trace_id_for_run(action(org="other")) != base


def test_trace_id_is_a_valid_w3c_id():
    trace_id = trace_id_for_run(action())
    assert 0 < trace_id <= MAX_TRACE_ID
    assert len(format_trace_id(trace_id)) == 32


def test_unset_identity_fields_do_not_crash():
    """Local runs leave org and friends unset; the id still has to be derivable."""
    trace_id = trace_id_for_run(ActionID(name="a0"))
    assert 0 < trace_id <= MAX_TRACE_ID


def test_run_name_defaults_to_action_name():
    """ActionID fills run_name from name when it is omitted, so these are one run."""
    assert trace_id_for_run(ActionID(name="solo")) == trace_id_for_run(ActionID(name="solo", run_name="solo"))
