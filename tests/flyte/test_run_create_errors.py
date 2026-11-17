"""`flyte run` against a backend that refuses CreateRun because the org is at a limit fails with a user error,
not a `RuntimeSystemError` that reaches Sentry as an SDK crash (FLYTE-SDK-70)."""

import mock
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from mock.mock import AsyncMock

import flyte
import flyte.errors
from flyte._initialize import _init_for_testing

from .test_rerun import _fake_prior_run, _mock_client_with_run

_QUEUE_DEPTH_MSG = 'action "acme:p:d:r1:a0": org "acme" is at its max_queue_depth limit (5), retry later'


async def _rerun_with_create_run_error(err: ConnectError):
    mock_client, mock_run_service, _, _ = _mock_client_with_run()
    mock_run_service.create_run.side_effect = err
    await _init_for_testing(client=mock_client, project="test", domain="test")
    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=_fake_prior_run())
        await flyte.with_runcontext(mode="remote").rerun.aio("r1")


@pytest.mark.asyncio
async def test_resource_exhausted_is_a_user_error():
    with pytest.raises(flyte.errors.ResourceExhaustedError) as exc_info:
        await _rerun_with_create_run_error(ConnectError(Code.RESOURCE_EXHAUSTED, _QUEUE_DEPTH_MSG))
    assert isinstance(exc_info.value, flyte.errors.RuntimeUserError)
    assert "max_queue_depth" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ConnectError)


@pytest.mark.asyncio
async def test_org_limit_refusal_is_a_user_error(org_limit_detail):
    err = ConnectError(Code.FAILED_PRECONDITION, "org at max_actions_per_run", details=[org_limit_detail()])
    with pytest.raises(flyte.errors.OrgLimitExceededError):
        await _rerun_with_create_run_error(err)


@pytest.mark.asyncio
async def test_max_concurrent_runs_refusal_names_the_limit(org_limit_detail):
    detail = org_limit_detail(
        limit="max_concurrent_runs",
        ceiling=1000,
        current=1000,
        requested=1,
        tier_default=1000,
        upgrade_url="https://www.union.ai/contact",
    )
    err = ConnectError(Code.RESOURCE_EXHAUSTED, 'org "acme" is at its max_concurrent_runs limit', details=[detail])
    with pytest.raises(flyte.errors.ResourceExhaustedError) as exc_info:
        await _rerun_with_create_run_error(err)
    assert exc_info.value.org_limit.limit == "max_concurrent_runs"
    assert exc_info.value.org_limit.ceiling == 1000
    assert str(exc_info.value) == (
        "Org limit max_concurrent_runs (1000) reached: 1000 in use, 1 more requested. "
        "Wait for running work to finish, or contact Union to raise it: https://www.union.ai/contact"
    )


@pytest.mark.asyncio
async def test_entitlement_refusal_is_a_user_error(org_entitlement_detail):
    detail = org_entitlement_detail(entitlement="committed_capacity", upgrade_url="https://www.union.ai/contact")
    err = ConnectError(Code.PERMISSION_DENIED, 'org "acme" is not entitled to committed_capacity', details=[detail])
    with pytest.raises(flyte.errors.OrgEntitlementRequiredError) as exc_info:
        await _rerun_with_create_run_error(err)
    assert exc_info.value.entitlement == "committed_capacity"
    assert "Contact Union to enable it: https://www.union.ai/contact" in str(exc_info.value)


@pytest.mark.asyncio
async def test_permission_denied_without_detail_is_unchanged():
    with pytest.raises(flyte.errors.RuntimeSystemError, match="Failed to create run: not a member"):
        await _rerun_with_create_run_error(ConnectError(Code.PERMISSION_DENIED, "not a member"))


@pytest.mark.asyncio
async def test_internal_error_stays_a_system_error():
    with pytest.raises(flyte.errors.RuntimeSystemError, match="Failed to create run"):
        await _rerun_with_create_run_error(ConnectError(Code.INTERNAL, "boom"))
