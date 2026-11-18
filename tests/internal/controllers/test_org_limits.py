"""Decoding the org limit and entitlement error details into typed errors with a readable message."""

import pickle

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

import flyte.errors
from flyte._internal.controllers.remote._org_limits import (
    _DETAILS,
    describe_limit,
    org_entitlement_required,
    org_limit_exceeded,
    org_refusal,
    resource_exhausted,
)

_URL = "https://www.union.ai/get-a-quote"
_SERVER_MSG = 'org "acme" would exceed its max_clusters quota (2 current + 1 requested > 2)'


def test_org_limit_exceeded_decodes_every_field(org_limit_detail):
    detail = org_limit_detail(
        limit="max_clusters", ceiling=4, current=4, requested=1, tier_default=2, upgrade_url=_URL, tier="Teams"
    )
    err = org_limit_exceeded(ConnectError(Code.FAILED_PRECONDITION, _SERVER_MSG, details=[detail]))

    assert err.org_limit == flyte.errors.OrgLimit(
        limit="max_clusters", ceiling=4, current=4, requested=1, tier_default=2, upgrade_url=_URL, tier="Teams"
    )
    assert str(err) == (
        "Org limit max_clusters (4) reached: 4 in use, 1 more requested. The Teams tier default is 2. "
        f"Contact sales to raise it: {_URL}"
    )


def test_tier_default_without_a_tier_name(org_limit_detail):
    detail = org_limit_detail(limit="max_clusters", ceiling=4, current=4, requested=1, tier_default=2)
    err = org_limit_exceeded(ConnectError(Code.FAILED_PRECONDITION, _SERVER_MSG, details=[detail]))

    assert err.org_limit.tier == ""
    assert "The org's tier default is 2." in str(err)


def test_legacy_detail_alone_is_a_refusal_with_the_servers_message(legacy_org_limit_detail):
    """A server from before the flyteidl2 details sends only cloudidl.common.OrgLimitExceeded."""
    err = org_limit_exceeded(ConnectError(Code.FAILED_PRECONDITION, _SERVER_MSG, details=[legacy_org_limit_detail()]))

    assert isinstance(err, flyte.errors.OrgLimitExceededError)
    assert err.org_limit is None
    assert str(err) == _SERVER_MSG


def test_legacy_detail_alongside_reads_the_public_one(org_limit_detail, legacy_org_limit_detail):
    detail = org_limit_detail(limit="max_clusters", ceiling=4, current=4, requested=1)
    error = ConnectError(Code.FAILED_PRECONDITION, _SERVER_MSG, details=[legacy_org_limit_detail(), detail])

    assert org_limit_exceeded(error).org_limit.limit == "max_clusters"


def test_tier_default_equal_to_ceiling_is_not_repeated(org_limit_detail):
    detail = org_limit_detail(limit="max_clusters", ceiling=2, current=2, requested=1, tier_default=2, upgrade_url=_URL)
    err = org_limit_exceeded(ConnectError(Code.FAILED_PRECONDITION, _SERVER_MSG, details=[detail]))

    assert err.org_limit.tier_default == 2
    assert (
        str(err) == f"Org limit max_clusters (2) reached: 2 in use, 1 more requested. Contact sales to raise it: {_URL}"
    )


def test_tier_default_zero_is_kept_apart_from_unset(org_limit_detail):
    """0 (unlimited) is a tier default; an unset one is None."""
    zero = org_limit_detail(limit="max_users", ceiling=5, current=5, requested=1, tier_default=0)
    unset = org_limit_detail(limit="max_users", ceiling=5, current=5, requested=1)

    assert org_limit_exceeded(ConnectError(Code.FAILED_PRECONDITION, "", details=[zero])).org_limit.tier_default == 0
    assert (
        org_limit_exceeded(ConnectError(Code.FAILED_PRECONDITION, "", details=[unset])).org_limit.tier_default is None
    )


def test_older_server_without_tier_default_or_url(org_limit_detail):
    """A server from before tier_default and upgrade_url still gets a readable message."""
    detail = org_limit_detail(limit="max_clusters", ceiling=2, current=2, requested=1)
    err = org_limit_exceeded(ConnectError(Code.FAILED_PRECONDITION, _SERVER_MSG, details=[detail]))

    assert err.org_limit.upgrade_url == ""
    assert str(err) == "Org limit max_clusters (2) reached: 2 in use, 1 more requested. Contact sales to raise it."


def test_unreadable_detail_falls_back_to_the_server_message(org_limit_detail):
    err = org_limit_exceeded(ConnectError(Code.FAILED_PRECONDITION, _SERVER_MSG, details=[org_limit_detail()]))

    assert isinstance(err, flyte.errors.OrgLimitExceededError)
    assert err.org_limit is None
    assert str(err) == _SERVER_MSG


def test_resource_exhausted_with_detail_says_to_wait(org_limit_detail):
    detail = org_limit_detail(
        limit="max_concurrent_runs", ceiling=1000, current=1000, requested=1, tier_default=1000, upgrade_url=_URL
    )
    err = resource_exhausted(ConnectError(Code.RESOURCE_EXHAUSTED, "at max_concurrent_runs", details=[detail]))

    assert isinstance(err, flyte.errors.ResourceExhaustedError)
    assert isinstance(err, flyte.errors.SlowDownError), "still retried by the controller"
    assert err.org_limit.limit == "max_concurrent_runs"
    assert str(err) == (
        "Org limit max_concurrent_runs (1000) reached: 1000 in use, 1 more requested. "
        f"Wait for running work to finish, or contact sales to raise it: {_URL}"
    )


def test_resource_exhausted_without_detail_keeps_the_server_message():
    err = resource_exhausted(ConnectError(Code.RESOURCE_EXHAUSTED, "slow down"), prefix="Failed: ")

    assert err.org_limit is None
    assert str(err) == "Failed: slow down"


def test_failed_precondition_detail_is_not_a_resource_exhausted(org_limit_detail):
    detail = org_limit_detail(limit="max_clusters", ceiling=2, current=2, requested=1)
    assert resource_exhausted(ConnectError(Code.FAILED_PRECONDITION, "", details=[detail])) is None
    assert org_limit_exceeded(ConnectError(Code.RESOURCE_EXHAUSTED, "", details=[detail])) is None


def test_entitlement_refusal(org_entitlement_detail):
    detail = org_entitlement_detail(entitlement="allowed_clouds", values=["coreweave", "nebius"], upgrade_url=_URL)
    err = org_entitlement_required(
        ConnectError(Code.PERMISSION_DENIED, 'org "acme" is not entitled to allowed_clouds', details=[detail])
    )

    assert isinstance(err, flyte.errors.RuntimeUserError)
    assert (err.entitlement, err.values, err.upgrade_url) == ("allowed_clouds", ["coreweave", "nebius"], _URL)
    assert (
        str(err) == f"This org isn't entitled to allowed_clouds (coreweave, nebius). Contact sales to enable it: {_URL}"
    )


def test_bool_entitlement_without_url(org_entitlement_detail):
    detail = org_entitlement_detail(entitlement="private_control_plane_endpoint")
    err = org_entitlement_required(ConnectError(Code.PERMISSION_DENIED, "no", details=[detail]))

    assert str(err) == "This org isn't entitled to private_control_plane_endpoint. Contact sales to enable it."


def test_permission_denied_without_detail_is_not_an_entitlement_refusal():
    assert org_entitlement_required(ConnectError(Code.PERMISSION_DENIED, "not a member")) is None
    assert org_refusal(ConnectError(Code.PERMISSION_DENIED, "not a member")) is None


def test_org_refusal_needs_a_detail(org_limit_detail, org_entitlement_detail):
    limit = org_limit_detail(limit="max_queue_depth", ceiling=5, current=5, requested=1)
    assert isinstance(
        org_refusal(ConnectError(Code.RESOURCE_EXHAUSTED, "", details=[limit])), flyte.errors.ResourceExhaustedError
    )
    assert org_refusal(ConnectError(Code.RESOURCE_EXHAUSTED, "rate limited")) is None
    assert isinstance(
        org_refusal(ConnectError(Code.FAILED_PRECONDITION, "", details=[limit])), flyte.errors.OrgLimitExceededError
    )
    ent = org_entitlement_detail(entitlement="byok")
    assert isinstance(
        org_refusal(ConnectError(Code.PERMISSION_DENIED, "", details=[ent])), flyte.errors.OrgEntitlementRequiredError
    )


def test_describe_limit_without_requested():
    limit = flyte.errors.OrgLimit(limit="max_users", ceiling=5, current=5, requested=0)
    assert (
        describe_limit(limit, retry_later=False)
        == "Org limit max_users (5) reached: 5 in use. Contact sales to raise it."
    )


@pytest.mark.parametrize(
    "err",
    [
        flyte.errors.OrgLimitExceededError("m", org_limit=flyte.errors.OrgLimit("max_clusters", 2, 2, 1, 2, _URL)),
        flyte.errors.ResourceExhaustedError("m", org_limit=flyte.errors.OrgLimit("max_concurrent_runs", 1, 1, 1)),
        flyte.errors.OrgEntitlementRequiredError("m", entitlement="byok", values=["a"], upgrade_url=_URL),
    ],
)
def test_errors_pickle_with_their_fields(err):
    """Errors cross process boundaries (e.g. flyte.map), so the fields must survive pickling."""
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is type(err)
    assert back.__dict__ == err.__dict__
    assert str(back) == str(err)


def test_detail_type_names_are_the_generated_messages():
    for type_name, cls in _DETAILS.items():
        assert cls.DESCRIPTOR.full_name == type_name
