"""The routing policy: determinism, distribution, and its stability under config change."""

from __future__ import annotations

import pytest

from flyteplugins.routing import RoutingContext, RoutingDecision, route, select_profile
from flyteplugins.routing._tags import profiles_for_tag, split_tag

PROFILES = ("us-east", "eu-west", "gpu-pool")


def _ctx(**kw) -> RoutingContext:
    kw.setdefault("profiles", PROFILES)
    kw.setdefault("task_name", "train")
    kw.setdefault("project", "p")
    kw.setdefault("domain", "d")
    return RoutingContext(**kw)


class TestSelectProfile:
    def test_is_deterministic(self) -> None:
        assert select_profile(PROFILES, "k") == select_profile(PROFILES, "k")

    def test_does_not_depend_on_the_order_profiles_are_given(self) -> None:
        assert select_profile(PROFILES, "k") == select_profile(tuple(reversed(PROFILES)), "k")

    def test_uses_every_profile(self) -> None:
        chosen = {select_profile(PROFILES, f"key-{i}") for i in range(200)}
        assert chosen == set(PROFILES)

    def test_spreads_roughly_evenly(self) -> None:
        counts = dict.fromkeys(PROFILES, 0)
        for i in range(3000):
            counts[select_profile(PROFILES, f"key-{i}")] += 1
        # Rendezvous hashing is uniform in expectation; allow generous slack for 3000 samples.
        for p, n in counts.items():
            assert 800 < n < 1200, f"{p} got {n} of 3000"

    def test_only_the_removed_profile_s_keys_move(self) -> None:
        """Why rendezvous rather than `hash(key) % len(profiles)`.

        Modulo reshuffles nearly every key when the profile set changes, throwing away the cache
        locality the policy exists to provide. Rendezvous moves only the keys that belonged to the
        profile that went away.
        """
        keys = [f"key-{i}" for i in range(1000)]
        before = {k: select_profile(PROFILES, k) for k in keys}
        remaining = tuple(p for p in PROFILES if p != "eu-west")
        after = {k: select_profile(remaining, k) for k in keys}

        moved = [k for k in keys if before[k] != after[k]]
        # Every moved key must be one that used to live on the removed profile...
        assert all(before[k] == "eu-west" for k in moved)
        # ...and every key that lived there must have moved.
        assert set(moved) == {k for k in keys if before[k] == "eu-west"}


class TestRoute:
    def test_declines_when_there_are_no_profiles(self) -> None:
        assert route(_ctx(profiles=())) is None

    def test_declines_when_the_user_pinned_a_profile(self) -> None:
        """`--profile` is an explicit choice; a policy does not get to overrule it."""
        assert route(_ctx(active_profile="eu-west")) is None

    def test_returns_a_decision_for_a_run(self) -> None:
        d = route(_ctx(inputs={"x": 1}))
        assert isinstance(d, RoutingDecision)
        assert d.profile in PROFILES

    def test_same_task_and_arguments_route_to_the_same_profile(self) -> None:
        a = route(_ctx(inputs={"dataset": "s3://bucket/x", "epochs": 3}))
        b = route(_ctx(inputs={"epochs": 3, "dataset": "s3://bucket/x"}))  # order must not matter
        assert a.profile == b.profile

    def test_different_arguments_can_route_differently(self) -> None:
        """Data-location routing: placement follows the arguments, not just the task."""
        chosen = {route(_ctx(inputs={"dataset": f"s3://bucket/{i}"})).profile for i in range(100)}
        assert len(chosen) > 1

    def test_task_project_and_domain_are_part_of_the_key(self) -> None:
        by_task = {route(_ctx(task_name=f"t{i}", inputs={"x": 1})).profile for i in range(100)}
        by_domain = {route(_ctx(domain=f"d{i}", inputs={"x": 1})).profile for i in range(100)}
        by_project = {route(_ctx(project=f"p{i}", inputs={"x": 1})).profile for i in range(100)}
        assert len(by_task) > 1 and len(by_domain) > 1 and len(by_project) > 1

    def test_run_name_decodes_back_to_the_chosen_profile(self) -> None:
        d = route(_ctx(inputs={"x": 1}))
        assert profiles_for_tag(split_tag(d.run_name), PROFILES) == [d.profile]

    def test_repeat_runs_get_the_same_profile_but_different_names(self) -> None:
        """The pairing the scheme rests on: deterministic placement, unique names."""
        inputs = {"dataset": "s3://bucket/x"}
        a, b = route(_ctx(inputs=inputs)), route(_ctx(inputs=inputs))
        assert a.profile == b.profile
        assert a.run_name != b.run_name

    def test_does_not_name_a_run_the_caller_named(self) -> None:
        d = route(_ctx(inputs={"x": 1}, run_name="mine"))
        assert d.run_name is None
        assert d.profile in PROFILES

    def test_labels_record_the_decision(self) -> None:
        d = route(_ctx(inputs={"x": 1}))
        assert d.labels == {"routed-by": "consistent-hash", "routed-to": d.profile}

    def test_unhashable_arguments_do_not_break_routing(self) -> None:
        d = route(_ctx(inputs={"rows": [1, 2, 3], "opts": {"a": 1}}))
        assert d.profile in PROFILES

    @pytest.mark.parametrize("inputs", [{}, {"x": None}])
    def test_empty_or_null_arguments_still_route(self, inputs) -> None:
        assert route(_ctx(inputs=inputs)).profile in PROFILES
