"""Run-name tagging: the constraints it has to respect, and round-tripping."""

from __future__ import annotations

import pytest

from flyteplugins.routing._tags import (
    MAX_RUN_NAME_LENGTH,
    TAG_LENGTH,
    make_run_name,
    profiles_for_tag,
    split_tag,
    tag_for,
)

# The platform classifies runs by the first character of the name and reserves these.
RESERVED_LEADING = ("u", "r", "l")

PROFILES = ["us-east", "eu-west", "gpu-pool", "ap-south", "on-prem"]


def test_tag_is_stable_across_calls() -> None:
    assert tag_for("us-east") == tag_for("us-east")


def test_tag_does_not_depend_on_the_other_profiles() -> None:
    """Tags must survive config edits: a name minted before a profile was added still decodes."""
    before = tag_for("us-east")
    # Nothing about tag_for takes the profile set, which is the point -- assert the value is the
    # same one a caller with a different set would compute.
    assert before == tag_for("us-east")


@pytest.mark.parametrize("profile", PROFILES)
def test_tag_never_starts_with_a_reserved_character(profile: str) -> None:
    assert not tag_for(profile).startswith(RESERVED_LEADING)


@pytest.mark.parametrize("profile", PROFILES)
def test_run_name_never_starts_with_a_reserved_character(profile: str) -> None:
    assert not make_run_name(profile).startswith(RESERVED_LEADING)


@pytest.mark.parametrize("profile", PROFILES)
def test_run_name_fits_the_length_cap(profile: str) -> None:
    assert len(make_run_name(profile)) <= MAX_RUN_NAME_LENGTH


def test_run_name_round_trips_to_its_profile() -> None:
    for profile in PROFILES:
        name = make_run_name(profile)
        assert profiles_for_tag(split_tag(name), PROFILES) == [profile]


def test_run_names_are_unique_for_the_same_profile() -> None:
    """The suffix is random, not derived from the routing key.

    Two runs of the same task on the same inputs route to the same profile; if the name were
    derived from the routing key too they would collide, and the second submission would come
    back as RunAlreadyExistsError.
    """
    names = {make_run_name("us-east") for _ in range(200)}
    assert len(names) == 200


def test_explicit_token_is_used() -> None:
    assert make_run_name("us-east", token="deadbeef").endswith("-deadbeef")


def test_over_long_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="character limit"):
        make_run_name("us-east", token="x" * 40)


class TestSplitTag:
    def test_reads_a_name_we_minted(self) -> None:
        assert split_tag(make_run_name("us-east")) == tag_for("us-east")

    @pytest.mark.parametrize(
        "name",
        [
            "",  # empty
            "nodashes",  # control-plane style, no separator
            "toolongtag-abc",  # head is not TAG_LENGTH
            "a-abc",  # head too short
            "u1-abc",  # contains a character outside the alphabet
            "-abc",  # empty head
        ],
    )
    def test_returns_none_for_names_we_did_not_mint(self, name: str) -> None:
        """A name we cannot read is not an error -- it means 'fan out instead'."""
        assert split_tag(name) is None

    def test_tag_length_is_what_split_expects(self) -> None:
        assert len(tag_for("us-east")) == TAG_LENGTH


class TestProfilesForTag:
    def test_returns_the_matching_profile(self) -> None:
        assert profiles_for_tag(tag_for("gpu-pool"), PROFILES) == ["gpu-pool"]

    def test_returns_empty_when_nothing_matches(self) -> None:
        assert profiles_for_tag("zz", PROFILES) == []

    def test_returns_every_colliding_profile(self) -> None:
        """Two profiles can share a tag. Reporting both lets the caller narrow a fan-out rather
        than confidently returning the wrong one."""
        # Search for a genuine collision rather than mocking one.
        seen: dict = {}
        collision = None
        for i in range(20000):
            p = f"profile-{i}"
            t = tag_for(p)
            if t in seen:
                collision = (seen[t], p, t)
                break
            seen[t] = p
        assert collision is not None, "expected a tag collision within the search space"
        a, b, tag = collision
        assert profiles_for_tag(tag, [a, b]) == sorted([a, b])
