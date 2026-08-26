"""Encoding a routing decision into a run name, and reading it back out.

A run name is all `flyte get run <name>` is given, so it is the only place a decision can be
recorded without keeping state somewhere.

The control plane caps names at 30 characters and reserves names beginning with `u`, `r` or `l`
(it classifies runs by the first character). The tag alphabet therefore excludes those three,
which only works because the tag is hash-derived rather than spelled from the profile name.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Iterable, List

__all__ = ["TAG_LENGTH", "make_run_name", "profiles_for_tag", "split_tag", "tag_for"]

# `u`, `r` and `l` are omitted: the platform reserves names starting with them.
_ALPHABET = "abcdefghijkmnopqstvwxyz0123456789"

#: Characters of tag. Two gives 1089 distinct tags, which makes a collision between the handful of
#: profiles a config file realistically declares unlikely -- and `profiles_for_tag` handles the
#: case where one happens anyway rather than guessing.
TAG_LENGTH = 2

#: Control-plane cap on a run name.
MAX_RUN_NAME_LENGTH = 30


def _digest(*parts: str) -> int:
    """A stable 64-bit digest. Stable across processes and releases, unlike `hash()`."""
    return int.from_bytes(hashlib.sha256("\x00".join(parts).encode()).digest()[:8], "big")


def tag_for(profile: str) -> str:
    """The tag for a profile.

    Derived from the profile name alone, so adding or removing other profiles does not change it
    and names minted last month still decode today.
    """
    n = _digest("tag", profile)
    chars = []
    for _ in range(TAG_LENGTH):
        n, i = divmod(n, len(_ALPHABET))
        chars.append(_ALPHABET[i])
    return "".join(chars)


def make_run_name(profile: str, token: str | None = None) -> str:
    """Mint a tagged run name for a routing decision.

    The suffix is random, deliberately. Two runs of the same task on the same inputs route to the
    same profile, and would collide on a name if the suffix came from the routing key -- the
    second submission returning `RunAlreadyExistsError`. Placement is deterministic; names are not.

    Args:
        profile: The profile this run was routed to, encoded as the name's leading tag.
        token: Supplies the suffix instead of fresh randomness. For tests.
    """
    suffix = token if token is not None else secrets.token_hex(4)
    name = f"{tag_for(profile)}-{suffix}"
    if len(name) > MAX_RUN_NAME_LENGTH:
        raise ValueError(f"Run name {name!r} exceeds the {MAX_RUN_NAME_LENGTH}-character limit")
    return name


def split_tag(run_name: str) -> str | None:
    """The tag from a run name, or None if the name is not one we minted.

    Names the control plane generated, or a user chose, will not have this shape.
    """
    head, _, rest = run_name.partition("-")
    if not rest or len(head) != TAG_LENGTH:
        return None
    if any(c not in _ALPHABET for c in head):
        return None
    return head


def profiles_for_tag(tag: str, profiles: Iterable[str]) -> List[str]:
    """Every profile whose tag matches.

    Normally one. Two profiles can collide on a tag; returning both lets the caller decline rather
    than guess.
    """
    return sorted(p for p in profiles if tag_for(p) == tag)
