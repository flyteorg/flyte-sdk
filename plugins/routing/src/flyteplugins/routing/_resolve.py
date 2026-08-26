"""Finding which profile a run lives on, for the read path.

A run-addressed command is handed a name and nothing else. Routed runs carry their target in the
name, so decoding the tag is enough.

There is deliberately no search: a name this plugin did not mint resolves to None and the command
runs under whatever profile is in effect. Probing every cluster would cost a round trip per
profile on every lookup, and make one unreachable cluster everybody's problem on every command.
"""

from __future__ import annotations

from typing import List, Optional

import flyte.config as config

from ._tags import profiles_for_tag, split_tag

__all__ = ["candidate_profiles", "resolve_run_profile"]


def candidate_profiles(config_file=None) -> List[str]:
    """Profiles declared by the config file."""
    return sorted(config.list_profiles(config_file))


def resolve_run_profile(run_name: str, config_file=None) -> Optional[str]:
    """The profile holding `run_name`, or None to leave the ambient profile alone.

    None covers a name with no decodable tag and a tag naming a profile the config no longer
    declares. Both mean "not a run I placed" -- let the command run against the default.
    """
    profiles = candidate_profiles(config_file)
    if not profiles:
        return None

    tag = split_tag(run_name)
    if tag is None:
        return None

    matches = profiles_for_tag(tag, profiles)
    # Exactly one match is the answer. Zero means the tag is stale or not ours. Two means a tag
    # collision, and with no search to fall back on there is nothing to choose between them --
    # better to use the default than to pick one at random and report it confidently.
    return matches[0] if len(matches) == 1 else None
