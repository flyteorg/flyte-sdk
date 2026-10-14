"""Every provider plugin runs the same conformance check.

CI fails here if this plugin drifts from the shared format: a verifier that
raises instead of returning False, event constants that render as enum names, a
sample delivery that no constant spells, an unstable dedupe key.
"""

from __future__ import annotations

from flyte.extras.webhooks.testing import assert_provider_conforms

import flyteplugins.linear as plugin


def test_conformance():
    assert_provider_conforms(plugin)
