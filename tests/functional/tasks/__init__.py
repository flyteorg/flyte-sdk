"""Scenario task modules for the functional suite (one per scenario)."""

from __future__ import annotations

import os


def image_cache_bust() -> dict[str, str]:
    """Image env that busts the image-build cache when
    ``FLYTE_FUNCTIONAL_IMAGE_CACHE_BUST`` is set.

    Baked into each custom task image, so a non-empty value changes the image
    identity and forces a fresh build. Needed on backends whose object store is
    ephemeral or GC'd (a sandbox / k3d), where a stale cross-run build-cache hit can
    reference a collected output and fail the build. Empty by default, so a
    persistent store keeps normal cross-run caching.
    """
    return {"FLYTE_FUNCTIONAL_CACHE_BUST": os.environ.get("FLYTE_FUNCTIONAL_IMAGE_CACHE_BUST", "")}
