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


def suite_image_packages() -> list[str]:
    """Pip spec(s) to install *this suite* into a scenario's task/app image, or ``[]``
    to rely on the fast-register code bundle instead.

    When ``FLYTE_FUNCTIONAL_SUITE_SPEC`` is set — e.g. a released version
    ``flyte-functional-tests==0.1.0`` or a branch
    ``flyte-functional-tests @ git+https://github.com/flyteorg/flyte-sdk@<ref>#subdirectory=tests/functional`` —
    every scenario image installs the suite so the task/app pod can import its module
    from site-packages. This is required whenever the suite is a *non-editable* install:
    its source then lives in site-packages, not under the run's working dir, so flyte's
    ``loaded_modules`` code bundle (which excludes site-packages) never ships it.

    Unset (the default) => install nothing extra and rely on the code bundle — correct
    when the suite runs from a source checkout under the working dir (flyte-sdk dev, or a
    consumer that vendors/checks out the suite).
    """
    spec = os.environ.get("FLYTE_FUNCTIONAL_SUITE_SPEC")
    return [spec] if spec else []


def suite_base_image(*extra_pip: str, extra_env: dict[str, str] | None = None):
    """Shared scenario image recipe: ``from_debian_base`` + the suite install (when
    ``FLYTE_FUNCTIONAL_SUITE_SPEC`` is set) + any ``extra_pip`` packages + the image
    cache-bust env (plus any ``extra_env``).

    Centralised so every scenario's pod resolves its task module identically — from the
    installed suite (spec set) or the code bundle (spec unset) — without each module
    re-deriving the recipe.
    """
    import flyte  # type: ignore

    img = flyte.Image.from_debian_base()
    pkgs = [*suite_image_packages(), *extra_pip]
    if pkgs:
        img = img.with_pip_packages(*pkgs)
    return img.with_env_vars({**image_cache_bust(), **(extra_env or {})})
