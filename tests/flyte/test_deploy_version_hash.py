"""
The deploy auto-version hashes the pickled environments. Lineage added attributes to environments, tasks, apps and
`ArtifactValue`; a deploy that does not use lineage must still hash exactly what it did before they existed, or an
SDK upgrade re-versions every task (and re-binds every trigger) on the next deploy.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

import cloudpickle

import flyte
from flyte._deploy import _version_dumps
from flyte.app import AppEnvironment, Parameter
from flyte.app._parameter import ArtifactValue

env = flyte.TaskEnvironment(name="version_hash_env", image="ghcr.io/org/image:1", resources=flyte.Resources(cpu=1))


@env.task
async def double(x: int) -> int:
    return x * 2


@env.task(cache="auto")
async def main(x: int) -> int:
    return await double(x)


app = AppEnvironment(
    name="version-hash-app",
    image="ghcr.io/org/image:1",
    parameters=[Parameter(name="model", value=ArtifactValue(name="model"), mount="/model")],
)


@contextmanager
def _without_lineage_attributes(*envs):
    """Delete the attributes lineage added, so the objects pickle as they did before lineage (then restore them)."""
    saved = []

    def drop(obj, *names):
        for name in names:
            if name in obj.__dict__:
                saved.append((obj, name, obj.__dict__.pop(name)))

    private = []
    for e in envs:
        if isinstance(e, AppEnvironment):
            drop(e, "labels", "consumes_artifacts", "lineage")
            for p in e.parameters:
                if isinstance(p.value, ArtifactValue) and "_handle" in (p.value.__pydantic_private__ or {}):
                    private.append((p.value, p.value.__pydantic_private__.pop("_handle")))
        else:
            drop(e, "labels")
            for t in e.tasks.values():
                drop(t, "labels", "consumes_artifacts")
    try:
        yield
    finally:
        for obj, name, value in saved:
            obj.__dict__[name] = value
        for value, handle in private:
            value.__pydantic_private__["_handle"] = handle


def test_version_pickle_without_lineage_matches_pre_lineage_pickle():
    envs = {"version_hash_env": env, "version-hash-app": app}
    projected = _version_dumps(envs)
    with _without_lineage_attributes(env, app):
        pre_lineage = cloudpickle.dumps(envs)
    assert hashlib.md5(projected).hexdigest() == hashlib.md5(pre_lineage).hexdigest()


def test_version_pickle_keeps_lineage_attributes_that_are_set():
    labeled = env.clone_with(name="version_hash_labeled_env", labels={"team": "ml"})
    plain = _version_dumps({"e": env.clone_with(name="version_hash_labeled_env")})
    assert _version_dumps({"e": labeled}) != plain
    assert b"team" in _version_dumps({"e": labeled})
