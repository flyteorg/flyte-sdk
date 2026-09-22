from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from ._metadata import Metadata


@contextmanager
def produces(**outputs: Metadata) -> Iterator[None]:
    """
    Declare that outputs of the task called inside this block are artifacts.

    The caller names the outputs, so a task that knows nothing about artifacts can still produce
    them: its outputs are published by the platform exactly as if it had returned
    `flyte.artifacts.new(...)`, with the action that ran it recorded as the source. Keyword names are
    output slots, `o0` for the first output, `o1` for the second, and so on.

    ```python
    with flyte.artifacts.produces(o0=Metadata(name="events", partitions={"date": day})):
        await clean.override(produces_artifacts=True)(raw=raw)
    ```

    The called task must run with `produces_artifacts=True`: that flag is what lets the platform
    publish its outputs. If the task also wraps an output itself, this declaration wins for the
    artifact's name, version, partitions and parents, and the task's description, card and attrs fill
    in whatever it leaves empty.

    Every task called inside the block receives the declarations, so call one task per block. They
    are not passed on to the actions that task itself spawns. Outside a task this does nothing.
    """
    from flyteidl2.core import types_pb2

    from flyte._context import internal_ctx
    from flyte._internal.runtime.convert import PRODUCED_ARTIFACTS_CONTEXT_KEY, encode_declared_artifacts

    from ._metadata import to_produced_artifact

    # Validate eagerly, so a bad declaration fails in the caller rather than in the task it calls.
    declarations = [
        to_produced_artifact(md, output=slot, literal_type=types_pb2.LiteralType()) for slot, md in outputs.items()
    ]

    ctx = internal_ctx()
    tctx = ctx.data.task_context
    if tctx is None:
        yield
        return
    custom = {**tctx.custom_context, PRODUCED_ARTIFACTS_CONTEXT_KEY: encode_declared_artifacts(declarations)}
    with ctx.replace_task_context(tctx.replace(custom_context=custom)):
        yield
