"""
Deploy-time check of `Artifact.ref(...)` handles against the artifact registry.

A reference restates partitions that another codebase owns. Nothing at import time can tell whether the
restatement is still right, so `flyte.deploy` asks the registry for each referenced name's partition schema
(fixed by the owner's `Artifact.declare` or first version) and fails on a disagreement, with the line to paste.
A name the registry does not know yet is reported, not failed: the owner may simply not have deployed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flyte.errors import LineageDeclarationError

from ._handle import Artifact, _Granularity, dimension_type_name, type_name

_GRANULARITY_MARKER = {"hour": "Hourly", "day": "Daily", "week": "Weekly", "month": "Monthly"}


@dataclass
class ReferenceCheck:
    """
    What checking a deploy's references found.

    Attributes:
        checked: References whose partitions match the registry.
        notes: One line per reference that could not be checked (not in the registry yet, or no backend).
    """

    checked: int = 0
    notes: List[str] = field(default_factory=list)


def _where(h: Artifact) -> str:
    return f"{h.src_file}:{h.src_line}" if h.src_file else h.name


def _ref_dims(h: Artifact) -> Tuple[Optional[Tuple[str, str]], Tuple[str, ...]]:
    time: Optional[Tuple[str, str]] = None
    keys: List[str] = []
    for d, t in h.partitions.items():
        if isinstance(t, _Granularity):
            time = (d, t.registry)
        else:
            keys.append(d)
    return time, tuple(keys)


def _suggested(h: Artifact, schema: Any) -> str:
    """The `Artifact.ref(...)` line that matches the registry, keeping the reference's own str/int choices."""
    parts: List[str] = []
    if schema.time_key:
        parts.append(f'"{schema.time_key}": artifacts.{_GRANULARITY_MARKER.get(schema.granularity or "day")}')
    for k in schema.keys:
        t = h.partitions.get(k)
        parts.append(f'"{k}": {"int" if t is int else "str"}')
    args = [f'"{h.name}"']
    if h.type is not None:
        args.append(f"type={type_name(h.type).rsplit('.', 1)[-1]}")
    if parts:
        args.append("partitions={" + ", ".join(parts) + "}")
    if h.project:
        args.append(f'project="{h.project}"')
    if h.domain:
        args.append(f'domain="{h.domain}"')
    return f"artifacts.Artifact.ref({', '.join(args)})"


def compare(h: Artifact, schema: Any) -> Optional[str]:
    """Why reference `h` disagrees with the registry's partition schema, or None when it agrees."""
    time, keys = _ref_dims(h)
    want_time = (schema.time_key, schema.granularity or "day") if schema.time_key else None
    if time == want_time and set(keys) == set(schema.keys):
        return None
    stated = ", ".join(f"{d}: {dimension_type_name(t)}" for d, t in h.partitions.items()) or "none"
    owner: List[str] = []
    if schema.time_key:
        owner.append(f"{schema.time_key}: {_GRANULARITY_MARKER.get(schema.granularity or 'day')}")
    owner.extend(schema.keys)
    return (
        f"artifact reference {h.name!r} at {_where(h)} states partitions [{stated}], but its owner's declaration "
        f"in the registry has [{', '.join(owner) or 'none'}]. Update the reference:\n    "
        f"{h.name} = {_suggested(h, schema)}"
    )


def references(handles: Iterable[Artifact]) -> List[Artifact]:
    """The distinct references among `handles`, one per (name, project, domain)."""
    out: Dict[Tuple[str, Optional[str], Optional[str]], Artifact] = {}
    for h in handles:
        if h.reference:
            out.setdefault((h.name, h.project, h.domain), h)
    return list(out.values())


#: Per-reference and overall bounds on the registry lookups `check_references` makes; a timeout is a note.
REF_CHECK_TIMEOUT = 5.0
REF_CHECK_TOTAL_TIMEOUT = 20.0


async def check_references(handles: Iterable[Artifact]) -> ReferenceCheck:
    """
    Check every `Artifact.ref` in `handles` against the registry.

    Raises:
        LineageDeclarationError: when a reference's partitions disagree with its owner's, naming every one.
    """
    refs = references(handles)
    result = ReferenceCheck()
    if not refs:
        return result
    from flyte.remote import Artifact as RemoteArtifact

    async def schema_of(h: Artifact) -> Tuple[Artifact, Any, Optional[Exception]]:
        try:
            schema = await asyncio.wait_for(
                RemoteArtifact.get_schema.aio(h.name, project=h.project, domain=h.domain), REF_CHECK_TIMEOUT
            )
            return h, schema, None
        except asyncio.TimeoutError:
            return h, None, TimeoutError(f"timed out after {REF_CHECK_TIMEOUT:g}s")
        except Exception as e:  # NOT_FOUND, or no backend at all (an offline dry run)
            return h, None, e

    # A slow or unreachable registry must never hold up (or fail) a deploy: each lookup is bounded, and so is
    # the whole check; whatever is unfinished becomes a note.
    tasks = {asyncio.ensure_future(schema_of(h)): h for h in refs}
    done, pending = await asyncio.wait(tasks, timeout=REF_CHECK_TOTAL_TIMEOUT)
    for t in pending:
        t.cancel()
    outcomes: List[Tuple[Artifact, Any, Optional[Exception]]] = []
    for t, h in tasks.items():
        if t in done and not t.cancelled() and t.exception() is None:
            outcomes.append(t.result())
        else:
            outcomes.append((h, None, TimeoutError(f"overall check timed out after {REF_CHECK_TOTAL_TIMEOUT:g}s")))

    problems: List[str] = []
    for h, schema, err in outcomes:
        if schema is None:
            if _is_not_found(err):
                result.notes.append(
                    f"artifact reference {h.name!r} is not in the registry yet (no declaration or version), so its "
                    "partitions are unchecked; they are checked on the next deploy after its owner publishes"
                )
            else:
                result.notes.append(f"artifact reference {h.name!r} could not be checked against the registry ({err})")
            continue
        msg = compare(h, schema)
        if msg:
            problems.append(msg)
        else:
            result.checked += 1
    if problems:
        raise LineageDeclarationError("\n".join(problems))
    return result


def _is_not_found(err: Optional[Exception]) -> bool:
    code = getattr(err, "code", None)
    try:
        name = getattr(code() if callable(code) else code, "name", "")
    except Exception:
        name = ""
    text = f"{type(err).__name__} {err}".lower()
    return name in ("NOT_FOUND", "not_found") or "notfound" in text.replace("_", "") or "not found" in text
