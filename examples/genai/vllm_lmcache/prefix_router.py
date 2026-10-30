"""
Routing logic for the LMCache example: sticky sessions and prefix-aware routing, with all
shared state kept in Valkey.

Every vLLM worker has private L0 (GPU) and L1 (CPU) KV tiers and shares one L2 (Valkey).
A request that lands on a worker that no longer holds its prefix in L0/L1 still hits L2,
so routing only has to be good, not exact. The router therefore keeps an approximate
index: which worker last served each prefix block. Because the index lives in Valkey,
every router replica sees the same state.

This module has no FastAPI or tokenizer dependencies, so it can be unit tested against a
local ``valkey-server``. It works with any ``redis.asyncio``-compatible client
(``valkey.asyncio`` in the router image).

Keys (all with TTLs), under ``rt:{namespace}``:

- ``pfx:{block_hash}``: worker that last served the prompt prefix ending at this block
- ``sess:{session_id}``: worker a session is pinned to
- ``load:{worker}``: in-flight requests on a worker
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import random
from array import array
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

logger = logging.getLogger(__name__)

RouteMode = Literal["prefix", "sticky", "roundrobin", "random"]
ROUTE_MODES: tuple[str, ...] = ("prefix", "sticky", "roundrobin", "random")


@dataclass(frozen=True)
class RouterConfig:
    # Tenant and model, so two deployments sharing a Valkey never read each other's index.
    namespace: str
    # Matches LMCache's chunk size, so "matched tokens" lines up with what L0/L1 can reuse.
    block_tokens: int = 256
    max_blocks: int = 128
    # Shorter matches are not worth sending a request to a busier worker for.
    min_match_tokens: int = 512
    # A prefix or session owner is used only while its load is within this many requests
    # of the least-loaded worker. Beyond that, spilling over and paying for an L2 fetch
    # beats queueing.
    load_slack: int = 4
    max_inflight: int = 64
    # Roughly how long a prefix stays in a worker's L0/L1. Tune it from the bench.
    prefix_ttl_s: int = 600
    session_ttl_s: int = 1800
    load_ttl_s: int = 900
    # Valkey is on the request path; if it is slower than this, route round-robin instead.
    valkey_timeout_s: float = 0.02
    num_candidates: int = 3


@dataclass
class Decision:
    """A ranked list of workers (primary first) and why the primary was chosen."""

    candidates: list[str]
    reason: str
    matched_tokens: int = 0
    block_hashes: list[str] = field(default_factory=list)


def block_hashes(tokens: Sequence[int], block_tokens: int, max_blocks: int) -> list[str]:
    """Chained hashes over full token blocks: h_i = H(h_{i-1} || tokens_i).

    A block's hash depends on every token before it, so a worker that served block i has
    also processed blocks 0..i-1. Partial trailing blocks are ignored, as in LMCache.
    """
    out: list[str] = []
    prev = b""
    n = min(len(tokens) // block_tokens, max_blocks)
    for i in range(n):
        chunk = array("I", tokens[i * block_tokens : (i + 1) * block_tokens]).tobytes()
        prev = hashlib.blake2b(prev + chunk, digest_size=16).digest()
        out.append(prev.hex())
    return out


def _rendezvous(key: str, workers: Sequence[str]) -> list[str]:
    """Order workers by rendezvous hash, so identical cold prefixes converge on one worker."""
    return sorted(workers, key=lambda w: hashlib.blake2b(f"{key}|{w}".encode(), digest_size=8).digest(), reverse=True)


class PrefixRouter:
    def __init__(self, client: Any, workers: Sequence[str], config: RouterConfig):
        if not workers:
            raise ValueError("at least one worker is required")
        self._r = client
        self.workers = list(workers)
        self.cfg = config
        self._rr = itertools.count()
        self._prefix = f"rt:{config.namespace}"

    # -- keys ---------------------------------------------------------------------------

    def _pfx(self, h: str) -> str:
        return f"{self._prefix}:pfx:{h}"

    def _sess(self, s: str) -> str:
        return f"{self._prefix}:sess:{s}"

    def _load(self, w: str) -> str:
        return f"{self._prefix}:load:{w}"

    # -- selection ----------------------------------------------------------------------

    def _round_robin(self, healthy: Sequence[str], reason: str) -> Decision:
        start = next(self._rr) % len(healthy)
        ordered = list(healthy[start:]) + list(healthy[:start])
        return Decision(candidates=ordered[: self.cfg.num_candidates], reason=reason)

    async def _read_state(
        self, hashes: Sequence[str], session_id: str | None, workers: Sequence[str]
    ) -> tuple[dict[str, int], str | None, list[str | None]]:
        pipe = self._r.pipeline(transaction=False)
        pipe.mget([self._load(w) for w in workers])
        if session_id:
            pipe.get(self._sess(session_id))
        else:
            pipe.echo("")
        if hashes:
            pipe.mget([self._pfx(h) for h in hashes])
        res = await asyncio.wait_for(pipe.execute(), timeout=self.cfg.valkey_timeout_s)
        loads = {w: max(0, int(v or 0)) for w, v in zip(workers, res[0])}
        session_owner = _s(res[1]) if session_id else None
        owners = [_s(v) for v in res[2]] if hashes else []
        return loads, session_owner, owners

    async def pick(
        self,
        tokens: Sequence[int],
        session_id: str | None = None,
        mode: str = "prefix",
        healthy: Sequence[str] | None = None,
    ) -> Decision:
        """Choose a ranked list of workers for one request.

        ``healthy`` is the current ready-set; candidates are never drawn from outside it.
        """
        pool = [w for w in (healthy or self.workers) if w in self.workers] or self.workers
        hashes = block_hashes(tokens, self.cfg.block_tokens, self.cfg.max_blocks)

        if mode == "roundrobin":
            d = self._round_robin(pool, "roundrobin")
            d.block_hashes = hashes
            return d
        if mode == "random":
            shuffled = random.sample(pool, len(pool))
            return Decision(candidates=shuffled[: self.cfg.num_candidates], reason="random", block_hashes=hashes)

        try:
            loads, session_owner, owners = await self._read_state(hashes if mode == "prefix" else [], session_id, pool)
        except Exception as e:  # Valkey slow or down: lose affinity, keep serving.
            logger.warning("valkey read failed, routing round-robin: %r", e)
            d = self._round_robin(pool, "fallback")
            d.block_hashes = hashes
            return d

        min_load = min(loads.values())

        def ok(w: str | None) -> bool:
            return (
                w is not None
                and w in loads
                and loads[w] <= min_load + self.cfg.load_slack
                and loads[w] < self.cfg.max_inflight
            )

        # Deepest indexed block per worker. The owner of block i holds the whole prefix
        # up to i, so this is each worker's matched prefix length.
        depth: dict[str, int] = {}
        for i, w in enumerate(owners):
            if w is not None and w in loads:
                depth[w] = i + 1
        by_depth = sorted(depth, key=lambda w: depth[w], reverse=True)

        seed = hashes[0] if hashes else (session_id or "")
        by_load = sorted(_rendezvous(seed, pool), key=lambda w: loads[w])

        primary, reason, matched = None, "least_loaded", 0
        if ok(session_owner):
            primary, reason = session_owner, "session"
            matched = depth.get(session_owner, 0) * self.cfg.block_tokens
        else:
            for w in by_depth:
                if depth[w] * self.cfg.block_tokens < self.cfg.min_match_tokens:
                    break
                if ok(w):
                    primary, reason, matched = w, "prefix", depth[w] * self.cfg.block_tokens
                    break
            if primary is None and (session_owner or by_depth):
                reason = "spill"  # an owner exists but is overloaded or the match is too short
        if primary is None:
            primary = by_load[0]
            matched = depth.get(primary, 0) * self.cfg.block_tokens

        ranked: list[str] = []
        for w in [primary, *by_depth, *by_load]:
            if w not in ranked:
                ranked.append(w)
        return Decision(
            candidates=ranked[: self.cfg.num_candidates],
            reason=reason,
            matched_tokens=matched,
            block_hashes=hashes,
        )

    # -- bookkeeping --------------------------------------------------------------------

    async def record(self, worker: str, hashes: Sequence[str], session_id: str | None = None) -> None:
        """Remember that ``worker`` now holds this prefix (and session). Best effort."""
        try:
            pipe = self._r.pipeline(transaction=False)
            for h in hashes:
                pipe.set(self._pfx(h), worker, ex=self.cfg.prefix_ttl_s)
            if session_id:
                pipe.set(self._sess(session_id), worker, ex=self.cfg.session_ttl_s)
            await asyncio.wait_for(pipe.execute(), timeout=self.cfg.valkey_timeout_s * 5)
        except Exception as e:
            logger.warning("valkey record failed: %r", e)

    async def acquire(self, worker: str) -> None:
        try:
            pipe = self._r.pipeline(transaction=False)
            pipe.incr(self._load(worker))
            pipe.expire(self._load(worker), self.cfg.load_ttl_s)
            await asyncio.wait_for(pipe.execute(), timeout=self.cfg.valkey_timeout_s * 5)
        except Exception as e:
            logger.warning("valkey acquire failed: %r", e)

    async def release(self, worker: str) -> None:
        try:
            v = await asyncio.wait_for(self._r.decr(self._load(worker)), timeout=self.cfg.valkey_timeout_s * 5)
            if int(v) < 0:
                await self._r.set(self._load(worker), 0, ex=self.cfg.load_ttl_s)
        except Exception as e:
            logger.warning("valkey release failed: %r", e)

    async def loads(self) -> dict[str, int]:
        vals = await self._r.mget([self._load(w) for w in self.workers])
        return {w: max(0, int(v or 0)) for w, v in zip(self.workers, vals)}


def _s(v: Any) -> str | None:
    if v is None or v == b"" or v == "":
        return None
    return v.decode() if isinstance(v, bytes) else str(v)
