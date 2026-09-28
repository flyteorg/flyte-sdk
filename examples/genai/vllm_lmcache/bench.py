"""
Benchmark the LMCache router: compare routing modes, and measure L2 against recompute.

Usage::

    export ROUTER_URL=https://<router-endpoint>
    export FLYTE_API_KEY=<token>          # the router has requires_auth=True
    python examples/genai/vllm_lmcache/bench.py modes --sessions 24 --turns 4
    python examples/genai/vllm_lmcache/bench.py probe --prompt-tokens 8000

``modes`` runs the same multi-turn workload once per routing mode. Each mode gets freshly
salted prompts, so one mode never benefits from KV that an earlier mode stored in L2. It
reports TTFT p50/p95 and the L0 and LMCache hit rates computed from /stats deltas.

``probe`` measures one long prompt three ways: recomputed on worker A, fetched from L2 on
worker B (which has never seen it), and served from L0 on worker B. That shows whether L2
over this network is faster than recomputing for this model and GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import statistics
import time
import uuid

import httpx

MODEL = os.getenv("MODEL_ID", "qwen3-8b")
WORDS = (
    "orchestration cache tensor replica prefix router latency throughput valkey memory "
    "session token block worker cluster queue gateway model serving request response"
).split()


def document(seed: str, approx_tokens: int) -> str:
    # Roughly one token per word for this vocabulary.
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(approx_tokens))


def headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {"content-type": "application/json"}
    if key := os.getenv("FLYTE_API_KEY"):
        h["authorization"] = f"Bearer {key}"
    return {**h, **(extra or {})}


async def chat(
    client: httpx.AsyncClient, url: str, messages: list[dict], hdrs: dict[str, str], max_tokens: int = 32
) -> tuple[float, str, dict[str, str]]:
    """Stream one chat completion; return (ttft_seconds, text, response headers)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0}
    start = time.perf_counter()
    ttft, parts = None, []
    async with client.stream("POST", f"{url}/v1/chat/completions", json=body, headers=headers(hdrs)) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if ttft is None:
                ttft = time.perf_counter() - start
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            parts.append(delta.get("content") or "")
        return ttft or (time.perf_counter() - start), "".join(parts), dict(r.headers)


async def stats(client: httpx.AsyncClient, url: str) -> dict:
    r = await client.get(f"{url}/stats", headers=headers())
    r.raise_for_status()
    return r.json()


def rate(before: dict, after: dict, hits: str, total: str) -> float | None:
    d_total = after.get(total, 0) - before.get(total, 0)
    return round((after.get(hits, 0) - before.get(hits, 0)) / d_total, 3) if d_total else None


async def run_mode(client: httpx.AsyncClient, url: str, mode: str, args: argparse.Namespace) -> dict:
    salt = uuid.uuid4().hex[:8]
    system_prompts = [document(f"{salt}-{k}", args.system_tokens) for k in range(args.system_prompts)]
    ttfts: list[float] = []
    reasons: collections.Counter = collections.Counter()
    sem = asyncio.Semaphore(args.concurrency)

    async def session(i: int) -> None:
        sid = f"{salt}-s{i}"
        messages = [{"role": "system", "content": system_prompts[i % len(system_prompts)]}]
        for t in range(args.turns):
            messages.append({"role": "user", "content": f"Turn {t}: summarize the document in one line. {sid}"})
            async with sem:
                ttft, text, h = await chat(client, url, messages, {"x-route-mode": mode, "x-session-id": sid})
            ttfts.append(ttft)
            reasons[h.get("x-route-reason", "?")] += 1
            messages.append({"role": "assistant", "content": text})

    before = (await stats(client, url))["total"]
    start = time.perf_counter()
    await asyncio.gather(*(session(i) for i in range(args.sessions)))
    elapsed = time.perf_counter() - start
    after = (await stats(client, url))["total"]

    ttfts.sort()
    return {
        "mode": mode,
        "requests": len(ttfts),
        "ttft_p50_ms": round(statistics.median(ttfts) * 1000),
        "ttft_p95_ms": round(ttfts[int(0.95 * (len(ttfts) - 1))] * 1000),
        "req_per_s": round(len(ttfts) / elapsed, 2),
        "l0_hit_rate": rate(before, after, "vllm:prefix_cache_hits", "vllm:prefix_cache_queries"),
        "lmcache_hit_rate": rate(before, after, "lmcache:num_hit_tokens", "lmcache:num_requested_tokens"),
        "reasons": dict(reasons),
    }


async def modes(args: argparse.Namespace) -> None:
    url = os.environ["ROUTER_URL"].rstrip("/")
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        results = [await run_mode(client, url, m, args) for m in args.modes.split(",")]
    cols = ["mode", "requests", "ttft_p50_ms", "ttft_p95_ms", "req_per_s", "l0_hit_rate", "lmcache_hit_rate"]
    print("\n" + " | ".join(f"{c:>16}" for c in cols))
    for r in results:
        print(" | ".join(f"{r[c]!s:>16}" for c in cols) + f"   {r['reasons']}")


async def probe(args: argparse.Namespace) -> None:
    url = os.environ["ROUTER_URL"].rstrip("/")
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        healthy = (await client.get(f"{url}/health", headers=headers())).json()["healthy_workers"]
        if len(healthy) < 2:
            raise SystemExit(f"need two healthy workers, have {healthy}")
        a, b = healthy[:2]
        messages = [
            {"role": "system", "content": document(uuid.uuid4().hex, args.prompt_tokens)},
            {"role": "user", "content": "Reply with one word."},
        ]
        recompute, _, _ = await chat(client, url, messages, {"x-route-worker": a}, max_tokens=1)
        # LMCache writes to L2 asynchronously; give the store a moment to land.
        await asyncio.sleep(args.settle_s)
        from_l2, _, _ = await chat(client, url, messages, {"x-route-worker": b}, max_tokens=1)
        from_l0, _, _ = await chat(client, url, messages, {"x-route-worker": b}, max_tokens=1)
    print(f"prompt ~{args.prompt_tokens} tokens")
    print(f"  recompute on {a}:  {recompute * 1000:8.0f} ms")
    print(f"  L2 fetch on {b}:   {from_l2 * 1000:8.0f} ms")
    print(f"  L0 hit on {b}:     {from_l0 * 1000:8.0f} ms")
    print(f"  L2 speedup vs recompute: {recompute / from_l2:.2f}x")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("modes")
    m.add_argument("--modes", default="prefix,sticky,roundrobin,random")
    m.add_argument("--sessions", type=int, default=24)
    m.add_argument("--turns", type=int, default=4)
    m.add_argument("--system-prompts", type=int, default=4)
    m.add_argument("--system-tokens", type=int, default=6000)
    m.add_argument("--concurrency", type=int, default=12)
    pr = sub.add_parser("probe")
    pr.add_argument("--prompt-tokens", type=int, default=8000)
    pr.add_argument("--settle-s", type=float, default=3.0)
    args = p.parse_args()
    asyncio.run(modes(args) if args.cmd == "modes" else probe(args))


if __name__ == "__main__":
    main()
