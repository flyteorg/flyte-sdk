"""Browser-use agent backed by Lightpanda (https://lightpanda.io).

Lightpanda is a headless browser engine written from scratch in Zig for
automation and AI agents. It speaks the Chrome DevTools Protocol, so Playwright
drives it unchanged — but it ships as a single ~150 MB static binary with no
rendering stack, which makes it roughly an order of magnitude lighter than a
headless Chromium in both startup time and memory. That trade is a good fit for
Flyte: the browser lives *inside* the task image, so each tool call gets a real
JavaScript-executing browser without a Selenium grid or a browser service.

What this example wires together:

* ``browser_env`` — a :class:`flyte.TaskEnvironment` whose image installs the
  Lightpanda binary at build time (see ``LIGHTPANDA_INSTALL``). Each tool is an
  ``@browser_env.task``, so the agent's browsing runs as a durable Flyte action
  in this image, isolated from the lightweight chat-UI container.
* A CDP session helper that starts ``lightpanda serve`` once per container and
  connects Playwright to it over ``ws://``. Lightpanda allows one page per CDP
  connection, so concurrency comes from opening several connections — that is
  what :func:`read_pages` does.
* An A/B switch: set ``LIGHTPANDA_MODE=cloud`` to drive Lightpanda Cloud's
  hosted browsers over ``wss://`` instead of the in-container binary, with no
  other code change. Every tool reports ``elapsed_ms`` and ``engine`` so you can
  see what each mode actually costs.

Deploy the chat UI (set ANTHROPIC_SECRET_NAME if your cluster names the
Anthropic key something other than ``internal-anthropic-api-key``)::

    uv run python examples/agents/lightpanda_browser_agent.py

Verify the browser wiring locally first, without a cluster (needs a local
Lightpanda: ``curl -fsSL https://pkg.lightpanda.io/install.sh | bash``)::

    LIGHTPANDA_BIN=./lightpanda uv run python examples/agents/lightpanda_browser_agent.py --check

Note that Lightpanda has no rendering engine: there are no screenshots or PDFs,
and it is Beta software, so some Web APIs are still missing. For scripted,
non-agentic extraction its one-shot CLI is often enough and needs no CDP at all:
``lightpanda fetch --dump markdown https://example.com``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
import time
from typing import Any, AsyncIterator

import flyte
from flyte.ai.agents import Agent
from flyte.ai.chat import AgentChatAppEnvironment, CustomTheme

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# "local" runs the binary baked into the task image; "cloud" connects to
# Lightpanda Cloud instead. Both speak CDP, so only the endpoint changes.
MODE = os.environ.get("LIGHTPANDA_MODE", "local").strip().lower()
CLOUD_REGION = os.environ.get("LIGHTPANDA_CLOUD_REGION", "euwest")

LIGHTPANDA_BIN = os.environ.get("LIGHTPANDA_BIN", "/usr/local/bin/lightpanda")

# Whatever your cluster calls the Anthropic key; it is always mounted as
# ANTHROPIC_API_KEY, which is what litellm reads.
ANTHROPIC_SECRET = os.environ.get("ANTHROPIC_SECRET_NAME", "internal-anthropic-api-key")
CDP_HOST = "127.0.0.1"
CDP_PORT = int(os.environ.get("LIGHTPANDA_PORT", "9222"))

# Lightpanda serves one page per CDP connection and accepts 16 connections by
# default (--cdp-max-connections), so parallel reads open one connection each.
MAX_PARALLEL_PAGES = 8
NAV_TIMEOUT_MS = 20_000

# `uname -m` reports x86_64 / aarch64, which is exactly how the release assets
# are named — so one command covers both amd64 and arm64 builders.
LIGHTPANDA_INSTALL = [
    "curl -fsSL -o /usr/local/bin/lightpanda"
    " https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-$(uname -m)-linux",
    "chmod 755 /usr/local/bin/lightpanda",
]

browser_env = flyte.TaskEnvironment(
    name="lightpanda-browser-tools",
    image=(
        flyte.Image.from_debian_base()
        # ca-certificates is required: without it every HTTPS navigation fails
        # with PeerFailedVerification. Those two are the only system deps —
        # the binary itself links against nothing but libc.
        .with_apt_packages("ca-certificates", "curl")
        .with_commands(LIGHTPANDA_INSTALL)
        # `pip install playwright` bundles the Node driver, which is all we need
        # to talk CDP. Skip `playwright install` — we never launch a browser
        # ourselves, we connect to one.
        .with_pip_packages("playwright", "litellm")
        .with_env_vars({"LIGHTPANDA_DISABLE_TELEMETRY": "true"})
    ),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[
        flyte.Secret(key=ANTHROPIC_SECRET, as_env_var="ANTHROPIC_API_KEY"),
        # For LIGHTPANDA_MODE=cloud, add your Lightpanda Cloud token here:
        # flyte.Secret(key="lightpanda-cloud-token", as_env_var="LPD_TOKEN"),
    ],
    env_vars={"LIGHTPANDA_MODE": MODE},
)


# ---------------------------------------------------------------------------
# CDP session management
# ---------------------------------------------------------------------------


class _Session:
    """Per-container browser plumbing: the CDP server and the Playwright driver.

    Both are expensive to start (relative to a page load) and cheap to keep, so
    they are created once and reused by every tool call in the process.
    """

    proc: asyncio.subprocess.Process | None = None
    driver: Any = None
    lock = asyncio.Lock()


async def _is_listening(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection(host, port)
    except OSError:
        return False
    writer.close()
    await writer.wait_closed()
    return True


async def _wait_until_listening(host: str, port: int, timeout_s: float = 15.0) -> None:
    """Poll until the CDP server accepts TCP connections, or give up."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await _is_listening(host, port):
            return
        await asyncio.sleep(0.05)
    raise RuntimeError(f"lightpanda never started listening on {host}:{port}")


async def _start_local_server() -> None:
    """Start ``lightpanda serve`` unless something is already listening on the port.

    An already-listening server — a sidecar, or one left by an earlier call — is
    adopted as-is.
    """
    if await _is_listening(CDP_HOST, CDP_PORT):
        return
    if not os.path.exists(LIGHTPANDA_BIN):
        raise RuntimeError(
            f"lightpanda binary not found at {LIGHTPANDA_BIN}. Set LIGHTPANDA_BIN, or install it with "
            "`curl -fsSL https://pkg.lightpanda.io/install.sh | bash`."
        )
    _Session.proc = await asyncio.create_subprocess_exec(
        LIGHTPANDA_BIN,
        "serve",
        "--host",
        CDP_HOST,
        "--port",
        str(CDP_PORT),
        "--log-level",
        "warn",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await _wait_until_listening(CDP_HOST, CDP_PORT)


async def _ensure_session() -> tuple[Any, str]:
    """Return the shared Playwright driver plus the CDP endpoint to connect to."""
    from playwright.async_api import async_playwright

    async with _Session.lock:
        if _Session.driver is None:
            # Starting the driver spawns a Node process; doing it per tool call
            # costs seconds, which would dwarf the page loads themselves.
            _Session.driver = await async_playwright().start()
        if MODE == "cloud":
            return _Session.driver, _cloud_cdp_url()
        if _Session.proc is None or _Session.proc.returncode is not None:
            await _start_local_server()
        return _Session.driver, f"ws://{CDP_HOST}:{CDP_PORT}/"


def _cloud_cdp_url() -> str:
    token = os.environ.get("LPD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("LIGHTPANDA_MODE=cloud needs LPD_TOKEN (from https://cloud.lightpanda.io) in the env")
    return f"wss://{CLOUD_REGION}.cloud.lightpanda.io/ws?token={token}"


def _engine_label() -> str:
    return f"lightpanda-cloud/{CLOUD_REGION}" if MODE == "cloud" else "lightpanda-local"


@contextlib.asynccontextmanager
async def _page() -> AsyncIterator[Any]:
    """Yield a Playwright page attached to one fresh Lightpanda CDP connection."""
    driver, endpoint = await _ensure_session()
    browser = await driver.chromium.connect_over_cdp(endpoint)
    try:
        # Reuse the context the connection arrives with — a second one is
        # rejected ("Cannot have more than one browser context at a time") — but
        # always open a fresh page: navigating the placeholder page that comes
        # attached to the connection hangs instead of loading.
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = await context.new_page()
        yield page
    finally:
        await browser.close()


async def _goto(page: Any, url: str, wait_until: str = "load") -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    await page.goto(url, wait_until=wait_until, timeout=NAV_TIMEOUT_MS)
    return page.url or url


def _squeeze(text: str) -> str:
    """Collapse the whitespace runs that page text is full of."""
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


# ---------------------------------------------------------------------------
# Browser tools
# ---------------------------------------------------------------------------


@browser_env.task
async def read_page(url: str, max_chars: int = 4000, wait_until: str = "load") -> dict:
    """Open a URL in a real browser and return its title and visible text.

    Scripts run before the text is captured, so this reads client-rendered pages
    that a plain HTTP fetch returns empty. Use wait_until="networkidle" for sites
    that load their content late; "domcontentloaded" is fastest.
    """
    started = time.monotonic()
    async with _page() as page:
        final_url = await _goto(page, url, wait_until)
        title = await page.title()
        body = await page.inner_text("body")
    text = _squeeze(body)
    return {
        "url": final_url,
        "title": title,
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
        "elapsed_ms": _elapsed_ms(started),
        "engine": _engine_label(),
    }


@browser_env.task
async def extract_links(url: str, contains: str = "", limit: int = 40) -> dict:
    """List the links on a page, optionally keeping only those whose URL or text matches.

    Use this to find what to read next — pass one of the returned hrefs to
    read_page. Set contains to a substring like "/docs" to narrow the results.
    """
    started = time.monotonic()
    async with _page() as page:
        final_url = await _goto(page, url)
        links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({href: e.href, text: (e.innerText || '').trim()}))",
        )

    needle = contains.strip().lower()
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for link in links:
        href = (link.get("href") or "").strip()
        text = _squeeze(link.get("text") or "")[:120]
        if not href or href.startswith("javascript:") or href in seen:
            continue
        if needle and needle not in href.lower() and needle not in text.lower():
            continue
        seen.add(href)
        out.append({"href": href, "text": text})
        if len(out) >= limit:
            break

    return {
        "url": final_url,
        "count": len(out),
        "links": out,
        "elapsed_ms": _elapsed_ms(started),
        "engine": _engine_label(),
    }


@browser_env.task
async def extract_elements(url: str, selector: str, limit: int = 20) -> dict:
    """Return the text of every element matching a CSS selector.

    Prefer this over read_page when you know the structure and want just the
    parts — e.g. selector="article h2" for headlines, or "table tr" for rows.
    """
    started = time.monotonic()
    async with _page() as page:
        final_url = await _goto(page, url)
        texts = await page.eval_on_selector_all(selector, "els => els.map(e => (e.innerText || '').trim())")

    items = [_squeeze(t) for t in texts if t and t.strip()][:limit]
    return {
        "url": final_url,
        "selector": selector,
        "count": len(items),
        "items": items,
        "elapsed_ms": _elapsed_ms(started),
        "engine": _engine_label(),
    }


@browser_env.task
async def evaluate_on_page(url: str, expression: str) -> dict:
    """Evaluate a JavaScript expression against a loaded page and return its value.

    The escape hatch for anything the other tools cannot express: read attributes,
    count nodes, walk the DOM. The expression must evaluate to a JSON-serializable
    value, e.g. "document.querySelectorAll('img').length".
    """
    started = time.monotonic()
    async with _page() as page:
        final_url = await _goto(page, url)
        value = await page.evaluate(expression)
    return {
        "url": final_url,
        "expression": expression,
        "value": value,
        "elapsed_ms": _elapsed_ms(started),
        "engine": _engine_label(),
    }


async def _read_one(url: str, max_chars: int, limiter: asyncio.Semaphore) -> dict:
    started = time.monotonic()
    async with limiter:
        try:
            async with _page() as page:
                final_url = await _goto(page, url)
                title = await page.title()
                body = await page.inner_text("body")
        except Exception as exc:  # one dead link must not sink the batch
            return {"url": url, "error": f"{type(exc).__name__}: {exc}", "elapsed_ms": _elapsed_ms(started)}
    text = _squeeze(body)
    return {
        "url": final_url,
        "title": title,
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
        "elapsed_ms": _elapsed_ms(started),
    }


@browser_env.task
async def read_pages(urls: list[str], max_chars: int = 1200) -> dict:
    """Read several URLs concurrently and return a short excerpt of each.

    Much faster than calling read_page in a loop — use it whenever you have more
    than one page to look at. Pages that fail come back with an "error" key
    instead of text, so the rest of the batch still lands.
    """
    started = time.monotonic()
    limiter = asyncio.Semaphore(MAX_PARALLEL_PAGES)
    pages = await asyncio.gather(*[_read_one(u, max_chars, limiter) for u in urls[:20]])
    return {
        "pages": pages,
        "ok": sum(1 for p in pages if "error" not in p),
        "failed": sum(1 for p in pages if "error" in p),
        # Wall-clock for the whole batch — compare against the sum of the
        # per-page timings to see what the parallel connections bought you.
        "elapsed_ms": _elapsed_ms(started),
        "engine": _engine_label(),
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

INSTRUCTIONS = """\
You are a web research assistant driving a real headless browser (Lightpanda).

- read_page loads a URL and returns its text; extract_links finds where to go
  next; extract_elements pulls specific nodes by CSS selector; read_pages reads
  many URLs at once — prefer it over repeated read_page calls.
- Start from the URL the user gives you. If you need a page you have not been
  given, get there by following links you actually found, not by guessing URLs.
- When a CSS selector returns nothing, do not keep guessing: read the page or
  use evaluate_on_page to inspect the real markup, then select against that.
- The browser has no renderer, so screenshots and PDFs are not available. Say so
  rather than pretending.
- Quote what the page says and cite the URL you read it from. If a page came
  back empty or errored, report that instead of filling the gap from memory.
"""

agent = Agent(
    name="lightpanda-browser-agent",
    instructions=INSTRUCTIONS,
    model="claude-haiku-4-5",
    tools=[read_page, extract_links, extract_elements, evaluate_on_page, read_pages],
    max_turns=12,
)


@browser_env.task(report=True)
async def browse_entrypoint(message: str, memory: list[dict[str, Any]]) -> dict[str, Any]:
    """Parent task owning the agent loop, so each browser tool call is a durable sub-action."""
    result = await agent.run.aio(message, memory=memory)
    return {
        "summary": result.summary,
        "error": result.error,
        "attempts": result.attempts,
        "charts": [],
        "code": "",
    }


env = AgentChatAppEnvironment(
    name="lightpanda-browser-agent-ui",
    agent=agent,
    task_entrypoint=browse_entrypoint,
    title="Browser-use agent",
    subtitle="Lightpanda headless browser driven over CDP from durable Flyte tasks.",
    theme=CustomTheme(accent_color="#22C55E", accent_hover_color="#4ADE80", button_text_color="#0a0a0f"),
    passthrough_auth=True,
    prompt_nudges=[
        {"label": "Read a page", "prompt": "Read https://lightpanda.io and tell me what the project claims to do."},
        {
            "label": "Follow links",
            "prompt": "Starting at https://lightpanda.io, find the docs links and summarize what the docs cover.",
        },
        {
            "label": "Parallel crawl",
            "prompt": (
                "Read https://example.com, https://lightpanda.io and https://www.union.ai in one batch "
                "and compare how each site describes itself. Include the timing you observed."
            ),
        },
        {
            "label": "Scrape a table",
            "prompt": "Use a CSS selector to pull the headlines from https://news.ycombinator.com and list the top 10.",
        },
        {
            "label": "Check JS rendering",
            "prompt": (
                "Evaluate document.title and the number of images on https://lightpanda.io, "
                "then confirm whether the page needed JavaScript to render."
            ),
        },
    ],
    depends_on=[browser_env],
    image=(flyte.Image.from_debian_base().with_pip_packages("litellm", "fastapi", "uvicorn")),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=flyte.Secret(ANTHROPIC_SECRET, as_env_var="ANTHROPIC_API_KEY"),
)


# ---------------------------------------------------------------------------
# Local self-check — exercises the browser plumbing without a Flyte cluster
# ---------------------------------------------------------------------------


async def _selfcheck(url: str) -> None:
    print(f"engine: {_engine_label()}  binary: {LIGHTPANDA_BIN}")

    page = await read_page.func(url=url, max_chars=300)
    print(f"\nread_page          {page['elapsed_ms']:>5} ms  {page['title']!r}")
    print(f"  {page['text'][:160]!r}")

    links = await extract_links.func(url=url, limit=5)
    print(f"\nextract_links      {links['elapsed_ms']:>5} ms  {links['count']} links")
    for link in links["links"][:5]:
        print(f"  {link['href']}")

    heads = await extract_elements.func(url=url, selector="h1, h2", limit=5)
    print(f"\nextract_elements   {heads['elapsed_ms']:>5} ms  {heads['items']}")

    js = await evaluate_on_page.func(url=url, expression="document.querySelectorAll('a').length")
    print(f"\nevaluate_on_page   {js['elapsed_ms']:>5} ms  {js['value']} anchors")

    batch = await read_pages.func(urls=[url, "https://example.com", "https://lightpanda.io"], max_chars=120)
    serial_ms = sum(p["elapsed_ms"] for p in batch["pages"])
    print(f"\nread_pages         {batch['elapsed_ms']:>5} ms wall ({serial_ms} ms if run serially)")
    for entry in batch["pages"]:
        print(f"  {entry['elapsed_ms']:>5} ms  {entry.get('title') or entry.get('error')}")

    if _Session.driver is not None:
        await _Session.driver.stop()
    if _Session.proc is not None and _Session.proc.returncode is None:
        _Session.proc.terminate()
        await _Session.proc.wait()


if __name__ == "__main__":
    if "--check" in sys.argv:
        args = [a for a in sys.argv[1:] if not a.startswith("-")]
        asyncio.run(_selfcheck(args[0] if args else "https://lightpanda.io"))
    else:
        import pathlib

        flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
        deployments = flyte.deploy(env)
        print(f"Browser-use agent UI: {deployments[0].summary_repr()}")
