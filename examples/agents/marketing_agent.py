"""Marketing agent example — blog and social content (stub tools).

Demonstrates a CodeModeAgent that writes, edits, and reviews marketing copy
using hard-coded tool responses. Run::

    python examples/agents/marketing_agent.py
"""

from __future__ import annotations

import pathlib
from typing import Callable, Literal, TypedDict

import flyte
from flyte.ai import AgentChatAppEnvironment, CodeModeAgent, CustomTheme


class FormattedResponse(TypedDict):
    summary: str


_STUB_BRAND = {
    "voice": "Clear, confident, and human. Avoid jargon unless the audience is technical.",
    "tone_pillars": ["Trustworthy", "Innovative", "Approachable"],
    "words_to_avoid": ["revolutionary", "synergy", "world-class", "leverage (as verb)"],
    "cta_style": "One primary CTA per piece; specific verb + outcome.",
}

_STUB_BLOG_TYPES = {
    "thought_leadership": {
        "structure": ["Hook", "Problem", "Insight", "Proof", "CTA"],
        "target_length_words": "900-1400",
        "seo_notes": "One primary keyword in H1; related terms in H2s naturally.",
    },
    "product_update": {
        "structure": ["What shipped", "Why it matters", "How to try it", "What's next"],
        "target_length_words": "400-700",
        "seo_notes": "Link to docs/changelog; include release version.",
    },
    "customer_story": {
        "structure": ["Customer context", "Challenge", "Solution", "Results", "Quote", "CTA"],
        "target_length_words": "1000-1500",
        "seo_notes": "Use customer-approved metrics only.",
    },
}

_STUB_SOCIAL = {
    "linkedin": {"max_chars": 3000, "hashtags": 3, "tone": "Professional, story-first"},
    "x": {"max_chars": 280, "hashtags": 2, "tone": "Punchy, one idea per post"},
    "instagram_caption": {"max_chars": 2200, "hashtags": 8, "tone": "Visual-first, short lines"},
}


async def get_brand_voice_guidelines() -> dict[str, object]:
    """Return brand voice and editorial guardrails (stub)."""
    return {"brand": _STUB_BRAND}


async def get_blog_outline(
    post_type: Literal["thought_leadership", "product_update", "customer_story"],
) -> dict[str, object]:
    """Return a recommended outline for a blog post type (stub)."""
    return {post_type: _STUB_BLOG_TYPES.get(post_type, _STUB_BLOG_TYPES["thought_leadership"])}


async def get_social_channel_guide(
    channel: Literal["linkedin", "x", "instagram_caption"],
) -> dict[str, object]:
    """Return channel-specific constraints and tone (stub)."""
    return {channel: _STUB_SOCIAL.get(channel, _STUB_SOCIAL["linkedin"])}


async def get_editorial_checklist() -> list[str]:
    """Return a copy-editing checklist (stub)."""
    return [
        "Headline promises match body content",
        "Active voice in ≥80% of sentences",
        "Sources cited for statistics",
        "Inclusive language scan",
        "Accessibility: alt text for images",
        "Legal/compliance disclaimer if needed",
    ]


async def get_campaign_metrics_stub(
    campaign_id: str,
) -> dict[str, object]:
    """Return fake campaign performance for demos (stub; ignores campaign_id)."""
    return {
        "campaign_id": campaign_id or "demo-campaign",
        "impressions": 128_400,
        "clicks": 3_842,
        "ctr_pct": 2.99,
        "note": "Stub data for UI demo only.",
    }


async def format_response(
    title: str,
    body: str,
    links: list[str] | None = None,
) -> FormattedResponse:
    """Format the final assistant message as Markdown for the chat UI."""
    parts = [f"## {title}", "", body]
    if links:
        parts += ["", "**References:**", *[f"- {u}" for u in links]]
    return {"summary": "\n".join(parts)}


ALL_TOOLS: dict[str, Callable] = {
    "get_brand_voice_guidelines": get_brand_voice_guidelines,
    "get_blog_outline": get_blog_outline,
    "get_social_channel_guide": get_social_channel_guide,
    "get_editorial_checklist": get_editorial_checklist,
    "get_campaign_metrics_stub": get_campaign_metrics_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a marketing copy assistant. Use the tools for brand rules, outlines, \
and channel constraints — they return stub/demo data. Produce concrete drafts \
or edits in Markdown inside format_response. When revising copy, cite which \
checklist items you addressed. Always end by calling format_response(title, body, links).
"""

agent = CodeModeAgent(
    tools=ALL_TOOLS,
    model="claude-sonnet-4-6",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)

env = AgentChatAppEnvironment(
    name="marketing-agent-demo",
    agent=agent,
    title="Marketing agent (demo)",
    subtitle="Blog and social drafts with stub brand/channel data.",
    theme=CustomTheme(accent_color="#EC4899", accent_hover_color="#F472B6", button_text_color="#0a0a0f"),
    prompt_nudges=[
        {
            "label": "Blog draft",
            "prompt": "Write a short thought leadership blog intro about responsible AI in healthcare.",
        },
        {
            "label": "LinkedIn thread",
            "prompt": "Draft a 4-post LinkedIn thread announcing a product update for data engineers.",
        },
        {
            "label": "Edit pass",
            "prompt": "Review this copy for voice and checklist: We are revolutionary synergy platform.",
        },
    ],
    image=flyte.Image.from_debian_base(install_flyte=False)
    .with_pip_packages("litellm", "pydantic-monty==0.0.8", "uvicorn", "fastapi", "flyte[sandbox]")
    .with_local_v2(),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    handle = flyte.serve(env)
    print(f"Marketing agent: {handle.url}")
