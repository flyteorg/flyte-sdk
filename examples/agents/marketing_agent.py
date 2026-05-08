"""Marketing agent example — blog, social, and campaign workflows (stub tools).

Demonstrates editorial calendars, content pillars, personas, workflow stages,
and cross-channel repurposing using **hard-coded** tool data (no live CMS or
analytics). Informed by common content-ops patterns: pillars, personas, KPIs,
and draft → review → scheduled → published handoffs.

Run::

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

_STUB_PILLARS = [
    {
        "pillar": "Product education",
        "goal": "Help buyers evaluate fit faster",
        "content_types": ["how-to blogs", "comparison sheets", "demo clips"],
    },
    {
        "pillar": "Customer proof",
        "goal": "Reduce perceived risk",
        "content_types": ["case studies", "quotes in social", "webinar replays"],
    },
    {
        "pillar": "Thought leadership",
        "goal": "Shape category narrative",
        "content_types": ["data-backed POV posts", "executive bylines", "podcast spots"],
    },
    {
        "pillar": "Community & events",
        "goal": "Deepen practitioner relationships",
        "content_types": ["meetup recaps", "CFP announcements", "AMA threads"],
    },
]

_STUB_PERSONAS = [
    {
        "name": "Platform Dana",
        "role": "Senior data engineer",
        "pains": ["opaque orchestration costs", "lack of lineage for audits"],
        "channels": ["LinkedIn", "technical blog", "conference talks"],
    },
    {
        "name": "Ops Omar",
        "role": "IT / platform owner",
        "pains": ["multi-tenant isolation", "SSO and RBAC"],
        "channels": ["security briefs", "ROI one-pagers", "vendor webinars"],
    },
    {
        "name": "Leadership Lee",
        "role": "VP Engineering",
        "pains": ["time-to-value for ML programs", "vendor lock-in"],
        "channels": ["short LinkedIn posts", "board-ready summaries"],
    },
]

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

_STUB_WORKFLOW = [
    {"status": "idea", "owner": "content", "sla_days": 2},
    {"status": "draft", "owner": "writer", "sla_days": 5},
    {"status": "in_review", "owner": "editor", "sla_days": 3},
    {"status": "approved", "owner": "brand", "sla_days": 2},
    {"status": "scheduled", "owner": "demand_gen", "sla_days": 1},
    {"status": "published", "owner": "analytics", "sla_days": 0},
]

_STUB_CAMPAIGN_BRIEF_FIELDS = [
    "Objective (awareness / consideration / conversion)",
    "Primary audience (persona)",
    "Offer or CTA",
    "Channels & flight dates",
    "Success metrics (e.g. MQLs, CTR, share of voice)",
    "Legal / compliance notes",
    "Localization or regional variants",
]


async def get_brand_voice_guidelines() -> dict[str, object]:
    """Return brand voice and editorial guardrails (stub)."""
    return {"brand": _STUB_BRAND}


async def get_content_pillars_stub() -> list[dict[str, object]]:
    """Return strategic content pillars aligned to goals (stub)."""
    return list(_STUB_PILLARS)


async def get_audience_personas_stub() -> list[dict[str, object]]:
    """Return B2B-style audience personas with pains and channels (stub)."""
    return list(_STUB_PERSONAS)


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


async def get_content_workflow_stages_stub() -> list[dict[str, object]]:
    """Return editorial workflow stages with example SLAs (stub)."""
    return list(_STUB_WORKFLOW)


async def get_campaign_brief_template_stub() -> dict[str, object]:
    """Return fields to fill for a cross-channel campaign brief (stub)."""
    return {"required_sections": list(_STUB_CAMPAIGN_BRIEF_FIELDS)}


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


async def get_campaign_metrics_stub(campaign_id: str) -> dict[str, object]:
    """Return fake campaign performance for demos (stub; ignores campaign_id)."""
    return {
        "campaign_id": campaign_id or "demo-campaign",
        "impressions": 128_400,
        "clicks": 3_842,
        "ctr_pct": 2.99,
        "engagement_rate_pct": 4.1,
        "mqls_attributed_stub": 42,
        "note": "Stub data for UI demo only — not connected to ads or web analytics.",
    }


async def get_ab_test_results_stub(test_name: str) -> dict[str, object]:
    """Return illustrative A/B or multivariate test outcomes (stub)."""
    return {
        "test_name": test_name or "homepage_hero_q2",
        "variant_a_ctr_pct": 2.1,
        "variant_b_ctr_pct": 2.8,
        "winner": "B",
        "confidence_stub": "Illustrative only — replace with real stats engine.",
    }


async def get_competitor_messaging_scan_stub(
    competitor: str,
) -> dict[str, object]:
    """Return canned positioning phrases for a named competitor (stub, not scraped)."""
    library = {
        "acme": {
            "taglines_stub": ["Scale without limits", "AI-first operations"],
            "themes_stub": ["automation", "enterprise SLAs"],
        },
        "globex": {
            "taglines_stub": ["Trust built in", "Compliance by design"],
            "themes_stub": ["security", "regulated industries"],
        },
    }
    key = competitor.lower().strip() or "acme"
    return {"competitor": competitor, "insights_stub": library.get(key, library["acme"])}


async def get_email_nurture_outline_stub(
    sequence_goal: Literal["trial_activation", "event_followup", "re_engagement"],
) -> list[dict[str, str]]:
    """Return a 4-touch email skeleton for common nurture goals (stub)."""
    sequences = {
        "trial_activation": [
            {"day": "0", "subject_angle": "Welcome + single next step", "body_focus": "Time-to-value checklist"},
            {"day": "2", "subject_angle": "Social proof", "body_focus": "Customer quote + metric"},
            {"day": "5", "subject_angle": "Education", "body_focus": "Link to docs / short video"},
            {"day": "9", "subject_angle": "CTA", "body_focus": "Book success call"},
        ],
        "event_followup": [
            {"day": "0", "subject_angle": "Thanks for attending", "body_focus": "Deck + recording"},
            {"day": "3", "subject_angle": "Deep dive", "body_focus": "Technical blog"},
            {"day": "7", "subject_angle": "Peer story", "body_focus": "Case study"},
            {"day": "14", "subject_angle": "Soft CTA", "body_focus": "Office hours invite"},
        ],
        "re_engagement": [
            {"day": "0", "subject_angle": "We miss you", "body_focus": "What's new since last login"},
            {
                "day": "4",
                "subject_angle": "Win back offer",
                "body_focus": "Illustrative incentive — legal review required",
            },
            {"day": "10", "subject_angle": "Feedback", "body_focus": "One-question survey"},
            {"day": "17", "subject_angle": "Sunset or downgrade", "body_focus": "Clear options"},
        ],
    }
    return sequences[sequence_goal]


async def get_repurposing_matrix_stub(
    source_asset: Literal["long_blog", "webinar", "product_release"],
) -> list[dict[str, str]]:
    """Return how to slice one flagship asset into channel-specific pieces (stub)."""
    matrices = {
        "long_blog": [
            {"output": "LinkedIn carousel", "effort": "medium", "hook": "Pull 5 data points from post"},
            {"output": "X thread", "effort": "low", "hook": "One insight per tweet"},
            {"output": "Sales one-pager", "effort": "medium", "hook": "Problem/solution + CTA"},
        ],
        "webinar": [
            {"output": "Blog recap", "effort": "medium", "hook": "Q&A themes"},
            {"output": "Short clips", "effort": "high", "hook": "Timestamp best moments"},
            {"output": "Email series", "effort": "medium", "hook": "Repurpose 3 chapters"},
        ],
        "product_release": [
            {"output": "Changelog post", "effort": "low", "hook": "Ship list + migration notes"},
            {"output": "Demo script", "effort": "medium", "hook": "3 wow moments"},
            {"output": "Partner newsletter blurb", "effort": "low", "hook": "140 words + link"},
        ],
    }
    return matrices[source_asset]


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
    "get_content_pillars_stub": get_content_pillars_stub,
    "get_audience_personas_stub": get_audience_personas_stub,
    "get_blog_outline": get_blog_outline,
    "get_social_channel_guide": get_social_channel_guide,
    "get_content_workflow_stages_stub": get_content_workflow_stages_stub,
    "get_campaign_brief_template_stub": get_campaign_brief_template_stub,
    "get_editorial_checklist": get_editorial_checklist,
    "get_campaign_metrics_stub": get_campaign_metrics_stub,
    "get_ab_test_results_stub": get_ab_test_results_stub,
    "get_competitor_messaging_scan_stub": get_competitor_messaging_scan_stub,
    "get_email_nurture_outline_stub": get_email_nurture_outline_stub,
    "get_repurposing_matrix_stub": get_repurposing_matrix_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a senior marketing content strategist. Tools return **stub** calendars, \
personas, metrics, and competitor blurbs — never imply live data or legal approval. \
Ground recommendations in pillars, personas, and workflow stages. For launches, \
tie copy to campaign brief fields. Always end with format_response(title, body, links).
"""

agent = CodeModeAgent(
    tools=ALL_TOOLS,
    model="claude-haiku-4-5",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)

env = AgentChatAppEnvironment(
    name="marketing-agent-demo",
    agent=agent,
    title="Marketing agent (demo)",
    subtitle="Editorial workflows, pillars, and cross-channel stubs — not live analytics.",
    theme=CustomTheme(accent_color="#EC4899", accent_hover_color="#F472B6", button_text_color="#0a0a0f"),
    prompt_nudges=[
        {
            "label": "Pillar strategy",
            "prompt": "Using the stub pillars and personas, propose a 2-week content mix for LinkedIn + blog.",
        },
        {
            "label": "Blog + repurposing",
            "prompt": "Outline a thought leadership blog and show how to repurpose it using the repurposing matrix.",
        },
        {
            "label": "Product launch",
            "prompt": (
                "Draft a campaign brief outline for a minor product release "
                "using the brief template and workflow stages."
            ),
        },
        {
            "label": "Nurture series",
            "prompt": (
                "Turn the trial_activation email stub into concrete subject lines and 2-sentence bodies per touch."
            ),
        },
        {
            "label": "Edit harsh copy",
            "prompt": "Rewrite: 'We are a world-class revolutionary synergy AI OS.' Use brand voice and checklist.",
        },
        {
            "label": "A/B readout",
            "prompt": "Explain the stub A/B test results for stakeholders and what to try next.",
        },
        {
            "label": "Competitor POV",
            "prompt": "Compare our positioning vs stub Acme messaging without making unverifiable claims.",
        },
        {
            "label": "Webinar follow-up",
            "prompt": "Plan post-webinar emails and two social posts using personas and channel guides.",
        },
        {
            "label": "B2B vs exec",
            "prompt": "Same product update: adapt messaging for Platform Dana vs Leadership Lee using personas.",
        },
        {
            "label": "Crisis holding line",
            "prompt": (
                "Draft a cautious external holding statement template (no admission of fault) "
                "for a hypothetical outage."
            ),
        },
    ],
    image=flyte.Image.from_debian_base(install_flyte=False)
    .with_pip_packages("litellm", "pydantic-monty==0.0.17", "uvicorn", "fastapi", "flyte[sandbox]")
    .with_local_v2(),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    handle = flyte.serve(env)
    print(f"Marketing agent: {handle.url}")
