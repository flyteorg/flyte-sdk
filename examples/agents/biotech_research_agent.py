"""Biotech / omics research agent example — hypotheses and plans (stub tools).

Returns synthetic omics-style summaries and research planning text for demos.
Not for clinical or lab decision-making. Run::

    python examples/agents/biotech_research_agent.py
"""

from __future__ import annotations

import pathlib
from typing import Callable, Literal, TypedDict

import flyte
from flyte.ai import AgentChatAppEnvironment, CodeModeAgent, CustomTheme


class FormattedResponse(TypedDict):
    summary: str


_STUB_COHORT = {
    "n_samples": 48,
    "modality": "RNA-seq + phosphoproteomics (synthetic labels)",
    "top_pathways_stub": ["Interferon response", "Cell cycle G2/M", "MAPK signaling"],
    "qc_flags_stub": ["3 samples high ribosomal fraction", "1 batch effect cohort B"],
}

_STUB_HYPOTHESES = [
    {
        "id": "H1",
        "title": "Target X drives resistance via pathway P",
        "testable_prediction": "CRISPR knockdown of X restores drug sensitivity in vitro",
        "priority": "high",
    },
    {
        "id": "H2",
        "title": "Biomarker panel M stratifies responders",
        "testable_prediction": "Prospective validation in n≥30 independent biobank samples",
        "priority": "medium",
    },
    {
        "id": "H3",
        "title": "Microenvironment cytokine C modulates T cell exhaustion",
        "testable_prediction": "Cytokine blockade shifts scRNA-seq exhaustion signature",
        "priority": "medium",
    },
]


async def summarize_omics_dataset_stub(
    dataset_id: str,
) -> dict[str, object]:
    """Return a synthetic omics cohort summary (stub)."""
    return {"dataset_id": dataset_id or "DEMO-COHORT-001", "summary": _STUB_COHORT}


async def suggest_experimental_design_stub(
    goal: Literal["target_validation", "biomarker_discovery", "mechanism"],
) -> dict[str, object]:
    """Return a template experimental design outline (stub)."""
    designs = {
        "target_validation": {
            "arms": ["WT cell line", "KO cell line", "+ rescue"],
            "readouts": ["viability IC50", "Western for pathway nodes", "RNA-seq n=3 each"],
            "timeline_weeks_stub": 14,
        },
        "biomarker_discovery": {
            "arms": ["Responder", "Non-responder archival"],
            "readouts": ["Differential expression", "Phospho enrichment", "External validation set"],
            "timeline_weeks_stub": 20,
        },
        "mechanism": {
            "arms": ["Stimulation time course", "Genetic perturbation matrix"],
            "readouts": ["Multi-omics integration", "Reporter assays"],
            "timeline_weeks_stub": 18,
        },
    }
    return {"goal": goal, "design": designs[goal]}


async def rank_hypotheses_stub() -> list[dict[str, str]]:
    """Return a fixed ranked hypothesis list (stub)."""
    return list(_STUB_HYPOTHESES)


async def get_literature_context_stub(
    topic: str,
) -> dict[str, object]:
    """Return placeholder 'literature' pointers (stub, not real papers)."""
    return {
        "topic": topic or "general",
        "suggested_queries_stub": [
            f"{topic} AND (single-cell OR spatial)",
            f"{topic} AND (clinical trial OR biomarker)",
        ],
        "note": "Replace with PubMed / preprint search in production.",
    }


async def regulatory_ethics_checklist_stub() -> list[str]:
    """Return generic R&D ethics reminders (stub)."""
    return [
        "IRB / ethics approval for human samples",
        "Data use agreements and patient consent scope",
        "BL2+ requirements if handling primary cells",
        "Pre-registration for confirmatory analyses where applicable",
    ]


async def format_response(
    title: str,
    body: str,
    links: list[str] | None = None,
) -> FormattedResponse:
    """Final Markdown for the chat UI."""
    parts = [f"## {title}", "", body]
    if links:
        parts += ["", "**References:**", *[f"- {u}" for u in links]]
    return {"summary": "\n".join(parts)}


ALL_TOOLS: dict[str, Callable] = {
    "summarize_omics_dataset_stub": summarize_omics_dataset_stub,
    "suggest_experimental_design_stub": suggest_experimental_design_stub,
    "rank_hypotheses_stub": rank_hypotheses_stub,
    "get_literature_context_stub": get_literature_context_stub,
    "regulatory_ethics_checklist_stub": regulatory_ethics_checklist_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a biotech research strategist assisting with **demo / educational** \
omics workflows. Tool outputs are synthetic — never present them as real patient \
or lab results. Tie hypotheses to suggest_experimental_design_stub and ethics \
checklist items. Always end with format_response(title, body, links).
"""

agent = CodeModeAgent(
    tools=ALL_TOOLS,
    model="claude-sonnet-4-6",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)

env = AgentChatAppEnvironment(
    name="biotech-research-agent-demo",
    agent=agent,
    title="Biotech research agent (demo)",
    subtitle="Synthetic omics summaries, designs, and ranked hypotheses.",
    theme=CustomTheme(accent_color="#8B5CF6", accent_hover_color="#A78BFA", button_text_color="#0a0a0f"),
    prompt_nudges=[
        {"label": "Cohort summary", "prompt": "Summarize the stub omics cohort and top pathways for a team meeting."},
        {"label": "Next experiments", "prompt": "Propose a target-validation experimental plan using the stub tools."},
        {"label": "Hypotheses", "prompt": "Prioritize the stub hypotheses and suggest validation readouts."},
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
    print(f"Biotech research agent: {handle.url}")
