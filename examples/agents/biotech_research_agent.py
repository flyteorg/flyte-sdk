"""Biotech / omics research agent example — multi-omics, pathways, and plans (stub tools).

Reflects common computational biology steps: QC, differential signal, pathway
enrichment, target tractability, and translational planning — all **synthetic**.
Not for clinical, regulatory, or wet-lab decisions.

Run::

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

_STUB_QC_METRICS = {
    "rin_median_stub": 8.2,
    "pct_mapped_reads_median_stub": 92.4,
    "library_complexity_stub": "acceptable",
    "note": "Synthetic QC panel — replace with pipeline outputs.",
}

_STUB_DEG = [
    {"gene": "STAT1", "log2fc_stub": 2.1, "adj_p_stub": 1e-6},
    {"gene": "CDK1", "log2fc_stub": 1.4, "adj_p_stub": 2e-5},
    {"gene": "TGFB1", "log2fc_stub": -1.1, "adj_p_stub": 4e-4},
]

_STUB_PATHWAY_GSEA = [
    {"pathway": "Interferon alpha response", "nes_stub": 1.9, "fdr_stub": 0.02},
    {"pathway": "E2F targets", "nes_stub": 1.6, "fdr_stub": 0.04},
    {"pathway": "TNF signaling", "nes_stub": -1.5, "fdr_stub": 0.08},
]

_STUB_CLUSTERS = [
    {"cluster_id": "C0", "label_stub": "Cycling T cells", "top_genes_stub": ["MKI67", "TOP2A"]},
    {"cluster_id": "C1", "label_stub": "Exhausted-like", "top_genes_stub": ["PDCD1", "LAG3"]},
    {"cluster_id": "C2", "label_stub": "NK-like", "top_genes_stub": ["NKG7", "GNLY"]},
]

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

_STUB_TARGETS = [
    {"gene": "X", "tractability_stub": "high", "modality_fit_stub": ["small_molecule", "PROTAC"]},
    {"gene": "Y", "tractability_stub": "medium", "modality_fit_stub": ["antibody", "ADC"]},
]


async def summarize_omics_dataset_stub(dataset_id: str) -> dict[str, object]:
    """Return a synthetic omics cohort summary (stub)."""
    return {"dataset_id": dataset_id or "DEMO-COHORT-001", "summary": _STUB_COHORT}


async def get_sample_qc_metrics_stub() -> dict[str, object]:
    """Return synthetic bulk RNA-seq style QC aggregates (stub)."""
    return dict(_STUB_QC_METRICS)


async def get_differential_expression_top_stub(
    contrast: str,
) -> list[dict[str, object]]:
    """Return top DE genes for a named contrast (stub; contrast not parsed)."""
    return list(_STUB_DEG)


async def get_pathway_enrichment_stub() -> list[dict[str, object]]:
    """Return GSEA-style pathway rows (stub)."""
    return list(_STUB_PATHWAY_GSEA)


async def get_single_cell_clusters_stub() -> list[dict[str, object]]:
    """Return illustrative cluster summaries (stub)."""
    return list(_STUB_CLUSTERS)


async def get_drug_target_tractability_stub() -> list[dict[str, object]]:
    """Return synthetic tractability labels (stub)."""
    return list(_STUB_TARGETS)


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


async def get_cro_handoff_checklist_stub() -> list[str]:
    """Return items to package for a CRO or collaborator (stub)."""
    return [
        "Analysis-ready counts matrix + colData with batch labels",
        "Frozen analysis parameters (DESeq2 / similar) and version pins",
        "Primary and exploratory endpoints pre-specified",
        "Data transfer agreement and sample manifest",
        "Expected deliverables and timeline with QC gates",
    ]


async def get_literature_context_stub(topic: str) -> dict[str, object]:
    """Return placeholder literature search seeds (stub, not real papers)."""
    return {
        "topic": topic or "general",
        "suggested_queries_stub": [
            f"{topic} AND (single-cell OR spatial)",
            f"{topic} AND (clinical trial OR biomarker)",
            f"{topic} AND (GSEA OR pathway)",
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
    "get_sample_qc_metrics_stub": get_sample_qc_metrics_stub,
    "get_differential_expression_top_stub": get_differential_expression_top_stub,
    "get_pathway_enrichment_stub": get_pathway_enrichment_stub,
    "get_single_cell_clusters_stub": get_single_cell_clusters_stub,
    "get_drug_target_tractability_stub": get_drug_target_tractability_stub,
    "suggest_experimental_design_stub": suggest_experimental_design_stub,
    "rank_hypotheses_stub": rank_hypotheses_stub,
    "get_cro_handoff_checklist_stub": get_cro_handoff_checklist_stub,
    "get_literature_context_stub": get_literature_context_stub,
    "regulatory_ethics_checklist_stub": regulatory_ethics_checklist_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a computational biology / translational research strategist. All omics \
and pathway numbers are **synthetic**. Never present outputs as patient \
results or regulatory evidence. Connect QC → DE → pathways → hypotheses → \
experiments and CRO handoff. Always end with format_response(title, body, links).
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
    subtitle="Multi-omics, pathways, and study design — synthetic data only.",
    theme=CustomTheme(accent_color="#8B5CF6", accent_hover_color="#A78BFA", button_text_color="#0a0a0f"),
    prompt_nudges=[
        {"label": "Cohort stand-up", "prompt": "Summarize cohort + QC for a Monday team sync."},
        {
            "label": "Pathway story",
            "prompt": "Chain DE genes into pathway enrichment and a biological narrative with caveats.",
        },
        {
            "label": "Single-cell angle",
            "prompt": "Relate single_cell clusters to DE and hypotheses for a joint lab meeting.",
        },
        {
            "label": "Target review",
            "prompt": "Discuss tractability table and what wet-lab validation you would order first.",
        },
        {
            "label": "Biomarker plan",
            "prompt": "Use biomarker_discovery experimental design plus ethics checklist for a translational outline.",
        },
        {
            "label": "CRO package",
            "prompt": "What would you send a CRO given the handoff checklist and stub cohort metadata?",
        },
        {
            "label": "Grant aims",
            "prompt": "Draft one-page specific aims language (hypothesis-driven) grounded in stub pathway themes.",
        },
        {"label": "Literature seeds", "prompt": "Turn literature_context_stub into a concrete PubMed search plan."},
        {
            "label": "Mechanism program",
            "prompt": "Propose a mechanism goal experimental matrix using time course + perturbation arms.",
        },
        {"label": "Risk & ethics", "prompt": "List top scientific risks and tie each to ethics or QC mitigations."},
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
    print(f"Biotech research agent: {handle.url}")
