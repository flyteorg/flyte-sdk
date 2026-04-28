"""Union/Flyte Sales Agent — AI orchestration pre-sales assistant
================================================================

A chat agent that answers questions about Union and Flyte, positioned
for prospects evaluating AI/ML orchestration solutions. It can look up
official documentation, website content, and GitHub repositories, and
can make informed comparisons against competitors (Ray, Prefect, Dagster,
Kubeflow, Modal, Airflow).

The agent is optimized for the **top of the sales funnel**: engineers and
technical leaders researching orchestration platforms. It can also handle
mid-funnel (evaluation/POC) and bottom-funnel (procurement) questions.

Architecture::

    Browser (FastAPI Chat UI)
      └── AgentChatAppEnvironment
            └── CodeModeAgent.run(message, history)
                  ├── LLM call (generate code using tool functions)
                  ├── Monty sandbox execution
                  └── retry on failure

Install dependencies::

    pip install 'flyte[sandbox]' litellm

Run::

    python examples/agents/sales_agent.py
"""

from __future__ import annotations

import pathlib
from typing import Callable, Literal, TypedDict

import flyte
from flyte.ai import AgentChatAppEnvironment, CodeModeAgent, CustomTheme

# ---------------------------------------------------------------------------
# TypedDict return types — visible in inspect.signature so the LLM knows
# the exact shape of data returned by each tool.
# ---------------------------------------------------------------------------


class OverviewInfo(TypedDict, total=False):
    name: str
    tagline: str
    url: str
    website_url: str
    docs_url: str
    description: str
    origin: str
    status: str
    github_stars: str
    license: str


class CompetitorProfile(TypedDict):
    name: str
    category: str
    strengths: list[str]
    weaknesses_vs_flyte: list[str]
    when_competitor_wins: list[str]


class CompetitorSummary(TypedDict):
    key: str
    name: str
    category: str


class WhyFlyteDimension(TypedDict):
    dimension: str
    summary: str
    key_points: list[str]


class WhyFlyteSummary(TypedDict):
    key: str
    dimension: str
    summary: str


class FunnelStage(TypedDict):
    stage: str
    description: str
    tone: str
    focus_areas: list[str]
    sample_questions: list[str]


class LinkGroup(TypedDict, total=False):
    union: str
    flyte: str
    union_docs: str
    flyte_docs: str
    flyte_core: str
    flyte_sdk_v2: str
    flytekit_v1: str


class FormattedResponse(TypedDict):
    summary: str


# ---------------------------------------------------------------------------
# Knowledge bases — structured data the agent's tools can look up
# ---------------------------------------------------------------------------

_UNION_OVERVIEW = {
    "overview": {
        "name": "Union",
        "tagline": "The AI orchestration platform built on Flyte",
        "url": "https://www.union.ai/",
        "docs_url": "https://www.union.ai/docs/v2/union/user-guide/",
        "description": (
            "Union is an enterprise-grade managed platform built on top of "
            "Apache Flyte. It provides a fully managed control plane, "
            "multi-tenant isolation, a hosted data plane, RBAC, SSO, audit "
            "logging, and premium support. Union adds features on top of OSS "
            "Flyte including Union Actors (long-running stateful workers), "
            "Union Serverless (pay-per-use ephemeral compute), blazing-fast "
            "image building, a modern SDK (flyte v2), and a polished UI."
        ),
    },
    "features": [
        "Fully managed Flyte — zero ops overhead",
        "On-prem / VPC / hybrid deployment options",
        "SOC 2 Type II, HIPAA-eligible, FedRAMP (in progress)",
        "Multi-cluster federation across clouds (AWS, GCP, Azure)",
        "Union Actors — persistent GPU workers for interactive & long-running jobs",
        "Union Serverless — pay-per-second ephemeral compute",
        "Sub-second image builds with smart caching",
        "Native Kubernetes integration",
        "Built-in data lineage and artifact tracking",
        "Enterprise RBAC, SSO (OIDC/SAML), audit logs",
    ],
    "ideal_for": [
        "Teams running ML/AI workloads at scale",
        "Organizations needing on-prem or hybrid deployment",
        "Enterprises with strict security and compliance requirements",
        "Teams wanting managed Flyte without DevOps burden",
        "Companies orchestrating GPU-heavy training and inference pipelines",
    ],
}

_FLYTE_OVERVIEW = {
    "overview": {
        "name": "Flyte",
        "tagline": "The open-source AI & data orchestration platform",
        "website_url": "https://flyte.org/",
        "docs_url": "https://www.union.ai/docs/v2/flyte/user-guide/",
        "description": (
            "Flyte is an open-source, Kubernetes-native workflow orchestration "
            "platform originally created at Lyft. It is designed for ML, data "
            "engineering, and AI workloads. Flyte provides strong typing, "
            "immutable and versioned workflows, automatic data lineage, "
            "caching, intra-task checkpointing, and multi-tenant isolation — "
            "all running on Kubernetes."
        ),
        "origin": "Lyft",
        "status": "CNCF graduated project",
        "github_stars": "6000+",
        "license": "Apache 2.0",
    },
    "features": [
        "Kubernetes-native — runs anywhere K8s runs",
        "Strongly typed tasks and workflows with automatic ser/de",
        "Immutable, versioned, and reproducible executions",
        "First-class caching with cache keys and cache versions",
        "Intra-task checkpointing for long-running jobs",
        "Dynamic workflows and map tasks for data parallelism",
        "Multi-tenant with project/domain isolation",
        "Rich plugin ecosystem (Spark, Ray, Dask, MPI, Sagemaker, etc.)",
        "Language-agnostic (Python, Java, Go SDKs)",
        "Built-in data catalog and artifact tracking",
        "Declarative infrastructure via Kubernetes CRDs",
    ],
    "adoption": [
        "Lyft (creator)",
        "Spotify",
        "Freenome",
        "Woven by Toyota",
        "Recogni",
        "LatchBio",
        "Pachama",
        "Striveworks",
    ],
    "repos": {
        "flyte": "https://github.com/flyteorg/flyte",
        "flyte-sdk-v2": "https://github.com/flyteorg/flyte-sdk",
        "flytekit-v1": "https://github.com/flyteorg/flytekit",
    },
}

_COMPETITOR_PROFILES: dict[str, dict] = {
    "airflow": {
        "name": "Apache Airflow",
        "category": "Workflow orchestration",
        "strengths": [
            "Massive community and ecosystem",
            "Huge library of pre-built operators/hooks",
            "Well-understood DAG model",
            "Battle-tested for ETL and data engineering",
        ],
        "weaknesses_vs_flyte": [
            "No native strong typing — data passed via XCom (JSON blobs)",
            "DAGs are not immutable or versioned — hard to reproduce past runs",
            "No built-in data lineage or artifact catalog",
            "Limited support for ML-specific patterns (map tasks, GPU scheduling)",
            "No intra-task checkpointing",
            "Scheduler-centric — not designed for compute-heavy AI workloads",
            "Scaling the scheduler is operationally complex",
            "No native Kubernetes task execution (requires KubernetesPodOperator workarounds)",
        ],
        "when_competitor_wins": [
            "Existing Airflow investment with purely ETL workloads",
            "Need for very specific pre-built operator (e.g. legacy Hadoop)",
            "Team already deeply skilled in Airflow",
        ],
    },
    "prefect": {
        "name": "Prefect",
        "category": "Workflow orchestration",
        "strengths": [
            "Pythonic, decorator-based API — low learning curve",
            "Good local development experience",
            "Prefect Cloud is easy to get started with",
            "Nice observability UI",
        ],
        "weaknesses_vs_flyte": [
            "No strong typing between tasks — relies on Python duck typing",
            "No immutable/versioned workflow definitions",
            "Not Kubernetes-native — requires separate infrastructure for scaling",
            "No built-in data catalog or lineage",
            "Limited multi-tenancy (no project/domain isolation)",
            "No intra-task checkpointing",
            "Weaker support for heterogeneous compute (GPU, Spark, etc.)",
            "No on-prem enterprise deployment story as mature as Union",
        ],
        "when_competitor_wins": [
            "Small teams wanting quick setup without K8s",
            "Primarily Python-centric data pipelines, not heavy ML",
            "Teams that prefer Prefect's event-driven triggers model",
        ],
    },
    "dagster": {
        "name": "Dagster",
        "category": "Data orchestration / asset-based",
        "strengths": [
            "Software-defined assets model — good for data mesh",
            "Strong local dev and testing experience",
            "Good type system for IO managers",
            "Nice integration with dbt",
        ],
        "weaknesses_vs_flyte": [
            "Asset-centric model less natural for ML training pipelines",
            "Not Kubernetes-native — Dagster Cloud uses its own infra",
            "Limited GPU/HPC scheduling capabilities",
            "No built-in map tasks or dynamic fan-out at Flyte's scale",
            "Weaker multi-cloud and hybrid deployment story",
            "Less mature enterprise features (RBAC, audit, compliance)",
            "Smaller adoption in ML/AI-heavy organizations",
        ],
        "when_competitor_wins": [
            "Data engineering teams using dbt and asset-centric thinking",
            "Organizations focused on data quality and observability over ML",
            "Teams that prefer the software-defined assets paradigm",
        ],
    },
    "ray": {
        "name": "Ray / Anyscale",
        "category": "Distributed computing framework",
        "strengths": [
            "Best-in-class distributed computing primitives",
            "Ray Train / Ray Serve for ML training and serving",
            "Great for distributed model training (PyTorch, etc.)",
            "Anyscale provides managed Ray",
        ],
        "weaknesses_vs_flyte": [
            "Ray is a compute framework, not a workflow orchestrator — no DAGs, no versioning",
            "No built-in workflow scheduling, retries, or caching",
            "No data lineage or artifact management",
            "Requires separate orchestration layer for production pipelines",
            "Cluster management is complex without Anyscale",
            "No multi-tenant isolation (project/domain model)",
            "Flyte has a native Ray plugin — you can use Ray *inside* Flyte tasks",
        ],
        "when_competitor_wins": [
            "Pure distributed computing / training workloads with no orchestration needs",
            "Teams building custom distributed applications",
            "When the primary need is Ray Serve for model serving",
        ],
    },
    "kubeflow": {
        "name": "Kubeflow Pipelines",
        "category": "ML orchestration on Kubernetes",
        "strengths": [
            "Kubernetes-native like Flyte",
            "Good integration with TensorFlow ecosystem",
            "Part of the broader Kubeflow ML platform",
            "Backed by Google",
        ],
        "weaknesses_vs_flyte": [
            "Pipeline definitions use YAML/IR — less Pythonic developer experience",
            "Weaker type system — relies on artifact passing, not strongly typed",
            "More complex operational setup and maintenance",
            "Less active open-source community (fewer contributors, slower release cadence)",
            "No equivalent of Flyte's dynamic workflows or map tasks",
            "No intra-task checkpointing",
            "Union/Flyte has a cleaner multi-tenant model",
            "Kubeflow Pipelines v2 is still maturing, fragmented from v1",
        ],
        "when_competitor_wins": [
            "Organizations already invested in the full Kubeflow ecosystem (KFServing, Katib, etc.)",
            "Teams using Vertex AI Pipelines (Google Cloud managed Kubeflow)",
            "Heavy TensorFlow shops wanting tight ecosystem integration",
        ],
    },
    "modal": {
        "name": "Modal",
        "category": "Serverless compute for AI",
        "strengths": [
            "Exceptional developer experience — very fast iteration",
            "Instant cold starts and sub-second container provisioning",
            "Great for prototyping and small-to-mid scale GPU workloads",
            "Simple pricing model",
            "Good for serving / inference workloads",
        ],
        "weaknesses_vs_flyte": [
            "No on-prem or VPC deployment — SaaS-only",
            "No workflow DAG orchestration — it's a compute platform, not an orchestrator",
            "No data lineage, artifact catalog, or versioned workflow definitions",
            "Limited enterprise features (RBAC, SSO, audit logging, compliance certs)",
            "Vendor lock-in — no open-source core, no self-hosting option",
            "Not suitable for organizations with data sovereignty requirements",
            "No multi-tenant project/domain isolation model",
            "Limited integration with broader data ecosystem (Spark, Hive, etc.)",
        ],
        "when_competitor_wins": [
            "Rapid prototyping and experimentation",
            "Small teams that don't need enterprise controls",
            "Inference/serving workloads with bursty GPU needs",
            "Developers who want the absolute simplest getting-started experience",
        ],
    },
}

_WHY_FLYTE_UNION: dict[str, dict] = {
    "on_prem_and_hybrid": {
        "dimension": "On-Prem & Hybrid Deployment",
        "summary": (
            "Flyte runs on any Kubernetes cluster — on-prem, cloud, or hybrid. "
            "Union adds a managed control plane that can federate across multiple "
            "clusters and clouds. This is critical for regulated industries "
            "(healthcare, finance, defense) with data sovereignty requirements."
        ),
        "key_points": [
            "Run in your own VPC — data never leaves your network",
            "Hybrid: control plane in Union Cloud, data plane on-prem",
            "Multi-cluster federation across AWS, GCP, Azure simultaneously",
            "Air-gapped deployment options for defense/government",
        ],
    },
    "security_and_compliance": {
        "dimension": "Security & Compliance",
        "summary": (
            "Union is SOC 2 Type II certified and HIPAA-eligible. Flyte's "
            "architecture enforces tenant isolation at the Kubernetes namespace "
            "level with separate service accounts, resource quotas, and network "
            "policies per project-domain."
        ),
        "key_points": [
            "SOC 2 Type II certified (Union)",
            "HIPAA-eligible (Union)",
            "FedRAMP authorization in progress (Union)",
            "Kubernetes namespace-level tenant isolation",
            "RBAC with OIDC/SAML SSO integration",
            "Audit logging for all API calls and executions",
            "Secrets management integrated with K8s secrets, Vault, AWS SM",
        ],
    },
    "reproducibility": {
        "dimension": "Reproducibility & Versioning",
        "summary": (
            "Every Flyte workflow registration creates an immutable, versioned "
            "snapshot of code, container image, and input/output types. Any past "
            "execution can be exactly reproduced. This is essential for regulated "
            "ML (model governance, audit trails)."
        ),
        "key_points": [
            "Immutable workflow versions — code + container + types",
            "Every execution is fully traceable",
            "Built-in data catalog tracks all artifacts",
            "Cache keys derived from input hashes — deterministic reuse",
            "Critical for FDA, financial model governance, etc.",
        ],
    },
    "scalability": {
        "dimension": "Scalability",
        "summary": (
            "Flyte scales to thousands of concurrent workflows with millions of "
            "tasks. The architecture separates the control plane (scheduling, "
            "metadata) from the data plane (execution), so they scale independently. "
            "Map tasks can fan out to 10,000+ parallel executions."
        ),
        "key_points": [
            "Proven at Lyft scale (thousands of daily workflows)",
            "Map tasks for embarrassingly parallel workloads (10K+ fan-out)",
            "Control plane / data plane separation",
            "Horizontal scaling via Kubernetes",
            "Union Serverless auto-scales to zero",
        ],
    },
    "ml_native": {
        "dimension": "ML/AI Native Design",
        "summary": (
            "Unlike general-purpose orchestrators (Airflow, Prefect), Flyte was "
            "purpose-built for ML. Strong typing catches errors before execution. "
            "Intra-task checkpointing saves GPU hours on long training runs. "
            "Native plugins for Spark, Ray, MPI, and more."
        ),
        "key_points": [
            "Strong typing with automatic serialization/deserialization",
            "Intra-task checkpointing (resume training from last checkpoint)",
            "GPU-aware scheduling with resource requests",
            "Native Ray, Spark, Dask, MPI task plugins",
            "Dynamic workflows for hyperparameter search, NAS, etc.",
            "Artifact lineage from data to model to deployment",
        ],
    },
    "developer_experience": {
        "dimension": "Developer Experience",
        "summary": (
            "The Flyte v2 SDK (flyte-sdk) provides a Pythonic, decorator-based "
            "API. Local execution works identically to remote — write once, "
            "run anywhere. The new SDK dramatically simplifies the API surface "
            "compared to flytekit v1."
        ),
        "key_points": [
            "@flyte.task and @flyte.workflow decorators — minimal boilerplate",
            "Local execution mirrors remote behavior exactly",
            "Type-safe inputs/outputs with Python type hints",
            "flyte.Image for declarative container image building",
            "flyte.app for deploying long-running services (FastAPI, Gradio, etc.)",
            "Hot-reload and interactive development with Union Actors",
        ],
    },
    "cost_efficiency": {
        "dimension": "Cost Efficiency",
        "summary": (
            "Flyte's caching avoids redundant computation. Spot/preemptible "
            "instance support with checkpointing minimizes GPU costs. Union "
            "Serverless provides pay-per-second billing with auto-scale-to-zero."
        ),
        "key_points": [
            "Deterministic caching — never re-run identical computation",
            "Spot instance support with automatic checkpointing/retry",
            "Union Serverless: pay only for compute seconds used",
            "Resource quotas prevent runaway costs",
            "Multi-cluster scheduling routes to cheapest available capacity",
        ],
    },
}

_SALES_FUNNEL_GUIDANCE = {
    "top_of_funnel": {
        "stage": "Awareness / Research",
        "description": "Prospect is exploring AI orchestration options",
        "tone": "Educational, value-focused, not pushy",
        "focus_areas": [
            "What is AI orchestration and why it matters",
            "High-level Flyte/Union differentiators",
            "Social proof (adopters, GitHub stats, CNCF status)",
            "Competitive landscape overview",
        ],
        "sample_questions": [
            "What is Flyte?",
            "How does Flyte compare to Airflow?",
            "Why would I use Union instead of managing Flyte myself?",
            "What makes Flyte different from other orchestrators?",
        ],
    },
    "mid_funnel": {
        "stage": "Evaluation / POC",
        "description": "Prospect is actively comparing solutions",
        "tone": "Technical, detailed, honest about trade-offs",
        "focus_areas": [
            "Deep technical comparisons on specific dimensions",
            "Architecture and deployment options",
            "Migration paths from existing tools",
            "Security and compliance details",
        ],
        "sample_questions": [
            "Can Flyte run on-prem in an air-gapped environment?",
            "How does Flyte handle GPU scheduling for training jobs?",
            "What's the migration path from Airflow to Flyte?",
            "How does multi-tenant isolation work?",
        ],
    },
    "bottom_funnel": {
        "stage": "Decision / Procurement",
        "description": "Prospect is ready to buy or adopt",
        "tone": "Confident, ROI-focused, address last objections",
        "focus_areas": [
            "Total cost of ownership",
            "Enterprise support and SLAs",
            "Compliance certifications",
            "Success stories and case studies",
        ],
        "sample_questions": [
            "What are Union's pricing tiers?",
            "Is Union SOC 2 certified?",
            "What support SLAs does Union offer?",
            "Can you help us build a business case?",
        ],
    },
}


# ---------------------------------------------------------------------------
# Tool functions — these are what the agent calls inside the sandbox
# ---------------------------------------------------------------------------


async def lookup_union_info(
    topic: Literal["overview", "features", "ideal_for", "all"],
) -> dict[str, OverviewInfo | list[str]]:
    """Look up information about Union (the managed Flyte platform).

    Returns dict keyed by *topic*:
      - "overview" -> {"overview": OverviewInfo}
      - "features" -> {"features": list[str]}
      - "ideal_for" -> {"ideal_for": list[str]}
      - "all" -> {"overview": ..., "features": [...], "ideal_for": [...]}

    Access the data with: result = lookup_union_info("overview"); info = result["overview"]
    """
    if topic in _UNION_OVERVIEW:
        return {topic: _UNION_OVERVIEW[topic]}
    return dict(_UNION_OVERVIEW)


async def lookup_flyte_info(
    topic: Literal["overview", "features", "adoption", "repos", "all"],
) -> dict[str, OverviewInfo | list[str] | dict[str, str]]:
    """Look up information about Flyte (the open-source orchestration platform).

    Returns dict keyed by *topic*:
      - "overview" -> {"overview": OverviewInfo}  (keys: name, tagline, website_url, docs_url, description, origin,
        status, github_stars, license)
      - "features" -> {"features": list[str]}
      - "adoption" -> {"adoption": list[str]}
      - "repos"    -> {"repos": {"flyte": url, "flyte-sdk-v2": url, "flytekit-v1": url}}
      - "all"      -> all of the above combined

    Access the data with: result = lookup_flyte_info("overview"); info = result["overview"]
    The features list contains plain strings (not dicts), e.g. ["Kubernetes-native — runs anywhere K8s runs", ...]
    The adoption list contains plain strings, e.g. ["Lyft (creator)", "Spotify", ...]
    """
    if topic in _FLYTE_OVERVIEW:
        return {topic: _FLYTE_OVERVIEW[topic]}
    return dict(_FLYTE_OVERVIEW)


async def compare_with_competitor(
    competitor: Literal["airflow", "prefect", "dagster", "ray", "kubeflow", "modal"],
) -> dict[str, CompetitorProfile]:
    """Get a detailed comparison of Flyte/Union vs a specific competitor.

    Returns dict keyed by competitor name:
      {"airflow": {"name": str, "category": str, "strengths": list[str],
                   "weaknesses_vs_flyte": list[str], "when_competitor_wins": list[str]}}

    Access: result = compare_with_competitor("airflow"); profile = result["airflow"]
    """
    key = competitor.lower().strip()
    profile = _COMPETITOR_PROFILES.get(key)
    if profile is None:
        available = ", ".join(sorted(_COMPETITOR_PROFILES.keys()))
        return {"error": f"Unknown competitor '{competitor}'. Available: {available}"}
    return {key: dict(profile)}


async def list_competitors() -> list[CompetitorSummary]:
    """List all competitors that can be compared against Flyte/Union.

    Returns list of {"key": str, "name": str, "category": str}.
    """
    return [{"key": k, "name": v["name"], "category": v["category"]} for k, v in _COMPETITOR_PROFILES.items()]


async def lookup_why_flyte(
    dimension: Literal[
        "on_prem_and_hybrid",
        "security_and_compliance",
        "reproducibility",
        "scalability",
        "ml_native",
        "developer_experience",
        "cost_efficiency",
        "all",
    ],
) -> dict[str, WhyFlyteDimension]:
    """Look up a specific technical reason for choosing Flyte/Union.

    Returns dict keyed by *dimension*:
      {"ml_native": {"dimension": str, "summary": str, "key_points": list[str]}}

    For "all", returns all dimensions keyed by their names.
    Access: result = lookup_why_flyte("ml_native"); info = result["ml_native"]
    """
    if dimension == "all":
        return _WHY_FLYTE_UNION
    entry = _WHY_FLYTE_UNION.get(dimension)
    if entry is None:
        available = ", ".join(sorted(_WHY_FLYTE_UNION.keys()))
        return {"error": f"Unknown dimension '{dimension}'. Available: {available}"}
    return {dimension: dict(entry)}


async def list_why_flyte_dimensions() -> list[WhyFlyteSummary]:
    """List all available 'why Flyte/Union' dimensions.

    Returns list of {"key": str, "dimension": str, "summary": str}.
    """
    return [
        {"key": k, "dimension": v["dimension"], "summary": v["summary"][:120] + "..."}
        for k, v in _WHY_FLYTE_UNION.items()
    ]


async def get_sales_funnel_guidance(
    stage: Literal["top_of_funnel", "mid_funnel", "bottom_funnel", "all"],
) -> dict[str, FunnelStage]:
    """Get guidance on how to respond based on the prospect's funnel stage.

    Returns dict keyed by *stage*:
      {"top_of_funnel": {"stage": str, "description": str, "tone": str,
                         "focus_areas": list[str], "sample_questions": list[str]}}

    For "all", returns all stages keyed by their names.
    """
    if stage == "all":
        return dict(_SALES_FUNNEL_GUIDANCE)
    entry = _SALES_FUNNEL_GUIDANCE.get(stage)
    if entry is None:
        available = ", ".join(sorted(_SALES_FUNNEL_GUIDANCE.keys()))
        return {"error": f"Unknown stage '{stage}'. Available: {available}"}
    return {stage: dict(entry)}


async def get_resource_links(
    resource_type: Literal["websites", "docs", "repos", "all"],
) -> dict[str, LinkGroup]:
    """Get official links for Union and Flyte resources.

    Returns dict keyed by *resource_type*:
      - "websites" -> {"websites": {"union": url, "flyte": url}}
      - "docs"     -> {"docs": {"union_docs": url, "flyte_docs": url}}
      - "repos"    -> {"repos": {"flyte_core": url, "flyte_sdk_v2": url, "flytekit_v1": url}}
      - "all"      -> all three combined

    Access: result = get_resource_links("all"); docs = result["docs"]
    """
    _links = {
        "websites": {
            "union": "https://www.union.ai/",
            "flyte": "https://flyte.org/",
        },
        "docs": {
            "union_docs": "https://www.union.ai/docs/v2/union/user-guide/",
            "flyte_docs": "https://www.union.ai/docs/v2/flyte/user-guide/",
        },
        "repos": {
            "flyte_core": "https://github.com/flyteorg/flyte",
            "flyte_sdk_v2": "https://github.com/flyteorg/flyte-sdk",
            "flytekit_v1": "https://github.com/flyteorg/flytekit",
        },
    }
    if resource_type in _links:
        return {resource_type: _links[resource_type]}
    return dict(_links)


def _format_code_example(title: str, description: str, code: str) -> FormattedResponse:
    return {
        "summary": (
            "## "
            + title
            + "\n\n"
            + description
            + "\n\n"
            + "```python\n"
            + code
            + "```\n\n"
            + "**Learn more:**\n"
            + "- https://www.union.ai/docs/v2/union/user-guide/\n"
            + "- https://github.com/flyteorg/flyte-sdk"
        )
    }


async def get_biotech_example() -> FormattedResponse:
    """Get a Flyte v2 example workflow for a biotech use case.

    Returns {"summary": str} with a formatted Markdown response containing
    a protein sequence analysis pipeline example. This is a final response —
    do NOT pass it to format_response, just return it directly.
    """
    return _format_code_example(
        "Biotech: Protein Sequence Analysis",
        "A simple Flyte v2 pipeline that validates a protein sequence, extracts features, and classifies its family.",
        (
            "import flyte\n"
            "\n"
            "env = flyte.TaskEnvironment(name='biotech')\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def validate_sequence(seq: str) -> str:\n"
            "    valid = set('ACDEFGHIKLMNPQRSTVWY')\n"
            "    cleaned = seq.upper().strip()\n"
            "    if not all(c in valid for c in cleaned):\n"
            "        raise ValueError('Invalid amino acid characters')\n"
            "    return cleaned\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def extract_features(seq: str) -> dict[str, float]:\n"
            "    length = len(seq)\n"
            "    hydrophobic = sum(1 for c in seq if c in 'AILMFVPW') / length\n"
            "    charged = sum(1 for c in seq if c in 'DEKRH') / length\n"
            "    return {'length': length, 'hydrophobic_frac': hydrophobic, 'charged_frac': charged}\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def classify_family(features: dict[str, float]) -> str:\n"
            "    if features['length'] < 50:\n"
            "        return 'peptide'\n"
            "    if features['hydrophobic_frac'] > 0.4:\n"
            "        return 'membrane-protein'\n"
            "    return 'globular-protein'\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def protein_analysis(seq: str) -> str:\n"
            "    cleaned = await validate_sequence(seq)\n"
            "    features = await extract_features(cleaned)\n"
            "    return await classify_family(features)\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    flyte.init_from_config()\n"
            "    run = flyte.run(protein_analysis, seq='MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH')\n"
            "    print(run.url)\n"
            "    run.wait()\n"
        ),
    )


async def get_autonomous_vehicles_example() -> FormattedResponse:
    """Get a Flyte v2 example workflow for an autonomous vehicles use case.

    Returns {"summary": str} with a formatted Markdown response containing
    a sensor fusion pipeline example. This is a final response —
    do NOT pass it to format_response, just return it directly.
    """
    return _format_code_example(
        "Autonomous Vehicles: Sensor Fusion Pipeline",
        "A simple Flyte v2 pipeline that preprocesses lidar and camera data, fuses them, and runs a prediction model.",
        (
            "import asyncio\n"
            "\n"
            "import flyte\n"
            "\n"
            "env = flyte.TaskEnvironment(name='av-fusion')\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def preprocess_lidar(raw_path: str) -> list[dict[str, float]]:\n"
            "    return [{'x': 1.0, 'y': 2.0, 'z': 0.3, 'intensity': 0.9}]\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def preprocess_camera(image_path: str) -> list[dict]:\n"
            "    return [{'label': 'car', 'bbox': [100, 200, 300, 400], 'confidence': 0.92}]\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def fuse_and_predict(\n"
            "    lidar_points: list[dict[str, float]],\n"
            "    detections: list[dict],\n"
            ") -> list[dict]:\n"
            "    fused = []\n"
            "    for det in detections:\n"
            "        det['depth'] = lidar_points[0]['z'] if lidar_points else None\n"
            "        fused.append(det)\n"
            "    return [{'label': obj['label'], 'trajectory': 'straight', 'risk': 'low'} for obj in fused]\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def sensor_fusion_pipeline(lidar_path: str, camera_path: str) -> list[dict]:\n"
            "    points, detections = await asyncio.gather(\n"
            "        preprocess_lidar(lidar_path),\n"
            "        preprocess_camera(camera_path),\n"
            "    )\n"
            "    return await fuse_and_predict(points, detections)\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    flyte.init_from_config()\n"
            "    run = flyte.run(sensor_fusion_pipeline, lidar_path='s3://data/lidar.bin', camera_path='s3://data/frame.jpg')\n"
            "    print(run.url)\n"
            "    run.wait()\n"
        ),
    )


async def get_ai_agents_example() -> FormattedResponse:
    """Get a Flyte v2 example workflow for an AI agents use case.

    Returns {"summary": str} with a formatted Markdown response containing
    a tool-calling agent loop example. This is a final response —
    do NOT pass it to format_response, just return it directly.
    """
    return _format_code_example(
        "AI Agents: Tool-Calling Agent Loop",
        "A simple Flyte v2 pipeline that plans actions, executes tool calls, and synthesizes a final response.",
        (
            "import flyte\n"
            "\n"
            "env = flyte.TaskEnvironment(name='ai-agent')\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def plan_actions(query: str) -> list[str]:\n"
            "    tools = []\n"
            "    if 'weather' in query.lower():\n"
            "        tools.append('get_weather')\n"
            "    if 'news' in query.lower():\n"
            "        tools.append('get_news')\n"
            "    if not tools:\n"
            "        tools.append('general_knowledge')\n"
            "    return tools\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def execute_tools(tools: list[str]) -> dict[str, str]:\n"
            "    return {t: 'Result from ' + t for t in tools}\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def synthesize(query: str, tool_results: dict[str, str]) -> str:\n"
            "    lines = ['Query: ' + query]\n"
            "    for name, val in tool_results.items():\n"
            "        lines.append('- ' + name + ': ' + val)\n"
            "    return chr(10).join(lines)\n"
            "\n"
            "\n"
            "@env.task\n"
            "async def agent_loop(query: str) -> str:\n"
            "    tools = await plan_actions(query)\n"
            "    results = await execute_tools(tools)\n"
            "    return await synthesize(query, results)\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    flyte.init_from_config()\n"
            "    run = flyte.run(agent_loop, query='What is the weather and latest news?')\n"
            "    print(run.url)\n"
            "    run.wait()\n"
        ),
    )


async def format_response(
    title: str,
    body: str,
    links: list[str] | None = None,
) -> FormattedResponse:
    """Format a response with a title, body text, and optional reference links.

    MUST be the last call in your code. Returns {"summary": str} which the
    chat UI renders as Markdown.

    Args:
        title: A short heading for the response.
        body: The main response text (Markdown supported).
        links: Optional list of relevant URLs to include as references.
    """
    parts = [f"## {title}", "", body]
    if links:
        parts.append("")
        parts.append("**Learn more:**")
        for link in links:
            parts.append(f"- {link}")
    return {"summary": "\n".join(parts)}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

ALL_TOOLS: dict[str, Callable] = {
    "lookup_union_info": lookup_union_info,
    "lookup_flyte_info": lookup_flyte_info,
    "compare_with_competitor": compare_with_competitor,
    "list_competitors": list_competitors,
    "lookup_why_flyte": lookup_why_flyte,
    "list_why_flyte_dimensions": list_why_flyte_dimensions,
    "get_sales_funnel_guidance": get_sales_funnel_guidance,
    "get_resource_links": get_resource_links,
    "get_biotech_example": get_biotech_example,
    "get_autonomous_vehicles_example": get_autonomous_vehicles_example,
    "get_ai_agents_example": get_ai_agents_example,
    "format_response": format_response,
}


# ---------------------------------------------------------------------------
# Agent + Environment
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PREFIX = """\
You are a knowledgeable, helpful, and honest pre-sales technical assistant \
for Union and Flyte — the AI/ML orchestration platform. Your goal is to help \
prospects understand why Flyte and Union are excellent choices for AI \
orchestration, while being transparent about trade-offs.

Guidelines:
- Be educational and value-focused, not pushy or salesy.
- Always back claims with specific technical facts from the tool functions.
- When comparing against competitors, be fair — acknowledge competitor \
strengths before explaining Flyte/Union advantages.
- If you don't know something, say so and point to official resources.
- For top-of-funnel questions, keep answers accessible and high-level.
- For mid-funnel questions, go deep on technical details.
- Always include relevant links to docs, website, or repos when appropriate.
- Your final call MUST be format_response(title, body, links) which returns \
{"summary": str}. The body supports Markdown.

CRITICAL — Data access patterns:
Every lookup function returns a dict keyed by the topic/dimension name.
- result = lookup_flyte_info("overview")  =>  result["overview"]["description"]
- result = lookup_flyte_info("all") =>  result["overview"], result["features"], result["adoption"], result["repos"]
- result["features"] is a list[str], NOT a list of dicts. Each item is a plain string like "Kubernetes-native —
  runs anywhere K8s runs".
- result["adoption"] is a list[str], e.g. ["Lyft (creator)", "Spotify", ...].
- result = compare_with_competitor("airflow")  =>  result["airflow"]["strengths"]
- result = lookup_why_flyte("ml_native")  =>  result["ml_native"]["key_points"]
- result = get_resource_links("all")  =>  result["docs"]["flyte_docs"], result["repos"]["flyte_core"]

Do NOT invent dict keys that don't exist. Use only the keys documented above.
When iterating features or adoption, iterate the list directly: for f in features: ...
Features are strings, not dicts — do NOT access .name or ["name"] on them.

Code example tools (return pre-formatted responses — do NOT pass to format_response):
- result = await get_biotech_example()       => result is {"summary": str}
- result = await get_autonomous_vehicles_example() => result is {"summary": str}
- result = await get_ai_agents_example()     => result is {"summary": str}
These tools already return a complete formatted response. Just return the result \
dict directly — do NOT call format_response after calling these tools.
"""

agent = CodeModeAgent(
    tools=ALL_TOOLS,
    model="claude-haiku-4-5",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)

env = AgentChatAppEnvironment(
    name="union-flyte-sales-agent",
    agent=agent,
    title="Union Sales Agent",
    subtitle=(
        "A chat agent that answers questions about Union and Flyte for AI leaders and technical decision makers."
    ),
    theme=CustomTheme(
        accent_color="#E6A71F",
        accent_hover_color="#F0BA4A",
        button_text_color="#0a0a0f",
    ),
    logo_url="https://www.union.ai/docs/v2/union/images/icon-logo.svg",
    prompt_nudges=[
        {
            "label": "What is Flyte?",
            "prompt": "What is Flyte and why should I care about it for my ML pipelines?",
        },
        {
            "label": "Flyte vs Airflow",
            "prompt": "How does Flyte compare to Apache Airflow for ML workloads?",
        },
        {
            "label": "Why Union?",
            "prompt": ("Why would I use Union instead of self-managing open-source Flyte?"),
        },
        {
            "label": "On-prem & Security",
            "prompt": ("Can Flyte/Union run on-prem? What security and compliance certifications does Union have?"),
        },
        {
            "label": "Compare all competitors",
            "prompt": (
                "Give me a high-level comparison of Flyte/Union against "
                "Ray, Prefect, Dagster, Kubeflow, Modal, and Airflow."
            ),
        },
        {
            "label": "Getting started",
            "prompt": ("Where can I find the docs, GitHub repos, and quickstart guides for Flyte and Union?"),
        },
        {
            "label": "Biotech example",
            "prompt": "Show me a Flyte v2 example for a biotech use case like protein sequence analysis.",
        },
        {
            "label": "Autonomous vehicles example",
            "prompt": "Show me a Flyte v2 example for an autonomous vehicles sensor fusion pipeline.",
        },
        {
            "label": "AI agents example",
            "prompt": "Show me a Flyte v2 example for building an AI agent with tool calling.",
        },
    ],
    additional_buttons=[
        {
            "button_text": "Chat with an engineer",
            "button_url": "https://www.union.ai/consultation",
        },
    ],
    image=flyte.Image.from_debian_base(install_flyte=False)
    .with_pip_packages(
        "litellm",
        "pydantic-monty==0.0.8",
        "uvicorn",
        "fastapi",
        "flyte[sandbox]",
    )
    .with_local_v2(),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_handle = flyte.serve(env)
    print(f"Sales Agent running at: {app_handle.url}")
