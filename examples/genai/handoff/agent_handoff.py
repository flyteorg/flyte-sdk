"""
Agent Handoff System Example

This example demonstrates a deterministic workflow for routing user queries to specialized agents:
1. Extract relevant workflow tags from the user query
2. Filter the agent registry by matching tags
3. Score candidate agents using semantic similarity (OpenAI embeddings)
4. Hand off to the top-scoring agent if it exceeds the threshold
5. Raise an error if no agent meets the threshold

The system uses 10 dummy agents, each with tags and descriptions.
"""

import asyncio
import pathlib
from typing import List, Optional

import flyte
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Configure reusable environment with OpenAI API key
env = flyte.TaskEnvironment(
    "agent-handoff",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=[flyte.Secret(key="openai-api-key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base().with_uv_project(
        pyproject_file=pathlib.Path(__file__).parent / "pyproject.toml", pre=True
    ),
    reusable=flyte.ReusePolicy(
        replicas=(1, 5),
        concurrency=10,
        idle_ttl=30 * 60,
    ),
    cache=flyte.Cache("auto", version_override="1.0"),
)


# Data Models
class Agent(BaseModel):
    """Represents a specialized agent in the registry."""

    id: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    threshold: float = 0.7  # Minimum similarity score to hand off


class AgentScore(BaseModel):
    """Agent with computed similarity score."""

    agent: Agent
    score: float
    reasoning: str = ""


class HandoffResult(BaseModel):
    """Result of the agent handoff process."""

    selected_agent: Optional[Agent] = None
    all_scores: List[AgentScore] = Field(default_factory=list)
    extracted_tags: List[str] = Field(default_factory=list)
    filtered_count: int = 0
    handoff_successful: bool = False
    error_message: Optional[str] = None


# Agent Registry - 10 Specialized Agents
AGENT_REGISTRY = [
    Agent(
        id="data-analyst",
        name="Data Analytics Agent",
        description="Specializes in data analysis, SQL queries, data visualization, "
        "and generating insights from datasets",
        tags=["data", "analytics", "sql", "visualization", "reporting"],
    ),
    Agent(
        id="code-reviewer",
        name="Code Review Agent",
        description="Expert at reviewing code, identifying bugs, suggesting improvements, and ensuring best practices",
        tags=["code", "review", "testing", "quality", "debugging"],
    ),
    Agent(
        id="doc-writer",
        name="Documentation Agent",
        description="Creates comprehensive technical documentation, API docs, user guides, and tutorials",
        tags=["documentation", "writing", "technical", "guides", "api"],
    ),
    Agent(
        id="ml-engineer",
        name="Machine Learning Agent",
        description="Builds and deploys ML models, handles feature engineering, model training, and optimization",
        tags=["ml", "ai", "models", "training", "deployment"],
    ),
    Agent(
        id="devops-agent",
        name="DevOps Automation Agent",
        description="Manages CI/CD pipelines, infrastructure as code, deployment automation, and monitoring",
        tags=["devops", "cicd", "infrastructure", "deployment", "automation"],
    ),
    Agent(
        id="security-audit",
        name="Security Audit Agent",
        description="Performs security audits, vulnerability scanning, penetration testing, and compliance checks",
        tags=["security", "audit", "vulnerability", "compliance", "testing"],
    ),
    Agent(
        id="api-designer",
        name="API Design Agent",
        description="Designs RESTful APIs, GraphQL schemas, API specifications, and integration patterns",
        tags=["api", "rest", "graphql", "design", "integration"],
    ),
    Agent(
        id="customer-support",
        name="Customer Support Agent",
        description="Handles customer inquiries, troubleshooting, FAQ generation, and support ticket management",
        tags=["support", "customer", "troubleshooting", "help", "tickets"],
    ),
    Agent(
        id="data-pipeline",
        name="Data Pipeline Agent",
        description="Builds ETL pipelines, manages data ingestion, transformation, and orchestration workflows",
        tags=["data", "pipeline", "etl", "ingestion", "orchestration"],
    ),
    Agent(
        id="frontend-dev",
        name="Frontend Development Agent",
        description="Develops UI components, handles state management, responsive design, and frontend optimization",
        tags=["frontend", "ui", "react", "design", "components"],
    ),
]


@flyte.trace
async def extract_tags_from_query(query: str) -> List[str]:
    """
    Use OpenAI to extract relevant workflow tags from the user query.

    Args:
        query: User's natural language query

    Returns:
        List of extracted tags
    """
    client = AsyncOpenAI()

    # Get all unique tags from the registry
    all_tags = sorted({tag for agent in AGENT_REGISTRY for tag in agent.tags})

    prompt = f"""Given the following user query, identify the most relevant workflow tags from the list below.
Return only the tags that are directly relevant to the query, as a comma-separated list.

Available tags: {", ".join(all_tags)}

User query: "{query}"

Instructions:
- Return 1-5 most relevant tags
- Only use tags from the available list
- Return as comma-separated values
- If no tags are relevant, return "none"

Always return 5 tags.

Relevant tags:"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )

    tags_text = response.choices[0].message.content.strip().lower()

    if tags_text == "none":
        return []

    # Parse and validate tags
    extracted = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
    # Only return valid tags that exist in the registry
    valid_tags = [tag for tag in extracted if tag in all_tags]

    return valid_tags


def filter_agents_by_tags(agents: List[Agent], tags: List[str]) -> List[Agent]:
    """
    Filter agents that have at least one matching tag.
    Pure function - no external calls, deterministic logic only.

    Args:
        agents: List of all agents
        tags: Tags to filter by

    Returns:
        Filtered list of agents
    """
    if not tags:
        return agents

    filtered = [agent for agent in agents if any(tag in agent.tags for tag in tags)]
    return filtered if filtered else agents  # Return all if no matches


@flyte.trace
async def get_embedding(text: str) -> List[float]:
    """
    Get OpenAI embedding for text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    client = AsyncOpenAI()
    response = await client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


@flyte.trace
async def score_agents(query: str, agents: List[Agent]) -> List[AgentScore]:
    """
    Score agents based on semantic similarity between query and agent descriptions.

    Args:
        query: User query
        agents: List of candidate agents

    Returns:
        List of scored agents, sorted by score (highest first)
    """
    # Get embeddings
    query_embedding = await get_embedding(query)

    async def score_single_agent(agent: Agent) -> AgentScore:
        agent_embedding = await get_embedding(agent.description)

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(query_embedding, agent_embedding))
        query_norm = sum(x * x for x in query_embedding) ** 0.5
        agent_norm = sum(x * x for x in agent_embedding) ** 0.5
        similarity = dot_product / (query_norm * agent_norm)

        reasoning = "Similarity score based on semantic match between query and agent description"

        return AgentScore(agent=agent, score=float(similarity), reasoning=reasoning)

    # Score all agents in parallel
    scores = await asyncio.gather(*[score_single_agent(agent) for agent in agents])

    # Sort by score descending
    scores.sort(key=lambda x: x.score, reverse=True)

    return scores


def select_agent(scored_agents: List[AgentScore], threshold: float = 0.7) -> Optional[Agent]:
    """
    Select the top-scoring agent if it meets the threshold.
    Pure function - deterministic selection logic only.

    Args:
        scored_agents: List of agents with scores
        threshold: Minimum score required for handoff

    Returns:
        Selected agent or None if no agent meets threshold
    """
    if not scored_agents:
        return None

    top_agent_score = scored_agents[0]
    if top_agent_score.score >= threshold:
        return top_agent_score.agent

    return None


@env.task
async def handoff_to_agent(agent: Agent, query: str) -> str:
    """
    Simulate handing off the query to the selected agent.

    Args:
        agent: Selected agent
        query: User query

    Returns:
        Response from the agent
    """
    # Simulate agent processing
    await asyncio.sleep(0.5)

    return f"""> Agent {agent.name} (ID: {agent.id}) is now handling your request:

Query: "{query}"

Agent Description: {agent.description}

Tags: {", ".join(agent.tags)}

[Agent would process the query here and return results]
"""


@env.task
async def agent_handoff_workflow(query: str, threshold: float = 0.7, show_top_n: int = 10) -> HandoffResult:
    """
    Main workflow for agent handoff system.

    Args:
        query: User's natural language query
        threshold: Minimum similarity score for handoff
        show_top_n: Number of top agents to show if handoff fails

    Returns:
        HandoffResult with selected agent or error information
    """
    result = HandoffResult()

    # Consolidated agent selection group with traces underneath
    with flyte.group("agent-selection"):
        # Extract tags from query (trace)
        extracted_tags = await extract_tags_from_query(query)
        result.extracted_tags = extracted_tags

        # Filter agents by tags (pure function, no await)
        filtered_agents = filter_agents_by_tags(AGENT_REGISTRY, extracted_tags)
        result.filtered_count = len(filtered_agents)

        # Score candidate agents (trace)
        scored_agents = await score_agents(query, filtered_agents)
        result.all_scores = scored_agents[:show_top_n]  # Keep top N for reporting

        # Select agent if score meets threshold (pure function, no await)
        selected = select_agent(scored_agents, threshold)

        if selected:
            result.selected_agent = selected
            result.handoff_successful = True
        else:
            # No agent meets threshold
            result.handoff_successful = False
            top_scores_text = "\n".join(
                [
                    f"  {i + 1}. {s.agent.name} (ID: {s.agent.id}) - Score: {s.score:.3f}"
                    for i, s in enumerate(scored_agents[:show_top_n])
                ]
            )
            result.error_message = (
                f"No agent met the threshold ({threshold}). "
                f"Top {show_top_n} candidates:\n{top_scores_text}\n\n"
                f"Please clarify your query or lower the threshold."
            )

    return result


@env.task
async def run_handoff(query: str, threshold: float = 0.7) -> str:
    """
    Execute the complete handoff workflow and return a user-friendly result.

    Args:
        query: User query
        threshold: Score threshold for handoff

    Returns:
        Human-readable result string
    """
    result = await agent_handoff_workflow(query, threshold)

    if result.handoff_successful:
        response = await handoff_to_agent(result.selected_agent, query)
        return f""" HANDOFF SUCCESSFUL

Extracted Tags: {", ".join(result.extracted_tags) if result.extracted_tags else "none"}
Filtered Agents: {result.filtered_count}
Selected Agent: {result.selected_agent.name} (Score: {result.all_scores[0].score:.3f})

{response}
"""
    else:
        # Raise error if no agent meets threshold
        raise RuntimeError(f"L HANDOFF FAILED\n\n{result.error_message}")


# Example queries for different scenarios
EXAMPLE_QUERIES = [
    # Should match Data Analytics Agent
    "I need help analyzing sales data and creating a visualization dashboard",
    # Should match ML Engineer Agent
    "Can you help me train a neural network for image classification?",
    # Should match DevOps Agent
    "I want to set up a CI/CD pipeline for my application deployment",
    # Should match Security Audit Agent
    "Perform a security audit and check for vulnerabilities in my application",
    # Should match API Design Agent
    "Design a RESTful API for a user management system",
    # Ambiguous query - might fail threshold
    "I need some help with my project",
]

if __name__ == "__main__":
    flyte.init_from_config()

    # Example 1: Successful handoff
    print("=" * 80)
    print("Example 1: Data Analytics Query")
    print("=" * 80)
    run1 = flyte.run(run_handoff, EXAMPLE_QUERIES[1], threshold=0.5)
    print(f"Run URL: {run1.url}")
    run1.wait()

    # # Example 2: ML Query
    # print("\n" + "=" * 80)
    # print("Example 2: Machine Learning Query")
    # print("=" * 80)
    # run2 = flyte.run(run_handoff, EXAMPLE_QUERIES[1], threshold=0.6)
    # print(f"Run URL: {run2.url}")
    # run2.wait()
    #
    # # Example 3: Ambiguous query with high threshold (should fail)
    # print("\n" + "=" * 80)
    # print("Example 3: Ambiguous Query (High Threshold)")
    # print("=" * 80)
    # try:
    #     run3 = flyte.run(run_handoff, EXAMPLE_QUERIES[5], threshold=0.8)
    #     print(f"Run URL: {run3.url}")
    #     run3.wait()
    # except Exception as e:
    #     print(f"Expected failure: {e}")
    #
    # # Example 4: Get detailed workflow result
    # print("\n" + "=" * 80)
    # print("Example 4: DevOps Query with Detailed Result")
    # print("=" * 80)
    # run4 = flyte.run(agent_handoff_workflow, EXAMPLE_QUERIES[2], threshold=0.6)
    # print(f"Run URL: {run4.url}")
    # run4.wait()
    # result = run4.result()
    # print("\nDetailed Result:")
    # print(f"  Extracted Tags: {result.extracted_tags}")
    # print(f"  Filtered Count: {result.filtered_count}")
    # print(f"  Selected Agent: {result.selected_agent.name if result.selected_agent else 'None'}")
    # print("  Top 3 Scores:")
    # for i, score in enumerate(result.all_scores[:3], 1):
    #     print(f"    {i}. {score.agent.name}: {score.score:.3f}")
