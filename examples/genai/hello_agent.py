import asyncio
from typing import List

from pydantic import BaseModel

import flyte

env = flyte.TaskEnvironment("hello-agent")


class SearchResult(BaseModel):
    company: str
    info: str


class ResearchState(BaseModel):
    query: str
    results: List[SearchResult] = []


@env.task
async def search_subagent(query: str, idx: int) -> SearchResult:
    await asyncio.sleep(0.1)
    return SearchResult(company=f"Company{idx}", info=f"Info about {query} {idx}")


@env.task
async def lead_agent(state: ResearchState, num_subagents: int = 3) -> ResearchState:
    tasks = [search_subagent(state.query, i) for i in range(num_subagents)]
    state.results = await asyncio.gather(*tasks)
    return state


if __name__ == "__main__":
    import flyte.git

    flyte.init_from_config(flyte.git.config_from_root())
    state = ResearchState(query="AI agent companies 2025")
    run = flyte.run(lead_agent, state)
    print(run.url)
    run.wait(run)
