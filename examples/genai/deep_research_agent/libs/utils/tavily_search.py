import asyncio
import os
from dataclasses import dataclass
from typing import Optional

import flyte


@dataclass(frozen=True, kw_only=True)
class SearchResult:
    title: str
    link: str
    content: str
    raw_content: Optional[str] = None

    def __str__(self, include_raw=True):
        result = f"Title: {self.title}\nLink: {self.link}\nContent: {self.content}"
        if include_raw and self.raw_content:
            result += f"\nRaw Content: {self.raw_content}"
        return result

    def short_str(self):
        return self.__str__(include_raw=False)


@dataclass(frozen=True, kw_only=True)
class SearchResults:
    results: list[SearchResult]

    def __str__(self, short=False):
        if short:
            result_strs = [result.short_str() for result in self.results]
        else:
            result_strs = [str(result) for result in self.results]
        return "\n\n".join(f"[{i + 1}] {result_str}" for i, result_str in enumerate(result_strs))

    def __add__(self, other):
        return SearchResults(results=self.results + other.results)

    def short_str(self):
        return self.__str__(short=True)


def extract_tavily_results(response) -> SearchResults:
    """Extract key information from Tavily search results."""
    results = []
    for item in response.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                link=item.get("url", ""),
                content=item.get("content", ""),
                raw_content=item.get("raw_content", ""),
            )
        )
    return SearchResults(results=results)


def tavily_search(query: str, max_results=3, include_raw: bool = True) -> SearchResults:
    """
    Perform a search using the Tavily Search API with the official client.

    Parameters:
        query (str): The search query.
        search_depth (str): The depth of search - 'basic' or 'deep'.
        max_results (int): Maximum number of results to return.

    Returns:
        list: Formatted search results with title, link, and snippet.
    """
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    client = TavilyClient(api_key)

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_raw_content=include_raw,
    )

    return extract_tavily_results(response)


@flyte.trace
async def atavily_search_results(query: str, max_results: int = 3, include_raw: bool = True) -> SearchResults:
    """
    Perform asynchronous search using the Tavily Search API with the official client.

    Parameters:
        query (str): The search query.
        max_results (int): Maximum number of results to return.
    """
    from tavily import AsyncTavilyClient

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    client = AsyncTavilyClient(api_key)

    response = await client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_raw_content=include_raw,
    )

    return extract_tavily_results(response)


if __name__ == "__main__":
    print(asyncio.run(atavily_search_results("What is the capital of France?")))
