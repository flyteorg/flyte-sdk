from dataclasses import dataclass

from pydantic import BaseModel, Field

from libs.utils.tavily_search import SearchResult, SearchResults


class ResearchPlan(BaseModel):
    queries: list[str] = Field(description="A list of search queries to thoroughly research the topic")


class SourceList(BaseModel):
    sources: list[int] = Field(description="A list of source numbers from the search results")


@dataclass(frozen=True, kw_only=True)
class DeepResearchResult(SearchResult):
    """Wrapper on top of SearchResults to adapt it to the DeepResearch.

    This class extends the basic SearchResult by adding a filtered version of the raw content
    that has been processed and refined for the specific research context. It maintains
    the original search result while providing additional research-specific information.

    Attributes:
        filtered_raw_content: A processed version of the raw content that has been filtered
                             and refined for relevance to the research topic
    """

    filtered_raw_content: str

    def __str__(self):
        return f"Title: {self.title}\nLink: {self.link}\nRefined Content: {self.filtered_raw_content[:10000]}"

    def short_str(self):
        return f"Title: {self.title}\nLink: {self.link}\nRaw Content: {self.content[:10000]}"


@dataclass(frozen=True, kw_only=True)
class DeepResearchResults(SearchResults):
    results: list[DeepResearchResult]

    def __add__(self, other):
        return DeepResearchResults(results=self.results + other.results)

    def dedup(self):
        def deduplicate_by_link(results):
            seen_links = set()
            unique_results = []

            for result in results:
                if result.link not in seen_links:
                    seen_links.add(result.link)
                    unique_results.append(result)

            return unique_results

        return DeepResearchResults(results=deduplicate_by_link(self.results))
