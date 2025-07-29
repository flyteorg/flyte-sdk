# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==0.2.0b23",
#    "pydantic==2.11.5",
#    "litellm==1.72.2",
#    "pypandoc==1.15",
#    "pandoc==2.4",
#    "tavily-python==0.7.5",
#    "commonmark==0.9.1",
#    "xhtml2pdf==0.2.17",
# ]
# ///

import asyncio
import json
import re
from pathlib import Path

import anyio
import yaml
from libs.utils.data_types import (
    DeepResearchResult,
    DeepResearchResults,
    ResearchPlan,
    SourceList,
)
from libs.utils.llms import asingle_shot_llm_call
from libs.utils.log import AgentLogger
from libs.utils.tavily_search import atavily_search_results

import flyte
from flyte.io._file import File

TIME_LIMIT_MULTIPLIER = 5
MAX_COMPLETION_TOKENS = 4096
REGISTRY = "ghcr.io/flyteorg"

logging = AgentLogger("together.open_deep_research")

env = flyte.TaskEnvironment(
    name="deep-research-agent",
    secrets=[
        # TODO: Replace with your own secrets
        flyte.Secret(key="JOE_TOGETHER_API_KEY", as_env_var="TOGETHER_API_KEY"),
        flyte.Secret(key="JOE_TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    image=flyte.Image.from_uv_script(
        __file__, name="deep-research-agent", registry=REGISTRY, arch=("linux/amd64", "linux/arm64"), pre=True
    )
    .with_apt_packages("ca-certificates", "pandoc", "texlive-xetex")
    .with_source_file(Path("prompts.yaml"), "/root"),
)


@env.task
async def generate_research_queries(
    topic: str,
    planning_model: str,
    json_model: str,
    prompts_file: File,
) -> list[str]:
    async with prompts_file.open() as fh:
        yaml_contents = fh.read()

    prompts = yaml.safe_load(yaml_contents)
    PLANNING_PROMPT = prompts["planning_prompt"]

    plan = ""
    logging.info(f"\n\nGenerated deep research plan for topic: {topic}\n\nPlan:")
    async for chunk in asingle_shot_llm_call(
        model=planning_model,
        system_prompt=PLANNING_PROMPT,
        message=f"Research Topic: {topic}",
        response_format=None,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    ):
        plan += chunk
        print(chunk, end="", flush=True)

    SEARCH_PROMPT = prompts["plan_parsing_prompt"]

    response_json = ""
    async for chunk in asingle_shot_llm_call(
        model=json_model,
        system_prompt=SEARCH_PROMPT,
        message=f"Plan to be parsed: {plan}",
        response_format={
            "type": "json_object",
            "schema": ResearchPlan.model_json_schema(),
        },
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    ):
        response_json += chunk

    plan = json.loads(response_json)
    return plan["queries"]


async def _summarize_content_async(
    raw_content: str,
    query: str,
    prompt: str,
    summarization_model: str,
) -> str:
    """Summarize content asynchronously using the LLM"""
    logging.info("Summarizing content asynchronously using the LLM")

    result = ""
    async for chunk in asingle_shot_llm_call(
        model=summarization_model,
        system_prompt=prompt,
        message=f"<Raw Content>{raw_content}</Raw Content>\n\n<Research Topic>{query}</Research Topic>",
        response_format=None,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    ):
        result += chunk
    return result


@env.task
async def search_and_summarize(
    query: str,
    prompts_file: File,
    summarization_model: str,
) -> DeepResearchResults:
    """Perform a search for a single query"""

    if len(query) > 400:
        # NOTE: we are truncating the query to 400 characters to avoid Tavily Search issues
        query = query[:400]
        logging.info(f"Truncated query to 400 characters: {query}")

    response = await atavily_search_results(query)

    logging.info("Tavily Search Called.")

    async with prompts_file.open() as fh:
        yaml_contents = fh.read()

    prompts = yaml.safe_load(yaml_contents)
    RAW_CONTENT_SUMMARIZER_PROMPT = prompts["raw_content_summarizer_prompt"]

    with flyte.group("summarize-content"):
        # Create tasks for summarization
        summarization_tasks = []
        result_info = []
        for result in response.results:
            if result.raw_content is None:
                continue
            task = _summarize_content_async(
                result.raw_content,
                query,
                RAW_CONTENT_SUMMARIZER_PROMPT,
                summarization_model,
            )
            summarization_tasks.append(task)
            result_info.append(result)

        # Use return_exceptions=True to prevent exceptions from propagating
        summarized_contents = await asyncio.gather(*summarization_tasks, return_exceptions=True)

    # Filter out exceptions
    summarized_contents = [result for result in summarized_contents if not isinstance(result, Exception)]

    formatted_results = []
    for result, summarized_content in zip(result_info, summarized_contents):
        formatted_results.append(
            DeepResearchResult(
                title=result.title,
                link=result.link,
                content=result.content,
                raw_content=result.raw_content,
                filtered_raw_content=summarized_content,
            )
        )
    return DeepResearchResults(results=formatted_results)


@env.task
async def search_all_queries(queries: list[str], summarization_model: str, prompts_file: File) -> DeepResearchResults:
    """Execute searches for all queries in parallel"""
    tasks = []
    results_list = []

    with flyte.group("parallel_search_and_summarize"):
        tasks = [search_and_summarize(query, prompts_file, summarization_model) for query in queries]

        if tasks:
            res_list = await asyncio.gather(*tasks)

    results_list.extend(res_list)

    # Combine all results
    combined_results = DeepResearchResults(results=[])
    for results in results_list:
        combined_results = combined_results + results

    return combined_results


@env.task
async def evaluate_research_completeness(
    topic: str,
    results: DeepResearchResults,
    queries: list[str],
    prompts_file: File,
    planning_model: str,
    json_model: str,
) -> list[str]:
    """
    Evaluate if the current search results are sufficient or if more research is needed.
    Returns an empty list if research is complete, or a list of additional queries if more research is needed.
    """

    # Format the search results for the LLM
    formatted_results = str(results)

    async with prompts_file.open() as fh:
        yaml_contents = fh.read()

    prompts = yaml.safe_load(yaml_contents)

    EVALUATION_PROMPT = prompts["evaluation_prompt"]

    logging.info("\nEvaluation: ")
    evaluation = ""
    async for chunk in asingle_shot_llm_call(
        model=planning_model,
        system_prompt=EVALUATION_PROMPT,
        message=(
            f"<Research Topic>{topic}</Research Topic>\n\n"
            f"<Search Queries Used>{queries}</Search Queries Used>\n\n"
            f"<Current Search Results>{formatted_results}</Current Search Results>"
        ),
        response_format=None,
        max_completion_tokens=None,
    ):
        evaluation += chunk
        print(chunk, end="", flush=True)

    EVALUATION_PARSING_PROMPT = prompts["evaluation_parsing_prompt"]

    response_json = ""
    async for chunk in asingle_shot_llm_call(
        model=json_model,
        system_prompt=EVALUATION_PARSING_PROMPT,
        message=f"Evaluation to be parsed: {evaluation}",
        response_format={
            "type": "json_object",
            "schema": ResearchPlan.model_json_schema(),
        },
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    ):
        response_json += chunk

    evaluation = json.loads(response_json)
    return evaluation["queries"]


@env.task
async def filter_results(
    topic: str,
    results: DeepResearchResults,
    prompts_file: File,
    planning_model: str,
    json_model: str,
    max_sources: int,
) -> DeepResearchResults:
    """Filter the search results based on the research plan"""

    # Format the search results for the LLM, without the raw content
    formatted_results = str(results)

    async with prompts_file.open() as fh:
        yaml_contents = fh.read()

    prompts = yaml.safe_load(yaml_contents)
    FILTER_PROMPT = prompts["filter_prompt"]

    logging.info("\nFilter response: ")
    filter_response = ""
    async for chunk in asingle_shot_llm_call(
        model=planning_model,
        system_prompt=FILTER_PROMPT,
        message=(
            f"<Research Topic>{topic}</Research Topic>\n\n"
            f"<Current Search Results>{formatted_results}</Current Search Results>"
        ),
        response_format=None,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    ):
        filter_response += chunk
        print(chunk, end="", flush=True)

    logging.info(f"Filter response: {filter_response}")

    FILTER_PARSING_PROMPT = prompts["filter_parsing_prompt"]

    response_json = ""
    async for chunk in asingle_shot_llm_call(
        model=json_model,
        system_prompt=FILTER_PARSING_PROMPT,
        message=f"Filter response to be parsed: {filter_response}",
        response_format={
            "type": "json_object",
            "schema": SourceList.model_json_schema(),
        },
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    ):
        response_json += chunk

    sources = json.loads(response_json)["sources"]

    logging.info(f"Filtered sources: {sources}")

    if max_sources != -1:
        sources = sources[:max_sources]

    # Filter the results based on the source list
    filtered_results = [results.results[i - 1] for i in sources if i - 1 < len(results.results)]

    return DeepResearchResults(results=filtered_results)


def _remove_thinking_tags(answer: str) -> str:
    """Remove content within <think> tags"""
    while "<think>" in answer and "</think>" in answer:
        start = answer.find("<think>")
        end = answer.find("</think>") + len("</think>")
        answer = answer[:start] + answer[end:]
    return answer


@env.task
async def generate_research_answer(
    topic: str,
    results: DeepResearchResults,
    remove_thinking_tags: bool,
    prompts_file: File,
    answer_model: str,
) -> str:
    """
    Generate a comprehensive answer to the research topic based on the search results.
    Returns a detailed response that synthesizes information from all search results.
    """

    formatted_results = str(results)
    async with prompts_file.open() as fh:
        yaml_contents = fh.read()

    prompts = yaml.safe_load(yaml_contents)
    ANSWER_PROMPT = prompts["answer_prompt"]

    answer = ""
    async for chunk in asingle_shot_llm_call(
        model=answer_model,
        system_prompt=ANSWER_PROMPT,
        message=f"Research Topic: {topic}\n\nSearch Results:\n{formatted_results}",
        response_format=None,
        # NOTE: This is the max_token parameter for the LLM call on Together AI,
        # may need to be changed for other providers
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    ):
        answer += chunk

    # this is just to avoid typing complaints
    if answer is None or not isinstance(answer, str):
        logging.error("No answer generated")
        return "No answer generated"

    if remove_thinking_tags:
        # Remove content within <think> tags
        answer = _remove_thinking_tags(answer)

    # Remove markdown code block markers if they exist at the beginning
    if answer.lstrip().startswith("```"):
        # Find the first line break after the opening backticks
        first_linebreak = answer.find("\n", answer.find("```"))
        if first_linebreak != -1:
            # Remove everything up to and including the first line break
            answer = answer[first_linebreak + 1 :]

        # Remove closing code block if it exists
        if answer.rstrip().endswith("```"):
            answer = answer.rstrip()[:-3].rstrip()

    return answer.strip()


@env.task(retries=flyte.RetryStrategy(count=3, backoff=10, backoff_factor=2))
async def research_topic(
    topic: str,
    budget: int = 3,
    remove_thinking_tags: bool = True,
    max_queries: int = 5,
    answer_model: str = "together_ai/deepseek-ai/DeepSeek-V3",
    planning_model: str = "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
    json_model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_sources: int = 40,
    summarization_model: str = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    prompts_file: File | str = "prompts.yaml",
) -> str:
    """Main method to conduct research on a topic. Will be used for weave evals."""
    if isinstance(prompts_file, str):
        prompts_file = await File.from_local(prompts_file)

    # Step 1: Generate initial queries
    queries = await generate_research_queries(
        topic=topic,
        planning_model=planning_model,
        json_model=json_model,
        prompts_file=prompts_file,
    )
    queries = [topic, *queries[: max_queries - 1]]
    all_queries = queries.copy()
    logging.info(f"Initial queries: {queries}")

    if len(queries) == 0:
        logging.error("No initial queries generated")
        return "No initial queries generated"

    # Step 2: Perform initial search
    results = await search_all_queries(queries, summarization_model, prompts_file)
    logging.info(f"Initial search complete, found {len(results.results)} results")

    # Step 3: Conduct iterative research within budget
    for iteration in range(budget):
        # Evaluate if more research is needed
        additional_queries = await evaluate_research_completeness(
            topic=topic,
            results=results,
            queries=all_queries,
            prompts_file=prompts_file,
            planning_model=planning_model,
            json_model=json_model,
        )

        # Filter out empty strings and check if any queries remain
        additional_queries = [q for q in additional_queries if q]
        if not additional_queries:
            logging.info("No need for additional research")
            break

        # for debugging purposes we limit the number of queries
        additional_queries = additional_queries[:max_queries]
        logging.info(f"Additional queries: {additional_queries}")

        # Expand research with new queries
        new_results = await search_all_queries(additional_queries, summarization_model, prompts_file)
        logging.info(f"Follow-up search complete, found {len(new_results.results)} results")

        results = results + new_results
        all_queries.extend(additional_queries)

    # Step 4: Generate final answer with feedback loop
    logging.info(f"Generating final answer for topic: {topic}")
    results = results.dedup()
    logging.info(f"Deduplication complete, kept {len(results.results)} results")
    filtered_results = await filter_results(
        topic=topic,
        results=results,
        prompts_file=prompts_file,
        planning_model=planning_model,
        json_model=json_model,
        max_sources=max_sources,
    )
    logging.info(f"LLM Filtering complete, kept {len(filtered_results.results)} results")

    # Generate final answer
    answer = await generate_research_answer(
        topic=topic,
        results=filtered_results,
        remove_thinking_tags=remove_thinking_tags,
        prompts_file=prompts_file,
        answer_model=answer_model,
    )

    return answer


@env.task
async def generate_pdf(answer: str, filename: str = "research_report.pdf") -> str:
    """
    Generate a PDF report from the markdown formatted research answer.
    Uses the first line of the answer as the title.

    Attempts to use pypandoc first, with fallbacks to:
    1. commonmark + xhtml2pdf if pypandoc fails
    2. A clear error message if all methods fail
    """
    import pypandoc

    # Extract the first line as title and rest as content
    lines = answer.split("\n")
    title = lines[0].strip("# ")  # Remove any markdown heading characters
    content = "\n".join(lines[1:]).strip()  # Get the rest of the content

    # Remove mermaid diagram blocks for pdf rendering
    content = re.sub(
        r"\*Figure.*?\*.*?```mermaid.*?```|```mermaid.*?```.*?\*Figure.*?\*",
        "\n",
        content,
        flags=re.DOTALL,
    )
    content = content.strip()  # Remove any extra whitespace that might remain

    disclaimer = (
        "Disclaimer: This AI-generated report may contain hallucinations, bias, or inaccuracies. "
        "Always verify information "
        "from independent sources before making decisions based on this content."
    )
    content = f"{disclaimer}\n\n{content}"

    # Create markdown with the extracted title - properly quote the title for YAML
    markdown_with_title = f'---\ntitle: "{title}"\n---\n\n{content}'

    # Try pypandoc first
    try:
        pdf_options = [
            "--pdf-engine=pdflatex",
            "--variable",
            "urlcolor=blue",
            "--variable",
            "colorlinks=true",
            "--variable",
            "linkcolor=blue",
            "--variable",
            "geometry:margin=1in",
        ]

        pypandoc.convert_text(
            markdown_with_title,
            "pdf",
            format="markdown",
            outputfile=filename,
            extra_args=pdf_options,
        )
        print(f"PDF generated successfully using pypandoc: {filename}")
        return filename
    except Exception as pandoc_error:
        print(f"Pypandoc conversion failed: {pandoc_error!s}")
        print("Trying alternative conversion methods...")

    # Try commonmark + xhtml2pdf as a backup
    try:
        import commonmark
        from xhtml2pdf import pisa

        # Convert markdown to HTML using commonmark
        html_content = commonmark.commonmark(content)

        # Add basic HTML structure with the title
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333366; }}
                a {{ color: #0066cc; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                blockquote {{ border-left: 3px solid #ccc; padding-left: 10px; color: #666; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {html_content}
        </body>
        </html>
        """

        # Convert HTML to PDF using xhtml2pdf
        async with await anyio.open_file(filename, "w+b") as pdf_file:
            pisa_status = pisa.CreatePDF(html_doc, dest=pdf_file)

        if pisa_status.err:
            raise Exception("xhtml2pdf encountered errors")
        else:
            print(f"PDF generated successfully using commonmark + xhtml2pdf: {filename}")
            return filename

    except Exception as alt_error:
        error_msg = f"All PDF conversion methods failed. Last error: {alt_error!s}"
        print(error_msg)
        raise Exception(error_msg)


@env.task
async def main(
    topic: str = "Top 10 developing countries by GDP in 2024",
    prompts_file: File | str = "/root/prompts.yaml",
    budget: int = 2,
    remove_thinking_tags: bool = True,
    max_queries: int = 3,
    answer_model: str = "together_ai/deepseek-ai/DeepSeek-V3",
    planning_model: str = "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
    json_model: str = "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_sources: int = 20,
    summarization_model: str = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
) -> File:
    if isinstance(prompts_file, str):
        prompts_file = await File.from_local(prompts_file)

    answer = await research_topic(
        topic=topic,
        budget=budget,
        remove_thinking_tags=remove_thinking_tags,
        max_queries=max_queries,
        answer_model=answer_model,
        planning_model=planning_model,
        json_model=json_model,
        max_sources=max_sources,
        summarization_model=summarization_model,
        prompts_file=prompts_file,
    )

    filename = await generate_pdf(answer)
    return File(path=filename)


if __name__ == "__main__":
    # Local execution
    # flyte.init()
    # flyte.run(main)

    # Remote execution
    # TODO: Replace with your own Flyte config file path
    flyte.init_from_config("/Users/ytong/.flyte/config-k3d.yaml")
    run = flyte.run(main)
    print(run.url)
    run.wait(run)
