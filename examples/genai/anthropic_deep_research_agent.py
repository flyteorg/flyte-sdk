"""Deep Research Agent using Anthropic Claude with Flyte.

This example demonstrates a multi-agent architecture for deep research on any
topic. The orchestrator agent decomposes a user query into smaller research
questions, fans them out to parallel research sub-agents (each with web search,
Python code execution, and report-building tools), then synthesizes the answers
into a comprehensive HTML report.

A critic agent evaluates the final output for accuracy, comprehensibility, and
groundedness in the sources. If the output is not satisfactory, the agent
loops back to do targeted follow-up research and re-synthesize, up to a
configurable maximum number of refinement iterations.

Architecture:
    1. Orchestrator agent: decomposes the query → list of sub-questions
    2. Research sub-agents (parallel): each answers one sub-question using:
       - web_search (Tavily API - general web search)
       - execute_python (sandboxed code execution via a dedicated TaskEnvironment
         for lightweight computation, data analysis, or calculations)
       - build_report_section (builds an HTML fragment for the sub-question)
    3. Synthesis agent: merges all sub-answers into a final rich HTML report
    4. Critic agent: evaluates quality → pass or provide follow-up questions
    5. Refinement loop (up to max_refinements): targeted research on gaps,
       then re-synthesis and re-evaluation
"""

import asyncio
import json
import textwrap

from flyteplugins.anthropic import function_tool, run_agent

import flyte
import flyte.report
from flyte._image import DIST_FOLDER, PythonWheels

# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

sandbox_env = flyte.TaskEnvironment(
    "python-sandbox",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=(
        flyte.Image.from_debian_base(python_version=(3, 13)).with_pip_packages(
            "numpy", "pandas", "scikit-learn", "matplotlib"
        )
    ),
)

agent_env = flyte.TaskEnvironment(
    "deep-research-agent",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=[
        flyte.Secret(key="niels-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="niels-tavily-api-key", as_env_var="TAVILY_API_KEY"),
    ],
    image=(
        flyte.Image.from_debian_base(python_version=(3, 13))
        .clone(
            addl_layer=PythonWheels(wheel_dir=DIST_FOLDER, package_name="flyteplugins-anthropic", pre=True),
            extendable=True,
        )
        .with_pip_packages("tavily-python", "markdown")
    ),
    depends_on=[sandbox_env],
)

# ---------------------------------------------------------------------------
# Token-budget constants (character limits, ~4 chars ≈ 1 token)
# ---------------------------------------------------------------------------

MAX_SEARCH_SNIPPET_CHARS = 400  # per search result content snippet
MAX_SEARCH_RESULTS = 3  # default results per web search call
MAX_EXEC_OUTPUT_CHARS = 3000  # stdout/stderr cap from execute_python
MAX_SUMMARY_CHARS_FOR_CRITIC = 600  # per sub-result when evaluating quality
MAX_SUMMARY_CHARS_FOR_SYNTHESIS = 1500  # per sub-result when synthesizing


def _truncate(text: str, max_chars: int, suffix: str = "\n... [truncated]") -> str:
    """Truncate text to max_chars, appending a suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@flyte.trace
async def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """Search the web for information on any topic.

    Uses the Tavily API to perform an advanced web search. Returns a compact
    JSON string containing the search results with titles, URLs, and content
    snippets (each capped to ~400 chars to keep token usage low).
    """
    import os

    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
    )
    results = []
    for r in response.get("results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": _truncate(r.get("content", ""), MAX_SEARCH_SNIPPET_CHARS),
            }
        )
    # Compact JSON (no indent) to save tokens
    return json.dumps(results, separators=(",", ":"))


@sandbox_env.task
async def execute_python(code: str) -> str:
    """Execute Python code in a sandboxed environment and return the output.

    The sandbox has numpy, pandas, scikit-learn, and matplotlib available.
    Use this for lightweight computation: calculations, data analysis,
    number crunching, conversions, or validating quantitative claims from
    research. The code must print its results to stdout.

    If the code creates matplotlib figures, a short description of each
    figure is appended. The full plot images are NOT included in the output
    to keep token counts low.

    Returns the captured stdout/stderr (capped at ~3000 chars).
    """
    import io
    import sys
    import traceback

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured = io.StringIO()
    sys.stdout = captured
    sys.stderr = captured

    try:
        exec_globals: dict = {}
        exec(code, exec_globals)
    except Exception:
        traceback.print_exc(file=captured)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    output = captured.getvalue() or "(no output)"
    output = _truncate(output, MAX_EXEC_OUTPUT_CHARS)

    # Note any open matplotlib figures (description only — no base64)
    try:
        import matplotlib.pyplot as plt

        fig_nums = plt.get_fignums()
        for i, num in enumerate(fig_nums, start=1):
            fig = plt.figure(num)
            w, h = fig.get_size_inches()
            n_axes = len(fig.axes)
            output += f"\n[Generated Figure {i}: {w:.0f}x{h:.0f} in, {n_axes} axes]"
        if fig_nums:
            plt.close("all")
    except Exception:
        pass  # matplotlib may not have been used

    return output


# Shared CSS for report sections (used by both standalone section reports
# and the final assembled report).
_SECTION_CSS = """\
    body {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
        min-height: 100vh;
        padding: 40px 20px;
    }
    .report-section {
        max-width: 900px;
        margin: 0 auto 24px;
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 28px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .report-section h3 {
        font-size: 1.3em;
        color: #bb86fc;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(187, 134, 252, 0.2);
    }
    .section-content { line-height: 1.7; color: #ccc; }
    .section-content p { margin-bottom: 12px; }
    .section-content h4, .section-content h5 { color: #ce93d8; margin: 16px 0 8px; }
    .section-content ul, .section-content ol { margin: 8px 0 12px 20px; }
    .section-content li { margin-bottom: 4px; }
    .section-content strong { color: #e0e0e0; }
    .section-content em { color: #b0bec5; }
    .section-content blockquote {
        border-left: 3px solid #7b2ff7;
        padding: 8px 16px;
        margin: 12px 0;
        background: rgba(123, 47, 247, 0.08);
        color: #b0bec5;
    }
    .section-content img {
        max-width: 100%;
        border-radius: 8px;
        margin: 16px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .section-content table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
    }
    .section-content th, .section-content td {
        border: 1px solid rgba(255,255,255,0.15);
        padding: 8px 12px;
        text-align: left;
    }
    .section-content th {
        background: rgba(123, 47, 247, 0.15);
        color: #bb86fc;
    }
    .section-content code {
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 0.9em;
    }
    .section-content > code,
    .section-content p code {
        background: rgba(0,0,0,0.35);
        padding: 2px 6px;
        border-radius: 4px;
        color: #a5d6a7;
    }
    .section-content pre {
        background: rgba(0,0,0,0.4);
        border-radius: 8px;
        padding: 16px;
        overflow-x: auto;
        font-size: 0.85em;
        line-height: 1.5;
        margin: 12px 0;
    }
    .section-content pre code {
        background: none;
        padding: 0;
        color: #a5d6a7;
    }
    .sources {
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px dashed rgba(255,255,255,0.1);
    }
    .sources h4 { color: #03dac6; font-size: 0.95em; margin-bottom: 8px; }
    .sources ul { list-style: none; padding: 0; margin: 0; }
    .sources li { margin-bottom: 4px; }
    .sources a {
        color: #80cbc4;
        text-decoration: none;
        font-size: 0.9em;
        word-break: break-all;
    }
    .sources a:hover { text-decoration: underline; }
    .code-block {
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px dashed rgba(255,255,255,0.1);
    }
    .code-block h4 { color: #cf6679; font-size: 0.95em; margin-bottom: 8px; }
    .code-block pre {
        background: rgba(0,0,0,0.4);
        border-radius: 8px;
        padding: 16px;
        overflow-x: auto;
        font-size: 0.85em;
        line-height: 1.5;
    }
    .code-block code {
        color: #a5d6a7;
        font-family: 'Fira Code', 'Consolas', monospace;
    }
    .code-block img {
        max-width: 100%;
        border-radius: 8px;
        margin: 12px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
"""


@agent_env.task(report=True)
async def build_report_section(
    section_title: str,
    section_content: str,
    sources: str = "",
    code_snippets: str = "",
) -> str:
    """Build an HTML report section for one research sub-question.

    Takes the research findings for a single sub-question and formats them
    as a styled HTML fragment. The fragment will later be assembled into the
    final report.

    Args:
        section_title: Title of this research sub-question.
        section_content: Markdown-formatted answer text. May include code
            blocks (triple-backtick fenced), tables, and standard markdown.
        sources: Comma-separated list of source URLs referenced.
        code_snippets: Key Python code and output as markdown fenced code
            blocks. Keep concise — only the most important snippets.
    """
    import markdown as md

    md_extensions = ["fenced_code", "tables", "nl2br"]
    content_html = md.markdown(section_content, extensions=md_extensions)

    sources_html = ""
    if sources.strip():
        source_items = [
            f'<li><a href="{s.strip()}" target="_blank">{s.strip()}</a></li>' for s in sources.split(",") if s.strip()
        ]
        sources_html = f"""
        <div class="sources">
            <h4>Sources</h4>
            <ul>{"".join(source_items)}</ul>
        </div>"""

    code_html = ""
    if code_snippets.strip():
        code_html = f"""
        <div class="code-block">
            <h4>Code / Output</h4>
            {md.markdown(code_snippets, extensions=md_extensions)}
        </div>"""

    section_html = f"""
    <div class="report-section">
        <h3>{section_title}</h3>
        <div class="section-content">{content_html}</div>
        {sources_html}
        {code_html}
    </div>"""

    # Render this section as a standalone report
    await flyte.report.replace.aio(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{section_title}</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <style>{_SECTION_CSS}</style>
    </head>
    <body>
        {section_html}
        <script>hljs.highlightAll();</script>
    </body>
    </html>""")
    await flyte.report.flush.aio()

    return section_html


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert researcher. You have access to web search, Python code
    execution, and a report-section builder.

    Your goal is to thoroughly answer the research question you are given.

    **Token-efficiency rules** (the conversation history has limited space):
    - Make at most 2-3 web searches. Formulate precise queries.
    - Only use execute_python if the question requires calculation, data
      analysis, or quantitative validation. Keep code short and focused.
    - Do NOT repeat large tool outputs in your responses. Summarize instead.

    Strategy:
    1. Use web_search to find relevant, authoritative information.
    2. Optionally use execute_python for calculations, conversions, data
       analysis, or to validate quantitative claims. Print only key results.
    3. Once you have gathered enough information, call build_report_section:
       - section_title: clear title for this sub-question.
       - section_content: answer in **markdown** (2-4 paragraphs). Use
         headings, bold, bullet lists, tables, and fenced code blocks as
         appropriate.
       - sources: comma-separated source URLs.
       - code_snippets: any key Python code and its output as a markdown
         fenced code block. Keep it concise. Omit if no code was run.
    4. Return a concise (1-2 paragraph) markdown summary as your final text
       response. Do NOT repeat the full section_content.
""")

research_tools = [
    function_tool(web_search),
    function_tool(execute_python),
    function_tool(build_report_section),
]


@agent_env.task
async def run_research_sub_agent(question: str, name: str) -> dict:
    """Run a single research sub-agent for one sub-question."""
    with flyte.group(f"researcher-{name}"):
        summary = await run_agent(
            prompt=f"Research the following question thoroughly:\n\n{question}",
            tools=research_tools,
            system=RESEARCH_SYSTEM_PROMPT,
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            max_iterations=10,
        )
        return {"name": name, "question": question, "summary": summary}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert research planner. Given a broad research query, your
    job is to decompose it into 3-5 specific, self-contained research
    sub-questions that, when answered together, will comprehensively address
    the original query.

    Return ONLY a JSON array of objects, where each object has:
    - "name": a short, human-readable kebab-case identifier (2-4 words,
      e.g. "cost-comparison", "historical-timeline", "environmental-impact")
    - "question": the full research sub-question text

    Do not include any other text or explanation.

    Example output:
    [{"name": "current-market-size", "question": "What is the current global market size for electric vehicles?"},
     {"name": "cost-vs-gas", "question": "How do the total ownership costs of EVs compare to gasoline vehicles?"},
     {"name": "policy-incentives", "question": "What government policies and incentives are driving EV adoption?"}]
""")

SYNTHESIS_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert research synthesizer. You will be given a collection of
    research summaries answering sub-questions of a larger research query.

    Your job is to synthesize these into a single coherent executive summary
    in **markdown format** that addresses the original query. The summary
    should:
    1. Start with a high-level overview (2-3 sentences)
    2. Highlight the most important findings across all sub-questions
    3. Identify connections and themes across the sub-topics
    4. Note any contradictions or open questions
    5. End with a brief conclusion and future directions

    Use markdown formatting: bold for emphasis, bullet lists for key points,
    and tables for comparisons when appropriate. Write in clear, well-
    organized prose suitable for a knowledgeable general audience.
    Keep it to 3-5 paragraphs.
""")

CRITIC_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a rigorous research reviewer. You evaluate research reports for
    quality along three dimensions:

    1. **Accuracy**: Are the factual claims correct? Are names, dates,
       statistics, and quantitative results accurately reported?
    2. **Comprehensibility**: Is the report clearly written? Does it flow
       logically? Would a knowledgeable reader find it easy to follow?
    3. **Groundedness**: Are claims supported by cited sources? Are there
       unsupported assertions or hallucinated details?

    You MUST respond with ONLY a JSON object (no other text) in this format:
    {
        "satisfactory": true/false,
        "accuracy_score": 1-5,
        "comprehensibility_score": 1-5,
        "groundedness_score": 1-5,
        "critique": "Brief explanation of weaknesses (empty string if satisfactory)",
        "follow_up_questions": ["specific question to address gap 1", ...]
    }

    Set "satisfactory" to true ONLY if ALL scores are >= 4.
    When "satisfactory" is true, "follow_up_questions" should be an empty list.
    When "satisfactory" is false, provide 1-3 targeted follow-up questions that
    would address the most important gaps.
""")

REFINEMENT_SYNTHESIS_PROMPT = textwrap.dedent("""\
    You are an expert research synthesizer performing a refinement pass.
    You will receive:
    - The original research query
    - The previous executive summary (which was found lacking)
    - A critique explaining what was weak
    - New research summaries that address the identified gaps

    Your job is to produce an IMPROVED executive summary in **markdown format**
    that:
    1. Retains accurate content from the previous summary
    2. Integrates the new findings to address the critique
    3. Fixes any accuracy, comprehensibility, or groundedness issues
    4. Maintains clear, well-organized prose suitable for a knowledgeable
       general audience

    Use markdown formatting: bold for emphasis, bullet lists for key points,
    and tables for comparisons when appropriate. Keep it to 3-5 paragraphs.
""")


def _build_final_report(
    query: str,
    executive_summary: str,
    section_htmls: list[str],
    sub_results: list[dict],
) -> str:
    """Assemble the final HTML research report."""
    import markdown as md

    sections_combined = "\n".join(section_htmls)
    md_extensions = ["fenced_code", "tables", "nl2br"]
    summary_html = md.markdown(executive_summary, extensions=md_extensions)

    return textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deep Research Report</title>
        <link rel="stylesheet"
              href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            {_SECTION_CSS}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 30px;
                background: rgba(255,255,255,0.05);
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .header h1 {{
                font-size: 2.2em;
                background: linear-gradient(90deg, #00d2ff, #7b2ff7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }}
            .header .query {{
                font-size: 1.1em;
                color: #aaa;
                font-style: italic;
            }}
            .executive-summary {{
                background: rgba(0, 210, 255, 0.08);
                border-left: 4px solid #00d2ff;
                border-radius: 0 12px 12px 0;
                padding: 24px 28px;
                margin-bottom: 36px;
                line-height: 1.7;
                color: #ccc;
            }}
            .executive-summary h2 {{
                color: #00d2ff;
                margin-bottom: 16px;
                font-size: 1.4em;
            }}
            .executive-summary p {{ margin-bottom: 12px; }}
            .executive-summary strong {{ color: #e0e0e0; }}
            .executive-summary ul, .executive-summary ol {{ margin: 8px 0 12px 20px; }}
            .executive-summary li {{ margin-bottom: 4px; }}
            .executive-summary code {{
                font-family: 'Fira Code', 'Consolas', monospace;
                background: rgba(0,0,0,0.3);
                padding: 2px 6px;
                border-radius: 4px;
                color: #a5d6a7;
                font-size: 0.9em;
            }}
            .executive-summary pre {{
                background: rgba(0,0,0,0.4);
                border-radius: 8px;
                padding: 16px;
                overflow-x: auto;
                font-size: 0.85em;
                line-height: 1.5;
                margin: 12px 0;
            }}
            .executive-summary pre code {{
                background: none;
                padding: 0;
            }}
            .executive-summary table {{
                width: 100%;
                border-collapse: collapse;
                margin: 16px 0;
            }}
            .executive-summary th, .executive-summary td {{
                border: 1px solid rgba(255,255,255,0.15);
                padding: 8px 12px;
                text-align: left;
            }}
            .executive-summary th {{
                background: rgba(0, 210, 255, 0.12);
                color: #00d2ff;
            }}
            .executive-summary img {{
                max-width: 100%;
                border-radius: 8px;
                margin: 16px 0;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .report-section:hover {{
                border-color: rgba(123, 47, 247, 0.4);
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 0.85em;
                margin-top: 40px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Deep Research Report</h1>
                <div class="query">"{query}"</div>
            </div>

            <div class="executive-summary">
                <h2>Executive Summary</h2>
                {summary_html}
            </div>

            {sections_combined}

            <div class="footer">
                Generated by Deep Research Agent &middot; Powered by Anthropic Claude &amp; Flyte
            </div>
        </div>
        <script>hljs.highlightAll();</script>
    </body>
    </html>""")


@agent_env.task
async def evaluate_quality(
    query: str,
    executive_summary: str,
    sub_results: list[dict],
) -> dict:
    """Run the critic agent to evaluate the research output quality.

    Sub-result summaries are truncated to keep the prompt within a reasonable
    token budget.

    Returns a dict with keys: satisfactory, accuracy_score,
    comprehensibility_score, groundedness_score, critique, follow_up_questions.
    """
    sub_summaries = "\n\n".join(
        f"### {r['question']}\n{_truncate(r['summary'], MAX_SUMMARY_CHARS_FOR_CRITIC)}" for r in sub_results
    )
    raw = await run_agent(
        prompt=(
            f"Original query: {query}\n\n"
            f"Executive summary:\n{executive_summary}\n\n"
            f"Research sub-sections (abbreviated):\n{sub_summaries}\n\n"
            f"Evaluate this research output. Respond with ONLY a JSON object."
        ),
        system=CRITIC_SYSTEM_PROMPT,
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        max_iterations=1,
    )
    try:
        evaluation = json.loads(raw)
    except json.JSONDecodeError:
        # If the critic didn't return valid JSON, treat as satisfactory
        # to avoid infinite loops
        evaluation = {
            "satisfactory": True,
            "accuracy_score": 4,
            "comprehensibility_score": 4,
            "groundedness_score": 4,
            "critique": "",
            "follow_up_questions": [],
        }
    return evaluation


def _build_section_htmls(sub_results: list[dict]) -> list[str]:
    """Build fallback HTML sections from sub-agent results."""
    import markdown as md

    md_extensions = ["fenced_code", "tables", "nl2br"]
    section_htmls = []
    for result in sub_results:
        content_html = md.markdown(result["summary"], extensions=md_extensions)
        fallback_section = f"""
        <div class="report-section">
            <h3>{result["question"]}</h3>
            <div class="section-content">
                {content_html}
            </div>
        </div>"""
        section_htmls.append(fallback_section)
    return section_htmls


@agent_env.task
async def synthesize_summaries(
    query: str,
    sub_results: list[dict],
    system_prompt: str = "",
    previous_summary: str = "",
    critique: str = "",
) -> str:
    """Synthesize research sub-results into a single executive summary.

    Sub-result summaries are truncated to stay within token budget.
    When previous_summary and critique are provided, this performs a
    refinement pass that improves the previous summary based on the critique
    and the new research findings.
    """
    if previous_summary:
        # Refinement mode
        new_summaries = "\n\n".join(
            f"### Follow-up: {r['question']}\n{_truncate(r['summary'], MAX_SUMMARY_CHARS_FOR_SYNTHESIS)}"
            for r in sub_results
        )
        prompt = (
            f"Original query: {query}\n\n"
            f"Previous executive summary:\n{previous_summary}\n\n"
            f"Critique of previous summary:\n{critique}\n\n"
            f"New research addressing the gaps:\n{new_summaries}\n\n"
            f"Please produce an improved executive summary."
        )
    else:
        # Initial synthesis
        summaries_text = "\n\n".join(
            f"### Sub-question: {r['question']}\n{_truncate(r['summary'], MAX_SUMMARY_CHARS_FOR_SYNTHESIS)}"
            for r in sub_results
        )
        prompt = (
            f"Original query: {query}\n\n"
            f"Research summaries:\n{summaries_text}\n\n"
            f"Please synthesize these into a coherent executive summary."
        )

    return await run_agent(
        prompt=prompt,
        system=system_prompt or SYNTHESIS_SYSTEM_PROMPT,
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        max_iterations=1,
    )


@agent_env.task
async def decompose_query(query: str) -> list[dict[str, str]]:
    """Decompose a research query into named sub-questions.

    Uses Claude to break a broad research query into 3-5 specific,
    self-contained sub-questions, each with a short kebab-case name and
    the full question text.

    Returns a list of dicts with "name" and "question" keys.
    """
    raw = await run_agent(
        prompt=(f"Decompose this research query into 3-5 specific sub-questions:\n\n{query}"),
        system=DECOMPOSE_SYSTEM_PROMPT,
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        max_iterations=1,
    )
    # Parse the JSON array of {name, question} objects
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            parsed = [parsed]
        # Normalise: accept both list-of-dicts and list-of-strings
        sub_questions: list[dict[str, str]] = []
        for idx, item in enumerate(parsed, start=1):
            if isinstance(item, dict):
                sub_questions.append(
                    {
                        "name": item.get("name", f"sub-q-{idx}"),
                        "question": item.get("question", str(item)),
                    }
                )
            else:
                sub_questions.append(
                    {
                        "name": f"sub-q-{idx}",
                        "question": str(item),
                    }
                )
    except json.JSONDecodeError:
        # Fallback: treat the response as a single question
        sub_questions = [{"name": "sub-q-1", "question": raw}]

    return sub_questions


@agent_env.task(report=True)
async def deep_research_agent(query: str, max_refinements: int = 2) -> str:
    """Run the deep research agent.

    This orchestrator:
    1. Decomposes the query into sub-questions using Claude
    2. Fans out parallel research sub-agents (each with web search,
       optional code execution, and report-building tools)
    3. Synthesizes all findings into a final rich HTML report
    4. Evaluates the output for accuracy, comprehensibility, and groundedness
    5. If not satisfactory, refines via targeted follow-up research (up to
       max_refinements iterations)
    """

    # --- Step 1: Decompose the query into sub-questions ---
    sub_questions = await decompose_query(query)

    print(f"Decomposed into {len(sub_questions)} sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. [{sq['name']}] {sq['question']}")

    # --- Step 2: Fan out parallel research sub-agents ---
    with flyte.group("parallel-research"):
        tasks = [run_research_sub_agent(sq["question"], sq["name"]) for sq in sub_questions]
        sub_results = list(await asyncio.gather(*tasks))

    # --- Step 3: Synthesize into executive summary ---
    with flyte.group("synthesize"):
        executive_summary = await synthesize_summaries(query, sub_results)

    # --- Step 4: Evaluate & refine loop ---
    # Keep a sliding window of the most recent sub_results for the critic
    # to avoid token explosion as refinement rounds accumulate results.
    latest_sub_results = sub_results
    for refinement_round in range(max_refinements):
        with flyte.group(f"evaluate-{refinement_round}"):
            evaluation = await evaluate_quality(
                query,
                executive_summary,
                latest_sub_results,
            )

        print(
            f"Evaluation round {refinement_round + 1}: "
            f"satisfactory={evaluation.get('satisfactory')}, "
            f"accuracy={evaluation.get('accuracy_score')}, "
            f"comprehensibility={evaluation.get('comprehensibility_score')}, "
            f"groundedness={evaluation.get('groundedness_score')}"
        )

        if evaluation.get("satisfactory", True):
            print("Quality check passed — finalizing report.")
            break

        critique = evaluation.get("critique", "")
        follow_ups = evaluation.get("follow_up_questions", [])

        if not follow_ups:
            print("Critic found issues but no follow-up questions — finalizing.")
            break

        print(f"Quality check failed (round {refinement_round + 1}). Critique: {critique}")
        print(f"Follow-up questions: {follow_ups}")

        # --- Step 4a: Targeted follow-up research ---
        with flyte.group(f"refine-research-{refinement_round}"):
            refinement_tasks = [
                run_research_sub_agent(
                    q,
                    f"followup-r{refinement_round}-{idx}",
                )
                for idx, q in enumerate(follow_ups, start=1)
            ]
            new_results = list(await asyncio.gather(*refinement_tasks))
            sub_results.extend(new_results)
            # The critic only sees the latest round's results (not all
            # accumulated history) to keep the evaluation prompt lean.
            latest_sub_results = new_results

        # --- Step 4b: Re-synthesize with the new findings ---
        with flyte.group(f"refine-synthesize-{refinement_round}"):
            executive_summary = await synthesize_summaries(
                query,
                new_results,
                system_prompt=REFINEMENT_SYNTHESIS_PROMPT,
                previous_summary=executive_summary,
                critique=critique,
            )
    else:
        print(f"Reached maximum refinement iterations ({max_refinements}). Finalizing with best available output.")

    # --- Step 5: Build and publish final HTML report ---
    section_htmls = _build_section_htmls(sub_results)
    html_report = _build_final_report(
        query=query,
        executive_summary=executive_summary,
        section_htmls=section_htmls,
        sub_results=sub_results,
    )
    await flyte.report.replace.aio(html_report)
    await flyte.report.flush.aio()

    return executive_summary


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        deep_research_agent,
        query=(
            "What are the economic, environmental, and geopolitical impacts "
            "of the global transition to renewable energy? Compare solar, "
            "wind, and nuclear in terms of cost, scalability, and adoption "
            "barriers as of 2026."
        ),
    )
    print(f"View at: {run.url}")
