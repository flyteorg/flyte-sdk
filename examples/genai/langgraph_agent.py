# /// script
# requires-python = "==3.13"
# dependencies = [
#     "langgraph==1.0.7",
#     "langchain-core==0.3.28",
#     "flyte>=2.0.0b53",
# ]
# ///

"""
LangGraph Tool-Calling Agent Example

This example demonstrates how to build a simple tool-calling agent using LangGraph.
The agent is fully self-contained with no network calls - it uses a mock LLM that
simulates tool-calling behavior for demonstration purposes.

The agent can:
1. Add two numbers
2. Multiply two numbers
3. Get the current time (mocked)
"""

import asyncio
from typing import Annotated, Any, Callable, Coroutine, Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

import flyte
import flyte.report

agent_env = flyte.TaskEnvironment(
    name="langgraph-agent",
    image=(
        flyte.Image.from_debian_base().with_pip_packages(
            "langgraph==1.0.7",
            "langchain==1.2.7",
            "langchain-anthropic==1.3.1",
        )
    ),
    resources=flyte.Resources(cpu=1),
)


# --- Tool Definitions ---


@flyte.trace
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@flyte.trace
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@flyte.trace
async def get_current_time() -> str:
    """Get the current time."""
    return "2025-02-01 12:00:00 UTC"


# Tool registry mapping tool names to functions
TOOLS: dict[str, Callable[[Any], Coroutine[Any, Any, Any]]] = {
    "add": add,
    "multiply": multiply,
    "get_current_time": get_current_time,
}


# --- State Definition ---


class AgentState(TypedDict):
    """State for the agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]


# --- Mock LLM Node ---


def mock_llm(state: AgentState) -> dict:
    """
    Mock LLM that simulates tool-calling behavior.

    This is a simple rule-based mock that:
    - Detects math operations and calls appropriate tools
    - Detects time queries and calls get_current_time
    - Otherwise returns a direct response
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message is a tool result, generate a final response
    if isinstance(last_message, ToolMessage):
        # Collect all tool results from recent messages
        tool_results = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_results.append(f"{msg.name}: {msg.content}")
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                break

        response = AIMessage(content=f"Based on the calculations, the results are: {', '.join(reversed(tool_results))}")
        return {"messages": [response]}

    # Get the user's query
    query = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            query = msg.content.lower()

    # Simple pattern matching for tool calls
    tool_calls = []

    # Check for addition
    if "add" in query or "plus" in query or "sum" in query:
        # Extract numbers from query (simple regex-free approach)
        numbers = [int(word) for word in query.split() if word.isdigit()]
        if len(numbers) >= 2:
            tool_calls.append(
                {
                    "id": "call_add",
                    "name": "add",
                    "args": {"a": numbers[0], "b": numbers[1]},
                }
            )

    # Check for multiplication
    if "multiply" in query or "times" in query or "product" in query:
        numbers = [int(word) for word in query.split() if word.isdigit()]
        if len(numbers) >= 2:
            tool_calls.append(
                {
                    "id": "call_multiply",
                    "name": "multiply",
                    "args": {"a": numbers[0], "b": numbers[1]},
                }
            )

    # Check for time query
    if "time" in query or "clock" in query:
        tool_calls.append(
            {
                "id": "call_time",
                "name": "get_current_time",
                "args": {},
            }
        )

    if tool_calls:
        response = AIMessage(content="", tool_calls=tool_calls)
    else:
        response = AIMessage(
            content="I can help you with math operations (add, multiply) or tell you the current time. "
            "Try asking something like 'add 5 and 3' or 'what time is it?'"
        )

    return {"messages": [response]}


# --- Tool Executor Node ---


async def execute_tools(state: AgentState) -> dict:
    """Execute the tools requested by the LLM."""
    messages = state["messages"]
    last_message = messages[-1]

    tool_messages = []
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in TOOLS:
                result = await TOOLS[tool_name](**tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                )
            else:
                tool_messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {tool_name}",
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                )

    return {"messages": tool_messages}


# --- Routing Logic ---


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM made tool calls, execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # Otherwise, we're done
    return "end"


# --- Build the Graph ---


def build_agent_graph() -> StateGraph:
    """Build the LangGraph agent graph."""
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("llm", mock_llm)
    graph.add_node("tools", execute_tools)

    # Set entry point
    graph.set_entry_point("llm")

    # Add conditional edges
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # Tools always go back to LLM
    graph.add_edge("tools", "llm")

    return graph.compile()


# --- Report Helper ---


def generate_html_report(
    mermaid_diagram: str,
    query: str | None = None,
    messages: list[dict] | None = None,
) -> str:
    """
    Generate an HTML report for the agent run.

    Args:
        mermaid_diagram: Mermaid diagram string for the agent graph
        query: The user's query (optional)
        messages: List of message dictionaries from the conversation (optional)

    Returns:
        HTML string for the report
    """
    messages_html = ""
    if messages:
        messages_html = "".join(
            f'<div class="message {msg["type"]}">{msg["type"]}: {msg.get("content") or msg.get("tool_calls", "")}</div>'
            for msg in messages
        )

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LangGraph Agent Visualization</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                color: #e8e8e8;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{
                text-align: center;
                font-size: 2em;
                margin-bottom: 10px;
                color: #00d4ff;
            }}
            h2 {{
                color: #00d4ff;
                border-bottom: 2px solid #00d4ff;
                padding-bottom: 10px;
            }}
            .subtitle {{
                text-align: center;
                color: #888;
                margin-bottom: 30px;
            }}
            .graph-container {{
                background: rgba(30, 41, 59, 0.95);
                border-radius: 15px;
                padding: 30px;
                margin: 20px 0;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
                border: 1px solid rgba(0, 212, 255, 0.3);
            }}
            .mermaid {{
                display: flex;
                justify-content: center;
            }}
            .mermaid svg {{
                max-width: 100%;
            }}
            .results-container {{
                background: rgba(0, 0, 0, 0.3);
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
            }}
            .query-result {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #00d4ff;
            }}
            .query {{
                font-weight: bold;
                color: #00d4ff;
                margin-bottom: 10px;
            }}
            .message {{
                padding: 8px 12px;
                margin: 5px 0;
                border-radius: 8px;
                font-size: 0.9em;
            }}
            .message.HumanMessage {{
                background: rgba(0, 212, 255, 0.2);
            }}
            .message.AIMessage {{
                background: rgba(0, 255, 136, 0.2);
            }}
            .message.ToolMessage {{
                background: rgba(255, 193, 7, 0.2);
            }}
            .tools-list {{
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                justify-content: center;
                margin: 20px 0;
            }}
            .tool-card {{
                background: rgba(0, 212, 255, 0.1);
                border: 1px solid #00d4ff;
                border-radius: 10px;
                padding: 15px 20px;
                text-align: center;
            }}
            .tool-name {{
                font-weight: bold;
                color: #00d4ff;
            }}
            .tool-desc {{
                font-size: 0.85em;
                color: #aaa;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LangGraph Tool-Calling Agent</h1>
            <p class="subtitle">A self-contained agent demonstrating the ReAct pattern with mock LLM</p>

            <h2>Agent Graph Structure</h2>
            <div class="graph-container">
                <pre class="mermaid">
{mermaid_diagram}
                </pre>
            </div>

            <h2>Available Tools</h2>
            <div class="tools-list">
                <div class="tool-card">
                    <div class="tool-name">add(a, b)</div>
                    <div class="tool-desc">Add two numbers together</div>
                </div>
                <div class="tool-card">
                    <div class="tool-name">multiply(a, b)</div>
                    <div class="tool-desc">Multiply two numbers together</div>
                </div>
                <div class="tool-card">
                    <div class="tool-name">get_current_time()</div>
                    <div class="tool-desc">Get the current time</div>
                </div>
            </div>

            {
        f'''<h2>Query & Results</h2>
            <div class="results-container">
                <div class="query-result">
                    <div class="query">{query}</div>
                    {messages_html}
                </div>
            </div>'''
        if query and messages
        else ""
    }
        </div>

        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'base',
                themeVariables: {{
                    primaryColor: '#67e8f9',
                    primaryTextColor: '#0f172a',
                    primaryBorderColor: '#00d4ff',
                    lineColor: '#00d4ff',
                    secondaryColor: '#a5f3fc',
                    tertiaryColor: '#cffafe',
                    background: '#1e293b',
                    mainBkg: '#67e8f9',
                    nodeBorder: '#0891b2',
                    clusterBkg: '#1e293b',
                    clusterBorder: '#00d4ff',
                    titleColor: '#00d4ff',
                    edgeLabelBackground: '#1e293b',
                    nodeTextColor: '#0f172a',
                    textColor: '#e8e8e8',
                    labelTextColor: '#e8e8e8'
                }}
            }});
        </script>
    </body>
    </html>
    """


# --- Flyte Tasks ---


class AgentOutput(TypedDict):
    type: str
    content: str
    tool_calls: NotRequired[list[dict]]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


@agent_env.task(report=True)
async def run_agent(query: str) -> list[AgentOutput]:
    """
    Run the LangGraph agent with a query.

    Args:
        query: The user's question or request

    Returns:
        List of message dictionaries from the conversation
    """
    agent = build_agent_graph()

    # Run the agent
    print(f"Query: {query}")
    print("-" * 40)

    initial_state = {"messages": [HumanMessage(content=query)]}
    result = await agent.ainvoke(initial_state)

    # Convert messages to serializable format
    output_messages = []
    for msg in result["messages"]:
        msg_dict = {
            "type": msg.__class__.__name__,
            "content": msg.content,
        }
        if isinstance(msg, AIMessage) and msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
        if isinstance(msg, ToolMessage):
            msg_dict["tool_call_id"] = msg.tool_call_id
            msg_dict["name"] = msg.name
        output_messages.append(AgentOutput(**msg_dict))

    # Print the conversation
    for msg in output_messages:
        print(f"{msg['type']}: {msg.get('content', msg.get('tool_calls', ''))}")

    # Generate and publish the report
    mermaid_diagram = agent.get_graph().draw_mermaid()
    html_report = generate_html_report(mermaid_diagram, query=query, messages=output_messages)
    await flyte.report.replace.aio(html_report)
    await flyte.report.flush.aio()

    return output_messages


@agent_env.task(report=True)
async def main() -> list[list[AgentOutput]]:
    """
    Run multiple example queries through the agent.

    Returns:
        List of lists of AgentOutput objects for each query
    """

    agent: CompiledStateGraph = build_agent_graph()

    # Generate and publish top-level report with just the agent graph
    mermaid_diagram = agent.get_graph().draw_mermaid()
    html_report = generate_html_report(mermaid_diagram)
    await flyte.report.replace.aio(html_report)
    await flyte.report.flush.aio()

    queries = [
        "What is 15 plus 27?",
        "Can you multiply 6 times 8?",
        "What time is it?",
        "Add 100 and 200, then also multiply 5 times 10",
        "What is the result if you add 5 and 10, then multiply that by 3?",
        "Please give me the product of 12 and 11.",
        "Can you add zero to forty-two?",
        "What's the current time and also add 20 and 22?",
        "Multiply 7 by 9 and then tell me what time it is.",
        "Calculate 25 plus 17.",
        "Multiply the sum of 2 and 3 by 4.",
        "Add 50 to 25, then multiply the result by 2.",
        "What is the sum of 1000 and 2000?",
        "Multiply 13 and 8, then add 5.",
        "If you add 60 and 40, what do you get?",
        "Find the product of 9 and 11.",
        "Can you multiply 0 by 99?",
        "What is the time right now?",
        "After adding 7 and 3, multiply the result by 5.",
        "What is 23 times 15?",
        "Add 33 to 44 and then multiply that sum by 2.",
    ]

    coros = []
    for query in queries:
        print(f"\n{'=' * 50}")
        coros.append(run_agent(query))
        print(f"{'=' * 50}\n")

    results = await asyncio.gather(*coros)
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.with_runcontext(mode="remote").run(main)
    print(r.url)
