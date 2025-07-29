from typing import Any

from agents.utils.utils import AgentState, memory_init
from flyte_env import QUICK_THINKING_LLM, env
from langchain_openai import ChatOpenAI


async def _get_reflection_prompt() -> str:
    """Get the system prompt for reflection."""
    return """
You are an expert financial analyst tasked with reviewing trading decisions/analysis
and providing a comprehensive, step-by-step analysis.
Your goal is to deliver detailed insights into investment decisions and highlight opportunities for improvement,
adhering strictly to the following guidelines:

1. Reasoning:
- For each trading decision, determine whether it was correct or incorrect.
A correct decision results in an increase in returns, while an incorrect decision does the opposite.
- Analyze the contributing factors to each success or mistake. Consider:
    - Market intelligence.
    - Technical indicators.
    - Technical signals.
    - Price movement analysis.
    - Overall market data analysis
    - News analysis.
    - Social media and sentiment analysis.
    - Fundamental data analysis.
    - Weight the importance of each factor in the decision-making process.

2. Improvement:
- For any incorrect decisions, propose revisions to maximize returns.
- Provide a detailed list of corrective actions or improvements,
including specific recommendations (e.g., changing a decision from HOLD to BUY on a particular date).

3. Summary:
- Summarize the lessons learned from the successes and mistakes.
- Highlight how these lessons can be adapted for future trading scenarios
and draw connections between similar situations to apply the knowledge gained.

4. Query:
- Extract key insights from the summary into a concise sentence of no more than 1000 tokens.
- Ensure the condensed sentence captures the essence of the lessons and reasoning for easy reference.

Adhere strictly to these instructions, and ensure your output is detailed, accurate, and actionable.
You will also be given objective descriptions of the market from a price movements, technical indicator, news,
and sentiment perspective to provide more context for your analysis.
"""


async def _extract_current_situation(current_state: AgentState) -> str:
    """Extract the current market situation from the state."""
    curr_market_report = current_state.market_report
    curr_sentiment_report = current_state.sentiment_report
    curr_news_report = current_state.news_report
    curr_fundamentals_report = current_state.fundamentals_report

    return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"


async def _reflect_on_component(report: str, situation: str, returns) -> str:
    """Generate reflection for a component."""
    messages = [
        ("system", await _get_reflection_prompt()),
        (
            "human",
            f"Returns: {returns}\n\nAnalysis/Decision: {report}\n\nObjective Market Reports for Reference: {situation}",
        ),
    ]

    result = ChatOpenAI(model=QUICK_THINKING_LLM).invoke(messages).content
    return result


async def reflect_and_store(
    agent_name: str,
    history_or_decision: Any,
    current_state: AgentState,
    returns: str,
):
    situation = await _extract_current_situation(current_state)
    result = await _reflect_on_component(history_or_decision, situation, returns)

    memory = await memory_init(agent_name)
    memory.add_situations([(situation, result)])


@env.task
async def reflect_bull_researcher(state: AgentState, returns: str):
    await reflect_and_store("bull-researcher", state.investment_debate_state.bull_history, state, returns)


@env.task
async def reflect_bear_researcher(state: AgentState, returns: str):
    await reflect_and_store("bear-researcher", state.investment_debate_state.bear_history, state, returns)


@env.task
async def reflect_trader(state: AgentState, returns: str):
    await reflect_and_store("trader", state.trader_investment_plan, state, returns)


@env.task
async def reflect_research_manager(state: AgentState, returns: str):
    await reflect_and_store("research-manager", state.investment_debate_state.judge_decision, state, returns)


@env.task
async def reflect_risk_manager(state: AgentState, returns: str):
    await reflect_and_store("risk-manager", state.risk_debate_state.judge_decision, state, returns)
