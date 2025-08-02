# /// script
# requires-python = "==3.13"
# dependencies = [
#     "flyte==0.2.0b33",
#     "backtrader==1.9.78.123",
#     "boto3==1.39.9",
#     "chainlit==2.5.5",
#     "eodhd==1.0.32",
#     "feedparser==6.0.11",
#     "finnhub-python==2.4.23",
#     "langchain-anthropic==0.3.15",
#     "langchain-experimental==0.3.4",
#     "langchain-google-genai==2.1.5",
#     "langchain-openai==0.3.23",
#     "pandas==2.3.0",
#     "parsel==1.10.0",
#     "praw==7.8.1",
#     "pytz==2025.2",
#     "questionary==2.1.0",
#     "redis==6.2.0",
#     "requests==2.32.4",
#     "stockstats==0.6.5",
#     "tqdm==4.67.1",
#     "tushare==1.4.21",
#     "typing-extensions==4.14.0",
#     "yfinance==0.2.63",
# ]
# ///
import asyncio
from copy import deepcopy

import agents
import agents.analysts
from agents.managers import create_research_manager, create_risk_manager
from agents.researchers import create_bear_researcher, create_bull_researcher
from agents.risk_debators import (
    create_neutral_debator,
    create_risky_debator,
    create_safe_debator,
)
from agents.trader import create_trader
from agents.utils.utils import AgentState
from flyte_env import DEEP_THINKING_LLM, QUICK_THINKING_LLM, env, flyte
from langchain_openai import ChatOpenAI
from reflection import (
    reflect_bear_researcher,
    reflect_bull_researcher,
    reflect_research_manager,
    reflect_risk_manager,
    reflect_trader,
)


@env.task
async def process_signal(full_signal: str, QUICK_THINKING_LLM: str) -> str:
    """Process a full trading signal to extract the core decision."""

    messages = [
        {
            "role": "system",
            "content": """You are an efficient assistant designed to analyze paragraphs or
financial reports provided by a group of analysts.
Your task is to extract the investment decision: SELL, BUY, or HOLD.
Provide only the extracted decision (SELL, BUY, or HOLD) as your output,
without adding any additional text or information.""",
        },
        {"role": "human", "content": full_signal},
    ]

    return ChatOpenAI(model=QUICK_THINKING_LLM).invoke(messages).content


async def run_analyst(analyst_name, state, online_tools):
    # Create a copy of the state for isolation
    run_fn = getattr(agents.analysts, f"create_{analyst_name}_analyst")

    # Run the analyst's chain
    result_state = await run_fn(QUICK_THINKING_LLM, state, online_tools)

    # Determine the report key
    report_key = "sentiment_report" if analyst_name == "social_media" else f"{analyst_name}_report"
    report_value = getattr(result_state, report_key)

    return result_state.messages[1:], report_key, report_value


@env.task
async def main(
    selected_analysts: list[str] = [
        "market",
        "fundamentals",
        "news",
        "social_media",
    ],
    max_debate_rounds: int = 1,
    max_risk_discuss_rounds: int = 1,
    online_tools: bool = True,  # offline dataset isn't available yet
    company_name: str = "NVDA",
    trade_date: str = "2024-05-12",
) -> tuple[str, AgentState]:
    if not selected_analysts:
        raise ValueError(
            "No analysts selected. Please select at least one analyst from market, fundamentals, news, or social_media."
        )

    state = AgentState(
        messages=[{"role": "human", "content": company_name}],
        company_of_interest=company_name,
        trade_date=str(trade_date),
    )

    # Run all analysts concurrently
    results = await asyncio.gather(
        *[run_analyst(analyst, deepcopy(state), online_tools) for analyst in selected_analysts]
    )

    # Flatten and append all resulting messages into the shared state
    for messages, report_attr, report in results:
        state.messages.extend(messages)
        setattr(state, report_attr, report)

    # Bull/Bear debate loop
    state = await create_bull_researcher(QUICK_THINKING_LLM, state)  # Start with bull
    while state.investment_debate_state.count < 2 * max_debate_rounds:
        current = state.investment_debate_state.current_response
        if current.startswith("Bull"):
            state = await create_bear_researcher(QUICK_THINKING_LLM, state)
        else:
            state = await create_bull_researcher(QUICK_THINKING_LLM, state)

    state = await create_research_manager(DEEP_THINKING_LLM, state)
    state = await create_trader(QUICK_THINKING_LLM, state)

    # Risk debate loop
    state = await create_risky_debator(QUICK_THINKING_LLM, state)  # Start with risky
    while state.risk_debate_state.count < 3 * max_risk_discuss_rounds:
        speaker = state.risk_debate_state.latest_speaker
        if speaker == "Risky":
            state = await create_safe_debator(QUICK_THINKING_LLM, state)
        elif speaker == "Safe":
            state = await create_neutral_debator(QUICK_THINKING_LLM, state)
        else:
            state = await create_risky_debator(QUICK_THINKING_LLM, state)

    state = await create_risk_manager(DEEP_THINKING_LLM, state)
    decision = await process_signal(state.final_trade_decision, QUICK_THINKING_LLM)

    return decision, state


@env.task
async def reflect_and_store(state: AgentState, returns: str) -> str:
    await asyncio.gather(
        reflect_bear_researcher(state, returns),
        reflect_bull_researcher(state, returns),
        reflect_trader(state, returns),
        reflect_risk_manager(state, returns),
        reflect_research_manager(state, returns),
    )

    return "Reflection completed."


# Run the reflection task after the main function
@env.task(cache="disable")
async def reflect_on_decisions(
    returns: str,
    selected_analysts: list[str] = [
        "market",
        "fundamentals",
        "news",
        "social_media",
    ],
    max_debate_rounds: int = 1,
    max_risk_discuss_rounds: int = 1,
    online_tools: bool = True,
    company_name: str = "NVDA",
    trade_date: str = "2024-05-12",
) -> str:
    _, state = await main(
        selected_analysts,
        max_debate_rounds,
        max_risk_discuss_rounds,
        online_tools,
        company_name,
        trade_date,
    )

    return await reflect_and_store(state, returns)


if __name__ == "__main__":
    flyte.init_from_config("config.yaml")
    run = flyte.run(main)
    print(run.url)

    # run = flyte.run(reflect_on_decisions, "+3.2% gain over 5 days")
    # print(run.url)
