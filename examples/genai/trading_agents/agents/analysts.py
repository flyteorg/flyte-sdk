import asyncio

from agents.utils.utils import AgentState
from flyte_env import env
from langchain_core.messages import ToolMessage, convert_to_openai_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from tools import toolkit

import flyte

MAX_ITERATIONS = 5


async def run_chain_with_tools(
    type: str, state: AgentState, llm: str, system_message: str, tool_names: list[str]
) -> AgentState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK; another assistant with different tools"
                " will help where you left off. Execute what you can to make progress."
                " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}"
                " For your reference, the current date is {current_date}. The company we want to look at is {ticker}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join(tool_names))
    prompt = prompt.partial(current_date=state.trade_date)
    prompt = prompt.partial(ticker=state.company_of_interest)

    chain = prompt | ChatOpenAI(model=llm).bind_tools([getattr(toolkit, tool_name).func for tool_name in tool_names])

    iteration = 0
    while iteration < MAX_ITERATIONS:
        result = await chain.ainvoke(state.messages)
        state.messages.append(convert_to_openai_messages(result))

        if not result.tool_calls:
            # Final response — no tools required
            setattr(state, f"{type}_report", result.content or "")
            break

        # Run all tool calls in parallel
        async def run_single_tool(tool_call):
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = getattr(toolkit, tool_name, None)

            if not tool:
                return None

            content = await tool(**tool_args)
            return ToolMessage(tool_call_id=tool_call["id"], name=tool_name, content=content)

        with flyte.group(f"tool_calls_iteration_{iteration}"):
            tool_messages = await asyncio.gather(*[run_single_tool(tc) for tc in result.tool_calls])

        # Add valid tool results to state
        tool_messages = [msg for msg in tool_messages if msg]
        state.messages.extend(convert_to_openai_messages(tool_messages))

        iteration += 1
    else:
        # Reached iteration cap — optionally raise or log
        print(f"Max iterations ({MAX_ITERATIONS}) reached for {type}")

    return state


@env.task
async def create_fundamentals_analyst(llm: str, state: AgentState, online_tools: bool) -> AgentState:
    if online_tools:
        tools = [toolkit.get_fundamentals_openai]
    else:
        tools = [
            toolkit.get_finnhub_company_insider_sentiment,
            toolkit.get_finnhub_company_insider_transactions,
            toolkit.get_simfin_balance_sheet,
            toolkit.get_simfin_cashflow,
            toolkit.get_simfin_income_stmt,
        ]

    system_message = (
        "You are a researcher tasked with analyzing fundamental information over the past week about a company. "
        "Please write a comprehensive report of the company's fundamental information such as financial documents, "
        "company profile, basic company financials, company financial history, insider sentiment, and insider "
        "transactions to gain a full view of the company's "
        "fundamental information to inform traders. Make sure to include as much detail as possible. "
        "Do not simply state the trends are mixed, "
        "provide detailed and finegrained analysis and insights that may help traders make decisions. "
        "Make sure to append a Markdown table at the end of the report to organize key points in the report, "
        "organized and easy to read."
    )

    tool_names = [tool.func.__name__ for tool in tools]

    return await run_chain_with_tools("fundamentals", state, llm, system_message, tool_names)


@env.task
async def create_market_analyst(llm: str, state: AgentState, online_tools: bool) -> AgentState:
    if online_tools:
        tools = [
            toolkit.get_YFin_data_online,
            toolkit.get_stockstats_indicators_report_online,
        ]
    else:
        tools = [
            toolkit.get_YFin_data,
            toolkit.get_stockstats_indicators_report,
        ]

    system_message = (
        """You are a trading assistant tasked with analyzing financial markets.
Your role is to select the **most relevant indicators** for a given market condition
or trading strategy from the following list.
The goal is to choose up to **8 indicators** that provide complementary insights without redundancy.
Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator.
Usage: Identify trend direction and serve as dynamic support/resistance.
Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark.
Usage: Confirm overall market trend and identify golden/death cross setups.
Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average.
Usage: Capture quick shifts in momentum and potential entry points.
Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs.
Usage: Look for crossovers and divergence as signals of trend changes.
Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line.
Usage: Use crossovers with the MACD line to trigger trades.
Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal.
Usage: Visualize momentum strength and spot divergence early.
Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions.
Usage: Apply 70/30 thresholds and watch for divergence to signal reversals.
Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands.
Usage: Acts as a dynamic benchmark for price movement.
Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line.
Usage: Signals potential overbought conditions and breakout zones.
Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line.
Usage: Indicates potential oversold conditions.
Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility.
Usage: Set stop-loss levels and adjust position sizes based on current market volatility.
Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume.
Usage: Confirm trends by integrating price action with volume data.
Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information.
Avoid redundancy (e.g., do not select both rsi and stochrsi).
Also briefly explain why they are suitable for the given market context.
When you tool call, please use the exact name of the indicators provided above as they are defined parameters,
otherwise your call will fail.
Please make sure to call get_YFin_data first to retrieve the CSV that is needed to generate indicators.
Write a very detailed and nuanced report of the trends you observe.
Do not simply state the trends are mixed, provide detailed and finegrained analysis
and insights that may help traders make decisions."""
        """ Make sure to append a Markdown table at the end of the report to
        organize key points in the report, organized and easy to read."""
    )

    tool_names = [tool.func.__name__ for tool in tools]
    return await run_chain_with_tools("market", state, llm, system_message, tool_names)


@env.task
async def create_news_analyst(llm: str, state: AgentState, online_tools: bool) -> AgentState:
    if online_tools:
        tools = [
            toolkit.get_global_news_openai,
            toolkit.get_google_news,
        ]
    else:
        tools = [
            toolkit.get_finnhub_news,
            toolkit.get_reddit_news,
            toolkit.get_google_news,
        ]

    system_message = (
        "You are a news researcher tasked with analyzing recent news and trends over the past week. "
        "Please write a comprehensive report of the current state of the world that is relevant for "
        "trading and macroeconomics. "
        "Look at news from EODHD, and finnhub to be comprehensive. Do not simply state the trends are mixed, "
        "provide detailed and finegrained analysis and insights that may help traders make decisions."
        """ Make sure to append a Markdown table at the end of the report to organize key points in the report,
        organized and easy to read."""
    )

    tool_names = [tool.func.__name__ for tool in tools]

    return await run_chain_with_tools("news", state, llm, system_message, tool_names)


@env.task
async def create_social_media_analyst(llm: str, state: AgentState, online_tools: bool) -> AgentState:
    if online_tools:
        tools = [toolkit.get_stock_news_openai]
    else:
        tools = [toolkit.get_reddit_stock_info]

    system_message = (
        "You are a social media and company specific news researcher/analyst tasked with analyzing social media posts, "
        "recent company news, and public sentiment for a specific company over the past week. "
        "You will be given a company's name your objective is to write a comprehensive long report "
        "detailing your analysis, insights, and implications for traders and investors on this company's current state "
        "after looking at social media and what people are saying about that company, "
        "analyzing sentiment data of what people feel each day about the company, and looking at recent company news. "
        "Try to look at all sources possible from social media to sentiment to news. Do not simply state the trends "
        "are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
        """ Make sure to append a Makrdown table at the end of the report to organize key points in the report,
        organized and easy to read."""
    )

    tool_names = [tool.func.__name__ for tool in tools]

    return await run_chain_with_tools("sentiment", state, llm, system_message, tool_names)
