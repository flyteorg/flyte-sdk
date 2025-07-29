from agents.utils.utils import AgentState, InvestmentDebateState, memory_init
from flyte_env import env
from langchain_openai import ChatOpenAI


@env.task
async def create_bear_researcher(llm: str, state: AgentState) -> AgentState:
    investment_debate_state = state.investment_debate_state
    history = investment_debate_state.history
    bear_history = investment_debate_state.bear_history

    current_response = investment_debate_state.current_response
    market_research_report = state.market_report
    sentiment_report = state.sentiment_report
    news_report = state.news_report
    fundamentals_report = state.fundamentals_report

    memory = await memory_init(name="bear-researcher")

    curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
    past_memories = memory.get_memories(curr_situation, n_matches=2)

    past_memory_str = ""
    for rec in past_memories:
        past_memory_str += rec["recommendation"] + "\n\n"

    prompt = f"""You are a Bear Analyst making the case against investing in the stock.
Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators.
Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability,
or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation,
or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning,
exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points
and debating effectively rather than simply listing facts.

Resources available:

Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate
that demonstrates the risks and weaknesses of investing in the stock.
You must also address reflections and learn from lessons and mistakes you made in the past.
"""

    response = ChatOpenAI(model=llm).invoke(prompt)

    argument = f"Bear Analyst: {response.content}"

    new_investment_debate_state = InvestmentDebateState(
        history=history + "\n" + argument,
        bear_history=bear_history + "\n" + argument,
        bull_history=investment_debate_state.bull_history,
        current_response=argument,
        count=investment_debate_state.count + 1,
    )

    state.investment_debate_state = new_investment_debate_state
    return state


@env.task
async def create_bull_researcher(llm: str, state: AgentState) -> AgentState:
    investment_debate_state = state.investment_debate_state
    history = investment_debate_state.history
    bull_history = investment_debate_state.bull_history

    current_response = investment_debate_state.current_response
    market_research_report = state.market_report
    sentiment_report = state.sentiment_report
    news_report = state.news_report
    fundamentals_report = state.fundamentals_report

    memory = await memory_init(name="bull-researcher")

    curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
    past_memories = memory.get_memories(curr_situation, n_matches=2)

    past_memory_str = ""
    for rec in past_memories:
        past_memory_str += rec["recommendation"] + "\n\n"

    prompt = f"""You are a Bull Analyst advocating for investing in the stock.
Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages,
and positive market indicators.
Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing
concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points
and debating effectively rather than just listing data.

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate
that demonstrates the strengths of the bull position.
You must also address reflections and learn from lessons and mistakes you made in the past.
"""

    response = ChatOpenAI(model=llm).invoke(prompt)

    argument = f"Bull Analyst: {response.content}"

    new_investment_debate_state = InvestmentDebateState(
        history=history + "\n" + argument,
        bull_history=bull_history + "\n" + argument,
        bear_history=investment_debate_state.bear_history,
        current_response=argument,
        count=investment_debate_state.count + 1,
    )

    state.investment_debate_state = new_investment_debate_state
    return state
