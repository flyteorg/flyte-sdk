from agents.utils.utils import AgentState, memory_init
from flyte_env import env
from langchain_core.messages import convert_to_openai_messages
from langchain_openai import ChatOpenAI


@env.task
async def create_trader(llm: str, state: AgentState) -> AgentState:
    company_name = state.company_of_interest
    investment_plan = state.investment_plan
    market_research_report = state.market_report
    sentiment_report = state.sentiment_report
    news_report = state.news_report
    fundamentals_report = state.fundamentals_report

    memory = await memory_init(name="trader")

    curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
    past_memories = memory.get_memories(curr_situation, n_matches=2)

    past_memory_str = ""
    for rec in past_memories:
        past_memory_str += rec["recommendation"] + "\n\n"

    context = {
        "role": "user",
        "content": f"Based on a comprehensive analysis by a team of analysts, "
        f"here is an investment plan tailored for {company_name}. "
        "This plan incorporates insights from current technical market trends, "
        "macroeconomic indicators, and social media sentiment. "
        "Use this plan as a foundation for evaluating your next trading decision.\n\n"
        f"Proposed Investment Plan: {investment_plan}\n\n"
        "Leverage these insights to make an informed and strategic decision.",
    }

    messages = [
        {
            "role": "system",
            "content": f"""You are a trading agent analyzing market data to make investment decisions.
Based on your analysis, provide a specific recommendation to buy, sell, or hold.
End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'
to confirm your recommendation.
Do not forget to utilize lessons from past decisions to learn from your mistakes.
Here is some reflections from similar situatiosn you traded in and the lessons learned: {past_memory_str}""",
        },
        context,
    ]

    result = ChatOpenAI(model=llm).invoke(messages)

    state.messages.append(convert_to_openai_messages(result))
    state.trader_investment_plan = result.content
    state.sender = "Trader"

    return state
