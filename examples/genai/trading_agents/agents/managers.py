from agents.utils.utils import (
    AgentState,
    InvestmentDebateState,
    RiskDebateState,
    memory_init,
)
from flyte_env import env
from langchain_openai import ChatOpenAI


@env.task
async def create_research_manager(llm: str, state: AgentState) -> AgentState:
    history = state.investment_debate_state.history
    investment_debate_state = state.investment_debate_state
    market_research_report = state.market_report
    sentiment_report = state.sentiment_report
    news_report = state.news_report
    fundamentals_report = state.fundamentals_report

    memory = await memory_init(name="research-manager")

    curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
    past_memories = memory.get_memories(curr_situation, n_matches=2)

    past_memory_str = ""
    for rec in past_memories:
        past_memory_str += rec["recommendation"] + "\n\n"

    prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate
this round of debate and make a definitive decision:
align with the bear analyst, the bull analyst,
or choose Hold only if it is strongly justified based on the arguments presented.

Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning.
Your recommendation—Buy, Sell, or Hold—must be clear and actionable.
Avoid defaulting to Hold simply because both sides have valid points;
commit to a stance grounded in the debate's strongest arguments.

Additionally, develop a detailed investment plan for the trader. This should include:

Your Recommendation: A decisive stance supported by the most convincing arguments.
Rationale: An explanation of why these arguments lead to your conclusion.
Strategic Actions: Concrete steps for implementing the recommendation.
Take into account your past mistakes on similar situations.
Use these insights to refine your decision-making and ensure you are learning and improving.
Present your analysis conversationally, as if speaking naturally, without special formatting.

Here are your past reflections on mistakes:
\"{past_memory_str}\"

Here is the debate:
Debate History:
{history}"""
    response = ChatOpenAI(model=llm).invoke(prompt)

    new_investment_debate_state = InvestmentDebateState(
        judge_decision=response.content,
        history=investment_debate_state.history,
        bear_history=investment_debate_state.bear_history,
        bull_history=investment_debate_state.bull_history,
        current_response=response.content,
        count=investment_debate_state.count,
    )

    state.investment_debate_state = new_investment_debate_state
    state.investment_plan = response.content

    return state


@env.task
async def create_risk_manager(llm: str, state: AgentState) -> AgentState:
    history = state.risk_debate_state.history
    risk_debate_state = state.risk_debate_state
    trader_plan = state.investment_plan
    market_research_report = state.market_report
    sentiment_report = state.sentiment_report
    news_report = state.news_report
    fundamentals_report = state.fundamentals_report

    memory = await memory_init(name="risk-manager")

    curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
    past_memories = memory.get_memories(curr_situation, n_matches=2)

    past_memory_str = ""
    for rec in past_memories:
        past_memory_str += rec["recommendation"] + "\n\n"

    prompt = f"""As the Risk Management Judge and Debate Facilitator,
your goal is to evaluate the debate between three risk analysts—Risky,
Neutral, and Safe/Conservative—and determine the best course of action for the trader.
Your decision must result in a clear recommendation: Buy, Sell, or Hold.
Choose Hold only if strongly justified by specific arguments, not as a fallback when all sides seem valid.
Strive for clarity and decisiveness.

Guidelines for Decision-Making:
1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context.
2. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate.
3. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**,
and adjust it based on the analysts' insights.
4. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments
and improve the decision you are making now to make sure you don't make a wrong BUY/SELL/HOLD call that loses money.

Deliverables:
- A clear and actionable recommendation: Buy, Sell, or Hold.
- Detailed reasoning anchored in the debate and past reflections.

---

**Analysts Debate History:**
{history}

---

Focus on actionable insights and continuous improvement.
Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes."""

    response = ChatOpenAI(model=llm).invoke(prompt)

    new_risk_debate_state = RiskDebateState(
        judge_decision=response.content,
        history=risk_debate_state.history,
        risky_history=risk_debate_state.risky_history,
        safe_history=risk_debate_state.safe_history,
        neutral_history=risk_debate_state.neutral_history,
        latest_speaker="Judge",
        current_risky_response=risk_debate_state.current_risky_response,
        current_safe_response=risk_debate_state.current_safe_response,
        current_neutral_response=risk_debate_state.current_neutral_response,
        count=risk_debate_state.count,
    )

    state.risk_debate_state = new_risk_debate_state
    state.final_trade_decision = response.content

    return state
