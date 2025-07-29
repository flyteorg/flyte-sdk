from agents.utils.utils import AgentState, RiskDebateState
from flyte_env import env
from langchain_openai import ChatOpenAI


@env.task
async def create_risky_debator(llm: str, state: AgentState) -> AgentState:
    risk_debate_state = state.risk_debate_state
    history = risk_debate_state.history
    risky_history = risk_debate_state.risky_history

    current_safe_response = risk_debate_state.current_safe_response
    current_neutral_response = risk_debate_state.current_neutral_response

    market_research_report = state.market_report
    sentiment_report = state.sentiment_report
    news_report = state.news_report
    fundamentals_report = state.fundamentals_report

    trader_decision = state.trader_investment_plan

    prompt = f"""As the Risky Risk Analyst, your role is to actively champion high-reward, high-risk opportunities,
emphasizing bold strategies and competitive advantages.
When evaluating the trader's decision or plan, focus intently on the potential upside, growth potential,
and innovative benefits—even when these come with elevated risk.
Use the provided market data and sentiment analysis to strengthen your arguments and challenge the opposing views.
Specifically, respond directly to each point made by the conservative and neutral analysts,
countering with data-driven rebuttals and persuasive reasoning.
Highlight where their caution might miss critical opportunities or where their assumptions may be overly conservative.
Here is the trader's decision:

{trader_decision}

Your task is to create a compelling case for the trader's decision by questioning and critiquing the conservative
and neutral stances to demonstrate why your high-reward perspective offers the best path forward.
Incorporate insights from the following sources into your arguments:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history}
Here are the last arguments from the conservative analyst: {current_safe_response}
Here are the last arguments from the neutral analyst: {current_neutral_response}.
If there are no responses from the other viewpoints, do not halluncinate and just present your point.

Engage actively by addressing any specific concerns raised, refuting the weaknesses in their logic,
and asserting the benefits of risk-taking to outpace market norms.
Maintain a focus on debating and persuading, not just presenting data.
Challenge each counterpoint to underscore why a high-risk approach is optimal.
Output conversationally as if you are speaking without any special formatting."""

    response = ChatOpenAI(model=llm).invoke(prompt)

    argument = f"Risky Analyst: {response.content}"

    new_risk_debate_state = RiskDebateState(
        history=history + "\n" + argument,
        risky_history=risky_history + "\n" + argument,
        safe_history=risk_debate_state.safe_history,
        neutral_history=risk_debate_state.neutral_history,
        latest_speaker="Risky",
        current_risky_response=argument,
        current_safe_response=current_safe_response,
        current_neutral_response=current_neutral_response,
        count=risk_debate_state.count + 1,
    )

    state.risk_debate_state = new_risk_debate_state
    return state


@env.task
async def create_safe_debator(llm: str, state: AgentState) -> AgentState:
    risk_debate_state = state.risk_debate_state
    history = risk_debate_state.history
    safe_history = risk_debate_state.safe_history

    current_risky_response = risk_debate_state.current_risky_response
    current_neutral_response = risk_debate_state.current_neutral_response

    market_research_report = state.market_report
    sentiment_report = state.sentiment_report
    news_report = state.news_report
    fundamentals_report = state.fundamentals_report

    trader_decision = state.trader_investment_plan

    prompt = f"""As the Safe/Conservative Risk Analyst, your primary objective is to protect assets,
minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation,
carefully assessing potential losses, economic downturns, and market volatility.
When evaluating the trader's decision or plan, critically examine high-risk elements,
pointing out where the decision may expose the firm to undue risk and where more cautious
alternatives could secure long-term gains.
Here is the trader's decision:

{trader_decision}

Your task is to actively counter the arguments of the Risky and Neutral Analysts,
highlighting where their views may overlook potential threats or fail to prioritize sustainability.
Respond directly to their points, drawing from the following data sources
to build a convincing case for a low-risk approach adjustment to the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history}
Here is the last response from the risky analyst: {current_risky_response}
Here is the last response from the neutral analyst: {current_neutral_response}.
If there are no responses from the other viewpoints, do not halluncinate and just present your point.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked.
Address each of their counterpoints to showcase why a conservative stance is ultimately the
safest path for the firm's assets.
Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy
over their approaches.
Output conversationally as if you are speaking without any special formatting."""

    response = ChatOpenAI(model=llm).invoke(prompt)

    argument = f"Safe Analyst: {response.content}"

    new_risk_debate_state = RiskDebateState(
        history=history + "\n" + argument,
        risky_history=risk_debate_state.risky_history,
        safe_history=safe_history + "\n" + argument,
        neutral_history=risk_debate_state.neutral_history,
        latest_speaker="Safe",
        current_risky_response=current_risky_response,
        current_safe_response=argument,
        current_neutral_response=current_neutral_response,
        count=risk_debate_state.count + 1,
    )

    state.risk_debate_state = new_risk_debate_state
    return state


@env.task
async def create_neutral_debator(llm: str, state: AgentState) -> AgentState:
    risk_debate_state = state.risk_debate_state
    history = risk_debate_state.history
    neutral_history = risk_debate_state.neutral_history

    current_risky_response = risk_debate_state.current_risky_response
    current_safe_response = risk_debate_state.current_safe_response

    market_research_report = state.market_report
    sentiment_report = state.sentiment_report
    news_report = state.news_report
    fundamentals_report = state.fundamentals_report

    trader_decision = state.trader_investment_plan

    prompt = f"""As the Neutral Risk Analyst, your role is to provide a balanced perspective,
weighing both the potential benefits and risks of the trader's decision or plan.
You prioritize a well-rounded approach, evaluating the upsides
and downsides while factoring in broader market trends,
potential economic shifts, and diversification strategies.Here is the trader's decision:

{trader_decision}

Your task is to challenge both the Risky and Safe Analysts,
pointing out where each perspective may be overly optimistic or overly cautious.
Use insights from the following data sources to support a moderate, sustainable strategy
to adjust the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history}
Here is the last response from the risky analyst: {current_risky_response}
Here is the last response from the safe analyst: {current_safe_response}.
If there are no responses from the other viewpoints, do not halluncinate and just present your point.

Engage actively by analyzing both sides critically, addressing weaknesses in the risky
and conservative arguments to advocate for a more balanced approach.
Challenge each of their points to illustrate why a moderate risk strategy might offer the best of both worlds,
providing growth potential while safeguarding against extreme volatility.
Focus on debating rather than simply presenting data, aiming to show that a balanced view can lead to
the most reliable outcomes. Output conversationally as if you are speaking without any special formatting."""

    response = ChatOpenAI(model=llm).invoke(prompt)

    argument = f"Neutral Analyst: {response.content}"

    new_risk_debate_state = RiskDebateState(
        history=history + "\n" + argument,
        risky_history=risk_debate_state.risky_history,
        safe_history=risk_debate_state.safe_history,
        neutral_history=neutral_history + "\n" + argument,
        latest_speaker="Neutral",
        current_risky_response=current_risky_response,
        current_safe_response=current_safe_response,
        current_neutral_response=argument,
        count=risk_debate_state.count + 1,
    )

    state.risk_debate_state = new_risk_debate_state
    return state
