# Trading Agents: A Multi-Agent Example

TradingAgents is a multi-agent system that simulates the collaborative decision-making dynamics of a real-world trading firm. It brings together multiple specialized agents — including a fundamental analyst, sentiment expert, technical analyst, trader, and risk manager — each powered by an LLM. These agents interact with one another to analyze market data, assess risk, and arrive at trading decisions through structured dialogue.

This example is inspired by and adapted from [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).

![Architecture](https://github.com/TauricResearch/TradingAgents/raw/main/assets/schema.png)

Set up secrets and run the trading simulation using the following commands:

```
flyte create secret openai_api_key <YOUR_OPENAI_API_KEY>
flyte create secret finnhub_api_key <YOUR_FINNHUB_API_KEY> # https://finnhub.io/
uv run main.py
```

After the trading simulation, you can run the `reflect_on_decisions` task to:

- Store the agent conversation history in an S3 vector store
- Enable agents to reference prior decisions and learnings in future runs

This makes the system more context-aware over time.

This example relies on S3-based vector storage and currently supports AWS tenants only.
Ensure the IAM role has the following permissions to use the S3 vector store:

```
s3vectors:CreateVectorBucket
s3vectors:CreateIndex
s3vectors:PutVectors
s3vectors:GetIndex
s3vectors:GetVectors
s3vectors:QueryVectors
s3vectors:GetVectorBucket
```
