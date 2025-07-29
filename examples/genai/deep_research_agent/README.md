# Open Deep Research

To run locally, use these commands:

```
brew install pandoc
brew install basictex (restart terminal)
export TOGETHER_API_KEY=<>
export TAVILY_API_KEY=<>
uv run agent.py
```

To run on a remote cluster, use the following commands:

```
flyte create secret --project andrew --domain development TOGETHER_API_KEY <>
flyte create secret --project andrew --domain development TAVILY_API_KEY <>
uv run agent.py
```

The example uses W&B `weave` for tracing and evaluations.

To run the evaluations locally, use these commands:

```
export HUGGINGFACE_TOKEN=<> # https://huggingface.co/settings/tokens
export WANDB_API_KEY=<> # https://wandb.ai/settings
uv run weave_evals.py
```