# MLE Agent

These examples demonstrate how to build a self-healing ML engineer agent using
Flyte.

## The problem: a vanilla ML job with no safety net

Most training pipelines are a single script or notebook cell: linear stages, no
checkpointing, and failure only surfaces at the end. You eat the full wall-clock
and compute bill before you learn anything went wrong.

```mermaid
flowchart LR
    s1["Ingest data<br/><i>✓ 12 min</i>"]
    s2["Preprocess & featurize<br/><i>✓ 38 min</i>"]
    s3["Train / tune<br/><i>✓ 5h 42m</i>"]
    s4["Evaluate & export model<br/><i>✗ crash</i>"]
    s1 --> s2 --> s3 --> s4

    classDef ok fill:#1a3d2f,stroke:#3fb950,color:#aff5b4
    classDef fail fill:#3d1a1f,stroke:#f85149,color:#ffb4b0
    class s1,s2,s3 ok
    class s4 fail
```

Classic solutions to this:
- ↪️ Automatic retries with exponential backoff for intermittent failures
- 🎒 Caching to avoid re-running previous successful steps
- 🐛 Manual debugging with logs and stack traces

## Agentic self-healing

```mermaid
flowchart TB
    subgraph slide1["① You kick off a normal job"]
        script["train.py · notebook cell · cron job"]
        cluster["Runs on a remote GPU node"]
        script --> cluster
    end

    subgraph slide2["② Linear pipeline"]
        direction LR
        s1["Ingest data<br/><i>✓ 12 min</i>"]
        s2["Preprocess & featurize<br/><i>✓ 38 min</i>"]
        s3["Train / tune<br/><i>✓ 5h 42m</i>"]
        s4["Evaluate & export model<br/><i>✗ crash</i>"]
        s1 --> s2 --> s3 --> s4
    end

    subgraph slide3["③ Sunk cost before you see the error"]
        clock["~6.5 hours elapsed"]
        bill["GPU hours already billed"]
        empty["No model.pkl · no metrics artifact"]
        s4 --> clock
        s4 --> bill
        s4 --> empty
    end

    subgraph slide4["④ The usual recovery path"]
        direction TB
        grep["Scroll cluster logs for the stack trace"]
        guess["Tweak memory, deps, or export code by hand"]
        rerun["Re-submit the whole pipeline from step ①"]
        wait["Wait another 6+ hours to find out if you guessed right"]
        grep --> guess --> rerun --> wait
        wait -.->|often fails again| grep
    end

    cluster --> s1
    empty --> grep

    classDef ok fill:#1a3d2f,stroke:#3fb950,color:#aff5b4
    classDef fail fill:#3d1a1f,stroke:#f85149,color:#ffb4b0
    classDef pain fill:#3d2a14,stroke:#d29922,color:#f5d08b
    class s1,s2,s3 ok
    class s4,empty fail
    class clock,bill,grep,guess,rerun,wait pain
```

### Examples in this directory

| Script | Sandbox | What it demonstrates |
| --- | --- | --- |
| `mle_pipeline.py` | `TaskEnvironment` | Vanilla linear pipeline above — ingest → preprocess → train succeed; export crashes |
| `mle_tool_builder_agent.py` | `TaskEnvironment` training sub-jobs | Agent writes its own training code; OOM and code errors trigger resource tuning and rewrites |
| `mle_tool_builder_agent_interactive.py` | `union.sandbox` (interactive session) | Same loop, using a live multi-turn sandbox |
| `mle_orchestrator_agent.py` | Monty orchestration sandbox | Agent composes pre-defined Flyte tools into a pipeline |
| `mle_orchestrator_agent_interactive.py` | `union.sandbox` (interactive session) | Same orchestration loop in an interactive sandbox |
| `agent_launcher_app.py` | — | Web UI to launch either agent and watch runs |
