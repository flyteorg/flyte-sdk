# /// script
# requires-python = "==3.13"
# dependencies = [
#    "pandas",
#    "pyarrow",
#    "flyte>=0.2.0b10",
# ]
# ///


import asyncio
import random
from typing import List, Union

import pandas as pd

import flyte

driver = flyte.TaskEnvironment(
    name="driver",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    image=flyte.Image.from_uv_script(__file__, name="flyte", registry="ghcr.io/flyteorg", pre=True).with_apt_packages(
        "ca-certificates"
    ),
)


# Utility to simulate async work
def log(name: str) -> str:
    print(f"[{name}] starting")
    return name


def done(name: str) -> None:
    print(f"[{name}] done")


async def simulate_task(name: str, delay: Union[float, None] = None) -> str:
    log(name)
    await asyncio.sleep(delay or random.uniform(0.1, 0.5))
    done(name)
    return name


# -- Stage 1: Data Ingestion --
@driver.task
async def fetch_user_events() -> pd.DataFrame:
    await simulate_task("fetch_user_events")
    return pd.DataFrame({"user_id": [1, 2], "event": ["click", "purchase"]})


@driver.task
async def fetch_sales_data() -> pd.DataFrame:
    await simulate_task("fetch_sales_data")
    return pd.DataFrame({"order_id": [101, 102], "amount": [250, 180]})


@driver.task
async def fetch_support_tickets() -> pd.DataFrame:
    await simulate_task("fetch_support_tickets")
    return pd.DataFrame({"ticket_id": [501, 502], "sentiment": ["positive", "negative"]})


# -- Stage 2: Data Validation --
@driver.task
async def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    await simulate_task("validate_schema")
    return df


@driver.task
async def validate_missingness(df: pd.DataFrame) -> pd.DataFrame:
    await simulate_task("validate_missingness")
    return df.dropna()


@driver.task
async def validate_distribution(df: pd.DataFrame) -> pd.DataFrame:
    await simulate_task("validate_distribution")
    return df


# -- Stage 3: Feature Engineering --
@driver.task
async def fe_user_behavior(user_events: pd.DataFrame) -> pd.DataFrame:
    await simulate_task("fe_user_behavior")
    return user_events.assign(event_count=1)


@driver.task
async def fe_sales_metrics(sales_data: pd.DataFrame) -> pd.DataFrame:
    await simulate_task("fe_sales_metrics")
    return sales_data.assign(amount_log=sales_data["amount"].apply(lambda x: round(x**0.5, 2)))


@driver.task
async def fe_sentiment(tickets: pd.DataFrame) -> pd.DataFrame:
    await simulate_task("fe_sentiment")
    return tickets.assign(sentiment_score=tickets["sentiment"].map({"positive": 1, "negative": -1}))


# -- Stage 4: Feature Join --
@driver.task
async def join_features(features: List[pd.DataFrame]) -> pd.DataFrame:
    await simulate_task("join_features")
    return pd.concat(features, axis=1)


# -- Stage 5: Model Training --
@driver.task
async def train_model_A(features: pd.DataFrame) -> str:
    await simulate_task("train_model_A")
    return "model_A"


@driver.task
async def train_model_B(features: pd.DataFrame) -> str:
    await simulate_task("train_model_B")
    return "model_B"


# -- Stage 6: Validation --
@driver.task
async def cross_validate_model(model: str) -> str:
    await simulate_task(f"cross_validate_{model}")
    return f"{model}_cv_score"


# -- Stage 7: Drift Detection --
@driver.task
async def compute_data_drift(features: pd.DataFrame) -> str:
    await simulate_task("compute_data_drift")
    return "data_drift_metrics"


@driver.task
async def compute_concept_drift(models: List[str]) -> str:
    await simulate_task("compute_concept_drift")
    return "concept_drift_metrics"


# -- Stage 8: Model Selection --
@driver.task
async def select_best_model(cv_scores: List[str]) -> str:
    await simulate_task("select_best_model")
    return "best_model"


# -- Stage 9: Shadow Deployment --
@driver.task
async def deploy_model_shadow(model: str) -> str:
    await simulate_task("deploy_model_shadow")
    return f"shadow_{model}"


@driver.task
async def monitor_latency(model: str) -> str:
    await simulate_task("monitor_latency")
    return f"{model}_latency"


@driver.task
async def monitor_predictions(model: str) -> str:
    await simulate_task("monitor_predictions")
    return f"{model}_predictions"


# -- Stage 10: Canary Testing --
@driver.task
async def canary_traffic_split(model: str) -> str:
    await simulate_task("canary_traffic_split")
    return f"canary_{model}"


@driver.task
async def compare_to_baseline(model: str) -> str:
    await simulate_task("compare_to_baseline")
    return f"{model}_comparison"


# -- Stage 11: Model Promotion --
@driver.task
async def promote_model_if_safe(metrics: List[str]) -> str:
    await simulate_task("promote_model_if_safe")
    return "promoted_model"


# -- Stage 12: Feedback Loop --
@driver.task
async def collect_user_feedback() -> pd.DataFrame:
    await simulate_task("collect_user_feedback")
    return pd.DataFrame({"feedback_id": [1, 2], "label": ["good", "bad"]})


@driver.task
async def auto_label_feedback(feedback: pd.DataFrame) -> pd.DataFrame:
    await simulate_task("auto_label_feedback")
    feedback["auto_label"] = feedback["label"].map({"good": 1, "bad": 0})
    return feedback


@driver.task
async def store_for_online_learning(labeled_feedback: pd.DataFrame) -> str:
    await simulate_task("store_for_online_learning")
    return "stored_feedback"


# -- Stage 13: Retraining Trigger --
@driver.task
async def trigger_retrain_if_drift_or_feedback(drift_metrics: str, feedback: pd.DataFrame) -> str:
    await simulate_task("trigger_retrain_if_drift_or_feedback")
    return "retrain_triggered"


# -- Main DAG Orchestration --
@driver.task
async def main() -> None:
    # Stage 1
    user_events, sales_data, support_tickets = await asyncio.gather(
        fetch_user_events(), fetch_sales_data(), fetch_support_tickets()
    )

    # Stage 2
    user_events = await validate_schema(user_events)
    sales_data = await validate_schema(sales_data)
    support_tickets = await validate_schema(support_tickets)

    user_events = await validate_missingness(user_events)
    sales_data = await validate_missingness(sales_data)
    support_tickets = await validate_missingness(support_tickets)

    user_events = await validate_distribution(user_events)
    sales_data = await validate_distribution(sales_data)
    support_tickets = await validate_distribution(support_tickets)

    # Stage 3
    user_behavior, sales_metrics, sentiment = await asyncio.gather(
        fe_user_behavior(user_events), fe_sales_metrics(sales_data), fe_sentiment(support_tickets)
    )

    # Stage 4
    joined_features = await join_features([user_behavior, sales_metrics, sentiment])

    # Stage 5
    model_A_task = train_model_A(joined_features)
    model_B_task = train_model_B(joined_features)
    model_A, model_B = await asyncio.gather(model_A_task, model_B_task)

    # Stage 6 + 7 in parallel
    cv_A_task = cross_validate_model(model_A)
    cv_B_task = cross_validate_model(model_B)
    data_drift_task = compute_data_drift(joined_features)
    concept_drift_task = compute_concept_drift([model_A, model_B])
    cv_A, cv_B, data_drift, concept_drift = await asyncio.gather(
        cv_A_task, cv_B_task, data_drift_task, concept_drift_task
    )

    # Stage 8
    best_model = await select_best_model([cv_A, cv_B])

    # Stage 9
    await asyncio.gather(deploy_model_shadow(best_model), monitor_latency(best_model), monitor_predictions(best_model))

    # Stage 10
    await asyncio.gather(canary_traffic_split(best_model), compare_to_baseline(best_model))

    # Stage 11
    await promote_model_if_safe([data_drift, concept_drift])

    # Stage 12
    feedback = await collect_user_feedback()
    labeled_feedback = await auto_label_feedback(feedback)
    await store_for_online_learning(labeled_feedback)

    # Stage 13
    await trigger_retrain_if_drift_or_feedback(data_drift, labeled_feedback)


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.run(main)

    # Run with:
    # uv run --prerelease=allow examples/ml/rfe.py
