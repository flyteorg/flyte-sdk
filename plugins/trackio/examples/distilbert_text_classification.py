from typing import Any

import flyte
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from flyteplugins.trackio import (
    get_trackio_run,
    trackio_config,
    trackio_init,
)

env = flyte.TaskEnvironment(name="trackio")

# remember to install psutil and accelerate


@trackio_init
@env.task
def train() -> dict[str, Any]:

    dataset = load_dataset("stanfordnlp/imdb")

    train_ds = dataset["train"].shuffle(seed=42).select(range(60))
    eval_ds = dataset["test"].shuffle(seed=42).select(range(35))

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
        )

    train_ds = train_ds.map(preprocess, batched=True)
    eval_ds = eval_ds.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./outputs",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            logging_steps=10,
            dataloader_pin_memory=False,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()

    metrics = trainer.evaluate()

    run = get_trackio_run()

    run.log(metrics)

    return metrics


cfg = trackio_config(
    project="distilbert-imdb",
    space_id="AINovice2005/distilbert-demo",
    bucket_id="AINovice2005/distilbert-storage",
    auto_log_cpu=True,
)

flyte.with_runcontext(
    custom_context=cfg.to_dict(),
).run(train)
