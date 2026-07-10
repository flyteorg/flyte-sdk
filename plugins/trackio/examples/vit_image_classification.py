"""
Workflow-based Trackio integration with Flyte.

This example demonstrates how Trackio can be used across a complete Flyte
machine learning workflow.

The workflow trains a Vision Transformer (ViT) on the Beans image
classification dataset using the Hugging Face Transformers library.
The pipeline is divided into multiple Flyte tasks responsible for
dataset preprocessing, model training, and evaluation.

Each task is decorated with ``@trackio_init`` and contributes metrics to
the same Trackio experiment, providing a unified view of the workflow
execution.

This example demonstrates:

* Multi-task Flyte workflows with Trackio.
* Automatic experiment initialization and lifecycle management.
* Logging preprocessing, training, and evaluation metrics.
* Hugging Face Transformers integration.
* CPU-friendly Vision Transformer training.
* Running the workflow locally with ``flyte.with_runcontext``.

"""

import evaluate
import flyte
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoImageProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    ViTForImageClassification,
)

from flyteplugins.trackio import (
    get_trackio_run,
    trackio_config,
    trackio_init,
)

env = flyte.TaskEnvironment(name="trackio")


class TrackioCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        run = get_trackio_run()
        if run and logs:
            run.log(logs)


@trackio_init
@env.task
def preprocess() -> tuple[str, str]:

    dataset = load_dataset("AI-Lab-Makerere/beans")

    train_ds = dataset["train"].shuffle(seed=42).select(range(100))
    valid_ds = dataset["validation"].shuffle(seed=42).select(range(20))

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def transform(example):
        image = example["image"].convert("RGB")
        example["pixel_values"] = processor(
            image,
            return_tensors="pt",
        )["pixel_values"][0]
        return example

    train_ds = train_ds.map(transform)
    valid_ds = valid_ds.map(transform)

    train_ds.set_format(
        "torch",
        columns=["pixel_values", "labels"],
    )

    valid_ds.set_format(
        "torch",
        columns=["pixel_values", "labels"],
    )

    train_path = "./train_ds"
    valid_path = "./valid_ds"

    train_ds.save_to_disk(train_path)
    valid_ds.save_to_disk(valid_path)

    run = get_trackio_run()

    run.log(
        {
            "train_samples": len(train_ds),
            "validation_samples": len(valid_ds),
            "classes": len(dataset["train"].features["labels"].names),
        }
    )

    return train_path, valid_path


@trackio_init
@env.task
def train(
    train_path: str,
    valid_path: str,
) -> str:

    train_ds = load_from_disk(train_path)
    valid_ds = load_from_disk(valid_path)

    label_names = train_ds.features["labels"].names

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={v: k for k, v in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./vit-output",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=5,
            remove_unused_columns=False,
            report_to=[],
            dataloader_pin_memory=False,
        ),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        callbacks=[TrackioCallback()],
    )

    trainer.train()

    model_path = "./model"

    trainer.save_model(model_path)

    get_trackio_run().log(
        {
            "training_complete": True,
        }
    )

    return model_path


@trackio_init
@env.task
def evaluate_model(
    model_path: str,
    valid_path: str,
) -> dict[str, float]:

    valid_ds = load_from_disk(valid_path)

    model = ViTForImageClassification.from_pretrained(
        model_path,
    )

    trainer = Trainer(model=model)

    predictions = trainer.predict(valid_ds)

    preds = np.argmax(predictions.predictions, axis=1)

    labels = predictions.label_ids

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    metrics = {
        **accuracy.compute(
            predictions=preds,
            references=labels,
        ),
        **f1.compute(
            predictions=preds,
            references=labels,
            average="weighted",
        ),
    }

    get_trackio_run().log(metrics)

    return metrics


@flyte.workflow
def vit_pipeline():

    train_path, valid_path = preprocess()

    model_path = train(
        train_path,
        valid_path,
    )

    return evaluate_model(
        model_path,
        valid_path,
    )


if __name__ == "__main__":
    cfg = trackio_config(
        project="vit-beans-demo",
        space_id="AINovice2005/vit-demo",
        bucket_id="AINovice2005/vit-storage",
        auto_log_cpu=True,
    )

    flyte.with_runcontext(
        custom_context=cfg.to_dict(),
    ).run(vit_pipeline)
