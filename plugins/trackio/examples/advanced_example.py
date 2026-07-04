import evaluate
import flyte
import numpy as np
from datasets import load_dataset
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
    """Automatically log Trainer metrics to Trackio."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            run = get_trackio_run()
            if run is not None:
                run.log(logs)


@trackio_init
@env.task
def train_and_evaluate() -> dict[str, float]:
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
        type="torch",
        columns=["pixel_values", "labels"],
    )

    valid_ds.set_format(
        type="torch",
        columns=["pixel_values", "labels"],
    )

    run = get_trackio_run()

    run.log(
        {
            "train_samples": len(train_ds),
            "validation_samples": len(valid_ds),
            "num_classes": len(dataset["train"].features["labels"].names),
        }
    )

    label_names = dataset["train"].features["labels"].names

    id2label = dict(enumerate(label_names))
    label2id = {label: idx for idx, label in id2label.items()}

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir="./vit-output",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=processor,
        callbacks=[TrackioCallback()],
    )

    trainer.train()

    predictions = trainer.predict(valid_ds)

    preds = np.argmax(
        predictions.predictions,
        axis=1,
    )
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

    run.log(
        {
            "train_samples": len(train_ds),
            "validation_samples": len(valid_ds),
            "num_classes": len(label_names),
            "class_names": label_names,
        }
    )

    trainer.save_model("./model")

    return metrics


cfg = trackio_config(
    project="vit-beans-demo",
    space_id="AINovice2005/vit-beans-dashboard",
    bucket_id="AINovice2005/vit-beans-storage",
)

flyte.with_runcontext(
    custom_context=cfg.to_dict(),
).run(train_and_evaluate)
