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


#
# Automatically log Trainer metrics to Trackio
#
class TrackioCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            run = get_trackio_run()
            if run:
                run.log(logs)


#
# Task 1
#
@trackio_init
@env.task
def preprocess():

    dataset = load_dataset("beans")

    train_ds = dataset["train"].shuffle(seed=42).select(range(200))
    valid_ds = dataset["validation"].shuffle(seed=42).select(range(50))

    processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224"
    )

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

    return train_ds, valid_ds


#
# Task 2
#
@trackio_init
@env.task
def train(train_ds, valid_ds):

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=3,
    )

    training_args = TrainingArguments(
        output_dir="./vit-output",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=None,
        callbacks=[TrackioCallback()],
    )

    trainer.train()

    run = get_trackio_run()

    run.log(
        {
            "training_completed": True,
        }
    )

    trainer.save_model("./model")

    return trainer


#
# Task 3
#
@trackio_init
@env.task
def evaluate_model(trainer):

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    predictions = trainer.predict(trainer.eval_dataset)

    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

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

    run = get_trackio_run()

    run.log(metrics)

    return metrics


#
# Flyte Workflow
#
@flyte.workflow
def vit_training_pipeline():

    train_ds, valid_ds = preprocess()

    trainer = train(train_ds, valid_ds)

    return evaluate_model(trainer)


#
# Execute
#
flyte.with_runcontext(
    custom_context=trackio_config(
        project="vit-beans-demo",
        space_id="AINovice2005/vit-beans-dashboard",
        bucket_id="AINovice2005/vit-beans-storage",
        private=True,
        tags=["flyte", "vision", "vit"],
    )
).run(vit_training_pipeline)