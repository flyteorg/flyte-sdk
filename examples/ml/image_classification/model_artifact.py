"""Artifact names shared by the training, batch-inference, and serving scripts.

The three entrypoints in this example never import each other — they are built
into different images with disjoint dependency extras (`training`, `batch`,
`serving`), so importing `training.py` from `serving.py` would drag `datasets`
and `accelerate` into the serving image. They do have to agree on the *names* of
the artifacts they pass between themselves, though, which is all this module
holds: plain strings, no imports, cheap to pull into any of the three images.

The names are the coupling point of the whole example:

    training.finetune_image_model
        └── publishes MODEL_ARTIFACT_NAME  (a Dir artifact: weights + processor)
                ├── OnArtifact trigger fires batch_inference.batch_inference_demo
                │       └── publishes PREDICTIONS_ARTIFACT_NAME (a DataFrame artifact)
                └── serving.env mounts it via flyte.app.ArtifactValue

Nothing here is a Flyte requirement — an artifact name is just a string scoped to
the project/domain. Keeping them in one module is how you avoid a typo silently
turning into "the trigger never fires".
"""

#: The fine-tuned classifier published by `training.finetune_image_model`.
#: A new version is created on every successful (non-cached) training run.
MODEL_ARTIFACT_NAME = "beans-classifier"

#: The batch-inference predictions published by `batch_inference.batch_inference_demo`.
#: Its lineage points back at the exact model version that produced it.
PREDICTIONS_ARTIFACT_NAME = "beans-classifier-predictions"
