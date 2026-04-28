import flyte

# Image refs are resolved at runtime via init_from_config(images=...) in main.py.
# from_ref_name() is used instead of hardcoding the URI because this file is
# copied into both images — hardcoding would create a circular reference (the
# image baked into itself would reference its own tag).
env_train = flyte.TaskEnvironment(
    name="training",
    image=flyte.Image.from_ref_name("training"),
)

env_data = flyte.TaskEnvironment(
    name="data-prep",
    image=flyte.Image.from_ref_name("data-prep"),
    depends_on=[env_train],
)
