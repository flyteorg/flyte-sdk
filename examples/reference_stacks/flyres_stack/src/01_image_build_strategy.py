"""
Image Build Strategy: Base + Layered Images for ML Workflows

This example demonstrates an efficient image building strategy that:
1. Builds a base image with PyTorch and CUDA (built infrequently, out-of-band)
2. Builds a second layer with experimental packages (changes more quickly)

This approach minimizes build times by caching the slow CUDA-dependent parts.
"""

# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte",
# ]
# ///

import flyte
from flyte import Image, Resources


def get_base_image():
    """
    Build a base image with PyTorch and CUDA.

    This is built out-of-band (e.g., nightly or on-demand) since it includes
    the heavy CUDA-dependent packages that take the longest to build.
    """
    return (
        Image.from_debian_base(name="ml-base", python_version=(3, 12))
        .with_pip_packages("torch==2.4.0", "torchvision==0.19.0")
        .with_apt_packages("cuda-compiler-12-4", "libcudnn8-dev")
        .with_env_vars({"CUDA_VERSION": "12.4"})
    )


def get_experimental_image(base: Image) -> Image:
    """
    Build a second layer on top of the base image with experimental packages.

    This layer changes more frequently and should rebuild faster since
    the base image is cached.
    """
    return (
        base.with_pip_packages(
            "transformers==4.42.0",
            "accelerate==0.31.0",
            "peft==0.11.0",
        )
        .with_apt_packages("git")
        .with_env_vars({"EXPERIMENTAL": "true"})
    )


# Define the images
base_image = get_base_image()
experimental_image = get_experimental_image(base_image)

# Create task environments with different images
base_env = flyte.TaskEnvironment(
    name="ml_base_task",
    image=base_image,
    resources=Resources(cpu=(2, 4), memory=("1Gi", "8Gi")),
)

exp_env = flyte.TaskEnvironment(
    name="ml_experimental_task",
    image=experimental_image,
    resources=Resources(cpu=(2, 4), memory=("1Gi", "8Gi")),
)


@base_env.task
async def verify_pytorch_version() -> dict:
    """Verify PyTorch and CUDA are installed correctly."""
    import torch

    return {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }


@exp_env.task
async def verify_experimental_packages() -> dict:
    """Verify experimental packages can import correctly."""
    import accelerate
    import transformers

    return {
        "transformers_version": transformers.__version__,
        "accelerate_version": accelerate.__version__,
    }


@base_env.task
async def build_model_checkpoint() -> str:
    """
    Simulate building a model checkpoint with base packages.

    In production, this could be training a small model or downloading
    weights from Hugging Face.
    """
    import torch

    # Create a simple model
    torch.nn.Linear(10, 5)
    return "base_model_checkpoint.pth"


@exp_env.task
async def fine_tune_model(checkpoint_path: str) -> str:
    """
    Fine-tune the base model using experimental packages.

    This demonstrates how the layered approach allows different
    versions of experimental packages to be tested without rebuilding
    the entire image.
    """

    # Load and fine-tune (simulated)
    checkpoint = {"path": checkpoint_path, "fine_tuned_with": "experimental_packages"}
    return str(checkpoint)


@exp_env.task(depends_on=[verify_experimental_packages])
async def full_training_workflow() -> dict:
    """Run the full training workflow."""

    # Build base model
    checkpoint = await build_model_checkpoint()

    # Fine-tune with experimental packages
    tuned_checkpoint = await fine_tune_model(checkpoint)

    return {
        "base_model": checkpoint,
        "fine_tuned_model": tuned_checkpoint,
        "workflow_status": "complete",
    }


if __name__ == "__main__":
    flyte.init_from_config()

    # Build images (in practice, these would be built separately)
    print("Building base image...")
    base_result = flyte.build(base_image, force=False, wait=False)
    print(f"Base image URI: {base_result.uri}")

    print("\nBuilding experimental image (layered on base)...")
    exp_result = flyte.build(experimental_image, force=False, wait=False)
    print(f"Experimental image URI: {exp_result.uri}")

    # Run tasks
    print("\nVerifying base environment...")
    result1 = flyte.run(verify_pytorch_version)
    print("PyTorch check: {}".format(result1.outputs[0]))

    print("\nVerifying experimental packages...")
    result2 = flyte.run(verify_experimental_packages)
    print("Experimental packages: {}".format(result2.outputs[0]))

    print("\nRunning full training workflow...")
    result3 = flyte.run(full_training_workflow)
    print("Training result: {}".format(result3.outputs[0]))
