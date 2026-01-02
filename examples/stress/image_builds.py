import asyncio

import flyte
from flyte import Image

# Create 10 different images to trigger 10 remote builds with large packages
images = [
    Image.from_debian_base()
    .with_apt_packages("vim", "wget")
    .with_pip_packages("tensorflow")
    .with_env_vars({"hello": "world1", "build_id": "1"}),
    Image.from_debian_base()
    .with_apt_packages("curl", "git")
    .with_pip_packages("torch")
    .with_env_vars({"hello": "world2", "build_id": "2"}),
    Image.from_debian_base()
    .with_apt_packages("vim", "curl")
    .with_pip_packages("transformers")
    .with_env_vars({"hello": "world3", "build_id": "3"}),
    Image.from_debian_base()
    .with_apt_packages("wget", "git")
    .with_pip_packages("torch", "torchvision")
    .with_env_vars({"hello": "world4", "build_id": "4"}),
    Image.from_debian_base()
    .with_apt_packages("vim", "git", "curl")
    .with_pip_packages("tensorflow", "keras")
    .with_env_vars({"hello": "world5", "build_id": "5"}),
    Image.from_debian_base()
    .with_apt_packages("wget", "curl")
    .with_pip_packages("scipy", "scikit-learn")
    .with_env_vars({"hello": "world6", "build_id": "6"}),
    Image.from_debian_base()
    .with_apt_packages("vim", "wget", "git")
    .with_pip_packages("pandas", "numpy", "matplotlib")
    .with_env_vars({"hello": "world7", "build_id": "7"}),
    Image.from_debian_base()
    .with_apt_packages("curl", "wget")
    .with_pip_packages("xgboost", "lightgbm")
    .with_env_vars({"hello": "world8", "build_id": "8"}),
    Image.from_debian_base()
    .with_apt_packages("git", "vim")
    .with_pip_packages("opencv-python", "pillow")
    .with_env_vars({"hello": "world9", "build_id": "9"}),
    Image.from_debian_base()
    .with_apt_packages("wget", "git", "curl")
    .with_pip_packages("ray", "dask")
    .with_env_vars({"hello": "world10", "build_id": "10"}),
]


async def main():
    print("Starting stress test with 10 image builds in parallel...")
    build_tasks = [flyte.build.aio(image) for image in images]
    await asyncio.gather(*build_tasks)
    print("\n✅ All 10 image builds triggered successfully!")


if __name__ == "__main__":
    flyte.init_from_config()
    asyncio.run(main())
