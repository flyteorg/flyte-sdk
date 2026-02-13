"""
Example demonstrating the extendable parameter for images.

By default, images are NOT extendable (extendable=False), meaning you cannot add layers on top of them.
You must set extendable=True to allow layering, which is useful for building up images incrementally.
"""

from flyte import Image

# 1. By default, images are NOT extendable
base_image = Image.from_debian_base(registry="localhost", name="my-base")
print(f"Base image extendable: {base_image.extendable}")  # False

# 2. To add layers, you must make the image extendable
extendable_image = base_image.clone(name="extendable-base", extendable=True)
print(f"Extendable image extendable: {extendable_image.extendable}")  # True

# 3. Now you can add layers to extendable images
extended_image = extendable_image.with_pip_packages("numpy", "pandas")
print(f"Extended image has {len(extended_image._layers)} layers")

# 4. You can continue adding more layers
fully_extended = extended_image.with_apt_packages("vim", "curl")
print(f"Fully extended image has {len(fully_extended._layers)} layers")

# 5. Trying to add layers to a non-extendable image will raise an error
try:
    base_image.with_pip_packages("requests")
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: Cannot add additional layers to a non-extendable image.
    # Please create the image with extendable=True in the clone() call.

# 6. You can lock down an extendable image by cloning with extendable=False
final_image = fully_extended.clone(name="final-image", extendable=False)
print(f"Final image extendable: {final_image.extendable}")  # False

# 7. The extendable property is preserved when cloning without specifying it
cloned_final = final_image.clone(name="cloned-final")
print(f"Cloned final image extendable: {cloned_final.extendable}")  # False (preserved)

cloned_extendable = extendable_image.clone(name="cloned-extendable")
print(f"Cloned extendable image extendable: {cloned_extendable.extendable}")  # True (preserved)


# Use case: Create an extendable base image, build it up, then optionally lock it down
def create_production_image() -> Image:
    """
    Example workflow: create an extendable base, build up dependencies,
    then optionally lock it down for production use.
    """
    # Start with extendable base
    img = Image.from_debian_base(registry="ghcr.io/myorg", name="myapp", extendable=True)

    # Add dependencies
    img = img.with_apt_packages("git", "curl", "vim")
    img = img.with_pip_packages("fastapi", "uvicorn", "sqlalchemy")
    img = img.with_workdir("/app")

    # Optionally lock it down for production (though it's already not extendable by default)
    # This step is actually unnecessary since we could just clone without extendable=True
    production_img = img.clone(name="myapp-production", extendable=False)

    return production_img


if __name__ == "__main__":
    prod_img = create_production_image()
    print(f"\nProduction image: {prod_img.name}")
    print(f"Production image extendable: {prod_img.extendable}")
