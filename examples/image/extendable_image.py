"""
Example demonstrating the extendable parameter for images.

By default, images are extendable (extendable=True), meaning you can add layers on top of them.
You can set extendable=False to prevent further layering, which is useful for creating
final production images that should not be modified.
"""

from flyte import Image

# 1. By default, images are extendable
base_image = Image.from_debian_base(registry="localhost", name="my-base")
print(f"Base image extendable: {base_image.extendable}")  # True

# 2. You can add layers to extendable images
extended_image = base_image.with_pip_packages("numpy", "pandas")
print(f"Extended image has {len(extended_image._layers)} layers")

# 3. You can continue adding more layers
fully_extended = extended_image.with_apt_packages("vim", "curl")
print(f"Fully extended image has {len(fully_extended._layers)} layers")

# 4. Create a non-extendable final image
final_image = fully_extended.clone(name="final-image", extendable=False)
print(f"Final image extendable: {final_image.extendable}")  # False

# 5. Trying to add layers to a non-extendable image will raise an error
try:
    final_image.with_pip_packages("requests")
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: Cannot add additional layers to a non-extendable image.
    # Please create the image with extendable=True in the clone() call.

# 6. You can make a non-extendable image extendable again by cloning with extendable=True
extendable_again = final_image.clone(name="modified-image", extendable=True)
print(f"Modified image extendable: {extendable_again.extendable}")  # True

# 7. Now you can add layers again
modified_with_packages = extendable_again.with_pip_packages("requests")
print(f"Modified image with packages has {len(modified_with_packages._layers)} layers")

# 8. The extendable property is preserved when cloning without specifying it
cloned_final = final_image.clone(name="cloned-final")
print(f"Cloned final image extendable: {cloned_final.extendable}")  # False (preserved)

cloned_extendable = base_image.clone(name="cloned-base")
print(f"Cloned base image extendable: {cloned_extendable.extendable}")  # True (preserved)


# Use case: Create a base image that can be extended, then lock it down for production
def create_production_image() -> Image:
    """
    Example workflow: build up an image with all necessary dependencies,
    then lock it down for production use.
    """
    # Start with extendable base
    img = Image.from_debian_base(registry="ghcr.io/myorg", name="myapp")

    # Add dependencies
    img = img.with_apt_packages("git", "curl", "vim")
    img = img.with_pip_packages("fastapi", "uvicorn", "sqlalchemy")
    img = img.with_workdir("/app")

    # Lock it down for production - no one can accidentally add more layers
    production_img = img.clone(name="myapp-production", extendable=False)

    return production_img


if __name__ == "__main__":
    prod_img = create_production_image()
    print(f"\nProduction image: {prod_img.name}")
    print(f"Production image extendable: {prod_img.extendable}")
