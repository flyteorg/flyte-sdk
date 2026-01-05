from typing import Tuple

from flyte._image import Architecture
from flyte.extend import ImageBuilder, ImageChecker


IMAGE_TASK_NAME = "build-image"
IMAGE_TASK_PROJECT = "system"
IMAGE_TASK_DOMAIN = "production"


class MyImageChecker(ImageChecker):
    _images_client = None

    @classmethod
    async def image_exists(
        cls, repository: str, tag: str, arch: Tuple[Architecture, ...] = ("linux/amd64",)
    ) -> Optional[str]:
       return True


class MyImageBuilder(ImageBuilder):
    def get_checkers(self) -> Optional[typing.List[typing.Type[ImageChecker]]]:
        """Return the image checker."""
        return [MyImageChecker]

    async def build_image(self, image: Image, dry_run: bool = False) -> str:
        print("Building image locally...")
        return image.uri
