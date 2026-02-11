import typing
from typing import Optional, Tuple

from flyte import Image
from flyte.extend import Architecture, ImageBuilder, ImageChecker


class MyImageChecker(ImageChecker):
    _images_client = None

    @classmethod
    async def image_exists(
        cls, repository: str, tag: str, arch: Tuple[Architecture, ...] = ("linux/amd64",)
    ) -> Optional[str]:
        return f"{repository}:{tag}"


class MyImageBuilder(ImageBuilder):
    def get_checkers(self) -> Optional[typing.List[typing.Type[ImageChecker]]]:
        """Return the image checker."""
        return [MyImageChecker]

    async def build_image(
        self,
        image: Image,
        dry_run: bool = False,
        wait: bool = True,
    ) -> str:
        print("Building image locally...")
        return image.uri
