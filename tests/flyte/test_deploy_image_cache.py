from unittest.mock import AsyncMock, patch

import sys
import pytest

from flyte._deploy import DeploymentPlan, _build_images
from flyte._image import Image
from flyte._task_environment import TaskEnvironment


@pytest.mark.parametrize("python_version,expected_py_version", [
    (None, "{}.{}".format(sys.version_info.major, sys.version_info.minor)),  # Use local python version
    ((3, 10), "3.10"),
])
@pytest.mark.asyncio
async def test_create_image_cache_lookup(python_version, expected_py_version):
    """Test that _build_images creates the correct nested dictionary structure for ImageCache."""

    if python_version is None:
        mock_image = Image.from_debian_base().with_pip_packages("numpy")
    else:
        mock_image = Image.from_debian_base(python_version=python_version).with_pip_packages("numpy")

    mock_image_identifier = f"test_identifier_{expected_py_version.replace('.', '_')}"
    fake_image_uri = f"registry.example.com/test-py{expected_py_version}:latest"

    with patch.object(
        type(mock_image), "identifier", new_callable=lambda: property(lambda self: mock_image_identifier)
    ):
        mock_env = TaskEnvironment(name="test_env", image=mock_image)
        deployment_plan = DeploymentPlan(envs={"test_env": mock_env})

        with patch("flyte._deploy._build_image_bg", new_callable=AsyncMock) as mock_build:
            mock_build.return_value = ("test_env", fake_image_uri)

            image_cache = await _build_images(deployment_plan)

            # Check that identifier is present in the image_lookup dict
            assert mock_image_identifier in image_cache.image_lookup

            # Check the image_lookup dict contains the expected python version
            version_lookup = image_cache.image_lookup[mock_image_identifier]
            # Make sure there's only one python version presented
            assert len(version_lookup) == 1
            assert version_lookup[expected_py_version] == fake_image_uri
