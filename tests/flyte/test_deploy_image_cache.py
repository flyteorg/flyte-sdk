from unittest.mock import AsyncMock, patch

import pytest

from flyte._deploy import DeploymentPlan, _build_images
from flyte._image import AVAIL_PY_VERSIONS, Image
from flyte._task_environment import TaskEnvironment


@pytest.mark.asyncio
async def test_build_images_creates_correct_image_lookup_structure():
    """Test that _build_images creates the correct nested dictionary structure for ImageCache."""

    mock_image = Image.from_debian_base().with_pip_packages("numpy")
    mock_image_identifier = "test_identifier_123"
    fake_image_uri = "registry.example.com/test:latest"

    with patch.object(
        type(mock_image), "identifier", new_callable=lambda: property(lambda self: mock_image_identifier)
    ):
        mock_env = TaskEnvironment(name="test_env", image=mock_image)
        deployment_plan = DeploymentPlan(envs={"test_env": mock_env})

        with patch("flyte._deploy._build_image_bg", new_callable=AsyncMock) as mock_build:
            mock_build.return_value = ("test_env", fake_image_uri)

            image_cache = await _build_images(deployment_plan)

            # Check out identifier presented in the image_lookup dict
            assert mock_image_identifier in image_cache.image_lookup

            # Check the image_lookup dict contains all supported python version
            version_lookup = image_cache.image_lookup[mock_image_identifier]
            assert isinstance(version_lookup, dict)
            for py_version in AVAIL_PY_VERSIONS:
                assert py_version in version_lookup
                assert version_lookup[py_version] == fake_image_uri
