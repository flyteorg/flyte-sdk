import tempfile
from pathlib import Path

import pytest

from flyte import Image
from flyte._image import DockerIgnore
from flyte._internal.imagebuild.utils import get_and_list_dockerignore


def test_get_and_list_dockerignore_with_dockerignore_file():
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)
        dockerignore_file = src_dir / ".dockerignore"
        dockerignore_file.write_text("*.py\nsrc/\n.cache\n# This is a comment\n\n*.txt\n \n  \n\t\n")
        image = Image.from_debian_base()
        patterns = get_and_list_dockerignore(image, src_dir)
        expected_patterns = ["*.py", "src/", ".cache", "*.txt"]

        assert patterns == expected_patterns


def test_get_and_list_dockerignore_with_dockerignore_layer():
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)
        root_dockerignore = src_dir / ".dockerignore"
        root_dockerignore.write_text("*.py\nsrc/\n")
        custom_dockerignore = src_dir / "custom.dockerignore"
        custom_dockerignore.write_text("*.txt\n.cache\n")
        image = Image.from_debian_base().with_dockerignore(custom_dockerignore)
        patterns = get_and_list_dockerignore(image, src_dir)
        expected_patterns = ["*.txt", ".cache"]

        assert patterns == expected_patterns


def test_get_and_list_dockerignore_not_found():
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)
        image = Image.from_debian_base()
        patterns = get_and_list_dockerignore(image, src_dir)

        assert patterns == []

def test_get_and_list_dockerignore_layer_priority():
    with tempfile.TemporaryDirectory() as tmp_context:
        src_dir = Path(tmp_context)
        local_dockerignore = src_dir / ".dockerignore"
        local_dockerignore.write_text("*.py\nsrc/\n")
        layer_dockerignore = src_dir / "custom.dockerignore"
        layer_dockerignore.write_text("*.txt\n.cache\n")
        image = Image.from_debian_base().with_dockerignore(layer_dockerignore)
        patterns = get_and_list_dockerignore(image, src_dir)
        expected_patterns = ["*.txt", ".cache"]
        
        assert patterns == expected_patterns
