from flyte._image import Image


def test_image_from_debian_base():
    img = Image.from_debian_base()
    assert img.base_image is not None
    assert img.extendable is True


def test_image_from_base():
    img = Image.from_base("python:3.12-slim")
    assert img.base_image == "python:3.12-slim"


def test_image_from_ref_name():
    img = Image.from_ref_name("my-image")
    assert img.name == "my-image"


def test_image_clone():
    img = Image.from_base("python:3.12-slim")
    clone = img.clone()
    assert clone.base_image == img.base_image


def test_image_with_pip_packages():
    img = Image.from_debian_base().with_pip_packages("numpy", "pandas")
    layers = img._layers
    assert len(layers) > 0


def test_image_with_requirements():
    img = Image.from_debian_base().with_requirements("requirements.txt")
    layers = img._layers
    assert len(layers) > 0


def test_image_with_env_vars():
    img = Image.from_debian_base().with_env_vars({"MY_VAR": "value"})
    layers = img._layers
    assert len(layers) > 0


def test_image_with_workdir():
    img = Image.from_debian_base().with_workdir("/app")
    layers = img._layers
    assert len(layers) > 0


def test_image_chaining():
    img = Image.from_debian_base().with_pip_packages("numpy").with_env_vars({"KEY": "VAL"}).with_workdir("/workspace")
    assert len(img._layers) >= 3


def test_image_importable():
    import flyte

    assert flyte.Image is Image
