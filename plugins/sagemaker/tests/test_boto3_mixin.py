import pytest

from flyteplugins.sagemaker._utils import (
    normalize_integers,
    render_config,
    resolve_images,
    resolve_region,
    stable_value,
    triton_image_uri,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, 1),
        (1.5, 1.5),
        ([1.0, {"n": 2.0}], [1, {"n": 2}]),
        ({"a": {"b": 3.0}}, {"a": {"b": 3}}),
    ],
)
def test_convert_floats_with_no_fraction_to_ints_recursive(value, expected):
    assert normalize_integers(value) == expected


def test_sorted_dict_str_preserves_semantically_significant_list_order():
    assert stable_value({"b": 1, "a": [2, 1]}) == "{a: [2, 1], b: 1}"


@pytest.mark.parametrize("numeric", [1, 1.0])
def test_hashes_normalised_integer_values_consistently(numeric):
    rendered, token = render_config(
        {"Count": numeric, "Name": "job-{idempotence_token}"},
        inputs={"x": 1},
        region="us-west-2",
    )
    assert rendered["Count"] == 1
    assert len(token) == 16


def test_input_and_image_placeholder_rendering():
    rendered, _ = render_config(
        {"Name": "{inputs.name}", "Image": "{images.main}", "Nested": "x-{inputs.name}"},
        inputs={"name": "demo"},
        images={"main": "repo/image:tag"},
        region="us-west-2",
    )
    assert rendered == {"Name": "demo", "Image": "repo/image:tag", "Nested": "x-demo"}


def test_idempotence_token_is_truncated_to_sagemaker_name_limit():
    rendered, token = render_config(
        {"ModelName": ("xgboost-random-string-1234567890123456789012345678-{idempotence_token}")},
        region="us-west-2",
    )
    assert len(token) == 16
    assert len(rendered["ModelName"]) == 63
    assert rendered["ModelName"].endswith(token[:12])


@pytest.mark.parametrize("group", ["inputs", "images"])
def test_missing_placeholder_fails(group):
    with pytest.raises(ValueError, match="Missing value"):
        render_config({"Value": f"{{{group}.missing}}"}, region="us-west-2")


def test_region_runtime_input_wins():
    assert resolve_region("us-east-1", {"region": "us-west-2"}) == "us-west-2"


def test_region_is_required():
    with pytest.raises(ValueError, match="Region parameter is required"):
        resolve_region(None, None)


@pytest.mark.parametrize("region", ["us-west-2", "cn-north-1"])
def test_triton_image_uri_is_region_rendered(region):
    resolved = resolve_images({"triton": triton_image_uri("24.01")}, region)["triton"]
    assert region in resolved
    assert "24.01-py3" in resolved
