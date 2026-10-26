"""Unit tests for DeepInfer sampling / structured-output helpers."""

import pytest

from flyteplugins.deepinfer import sampling_params, structured_output


def test_sampling_defaults_left_to_upstream():
    assert sampling_params() == {}


def test_sampling_overrides():
    p = sampling_params(temperature=0, max_tokens=512, top_logprobs=3)
    assert p["temperature"] == 0
    assert p["max_tokens"] == 512
    assert p["top_logprobs"] == 3


def test_sampling_passthrough_unknown():
    p = sampling_params(custom_flag=True)
    assert p["custom_flag"] is True


def test_structured_output_regex():
    so = structured_output("regex", r"[A-Z]{2}-[0-9]{3}")
    assert so == {"schema_type": "regex", "schema": r"[A-Z]{2}-[0-9]{3}"}


def test_structured_output_json_extra():
    so = structured_output("json", "{}", trigger_token_ids=[1, 2, 3], bound_token_id=4)
    assert so["schema_type"] == "json"
    assert so["schema"] == "{}"
    assert so["trigger_token_ids"] == [1, 2, 3]
    assert so["bound_token_id"] == 4


def test_structured_output_rejects_bad_type():
    with pytest.raises(ValueError):
        structured_output("xml", "...")


def test_composition():
    p = sampling_params(temperature=0, so_args=structured_output("ebnf", 'root ::= "x"'))
    assert p["so_args"]["schema_type"] == "ebnf"
