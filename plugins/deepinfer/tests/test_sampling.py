"""Unit tests for `generate`'s sampling / structured-output options."""

import json

import pytest
from pydantic import BaseModel

from flyteplugins.deepinfer._sampling import build_sampling


def test_unset_options_left_to_upstream():
    params, parse = build_sampling()
    assert params == {}
    assert parse("x") == "x"


def test_options_and_passthrough():
    params, _ = build_sampling(temperature=0, max_tokens=512, top_logprobs=3)
    assert params == {"temperature": 0, "max_tokens": 512, "top_logprobs": 3}


def test_regex_and_grammar():
    assert build_sampling(regex="[a-z]+")[0]["so_args"] == {"schema_type": "regex", "schema": "[a-z]+"}
    assert build_sampling(grammar='root ::= "x"')[0]["so_args"] == {"schema_type": "ebnf", "schema": 'root ::= "x"'}


class Item(BaseModel):
    name: str
    count: int


def test_json_schema_pydantic():
    params, parse = build_sampling(json_schema=Item)
    assert params["so_args"]["schema_type"] == "json"
    assert json.loads(params["so_args"]["schema"]) == Item.model_json_schema()
    assert parse('{"name": "a", "count": 2}') == Item(name="a", count=2)


def test_json_schema_dict_and_string():
    schema = {"type": "object"}
    params, parse = build_sampling(json_schema=schema)
    assert params["so_args"]["schema"] == json.dumps(schema)
    assert parse('{"a": 1}') == {"a": 1}
    params, _ = build_sampling(json_schema='{"type": "object"}')
    assert params["so_args"]["schema"] == '{"type": "object"}'


def test_at_most_one_constraint():
    with pytest.raises(ValueError, match="at most one"):
        build_sampling(json_schema=Item, regex="x")


def test_bad_json_schema_type():
    with pytest.raises(TypeError):
        build_sampling(json_schema=42)
