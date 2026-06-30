from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from omegaconf import MISSING, OmegaConf

from flyteplugins.omegaconf.report import log_yaml, to_html, to_yaml


def test_to_yaml_hides_payload_shape_for_kind_and_values_keys():
    cfg = OmegaConf.create({"kind": "training-job", "values": {"lr": 0.001, "epochs": 10}})

    assert to_yaml(cfg) == "kind: training-job\nvalues:\n  lr: 0.001\n  epochs: 10\n"


def test_to_yaml_handles_missing_paths_and_tuples():
    cfg = OmegaConf.create(
        {
            "data_path": MISSING,
            "epochs": 10,
            "model_path": Path("/tmp/model.bin"),
            "layers": (64, 128),
        }
    )

    assert to_yaml(cfg) == "data_path: ???\nepochs: 10\nmodel_path: /tmp/model.bin\nlayers:\n- 64\n- 128\n"


def test_to_yaml_renders_utf8_bytes_as_text():
    cfg = OmegaConf.create({"payload": b"default-token"})

    assert to_yaml(cfg) == "payload: default-token\n"


def test_to_yaml_renders_non_utf8_bytes_as_base64_text():
    cfg = OmegaConf.create({"payload": b"\xff\xfe"})

    assert to_yaml(cfg) == "payload: //4=\n"


def test_to_html_escapes_title_and_yaml():
    cfg = OmegaConf.create({"name": "<unsafe>"})

    rendered = to_html(cfg, title="<Config>")

    assert "<style>" in rendered
    assert ".omegaconf-report pre" in rendered
    assert "&lt;Config&gt;" in rendered
    assert "&lt;unsafe&gt;" in rendered
    assert "<unsafe>" not in rendered


@pytest.mark.asyncio
async def test_log_yaml_writes_report_tab():
    cfg = OmegaConf.create({"lr": 0.001})
    tab = Mock()

    with patch("flyte.report.get_tab", return_value=tab):
        await log_yaml.aio(cfg, title="Input config", tab="Config")

    tab.log.assert_called_once()
    html = tab.log.call_args.args[0]
    assert "Input config" in html
    assert "lr: 0.001" in html
