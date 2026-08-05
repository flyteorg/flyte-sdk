from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def hydra_conf(tmp_path: Path) -> Path:
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "config.yaml").write_text(
        """data:
  path: s3://bucket/train
  dataset: imagenet
training:
  epochs: 3
  batch_size: 16
model:
  name: resnet50
task_env:
  pipeline:
    resources:
      cpu: '2'
      memory: 4Gi
""",
    )
    return conf
