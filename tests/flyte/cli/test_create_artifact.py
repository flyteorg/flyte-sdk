"""Tests for the `--card` options on `flyte create artifact`."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from flyte.cli._create import _infer_card_format, artifact


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("card.html", "html"),
        ("card.HTM", "html"),
        ("card.md", "md"),
        ("card.markdown", "md"),
        ("card.yml", "yaml"),
        ("card", "html"),
    ],
)
def test_infer_card_format(filename: str, expected: str):
    assert _infer_card_format(Path(filename)) == expected


def test_infer_card_format_rejects_unknown_extension():
    import rich_click as click

    with pytest.raises(click.BadParameter):
        _infer_card_format(Path("card.pdf"))


def _run(args, tmp_path: Path):
    """Invoke `flyte create artifact` with the remote calls stubbed out."""
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")

    # log_level must be a real int: the CommandBase error handler compares it against logging.DEBUG.
    cfg = MagicMock(log_level=logging.INFO)
    created_card = object()

    with (
        patch("flyte.io.File.from_local_sync", return_value="file-literal") as from_local,
        patch("flyte.artifacts.Card.create_from", return_value=created_card) as create_from,
        patch("flyte.remote.Artifact.create") as create,
    ):
        create.return_value = MagicMock(name="art", version="v1")
        result = CliRunner().invoke(
            artifact,
            ["my_model", "--from-file", str(model), *args],
            obj=cfg,
        )
    return result, from_local, create_from, create, created_card


def test_card_is_uploaded_and_attached(tmp_path: Path):
    card = tmp_path / "model_card.html"
    card.write_text("<h1>Model</h1>")

    result, _, create_from, create, created_card = _run(["--card", str(card), "--card-type", "model"], tmp_path)

    assert result.exit_code == 0, result.output
    create_from.assert_called_once()
    assert create_from.call_args.kwargs["local_path"] == card
    assert create_from.call_args.kwargs["format"] == "html"
    assert create_from.call_args.kwargs["card_type"] == "model"
    assert create.call_args.kwargs["card"] is created_card


def test_explicit_card_format_overrides_extension(tmp_path: Path):
    card = tmp_path / "card.txt"
    card.write_text("<h1>Model</h1>")

    result, _, create_from, _, _ = _run(["--card", str(card), "--card-format", "html"], tmp_path)

    assert result.exit_code == 0, result.output
    assert create_from.call_args.kwargs["format"] == "html"


def test_unknown_card_extension_is_rejected(tmp_path: Path):
    card = tmp_path / "card.pdf"
    card.write_bytes(b"%PDF")

    result, _, create_from, create, _ = _run(["--card", str(card)], tmp_path)

    assert result.exit_code != 0
    assert "cannot infer a card format" in result.output
    create_from.assert_not_called()
    create.assert_not_called()


def test_no_card_passes_none(tmp_path: Path):
    result, _, create_from, create, _ = _run([], tmp_path)

    assert result.exit_code == 0, result.output
    create_from.assert_not_called()
    assert create.call_args.kwargs["card"] is None
