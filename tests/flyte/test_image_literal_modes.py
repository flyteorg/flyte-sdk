"""
`project_install_mode` and `copy_style` are `Literal`-typed layer fields, and nothing
enforces a `Literal` at runtime. Every consumer of them is a two-way
`if value == "<one option>": ... else: ...`, so an unrecognized value never fails -- it
silently selects the other branch, and which branch that is depends on which consumer
you land in.

These tests pin both halves: that construction now rejects such a value, and *why* it has
to be construction that rejects it.
"""

import hashlib
from pathlib import Path

import pytest

from flyte._image import (
    _CODE_BUNDLE_COPY_STYLES,
    _PROJECT_INSTALL_MODES,
    CodeBundleLayer,
    CopyConfig,
    Image,
    PixiProject,
    PoetryProject,
    UVProject,
    resolve_code_bundle_layer,
)


def _project_layers(mode):
    """One of each class that carries `project_install_mode`. Paths need not exist:
    path validation happens at build time, in `validate()`."""
    return (
        UVProject(pyproject=Path("pyproject.toml"), project_install_mode=mode),
        PoetryProject(pyproject=Path("pyproject.toml"), poetry_lock=Path("poetry.lock"), project_install_mode=mode),
        PixiProject(manifest=Path("pixi.toml"), project_install_mode=mode),
    )


# --- the guard -------------------------------------------------------------


@pytest.mark.parametrize("bad", ["dependencies", "install", "Install_Project", "DEPENDENCIES_ONLY", "", None])
def test_unknown_project_install_mode_is_rejected_at_construction(bad):
    """All three project layers. Only `UVProject` had a consumer that rejected a bad mode
    (the remote builder's `match`), and only on the remote path -- so a guard covering just
    that one would leave the other two exactly as silent as before."""
    for build in (
        lambda: UVProject(pyproject=Path("pyproject.toml"), project_install_mode=bad),
        lambda: PoetryProject(
            pyproject=Path("pyproject.toml"), poetry_lock=Path("poetry.lock"), project_install_mode=bad
        ),
        lambda: PixiProject(manifest=Path("pixi.toml"), project_install_mode=bad),
    ):
        with pytest.raises(ValueError, match="Invalid project_install_mode"):
            build()


@pytest.mark.parametrize("good", ["dependencies_only", "install_project"])
def test_known_project_install_modes_are_accepted(good):
    """The guard must not narrow the documented set."""
    for layer in _project_layers(good):
        assert layer.project_install_mode == good


@pytest.mark.parametrize("bad", ["modules", "loaded modules", "All", "none", "", None])
def test_unknown_copy_style_is_rejected_at_construction(bad):
    with pytest.raises(ValueError, match="Invalid copy_style"):
        CodeBundleLayer(copy_style=bad)


@pytest.mark.parametrize("good", ["loaded_modules", "all"])
def test_known_copy_styles_are_accepted(good):
    assert CodeBundleLayer(copy_style=good).copy_style == good


def test_the_allowed_sets_are_derived_from_the_type_aliases():
    """Restating them would let a guard drift from the `Literal` it guards."""
    assert _PROJECT_INSTALL_MODES == ("dependencies_only", "install_project")
    assert _CODE_BUNDLE_COPY_STYLES == ("loaded_modules", "all")


def test_the_public_image_dsl_rejects_it_too():
    """`Image.with_*` is where a user actually types the value."""
    image = Image.from_debian_base()
    with pytest.raises(ValueError, match="Invalid project_install_mode"):
        image.with_uv_project(Path("pyproject.toml"), project_install_mode="dependencies")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid copy_style"):
        image.with_code_bundle(copy_style="modules")  # type: ignore[arg-type]


# --- why it has to be construction -----------------------------------------
#
# Each of these smuggles a bad value past `__post_init__` (the frozen dataclass's own
# back door) to show what the consumers do with it. They are the reason the guard sits
# at construction rather than in any one consumer.


def _smuggle(layer, field, value):
    object.__setattr__(layer, field, value)
    return layer


def test_code_bundle_copy_style_consumers_pick_opposite_branches(monkeypatch):
    """The sharpest evidence that no value outside the set can be meaningful: the two
    consumers of `copy_style` disagree about what an unknown one means.

    `update_hash` is `if copy_style == "loaded_modules": ... else: <hash the whole tree>`,
    so an unknown value hashes as "all". `resolve_code_bundle_layer` is
    `if copy_style == "all": ... else: <keep the layer, copy only imported modules>`, so
    the same value builds as "loaded_modules". The image would be tagged with a digest
    computed over files it does not contain.
    """
    import flyte._utils as _utils

    root = Path(__file__).parent
    hashed = []
    monkeypatch.setattr(_utils, "update_hasher_for_source", lambda source, *a, **kw: hashed.append(source))

    # Consumer 1 -- update_hash takes the "all" branch, which hands the whole root_dir to
    # the hasher. ("loaded_modules" hands it a list of individual module files instead.)
    smuggled = _smuggle(CodeBundleLayer(copy_style="all", root_dir=root), "copy_style", "modules")
    smuggled.update_hash(hashlib.md5())
    assert hashed == [root]


def test_the_code_bundle_resolver_branches_the_other_way(tmp_path):
    """The other half of the contradiction. `resolve_code_bundle_layer` keys its branch on
    `copy_style == "all"`, so *everything else* -- an unknown value included -- keeps the
    CodeBundleLayer and copies only imported modules.

    Together with the test above, that is the contradiction: the same unknown value hashes
    as "all" and builds as "loaded_modules". No value outside the set can be meaningful,
    which is why the guard belongs at construction rather than in either consumer.
    """

    def kinds_for(copy_style):
        image = Image.from_debian_base()
        object.__setattr__(image, "_layers", (CodeBundleLayer(copy_style=copy_style),))
        return [type(layer) for layer in resolve_code_bundle_layer(image, "none", tmp_path)._layers]

    assert kinds_for("all") == [CopyConfig]
    assert kinds_for("loaded_modules") == [CodeBundleLayer]


def test_an_unknown_project_install_mode_means_install_project_to_every_local_consumer():
    """`update_hash` -- and, identically, the docker builder and the pixi layer helper --
    are `if mode == "dependencies_only": ... else: ...`, so an unknown mode silently
    installs the whole project: it copies the entire project directory into the image
    rather than just the manifest.

    Only the remote builder's `match` has a `case _: raise ValueError(...)`, and only for
    `UVProject`. The invariant was already written down there; nothing else enforced it.
    """
    here = Path(__file__).parent
    layer = _smuggle(
        PoetryProject(pyproject=here / "pyproject.toml", poetry_lock=here / "poetry.lock"),
        "project_install_mode",
        "install",
    )

    unknown_digest = hashlib.md5()
    layer.update_hash(unknown_digest)
    install_digest = hashlib.md5()
    PoetryProject(
        pyproject=here / "pyproject.toml",
        poetry_lock=here / "poetry.lock",
        project_install_mode="install_project",
    ).update_hash(install_digest)
    assert unknown_digest.hexdigest() == install_digest.hexdigest()


def test_the_guard_matches_copyconfigs_existing_one():
    """`CopyConfig.path_type` is the same shape -- a `Literal`-typed layer field -- and
    `_image.py` has always rejected an out-of-set value for it at construction. These
    fields were the exception, not the rule."""
    with pytest.raises(ValueError, match="Invalid path_type"):
        CopyConfig(path_type=2, src=Path("."), dst=".")  # type: ignore[arg-type]
