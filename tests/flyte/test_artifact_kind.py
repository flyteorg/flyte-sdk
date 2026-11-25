"""The reserved kind discriminator: how it is written, and how it reads back.

An artifact's kind is stored under a reserved `flyte.io/kind` attr rather than a
typed proto field, so these tests pin the two things that convention depends on:
the precedence when several sources disagree, and the fallback for artifacts
published before the key existed.
"""

import pytest
from flyteidl2.artifact import artifact_pb2
from flyteidl2.core import artifact_id_pb2

from flyte.artifacts import KIND_KEY, Metadata
from flyte.artifacts._metadata import resolve_attrs
from flyte.remote import Artifact


def _stored(user_metadata=None, card_type: str = "") -> Artifact:
    """An Artifact as the service would return it, with only the kind signals set."""
    card = artifact_id_pb2.ArtifactCard(uri="s3://c", format="html", type=card_type) if card_type else None
    return Artifact(
        artifact_pb2.Artifact(
            artifact_id=artifact_pb2.ArtifactIdentifier(
                name=artifact_pb2.ArtifactName(org="o", project="p", domain="d", name="a"),
                version="v1",
            ),
            spec=artifact_pb2.ArtifactSpec(
                info=artifact_id_pb2.ArtifactInfo(user_metadata=user_metadata or {}, card=card),
            ),
        )
    )


# --- writing ---------------------------------------------------------------


def test_create_model_metadata_stamps_kind():
    """A model is identifiable without relying on the attr key shape or a card."""
    md = Metadata.create_model_metadata(name="m")
    assert resolve_attrs(md)[KIND_KEY] == "model"


def test_kind_field_is_merged_into_attrs():
    md = Metadata(name="m", kind="data")
    assert resolve_attrs(md)[KIND_KEY] == "data"


def test_explicit_reserved_key_beats_kind_field():
    """Writing the namespaced key by hand is deliberate, so it wins."""
    md = Metadata(name="m", kind="data", attrs={KIND_KEY: "model"})
    assert resolve_attrs(md)[KIND_KEY] == "model"


def test_create_model_metadata_overrides_user_kind():
    """create_model_metadata is unambiguous about what it builds: model-specific
    keys are documented to win, and kind is one of them."""
    md = Metadata.create_model_metadata(name="m", attrs={KIND_KEY: "data"})
    assert resolve_attrs(md)[KIND_KEY] == "model"


def test_user_attrs_are_preserved_alongside_kind():
    md = Metadata(name="m", kind="model", attrs={"team": "ml"})
    attrs = resolve_attrs(md)
    assert attrs == {"team": "ml", KIND_KEY: "model"}


def test_no_kind_leaves_attrs_untouched():
    """Nothing is stamped unless asked for, so unrelated artifacts stay clean."""
    assert resolve_attrs(Metadata(name="m", attrs={"team": "ml"})) == {"team": "ml"}
    assert resolve_attrs(Metadata(name="m")) == {}


# --- reading back ----------------------------------------------------------


def test_kind_reads_the_reserved_key():
    assert _stored({KIND_KEY: "model"}).kind == "model"


def test_kind_falls_back_to_card_type():
    """Artifacts published before the reserved key existed still classify."""
    assert _stored(card_type="model").kind == "model"


def test_reserved_key_wins_over_card_type():
    """The card describes rendering; the key describes the thing itself."""
    assert _stored({KIND_KEY: "data"}, card_type="model").kind == "data"


def test_kind_defaults_to_generic():
    """Never None and never raising: unlabelled and not-a-model are the same answer."""
    assert _stored().kind == "generic"


def test_unrecognized_kind_falls_through_to_generic():
    """A value outside the known set is not passed through as-is; callers can rely
    on the return being one of the three."""
    assert _stored({KIND_KEY: "sometthing-else"}).kind == "generic"


def test_model_without_a_card_still_reports_model():
    """The hf_model case: a prefetched model whose repo had no README carries no
    card at all, so the reserved key is the only signal."""
    md = Metadata.create_model_metadata(name="m")
    assert _stored(resolve_attrs(md)).kind == "model"


# --- validation at the write side ------------------------------------------
#
# `test_unrecognized_kind_falls_through_to_generic` above pins the *read* side's
# tolerance, which is right for artifacts published by older SDKs or other writers.
# These pin the other half: this SDK should never be able to write such a value in
# the first place, because the read side's tolerance is indistinguishable from
# silence once it has.


def test_unknown_kind_is_rejected_at_construction():
    """`Kind` is a Literal, and nothing enforces a Literal at runtime."""
    for bad in ("modl", "Model", "MODEL", "dataset", ""):
        with pytest.raises(ValueError, match="kind must be one of"):
            Metadata(name="m", kind=bad)  # type: ignore[arg-type]


def test_unknown_kind_would_otherwise_be_silently_discarded():
    """Why construction is the right place to catch it: a typo'd kind reaches the
    wire under the reserved attr and reads back as "generic", so the artifact is
    misclassified permanently with nothing raised at either end."""
    assert _stored({KIND_KEY: "Model"}).kind == "generic"


def test_known_kinds_are_accepted():
    """The guard must not narrow the documented set."""
    for good in ("model", "data", "generic"):
        assert resolve_attrs(Metadata(name="m", kind=good))[KIND_KEY] == good  # type: ignore[arg-type]
    assert resolve_attrs(Metadata(name="m", kind=None)) == {}


def test_kind_validation_is_shared_with_the_imperative_path():
    """`Artifact.create(kind=...)` and `Artifact.listall(kind=...)` take the same
    values as `Metadata(kind=...)`, for the same reason `_validate_parents` is
    shared: the declarative and imperative paths must agree."""
    from flyte.artifacts._metadata import _validate_kind

    with pytest.raises(ValueError, match="kind must be one of"):
        _validate_kind("modl")
    assert _validate_kind("model") is None
    assert _validate_kind(None) is None


def test_artifact_name_is_required_at_construction():
    """`Artifact.create` has always rejected a missing name; the declarative path
    reached the wire with an empty one."""
    with pytest.raises(ValueError, match="name must not be empty"):
        Metadata(name="")
    with pytest.raises(TypeError, match="name must be a string"):
        Metadata(name=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="name must be a string"):
        Metadata(name=3)  # type: ignore[arg-type]
    assert Metadata(name="m").name == "m"
