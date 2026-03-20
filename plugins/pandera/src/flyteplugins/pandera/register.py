from .transformers.pandas import register_pandera_type_transformers


def register_type_transformers() -> None:
    """Register all flyteplugins-pandera type transformers."""
    register_pandera_type_transformers()
