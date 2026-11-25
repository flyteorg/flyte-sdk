"""Kubernetes namespace configuration shared by task definitions."""

import re

TASK_NAMESPACE_CONFIG_KEY = "kubernetes.namespace"


def validate_namespace(namespace: str | None) -> None:
    if namespace is None:
        return
    if not isinstance(namespace, str):
        raise TypeError(f"Expected namespace to be of type str, got {type(namespace)}")
    if len(namespace) > 63 or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", namespace) is None:
        raise ValueError(
            "namespace must be a Kubernetes namespace name: 1-63 lowercase letters, digits or hyphens, "
            "starting and ending with a letter or digit"
        )
