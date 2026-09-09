from __future__ import annotations

import importlib.util
import os
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

from flyte.app._parameter import ArtifactValue, Parameter, RunOutput

if TYPE_CHECKING:
    import flyte


def _is_artifact(value: object) -> bool:
    """True if `value` is a deploy-time-resolved artifact reference (not a literal string)."""
    return isinstance(value, (ArtifactValue, RunOutput))

# The app-serde requires the primary container to be named "app" and to exist in the pod
# spec (flyte/app/_runtime/app_serde.py); a fuse mount attaches to that container.
_APP_CONTAINER = "app"

# Volume/mount name for the model PVC in fuse mode. Kept stable ("model") so a cloned env
# that re-runs delivery is idempotent by name rather than stacking duplicate volumes.
_MODEL_VOLUME_NAME = "model"

# Vendor-neutral env contract for fuse delivery, read at deploy time. A platform advertises the
# read-only, object-store-backed PVC it provisions (over the data-bucket root) via FLYTE_MODEL_PVC,
# and optionally the mount path via FLYTE_MODEL_MOUNT, so the app author need not name either. The
# SDK carries no vendor-specific default; explicit fields override the env.
_MODEL_PVC_ENV_VAR = "FLYTE_MODEL_PVC"
_MODEL_MOUNT_ENV_VAR = "FLYTE_MODEL_MOUNT"


@dataclass(kw_only=True)
class ModelDeliveryMixin:
    """Shared model-delivery surface for LLM-serving app environments (llama.cpp, vLLM, SGLang).

    These plugins are structurally identical: each binds a `model_path` (a remote URI, a
    `RunOutput`, or an `ArtifactValue`) plus an optional speculative-decoding `draft_model_path`,
    and reaches the weights into the container one of two ways. Rather than have each plugin
    re-implement the fields, validation, parameter wiring, and fuse-mount plumbing in its own
    `__post_init__`, that logic lives here once.

    This is a **toolkit mixin**, deliberately not a cooperative `__post_init__` chain: it
    contributes the shared *fields* and a set of *explicit helper methods the plugin calls*.
    A plugin keeps ownership of its `__post_init__` (and its engine-specific server command),
    and simply calls the helpers in order. That avoids the fragile
    `super().__post_init__()` dance across a dataclass MRO — a plugin that forgets a call fails
    loudly at that call site rather than silently skipping delivery.

    Mix it in **before** `AppEnvironment` so its fields resolve ahead of the base's:

        @dataclass(kw_only=True)
        class LlamaCppAppEnvironment(ModelDeliveryMixin, flyte.app.AppEnvironment):
            def __post_init__(self):
                if self.env_vars is None:
                    self.env_vars = {}
                self._validate_model_delivery()
                model_dir, draft_dir = self._resolve_model_dirs()
                self.args = build_engine_command(model_dir=model_dir, draft_dir=draft_dir, ...)
                self._apply_model_delivery()   # sets self.parameters and/or self.pod_template
                super().__post_init__()

    Delivery modes (`model_delivery`):

    - ``"download"`` (default): the bound ``model_path``/``draft_model_path`` are copied into the
      container's local disk via download `Parameter`s before the server starts. Binding an
      `ArtifactValue` here records the App→artifact lineage edge for free.
    - ``"fuse"``: the weights are read in place from a read-only, object-store-backed PVC
      (gcsfuse on GKE, Mountpoint-S3 on EKS) mounted via `PodTemplate.allow_object_store_volume`
      — no copy to local disk, releases cleanly on scale-to-zero, no privilege. ``model_path``
      may be either an `ArtifactValue`/`RunOutput` — resolved to its object-store URI at deploy,
      streamed in place from the bucket-root mount, lineage recorded automatically — or a literal
      *relative subpath* string under the mount (with `model_artifact` binding lineage separately).

    The read-only PVC is named by ``model_pvc`` (explicit), else the vendor-neutral
    ``FLYTE_MODEL_PVC`` env the platform injects; the SDK carries no vendor-specific default.

    Engines whose download path differs (e.g. vLLM's blob-streaming loader) override
    `_download_parameters`; a plugin that serves an `ArtifactValue` over fuse sets
    `_fuse_model_uri_env_var` so the resolved URI reaches its runtime shim.
    """

    # Plugins that can serve a fuse-mounted ``ArtifactValue`` set this to the env var their runtime
    # shim reads the resolved model URI from (e.g. llama.cpp's ``FLYTE_LLAMACPP_MODEL_URI``). Left
    # None, ``model_path=ArtifactValue`` in fuse mode is rejected (the engine can't locate it).
    _fuse_model_uri_env_var: ClassVar[str | None] = None

    extra_args: str | list[str] = ""
    model_path: str | RunOutput | ArtifactValue = ""
    model_hf_path: str = ""
    model_id: str = ""
    draft_model_path: str | RunOutput | ArtifactValue = ""
    draft_model_hf_path: str = ""
    model_delivery: Literal["download", "fuse"] = "download"
    model_pvc: str = ""
    model_mount_path: str = "/tmp/models"
    fuse_pod_annotations: dict[str, str] | None = None
    model_artifact: RunOutput | ArtifactValue | None = None
    # Under /tmp, and that is not cosmetic: ``fserve`` materializes each mounted Parameter
    # through ``_ensure_dest_writable``, which needs the *image's* user to be able to create the
    # parent directory. The released Flyte base image runs non-root, so a mount at the
    # filesystem root -- or under /root -- fails with "Permission denied" before the engine ever
    # starts. /tmp is writable for any user and lives on the same overlay filesystem the weights
    # are already budgeted against by ``disk=``.
    _model_mount_path: str = field(default="/tmp/flyte/model", init=False)
    _draft_model_mount_path: str = field(default="/tmp/flyte/draft-model", init=False)

    @property
    def _is_fuse(self) -> bool:
        return self.model_delivery == "fuse"

    @property
    def _fuse_model_is_artifact(self) -> bool:
        """Fuse mode serving an `ArtifactValue`/`RunOutput` directly (vs a literal subpath string)."""
        return self._is_fuse and _is_artifact(self.model_path)

    def _resolved_model_pvc(self) -> str:
        """RO PVC claim name: explicit `model_pvc`, else the platform-injected `FLYTE_MODEL_PVC`."""
        return self.model_pvc or os.environ.get(_MODEL_PVC_ENV_VAR, "")

    def _validate_model_delivery(self) -> None:
        """Validate the shared model-serving invariants and the selected delivery mode.

        Raises `ValueError` on any misconfiguration. Safe to call once at the top of the
        plugin's `__post_init__`, before building the engine command.
        """
        cls = type(self).__name__
        if self._server is not None:
            raise ValueError(f"server function cannot be set for {cls}")
        if self._on_startup is not None:
            raise ValueError(f"on_startup function cannot be set for {cls}")
        if self._on_shutdown is not None:
            raise ValueError(f"on_shutdown function cannot be set for {cls}")

        if self.model_id == "":
            raise ValueError("model_id must be defined")

        if self.model_path == "" and self.model_hf_path == "":
            raise ValueError("model_path or model_hf_path must be defined")
        if self.model_path != "" and self.model_hf_path != "":
            raise ValueError("model_path and model_hf_path cannot be set at the same time")
        if self.draft_model_path != "" and self.draft_model_hf_path != "":
            raise ValueError("draft_model_path and draft_model_hf_path cannot be set at the same time")

        if self.args:
            raise ValueError(f"args cannot be set for {cls}. Use `extra_args` to add extra arguments.")
        if self.parameters:
            raise ValueError(f"parameters cannot be set for {cls}")

        if self._is_fuse:
            self._validate_fuse()
        elif self.model_pvc or self.fuse_pod_annotations or self.model_artifact is not None:
            raise ValueError(
                "model_pvc/fuse_pod_annotations/model_artifact only apply when model_delivery='fuse' "
                "(in download mode, bind model_path to an ArtifactValue for lineage instead)"
            )

    def _validate_fuse(self) -> None:
        # In fuse mode the weights are read from a mounted PVC, so a HF-repo source (which
        # downloads at startup) is incompatible.
        if not self._resolved_model_pvc():
            raise ValueError(
                f"model_delivery='fuse' needs a read-only PVC: set model_pvc, or have the platform "
                f"advertise one via the {_MODEL_PVC_ENV_VAR} env var."
            )
        if self.model_hf_path or self.draft_model_hf_path:
            raise ValueError("model_hf_path/draft_model_hf_path cannot be used with model_delivery='fuse'")

        if _is_artifact(self.model_path):
            # Serve the artifact directly: its object-store URI is resolved at deploy and read in
            # place from the bucket-root mount. Needs the engine's shim to accept the URI (via
            # _fuse_model_uri_env_var); lineage then comes from the artifact binding itself.
            if self._fuse_model_uri_env_var is None:
                raise ValueError(
                    f"{type(self).__name__} does not support model_path=ArtifactValue/RunOutput in "
                    "fuse mode; pass model_path as a relative subpath string under the mount instead."
                )
            if self.model_artifact is not None:
                raise ValueError(
                    "model_artifact is redundant when model_path is itself an ArtifactValue/RunOutput; "
                    "drop it (the artifact binding already records lineage)."
                )
        if self.draft_model_path and not isinstance(self.draft_model_path, str):
            raise ValueError(
                "model_delivery='fuse' supports a draft model only as a relative subpath string under "
                "the mount, not an ArtifactValue/RunOutput."
            )

    def _resolve_extra_args(self) -> list[str]:
        """`extra_args` normalized to a list (shell-split when given as a string)."""
        if isinstance(self.extra_args, str):
            return shlex.split(self.extra_args)
        return list(self.extra_args)

    def _resolve_model_dirs(self) -> tuple[str, str]:
        """Return `(model_dir, draft_dir)` — where the engine command should read the weights.

        Download: the private per-model mount paths the download `Parameter`s land at. Fuse: the
        model's relative subpath under the read-only PVC mount. An empty string means "not set"
        (no `model_path`/`draft_model_path`), which the caller maps to its HF-repo source.
        """
        if self._is_fuse:
            base = self.model_mount_path.rstrip("/")
            if _is_artifact(self.model_path):
                # The resolved artifact URI arrives at runtime in this env var (injected by the
                # non-downloading Parameter below); the shim strips scheme://bucket and joins the
                # mount. fserve expands a `$VAR` token (bare name, no braces) against the env
                # before exec (flyte/_bin/serve.py), and _shell_safe leaves `$`-prefixed tokens
                # unquoted, so emit `$NAME` rather than shell-style `${NAME}`.
                model_dir = "$" + str(self._fuse_model_uri_env_var)
            elif self.model_path:
                model_dir = f"{base}/{str(self.model_path).strip('/')}"
            else:
                model_dir = ""
            draft_dir = f"{base}/{str(self.draft_model_path).strip('/')}" if self.draft_model_path else ""
            return model_dir, draft_dir
        return self._model_mount_path, self._draft_model_mount_path

    def _apply_model_delivery(self) -> None:
        """Wire up delivery: attach the fuse PVC (+ artifact/lineage param) or download parameters.

        Sets `self.parameters` and, in fuse mode, `self.pod_template` (and, when serving an
        artifact, the `FLYTE_MODEL_MOUNT` env the runtime shim resolves the URI against). Call
        after building the engine command and after the shared validation.
        """
        if self._is_fuse:
            # Attaching the model volume mutates the pod template via the kubernetes client
            # models, a deploy-time-only concern. `__post_init__` also runs when `fserve`
            # reconstructs this env inside the serving image, where `kubernetes` is absent and
            # the volume is already on the deploy-time-serialized pod spec — so skip the
            # mutation there rather than hard-failing the container on import.
            if importlib.util.find_spec("kubernetes") is not None:
                self.pod_template = self._attach_fuse_volume(self.pod_template)
            param = self._fuse_parameter()
            if param is not None:
                self.parameters = [param]
            if self._fuse_model_is_artifact:
                # The shim resolves the injected artifact URI against this mount (bucket root).
                self.env_vars[_MODEL_MOUNT_ENV_VAR] = self.model_mount_path
        else:
            params = self._download_parameters()
            if params:
                self.parameters = params

    def _download_parameters(self) -> list[Parameter]:
        """Download-mode `Parameter`s: copy `model_path`/`draft_model_path` to local disk.

        Engines with a bespoke loader (e.g. vLLM's blob streaming) override this.
        """
        params: list[Parameter] = []
        if self.model_path:
            params.append(
                Parameter(name="model_path", value=self.model_path, download=True, mount=self._model_mount_path)
            )
        if self.draft_model_path:
            params.append(
                Parameter(
                    name="draft_model_path",
                    value=self.draft_model_path,
                    download=True,
                    mount=self._draft_model_mount_path,
                )
            )
        return params

    def _fuse_parameter(self) -> Parameter | None:
        """The single fuse-mode `Parameter`, if any — a non-downloading artifact binding.

        Deploy-time materialization sets the value's `resolved_version_id`, which
        `collect_artifact_ids` reads to record the App→artifact edge. `download=False` means
        nothing is copied (the bytes come from the RO PVC).

        - `model_path` is an artifact: bound with `env_var` so the resolved object-store URI is
          also injected for the runtime shim — one param does lineage *and* location.
        - else `model_artifact` set (with a literal-subpath `model_path`): lineage-only, no env
          var (the subpath already locates the weights).
        """
        if _is_artifact(self.model_path):
            return Parameter(
                name="model", value=self.model_path, download=False, env_var=self._fuse_model_uri_env_var
            )
        if self.model_artifact is not None:
            return Parameter(name="model", value=self.model_artifact, download=False)
        return None

    def _attach_fuse_volume(self, pod_template: "str | flyte.PodTemplate | None") -> "flyte.PodTemplate":
        """Mount the read-only, object-store-backed model PVC into the app pod.

        Delegates to `PodTemplate.allow_object_store_volume` (the unprivileged, Knative-friendly,
        scale-to-zero-clean object-store CSI primitive). A `None` template is created fresh with
        the required primary container; an existing one is extended (its probe/scheduling
        survive). A string pod-template reference cannot be mounted.
        """
        from kubernetes.client.models import V1Container, V1PodSpec

        import flyte

        if isinstance(pod_template, str):
            raise ValueError(
                "model_delivery='fuse' needs a PodTemplate object (or none) to attach the model "
                "volume; a string pod-template reference cannot be mounted."
            )
        if pod_template is None:
            pod_template = flyte.PodTemplate(
                primary_container_name=_APP_CONTAINER,
                pod_spec=V1PodSpec(containers=[V1Container(name=_APP_CONTAINER)]),
            )
        if pod_template.primary_container_name != _APP_CONTAINER:
            raise ValueError(f"fuse delivery requires the pod template's primary container to be '{_APP_CONTAINER}'")

        return pod_template.allow_object_store_volume(
            claim_name=self._resolved_model_pvc(),
            mount_path=self.model_mount_path,
            read_only=True,
            name=_MODEL_VOLUME_NAME,
            annotations=self.fuse_pod_annotations,
        )
