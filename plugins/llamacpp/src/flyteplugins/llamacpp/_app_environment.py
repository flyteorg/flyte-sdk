from __future__ import annotations

import importlib.util
import os
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Literal, Optional, Union

import flyte.app
import rich.repr
from flyte import Environment, Image, Resources, SecretRequest
from flyte.app import ArtifactValue, Parameter, RunOutput
from flyte.app._types import Port
from flyte.models import SerializationContext

from flyteplugins.llamacpp._image import DEFAULT_LLAMA_CPP_IMAGE

# The app-serde requires the primary container to be named "app" and to exist in the pod spec
# (flyte/app/_runtime/app_serde.py); the object-store mount attaches to that container.
_APP_CONTAINER = "app"
# Stable volume/mount name so a cloned env that re-applies delivery is idempotent by name.
_MODEL_VOLUME_NAME = "model"
# Vendor-neutral env a platform advertises for the read-only, object-store-backed model PVC, plus
# the mount path the runtime shim resolves an artifact's object-store URI against.
_MODEL_PVC_ENV_VAR = "FLYTE_MODEL_PVC"
_MODEL_MOUNT_ENV_VAR = "FLYTE_MODEL_MOUNT"
# The env var the `llama-cpp-fserve` shim reads the resolved model URI from when serving a bound
# ArtifactValue over a mount (a non-downloading Parameter injects it; also records lineage).
_MODEL_URI_ENV_VAR = "FLYTE_LLAMACPP_MODEL_URI"
# Private in-container mount points for downloaded weights (under /tmp so the non-root image user
# can create them; the released Flyte base image runs non-root and a mount at / or /root fails).
_DOWNLOAD_MODEL_MOUNT = "/tmp/flyte/model"
_DOWNLOAD_DRAFT_MOUNT = "/tmp/flyte/draft-model"


def _is_artifact(value: object) -> bool:
    """True if `value` is a deploy-time-resolved artifact reference (not a literal subpath string)."""
    return isinstance(value, (ArtifactValue, RunOutput))


def _shell_safe(args: list[str]) -> list[str]:
    """Quote args that have to survive a trip through a shell.

    `fserve` runs the app with `Popen(" ".join(args), shell=True)` (flyte/_bin/serve.py), so
    any token carrying spaces or quotes -- a `--chat-template` blob, say -- reaches the server
    mangled unless it is quoted here. `shlex.quote` is the identity function for ordinary
    tokens, so this is a no-op for everything else.

    Tokens starting with `$` are left alone: `fserve` expands those against the container
    environment *before* joining, and quoting would turn the marker into a literal.
    """
    return [arg if arg.startswith("$") else shlex.quote(arg) for arg in args]


def build_fserve_command(
    *,
    model_id: str,
    port: int,
    model_dir: str | None = None,
    model_hf_path: str | None = None,
    draft_model_dir: str | None = None,
    draft_model_hf_path: str | None = None,
    extra_args: Iterable[str] = (),
    host: str = "0.0.0.0",
) -> list[str]:
    """Build the shell-safe `llama-cpp-fserve` argv that serves a GGUF model with llama.cpp.

    This is exactly the command `LlamaCppAppEnvironment` runs, exposed for serving shapes that
    are not a Flyte App -- e.g. a llama.cpp server running as a native sidecar in a task pod.
    The `llama-cpp-fserve` shim resolves a `--model-dir`/`--draft-model-dir` (a directory whose
    concrete `.gguf` filename is unknown until runtime) to a file and execs llama-server.

    Provide the model as either a mounted directory (`model_dir`, resolved by the shim) or a
    HuggingFace repo (`model_hf_path`, which llama-server downloads at startup) -- exactly one.
    The optional speculative-decoding draft is the same, via `draft_model_dir` /
    `draft_model_hf_path` (at most one). `extra_args` are appended verbatim (e.g. `--ctx-size`,
    `--flash-attn`); if they already carry `--host`, the default host bind is skipped.
    """
    extra = list(extra_args)
    if bool(model_dir) == bool(model_hf_path):
        raise ValueError("exactly one of model_dir or model_hf_path must be provided")
    if draft_model_dir and draft_model_hf_path:
        raise ValueError("provide at most one of draft_model_dir or draft_model_hf_path")
    model_args = ["--model-dir", model_dir] if model_dir else ["--hf-repo", model_hf_path]
    if draft_model_dir:
        draft_args = ["--draft-model-dir", draft_model_dir]
    elif draft_model_hf_path:
        draft_args = ["--hf-repo-draft", draft_model_hf_path]
    else:
        draft_args = []
    host_args = [] if "--host" in extra else ["--host", host]
    return _shell_safe(
        [
            "llama-cpp-fserve",
            *model_args,
            "--alias",
            model_id,
            *host_args,
            "--port",
            str(port),
            *draft_args,
            *extra,
        ]
    )


@dataclass
class ObjectStoreMount:
    """Serve the model **in place** from a read-only, object-store-backed mount (no copy).

    Set `LlamaCppAppEnvironment.mount = ObjectStoreMount(...)` to select this over the default
    `download` delivery. The weights are read lazily from a read-only PVC that exposes the data
    bucket (gcsfuse on GKE, Mountpoint-S3 on EKS) -- nothing is copied to local disk and the mount
    releases cleanly when the app scales to zero.

    Sizing note: the CSI file-cache lands on the pod's ephemeral storage, so `resources.disk` must
    cover ~the model size, else the pod is evicted mid-load.

    Args:
        pvc: Claim name of the read-only, object-store-backed PVC to mount. Defaults to the
            platform-advertised `FLYTE_MODEL_PVC` env when unset, so on a configured dataplane the
            app author names nothing. The PVC exposes the data bucket root.
        mount_path: Where the PVC is mounted inside the container. Defaults to `/tmp/models`.
        model_path: Either an `ArtifactValue`/`RunOutput` served **directly** -- its object-store
            URI is resolved at deploy, streamed in place from the bucket-root mount, and the
            App->artifact lineage edge is recorded automatically -- or a literal relative subpath
            string under the mount (e.g. `"qwen3-32b/Q4_K_M"`; empty serves from the mount root).
        draft_model_path: Optional relative subpath under the mount to a draft GGUF for
            speculative decoding.
    """

    pvc: str = ""
    mount_path: str = "/tmp/models"
    model_path: str | RunOutput | ArtifactValue = ""
    draft_model_path: str = ""

    def resolved_pvc(self) -> str:
        """RO PVC claim name: explicit `pvc`, else the platform-injected `FLYTE_MODEL_PVC`."""
        return self.pvc or os.environ.get(_MODEL_PVC_ENV_VAR, "")

    def _subdir(self, subpath: str) -> str:
        base = self.mount_path.rstrip("/")
        return f"{base}/{subpath.strip('/')}" if subpath else base


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class LlamaCppAppEnvironment(flyte.app.AppEnvironment):
    """
    App environment backed by llama.cpp (llama-server) for serving GGUF models.

    This environment serves an OpenAI-compatible endpoint (under `/v1`) plus the llama.cpp
    Web UI, with the specified GGUF model and configuration. llama.cpp shines where vLLM and
    SGLang don't fit: quantized GGUF weights, partial CPU offload of models larger than VRAM,
    and CPU-only serving.

    The model reaches the server one of three ways (exactly one):
      * `model_path` -- a remote artifact/URI **downloaded** into the container (default).
      * `model_hf_path` -- a HuggingFace GGUF repo llama-server downloads at startup.
      * `mount` -- an `ObjectStoreMount`: the weights are read **in place** from a
        read-only object-store PVC (no copy; scale-to-zero clean).

    Args:
        name: The name of the application.
        port: Port the application listens on. Defaults to 8080.
        requests: Compute resource requests for application.
        secrets: Secrets that are requested for application.
        limits: Compute resource limits for application.
        env_vars: Environment variables to set for the application.
        scaling: Scaling configuration for the app environment.
        domain: Domain to use for the app.
        cluster_pool: The target cluster_pool where the app should be deployed.
        requires_auth: Whether the public URL requires authentication.
        type: Type of app.
        extra_args: Extra args to pass to `llama-server`, e.g. `"--ctx-size 32768 --jinja"`.
            Run `llama-server --help` or see
            https://github.com/ggml-org/llama.cpp/tree/master/tools/server for details.
        model_path: Remote path to the GGUF weights -- a directory containing `.gguf` file(s)
            or a direct path to one (e.g. `s3://bucket/path/to/model`), or a
            `RunOutput`/`ArtifactValue` resolved at deploy time. Downloaded into the container;
            binding an `ArtifactValue` also records the App->artifact lineage edge. Mutually
            exclusive with `model_hf_path` and `mount`.
        model_hf_path: HuggingFace GGUF repo, optionally with a quant tag (e.g.
            `ggml-org/gemma-3-4b-it-GGUF:Q4_K_M`). Passed as `--hf-repo`; llama-server downloads
            it at startup. Mutually exclusive with `model_path` and `mount`.
        mount: An `ObjectStoreMount` to serve the weights in place from a read-only
            object-store PVC instead of downloading. Mutually exclusive with `model_path` /
            `model_hf_path`; the model (and optional draft) subpaths live on the mount.
        model_id: Model id exposed by the server (llama-server's `--alias`) -- the string API
            clients send as `"model"`. Optional; defaults to the app `name` when unset.
        draft_model_path: Remote path/artifact to a draft GGUF for speculative decoding,
            downloaded alongside the target and passed as `--model-draft` (download mode only).
        draft_model_hf_path: HuggingFace GGUF repo for the draft model (HF mode only).
        pod_annotations: Extra pod annotations to set (any delivery mode). On GKE the gcsfuse
            sidecar injector requires `{"gke-gcsfuse/volumes": "true"}` for an object-store mount.
    """

    port: int | Port = 8080
    type: str = "llama.cpp"
    image: str | Image | Literal["auto"] = DEFAULT_LLAMA_CPP_IMAGE

    extra_args: str | list[str] = ""
    model_id: str = ""
    # Source (exactly one): download an artifact, pull from HF at startup, or mount in place.
    model_path: str | RunOutput | ArtifactValue = ""
    model_hf_path: str = ""
    mount: Optional[ObjectStoreMount] = None
    # Draft (optional) -- follows the model's delivery mode.
    draft_model_path: str | RunOutput | ArtifactValue = ""
    draft_model_hf_path: str = ""
    # General pod annotations (not delivery-specific); applied to the app pod in any mode.
    pod_annotations: Optional[dict[str, str]] = None

    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}

        self._validate()

        extra_args = self._resolve_extra_args()
        model_dir, model_hf, draft_dir, draft_hf = self._resolve_sources()

        # The GGUF filename inside a mounted/downloaded directory is unknown at deploy time, so
        # mounted/downloaded weights go through the `llama-cpp-fserve` shim, which resolves
        # `--model-dir` / `--draft-model-dir` to concrete .gguf paths and execs llama-server.
        self.args = build_fserve_command(
            model_id=self.model_id or self.name,
            port=self.get_port().port,
            model_dir=model_dir,
            model_hf_path=model_hf,
            draft_model_dir=draft_dir,
            draft_model_hf_path=draft_hf,
            extra_args=extra_args,
        )

        self._apply_delivery()

        self.links = [flyte.app.Link(path="/", title="llama.cpp Web UI", is_relative=True), *self.links]

        if self.image is None or self.image == "auto":
            self.image = DEFAULT_LLAMA_CPP_IMAGE

        super().__post_init__()

    # --- validation -----------------------------------------------------------------------

    def _validate(self) -> None:
        cls = type(self).__name__
        if self._server is not None:
            raise ValueError(f"server function cannot be set for {cls}")
        if self._on_startup is not None:
            raise ValueError(f"on_startup function cannot be set for {cls}")
        if self._on_shutdown is not None:
            raise ValueError(f"on_shutdown function cannot be set for {cls}")
        if self.args:
            raise ValueError(f"args cannot be set for {cls}. Use `extra_args` to add extra arguments.")
        if self.parameters:
            raise ValueError(f"parameters cannot be set for {cls}")

        sources = [bool(self.model_path), bool(self.model_hf_path), self.mount is not None]
        if sum(sources) != 1:
            raise ValueError("exactly one of model_path, model_hf_path, or mount must be set")

        if self.mount is not None:
            if self.draft_model_path or self.draft_model_hf_path:
                raise ValueError("with mount, set the draft via mount.draft_model_path (not draft_model_path/_hf_path)")
            if not self.mount.resolved_pvc():
                raise ValueError(
                    f"mount needs a read-only PVC: set mount.pvc, or have the platform advertise one "
                    f"via the {_MODEL_PVC_ENV_VAR} env var."
                )
        else:
            if self.draft_model_path and self.draft_model_hf_path:
                raise ValueError("draft_model_path and draft_model_hf_path cannot be set at the same time")
            if self.model_hf_path and self.draft_model_path:
                raise ValueError(
                    "draft_model_path (downloaded) cannot pair with model_hf_path; use draft_model_hf_path"
                )

    # --- source resolution ----------------------------------------------------------------

    def _resolve_extra_args(self) -> list[str]:
        if isinstance(self.extra_args, str):
            return shlex.split(self.extra_args)
        return list(self.extra_args)

    def _resolve_sources(self) -> tuple[str | None, str | None, str | None, str | None]:
        """Return (model_dir, model_hf_path, draft_model_dir, draft_model_hf_path) for the shim.

        Exactly one of model_dir / model_hf_path is non-None; the drafts are optional.
        """
        if self.mount is not None:
            if _is_artifact(self.mount.model_path):
                # The resolved artifact URI arrives at runtime in $FLYTE_LLAMACPP_MODEL_URI (injected
                # by the non-downloading Parameter in _apply_delivery); the shim strips
                # scheme://bucket and joins the mount. Emit a bare `$NAME` (fserve expands it).
                model_dir = "$" + _MODEL_URI_ENV_VAR
            else:
                model_dir = self.mount._subdir(self.mount.model_path)
            draft_dir = self.mount._subdir(self.mount.draft_model_path) if self.mount.draft_model_path else None
            return model_dir, None, draft_dir, None
        if self.model_hf_path:
            return None, self.model_hf_path, None, (self.draft_model_hf_path or None)
        # download mode
        draft_dir = _DOWNLOAD_DRAFT_MOUNT if self.draft_model_path else None
        return _DOWNLOAD_MODEL_MOUNT, None, draft_dir, (self.draft_model_hf_path or None)

    # --- delivery -------------------------------------------------------------------------

    def _apply_delivery(self) -> None:
        """Attach the object-store mount (mount mode) and/or download parameters + pod annotations."""
        pod_template = self.pod_template
        if self.mount is not None:
            if _is_artifact(self.mount.model_path):
                # Serve the bound artifact directly: one non-downloading Parameter injects its
                # resolved object-store URI (for the shim) and records the App->artifact edge.
                self.parameters = [
                    Parameter(name="model", value=self.mount.model_path, download=False, env_var=_MODEL_URI_ENV_VAR)
                ]
                self.env_vars[_MODEL_MOUNT_ENV_VAR] = self.mount.mount_path
            if importlib.util.find_spec("kubernetes") is not None:
                # Attaching the volume mutates the pod template via the kubernetes client models, a
                # deploy-time-only concern -- skip inside the serving image (kubernetes absent; the
                # volume is already on the deploy-time-serialized pod spec).
                pod_template = self._attach_mount(pod_template)
        else:
            params = self._download_parameters()
            if params:
                self.parameters = params

        if self.pod_annotations:
            pod_template = self._apply_pod_annotations(pod_template)

        if pod_template is not None:
            self.pod_template = pod_template

    def _download_parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        if self.model_path:
            params.append(
                Parameter(name="model_path", value=self.model_path, download=True, mount=_DOWNLOAD_MODEL_MOUNT)
            )
        if self.draft_model_path:
            params.append(
                Parameter(
                    name="draft_model_path", value=self.draft_model_path, download=True, mount=_DOWNLOAD_DRAFT_MOUNT
                )
            )
        return params

    def _base_pod_template(self, pod_template: "str | flyte.PodTemplate | None") -> "flyte.PodTemplate":
        from kubernetes.client.models import V1Container, V1PodSpec

        if isinstance(pod_template, str):
            raise ValueError("a string pod-template reference cannot be extended; pass a PodTemplate object or none")
        if pod_template is None:
            pod_template = flyte.PodTemplate(
                primary_container_name=_APP_CONTAINER,
                pod_spec=V1PodSpec(containers=[V1Container(name=_APP_CONTAINER)]),
            )
        if pod_template.primary_container_name != _APP_CONTAINER:
            raise ValueError(f"the pod template's primary container must be '{_APP_CONTAINER}'")
        return pod_template

    def _attach_mount(self, pod_template: "str | flyte.PodTemplate | None") -> "flyte.PodTemplate":
        """Mount the read-only, object-store-backed model PVC into the app pod.

        Delegates to `PodTemplate.allow_object_store_volume` (the unprivileged, Knative-friendly,
        scale-to-zero-clean object-store CSI primitive).
        """
        assert self.mount is not None
        pod_template = self._base_pod_template(pod_template)
        return pod_template.allow_object_store_volume(
            claim_name=self.mount.resolved_pvc(),
            mount_path=self.mount.mount_path,
            read_only=True,
            name=_MODEL_VOLUME_NAME,
        )

    def _apply_pod_annotations(self, pod_template: "str | flyte.PodTemplate | None") -> "flyte.PodTemplate":
        if importlib.util.find_spec("kubernetes") is None:
            # Serving image: annotations are already on the deploy-time-serialized pod spec.
            return pod_template  # type: ignore[return-value]
        pod_template = self._base_pod_template(pod_template)
        merged = {**(pod_template.annotations or {}), **(self.pod_annotations or {})}
        return replace(pod_template, annotations=merged)

    # --- framework hooks ------------------------------------------------------------------

    def container_args(self, serialization_context: SerializationContext) -> list[str]:
        """Return the container arguments for llama.cpp."""
        if isinstance(self.args, str):
            return shlex.split(self.args)
        return self.args or []

    def clone_with(
        self,
        name: str,
        image: Optional[Union[str, Image, Literal["auto"]]] = None,
        resources: Optional[Resources] = None,
        env_vars: Optional[dict[str, str]] = None,
        secrets: Optional[SecretRequest] = None,
        depends_on: Optional[list[Environment]] = None,
        description: Optional[str] = None,
        interruptible: Optional[bool] = None,
        **kwargs: Any,
    ) -> LlamaCppAppEnvironment:
        port = kwargs.pop("port", None)
        extra_args = kwargs.pop("extra_args", None)
        model_id = kwargs.pop("model_id", None)
        # Source/draft/mount overrides: only re-set when explicitly passed. For the string source
        # fields None clears to "" (empty); mount/pod_annotations accept None as "unset".
        overrides: dict[str, Any] = {}
        for key in ("model_path", "model_hf_path", "draft_model_path", "draft_model_hf_path"):
            if key in kwargs:
                overrides[key] = kwargs.pop(key) or ""
        for key in ("mount", "pod_annotations"):
            if key in kwargs:
                overrides[key] = kwargs.pop(key)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        kwargs = self._get_kwargs()
        kwargs["name"] = name
        kwargs["args"] = None
        kwargs["parameters"] = None
        if image is not None:
            kwargs["image"] = image
        if resources is not None:
            kwargs["resources"] = resources
        if env_vars is not None:
            kwargs["env_vars"] = env_vars
        if secrets is not None:
            kwargs["secrets"] = secrets
        if depends_on is not None:
            kwargs["depends_on"] = depends_on
        if description is not None:
            kwargs["description"] = description
        if interruptible is not None:
            kwargs["interruptible"] = interruptible
        if port is not None:
            kwargs["port"] = port
        if extra_args is not None:
            kwargs["extra_args"] = extra_args
        if model_id is not None:
            kwargs["model_id"] = model_id
        kwargs.update(overrides)
        return replace(self, **kwargs)
