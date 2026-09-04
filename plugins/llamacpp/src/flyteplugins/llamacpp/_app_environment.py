from __future__ import annotations

import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Union

import flyte.app
import rich.repr
from flyte import Environment, Image, Resources, SecretRequest
from flyte.app import ArtifactValue, Parameter, RunOutput
from flyte.app._types import Port
from flyte.models import SerializationContext

from flyteplugins.llamacpp._image import DEFAULT_LLAMA_CPP_IMAGE


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


# The app-serde requires the primary container to be named "app" and to exist in the pod
# spec (flyte/app/_runtime/app_serde.py); a fuse mount attaches to that container.
_APP_CONTAINER = "app"


def _attach_model_pvc(
    pod_template: str | flyte.PodTemplate | None,
    *,
    claim: str,
    mount_path: str,
    annotations: dict[str, str] | None,
) -> flyte.PodTemplate:
    """Mount a read-only, object-store-backed PVC into the app pod for fuse delivery.

    The PVC (a static, CSI-backed volume provisioned outside the SDK — gcsfuse on GKE,
    Mountpoint-S3 on EKS) is referenced by claim name and mounted read-only into the primary
    `app` container at `mount_path`; the served weights are read in place from a subdirectory
    under it. A PVC volume is Knative-friendly (on its podspec allow-list) and releases
    cleanly on scale-to-zero, unlike an inline CSI volume or a node device-plugin.

    Any `annotations` are set on the pod (the sole vendor-specific bit: GKE's gcsfuse sidecar
    injector requires `gke-gcsfuse/volumes: "true"`; Mountpoint-S3 needs none). An existing
    `PodTemplate` is extended in place (its readiness probe / scheduling survive); a `None`
    template is created fresh. A string pod-template reference cannot be mounted.
    """
    from kubernetes.client.models import (
        V1Container,
        V1PersistentVolumeClaimVolumeSource,
        V1PodSpec,
        V1Volume,
        V1VolumeMount,
    )

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
    if pod_template.pod_spec is None:
        pod_template.pod_spec = V1PodSpec(containers=[V1Container(name=_APP_CONTAINER)])

    spec = pod_template.pod_spec
    # Idempotent: clone_with() re-runs __post_init__ carrying the already-mounted template,
    # so bail if the model volume is already attached rather than double-adding it.
    if any(v.name == "model" for v in (spec.volumes or [])):
        return pod_template
    if not any(c.name == _APP_CONTAINER for c in (spec.containers or [])):
        spec.containers = [*(spec.containers or []), V1Container(name=_APP_CONTAINER)]

    spec.volumes = [
        *(spec.volumes or []),
        V1Volume(
            name="model",
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name=claim, read_only=True),
        ),
    ]
    for c in spec.containers:
        if c.name == _APP_CONTAINER:
            c.volume_mounts = [
                *(c.volume_mounts or []),
                V1VolumeMount(name="model", mount_path=mount_path, read_only=True),
            ]

    if annotations:
        pod_template.annotations = {**(pod_template.annotations or {}), **annotations}
    return pod_template


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class LlamaCppAppEnvironment(flyte.app.AppEnvironment):
    """
    App environment backed by llama.cpp (llama-server) for serving GGUF models.

    This environment serves an OpenAI-compatible endpoint (under `/v1`) plus the llama.cpp
    Web UI, with the specified GGUF model and configuration. llama.cpp shines where vLLM and
    SGLang don't fit: quantized GGUF weights, partial CPU offload of models larger than VRAM,
    and CPU-only serving.

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
            or a direct path to one (e.g. s3://bucket/path/to/model), or a
            `RunOutput`/`ArtifactValue` resolved at deploy time. The weights are downloaded
            into the container and the served `.gguf` is located at startup (for sharded
            models, the `-00001-of-` shard is picked; llama-server finds the rest).
        model_hf_path: Hugging Face GGUF repo, optionally with a quant tag (e.g.
            `ggml-org/gemma-3-4b-it-GGUF:Q4_K_M`). Passed to llama-server as `--hf-repo`,
            which downloads the weights at startup.
        model_id: Model id exposed by the server (llama-server's `--alias`).
        draft_model_path: Remote path to the draft model GGUF used for speculative decoding,
            or a `RunOutput`/`ArtifactValue` resolved at deploy time. Downloaded alongside the
            target model and passed as `--model-draft`. Tune the speculation via `extra_args`
            (`--draft-max`, `--draft-min`, `--gpu-layers-draft`, ...).
        draft_model_hf_path: Hugging Face GGUF repo for the draft model, as an alternative to
            `draft_model_path`. Passed as `--hf-repo-draft`.
        model_delivery: How the weights reach the container.
            - `"download"` (default): the bound `model_path` is copied into the container's
              local disk (via a `Parameter`) before the server starts. Simple; the whole
              model lands on the node's ephemeral disk.
            - `"fuse"`: the weights are read in place from a read-only, object-store-backed
              PVC mounted into the pod by a CSI driver (gcsfuse on GKE, Mountpoint-S3 on EKS)
              — lazy first-touch, nothing copied to local disk, and the mount releases cleanly
              when the app scales to zero. In this mode `model_path`/`draft_model_path` are
              *relative subpaths* under the mount (e.g. `"qwen3-32b/Q4_K_M"`), not remote URIs.
              The PVC itself is cloud infrastructure provisioned outside the SDK (see
              `model_pvc`).
        model_pvc: Claim name of the read-only, object-store-backed PVC to mount when
            `model_delivery="fuse"`. The PVC exposes the data bucket's model prefix; each
            served model is a subdirectory under it. Required for `"fuse"`, ignored otherwise.
        model_mount_path: Where the model PVC is mounted inside the container (fuse mode).
            Defaults to `/tmp/models`; `model_path`/`draft_model_path` are resolved relative
            to it.
        fuse_pod_annotations: Extra pod annotations to set in fuse mode — the one
            vendor-specific knob. On GKE the gcsfuse sidecar injector requires
            `{"gke-gcsfuse/volumes": "true"}`; on EKS (Mountpoint-S3) no annotation is needed.
    """

    port: int | Port = 8080
    type: str = "llama.cpp"
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
    image: str | Image | Literal["auto"] = DEFAULT_LLAMA_CPP_IMAGE
    # Under /tmp, and that is not cosmetic: ``fserve`` materializes each mounted Parameter
    # through ``_ensure_dest_writable``, which needs the *image's* user to be able to create the
    # parent directory. The released Flyte base image runs non-root, so a mount at the
    # filesystem root -- or under /root -- fails with "Permission denied" before the engine ever
    # starts. /tmp is writable for any user and lives on the same overlay filesystem the weights
    # are already budgeted against by ``disk=``.
    _model_mount_path: str = field(default="/tmp/flyte/model", init=False)
    _draft_model_mount_path: str = field(default="/tmp/flyte/draft-model", init=False)

    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}

        if self._server is not None:
            raise ValueError("server function cannot be set for LlamaCppAppEnvironment")

        if self._on_startup is not None:
            raise ValueError("on_startup function cannot be set for LlamaCppAppEnvironment")

        if self._on_shutdown is not None:
            raise ValueError("on_shutdown function cannot be set for LlamaCppAppEnvironment")

        if self.model_id == "":
            raise ValueError("model_id must be defined")

        if self.model_path == "" and self.model_hf_path == "":
            raise ValueError("model_path or model_hf_path must be defined")
        if self.model_path != "" and self.model_hf_path != "":
            raise ValueError("model_path and model_hf_path cannot be set at the same time")

        if self.draft_model_path != "" and self.draft_model_hf_path != "":
            raise ValueError("draft_model_path and draft_model_hf_path cannot be set at the same time")

        fuse = self.model_delivery == "fuse"
        if fuse:
            # In fuse mode the weights are read from a mounted PVC, so a HF-repo source
            # (which downloads at startup) is incompatible, and the paths are relative
            # subpaths under the mount rather than remote URIs / bound artifacts.
            if not self.model_pvc:
                raise ValueError("model_pvc is required when model_delivery='fuse'")
            if self.model_hf_path or self.draft_model_hf_path:
                raise ValueError("model_hf_path/draft_model_hf_path cannot be used with model_delivery='fuse'")
            if not isinstance(self.model_path, str) or not isinstance(self.draft_model_path, str):
                raise ValueError(
                    "model_delivery='fuse' locates weights by a relative subpath under the mount; "
                    "pass model_path/draft_model_path as strings (e.g. 'qwen3-32b/Q4_K_M'), not a "
                    "RunOutput/ArtifactValue. Resolve the artifact to its FUSE-visible subpath at "
                    "deploy time and pass that string."
                )
        elif self.model_pvc or self.fuse_pod_annotations:
            raise ValueError("model_pvc/fuse_pod_annotations only apply when model_delivery='fuse'")

        if self.args:
            raise ValueError("args cannot be set for LlamaCppAppEnvironment. Use `extra_args` to add extra arguments.")

        if isinstance(self.extra_args, str):
            extra_args = shlex.split(self.extra_args)
        else:
            extra_args = list(self.extra_args)

        # The GGUF filename inside a mounted directory is unknown at deploy time, so mounted
        # weights go through the `llama-cpp-fserve` shim, which resolves `--model-dir` /
        # `--draft-model-dir` to concrete .gguf paths and execs llama-server. In download mode
        # the shim looks under the downloaded-Parameter mount; in fuse mode it looks under the
        # RO PVC mount at the model's relative subpath.
        if fuse:
            base = self.model_mount_path.rstrip("/")
            model_dir = f"{base}/{str(self.model_path).strip('/')}"
            draft_dir = f"{base}/{str(self.draft_model_path).strip('/')}" if self.draft_model_path else ""
        else:
            model_dir = self._model_mount_path
            draft_dir = self._draft_model_mount_path

        # llama-server binds 127.0.0.1 by default (unreachable from outside the container), so
        # build_fserve_command adds --host 0.0.0.0 unless extra_args override it.
        self.args = build_fserve_command(
            model_id=self.model_id,
            port=self.get_port().port,
            model_dir=model_dir if self.model_path else None,
            model_hf_path=self.model_hf_path or None,
            draft_model_dir=draft_dir if self.draft_model_path else None,
            draft_model_hf_path=self.draft_model_hf_path or None,
            extra_args=extra_args,
        )

        if self.parameters:
            raise ValueError("parameters cannot be set for LlamaCppAppEnvironment")

        if fuse:
            # No download Parameters: the weights are read in place from the RO PVC. Attach
            # the PVC volume/mount (+ any vendor sidecar annotation) to the pod spec instead.
            self.pod_template = _attach_model_pvc(
                self.pod_template,
                claim=self.model_pvc,
                mount_path=self.model_mount_path,
                annotations=self.fuse_pod_annotations,
            )
        else:
            parameters: list[Parameter] = []
            if self.model_path:
                parameters.append(
                    Parameter(
                        name="model_path",
                        value=self.model_path,
                        download=True,
                        mount=self._model_mount_path,
                    )
                )
            if self.draft_model_path:
                parameters.append(
                    Parameter(
                        name="draft_model_path",
                        value=self.draft_model_path,
                        download=True,
                        mount=self._draft_model_mount_path,
                    )
                )
            if parameters:
                self.parameters = parameters

        self.links = [flyte.app.Link(path="/", title="llama.cpp Web UI", is_relative=True), *self.links]

        if self.image is None or self.image == "auto":
            self.image = DEFAULT_LLAMA_CPP_IMAGE

        super().__post_init__()

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
        if "model_path" in kwargs:
            set_model_path = True
            model_path = kwargs.pop("model_path", "") or ""
        else:
            set_model_path = False
            model_path = self.model_path
        if "model_hf_path" in kwargs:
            set_model_hf_path = True
            model_hf_path = kwargs.pop("model_hf_path", "") or ""
        else:
            set_model_hf_path = False
            model_hf_path = self.model_hf_path
        if "draft_model_path" in kwargs:
            set_draft_model_path = True
            draft_model_path = kwargs.pop("draft_model_path", "") or ""
        else:
            set_draft_model_path = False
            draft_model_path = self.draft_model_path
        if "draft_model_hf_path" in kwargs:
            set_draft_model_hf_path = True
            draft_model_hf_path = kwargs.pop("draft_model_hf_path", "") or ""
        else:
            set_draft_model_hf_path = False
            draft_model_hf_path = self.draft_model_hf_path
        model_id = kwargs.pop("model_id", None)

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
        if set_model_path:
            kwargs["model_path"] = model_path
        if set_model_hf_path:
            kwargs["model_hf_path"] = model_hf_path
        if set_draft_model_path:
            kwargs["draft_model_path"] = draft_model_path
        if set_draft_model_hf_path:
            kwargs["draft_model_hf_path"] = draft_model_hf_path
        if model_id is not None:
            kwargs["model_id"] = model_id
        return replace(self, **kwargs)
