from __future__ import annotations

import shlex
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, Optional, Union

import flyte.app
import rich.repr
from flyte import Environment, Image, Resources, SecretRequest
from flyte.app import ModelDeliveryMixin
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


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class LlamaCppAppEnvironment(ModelDeliveryMixin, flyte.app.AppEnvironment):
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
              when the app scales to zero. In this mode `model_path` is either an
              `ArtifactValue`/`RunOutput` (served **directly**: resolved to its object-store URI
              at deploy, streamed from the bucket-root mount, lineage recorded automatically) or
              a *relative subpath* string under the mount (e.g. `"qwen3-32b/Q4_K_M"`).
              `draft_model_path` is a relative subpath string. See `model_pvc`.
        model_pvc: Claim name of the read-only, object-store-backed PVC to mount when
            `model_delivery="fuse"`. Defaults to the platform-advertised `FLYTE_MODEL_PVC` env
            when unset, so on a configured dataplane the app author names nothing. The PVC
            exposes the data bucket root; the served weights are read from the artifact's key
            under it. Ignored outside `"fuse"`.
        model_mount_path: Where the model PVC is mounted inside the container (fuse mode).
            Defaults to `/tmp/models`; `model_path`/`draft_model_path` are resolved relative
            to it.
        fuse_pod_annotations: Extra pod annotations to set in fuse mode — the one
            vendor-specific knob. On GKE the gcsfuse sidecar injector requires
            `{"gke-gcsfuse/volumes": "true"}`; on EKS (Mountpoint-S3) no annotation is needed.
        model_artifact: Fuse-mode-with-a-subpath only. The `ArtifactValue`/`RunOutput` the
            served subpath corresponds to, bound purely for **lineage** — records the
            App→artifact edge without downloading. Redundant (and rejected) when `model_path`
            is *itself* an `ArtifactValue`, which already carries lineage; and rejected in
            download mode (bind `model_path` to an `ArtifactValue` there instead).
    """

    port: int | Port = 8080
    type: str = "llama.cpp"
    image: str | Image | Literal["auto"] = DEFAULT_LLAMA_CPP_IMAGE
    # The env var the `llama-cpp-fserve` shim reads the resolved model URI from when serving an
    # `ArtifactValue` over fuse. `ModelDeliveryMixin` binds the artifact to a non-downloading
    # Parameter with this env var, and references it (`${...}`) as the `--model-dir` value.
    _fuse_model_uri_env_var: ClassVar[str] = "FLYTE_LLAMACPP_MODEL_URI"

    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}

        # Shared model-serving validation (server/model_id/path-xor/args/parameters + delivery mode).
        self._validate_model_delivery()

        extra_args = self._resolve_extra_args()

        # The GGUF filename inside a mounted directory is unknown at deploy time, so mounted
        # weights go through the `llama-cpp-fserve` shim, which resolves `--model-dir` /
        # `--draft-model-dir` to concrete .gguf paths and execs llama-server. Download mode points
        # at the downloaded-Parameter mount; fuse-by-subpath at the RO PVC mount; fuse-by-artifact
        # at a `${FLYTE_LLAMACPP_MODEL_URI}` env ref the shim resolves against the mount at runtime.
        model_dir, draft_dir = self._resolve_model_dirs()

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

        # Attach the fuse PVC (+ artifact/lineage param) or the download parameters.
        self._apply_model_delivery()

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
