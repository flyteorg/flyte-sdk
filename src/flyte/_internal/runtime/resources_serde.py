from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from flyteidl2.core import tasks_pb2

from flyte._resources import CPUBaseType, DeviceClass, Resources

if TYPE_CHECKING:
    from flyte._image import Image

ACCELERATOR_DEVICE_MAP = {
    "A100": "nvidia-tesla-a100",
    "A100_80G": "nvidia-a100-80gb",
    "A100 80G": "nvidia-a100-80gb",
    "H100": "nvidia-tesla-h100",
    "A10": "nvidia-a10",
    "A10G": "nvidia-a10g",
    "A100G": "nvidia-a100g",
    "L4": "nvidia-l4",
    "L40s": "nvidia-l40s",
    "L4_VWS": "nvidia-l4-vws",
    "K80": "nvidia-tesla-k80",
    "M60": "nvidia-tesla-m60",
    "P4": "nvidia-tesla-p4",
    "P100": "nvidia-tesla-p100",
    "RTX_PRO_4500": "nvidia-rtx-pro-4500",
    "RTX_PRO_6000": "nvidia-rtx-pro-6000",
    "RTX PRO 6000": "nvidia-rtx-pro-6000",
    "T4": "nvidia-tesla-t4",
    "V100": "nvidia-tesla-v100",
    "V5E": "tpu-v5-lite-podslice",
    "V5P": "tpu-v5p-slice",
    "V6E": "tpu-v6e-slice",
}

DEPRECATED_ACCELERATOR_NOTATIONS = ["A100 80G", "RTX PRO 6000"]

# APPLE_SILICON's wire value, for flyteidl2 releases that predate it.
_APPLE_SILICON_DEVICE_CLASS = getattr(tasks_pb2.GPUAccelerator, "APPLE_SILICON", 5)

_DeviceClassToProto: Dict[DeviceClass, "tasks_pb2.GPUAccelerator.DeviceClass"] = {
    "GPU": tasks_pb2.GPUAccelerator.NVIDIA_GPU,
    "TPU": tasks_pb2.GPUAccelerator.GOOGLE_TPU,
    "NEURON": tasks_pb2.GPUAccelerator.AMAZON_NEURON,
    "AMD_GPU": tasks_pb2.GPUAccelerator.AMD_GPU,
    "HABANA_GAUDI": tasks_pb2.GPUAccelerator.HABANA_GAUDI,
    "APPLE_SILICON": _APPLE_SILICON_DEVICE_CLASS,
}


def _get_cpu_resource_entry(cpu: CPUBaseType) -> tasks_pb2.Resources.ResourceEntry:
    return tasks_pb2.Resources.ResourceEntry(
        name=tasks_pb2.Resources.ResourceName.CPU,
        value=str(cpu),
    )


def _get_memory_resource_entry(memory: str) -> tasks_pb2.Resources.ResourceEntry:
    return tasks_pb2.Resources.ResourceEntry(
        name=tasks_pb2.Resources.ResourceName.MEMORY,
        value=memory,
    )


def _get_gpu_resource_entry(gpu: int) -> tasks_pb2.Resources.ResourceEntry:
    return tasks_pb2.Resources.ResourceEntry(
        name=tasks_pb2.Resources.ResourceName.GPU,
        value=str(gpu),
    )


def _get_gpu_extended_resource_entry(resources: Resources) -> Optional[tasks_pb2.GPUAccelerator]:
    if resources is None:
        return None
    if resources.gpu is None or isinstance(resources.gpu, int):
        return None
    device = resources.get_device()
    if device is None:
        return None

    device_class = _DeviceClassToProto.get(device.device_class, tasks_pb2.GPUAccelerator.NVIDIA_GPU)
    if device.device is None:
        raise RuntimeError("Device type must be specified for GPU string.")
    else:
        device_type = device.device

    # Notations with spaces are deprecated, but we still support them for backwards compatibility.
    if device_type in DEPRECATED_ACCELERATOR_NOTATIONS:
        from flyte._logging import logger

        logger.warning(
            f"Accelerator notation {device_type} is deprecated. Please use the equivalent notation using underscores "
            "instead of spaces. See https://www.union.ai/docs/v2/union/api-reference/flyte-sdk/flyte/resources/ viable "
            "accelerator references."
        )
    device_type = ACCELERATOR_DEVICE_MAP.get(device_type, device_type)
    return tasks_pb2.GPUAccelerator(
        device=device_type,
        partition_size=device.partition or None,
        device_class=device_class,
    )


def _get_disk_resource_entry(disk: str) -> tasks_pb2.Resources.ResourceEntry:
    return tasks_pb2.Resources.ResourceEntry(
        name=tasks_pb2.Resources.ResourceName.EPHEMERAL_STORAGE,
        value=disk,
    )


def get_proto_extended_resources(resources: Resources | None) -> Optional[tasks_pb2.ExtendedResources]:
    """
    TODO Implement partitioning logic string handling for GPU

    Args:
        resources:
    """
    if resources is None:
        return None
    acc = _get_gpu_extended_resource_entry(resources)
    shm = resources.get_shared_memory()
    if acc is None and shm is None:
        return None
    proto_shm = None
    if shm is not None:
        proto_shm = tasks_pb2.SharedMemory(
            mount_path="/dev/shm",
            mount_name="flyte-shm",
            size_limit=shm,
        )
    return tasks_pb2.ExtendedResources(gpu_accelerator=acc, shared_memory=proto_shm)


def _convert_resources_to_resource_entries(
    resources: Resources | None,
) -> Tuple[List[tasks_pb2.Resources.ResourceEntry], List[tasks_pb2.Resources.ResourceEntry]]:
    request_entries: List[tasks_pb2.Resources.ResourceEntry] = []
    limit_entries: List[tasks_pb2.Resources.ResourceEntry] = []
    if resources is None:
        return request_entries, limit_entries
    if resources.cpu is not None:
        if isinstance(resources.cpu, tuple):
            request_entries.append(_get_cpu_resource_entry(resources.cpu[0]))
            limit_entries.append(_get_cpu_resource_entry(resources.cpu[1]))
        else:
            request_entries.append(_get_cpu_resource_entry(resources.cpu))

    if resources.memory is not None:
        if isinstance(resources.memory, tuple):
            request_entries.append(_get_memory_resource_entry(resources.memory[0]))
            limit_entries.append(_get_memory_resource_entry(resources.memory[1]))
        else:
            request_entries.append(_get_memory_resource_entry(resources.memory))

    if resources.gpu is not None:
        device = resources.get_device()
        if device is not None:
            request_entries.append(_get_gpu_resource_entry(device.quantity))

    if resources.disk is not None:
        request_entries.append(_get_disk_resource_entry(resources.disk))

    return request_entries, limit_entries


def get_proto_resources(resources: Resources | None) -> Optional[tasks_pb2.Resources]:
    """
    Get main resources IDL representation from the resources object

    Args:
        resources: User facing Resources object containing potentially both requests and limits

    Returns:
        The given resources as requests and limits
    """
    if resources is None:
        return None
    request_entries, limit_entries = _convert_resources_to_resource_entries(resources)
    if not request_entries and not limit_entries:
        return None

    return tasks_pb2.Resources(requests=request_entries, limits=limit_entries)


def get_apple_silicon_execution_env(
    resources: Resources | None, image: "Image | str | None", env_vars: Dict[str, str] | None
) -> Dict[str, str]:
    """
    Environment that tells the runtime to run a `gpu="AppleSilicon"` workload natively on the Mac, with the Python
    packages (and pip index options) to install there, taken from the image. A workload whose `env_vars` set
    `FLYTE_APPLE_SILICON_EXECUTION=container` stays in its container and calls a host-side MLX daemon instead.
    """
    import json

    device = resources.get_device() if resources is not None else None
    if device is None or device.device_class != "APPLE_SILICON":
        return {}
    if (env_vars or {}).get("FLYTE_APPLE_SILICON_EXECUTION", "host") == "container":
        return {}
    packages, pip_args = _image_host_requirements(image)
    env = {
        "_F_APPLE_SILICON_EXECUTION": "host",
        "_F_APPLE_SILICON_HOST_PACKAGES": json.dumps(packages),
    }
    if pip_args:
        env["_F_APPLE_SILICON_HOST_PIP_ARGS"] = json.dumps(pip_args)
    return env


def _image_host_requirements(image: object) -> Tuple[List[str], List[str]]:
    """
    The Python requirements an `Image` installs, and the pip index options they come from, for rebuilding its
    environment on the Mac. Only Python packages carry over; layers that can't (local wheels, a project's own code,
    system packages) are reported in a warning rather than silently dropped.
    """
    from flyte._image import (
        Image,
        PipOption,
        PipPackages,
        PoetryProject,
        PythonWheels,
        Requirements,
        UVProject,
        UVScript,
    )
    from flyte._logging import logger

    if not isinstance(image, Image):
        return [], []
    packages: List[str] = []
    pip_args: List[str] = []
    skipped: List[str] = []

    def add_options(layer: PipOption) -> None:
        if layer.index_url and "--index-url" not in pip_args:
            pip_args.extend(["--index-url", layer.index_url])
        for url in layer.extra_index_urls or ():
            if url not in pip_args:
                pip_args.extend(["--extra-index-url", url])
        if layer.pre and "--prerelease" not in pip_args:
            pip_args.extend(["--prerelease", "allow"])
        if layer.extra_args:
            skipped.append(f"pip extra_args {layer.extra_args!r}")

    for layer in image._layers:
        try:
            if isinstance(layer, Requirements):
                reqs, opts = _parse_requirements_file(Path(layer.file))
                packages.extend(reqs)
                pip_args.extend(o for o in opts if o not in pip_args or o.startswith("--"))
                add_options(layer)
            elif isinstance(layer, PipPackages):
                packages.extend(layer.packages or ())
                add_options(layer)
            elif isinstance(layer, (UVProject, PoetryProject)):
                packages.extend(_pyproject_dependencies(Path(layer.pyproject)))
                if isinstance(layer, UVProject):
                    add_options(layer)
                if layer.project_install_mode == "install_project":
                    skipped.append(f"the project in {layer.pyproject} itself (only its dependencies)")
            elif isinstance(layer, UVScript):
                packages.extend(_script_dependencies(Path(layer.script)))
                add_options(layer)
            elif isinstance(layer, PythonWheels) and layer.package_name != "flyte":
                skipped.append(f"local wheels in {layer.wheel_dir}")
        except (OSError, ValueError) as e:
            skipped.append(f"{type(layer).__name__} ({e})")
    if skipped:
        logger.warning(
            "gpu='AppleSilicon' runs on the Mac in an environment built from the image's Python packages; "
            f"these don't carry over: {'; '.join(skipped)}"
        )
    return _without_flyte(packages), pip_args


def _parse_requirements_file(path: Path, seen: set | None = None) -> Tuple[List[str], List[str]]:
    """Requirements and index options from a requirements file: continuation lines joined, `-r` includes followed."""
    seen = seen if seen is not None else set()
    path = path.resolve()
    if path in seen:
        return [], []
    seen.add(path)
    reqs: List[str] = []
    opts: List[str] = []
    text = path.read_text().replace("\\\n", " ")
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].strip() if not raw.lstrip().startswith("#") else ""
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ", "-c ", "--constraint ")):
            flag, _, target = line.partition(" ")
            if flag in ("-r", "--requirement"):
                more, more_opts = _parse_requirements_file(path.parent / target.strip(), seen)
                reqs.extend(more)
                opts.extend(more_opts)
            continue
        if line.startswith("-"):
            flag, _, value = line.replace("=", " ", 1).partition(" ")
            if flag in ("-i", "--index-url"):
                opts.extend(["--index-url", value.strip()])
            elif flag == "--extra-index-url":
                opts.extend(["--extra-index-url", value.strip()])
            elif flag in ("-f", "--find-links"):
                opts.extend(["--find-links", value.strip()])
            elif flag == "--pre":
                opts.extend(["--prerelease", "allow"])
            continue  # hashes and other per-file options don't affect what gets installed
        reqs.append(line.split(" --hash", 1)[0].strip())
    return reqs, opts


def _pyproject_dependencies(pyproject: Path) -> List[str]:
    data = _load_toml(pyproject.read_text())
    deps = list(data.get("project", {}).get("dependencies", []))
    poetry = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    for name, spec in poetry.items():
        if name == "python":
            continue
        version = spec if isinstance(spec, str) else spec.get("version", "*") if isinstance(spec, dict) else "*"
        deps.append(name if version in ("*", "") else f"{name}{_poetry_constraint(version)}")
    return deps


def _poetry_constraint(version: str) -> str:
    if version.startswith(("^", "~")):
        return f">={version[1:]}"
    return version if version[:1] in "<>=!" else f"=={version}"


def _script_dependencies(script: Path) -> List[str]:
    """The dependencies in a script's inline (PEP 723) metadata."""
    import re

    m = re.search(r"(?ms)^# /// script\s*$(?P<body>.*?)^# ///\s*$", script.read_text())
    if not m:
        return []
    body = "\n".join(line[2:] if line.startswith("# ") else line[1:] for line in m.group("body").splitlines())
    return list(_load_toml(body).get("dependencies", []))


def _load_toml(text: str) -> dict:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        raise ValueError("reading pyproject metadata needs Python 3.11+")
    return tomllib.loads(text)


def _without_flyte(packages: List[str]) -> List[str]:
    """
    Drop flyte itself: the host environment gets the host's own flyte. The asyncssh the default image installs
    alongside flyte (its `ssh` extra, for ssh debug in the pod) goes with it.
    """
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    from flyte._debug.constants import ASYNCSSH_REQUIREMENT

    out = []
    for spec in packages:
        if spec == ASYNCSSH_REQUIREMENT:
            continue
        try:
            if canonicalize_name(Requirement(spec).name) == "flyte":
                continue
        except InvalidRequirement:
            pass
        out.append(spec)
    return out
