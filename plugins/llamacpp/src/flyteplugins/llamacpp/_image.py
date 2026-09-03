"""Container image with llama.cpp built from source (CUDA-enabled by default)."""

from __future__ import annotations

import base64

import flyte

from flyteplugins.llamacpp._constants import (
    CUDA_HOME,
    CUDA_STUB_LIB,
    CUDA_TOOLKIT_PACKAGE,
    DEFAULT_CUDA_ARCH,
    LLAMA_CPP_INSTALL_DIR,
    LLAMA_CPP_REPO,
    NODE_HOME,
    NODE_VERSION,
)

_NODE_TARBALL = f"node-v{NODE_VERSION}-linux-x64"


def _node_install_commands() -> list[str]:
    return [
        f"wget -q https://nodejs.org/dist/v{NODE_VERSION}/{_NODE_TARBALL}.tar.xz -O /tmp/node.tar.xz",
        f"mkdir -p {NODE_HOME} && tar -xJf /tmp/node.tar.xz -C {NODE_HOME} --strip-components=1 && rm /tmp/node.tar.xz",
    ]


def _run_script(script: str) -> str:
    """Serialize a shell script for an image-builder RUN step without quoting hazards.

    The remote image builder mangles embedded quoting: double quotes inside a command
    are dropped, so a fat-binary `-DCMAKE_CUDA_ARCHITECTURES="80;86;89"` reaches `sh`
    as `80;86;89` and the `;` splits it into separate commands ("sh: 1: 86: not").
    Base64-encoding the whole script and decoding it into `sh` at build time keeps
    every character intact regardless of how the builder serializes RUN commands.
    """
    encoded = base64.b64encode(script.encode()).decode()
    return f"echo {encoded} | base64 -d | sh"


def _clone_commands(repo: str, ref: str | None) -> list[str]:
    commands = [f"git clone {repo} {LLAMA_CPP_INSTALL_DIR}"]
    if ref is not None:
        commands.append(f"cd {LLAMA_CPP_INSTALL_DIR} && git checkout {ref}")
    return commands


def build_llama_cpp_image(
    *,
    name: str = "llama-cpp-app-image",
    cuda: bool = True,
    cuda_arch: str = DEFAULT_CUDA_ARCH,
    repo: str = LLAMA_CPP_REPO,
    ref: str | None = None,
) -> flyte.Image:
    """Build a Debian image with llama-server compiled from source.

    Args:
        name: Name of the image.
        cuda: Build with CUDA support (GGML_CUDA=ON). Set to False for a CPU-only image.
        cuda_arch: Target CUDA architecture(s) for the kernel build, as a ";"-separated
            list of compute capabilities (e.g. "89" for L4/L40S, "80;86;89;90" for a fat
            binary that also covers A100/A10/H100). Ignored when `cuda=False`.
        repo: Git repository to build llama.cpp from.
        ref: Git ref (tag, branch, or commit) to check out. None builds the default
            branch tip; pin a release tag (e.g. "b6148") for reproducible builds.
    """
    image = flyte.Image.from_debian_base(name=name).with_apt_packages(
        "git",
        "build-essential",
        "cmake",
        "pkg-config",
        "wget",
        "curl",
        "ca-certificates",
        "libnuma-dev",
        "libssl-dev",
        "pciutils",
        "libcurl4-openssl-dev",
        "xz-utils",
    )

    if cuda:
        image = image.with_commands(
            [
                (
                    "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/"
                    "cuda-keyring_1.1-1_all.deb"
                ),
                "dpkg -i cuda-keyring_1.1-1_all.deb",
                "apt-get update",
                f"apt-get install -y {CUDA_TOOLKIT_PACKAGE}",
            ]
        )
        cmake_configure = (
            f"LIBRARY_PATH={CUDA_STUB_LIB}:$LIBRARY_PATH "
            f"cmake -S {LLAMA_CPP_INSTALL_DIR} -B {LLAMA_CPP_INSTALL_DIR}/build "
            "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON "
            "-DCMAKE_BUILD_TYPE=Release "
            f'-DCMAKE_CUDA_ARCHITECTURES="{cuda_arch}" '
            f'-DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,{CUDA_STUB_LIB}" '
            f'-DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,{CUDA_STUB_LIB}"'
        )
        cmake_build_prefix = f"LIBRARY_PATH={CUDA_STUB_LIB}:$LIBRARY_PATH "
        stub_commands = [f"ln -sf {CUDA_STUB_LIB}/libcuda.so {CUDA_STUB_LIB}/libcuda.so.1"]
    else:
        cmake_configure = (
            f"cmake -S {LLAMA_CPP_INSTALL_DIR} -B {LLAMA_CPP_INSTALL_DIR}/build "
            "-DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release"
        )
        cmake_build_prefix = ""
        stub_commands = []

    image = image.with_commands(
        _node_install_commands()
        + _clone_commands(repo, ref)
        + stub_commands
        + [
            # The cmake configure carries a ";"-separated CMAKE_CUDA_ARCHITECTURES;
            # ship it via _run_script so the ";" and quotes survive the builder.
            _run_script(cmake_configure),
            # Build llama-server. The llama-ui-assets target runs an npm build of
            # tools/ui to embed the Web UI; put node/npm on PATH. The UI's .npmrc
            # sets engine-strict=true, which makes a transitive dep's Node bound a
            # fatal EBADENGINE; override it so install proceeds (Node here satisfies
            # the UI's real Vite requirement).
            _run_script(
                f"PATH={NODE_HOME}/bin:$PATH npm_config_engine_strict=false "
                f"{cmake_build_prefix}"
                f"cmake --build {LLAMA_CPP_INSTALL_DIR}/build --config Release "
                "-j $(nproc) --target llama-server"
            ),
        ]
    )

    env_vars = {
        "PATH": f"{LLAMA_CPP_INSTALL_DIR}/build/bin:{CUDA_HOME}/bin:$PATH"
        if cuda
        else (f"{LLAMA_CPP_INSTALL_DIR}/build/bin:$PATH"),
        # Under /tmp so `-hf`-style downloads work with the non-root user the
        # released Flyte base image runs as.
        "LLAMA_CACHE": "/tmp/llama.cpp/cache",
    }
    if cuda:
        env_vars["CUDA_HOME"] = CUDA_HOME

    # The plugin itself provides the `llama-cpp-fserve` entrypoint (and pulls in flyte).
    return image.with_env_vars(env_vars).with_pip_packages("flyteplugins-llamacpp", pre=True)


DEFAULT_LLAMA_CPP_IMAGE = build_llama_cpp_image()
