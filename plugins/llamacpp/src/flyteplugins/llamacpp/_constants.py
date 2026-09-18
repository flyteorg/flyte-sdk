"""Constants shared by the llama.cpp image build and server shim."""

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_INSTALL_DIR = "/opt/llama.cpp"
LLAMA_SERVER_BINARY = f"{LLAMA_CPP_INSTALL_DIR}/build/bin/llama-server"

CUDA_HOME = "/usr/local/cuda-12.8"
# Compile-only CUDA subset for a headless llama.cpp (GGML_CUDA=ON) build: nvcc + the
# cudart/driver stubs + cuBLAS/cuRAND dev headers. The full `cuda-toolkit-12-8`
# metapackage also pulls in a GUI profiler and its desktop toolchain (~2 GB) that a
# headless build never uses, needlessly bloating the image and its build.
CUDA_TOOLKIT_PACKAGE = "cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-driver-dev-12-8 libcublas-dev-12-8 libcurand-dev-12-8"
# CUDA stubs let the linker resolve libcuda.so on GPU-less build machines; the real
# driver library is injected by the container runtime on the serving node.
CUDA_STUB_LIB = f"{CUDA_HOME}/lib64/stubs"

# Default target arch: 89 = Ada Lovelace (L4/L40S). Image builds run on CPU-only
# nodes, so CMAKE_CUDA_ARCHITECTURES=native is out. Pass a ";"-separated list to
# ``build_llama_cpp_image(cuda_arch=...)`` to produce a fat binary, e.g. "80;86;89;90"
# to also cover A100 (80), A10 (86), and H100 (90).
DEFAULT_CUDA_ARCH = "89"

# Node is required to build llama-server's embedded Web UI from source. The UI is a
# Vite 7 / SvelteKit app (needs Node >=22.12); without Node, cmake's llama-ui-assets
# target skips the npm build and falls back to a prebuilt UI download that is
# version-mismatched against embed.cpp and fails the build.
NODE_VERSION = "22.12.0"
NODE_HOME = "/opt/node"
