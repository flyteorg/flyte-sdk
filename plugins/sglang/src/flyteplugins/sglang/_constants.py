# DFLASH landed in SGLang on 2026-04-07 (PR #22077) and the Spec V2 engine it rides on became
# the default in the 2026-06 release line. 0.5.2 predates both, as well as the EAGLE3 support
# `speculative_config` on SGLangAppEnvironment depends on.
SGLANG_MIN_VERSION = (0, 5, 16)
SGLANG_MIN_VERSION_STR = ".".join(map(str, SGLANG_MIN_VERSION))

# The cache-aware router ships as its own package on its own release cadence, so this pin
# tracks SGLANG_MIN_VERSION by hand. A router/server mismatch shows up as workers failing to
# register rather than as an install error, so re-check it whenever SGLang is bumped.
SGLANG_ROUTER_VERSION = "0.3.2"

# CUDA 13, matching the wheels SGLang itself pins: 0.5.16 requires flashinfer_python[cu13] and
# nvidia-cutlass-dsl[cu13]. A toolkit from a different CUDA major leaves flashinfer half-loaded
# until something explodes deep inside an import.
CUDA_TOOLKIT_PACKAGE = "cuda-toolkit-13-0"
CUDA_HOME = "/usr/local/cuda-13.0"
