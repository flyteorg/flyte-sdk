# vLLM 0.20.1+ is required for the DFlash speculator, and the whole 0.2x line takes every
# speculative knob as a single ``--speculative-config`` JSON blob. Older pins (0.11.x) predate
# both, so `speculative_config` on VLLMAppEnvironment would have nothing to talk to.
VLLM_MIN_VERSION = (0, 26, 0)
VLLM_MIN_VERSION_STR = ".".join(map(str, VLLM_MIN_VERSION))

# Must equal the ``flashinfer-python`` pin of VLLM_MIN_VERSION -- vLLM pins it exactly, so a
# different version here is silently overwritten by the vLLM layer and leaves the prebuilt
# kernels below mismatched against the installed flashinfer.
FLASHINFER_VERSION = "0.6.14"

# The index has to match the CUDA major that vLLM's own wheels pin (0.26.0 ->
# nvidia-cutlass-dsl[cu13], i.e. CUDA 13). This is a hand-maintained pair: bump
# VLLM_MIN_VERSION and both of the above have to be re-checked together.
FLASHINFER_JIT_CACHE_INDEX_URL = "https://flashinfer.ai/whl/cu130"
