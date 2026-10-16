"""
Serve a Hugging Face model on the Mac's GPU from an app: `transformers` + `torch` on `mps`.

The app's own code loads the model onto `mps` and generates with it. A devbox container is a Linux VM with no Metal,
so a `gpu="AppleSilicon"` app runs natively on the Mac instead, in an environment built from the image's pip
packages. The devbox still gives the app its URL and logs, and forwards requests to it.

This runs workload code outside any sandbox, so it is opt-in, and the devbox binds every port to 127.0.0.1:

    flyte start devbox --apple-silicon
    flyte create config --devbox
    python examples/accelerators/apple_silicon_serve.py
    curl "http://hf-mps-flytesnacks-development.localhost:30081/generate?prompt=Write+a+haiku+about+a+laptop"

The model downloads into the Mac's Hugging Face cache (~/.cache/huggingface) on first start.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import flyte
from flyte.app import AppEnvironment

MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

app_env = AppEnvironment(
    name="hf-mps",
    image=flyte.Image.from_debian_base().with_pip_packages("torch", "transformers"),
    # Apple Silicon: the app runs natively on the Mac, so torch can use the Metal (mps) device.
    resources=flyte.Resources(cpu=1, memory="4Gi", gpu="AppleSilicon"),
    port=8080,
    requires_auth=False,
)


@app_env.server
def serve():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Fail loudly rather than quietly serving from the CPU.
    if not torch.backends.mps.is_available():
        raise RuntimeError("mps is not available: run this app with gpu='AppleSilicon' on an Apple Silicon devbox")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("mps")
    model.eval()
    lock = threading.Lock()  # one generation at a time on the GPU

    def generate(prompt: str, max_new_tokens: int) -> dict:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("mps")
        with lock, torch.inference_mode():
            start = time.perf_counter()
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return {
            "model": MODEL,
            "device": str(model.device),
            "dtype": str(model.dtype),
            "tokens_per_second": round(len(new_tokens) / elapsed, 1),
            "completion": tokenizer.decode(new_tokens, skip_special_tokens=True),
        }

    # The first generation on mps compiles Metal kernels; do it before taking traffic.
    generate("Hello", 8)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                status, payload = 200, {"status": "ok", "device": str(model.device)}
            elif url.path == "/generate":
                query = parse_qs(url.query)
                prompt = query.get("prompt", ["Say hello."])[0]
                max_new_tokens = int(query.get("max_new_tokens", ["64"])[0])
                status, payload = 200, generate(prompt, max_new_tokens)
            else:
                status, payload = 404, {"try": "/generate?prompt=..."}
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    # 127.0.0.1: on the host, only the devbox's port forward needs to reach the app.
    ThreadingHTTPServer(("127.0.0.1", app_env.get_port().port), Handler).serve_forever()


if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(app_env)
    print(app.url)
