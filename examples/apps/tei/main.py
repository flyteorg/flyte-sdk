"""
Deploy the TEI app and hit it with a client.

Usage:
    flyte deploy examples/apps/tei/app.py tei_cpu_env   # or tei_gpu_env
    python examples/apps/tei/main.py                    # deploys + calls /embed

The client expects TEI's `/embed` endpoint: POST JSON {"inputs": [...]} and
get back a list of embedding vectors.
"""

import httpx

import flyte
from app import tei_cpu_env  # switch to tei_gpu_env for GPU deployment

if __name__ == "__main__":
    flyte.init_from_config()

    deployed = flyte.deploy(tei_cpu_env)
    app = deployed[0]
    print(app.table_repr())

    endpoint = tei_cpu_env.endpoint.rstrip("/")
    print(f"TEI endpoint: {endpoint}")

    # /health returns 200 once the model finishes loading.
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{endpoint}/embed",
            json={"inputs": ["What is Flyte?", "Hugging Face text embeddings inference"]},
        )
        resp.raise_for_status()
        vectors = resp.json()
        print(f"Got {len(vectors)} embedding(s), dim={len(vectors[0])}")
