"""Gateway connectivity + model discovery probe.

Hits the demo model gateway to list the models each demo key can see and which
model IDs actually answer, so you can confirm the System 2 configuration in
``_config.py`` before running the benchmark.

    flyte run examples/typesafe_ai/probe_gw.py probe_gw
"""

import os

from _runtime import env

import flyte

BASE = os.environ.get("LLM_GATEWAY_BASE_URL", "https://llm-gateway.apps.demo.hosted.unionai.cloud").rstrip("/")


def _models(key, tries=3):
    import httpx

    for _ in range(tries):
        try:
            r = httpx.get(BASE + "/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=300)
            if r.status_code == 200:
                return [m.get("id") for m in r.json().get("data", [])]
            return [f"HTTP{r.status_code}"]
        except Exception as e:
            last = str(e)
    return [f"ERR {last}"]


def _chat(key, model):
    import httpx

    try:
        r = httpx.post(
            BASE + "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Reply with the word: pong"}],
            },
            timeout=60,
        )
        if r.status_code == 200:
            j = r.json()
            return f"200 text={j['choices'][0]['message']['content']!r} usage={j.get('usage', {})}"
        try:
            msg = r.json().get("error", {}).get("message", "")[:120]
        except Exception:
            msg = r.text[:120]
        return f"{r.status_code} {msg}"
    except Exception as e:
        return f"ERR {e}"


@env.task
async def probe_gw(
    qwen_models: list[str] | None = None,
    claude_models: list[str] | None = None,
) -> str:
    """List every model a key can see and test chat on the requested models."""
    q = os.environ.get("DEMO_QWEN_38_27B_API_KEY", "")
    a = os.environ.get("DEMO_GATEWAY_ANTHROPIC_API_KEY", "")
    out = [f"BASE={BASE}", f"qwen key present: {bool(q)}", f"anthropic key present: {bool(a)}"]

    qm = _models(q)
    out.append("QWEN_ALL=" + str(qm))
    am = _models(a)
    out.append("ANTH_ALL=" + str(am))

    qwen_models = qwen_models or (qm if qm and not qm[0].startswith(("ERR", "HTTP")) else ["Qwen/Qwen3.8-27B"])
    for m in qwen_models:
        out.append(f"QWEN_TEST {m} => {_chat(q, m)}")

    claude_models = claude_models or [m for m in am if "claude" in m.lower()]
    for m in claude_models[:12]:
        out.append(f"CLAUDE_TEST {m} => {_chat(a, m)}")

    return "\n".join(out)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(probe_gw)
    print(r.name, r.url)
    r.wait()
