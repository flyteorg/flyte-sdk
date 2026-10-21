import argparse

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", type=str, required=True, help="The app endpoint URL (without /v1).")
parser.add_argument("--model_id", type=str, default="qwen2.5-0.5b-instruct")
parser.add_argument("--api_key", type=str, default="<your-api-key>")
args = parser.parse_args()

client = OpenAI(base_url=f"{args.endpoint}/v1", api_key=args.api_key)

response = client.chat.completions.create(
    model=args.model_id,
    messages=[{"role": "user", "content": "Write a one-line hello in Python."}],
    stream=True,
)
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
