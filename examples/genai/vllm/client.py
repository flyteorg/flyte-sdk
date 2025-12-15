import argparse

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", type=str, required=True)
parser.add_argument("--model_id", type=str, required=True)
args = parser.parse_args()
endpoint = args.endpoint

client = OpenAI(
    base_url=f"{endpoint}/v1",
    api_key="<your-api-key>",
)

response = client.chat.completions.create(
    model=args.model_id,
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
