from openai import OpenAI

client = OpenAI(
    base_url="https://ancient-bush-70e72.apps.demo.hosted.unionai.cloud/v1",
    api_key="<your-api-key>",
)

response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
