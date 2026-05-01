import os
from openai import OpenAI

client = OpenAI(
    base_url="https://gateway.salesforceresearch.ai/openai/process/v1/",
    api_key="dummy",
    default_headers={"X-Api-Key": os.getenv("X_API_KEY")},
)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello from openai gateway!"}],
    temperature=0.7,
    top_p=0.9,
)
print(resp.choices[0].message.content)