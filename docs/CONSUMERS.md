# Consumers

Consumers may call only `http://host:4000/v1` for model requests and `http://host:8010/v1/status` for status. Never call raw vLLM ports. Use aliases `heavy-model`, `lite-model`, and `qwen-embed` in an OpenAI-compatible client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://host:4000/v1", api_key="local-dev-key")
reply = client.chat.completions.create(model="lite-model", messages=[{"role": "user", "content": "Extract fields"}])
vectors = client.embeddings.create(model="qwen-embed", input=["sample"])
```

Use `heavy-model` for quality schema inference, evidence review, and answer verification. Use `lite-model` for structured extraction, EDC mapping, and agent planning. The research harness should render `overall` and each alias's `state` from `/v1/status`, with unknown or unavailable shown as such; it should not infer health from a raw backend port.
