# Model profiles

Profiles define the three role aliases, exact model ID, GPU index, loopback port, active sequence limit, queued request limit, context window, and GPU memory fraction. GPU placement is one role per GPU. The A100 profile uses GPU 0 for Heavy at port 8002 (4 active, 8 queued), GPU 1 for Lite at 8003 (8 active, 16 queued), and GPU 2 for Embed at 8001 (32 active, 64 queued). LiteLLM enforces each role's finite in-flight cap as active plus queued and rejects excess requests with a rate-limit response; no request waits in an unbounded proxy queue.

Heavy defaults to `Qwen/Qwen3-30B-A3B-Instruct-FP8`; Lite defaults to `ibm-granite/granite-4.1-8b-instruct`. `EMBED_MODEL_ID` must provide the exact selected Qwen embedding ID because this repository does not identify one. Context defaults are 32768, 32768, and 8192 tokens; memory fractions are 0.92, 0.90, and 0.85. Treat these as conservative starting points, not measured guarantees.

Validation rejects unknown profile fields, zero or unbounded queue settings, and checks required vLLM flags against the installed CLI before launch. Active and queued limits are configurable positive integers. Benchmark representative requests before increasing context, GPU memory use, or concurrency. Queue admission must remain finite.
