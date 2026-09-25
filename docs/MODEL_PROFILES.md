# Model profiles

Profiles define the three role aliases, exact model ID, GPU index, loopback port, active sequence limit, queued request limit, context window, GPU memory fraction and, for chat roles, the KV-cache dtype. GPU placement is one role per GPU. The A100 profile uses GPU 0 for Heavy at port 8002 (4 active, 8 queued), GPU 1 for Lite at 8003 (8 active, 16 queued), and GPU 2 for Embed at 8001 (32 active, 64 queued). LiteLLM caps each role's in-flight requests at active plus queued (`max_parallel_requests`). Requests beyond the cap wait inside LiteLLM rather than being rejected, and that wait counts towards the 120 s gateway timeout.

| Role | Alias | Model in `a100-3x40` | Context | Memory fraction | KV cache |
| --- | --- | --- | --- | --- | --- |
| Heavy | `heavy-model` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (MoE, non-thinking) | 32768 | 0.92 | `fp8` |
| Lite | `lite-model` | `ibm-granite/granite-4.1-8b` | 32768 | 0.90 | model dtype |
| Embed | `qwen-embed` | `Qwen/Qwen3-Embedding-0.6B` (override with `EMBED_MODEL_ID`) | 8192 | 0.85 | n/a |

## KV-cache capacity

The KV cache limits how many tokens can be in flight at once, which matters more than the request count. vLLM prints it at startup (`uv run llm-setup logs --service heavy | grep -iE "KV cache size|maximum concurrency"`). Measured on 25 Sep 2026 with the model dtype:

| Role | KV cache | Full 32k requests at once |
| --- | --- | --- |
| Heavy | 69,296 tokens | 2.11 |
| Lite | 127,728 tokens | 3.90 |

Heavy is the bottleneck, but it keeps the model dtype (`kv_cache_dtype: auto`). An FP8 KV cache would roughly double its capacity, yet on A100 (compute capability 8.0) vLLM serves FP8 KV with the FlashInfer attention backend, whose kernels are JIT-compiled with `nvcc` at startup unless `flashinfer-jit-cache` (matching the installed flashinfer version) is installed. The Slurm image has no CUDA toolkit, so Heavy would die before `/health` answers and the whole start would fail. `start` and `verify` now refuse `fp8` on pre-Hopper GPUs when neither `nvcc` nor `flashinfer-jit-cache` is available. FP8 KV also uses a KV scale of 1.0 without calibration, which can cost accuracy: compare on the gold set before keeping it.

Allowed values are `auto` (default, the model dtype), `fp8`, `fp8_e4m3` and `fp8_e5m2`; the embedding role does not accept the field. Watch `preemptions` in `/v1/status`: a rising count under load means too many long requests run at once.

## Tool calling

The harness agent loop sends OpenAI-style `tools` with `tool_choice: "auto"` to `lite-model`. vLLM rejects such requests (HTTP 400, "auto tool choice requires --enable-auto-tool-choice and --tool-call-parser") unless the backend was started with a parser, so chat roles take `tool_call_parser`:

| Role | Parser | Why |
| --- | --- | --- |
| Lite (Granite 4.1) | `granite4` | Runs the agent tool loop |
| Heavy (Qwen3-2507) | `hermes` | Qwen's chat template uses Hermes-style tool calls; not needed today, available if a Heavy agent is configured |

The launcher then adds `--enable-auto-tool-choice --tool-call-parser <name>`, and `verify` checks the parser exists in the installed vLLM. `smoke` sends one tool-calling request per such role.

## Validation

Validation rejects unknown profile fields, zero or unbounded queue settings, and unknown KV-cache dtypes, and checks the required vLLM flags (including `--kv-cache-dtype` when used) against the installed CLI before launch. Active and queued limits are configurable positive integers. Benchmark representative requests before increasing context, GPU memory use, or concurrency (see `OPERATIONS.md`, "Load test"). Queue admission must remain finite.

## Gateway retries

The generated LiteLLM config retries only transient server errors (5xx, 503), once. Timeouts are not retried by the gateway: a timed-out generation usually means the backend is saturated, a retry adds load, and the harness's own per-call timeouts (45-120 s) are at or below the gateway's 120 s, so a gateway retry would arrive after the caller has given up anyway.
