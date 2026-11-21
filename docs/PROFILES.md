# Profiles

A profile says which models run on which GPUs and ports. Every service listens on 127.0.0.1.

```yaml
extends: orch.yaml              # optional: start from another profile (path relative to this file)
name: orch-gliner
gateway_port: 4000              # optional, default 4000
status_port: 8010               # optional, default 8010
models:
  orch:                         # the role; it names the services: orch, orch-2, orch-3, ...
    alias: orch-model           # the name clients send to the gateway
    model: Qwen/Qwen3.8-27B-FP8
    revision: 017b9c7a...       # optional; pin it, results are cached by model identity
    gpus: all                   # or a list such as [0] or [1, 2]; one vLLM process per GPU
    port: 8001                  # the first replica's port; replica i listens on port + i
    max_num_active_seqs: 8      # vLLM --max-num-seqs
    max_num_queued_reqs: 8      # the gateway admits active + queued requests per replica
    max_model_len: 65536
    gpu_memory_utilization: 0.92
    kv_cache_dtype: auto        # auto, fp8, fp8_e4m3, fp8_e5m2 (chat models only)
    tool_call_parser: hermes    # optional, chat models only; enables tool_choice "auto"
    task: generate              # or embed (pooling runner; no kv_cache_dtype, no tool_call_parser)
    extra_args: [--enable-prefix-caching]
gliner:                         # optional; present means on (enabled: false turns it off)
  gpu: 0                        # must be a GPU that serves a model: GLiNER shares it
  port: ${GLINER_PORT:-8090}
  model: urchade/gliner_medium-v2.1
  revision: 40ec4193...         # defaults to the pinned revision of the default model
  max_chars: 1500
```

- `${NAME}` and `${NAME:-default}` are replaced from the environment before the YAML is read.
- `extends` merges mappings key by key; lists and values in the child replace the base's.
- `gpus: all` takes every GPU of the allocation that no other model lists. The count comes from
  `CUDA_VISIBLE_DEVICES` (set by Slurm), else `nvidia-smi`, else `--gpus N`. At most one model may use it.
- GPU numbers are indexes into the allocation: with `CUDA_VISIBLE_DEVICES=4,5,6`, GPU 0 is device 4.
- Validation refuses unknown fields, two processes on one GPU, a port used twice, a profile that
  needs more GPUs than the allocation has, and vLLM flags the installed vLLM does not support.

`uv run llm-setup profile show --profile <file> --gpus N` prints what would run where. A session
stores the resolved profile (explicit GPU lists) in `runtime/<session>/profile.yaml`.

## The profiles

| Profile | 1 GPU | 3 GPUs |
| --- | --- | --- |
| `orch.yaml` | orch :8001 on GPU 0 | orch, orch-2, orch-3 on :8001-8003, GPUs 0-2 |
| `orch-gliner.yaml` | the same + GLiNER :8090 on GPU 0 | the same + GLiNER :8090 on GPU 0 |
| `heavy-lite-embed.yaml` | refused | heavy :8002, lite :8003, embed :8001 |

The orch replicas run the same pinned weights at temperature 0; batching can still make two
replicas answer slightly differently. Results are cached by the pinned identity, not the replica.

## Capacity

Measure before changing concurrency, context or memory fractions (OPERATIONS.md, "Load test").
The KV cache limits how many tokens are in flight at once, more than the request count; vLLM prints
its size at startup (`llm-setup logs heavy | grep -iE "KV cache size|maximum concurrency"`).
Measured on 25 Sep 2026 with the model dtype: heavy 69,296 tokens (2.1 full 32k requests),
lite 127,728 tokens (3.9). Watch `preemptions` in `/v1/status`: a rising count under load means
too many long requests run at once.

**FP8 KV cache.** It roughly doubles capacity, but on A100 (compute capability 8.0) vLLM serves it
with the FlashInfer attention backend, whose kernels compile with `nvcc` at startup unless
`flashinfer-jit-cache` is installed. The cluster image has no CUDA toolkit, so `start` refuses fp8 on
pre-Hopper GPUs when neither is present. FP8 KV also uses a KV scale of 1.0 without calibration:
compare on the gold set before keeping it.

**Tool calling.** Clients that send `tools` with `tool_choice: "auto"` need `tool_call_parser` on
that model (Qwen3: `hermes`; Granite 4.1: `granite4`), or vLLM answers HTTP 400. `llm-setup smoke`
sends one tool-calling request per such model.

**Gateway retries.** The generated LiteLLM config retries only transient server errors (5xx), once.
Timeouts and rate limits are not retried: they mean the backend is saturated, and the caller owns
its deadline. There are no fallbacks between models. `LITELLM_ROUTING_STRATEGY` overrides
`least-busy` for measurements.
