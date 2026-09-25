# Debugging

| Symptom | Evidence | Recovery |
|---|---|---|
| LiteLLM healthy, vLLM unavailable | `llm-setup status`, then `llm-setup logs --service heavy` (or `lite`/`embed`) | Check allocation, model ID, and backend log; restart the owned session after correcting the profile. |
| High waiting queue | Status `waiting`, queue p95, and LiteLLM log | Reduce caller concurrency and follow `INCIDENTS.md`; do not raise limits without a benchmark. |
| High KV cache | `kvCache` near 0.92 | Shorten context or lower concurrency, then validate the profile. |
| Model OOM | backend log and `nvidia-smi` | Stop the owned session; lower context or memory utilization and load one role at a time for diagnosis. |
| Heavy fails at first request with `Could not find nvcc` | Heavy log shows FlashInfer sampler JIT trying to find `/usr/local/cuda` | The launcher defaults to vLLM's native PyTorch sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`), which needs no CUDA compiler. If explicitly setting it to `1`, first load a compatible CUDA toolkit module that provides `nvcc`. |
| Unsupported vLLM flag or slow CLI startup | start error from `vllm serve --help=all` | Use options present in the installed version. The full-help check allows 600 seconds by default; adjust `LLM_SETUP_VLLM_HELP_TIMEOUT_SECONDS` (maximum 3600). Keep the Python environment on node-local scratch if the shared filesystem stalls during imports. |
| Embedding backend unavailable | embed log, `EMBED_MODEL_ID`, `/health` | Confirm exact model ID, access, and embedding task support. |
| Stale PID state | `verify` reports stale/not owned | `stop` only signals matching session-marked processes. Remove stale runtime state manually only after checking the exact session. |
| Slurm allocation ending | Slurm time remaining and service logs | Run `llm-setup stop`, then preserve needed non-sensitive diagnostics before allocation exit. |
| Agent requests fail with HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser` | The backend was started without a tool-call parser | Set `tool_call_parser` for the role in the profile (Lite: `granite4`), then `llm-setup stop` and `start`. `llm-setup smoke` tests tool calling. |
| `start` fails: Heavy exits with `Could not find nvcc` right after loading, with the FlashInfer attention backend in the log | `kv_cache_dtype: fp8` on A100 needs FlashInfer attention kernels, JIT-compiled with nvcc | Set `kv_cache_dtype: auto`, or install `flashinfer-jit-cache` for the installed flashinfer version. After a failed start, read `runtime/<session>/heavy.log` directly (`llm-setup logs` needs a current session). |
