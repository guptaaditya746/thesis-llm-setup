# Debugging

| Symptom | Evidence | Recovery |
|---|---|---|
| LiteLLM healthy, vLLM unavailable | `llm-setup status`, then `llm-setup logs --service heavy` (or `lite`/`embed`) | Check allocation, model ID, and backend log; restart the owned session after correcting the profile. |
| High waiting queue | Status `waiting`, queue p95, and LiteLLM log | Reduce caller concurrency and follow `INCIDENTS.md`; do not raise limits without a benchmark. |
| High KV cache | `kvCache` near 0.92 | Shorten context or lower concurrency, then validate the profile. |
| Model OOM | backend log and `nvidia-smi` | Stop the owned session; lower context or memory utilization and load one role at a time for diagnosis. |
| Unsupported vLLM flag | `profile validate` or start error from `vllm serve --help` | Use options present in the installed version and update profile validation deliberately. |
| Embedding backend unavailable | embed log, `EMBED_MODEL_ID`, `/health` | Confirm exact model ID, access, and embedding task support. |
| Stale PID state | `verify` reports stale/not owned | `stop` only signals matching session-marked processes. Remove stale runtime state manually only after checking the exact session. |
| Slurm allocation ending | Slurm time remaining and service logs | Run `llm-setup stop`, then preserve needed non-sensitive diagnostics before allocation exit. |
