# Debugging

Start with `uv run llm-setup doctor` on the compute node
(`srun --jobid <jobid> --overlap --pty bash -l`). It checks the allocation, key, profile, GPUs
(leftover processes), weights in the cache, ports, the session, every health endpoint, the gateway
key and aliases and the status API, and names the first problem. `llm-setup events` shows what the
supervisor saw; `llm-setup logs <service>` works for the last session too, also after a failed start.

| Symptom | Evidence | Recovery |
|---|---|---|
| `session … is still running` at start | a live process of an earlier session | `llm-setup stop`, or `scancel` its job. A session with no live process is cleared automatically. |
| `ports already in use: gliner :8090` (or another port) | `ss -ltnp \| grep :<port>`; `ps -eo user,pid,etime,cmd \| grep <port>` shows another user's server | Never stop it. Move ours: `GLINER_PORT=8091 slurm/submit.sh`, and point `GLINER_URL` there. For the gateway or status port, set `gateway_port`/`status_port` in a profile that `extends` yours. |
| `gpus: all needs the allocation's GPUs` | the command ran outside the GPU job | Run it inside the job, or pass `--gpus N`. |
| `the allocation has 1 GPU(s), but the profile places …` | the profile lists more GPUs than requested | `submit.sh --gpus 3`, or a profile with `gpus: all`. |
| Step in an existing job sees `GPUs: none visible` | `srun --overlap` without `--gres` | Use `slurm/submit.sh --job <jobid> --gpus N`, which asks for the GPUs. |
| `bash: slurm/job.sh: No such file or directory`, exit 127 | a hand-typed `srun --container-image=...` | The container starts in the image's directory; use `slurm/submit.sh`, which uses absolute paths. |
| Everything gone after closing the terminal | the stack ran in `srun --pty` | Use `slurm/submit.sh` (a batch job, or a detached step with `--job`). |
| Gateway unavailable during a run | `events` shows `gateway exited`, then `restarted` | Restarted within seconds; `gave up restarting` means it keeps crashing: `llm-setup logs gateway`. |
| Gateway fine, a model unavailable | `llm-setup status`, then `llm-setup logs orch-2` | Check the allocation, model ID and log; fix the profile, then restart the job. |
| `gateway key: rejected` | the shell's key differs from the gateway's | `set -a; source .env; set +a`; clients must send the same key. |
| High waiting queue or `kvCache` near 0.92 | `/v1/status`, queue p95 | Reduce caller concurrency first; change limits only after a load test (OPERATIONS.md). |
| Model OOM | the model's log, `nvidia-smi` | Lower `max_model_len` or `gpu_memory_utilization`; with GLiNER on GPU 0, see OPERATIONS.md, "GLiNER". |
| Model dies at first request with `Could not find nvcc` (FlashInfer sampler) | log mentions `/usr/local/cuda` | The default `VLLM_USE_FLASHINFER_SAMPLER=0` avoids it; set 1 only with a CUDA toolkit loaded. |
| Start fails right after loading, `Could not find nvcc` with the FlashInfer attention backend | `kv_cache_dtype: fp8` on A100 | Set `kv_cache_dtype: auto`, or install `flashinfer-jit-cache` (PROFILES.md, "Capacity"). |
| HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice` | the model has no tool-call parser | Set `tool_call_parser` in the profile, restart; `llm-setup smoke` tests it. |
| `timed out … vllm serve --help=all` or an unsupported flag | start error before any model launched | Raise `LLM_SETUP_VLLM_HELP_TIMEOUT_SECONDS` (default 600, max 3600) if the filesystem is slow; use flags the installed vLLM has. |
| Job stuck in `CG` long after `scancel`; `srun --jobid <id> --overlap …` hangs or says `aborted before step completely launched` | `sinfo -n <node>` shows `completing` or `completing*` (`*`: slurmd not answering) | A node problem only an admin can clear (Discord admin channel). Do not retry `srun` into the job: each try adds a step that cannot be cancelled. Start elsewhere: `slurm/submit.sh -- --exclude=<node>`. |
| Start stuck at `installing the compiler Triton needs` | the package mirror is slow | Build the image once with `slurm/build-image.sh`; `submit.sh` then skips `apt-get`. |
| Allocation about to end | `doctor` warns under an hour left | Save the non-sensitive logs you need; the job stops cleanly 3 minutes before the limit. |

## Incidents

When overloaded: capture `/v1/status`, the model logs and the last hour of
`runtime/<session>/status.sqlite3` (a copy up to a minute old; the live one is node-local). Queue time rising while inference time stays flat means
admission pressure; both rising means model or context pressure. Reduce caller concurrency first.
Collect profile, versions, logs and latency samples before changing settings; never tokens.

- **2026-11-21, gammaweb08, job 9844** (interactive, gpu:ampere:3). The first in-job start failed
  with exit 127 (relative script path inside the container); the second stopped with
  `ports already in use: gliner :8090`: another user's vLLM had listened on 0.0.0.0:8090 for a day.
  No model had loaded. Fixes: in-job start with absolute paths and `--gres` on the step (now
  `submit.sh --job`), the `GLINER_PORT` override, and a port hint in the start error.
- **2026-11-22, gammaweb08, jobs 10167 and 10178.** 10167 (CPU `srun --pty bash`) got signal 9 at
  20:49 and stayed in CG; the node showed `completing*` on and off. Every later
  `srun --jobid 10178 --overlap` hung (`aborted before step completely launched`), so steps
  10178.18–31 piled up and could not be cancelled; the stack's log stopped at 20:33 and after
  `scancel 10178` it was stuck in CG too. Another user's job on the node kept running, so our own
  processes may have blocked cleanup (unconfirmed; needs an admin to see `D`-state processes). Earlier
  that evening a SIGKILLed step left our ports held (`ports already in use` on the next start), and a
  start hung 14 minutes in `apt-get`. Fixes: `terminate()` grace 30 → 20 s (below Slurm `KillWait`
  30 s), `slurm/build-image.sh` (no `apt-get` at start), the two DEBUGGING rows above.
  Open: (1) expose health without a new Slurm step; (2) find what 10167 ran and why it got signal 9
  (64 GB memory limit?).
- 2026-11-22, gammaweb08: jobs 10178 (llm-stack) and 10167 (a bare `bash`) stuck in CG; the node went
  `MIXED+COMPLETING+NOT_RESPONDING`. All six service logs in `runtime/<session>/` stopped in the same
  second (20:37:17) and no "stopping" event was written: the Ceph mount hung, not our shutdown.
  Fix: the session folder is node-local, copied to `runtime/` every minute by `llm_setup.mirror` (the
  only session process writing to Ceph); the job log goes through a SIGTERM-ignoring `cat`; the
  package is installed non-editable. A hard Ceph hang still holds the mirror/`cat` in CG: admins.
