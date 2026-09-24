# Incidents

## Overload

1. Capture `/v1/status`, backend logs, and the last hour of SQLite samples.
2. Compare waiting count and queue p95 against end-to-end latency p95. Queue time rising while inference time stays stable points to admission pressure; both rising suggests model execution or context pressure.
3. Reduce caller concurrency first. Change capacity only after representative benchmark evidence.

## Recovery

1. Confirm the Slurm allocation and GPU process ownership with `nvidia-smi` and `llm-setup verify`.
2. Read the role log and correct the model ID, memory, or unsupported flag at the profile level.
3. Stop only the recorded session, start it again, then verify and smoke test.

Collect profile copy, versions, non-sensitive logs, status JSON, and queue/inference latency samples before changing settings. Never collect tokens or credentials.
