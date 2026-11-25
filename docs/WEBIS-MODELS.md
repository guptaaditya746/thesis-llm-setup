# Webis LLM API model inventory (2026-11-24)

## Observed service inventory

The authenticated endpoint `https://chat.web.webis.de/openai/models` returned 29 IDs on 2026-11-24. The public Ollama proxy `https://llm.srv.webis.de/ollama/api/tags` returned corresponding tags. This is an observation of the service at query time, not a stable catalog. All OpenAI-list entries reported `TextGeneration`; no embedding model appeared in the list, so the docs' `nomic-embed-text` example is not evidence it is currently deployed.

| Group | IDs |
| --- | --- |
| Qwen | `qwen3.5-9b`, `qwen3.8-27b`, `qwen3-30b-a3b`, `qwen3-30b-a3b-cpu`, `qwen3-8b`, `qwen3-1.7b`, `qwen3-1.7b-cpu`, `qwen3-0.6b-cpu` |
| DeepSeek R1 | `deepseek-r1-14b`, `deepseek-r1-8b`, `deepseek-r1-7b`, `deepseek-r1-1.5b-cpu` |
| GPT-OSS | `gpt-oss-20b`, `gpt-oss-20b-cpu` |
| Llama | `llama3.1-8b`, `llama3.2-3b`, `llama3-8b` |
| Gemma | `gemma4-e4b`, `gemma3-4b`, `gemma3-1b-cpu`, `gemma3-270m-cpu`, `gemma2-2b` |
| Other | `phi3-3.8b`, `mistral-7b`, `llava-13b`, `vicuna-7b`, `zephyr-7b`, `smollm2-1.7b-cpu`, `tinyllama-1.1b-cpu` |

The unauthenticated Ollama `/api/show` endpoint returned `Q4_0`, empty `model_info`, and the same parameters for all 29 aliases: temperature 0.7, top-p 0.9, top-k 40, `num_predict 128`. This establishes the proxy's reported defaults, not necessarily an API-enforced output limit. `/api/version` returned `0.1.42`. Tag metadata reported zero sizes and unknown parameter sizes, so it does not identify actual served artifacts or runtime resource allocation.

### API smoke check

On 2026-11-24, a single OpenAI-compatible chat completion to `https://llm.srv.webis.de/openai/v1/chat/completions` using `smollm2-1.7b-cpu` returned HTTP 200, `finish_reason=stop`, and `API_OK` in 0.79 seconds. This confirms the API was generating at that moment; one short request is not a latency estimate or a service-level guarantee.

### Output-budget check

On the authenticated OpenAI endpoint, `smollm2-1.7b-cpu` accepted `max_tokens=256` and returned 256 completion tokens with `finish_reason=length` (20.26 seconds) for a numbered-sequence prompt. Thus, the proxy's reported default `num_predict 128` is overridable through this API path. This is a parameter check, not a performance sample: the long numbered output is unlike the thesis task. A separate repetitive-word probe at requested budgets 128 and 192 received HTTP 500 (`prediction aborted, token repeat limit reached`), so those failures do not indicate a 128-token output cap.

For context, one-off demo-endpoint checks generated `API_OK` from `smollm2-1.7b-cpu` with HTTP 200 in 0.86 seconds, and from `qwen3-0.6b-cpu` in 3.66 seconds. A `qwen3.5-9b` request returned `2` in 4.70 seconds with a 256-token budget. These are smoke checks with different prompts and cache states. Webis documents the unauthenticated endpoint as demo-only and says not to use it for experiments; none of these demo-endpoint numbers are included in the authenticated pilot below.

### Authenticated latency pilot (2026-11-24)

This is a controlled service timing pilot, not a quality evaluation. It used the authenticated endpoint, streamed responses with usage included, disabled retries, ran sequentially, and randomized/interleaved model order. Each model received one smoke request, three warmups and 30 timed requests. The user prompt was the same 40-word explanation task with a three-digit run ID; `temperature=0`, `max_tokens=512`. No HTTP failures or timeouts occurred during this sample. The prompt is generic and the models emitted different numbers of tokens, so total latency is not equal-work throughput and does not pick the best thesis model.

| Model | Timed requests | TTFT p50 / p95 (s) | Total p50 / p95 (s) | Median prompt / completion tokens | Median completion tokens/s | `finish_reason=length` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `smollm2-1.7b-cpu` | 30/30 | 0.313 / 0.551 | 3.931 / 4.217 | 53 / 51 | 13.0 | 0 |
| `qwen3-0.6b-cpu` | 30/30 | 0.621 / 0.791 | 10.834 / 16.126 | 34 / 200.5 | 18.2 | 0 |
| `qwen3.5-9b` | 30/30 | 0.465 / 0.722 | 13.399 / 13.705 | 34 / 512 | 38.2 | 30 |

The `qwen3.5-9b` runs all used the entire 512-token budget and ended with `finish_reason=length`; the requested answer was not completed. Its timing therefore measures capped generation, not a successful answer. `qwen3-0.6b-cpu` completed but used about four times as many output tokens as SmolLM2. The observed total-latency differences are descriptive: paired bootstrap intervals for median differences were +6.915 s for Qwen3-0.6B minus SmolLM2 (95% CI 6.321–7.612), +9.481 s for Qwen3.5 minus SmolLM2 (9.382–9.599), and −2.624 s for Qwen3-0.6B minus Qwen3.5 (−2.941 to −1.797). The last comparison is especially unsuitable as a winner claim because Qwen3.5 never completed the answer.

Raw request metrics (no API key or response text) are in [`benchmark-results/webis-auth-latency-2026-11-24.jsonl`](benchmark-results/webis-auth-latency-2026-11-24.jsonl). The token-rate column counts all completion tokens, including reasoning; it is not a measure of useful answer throughput. This run can inform request timing for that prompt and endpoint on that date; repeat on representative thesis examples before making a model choice.

### Context-limit probe (inconclusive)

Context probes used the authenticated endpoint and attempted exact evidence-code retrieval from filler prompts, with the server-reported `prompt_tokens` as the token count. The context-size calibration and inference service became unreliable before five valid placements per size could be collected:

- For `qwen3-0.6b-cpu`, a prompt constructed for about 4K input tokens reported 2,050 prompt tokens on two attempts; a separately constructed ~2K prompt also reported 2,050. The target code was not returned exactly. One ~1K request and a larger request timed out. This pattern is **consistent with truncation or a roughly 2K effective input cap**, but the timed-out calls and failed exact retrieval mean it is not a verified context limit.
- For `qwen3.5-9b`, two ~2K-input probes used the full 1,024-token output budget for reasoning and did not return the requested code. The provider documents `chat_template_kwargs: {"enable_thinking": false}`, but sending that field and appending `/no_think` did not suppress reasoning in this Webis endpoint. This prevented a useful retrieval-accuracy context measurement.
- `smollm2-1.7b-cpu` context ladder requests at estimated 512 and 1K input sizes timed out at 35 seconds. A later 8-token smoke request also timed out at 25 seconds, although the authenticated model-list endpoint still returned HTTP 200 with 29 models. Inference was therefore unhealthy or delayed during this part of the session.

The incomplete probe records are preserved in [`benchmark-results/webis-auth-context-2026-11-24.jsonl`](benchmark-results/webis-auth-context-2026-11-24.jsonl), [`benchmark-results/webis-auth-context-small-models-2026-11-24.jsonl`](benchmark-results/webis-auth-context-small-models-2026-11-24.jsonl), and [`benchmark-results/webis-auth-context-calibration-2026-11-24.jsonl`](benchmark-results/webis-auth-context-calibration-2026-11-24.jsonl). Do not report an effective context limit from these attempts. Repeat only after short authenticated generation requests are healthy again; count input tokens from API usage and require the target code to be returned exactly at every tested size/position.

#### Qwen3.5-9B recheck (2026-11-25)

The upstream [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) specifies 262,144 native context tokens and says it can be extended to 1,010,000. That remains an upstream capability, not a Webis deployment setting.

The authenticated Webis catalog still returned HTTP 200 with 29 models and included `qwen3.5-9b`. The Ollama `/api/show` response returned HTTP 200 but had empty `model_info` and no `num_ctx`/context field; it reported `num_predict 128`. Fresh authenticated context probes at estimated 2K and 4K input sizes both timed out after 45 seconds without returning token usage. So the live Webis context limit remains unknown; the earlier repeated 2,050-token count on larger constructed prompts is suggestive of truncation, but not enough to claim a 2K cap.

The recheck is recorded in [`benchmark-results/webis-auth-qwen3.5-recheck-2026-11-25.jsonl`](benchmark-results/webis-auth-qwen3.5-recheck-2026-11-25.jsonl). The catalog and metadata endpoints respond, while chat generation has timed out during these context probes.

The initial, larger 30-request timing run used the unauthenticated VPN/demo endpoint before its documentation warning was noticed. Its raw trace [`benchmark-results/webis-demo-api-latency-2026-11-24.jsonl`](benchmark-results/webis-demo-api-latency-2026-11-24.jsonl) is retained as a demo-path trace only and is excluded from the authenticated results above; do not use it as experimental evidence.

## Context information and limits

Webis' OpenAI model-list response and Ollama `show` response do not expose the configured context cap. The following are **upstream family specifications**, not verified Webis deployment limits. Even where the original model supports a large context, the Webis alias may be configured lower; confirm by measuring the endpoint before relying on long prompts.

| Webis IDs | Upstream context reference | Source |
| --- | --- | --- |
| `qwen3.5-9b` | 262,144 native; model card says extensible to about 1M | [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B) |
| `qwen3.8-27b` | Model card describes a hosted 1M-context offering; do not assume that hosted setting applies to this Webis alias | [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B) |
| `qwen3-0.6b-cpu`, `qwen3-1.7b`, `qwen3-1.7b-cpu`, `qwen3-8b`, `qwen3-30b-a3b`, `qwen3-30b-a3b-cpu` | 32,768 native; up to 131,072 with YaRN configuration | [Qwen3-8B card](https://huggingface.co/Qwen/Qwen3-8B), [Qwen3-30B-A3B card](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| `gemma3-270m-cpu`, `gemma3-1b-cpu` | 32K for Gemma 3 1B; 270M is not established by the cited card | [Gemma 3 card](https://huggingface.co/google/gemma-3-1b-it) |
| `gemma3-4b` | 128K | [Gemma 3 4B card](https://huggingface.co/google/gemma-3-4b-it) |
| `gemma4-e4b` | 128K | [Gemma 4 E4B card](https://huggingface.co/google/gemma-4-E4B) |
| `llama3.1-8b` | 128K | [Meta Llama 3.1 8B card](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| `llama3.2-3b` | 128K | [Meta Llama 3.2 3B card](https://huggingface.co/meta-llama/Llama-3.2-3B) |
| `llama3-8b` | 8K | [Meta Llama 3 8B card](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| `gpt-oss-20b`, `gpt-oss-20b-cpu` | 131,072 | [OpenAI GPT-OSS-20B docs](https://developers.openai.com/api/docs/models/gpt-oss-20b) |
| `phi3-3.8b` | 128K if this alias uses Phi-3 Mini 128K; the API does not expose variant | [Microsoft Phi-3 Mini 128K card](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) |
| `deepseek-r1-1.5b-cpu`, `deepseek-r1-7b`, `deepseek-r1-8b`, `deepseek-r1-14b` | The official R1-distill serving example uses `--max-model-len 32768`; exact Webis aliases/architectures are not exposed | [DeepSeek R1 Distill Qwen-14B card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) |
| `mistral-7b` | 32,768 for the v0.3 family; alias version is unspecified | [Mistral 7B v0.3 config](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json) |
| `gemma2-2b`, `llava-13b`, `vicuna-7b`, `zephyr-7b`, `smollm2-1.7b-cpu`, `tinyllama-1.1b-cpu` | Not established from the live service metadata; upstream variants/configs must be identified first | — |

## Implications for a model comparison

### Proposed thesis pilot

**Shortlist, not a presumed ranking:**

- `qwen3.5-9b`: long-context-capable candidate at a smaller nominal size.
- `qwen3.8-27b`: larger Qwen candidate.
- `gpt-oss-20b`: separate model family and reasoning candidate.
- `deepseek-r1-14b`: distilled reasoning candidate.
- `qwen3-8b`: smaller baseline with an upstream 32K native context.

These choices span model families and sizes. They do not establish which one is best for the thesis task. If Gemma's image input matters, add `gemma4-e4b` as a separate multimodal candidate; do not compare it on a text-only task and infer image quality.

**Freeze the task and examples first.** For the evidence extraction workflow, use one atomic task per call (for example, extract one claim from one supplied source passage into a fixed JSON schema). Keep prompt, schema, source text, and acceptance rules identical across candidates. Include ordinary, confusing, and missing-evidence cases. Use a small development split to fix the prompt and a separate held-out set for the final comparison; keep papers and their duplicates/counterfactuals together in one split. A reasonable pilot is 40 development examples and 100 held-out examples, if those counts fit the corpus. Store gold fields and exact supporting spans outside the model prompt.

**Quality:** Run the same held-out items through every model in randomized, interleaved order. Use temperature 0 where accepted and record all parameters. Score fields with deterministic checks: schema validity, normalized claim/value correctness, and source-span match. Report exact record accuracy, field-level precision/recall/F1, invalid-output rate, and unsupported-claim rate. Blindly inspect disagreements and a sample of matches. If generation and verification are both needed, score them as separate roles; don't let a model grade its own output or treat two same-family calls as independent evidence.

**Latency and reliability:** First do one smoke request per model. Then use 3 warmups and 30 sequential timed requests per model on a fixed prompt/output size. Interleave model order to reduce time-of-day bias. Stream responses to measure time to first token; also record total latency, prompt/output tokens, output tokens per second, finish reason, HTTP failures, timeouts, and any retries. Report median and p95 latency plus paired bootstrap intervals over the identical examples. Keep automatic retries disabled in the primary analysis; report a separately tagged retry only as reliability recovery. Avoid concurrent benchmarking until Webis publishes a load policy.

**Context:** Test a controlled evidence-location task with the target passage placed at varying positions in padding text. Increase actual tokenizer-counted prompt size in steps (for example, 4K, 8K, 16K, 32K, then 64K and 128K only for candidates that survive the smaller steps). Use 5 distinct items/placements per size. Reserve output budget and count input plus output against the context. Measure both request acceptance and whether the model returns the required evidence correctly. Stop at service rejection or a clear accuracy collapse; distinguish that empirical effective limit from upstream context specifications.

**API controls and limits:** Pin API base URL, model ID, prompt/schema hash, client/library version, request timestamp, and all request parameters. Ask for the same bounded output budget (e.g. 256 tokens) and set temperature 0 if supported; record `max_tokens`/`max_completion_tokens` behavior and any rejected parameters rather than silently changing settings. The proxy advertises `num_predict 128`; verify that the OpenAI endpoint accepts an override before interpreting truncated outputs as model failures. Keep the request rate sequential and modest because no service rate limit is documented. Don't include private thesis data in API calls unless its use on Webis is approved for that corpus.

**Decision rule:** Set quality and reliability minimums before evaluating the held-out set. Among models that pass them, compare the quality/latency/cost-of-use tradeoff (Webis has no per-token price in the supplied docs, but wall time still matters). Report a Pareto set if no model dominates; do not collapse the result to a single “best” model unless the thesis workflow has a predeclared priority rule. No available evidence establishes a winner before this pilot runs.

## Sources

- Webis service usage instructions were supplied in the conversation; they document authenticated `https://chat.web.webis.de/openai/`, model enumeration, and the VPN-only Ollama proxy. The live data above was read from those service endpoints on 2026-11-24.
- Model context links above are first-party model cards/configuration or provider docs. They describe upstream models, not Webis' deployment configuration.
