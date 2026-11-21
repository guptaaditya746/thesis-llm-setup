"""The LiteLLM gateway configuration, generated per session from the profile.

Each model alias routes to its replicas (least busy first); a model with several replicas also
exposes ``<alias>-1``, ``<alias>-2``, ... to pin one. There are no fallbacks between models.
"""

from __future__ import annotations

import os
from typing import Any

from .profile import Profile

# A timeout or a rate limit means the backend is already saturated; retrying it from the gateway only
# adds load and hides the wait from the caller, who owns its own deadline. Only transient server
# errors are retried, once.
RETRY_POLICY = {
    "TimeoutErrorRetries": 0,
    "RateLimitErrorRetries": 0,
    "BadRequestErrorRetries": 0,
    "AuthenticationErrorRetries": 0,
    "ContentPolicyViolationErrorRetries": 0,
    "InternalServerErrorRetries": 1,
    "ServiceUnavailableErrorRetries": 1,
}


def render_litellm(profile: Profile) -> dict[str, Any]:
    deployments = []
    for instance in profile.instances:
        model = instance.model
        deployment = {"model_name": model.alias, "litellm_params": {
            "model": f"openai/{model.alias}", "api_base": f"{instance.url}/v1",
            "api_key": "os.environ/LLM_BACKEND_KEY", "drop_params": True,
            "max_parallel_requests": model.max_num_active_seqs + model.max_num_queued_reqs,
        }, "model_info": {"mode": "embedding" if model.embed else "chat"}}
        deployments.append(deployment)
        if len(model.gpus) > 1:
            deployments.append({**deployment, "model_name": f"{model.alias}-{instance.index + 1}"})
    return {"model_list": deployments,
            "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
            "litellm_settings": {"request_timeout": 600},
            "router_settings": {"routing_strategy": os.environ.get("LITELLM_ROUTING_STRATEGY", "least-busy"),
                                "num_retries": 1, "retry_policy": RETRY_POLICY, "fallbacks": []}}
