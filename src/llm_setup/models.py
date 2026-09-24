"""Model role descriptors and stable status serialization."""

from typing import Any


def model_targets(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {config["alias"]: {**config, "role": role,
            "backend": f"http://127.0.0.1:{config['port']}"}
            for role, config in profile["models"].items()}
