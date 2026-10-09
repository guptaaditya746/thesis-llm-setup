"""Model role descriptors and stable status serialization."""

from typing import Any


def model_targets(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    targets = {}
    for role, config in profile["models"].items():
        targets[config["alias"]] = {**config, "role": role,
                                    "backend": f"http://127.0.0.1:{config['port']}"}
    return targets


def model_instances(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten role replicas into individual supervised vLLM process configurations."""
    instances = []
    for role, config in profile["models"].items():
        for index, location in enumerate([config, *config.get("replicas", [])], start=1):
            item = {**config, **location, "role": role}
            item.pop("replicas", None)
            name = role if index == 1 else f"{role}-{index}"
            instances.append({"name": name, **item})
    return instances
