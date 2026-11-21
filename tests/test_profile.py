"""Profiles: environment substitution, extends, gpus: all, and validation."""

from pathlib import Path

import pytest
import yaml

from llm_setup.profile import ProfileError, load_profile, parse_profile, read_profile, resolve_gpus

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles"


def raw(name: str, gpu_count: int = 3) -> dict:
    return resolve_gpus(read_profile(PROFILES / f"{name}.yaml"), gpu_count)


def test_every_shipped_profile_validates():
    for path in PROFILES.glob("*.yaml"):
        assert load_profile(path, 3).instances


def test_gpus_all_gives_one_replica_per_allocated_gpu():
    one = load_profile(PROFILES / "orch.yaml", 1)
    three = load_profile(PROFILES / "orch.yaml", 3)
    assert [(i.name, i.gpu, i.port) for i in one.instances] == [("orch", 0, 8001)]
    assert [(i.name, i.gpu, i.port) for i in three.instances] == [
        ("orch", 0, 8001), ("orch-2", 1, 8002), ("orch-3", 2, 8003)]
    # The replicas run the same model settings, so cached results stay valid across GPU counts.
    assert one.models["orch"].model == three.models["orch"].model
    assert one.models["orch"].revision == three.models["orch"].revision


def test_gpus_all_counts_the_visible_gpus_and_explains_when_it_cannot(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert len(load_profile(PROFILES / "orch.yaml").instances) == 2
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr("llm_setup.profile.shutil.which", lambda _name: None)
    with pytest.raises(ProfileError, match="--gpus N"):
        load_profile(PROFILES / "orch.yaml")


def test_gpus_all_takes_only_gpus_no_other_model_claims():
    data = read_profile(PROFILES / "heavy-lite-embed.yaml")
    data["models"]["lite"]["gpus"] = "all"                  # heavy holds GPU 0, embed GPU 2
    assert resolve_gpus(data, 4)["models"]["lite"]["gpus"] == [1, 3]
    data = raw("orch")
    data["models"]["other"] = {**data["models"]["orch"], "gpus": "all", "alias": "x"}
    data["models"]["orch"]["gpus"] = "all"
    with pytest.raises(ProfileError, match="only one model"):
        resolve_gpus(data, 3)


def test_a_profile_needing_more_gpus_than_the_allocation_is_refused():
    with pytest.raises(ProfileError, match="has 1 GPU"):
        load_profile(PROFILES / "heavy-lite-embed.yaml", 1)


def test_extends_merges_the_base_and_environment_defaults_apply(monkeypatch):
    monkeypatch.delenv("GLINER_PORT", raising=False)
    base = load_profile(PROFILES / "orch.yaml", 3)
    gliner = load_profile(PROFILES / "orch-gliner.yaml", 3)
    assert gliner.models == base.models and gliner.name == "orch-gliner"
    assert gliner.gliner.port == 8090 and base.gliner is None
    monkeypatch.setenv("GLINER_PORT", "8091")
    assert load_profile(PROFILES / "orch-gliner.yaml", 3).gliner.port == 8091


def test_extends_is_relative_to_the_profile_and_mappings_merge(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(raw("orch")))
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nname: child\nmodels: {orch: {max_model_len: 4096}}\n")
    child = load_profile(tmp_path / "child.yaml")
    assert child.name == "child" and child.models["orch"].max_model_len == 4096
    assert child.models["orch"].alias == "orch-model"


def test_environment_variable_replaces_the_embedding_model(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_ID", "org/qwen-embedder")
    assert load_profile(PROFILES / "heavy-lite-embed.yaml").models["embed"].model == "org/qwen-embedder"


@pytest.mark.parametrize(("change", "message"), [
    (lambda d: d["models"]["lite"].update(gpus=[0]), "GPU 0 is assigned to both heavy and lite"),
    (lambda d: d["models"]["lite"].update(port=8002), "port 8002 is assigned"),
    (lambda d: d["models"]["heavy"].update(max_num_queued_reqs=0), "positive integer"),
    (lambda d: d["models"]["heavy"].update(max_num_queued_request=7), "unsupported fields"),
    (lambda d: d["models"]["heavy"].update(revision="  "), "revision"),
    (lambda d: d["models"]["heavy"].update(kv_cache_dtype="int4"), "kv_cache_dtype"),
    (lambda d: d["models"]["embed"].update(kv_cache_dtype="fp8"), "unsupported fields"),
    (lambda d: d["models"]["embed"].update(tool_call_parser="hermes"), "unsupported fields"),
    (lambda d: d["models"]["heavy"].update(gpu_memory_utilization=1.5), "gpu_memory_utilization"),
    (lambda d: d["models"]["heavy"].update(gpus=[]), "gpus"),
    (lambda d: d["models"]["lite"].update(alias="heavy-model"), "own alias"),
    (lambda d: d["models"].update(gateway=d["models"].pop("lite")), "reserved"),
    (lambda d: d.update(bind_host="0.0.0.0"), "unsupported profile fields"),
    (lambda d: d.update(gateway_port=8001), "port 8001 is assigned"),
    (lambda d: d["models"]["heavy"].pop("port"), "missing: port"),
])
def test_invalid_profiles_are_refused(change, message):
    data = raw("heavy-lite-embed")
    change(data)
    with pytest.raises(ProfileError, match=message):
        parse_profile(data)


def test_replica_ports_cannot_collide_with_another_model():
    data = raw("heavy-lite-embed")
    data["models"]["embed"]["gpus"] = [2, 3]     # embed on 8001 and 8002, heavy's port
    with pytest.raises(ProfileError, match="port 8002"):
        parse_profile(data)


def test_status_keys_name_replicas_by_port():
    assert list(load_profile(PROFILES / "orch.yaml", 3).status_keys()) == [
        "orch-model@8001", "orch-model@8002", "orch-model@8003"]
    assert list(load_profile(PROFILES / "orch.yaml", 1).status_keys()) == ["orch-model"]


def test_the_resolved_copy_a_session_stores_loads_without_gpus(tmp_path):
    profile = load_profile(PROFILES / "orch-gliner.yaml", 3)
    (tmp_path / "profile.yaml").write_text(yaml.safe_dump(profile.data))
    assert load_profile(tmp_path / "profile.yaml").instances == profile.instances
