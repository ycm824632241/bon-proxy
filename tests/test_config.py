from __future__ import annotations

import copy

import pytest
import yaml

from bon_proxy.config import AppConfig, ConfigLoadError, load_config


def test_load_valid_yaml(tmp_path, config_dict) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    config = load_config(path)

    assert config.answer.params.n == 3
    assert config.judge.params.n == 4
    assert config.server.log_level == "INFO"
    assert config.server.log_file is None
    assert config.server.log_payloads is False
    assert config.answer.base_url == "http://answer.test/v1"


def test_log_payloads_derives_jsonl_path(config_dict) -> None:
    config_dict["server"]["log_file"] = "/tmp/proxy.log"
    config_dict["server"]["log_payloads"] = True
    config = AppConfig.model_validate(config_dict)
    assert config.server.log_payloads_file == "/tmp/proxy.payloads.jsonl"


def test_log_payloads_requires_a_path(config_dict) -> None:
    config_dict["server"]["log_payloads"] = True
    with pytest.raises(ValueError, match="log_payloads requires"):
        AppConfig.model_validate(config_dict)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("server", "max_concurrency"), 0),
        (("answer", "timeout_seconds"), 0),
        (("answer", "params", "n"), 1),
        (("judge", "params", "n"), 1),
        (("answer", "params", "top_p"), 0),
        (("answer", "params", "temperature"), 2.1),
    ],
)
def test_config_rejects_invalid_values(config_dict, path, value) -> None:
    data = copy.deepcopy(config_dict)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        AppConfig.model_validate(data)


def test_config_rejects_unknown_fields(config_dict) -> None:
    config_dict["server"]["workers"] = 2

    with pytest.raises(ValueError):
        AppConfig.model_validate(config_dict)


def test_load_config_reports_invalid_yaml(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("server: [", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="invalid YAML"):
        load_config(path)


def test_load_config_requires_mapping(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- item", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="YAML mapping"):
        load_config(path)
