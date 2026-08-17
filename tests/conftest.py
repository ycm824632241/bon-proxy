from __future__ import annotations

from typing import Any

import pytest

from bon_proxy.config import AppConfig


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def config_dict() -> dict[str, Any]:
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "max_concurrency": 2,
            "log_level": "INFO",
        },
        "answer": {
            "base_url": "http://answer.test/v1",
            "api_key": "answer-secret",
            "model": "answer-model",
            "timeout_seconds": 1,
            "params": {
                "temperature": 1.0,
                "top_p": 0.95,
                "n": 3,
                "reasoning_effort": "max",
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "max",
                },
            },
        },
        "judge": {
            "base_url": "http://judge.test/v1",
            "api_key": "judge-secret",
            "model": "judge-model",
            "timeout_seconds": 1,
            "prompt": "Choose the best candidate.",
            "params": {
                "temperature": 0.1,
                "top_p": 0.9,
                "n": 4,
                "reasoning_effort": "max",
                "chat_template_kwargs": {"thinking": False},
            },
        },
    }


@pytest.fixture
def app_config(config_dict: dict[str, Any]) -> AppConfig:
    return AppConfig.model_validate(config_dict)
