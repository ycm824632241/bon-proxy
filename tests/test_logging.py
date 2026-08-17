from __future__ import annotations

import json
import logging

import httpx
import pytest

from bon_proxy.app import create_app
from bon_proxy.config import AppConfig
from bon_proxy.logging_setup import configure_logging
from tests.helpers import answer_response, choice, judge_response
from tests.test_api import RecordingTransport, basic_request, proxy_client


@pytest.mark.anyio
async def test_logs_do_not_include_prompts_or_api_keys(app_config, caplog) -> None:
    caplog.set_level(logging.INFO)
    answer = RecordingTransport([httpx.ConnectError("offline")])
    judge = RecordingTransport([])
    app = create_app(app_config, answer_transport=answer, judge_transport=judge)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=basic_request(messages=[{"role": "user", "content": "highly-sensitive-prompt"}]),
        )

    assert response.status_code == 502
    assert "highly-sensitive-prompt" not in caplog.text
    assert "answer-secret" not in caplog.text
    assert "judge-secret" not in caplog.text


def test_configure_logging_writes_file(tmp_path) -> None:
    log_file = tmp_path / "proxy.log"
    configure_logging("INFO", str(log_file))
    try:
        logging.getLogger("bon_proxy.test").info("hello-file-log")
        text = log_file.read_text(encoding="utf-8")
        assert "hello-file-log" in text
    finally:
        logging.basicConfig(level=logging.WARNING, force=True, handlers=[logging.StreamHandler()])


@pytest.mark.anyio
async def test_payload_log_writes_candidates(config_dict, tmp_path) -> None:
    log_file = tmp_path / "proxy.log"
    config_dict["server"]["log_file"] = str(log_file)
    config_dict["server"]["log_payloads"] = True
    config = AppConfig.model_validate(config_dict)
    answer = RecordingTransport(
        [
            answer_response(
                [
                    choice(0, "candidate zero", [10, 11]),
                    choice(1, "candidate one", [20, 21, 22, 23]),
                    choice(2, "candidate two", [30]),
                ]
            )
        ]
    )
    judge = RecordingTransport([judge_response(1)])

    async with proxy_client(config, answer, judge) as client:
        response = await client.post("/v1/chat/completions", json=basic_request())

    assert response.status_code == 200
    payload_path = tmp_path / "proxy.payloads.jsonl"
    lines = payload_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "ok"
    assert record["selected_index"] == 1
    assert record["votes"] == [1, 1, 1, 1]
    assert [item["content"] for item in record["candidates"]] == [
        "candidate zero",
        "candidate one",
        "candidate two",
    ]
    assert record["selected"]["content"] == "candidate one"
    assert record["request"]["messages"][0]["content"] == "Write quicksort"
    assert "api_key" not in record["request"]
