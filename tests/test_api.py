from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from openai import OpenAI

from bon_proxy.app import create_app
from bon_proxy.config import AppConfig
from tests.helpers import answer_response, choice, judge_response


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(request.headers)
        if not self.responses:
            raise AssertionError("unexpected upstream request")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return httpx.Response(200, json=result, request=request)


class BlockingTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.started.set()
        await self.release.wait()
        return httpx.Response(200, json=self.response, request=request)


@asynccontextmanager
async def proxy_client(
    config: AppConfig,
    answer_transport: httpx.AsyncBaseTransport,
    judge_transport: httpx.AsyncBaseTransport,
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        config,
        answer_transport=answer_transport,
        judge_transport=judge_transport,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client,
    ):
        yield client


def basic_request(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "virtual-model",
        "messages": [{"role": "user", "content": "Write quicksort"}],
    }
    body.update(overrides)
    return body


@pytest.mark.anyio
async def test_success_calls_answer_and_judge_once_and_returns_best(app_config) -> None:
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
    body = basic_request(
        n=99,
        temperature=0.2,
        top_p=0.2,
        chat_template_kwargs={"thinking": False},
        response_format={"type": "json_object"},
        custom_vllm_field="preserved",
    )

    async with proxy_client(app_config, answer, judge) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["choices"]) == 1
    assert payload["choices"][0]["index"] == 1
    assert payload["choices"][0]["message"]["content"] == "candidate one"
    assert payload["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }
    assert len(answer.requests) == 1
    assert len(judge.requests) == 1
    assert answer.requests[0]["model"] == "answer-model"
    assert answer.requests[0]["n"] == 3
    assert answer.requests[0]["temperature"] == 1.0
    assert answer.requests[0]["top_p"] == 0.95
    assert answer.requests[0]["return_token_ids"] is True
    assert answer.requests[0]["response_format"] == {"type": "json_object"}
    assert answer.requests[0]["custom_vllm_field"] == "preserved"
    assert answer.headers[0]["authorization"] == "Bearer answer-secret"
    assert judge.headers[0]["authorization"] == "Bearer judge-secret"
    assert judge.requests[0]["n"] == 4
    assert response.headers["x-request-id"].startswith("req_")


@pytest.mark.anyio
async def test_tool_call_choice_is_preserved(app_config) -> None:
    tool_call = {
        "id": "call_lookup",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"id":42}'},
    }
    answer = RecordingTransport(
        [
            answer_response(
                [
                    choice(0, "plain answer", [1]),
                    choice(1, None, [2, 3], tool_calls=[tool_call]),
                ]
            )
        ]
    )
    judge = RecordingTransport([judge_response(1)])

    async with proxy_client(app_config, answer, judge) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=basic_request(tools=[{"type": "function", "function": {"name": "lookup"}}]),
        )

    assert response.status_code == 200
    selected = response.json()["choices"][0]
    assert selected["finish_reason"] == "tool_calls"
    assert selected["message"]["content"] is None
    assert selected["message"]["tool_calls"] == [tool_call]


@pytest.mark.anyio
async def test_judge_invalid_output_falls_back_to_first_candidate(app_config) -> None:
    answer = RecordingTransport(
        [answer_response([choice(0, "fallback", [1, 2]), choice(1, "other", [3])])]
    )
    judge = RecordingTransport(
        [{"choices": [{"message": {"role": "assistant", "content": '{"best_index":99}'}}]}]
    )

    async with proxy_client(app_config, answer, judge) as client:
        response = await client.post("/v1/chat/completions", json=basic_request())

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fallback"
    assert response.json()["usage"]["completion_tokens"] == 2


@pytest.mark.anyio
async def test_judge_timeout_falls_back_to_first_candidate(app_config) -> None:
    answer = RecordingTransport(
        [answer_response([choice(0, "fallback", [1]), choice(1, "other", [2])])]
    )
    judge = RecordingTransport([httpx.ReadTimeout("slow judge")])

    async with proxy_client(app_config, answer, judge) as client:
        response = await client.post("/v1/chat/completions", json=basic_request())

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fallback"


@pytest.mark.anyio
async def test_single_candidate_skips_judge(app_config) -> None:
    answer = RecordingTransport([answer_response([choice(0, "only", [1, 2, 3])])])
    judge = RecordingTransport([])

    async with proxy_client(app_config, answer, judge) as client:
        response = await client.post("/v1/chat/completions", json=basic_request())

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "only"
    assert len(judge.requests) == 0


@pytest.mark.anyio
async def test_no_valid_candidates_returns_502(app_config) -> None:
    answer = RecordingTransport([answer_response([{"index": 0, "message": None}])])
    judge = RecordingTransport([])

    async with proxy_client(app_config, answer, judge) as client:
        response = await client.post("/v1/chat/completions", json=basic_request())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "no_valid_candidates"


@pytest.mark.anyio
async def test_missing_selected_token_ids_returns_502(app_config) -> None:
    invalid_choice = choice(0, "answer", [])
    invalid_choice.pop("token_ids")
    answer = RecordingTransport([answer_response([invalid_choice])])
    judge = RecordingTransport([])

    async with proxy_client(app_config, answer, judge) as client:
        response = await client.post("/v1/chat/completions", json=basic_request())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "missing_token_ids"


@pytest.mark.anyio
async def test_sglang_without_token_ids_returns_selected_candidate(config_dict) -> None:
    data = copy.deepcopy(config_dict)
    data["answer"]["return_token_ids"] = False
    config = AppConfig.model_validate(data)
    first = choice(0, "candidate zero", [])
    selected = choice(1, "candidate one", [])
    first.pop("token_ids")
    selected.pop("token_ids")
    upstream = answer_response([first, selected])
    upstream["usage"] = {"prompt_tokens": 11, "completion_tokens": 8, "total_tokens": 19}
    answer = RecordingTransport([upstream])
    judge = RecordingTransport([judge_response(1)])

    async with proxy_client(config, answer, judge) as client:
        response = await client.post("/v1/chat/completions", json=basic_request())

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "candidate one"
    assert response.json()["usage"] == upstream["usage"]
    assert "return_token_ids" not in answer.requests[0]


@pytest.mark.anyio
async def test_answer_timeout_returns_504(app_config) -> None:
    answer = RecordingTransport([httpx.ReadTimeout("slow answer")])
    judge = RecordingTransport([])

    async with proxy_client(app_config, answer, judge) as client:
        response = await client.post("/v1/chat/completions", json=basic_request())

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "answer_timeout"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "code"),
    [
        (basic_request(stream=True), "stream_not_supported"),
        (basic_request(stream="yes"), "invalid_stream"),
        ({"model": "virtual-model", "messages": []}, "invalid_messages"),
        ({"messages": [{"role": "user", "content": "hi"}]}, "invalid_model"),
    ],
)
async def test_request_validation_returns_openai_error(app_config, body, code) -> None:
    answer = RecordingTransport([])
    judge = RecordingTransport([])

    async with proxy_client(app_config, answer, judge) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["code"] == code
    assert not answer.requests


@pytest.mark.anyio
async def test_invalid_json_returns_openai_error(app_config) -> None:
    answer = RecordingTransport([])
    judge = RecordingTransport([])

    async with proxy_client(app_config, answer, judge) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=b"{",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


@pytest.mark.anyio
async def test_concurrency_limit_returns_429_without_queueing(config_dict) -> None:
    data = copy.deepcopy(config_dict)
    data["server"]["max_concurrency"] = 1
    config = AppConfig.model_validate(data)
    blocking_answer = BlockingTransport(
        answer_response([choice(0, "first", [1]), choice(1, "second", [2])])
    )
    judge = RecordingTransport([judge_response(0)])

    async with proxy_client(config, blocking_answer, judge) as client:
        first_task = asyncio.create_task(client.post("/v1/chat/completions", json=basic_request()))
        await asyncio.wait_for(blocking_answer.started.wait(), timeout=1)
        second = await client.post("/v1/chat/completions", json=basic_request())
        blocking_answer.release.set()
        first = await asyncio.wait_for(first_task, timeout=1)

    assert second.status_code == 429
    assert second.json()["error"]["code"] == "concurrency_limit"
    assert first.status_code == 200


@pytest.mark.anyio
async def test_health_and_models(app_config) -> None:
    answer = RecordingTransport([])
    judge = RecordingTransport([])

    async with proxy_client(app_config, answer, judge) as client:
        health = await client.get("/health")
        models = await client.get("/v1/models")

    assert health.json() == {"status": "ok"}
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "answer-model"


def test_official_openai_client_parses_proxy_response() -> None:
    payload = answer_response([choice(0, "selected answer", [1, 2])])
    payload["choices"][0].pop("token_ids")
    payload.pop("prompt_token_ids")
    payload["usage"] = {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAI(api_key="unused", base_url="http://proxy.test/v1", http_client=http_client)

    completion = client.chat.completions.create(
        model="virtual-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert len(completion.choices) == 1
    assert completion.choices[0].message.content == "selected answer"
    assert completion.usage is not None
    assert completion.usage.completion_tokens == 2
