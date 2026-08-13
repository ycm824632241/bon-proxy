"""FastAPI application exposing the OpenAI-compatible surface."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from bon_proxy.concurrency import ConcurrencyGate
from bon_proxy.config import AppConfig
from bon_proxy.errors import ProxyError
from bon_proxy.service import BestOfNService
from bon_proxy.upstream import VLLMClient

logger = logging.getLogger(__name__)


def create_app(
    config: AppConfig,
    *,
    answer_transport: httpx.AsyncBaseTransport | None = None,
    judge_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    answer_client = VLLMClient("answer", config.answer, transport=answer_transport)
    judge_client = VLLMClient("judge", config.judge, transport=judge_transport)
    service = BestOfNService(config, answer_client, judge_client)
    gate = ConcurrencyGate(config.server.max_concurrency)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="bon-proxy", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.service = service
    app.state.gate = gate

    @app.exception_handler(ProxyError)
    async def proxy_error_handler(request: Request, exc: ProxyError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        headers = {"x-request-id": request_id} if request_id else None
        return JSONResponse(exc.as_dict(), status_code=exc.status_code, headers=headers)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.answer.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "bon-proxy",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
        request.state.request_id = request_id
        body = await _read_request_body(request)
        _validate_request_body(body)

        async with gate.slot() as acquired:
            if not acquired:
                logger.warning("request=%s concurrency_limit", request_id)
                raise ProxyError(
                    429,
                    "The proxy concurrency limit has been reached.",
                    "rate_limit_error",
                    "concurrency_limit",
                )
            response = await service.complete(body, request_id)

        return JSONResponse(response, headers={"x-request-id": request_id})

    return app


async def _read_request_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProxyError(
            400,
            "The request body must be valid JSON.",
            "invalid_request_error",
            "invalid_json",
        ) from exc
    if not isinstance(body, dict):
        raise ProxyError(
            400,
            "The request body must be a JSON object.",
            "invalid_request_error",
            "invalid_body",
        )
    return body


def _validate_request_body(body: dict[str, Any]) -> None:
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ProxyError(
            400,
            "The model field is required and must be a non-empty string.",
            "invalid_request_error",
            "invalid_model",
            "model",
        )
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ProxyError(
            400,
            "The messages field is required and must be a non-empty array.",
            "invalid_request_error",
            "invalid_messages",
            "messages",
        )
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise ProxyError(
            400,
            "The stream field must be a boolean.",
            "invalid_request_error",
            "invalid_stream",
            "stream",
        )
    if stream:
        raise ProxyError(
            400,
            "Streaming is not supported by this Best-of-N proxy.",
            "invalid_request_error",
            "stream_not_supported",
            "stream",
        )
