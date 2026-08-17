"""Optional JSONL writer for Best-of-N request payloads."""

from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bon_proxy.config import ServerConfig


def sanitize_request(request_body: dict[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(request_body)
    body.pop("api_key", None)
    return body


def snapshot_choice(choice: dict[str, Any]) -> dict[str, Any]:
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    return {
        "index": choice.get("index"),
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
        "function_call": message.get("function_call"),
        "reasoning_content": message.get("reasoning_content"),
    }


def snapshot_candidates(choices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [snapshot_choice(choice) for choice in choices]


class RequestLogWriter:
    """Append one JSON object per completed (or failed) proxy workflow."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_server_config(cls, server: ServerConfig) -> RequestLogWriter:
        if not server.log_payloads or not server.log_payloads_file:
            return cls(None)
        return cls(Path(server.log_payloads_file).expanduser())

    def write(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
