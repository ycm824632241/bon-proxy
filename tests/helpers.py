from __future__ import annotations

import json
import time
from typing import Any


def choice(
    index: int,
    content: str | None,
    token_ids: list[int],
    *,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "index": index,
        "message": message,
        "finish_reason": finish_reason,
        "logprobs": None,
        "token_ids": token_ids,
    }


def answer_response(choices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-answer",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "answer-model",
        "choices": choices,
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": sum(len(item.get("token_ids", [])) for item in choices),
            "total_tokens": 11 + sum(len(item.get("token_ids", [])) for item in choices),
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
        "prompt_token_ids": [1, 2, 3],
    }


def judge_response(best_index: int | list[int]) -> dict[str, Any]:
    indexes = [best_index] * 4 if isinstance(best_index, int) else best_index
    return {
        "id": "chatcmpl-judge",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "judge-model",
        "choices": [
            {
                "index": choice_index,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"best_index": selected_index}),
                },
                "finish_reason": "stop",
            }
            for choice_index, selected_index in enumerate(indexes)
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
    }
