#!/usr/bin/env python3
"""Smoke-test proxy_v2 file logging (log_file + log_payloads_file).

Offline (default): mock upstream, no SGLang needed.
Live: hit a running proxy and check that payloads JSONL grows.

Examples:
  PYTHONPATH=src python scripts/smoke_logging.py
  PYTHONPATH=src python scripts/smoke_logging.py --live http://127.0.0.1:28085 \\
      --log-file /kwkj-k8s/llm_team/ycm/logs/proxy_v2_sglang30020.log \\
      --payloads-file /kwkj-k8s/llm_team/ycm/logs/proxy_v2_sglang30020.payloads.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bon_proxy.config import load_config  # noqa: E402
from bon_proxy.logging_setup import configure_logging  # noqa: E402
from tests.helpers import answer_response, choice, judge_response  # noqa: E402
from tests.test_api import RecordingTransport, basic_request, proxy_client  # noqa: E402


def _offline(tmp: Path) -> None:
    log_file = tmp / "proxy.log"
    payloads_file = tmp / "proxy.payloads.jsonl"
    raw = {
        "server": {
            "host": "127.0.0.1",
            "port": 18085,
            "max_concurrency": 2,
            "log_level": "INFO",
            "log_file": str(log_file),
            "log_payloads": True,
            "log_payloads_file": str(payloads_file),
        },
        "answer": {
            "base_url": "http://answer.test/v1",
            "api_key": "",
            "model": "answer-model",
            "timeout_seconds": 5,
            "params": {"temperature": 1.0, "top_p": 0.95, "n": 3},
        },
        "judge": {
            "base_url": "http://judge.test/v1",
            "api_key": "",
            "model": "judge-model",
            "timeout_seconds": 5,
            "prompt": "Choose the best candidate.",
            "params": {"temperature": 0.1, "top_p": 0.95, "n": 4},
        },
    }
    cfg_path = tmp / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(cfg_path)
    configure_logging("INFO", str(log_file))
    assert config.server.log_payloads_file == str(payloads_file)

    answer = RecordingTransport(
        [
            answer_response(
                [
                    choice(0, "smoke-cand-0", [1]),
                    choice(1, "smoke-cand-1", [2, 3]),
                    choice(2, "smoke-cand-2", [4]),
                ]
            )
        ]
    )
    judge = RecordingTransport([judge_response(1)])

    async def _run() -> None:
        async with proxy_client(config, answer, judge) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=basic_request(messages=[{"role": "user", "content": "smoke-hello"}]),
            )
            if response.status_code != 200:
                raise SystemExit(f"offline request failed: {response.status_code} {response.text}")

    import asyncio

    asyncio.run(_run())

    if not log_file.is_file() or "file_logging" not in log_file.read_text(encoding="utf-8"):
        # file_logging line is from configure_logging; process logs from service also ok
        text = log_file.read_text(encoding="utf-8") if log_file.is_file() else ""
        if "workflow_complete" not in text and "answer_complete" not in text:
            raise SystemExit(f"log_file missing process lines:\n{text[:500]}")

    lines = payloads_file.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 1:
        raise SystemExit(f"expected 1 payload line, got {len(lines)}")
    record = json.loads(lines[0])
    if record.get("status") != "ok":
        raise SystemExit(f"payload status not ok: {record}")
    if record.get("selected_index") != 1:
        raise SystemExit(f"selected_index != 1: {record.get('selected_index')}")
    contents = [c.get("content") for c in record.get("candidates") or []]
    if contents != ["smoke-cand-0", "smoke-cand-1", "smoke-cand-2"]:
        raise SystemExit(f"candidates mismatch: {contents}")
    if record.get("request", {}).get("messages", [{}])[0].get("content") != "smoke-hello":
        raise SystemExit("request messages not logged")
    print("OFFLINE SMOKE OK")
    print(f"  log_file          {log_file}")
    print(f"  log_payloads_file {payloads_file}")
    print(f"  selected_index    {record['selected_index']}")
    print(f"  votes             {record.get('votes')}")


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _live(base_url: str, log_file: Path, payloads_file: Path, model: str) -> None:
    health = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5.0)
    health.raise_for_status()
    before_log = _file_size(log_file)
    before_pay = _file_size(payloads_file)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: SMOKE_OK"}],
    }
    print(f"POST {base_url}/v1/chat/completions ...")
    response = httpx.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=body,
        timeout=180.0,
        headers={"x-request-id": "smoke_logging_live"},
    )
    print(f"  http {response.status_code}")
    if response.status_code != 200:
        raise SystemExit(f"live request failed: {response.status_code} {response.text[:400]}")
    after_log = _file_size(log_file)
    after_pay = _file_size(payloads_file)
    if after_pay <= before_pay:
        raise SystemExit(
            "payloads JSONL did not grow. Restart proxy so it loads log_payloads_file, then retry.\n"
            f"  {payloads_file} size {before_pay} -> {after_pay}"
        )
    last = payloads_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    record = json.loads(last)
    n_cand = len(record.get("candidates") or [])
    print("LIVE SMOKE OK")
    print(f"  log_file          {log_file} ({before_log} -> {after_log} bytes)")
    print(f"  log_payloads_file {payloads_file} ({before_pay} -> {after_pay} bytes)")
    print(f"  status            {record.get('status')}")
    print(f"  candidates        {n_cand}")
    print(f"  selected_index    {record.get('selected_index')}")
    print(f"  votes             {record.get('votes')}")
    selected = (record.get("selected") or {}).get("content")
    if isinstance(selected, str):
        print(f"  selected preview  {selected[:200]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test proxy_v2 logging")
    parser.add_argument("--live", metavar="BASE_URL", help="Hit a running proxy, e.g. http://127.0.0.1:28085")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("/kwkj-k8s/llm_team/ycm/logs/proxy_v2_sglang30020.log"),
    )
    parser.add_argument(
        "--payloads-file",
        type=Path,
        default=Path("/kwkj-k8s/llm_team/ycm/logs/proxy_v2_sglang30020.payloads.jsonl"),
    )
    parser.add_argument("--model", default="DeepSeek-V4-Flash-0731")
    ns = parser.parse_args()
    if ns.live:
        _live(ns.live, ns.log_file, ns.payloads_file, ns.model)
        return 0
    with tempfile.TemporaryDirectory(prefix="bon-proxy-smoke-") as tmp:
        _offline(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
