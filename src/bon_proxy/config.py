"""YAML configuration models and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigLoadError(ValueError):
    """Raised when the YAML configuration cannot be loaded or validated."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerConfig(StrictModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    max_concurrency: int = Field(default=32, ge=1)
    log_level: str = "INFO"
    # Optional process log (INFO lines: latency, votes, errors). stderr is always kept.
    log_file: str | None = None
    # When true, also append one JSON object per request: messages, N candidates,
    # judge votes, and the selected answer. Requires log_file or log_payloads_file.
    log_payloads: bool = False
    log_payloads_file: str | None = None

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("host must not be empty")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("log_level must be CRITICAL, ERROR, WARNING, INFO, or DEBUG")
        return normalized

    @field_validator("log_file", "log_payloads_file")
    @classmethod
    def empty_path_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def resolve_payload_log_path(self) -> Self:
        if not self.log_payloads:
            return self
        if self.log_payloads_file:
            return self
        if not self.log_file:
            raise ValueError("log_payloads requires log_file or log_payloads_file")
        log_path = Path(self.log_file)
        self.log_payloads_file = str(log_path.with_name(f"{log_path.stem}.payloads.jsonl"))
        return self


class GenerationParams(StrictModel):
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    n: int = Field(ge=1)
    reasoning_effort: str | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class UpstreamConfig(StrictModel):
    base_url: str
    api_key: str = ""
    model: str
    timeout_seconds: float = Field(gt=0)
    params: GenerationParams

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model must not be empty")
        return value


class AnswerConfig(UpstreamConfig):
    # vLLM and newer SGLang releases can return per-choice token IDs.  SGLang
    # v0.5.16 cannot, so deployments using that version must disable this.
    return_token_ids: bool = True

    @model_validator(mode="after")
    def validate_candidate_count(self) -> Self:
        if self.params.n < 2:
            raise ValueError("answer.params.n must be at least 2")
        return self


class JudgeConfig(UpstreamConfig):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("judge.prompt must not be empty")
        return value

    @model_validator(mode="after")
    def validate_judgement_count(self) -> Self:
        if self.params.n != 4:
            raise ValueError("judge.params.n must equal 4")
        return self


class AppConfig(StrictModel):
    server: ServerConfig
    answer: AnswerConfig
    judge: JudgeConfig


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an application configuration from YAML."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigLoadError(f"cannot read config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigLoadError(f"config file {config_path} must contain a YAML mapping")

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigLoadError(f"invalid configuration in {config_path}:\n{exc}") from exc
