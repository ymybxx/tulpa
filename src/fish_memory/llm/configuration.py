"""Resolved desktop module endpoints; CLI environment settings remain unchanged."""

from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class ClientArguments(TypedDict):
    base_url: str
    model: str
    api_key: str


class LanguageModelEndpoint(BaseModel):
    model_config = ConfigDict(
        frozen=True, populate_by_name=True, extra="forbid", hide_input_in_errors=True
    )

    base_url: str = Field(alias="baseUrl", min_length=1)
    model: str = Field(min_length=1)
    api_key: SecretStr = Field(alias="apiKey")

    @field_validator("api_key")
    @classmethod
    def require_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("model API key is required")
        return value

    def client_arguments(self) -> ClientArguments:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.api_key.get_secret_value(),
        }


class DesktopModelRoutes(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    chat: LanguageModelEndpoint
    reasoning: LanguageModelEndpoint
