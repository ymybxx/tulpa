import secrets
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from fish_memory.tts.catalog import get_tts_catalog

DEFAULT_UPLOAD_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"
DEFAULT_CUSTOMIZATION_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
)


class CustomVoiceProviderError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        provider_code: str,
        provider_message: str,
        request_id: str | None = None,
    ) -> None:
        super().__init__(provider_message)
        self.status_code = status_code
        self.provider_code = provider_code
        self.provider_message = provider_message
        self.request_id = request_id


class CustomVoice(BaseModel):
    id: str = Field(serialization_alias="voiceId")
    target_model: str = Field(serialization_alias="targetModel")
    status: str
    created_at: datetime | None = Field(
        default=None,
        serialization_alias="createdAt",
    )


class CustomVoiceList(BaseModel):
    items: list[CustomVoice]


class _UploadPolicyData(BaseModel):
    policy: str
    signature: str
    upload_dir: str
    upload_host: str
    oss_access_key_id: str
    x_oss_object_acl: str
    x_oss_forbid_overwrite: str


class _UploadPolicyResponse(BaseModel):
    data: _UploadPolicyData


class _CreateOutput(BaseModel):
    voice_id: str


class _CreateResponse(BaseModel):
    output: _CreateOutput


class _VoiceListItem(BaseModel):
    voice_id: str
    status: str = "UNKNOWN"
    gmt_create: datetime | None = None


class _VoiceListOutput(BaseModel):
    voice_list: list[_VoiceListItem] = Field(default_factory=list)


class _VoiceListResponse(BaseModel):
    output: _VoiceListOutput


class DashScopeVoiceClient:
    def __init__(
        self,
        *,
        api_key: str,
        customization_url: str = DEFAULT_CUSTOMIZATION_URL,
        upload_url: str = DEFAULT_UPLOAD_URL,
        http_client: httpx.AsyncClient | None = None,
        prefix_factory: Callable[[], str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._customization_url = customization_url
        self._upload_url = upload_url
        self._client = http_client or httpx.AsyncClient(timeout=60)
        self._owns_client = http_client is None
        self._prefix_factory = prefix_factory or (lambda: f"fish{secrets.token_hex(3)}")

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def create_voice(
        self,
        audio: bytes,
        *,
        target_model: str,
    ) -> CustomVoice:
        policy = await self._get_upload_policy(target_model)
        object_key = f"{policy.upload_dir}/voice.wav"
        await self._upload_audio(policy, object_key, audio)
        prefix = self._prefix_factory()
        response = await self._post_customization(
            {
                "model": "voice-enrollment",
                "input": {
                    "action": "create_voice",
                    "target_model": target_model,
                    "prefix": prefix,
                    "url": f"oss://{object_key}",
                    # DashScope currently reads only the first language hint.
                    "language_hints": ["zh"],
                    "enable_preprocess": True,
                    "max_prompt_audio_length": 20.0,
                },
            },
            resolve_oss=True,
        )
        parsed = _CreateResponse.model_validate(response)
        return CustomVoice(
            id=parsed.output.voice_id,
            target_model=target_model,
            status="OK",
        )

    async def list_voices(self) -> list[CustomVoice]:
        response = await self._post_customization(
            {
                "model": "voice-enrollment",
                "input": {
                    "action": "list_voice",
                    "page_size": 1000,
                    "page_index": 0,
                },
            }
        )
        parsed = _VoiceListResponse.model_validate(response)
        return [
            CustomVoice(
                id=item.voice_id,
                target_model=_target_model_from_voice_id(item.voice_id),
                status=item.status,
                created_at=item.gmt_create,
            )
            for item in parsed.output.voice_list
        ]

    async def delete_voice(self, voice_id: str) -> None:
        await self._post_customization(
            {
                "model": "voice-enrollment",
                "input": {
                    "action": "delete_voice",
                    "voice_id": voice_id,
                },
            }
        )

    async def _get_upload_policy(self, model: str) -> _UploadPolicyData:
        response = await self._client.get(
            self._upload_url,
            params={"action": "getPolicy", "model": model},
            headers=self._headers(),
        )
        _raise_for_provider_error(response)
        return _UploadPolicyResponse.model_validate(response.json()).data

    async def _upload_audio(
        self,
        policy: _UploadPolicyData,
        object_key: str,
        audio: bytes,
    ) -> None:
        response = await self._client.post(
            policy.upload_host,
            data={
                "OSSAccessKeyId": policy.oss_access_key_id,
                "Signature": policy.signature,
                "policy": policy.policy,
                "x-oss-object-acl": policy.x_oss_object_acl,
                "x-oss-forbid-overwrite": policy.x_oss_forbid_overwrite,
                "key": object_key,
                "success_action_status": "200",
            },
            files={"file": ("voice.wav", audio, "audio/wav")},
        )
        _raise_for_provider_error(response)

    async def _post_customization(
        self,
        payload: dict[str, object],
        *,
        resolve_oss: bool = False,
    ) -> object:
        headers = self._headers()
        if resolve_oss:
            headers["X-DashScope-OssResourceResolve"] = "enable"
        response = await self._client.post(
            self._customization_url,
            headers=headers,
            json=payload,
        )
        _raise_for_provider_error(response)
        return response.json()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }


def _target_model_from_voice_id(voice_id: str) -> str:
    model_ids = sorted(
        (model.id for model in get_tts_catalog()),
        key=len,
        reverse=True,
    )
    return next(
        (model_id for model_id in model_ids if voice_id.startswith(f"{model_id}-")),
        "unknown",
    )


def _raise_for_provider_error(response: httpx.Response) -> None:
    if response.is_success:
        return

    payload = _response_payload(response)
    provider_code = _string_value(payload, "code") or "provider_error"
    provider_message = (
        _string_value(payload, "message")
        or _string_value(payload, "Message")
        or response.reason_phrase
        or "DashScope request failed"
    )
    request_id = (
        _string_value(payload, "request_id")
        or _string_value(payload, "requestId")
        or response.headers.get("x-request-id")
    )
    raise CustomVoiceProviderError(
        status_code=response.status_code,
        provider_code=provider_code,
        provider_message=provider_message,
        request_id=request_id,
    )


def _response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _string_value(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None
