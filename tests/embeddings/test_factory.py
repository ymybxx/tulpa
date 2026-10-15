from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import SecretStr

from fish_memory.config import Settings
from fish_memory.embeddings.client import (
    LocalOnnxEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)
from fish_memory.embeddings.factory import create_embedding_client


def base_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql://test",
        "llm_provider": "openai",
        "llm_api_key": SecretStr("llm-key"),
        "llm_model": "qwen-plus",
        "llm_base_url": "https://llm.invalid/v1",
    }
    values.update(overrides)
    return Settings.model_construct(**values)


def test_default_factory_creates_local_multilingual_e5_client() -> None:
    client = create_embedding_client(base_settings())

    assert isinstance(client, LocalOnnxEmbeddingClient)
    assert client.space.provider == "local"
    assert client.space.model == "intfloat/multilingual-e5-small"
    assert client.space.dimensions == 384


@pytest.mark.asyncio
async def test_default_factory_loads_pinned_official_int8_onnx(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "onnx" / "model_qint8_avx512_vnni.onnx"
    artifact.parent.mkdir()
    artifact.touch()
    snapshot_calls: list[dict[str, object]] = []
    session_paths: list[str] = []

    def snapshot_download(**kwargs: object) -> str:
        snapshot_calls.append(kwargs)
        return str(tmp_path)

    class ArtifactSession:
        def get_inputs(self) -> list[Any]:
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
            ]

        def get_outputs(self) -> list[Any]:
            return [SimpleNamespace(type="tensor(float)")]

        def run(
            self,
            _outputs: object,
            inputs: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            batch_size = inputs["input_ids"].shape[0]
            return [np.ones((batch_size, 2, 384), dtype=np.float32)]

    class ArtifactTokenizer:
        def __call__(
            self,
            _texts: list[str],
            **_kwargs: object,
        ) -> dict[str, np.ndarray]:
            return {
                "input_ids": np.array([[1, 2]], dtype=np.int64),
                "attention_mask": np.array([[1, 1]], dtype=np.int64),
            }

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: ArtifactTokenizer(),
    )
    monkeypatch.setattr(
        "onnxruntime.InferenceSession",
        lambda model_path, **_kwargs: (
            session_paths.append(str(model_path)) or ArtifactSession()
        ),
    )
    client = create_embedding_client(base_settings())

    assert isinstance(client, LocalOnnxEmbeddingClient)
    await client.initialize()

    assert session_paths == [str(artifact)]
    assert snapshot_calls == [
        {
            "repo_id": "intfloat/multilingual-e5-small",
            "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
            "allow_patterns": [
                "onnx/model_qint8_avx512_vnni.onnx",
                "config.json",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            "local_files_only": True,
        }
    ]


def test_remote_factory_uses_only_dedicated_embedding_configuration() -> None:
    client = create_embedding_client(
        base_settings(
            embedding_provider="openai-compatible",
            embedding_api_key=SecretStr("embedding-key"),
            embedding_base_url="https://embedding.invalid/v1",
            embedding_model="remote-embedding",
            embedding_dimensions=768,
        )
    )

    assert isinstance(client, OpenAICompatibleEmbeddingClient)
    assert client.space.model == "remote-embedding"
    assert client.space.dimensions == 768
