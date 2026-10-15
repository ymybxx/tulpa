from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from fish_memory.embeddings import client as client_module
from fish_memory.embeddings.client import (
    EmbeddingSpace,
    LocalOnnxEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
    _resolve_cached_snapshot,
)


class FakeEmbeddings:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeOpenAI:
    def __init__(self, responses: list[Any]) -> None:
        self.embeddings = FakeEmbeddings(responses)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def embedding_response(
    vectors: list[list[float]],
    *,
    indexes: list[int] | None = None,
    input_tokens: int = 10,
) -> Any:
    resolved_indexes = indexes or list(range(len(vectors)))
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in zip(resolved_indexes, vectors, strict=True)
        ],
        usage=SimpleNamespace(prompt_tokens=input_tokens),
    )


def vector(value: float, dimensions: int = 1024) -> list[float]:
    return [value] * dimensions


def client(sdk: FakeOpenAI) -> OpenAICompatibleEmbeddingClient:
    return OpenAICompatibleEmbeddingClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="text-embedding-v4",
        dimensions=1024,
        client=cast(Any, sdk),
    )


@pytest.mark.asyncio
async def test_batches_more_than_ten_texts_and_preserves_order() -> None:
    sdk = FakeOpenAI(
        [
            embedding_response(
                [vector(float(index)) for index in reversed(range(10))],
                indexes=list(reversed(range(10))),
                input_tokens=20,
            ),
            embedding_response([vector(10.0), vector(11.0)], input_tokens=4),
        ]
    )
    result = await client(sdk).embed([f"fact-{index}" for index in range(12)])

    assert [item[0] for item in result.vectors] == [
        *[float(index) for index in range(10)],
        10.0,
        11.0,
    ]
    assert result.input_tokens == 24
    assert sdk.embeddings.calls == [
        {
            "model": "text-embedding-v4",
            "input": [f"fact-{index}" for index in range(10)],
            "dimensions": 1024,
            "encoding_format": "float",
        },
        {
            "model": "text-embedding-v4",
            "input": ["fact-10", "fact-11"],
            "dimensions": 1024,
            "encoding_format": "float",
        },
    ]


@pytest.mark.asyncio
async def test_empty_input_skips_provider() -> None:
    sdk = FakeOpenAI([])

    result = await client(sdk).embed([])

    assert result.vectors == ()
    assert result.input_tokens == 0
    assert sdk.embeddings.calls == []


@pytest.mark.asyncio
async def test_rejects_wrong_vector_count() -> None:
    sdk = FakeOpenAI([embedding_response([vector(1.0)])])

    with pytest.raises(ValueError, match="count"):
        await client(sdk).embed(["one", "two"])


@pytest.mark.asyncio
async def test_rejects_wrong_dimensions() -> None:
    sdk = FakeOpenAI([embedding_response([vector(1.0, dimensions=3)])])

    with pytest.raises(ValueError, match="dimension"):
        await client(sdk).embed(["one"])


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
async def test_rejects_non_finite_values(invalid: float) -> None:
    invalid_vector = vector(1.0)
    invalid_vector[5] = invalid
    sdk = FakeOpenAI([embedding_response([invalid_vector])])

    with pytest.raises(ValueError, match="finite"):
        await client(sdk).embed(["one"])


@pytest.mark.asyncio
async def test_closes_underlying_client() -> None:
    sdk = FakeOpenAI([])
    embedding_client = client(sdk)

    await embedding_client.close()

    assert sdk.closed is True


@pytest.mark.asyncio
async def test_remote_client_exposes_space_and_uses_same_api_for_queries() -> None:
    sdk = FakeOpenAI([embedding_response([vector(1.0)])])
    embedding_client = client(sdk)

    result = await embedding_client.embed_query(["weather"])

    assert embedding_client.space == EmbeddingSpace(
        provider="openai-compatible",
        model="text-embedding-v4",
        dimensions=1024,
    )
    assert result.vectors[0][0] == 1.0
    assert sdk.embeddings.calls[0]["input"] == ["weather"]


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str], **_: Any) -> dict[str, np.ndarray]:
        self.calls.append(texts)
        return {
            "input_ids": np.array([[1, 2], [3, 0]], dtype=np.int64),
            "attention_mask": np.array([[1, 1], [1, 0]], dtype=np.int64),
        }


class FakeOnnxSession:
    def get_inputs(self) -> list[Any]:
        return [
            SimpleNamespace(name="input_ids"),
            SimpleNamespace(name="attention_mask"),
        ]

    def get_outputs(self) -> list[Any]:
        return [SimpleNamespace(type="tensor(float)")]

    def run(
        self,
        _: object,
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        batch_size = inputs["input_ids"].shape[0]
        values = np.array(
            [
                [[3.0, 4.0], [0.0, 0.0]],
                [[0.0, 5.0], [9.0, 9.0]],
            ],
            dtype=np.float32,
        )
        return [values[:batch_size]]


def test_xlm_tokenizer_json_uses_lightweight_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "XLMRobertaTokenizer", "pad_token": "<pad>"})
    )
    calls: list[Any] = []

    class Backend:
        def token_to_id(self, token: str) -> int | None:
            return 1 if token == "<pad>" else None

        def enable_truncation(self, **kwargs: object) -> None:
            calls.append(("truncation", kwargs))

        def enable_padding(self, **kwargs: object) -> None:
            calls.append(("padding", kwargs))

        def encode_batch(self, texts, *, add_special_tokens):
            calls.append((texts, add_special_tokens))
            return [
                SimpleNamespace(
                    ids=[0, 10, 2], attention_mask=[1, 1, 1], type_ids=[0, 0, 0]
                ),
                SimpleNamespace(
                    ids=[0, 2, 1], attention_mask=[1, 1, 0], type_ids=[0, 0, 0]
                ),
            ]

    backend = Backend()
    monkeypatch.setattr("tokenizers.Tokenizer.from_file", lambda _: backend)
    monkeypatch.setattr(
        client_module, "_configure_xlm_roberta_tokenizer", lambda *_: None
    )

    tokenizer = client_module._load_local_tokenizer(tmp_path, max_tokens=512)
    encoded = tokenizer(
        ["query: weather", "query: "],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )

    np.testing.assert_array_equal(encoded["input_ids"], [[0, 10, 2], [0, 2, 1]])
    np.testing.assert_array_equal(encoded["attention_mask"], [[1, 1, 1], [1, 1, 0]])
    assert calls[0] == ("truncation", {"max_length": 512})
    assert calls[1][0] == "padding"


def _xlm_model_dir(path: Path) -> Path:
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "XLMRobertaTokenizer", "pad_token": "<pad>"})
    )
    (path / "sentencepiece.bpe.model").write_bytes(b"model")
    return path


def test_xlm_sentencepiece_model_maps_fairseq_ids_without_tokenizers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Processor:
        def __init__(self, *, model_file: str) -> None:
            assert model_file == str(tmp_path / "sentencepiece.bpe.model")

        def encode(self, text: str) -> list[int]:
            # SentencePiece id 0 is <unk>; real pieces start after <s>, </s>.
            return {"query: hi": [5, 0, 7], "query: long": [9] * 10, "q": [5]}[text]

    monkeypatch.setitem(
        sys.modules, "sentencepiece", SimpleNamespace(SentencePieceProcessor=Processor)
    )
    monkeypatch.setattr(
        "tokenizers.Tokenizer.from_file",
        lambda _: pytest.fail("the SentencePiece model must be preferred"),
    )

    tokenizer = client_module._load_local_tokenizer(
        _xlm_model_dir(tmp_path), max_tokens=5
    )
    encoded = tokenizer(["query: hi", "query: long", "q"])

    np.testing.assert_array_equal(
        encoded["input_ids"],
        [[0, 6, 3, 8, 2], [0, 10, 10, 10, 2], [0, 6, 2, 1, 1]],
    )
    np.testing.assert_array_equal(
        encoded["attention_mask"], [[1] * 5, [1] * 5, [1, 1, 1, 0, 0]]
    )
    np.testing.assert_array_equal(encoded["token_type_ids"], np.zeros((3, 5)))


def test_xlm_sentencepiece_failure_keeps_tokenizer_json_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(**_: object) -> None:
        raise OSError("corrupt model")

    monkeypatch.setitem(
        sys.modules, "sentencepiece", SimpleNamespace(SentencePieceProcessor=broken)
    )
    backend = SimpleNamespace(
        token_to_id=lambda token: 1 if token == "<pad>" else None,
        enable_truncation=lambda **_: None,
        enable_padding=lambda **_: None,
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.setattr("tokenizers.Tokenizer.from_file", lambda _: backend)
    monkeypatch.setattr(
        client_module, "_configure_xlm_roberta_tokenizer", lambda *_: None
    )

    tokenizer = client_module._load_local_tokenizer(
        _xlm_model_dir(tmp_path), max_tokens=512
    )

    assert isinstance(tokenizer, client_module._TokenizerJsonAdapter)


def test_nonstandard_xlm_tokenizer_falls_back_to_transformers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "XLMRobertaTokenizerFast"})
    )
    fallback = object()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, *, local_files_only: bool) -> object:
            assert path == str(tmp_path)
            assert local_files_only is True
            return fallback

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=AutoTokenizer),
    )
    monkeypatch.setattr(
        "tokenizers.Tokenizer.from_file",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr(
        client_module,
        "_configure_xlm_roberta_tokenizer",
        lambda *_: (_ for _ in ()).throw(ValueError("unsupported charsmap")),
    )

    assert client_module._load_local_tokenizer(tmp_path, max_tokens=512) is fallback


@pytest.mark.asyncio
async def test_local_e5_uses_asymmetric_prefixes_and_normalized_mean_pooling() -> None:
    tokenizer = FakeTokenizer()
    embedding_client = LocalOnnxEmbeddingClient(
        model="intfloat/multilingual-e5-small",
        dimensions=2,
        tokenizer=cast(Any, tokenizer),
        session=cast(Any, FakeOnnxSession()),
    )

    documents = await embedding_client.embed(["天气", "篮球"])
    query = await embedding_client.embed_query(["天气", "篮球"])

    assert embedding_client.space == EmbeddingSpace(
        provider="local",
        model="intfloat/multilingual-e5-small",
        dimensions=2,
    )
    assert tokenizer.calls == [
        ["passage: 天气", "passage: 篮球"],
        ["query: 天气", "query: 篮球"],
    ]
    assert documents.input_tokens == 3
    np.testing.assert_allclose(documents.vectors, ((0.6, 0.8), (0.0, 1.0)))
    np.testing.assert_allclose(query.vectors, documents.vectors)


@pytest.mark.asyncio
async def test_local_embedding_releases_lock_between_background_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_client = LocalOnnxEmbeddingClient(
        model="intfloat/multilingual-e5-small",
        dimensions=2,
        batch_size=1,
        tokenizer=cast(Any, FakeTokenizer()),
        session=cast(Any, FakeOnnxSession()),
    )
    calls: list[tuple[str, str]] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def encode(texts: list[str], prefix: str) -> Any:
        calls.append((prefix, texts[0]))
        if len(calls) == 1:
            first_started.set()
            release_first.wait(timeout=2)
        return client_module._EncodedBatch(vectors=((1.0, 0.0),), input_tokens=1)

    monkeypatch.setattr(embedding_client, "_encode_sync", encode)
    background = asyncio.create_task(embedding_client.embed(["a", "b"]))
    assert await asyncio.to_thread(first_started.wait, 1)
    query = asyncio.create_task(embedding_client.embed_query(["q"]))
    release_first.set()

    await asyncio.gather(background, query)

    assert calls == [
        ("passage: ", "a"),
        ("query: ", "q"),
        ("passage: ", "b"),
    ]


@pytest.mark.asyncio
async def test_local_e5_rejects_model_dimension_mismatch() -> None:
    embedding_client = LocalOnnxEmbeddingClient(
        model="intfloat/multilingual-e5-small",
        dimensions=384,
        tokenizer=cast(Any, FakeTokenizer()),
        session=cast(Any, FakeOnnxSession()),
    )

    with pytest.raises(ValueError, match="model output 2"):
        await embedding_client.initialize()


@pytest.mark.asyncio
async def test_local_e5_rejects_quantized_integer_output_vectors() -> None:
    class IntegerOutputSession(FakeOnnxSession):
        def get_outputs(self) -> list[Any]:
            return [SimpleNamespace(type="tensor(int8)")]

    embedding_client = LocalOnnxEmbeddingClient(
        model="intfloat/multilingual-e5-small",
        dimensions=2,
        tokenizer=cast(Any, FakeTokenizer()),
        session=cast(Any, IntegerOutputSession()),
    )

    with pytest.raises(ValueError, match="FP32"):
        await embedding_client.initialize()


def test_resolve_cached_snapshot_does_not_check_network() -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/cached/model"

    result = _resolve_cached_snapshot(
        snapshot_download,
        model="intfloat/multilingual-e5-small",
        allow_patterns=["onnx/model.onnx"],
    )

    assert result == Path("/cached/model")
    assert calls == [
        {
            "repo_id": "intfloat/multilingual-e5-small",
            "allow_patterns": ["onnx/model.onnx"],
            "local_files_only": True,
        }
    ]


def test_cached_local_model_emits_one_loading_log(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "onnx" / "model.onnx"
    model_path.parent.mkdir()
    model_path.touch()

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_: str(tmp_path),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "onnxruntime.InferenceSession",
        lambda *_args, **_kwargs: FakeOnnxSession(),
    )
    embedding_client = LocalOnnxEmbeddingClient(
        model="intfloat/multilingual-e5-small",
        dimensions=384,
    )

    with caplog.at_level(logging.INFO, logger="fish_memory.embeddings.client"):
        embedding_client._initialize_sync()

    messages = [
        record.getMessage()
        for record in caplog.records
        if "intfloat/multilingual-e5-small" in record.getMessage()
    ]
    assert messages == [
        f"Loading cached local embedding model "
        f"intfloat/multilingual-e5-small from {model_path}"
    ]


def test_download_snapshot_uses_plain_log_progress() -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/downloaded/model"

    download_snapshot = getattr(client_module, "_download_snapshot", None)
    assert download_snapshot is not None
    result = download_snapshot(
        snapshot_download,
        model="intfloat/multilingual-e5-small",
        allow_patterns=["onnx/model.onnx"],
    )

    assert result == Path("/downloaded/model")
    assert (
        getattr(calls[0]["tqdm_class"], "__name__", None)
        == "_EmbeddingDownloadProgress"
    )


def test_embedding_download_progress_logs_intermediate_file_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    progress_class = getattr(client_module, "_EmbeddingDownloadProgress", None)
    assert progress_class is not None

    with caplog.at_level(logging.INFO, logger="fish_memory.embeddings.client"):
        progress = progress_class(total=6, desc="Fetching 6 files")
        for _ in range(6):
            progress.update()
        progress.close()

    messages = [record.getMessage() for record in caplog.records]
    assert "本地 Embedding 模型下载进度: 0.0% (0/6 files)" in messages
    assert "本地 Embedding 模型下载进度: 50.0% (3/6 files)" in messages
    assert "本地 Embedding 模型下载进度: 100.0% (6/6 files)" in messages


def test_default_model_downloads_from_modelscope_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fish_memory.model_catalog import E5_SMALL_REVISION
    from fish_memory.model_sources import ModelDownloadError, ModelStatusReporter

    installs: list[str] = []

    def install(repo: str, revision: str, files: Any, **_kwargs: Any) -> Path:
        installs.append(f"{repo}@{revision}")
        return tmp_path

    monkeypatch.setattr(client_module, "install_hub_snapshot", install)
    reporter = ModelStatusReporter("embeddings", "test")

    def client(**options: Any) -> LocalOnnxEmbeddingClient:
        return LocalOnnxEmbeddingClient(
            model="intfloat/multilingual-e5-small", dimensions=384, **options
        )

    pinned = client(
        onnx_file="onnx/model_qint8_avx512_vnni.onnx", revision=E5_SMALL_REVISION
    )
    assert pinned._download_pinned(reporter) == tmp_path
    assert installs == [f"intfloat/multilingual-e5-small@{E5_SMALL_REVISION}"]
    # Anything not pinned in the catalog goes straight to the Hub endpoints.
    assert client(onnx_file="onnx/model.onnx")._download_pinned(reporter) is None
    assert len(installs) == 1

    def unreachable(*_args: Any, **_kwargs: Any) -> Path:
        raise ModelDownloadError("down")

    monkeypatch.setattr(client_module, "install_hub_snapshot", unreachable)
    assert pinned._download_pinned(reporter) is None
