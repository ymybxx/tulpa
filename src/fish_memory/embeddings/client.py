from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from huggingface_hub.utils import tqdm as hf_tqdm
from openai import AsyncOpenAI

from fish_memory.model_catalog import E5_SMALL_FILES, E5_SMALL_REPO, E5_SMALL_REVISION
from fish_memory.model_sources import (
    ModelDownloadError,
    ModelStatusReporter,
    hf_endpoints,
    install_hub_snapshot,
)

logger = logging.getLogger(__name__)


class _EmbeddingDownloadProgress(hf_tqdm):
    """Emit stable download milestones instead of terminal-only progress bars."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._last_logged_bucket = -1
        kwargs["disable"] = False
        super().__init__(*args, **kwargs)
        self._log_file_progress()

    def display(self, msg: str | None = None, pos: int | None = None) -> None:
        del msg, pos

    def update(self, n: float | None = 1) -> bool | None:
        result = super().update(n)
        self._log_file_progress()
        return result

    def _log_file_progress(self) -> None:
        total = int(self.total or 0)
        completed = min(int(self.n), total)
        if total < 1 or not str(self.desc).startswith("Fetching "):
            return
        percent = completed * 100 / total
        bucket = min(int(percent // 10), 10)
        if bucket <= self._last_logged_bucket:
            return
        self._last_logged_bucket = bucket
        logger.info(
            "本地 Embedding 模型下载进度: %.1f%% (%d/%d files)",
            percent,
            completed,
            total,
        )


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    provider: str
    model: str
    dimensions: int

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("embedding provider must not be empty")
        if not self.model.strip():
            raise ValueError("embedding model must not be empty")
        if not 1 <= self.dimensions <= 2000:
            raise ValueError("embedding dimensions must be between 1 and 2000")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    input_tokens: int | None


@dataclass(frozen=True, slots=True)
class _EncodedBatch:
    vectors: tuple[tuple[float, ...], ...]
    input_tokens: int


class EmbeddingClient(Protocol):
    @property
    def space(self) -> EmbeddingSpace: ...

    async def initialize(self) -> None: ...

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...

    async def embed_query(self, texts: Sequence[str]) -> EmbeddingResult: ...

    async def close(self) -> None: ...


def _resolve_cached_snapshot(
    snapshot_download: Callable[..., str],
    *,
    model: str,
    allow_patterns: list[str],
    revision: str | None = None,
) -> Path:
    options: dict[str, object] = {
        "repo_id": model,
        "allow_patterns": allow_patterns,
        "local_files_only": True,
    }
    if revision is not None:
        options["revision"] = revision
    return Path(snapshot_download(**options))


def _download_snapshot(
    snapshot_download: Callable[..., str],
    *,
    model: str,
    allow_patterns: list[str],
    revision: str | None = None,
) -> Path:
    options: dict[str, object] = {
        "repo_id": model,
        "allow_patterns": allow_patterns,
        "tqdm_class": _EmbeddingDownloadProgress,
    }
    if revision is not None:
        options["revision"] = revision
    # Mainland mirror first; the Hub cache is shared, whichever endpoint served it.
    failure: Exception | None = None
    for endpoint in hf_endpoints():
        try:
            return Path(snapshot_download(**options, endpoint=endpoint))
        except Exception as error:
            failure = error
            logger.warning("从 %s 下载嵌入模型失败，尝试下一个下载源", endpoint)
    assert failure is not None
    raise failure


class OpenAICompatibleEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        batch_size: int = 10,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        if batch_size < 1:
            raise ValueError("embedding batch size must be positive")
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def space(self) -> EmbeddingSpace:
        return EmbeddingSpace(
            provider="openai-compatible",
            model=self._model,
            dimensions=self._dimensions,
        )

    async def initialize(self) -> None:
        return None

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), input_tokens=0)

        vectors: list[tuple[float, ...]] = []
        total_input_tokens: int | None = 0
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
                encoding_format="float",
            )
            rows = sorted(response.data, key=lambda item: item.index)
            if len(rows) != len(batch):
                raise ValueError("embedding response count does not match input count")
            if [row.index for row in rows] != list(range(len(batch))):
                raise ValueError("embedding response indexes are invalid")
            vectors.extend(
                _validated_vector(
                    cast(Sequence[float], row.embedding),
                    dimensions=self._dimensions,
                )
                for row in rows
            )
            usage = cast(Any, getattr(response, "usage", None))
            input_tokens = getattr(usage, "prompt_tokens", None)
            if input_tokens is None:
                total_input_tokens = None
            elif total_input_tokens is not None:
                total_input_tokens += int(input_tokens)

        return EmbeddingResult(
            vectors=tuple(vectors),
            input_tokens=total_input_tokens,
        )

    async def embed_query(self, texts: Sequence[str]) -> EmbeddingResult:
        return await self.embed(texts)

    async def close(self) -> None:
        await self._client.close()


class LocalOnnxEmbeddingClient:
    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        model_dir: str | None = None,
        onnx_file: str = "onnx/model.onnx",
        revision: str | None = None,
        max_tokens: int = 512,
        batch_size: int = 32,
        num_threads: int = 2,
        tokenizer: Any | None = None,
        session: Any | None = None,
    ) -> None:
        self._space = EmbeddingSpace(
            provider="local",
            model=model,
            dimensions=dimensions,
        )
        if max_tokens < 1:
            raise ValueError("embedding max tokens must be positive")
        if batch_size < 1:
            raise ValueError("embedding batch size must be positive")
        if num_threads < 1:
            raise ValueError("embedding thread count must be positive")
        if (tokenizer is None) != (session is None):
            raise ValueError("tokenizer and session must be provided together")
        self._model_dir = model_dir
        self._onnx_file = onnx_file
        self._revision = revision
        self._max_tokens = max_tokens
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._tokenizer = tokenizer
        self._session = session
        self._lock = asyncio.Lock()

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    async def initialize(self) -> None:
        async with self._lock:
            if self._session is None:
                await asyncio.to_thread(self._initialize_sync)
            await asyncio.to_thread(self._validate_output_type_sync)
            detected = await asyncio.to_thread(self._detect_dimensions_sync)
        if detected != self._space.dimensions:
            raise ValueError(
                f"configured embedding dimension {self._space.dimensions} "
                f"does not match model output {detected}"
            )

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return await self._embed_prefixed(texts, "passage: ")

    async def embed_query(self, texts: Sequence[str]) -> EmbeddingResult:
        return await self._embed_prefixed(texts, "query: ")

    async def close(self) -> None:
        self._session = None
        self._tokenizer = None

    async def _embed_prefixed(
        self,
        texts: Sequence[str],
        prefix: str,
    ) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), input_tokens=0)
        self._require_initialized()

        vectors: list[tuple[float, ...]] = []
        input_tokens = 0
        for start in range(0, len(texts), self._batch_size):
            async with self._lock:
                batch = list(texts[start : start + self._batch_size])
                encoded_batch = await asyncio.to_thread(
                    self._encode_sync,
                    batch,
                    prefix,
                )
                vectors.extend(encoded_batch.vectors)
                input_tokens += encoded_batch.input_tokens
            if start + self._batch_size < len(texts):
                await asyncio.sleep(0)
        return EmbeddingResult(vectors=tuple(vectors), input_tokens=input_tokens)

    def _download_pinned(self, reporter: ModelStatusReporter) -> Path | None:
        """The default model from ModelScope, verified file by file; None otherwise."""
        pinned = {file.name for file in E5_SMALL_FILES}
        if (
            self._space.model != E5_SMALL_REPO
            or self._revision != E5_SMALL_REVISION
            or self._onnx_file not in pinned
        ):
            return None
        try:
            return install_hub_snapshot(
                E5_SMALL_REPO,
                E5_SMALL_REVISION,
                E5_SMALL_FILES,
                progress=reporter.downloading,
            )
        except (ModelDownloadError, OSError) as error:
            logger.warning("从魔搭准备嵌入模型失败，改用 Hugging Face: %s", error)
            return None

    def _initialize_sync(self) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError as exception:
            raise ImportError(
                "local embeddings require onnxruntime and huggingface-hub"
            ) from exception

        if self._model_dir:
            model_root = Path(self._model_dir).expanduser().resolve()
            model_source = "configured"
        else:
            allow_patterns = [
                self._onnx_file,
                "config.json",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ]
            try:
                # snapshot_download checks the network by default even when the cache is
                # complete. Resolve offline first so a network outage cannot block
                # application startup.
                model_root = _resolve_cached_snapshot(
                    snapshot_download,
                    model=self._space.model,
                    allow_patterns=allow_patterns,
                    revision=self._revision,
                )
                model_source = "cached"
            except LocalEntryNotFoundError:
                logger.info(
                    "Local embedding model %s is not cached; downloading it",
                    self._space.model,
                )
                reporter = ModelStatusReporter("embeddings", "记忆")
                reporter.preparing()
                try:
                    model_root = self._download_pinned(reporter) or _download_snapshot(
                        snapshot_download,
                        model=self._space.model,
                        allow_patterns=allow_patterns,
                        revision=self._revision,
                    )
                except Exception as error:
                    reporter.failed(str(error))
                    raise
                reporter.ready()
                model_source = "downloaded"
        model_path = model_root / self._onnx_file
        if not model_path.is_file():
            raise FileNotFoundError(f"embedding ONNX model not found: {model_path}")

        logger.info(
            "Loading %s local embedding model %s from %s",
            model_source,
            self._space.model,
            model_path,
        )
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self._num_threads
        self._tokenizer = _load_local_tokenizer(
            model_root,
            max_tokens=self._max_tokens,
        )
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    def _validate_output_type_sync(self) -> None:
        session = self._session
        if session is None:
            raise RuntimeError("embedding client is not initialized")
        outputs = session.get_outputs()
        output_type = outputs[0].type if outputs else None
        if output_type != "tensor(float)":
            raise ValueError(
                "local embedding ONNX must produce FP32 output vectors; "
                f"model output type is {output_type!r}"
            )

    def _detect_dimensions_sync(self) -> int:
        batch = self._encode_sync(
            ["dimension probe"],
            "passage: ",
            validate_dimensions=False,
        )
        return len(batch.vectors[0])

    def _encode_sync(
        self,
        texts: list[str],
        prefix: str,
        *,
        validate_dimensions: bool = True,
    ) -> _EncodedBatch:
        import numpy as np

        self._require_initialized()
        tokenizer = cast(Any, self._tokenizer)
        session = cast(Any, self._session)
        prefixed = [f"{prefix}{text}" for text in texts]
        encoded = tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=self._max_tokens,
            return_tensors="np",
        )
        input_names = {item.name for item in session.get_inputs()}
        inputs = {name: value for name, value in encoded.items() if name in input_names}
        if "token_type_ids" in input_names and "token_type_ids" not in inputs:
            inputs["token_type_ids"] = np.zeros_like(encoded["input_ids"])

        token_embeddings = np.asarray(session.run(None, inputs)[0])
        if getattr(token_embeddings, "ndim", 0) == 2:
            pooled = token_embeddings
        else:
            encoded_attention_mask = encoded.get("attention_mask")
            attention_mask = (
                np.asarray(encoded_attention_mask)
                if encoded_attention_mask is not None
                else np.ones(token_embeddings.shape[:2], dtype=np.float32)
            )
            mask = attention_mask[..., None].astype(np.float32)
            pooled = (token_embeddings * mask).sum(axis=1) / np.clip(
                mask.sum(axis=1),
                a_min=1e-9,
                a_max=None,
            )

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = pooled / norms
        rows = normalized.astype(float).tolist()
        if validate_dimensions:
            vectors = tuple(
                _validated_vector(row, dimensions=self._space.dimensions)
                for row in rows
            )
        else:
            vectors = tuple(tuple(float(value) for value in row) for row in rows)
        encoded_attention_mask = encoded.get("attention_mask")
        input_tokens = (
            int(np.asarray(encoded_attention_mask).sum())
            if encoded_attention_mask is not None
            else int(encoded["input_ids"].size)
        )
        return _EncodedBatch(vectors=vectors, input_tokens=input_tokens)

    def _require_initialized(self) -> None:
        if self._session is None or self._tokenizer is None:
            raise RuntimeError("embedding client is not initialized")


class _SentencePieceXlmRobertaAdapter:
    """XLM-R token ids straight from its SentencePiece model.

    The tokenizers runtime expands the 250k-piece Unigram vocabulary into a
    trie of per-node hash maps (about 400 MB resident); SentencePiece keeps
    the same vocabulary in a compact double-array trie (about 60 MB).
    """

    # fairseq ids that XLM-R places before the SentencePiece vocabulary.
    _BOS, _PAD, _EOS, _UNK = 0, 1, 2, 3
    _OFFSET = 1

    def __init__(self, model_path: Path, *, max_tokens: int) -> None:
        import sentencepiece

        self._processor = sentencepiece.SentencePieceProcessor(
            model_file=str(model_path)
        )
        self._max_body = max(0, max_tokens - 2)

    def _ids(self, text: str) -> list[int]:
        body = [
            piece + self._OFFSET if piece else self._UNK
            for piece in self._processor.encode(text)
        ]
        return [self._BOS, *body[: self._max_body], self._EOS]

    def __call__(
        self,
        texts: list[str],
        **_: object,
    ) -> dict[str, Any]:
        import numpy as np

        rows = [self._ids(text) for text in texts]
        width = max((len(row) for row in rows), default=0)
        input_ids = np.full((len(rows), width), self._PAD, dtype=np.int64)
        attention_mask = np.zeros((len(rows), width), dtype=np.int64)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = row
            attention_mask[index, : len(row)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        }


class _TokenizerJsonAdapter:
    """The small tokenizers runtime with the AutoTokenizer array contract."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_tokens: int,
        pad_id: int,
        pad_token: str,
        padding_side: str,
    ) -> None:
        tokenizer.enable_truncation(max_length=max_tokens)
        tokenizer.enable_padding(
            direction=padding_side,
            pad_id=pad_id,
            pad_type_id=0,
            pad_token=pad_token,
        )
        self._tokenizer = tokenizer

    def __call__(
        self,
        texts: list[str],
        **_: object,
    ) -> dict[str, Any]:
        import numpy as np

        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=True)
        return {
            "input_ids": np.asarray(
                [encoding.ids for encoding in encodings], dtype=np.int64
            ),
            "attention_mask": np.asarray(
                [encoding.attention_mask for encoding in encodings], dtype=np.int64
            ),
            "token_type_ids": np.asarray(
                [encoding.type_ids for encoding in encodings], dtype=np.int64
            ),
        }


def _load_local_tokenizer(model_root: Path, *, max_tokens: int) -> Any:
    tokenizer_path = model_root / "tokenizer.json"
    tokenizer_config = _read_tokenizer_config(model_root)
    tokenizer_class = tokenizer_config.get("tokenizer_class")
    is_xlm_roberta = tokenizer_class in {
        "XLMRobertaTokenizer",
        "XLMRobertaTokenizerFast",
    }
    sentencepiece_path = model_root / "sentencepiece.bpe.model"
    if is_xlm_roberta and sentencepiece_path.is_file():
        try:
            return _SentencePieceXlmRobertaAdapter(
                sentencepiece_path, max_tokens=max_tokens
            )
        except (ImportError, OSError, RuntimeError):
            logger.debug(
                "SentencePiece XLM-R tokenizer unavailable for %s",
                model_root,
                exc_info=True,
            )
    if tokenizer_path.is_file() and is_xlm_roberta:
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            _configure_xlm_roberta_tokenizer(tokenizer, tokenizer_path)
            pad_token = _special_token_text(tokenizer_config.get("pad_token"))
            for candidate in (pad_token, "<pad>", "[PAD]"):
                if candidate is None:
                    continue
                pad_id = tokenizer.token_to_id(candidate)
                if pad_id is not None:
                    padding_side = tokenizer_config.get("padding_side", "right")
                    if padding_side not in {"left", "right"}:
                        padding_side = "right"
                    return _TokenizerJsonAdapter(
                        tokenizer,
                        max_tokens=max_tokens,
                        pad_id=pad_id,
                        pad_token=candidate,
                        padding_side=padding_side,
                    )
        except (ImportError, OSError, ValueError):
            # tokenizer.json is not a uniform contract across custom XLM-R
            # exports. Keep those models compatible with their authoritative
            # Transformers tokenizer; the pinned desktop E5 artifact follows
            # the lightweight path above.
            logger.debug(
                "Lightweight XLM-R tokenizer adapter unavailable for %s",
                model_root,
                exc_info=True,
            )

    # Keep custom local model directories that only ship a slow tokenizer
    # compatible. The bundled multilingual-e5 path above never imports the
    # much larger transformers package.
    try:
        from transformers import AutoTokenizer
    except ImportError as exception:
        raise ImportError(
            "local embedding model needs tokenizer.json or transformers"
        ) from exception
    return AutoTokenizer.from_pretrained(str(model_root), local_files_only=True)


def _configure_xlm_roberta_tokenizer(tokenizer: Any, path: Path) -> None:
    """Apply the same encode-time components as XLMRobertaTokenizerFast."""
    try:
        from tokenizers import normalizers, pre_tokenizers

        serialized = path.read_text(encoding="utf-8")
        marker = '"precompiled_charsmap"'
        start = serialized.index(marker) + len(marker)
        start = serialized.index(":", start) + 1
        while start < len(serialized) and serialized[start].isspace():
            start += 1
        charsmap, _ = json.JSONDecoder().raw_decode(serialized, start)
        if not isinstance(charsmap, str):
            raise ValueError("precompiled_charsmap must be a string")
        tokenizer.normalizer = normalizers.Precompiled(base64.b64decode(charsmap))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(
                    replacement="▁",
                    prepend_scheme="always",
                    split=True,
                ),
            ]
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid XLM-R tokenizer.json") from error


def _read_tokenizer_config(model_root: Path) -> dict[str, Any]:
    path = model_root / "tokenizer_config.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _special_token_text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content:
            return content
    return None


def _validated_vector(
    vector: Sequence[float],
    *,
    dimensions: int,
) -> tuple[float, ...]:
    if len(vector) != dimensions:
        raise ValueError(
            f"embedding dimension {len(vector)} does not match expected {dimensions}"
        )
    normalized = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("embedding values must be finite")
    return normalized
