from fish_memory.config import Settings
from fish_memory.embeddings.client import (
    EmbeddingClient,
    LocalOnnxEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)

_DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-small"
# This is the upstream repository's only QInt8 ONNX artifact. The graph runs
# through ONNX Runtime's CPUExecutionProvider on both x86_64 and Apple arm64;
# LocalOnnxEmbeddingClient still rejects anything but FP32 vector output.
_DEFAULT_LOCAL_INT8_ONNX_FILE = "onnx/model_qint8_avx512_vnni.onnx"
_DEFAULT_LOCAL_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "local":
        use_default_int8 = settings.embedding_model == _DEFAULT_LOCAL_MODEL
        return LocalOnnxEmbeddingClient(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            model_dir=settings.embedding_local_model_dir,
            onnx_file=(
                _DEFAULT_LOCAL_INT8_ONNX_FILE if use_default_int8 else "onnx/model.onnx"
            ),
            revision=(_DEFAULT_LOCAL_MODEL_REVISION if use_default_int8 else None),
            max_tokens=settings.embedding_local_max_tokens,
            batch_size=settings.embedding_batch_size,
            num_threads=settings.embedding_local_num_threads,
        )

    api_key = settings.embedding_api_key
    base_url = settings.embedding_base_url
    if api_key is None or not base_url:
        raise ValueError(
            "remote embedding configuration requires a dedicated API key and base URL"
        )
    return OpenAICompatibleEmbeddingClient(
        api_key=api_key.get_secret_value(),
        base_url=base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
    )
