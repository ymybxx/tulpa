from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from fish_memory.config import Settings

KWS_MODEL_NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
KWS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{KWS_MODEL_NAME}.tar.bz2"
)


@dataclass(frozen=True, slots=True)
class KwsModelConfig:
    model_dir: Path
    chunk_size: int = 8
    num_threads: int = 1
    score: float = 1.0
    threshold: float = 0.25
    sample_rate: int = 16_000
    feature_dim: int = 80
    tokens_type: str = "phone+ppinyin"

    @classmethod
    def from_settings(cls, settings: Settings) -> "KwsModelConfig":
        model_dir = getattr(settings, "kws_model_dir", None)
        if model_dir is None or not model_dir.strip():
            model_dir = str(
                Path.cwd() / ".cache" / "fish-memory" / "kws" / KWS_MODEL_NAME
            )
        return cls(
            model_dir=Path(model_dir).expanduser(),
            chunk_size=getattr(settings, "kws_chunk_size", 8),
            num_threads=getattr(settings, "kws_num_threads", 1),
            score=getattr(settings, "kws_score", 1.0),
            threshold=getattr(settings, "kws_threshold", 0.25),
        )

    @property
    def model_name(self) -> str:
        return KWS_MODEL_NAME

    @property
    def model_url(self) -> str:
        return KWS_MODEL_URL

    def model_files(self) -> dict[str, Path]:
        if self.chunk_size not in (8, 16):
            raise ValueError("chunk_size must be 8 or 16")
        stem = f"epoch-13-avg-2-chunk-{self.chunk_size}-left-64"
        return {
            "encoder": self.model_dir / f"encoder-{stem}.int8.onnx",
            "decoder": self.model_dir / f"decoder-{stem}.onnx",
            "joiner": self.model_dir / f"joiner-{stem}.int8.onnx",
            "tokens": self.model_dir / "tokens.txt",
            "lexicon": self.model_dir / "en.phone",
        }

    def is_ready(self) -> bool:
        return all(
            path.is_file() and path.stat().st_size > 0
            for path in self.model_files().values()
        )


@dataclass(frozen=True, slots=True)
class Detection:
    keyword: str
    audio_offset_ms: int
    processing_ms: float


class KwsStartRequest(BaseModel):
    type: str = Field(pattern="^start$")
    wake_word: str = Field(alias="wakeWord", min_length=1, max_length=80)
    sample_rate: int = Field(alias="sampleRate")
    channels: int
    format: str
