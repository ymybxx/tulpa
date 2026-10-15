import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from fish_memory.kws.models import KwsModelConfig


class KeywordPreparationError(RuntimeError):
    """The configured wake word could not be converted to sherpa tokens."""


Text2TokenRunner = Callable[[list[str], Path], None]


def _run_text2token(arguments: list[str], output: Path) -> None:
    try:
        completed = subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"text2token failed{suffix}") from error
    if not output.exists() and completed.stdout.strip():
        output.write_text(completed.stdout, encoding="utf-8")


def _run_text2token_in_process(
    phrases: Sequence[str],
    output: Path,
    config: KwsModelConfig,
    paths: dict[str, Path],
) -> None:
    from sherpa_onnx import text2token

    # The default output_ids=False returns strings; the dependency annotates
    # both that mode and the integer-ID mode with a single union return type.
    encoded = cast(
        list[list[str]],
        text2token(
            list(phrases),
            tokens=str(paths["tokens"]),
            tokens_type=config.tokens_type,
            bpe_model=None,
            lexicon=str(paths["lexicon"]) if paths["lexicon"].is_file() else None,
        ),
    )
    if len(encoded) != len(phrases) or not all(encoded):
        raise RuntimeError("text2token produced no tokens")
    output.write_text(
        "".join(
            " ".join([*tokens, f"@{phrase}"]) + "\n"
            for tokens, phrase in zip(encoded, phrases, strict=True)
        ),
        encoding="utf-8",
    )


def wake_word_phrases(wake_words: str | Sequence[str]) -> tuple[str, ...]:
    values = (wake_words,) if isinstance(wake_words, str) else tuple(wake_words)
    phrases = tuple(dict.fromkeys(value.strip() for value in values))
    if not phrases or any(not phrase for phrase in phrases):
        raise KeywordPreparationError("wake word must not be empty")
    if any(char in phrase for phrase in phrases for char in "\r\n"):
        raise KeywordPreparationError("wake word must be a single line")
    return phrases


class KeywordFileManager:
    def __init__(
        self,
        config: KwsModelConfig,
        *,
        executable: str = "sherpa-onnx-cli",
        run_text2token: Text2TokenRunner | None = None,
    ) -> None:
        self.config = config
        self.executable = executable
        self.run_text2token = run_text2token
        self.work_dir = config.model_dir / ".fish-memory-kws"
        self.keywords_path = self.work_dir / "keywords.txt"

    def prepare(self, wake_words: str | Sequence[str]) -> Path:
        """Write one keyword line per wake word; any of them wakes the pet."""
        phrases = wake_word_phrases(wake_words)

        paths = self.config.model_files()
        if not paths["tokens"].is_file():
            raise KeywordPreparationError(f"missing KWS file: {paths['tokens']}")
        if (
            self.config.tokens_type == "phone+ppinyin"
            and not paths["lexicon"].is_file()
        ):
            raise KeywordPreparationError(f"missing KWS file: {paths['lexicon']}")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        raw_path: Path | None = None
        output_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.work_dir,
                prefix="keywords_raw-",
                suffix=".tmp",
                delete=False,
            ) as raw_file:
                raw_path = Path(raw_file.name)
                raw_file.write("".join(f"{phrase} @{phrase}\n" for phrase in phrases))
            output_path = self.work_dir / f"keywords-{os.getpid()}.tmp"
            arguments = [
                self.executable,
                "text2token",
                "--tokens",
                str(paths["tokens"]),
                "--tokens-type",
                self.config.tokens_type,
            ]
            if paths["lexicon"].is_file():
                arguments.extend(("--lexicon", str(paths["lexicon"])))
            arguments.extend((str(raw_path), str(output_path)))
            if self.run_text2token is None:
                _run_text2token_in_process(
                    phrases,
                    output_path,
                    self.config,
                    paths,
                )
            else:
                self.run_text2token(arguments, output_path)
            if (
                not output_path.is_file()
                or not output_path.read_text(encoding="utf-8").strip()
            ):
                raise KeywordPreparationError("text2token produced an empty file")
            os.replace(output_path, self.keywords_path)
            return self.keywords_path
        except KeywordPreparationError:
            raise
        except Exception as error:
            raise KeywordPreparationError(str(error)) from error
        finally:
            for temporary_path in (raw_path, output_path):
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
