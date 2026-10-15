import subprocess
from pathlib import Path

import pytest

import fish_memory.kws.keywords as keywords_module
from fish_memory.config import Settings
from fish_memory.kws.keywords import KeywordFileManager, KeywordPreparationError
from fish_memory.kws.models import KwsModelConfig


def test_kws_defaults_are_safe_without_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FISH_MEMORY_LLM_PROVIDER", "openai")
    monkeypatch.setenv("FISH_MEMORY_LLM_API_KEY", "test-key")
    monkeypatch.setenv("FISH_MEMORY_LLM_MODEL", "qwen-plus")
    monkeypatch.setenv("FISH_MEMORY_LLM_BASE_URL", "https://example.invalid/v1")
    settings = Settings(
        _env_file=None,
    )

    assert settings.kws_model_dir is None
    assert settings.wake_word == "小鱼"
    assert settings.kws_chunk_size == 8
    assert settings.kws_num_threads == 1

    config = KwsModelConfig.from_settings(settings)
    assert config.model_dir == (
        Path.cwd()
        / ".cache"
        / "fish-memory"
        / "kws"
        / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
    )


def test_kws_model_config_uses_official_chunk_8_files(tmp_path: Path) -> None:
    config = KwsModelConfig(
        model_dir=tmp_path,
        chunk_size=8,
        num_threads=1,
        score=1.0,
        threshold=0.25,
    )

    paths = config.model_files()

    assert paths["encoder"].name == ("encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
    assert paths["decoder"].name == "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"
    assert paths["joiner"].name == "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
    assert paths["tokens"].name == "tokens.txt"
    assert paths["lexicon"].name == "en.phone"


def test_keyword_manager_writes_raw_file_and_atomically_promotes_tokens(
    tmp_path: Path,
) -> None:
    (tmp_path / "tokens.txt").write_text("tokens\n", encoding="utf-8")
    (tmp_path / "en.phone").write_text("lexicon\n", encoding="utf-8")
    calls: list[tuple[list[str], Path, str]] = []

    def fake_converter(arguments: list[str], output: Path) -> None:
        calls.append(
            (arguments, output, Path(arguments[-2]).read_text(encoding="utf-8"))
        )
        output.write_text("小 鱼\n", encoding="utf-8")

    manager = KeywordFileManager(
        KwsModelConfig(
            model_dir=tmp_path,
            chunk_size=8,
            num_threads=1,
            score=1.0,
            threshold=0.25,
        ),
        run_text2token=fake_converter,
    )

    path = manager.prepare("小鱼")

    assert path.read_text(encoding="utf-8") == "小 鱼\n"
    assert calls
    assert calls[0][2] == "小鱼 @小鱼\n"
    assert not list(manager.work_dir.glob("*.tmp"))


def test_keyword_manager_uses_in_process_sherpa_converter_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "tokens.txt").write_text("tokens\n", encoding="utf-8")
    (tmp_path / "en.phone").write_text("lexicon\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_text2token(texts: list[str], **kwargs: object) -> list[list[str]]:
        calls.append((texts, kwargs))
        return [["ch", "ou", "b", "ao"]]

    monkeypatch.setattr("sherpa_onnx.text2token", fake_text2token)
    manager = KeywordFileManager(
        KwsModelConfig(
            model_dir=tmp_path,
            chunk_size=8,
            num_threads=1,
            score=1.0,
            threshold=0.25,
        )
    )

    path = manager.prepare("臭包")

    assert path.read_text(encoding="utf-8") == "ch ou b ao @臭包\n"
    assert calls == [
        (
            ["臭包"],
            {
                "tokens": str(tmp_path / "tokens.txt"),
                "tokens_type": "phone+ppinyin",
                "bpe_model": None,
                "lexicon": str(tmp_path / "en.phone"),
            },
        )
    ]


def test_failed_keyword_conversion_keeps_previous_file(tmp_path: Path) -> None:
    (tmp_path / "tokens.txt").write_text("tokens\n", encoding="utf-8")
    (tmp_path / "en.phone").write_text("lexicon\n", encoding="utf-8")

    def failing_converter(arguments: list[str], output: Path) -> None:
        raise RuntimeError("text2token failed")

    manager = KeywordFileManager(
        KwsModelConfig(
            model_dir=tmp_path,
            chunk_size=8,
            num_threads=1,
            score=1.0,
            threshold=0.25,
        ),
        run_text2token=failing_converter,
    )
    previous = manager.keywords_path
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_text("old keywords\n", encoding="utf-8")

    with pytest.raises(KeywordPreparationError, match="text2token failed"):
        manager.prepare("新词")

    assert previous.read_text(encoding="utf-8") == "old keywords\n"
    assert not list(manager.work_dir.glob("*.tmp"))


def test_text2token_error_includes_cli_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def failing_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["sherpa-onnx-cli", "text2token"],
            stderr="Please run pip install pypinyin before you continue",
        )

    monkeypatch.setattr(keywords_module.subprocess, "run", failing_run)

    with pytest.raises(RuntimeError, match="pypinyin"):
        keywords_module._run_text2token(
            ["sherpa-onnx-cli", "text2token"],
            tmp_path / "keywords.txt",
        )


def test_keyword_manager_rejects_multiline_wake_word(tmp_path: Path) -> None:
    (tmp_path / "tokens.txt").write_text("tokens\n", encoding="utf-8")
    (tmp_path / "en.phone").write_text("lexicon\n", encoding="utf-8")
    manager = KeywordFileManager(
        KwsModelConfig(
            model_dir=tmp_path,
            chunk_size=8,
            num_threads=1,
            score=1.0,
            threshold=0.25,
        )
    )

    with pytest.raises(KeywordPreparationError, match="single line"):
        manager.prepare("小鱼\n系统")


def test_keyword_manager_writes_one_line_per_wake_word(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "tokens.txt").write_text("tokens\n", encoding="utf-8")
    (tmp_path / "en.phone").write_text("lexicon\n", encoding="utf-8")
    tokens = {"阿澄": ["a", "ch", "eng"], "小澄": ["x", "iao", "ch", "eng"]}

    def fake_text2token(texts: list[str], **_kwargs: object) -> list[list[str]]:
        return [tokens[text] for text in texts]

    monkeypatch.setattr("sherpa_onnx.text2token", fake_text2token)
    manager = KeywordFileManager(
        KwsModelConfig(
            model_dir=tmp_path,
            chunk_size=8,
            num_threads=1,
            score=1.0,
            threshold=0.25,
        )
    )

    path = manager.prepare(("阿澄", " 小澄 ", "阿澄"))

    assert path.read_text(encoding="utf-8") == ("a ch eng @阿澄\nx iao ch eng @小澄\n")
    with pytest.raises(KeywordPreparationError):
        manager.prepare(("阿澄", " "))
