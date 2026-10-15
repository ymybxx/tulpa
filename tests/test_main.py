import runpy
from typing import Any

import pytest
from pydantic import ValidationError

import fish_memory.main as main_module


@pytest.fixture(autouse=True)
def valid_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_MEMORY_LLM_PROVIDER", "openai")
    monkeypatch.setenv("FISH_MEMORY_LLM_API_KEY", "test-key")
    monkeypatch.setenv("FISH_MEMORY_LLM_MODEL", "qwen-plus")
    monkeypatch.setenv("FISH_MEMORY_LLM_BASE_URL", "https://example.invalid/v1")


def test_server_host_defaults_to_loopback() -> None:
    assert main_module.read_server_host(None) == "127.0.0.1"


def test_server_host_accepts_environment_value() -> None:
    assert main_module.read_server_host("0.0.0.0") == "0.0.0.0"


def test_rejects_blank_server_host() -> None:
    with pytest.raises(ValueError, match="FISH_MEMORY_HOST"):
        main_module.read_server_host("   ")


def test_server_port_defaults_to_8000() -> None:
    assert main_module.read_server_port(None) == 8000


def test_server_port_accepts_environment_value() -> None:
    assert main_module.read_server_port("8123") == 8123


@pytest.mark.parametrize("value", ["nope", "0", "65536"])
def test_rejects_invalid_server_port(value: str) -> None:
    with pytest.raises(ValueError, match="FISH_MEMORY_PORT"):
        main_module.read_server_port(value)


def test_main_passes_environment_port_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_run(app: str, **kwargs: Any) -> None:
        calls.append((app, kwargs))

    monkeypatch.setenv("FISH_MEMORY_PORT", "8123")
    monkeypatch.setenv("FISH_MEMORY_HOST", "0.0.0.0")
    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

    main_module.main()

    assert calls == [
        (
            "fish_memory.app:create_app",
            {
                "factory": True,
                "host": "0.0.0.0",
                "port": 8123,
                "workers": 1,
            },
        )
    ]


def test_main_configures_application_logging_before_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_logging = getattr(main_module, "configure_application_logging", None)
    assert configure_logging is not None

    events: list[str] = []
    logging_config: dict[str, object] = {}
    application_logger = main_module.logging.getLogger("fish_memory")
    previous_level = application_logger.level
    monkeypatch.setattr(
        main_module.logging,
        "basicConfig",
        lambda **kwargs: (logging_config.update(kwargs), events.append("logging")),
    )
    monkeypatch.setattr(
        main_module.uvicorn,
        "run",
        lambda *_args, **_kwargs: events.append("uvicorn"),
    )

    try:
        main_module.main()
    finally:
        configured_level = application_logger.level
        application_logger.setLevel(previous_level)

    assert events == ["logging", "uvicorn"]
    assert logging_config["level"] == main_module.logging.WARNING
    assert configured_level == main_module.logging.INFO


def test_main_reports_invalid_configuration_before_starting_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uvicorn_called = False

    def invalid_settings() -> None:
        raise ValidationError.from_exception_data(
            "Settings",
            [
                {
                    "type": "string_too_short",
                    "loc": ("llm_api_key",),
                    "input": "",
                    "ctx": {"min_length": 1},
                }
            ],
        )

    def fake_run(*args: Any, **kwargs: Any) -> None:
        nonlocal uvicorn_called
        uvicorn_called = True

    monkeypatch.setattr(main_module, "Settings", invalid_settings, raising=False)
    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    message = str(exc_info.value)
    assert "fish-memory cannot start" in message
    assert "FISH_MEMORY_LLM_API_KEY" in message
    assert "FISH_MEMORY_LLM_API_KEY=..." in message
    assert uvicorn_called is False


def test_module_execution_starts_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_run(app: str, **kwargs: Any) -> None:
        calls.append((app, kwargs))

    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

    runpy.run_path(main_module.__file__, run_name="__main__")

    assert len(calls) == 1


def test_main_dispatches_auth_command_without_starting_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(main_module.sys, "argv", ["fish-memory", "auth", "show"])
    monkeypatch.setattr(
        main_module,
        "run_auth_command_from_environment",
        lambda arguments: calls.append(arguments) or 0,
        raising=False,
    )
    monkeypatch.setattr(
        main_module.uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("uvicorn must not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 0
    assert calls == [["show"]]
