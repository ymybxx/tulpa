import json

from fish_memory.config import Settings

LLM_ENV = {
    "FISH_MEMORY_LLM_PROVIDER": "openai",
    "FISH_MEMORY_LLM_API_KEY": "test-key",
    "FISH_MEMORY_LLM_MODEL": "qwen-plus",
    "FISH_MEMORY_LLM_BASE_URL": "https://example.invalid/v1",
}


def test_existing_env_file_still_configures_all_text_modules(tmp_path, monkeypatch):
    for name in (*LLM_ENV, "FISH_MEMORY_DESKTOP_MODEL_ROUTES"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text("\n".join(f"{key}={value}" for key, value in LLM_ENV.items()))
    settings = Settings(_env_file=path)
    routes = settings.resolved_llm_routes()
    assert routes.chat == routes.reasoning
    assert routes.chat.model == "qwen-plus"
    assert routes.chat.api_key.get_secret_value() == "test-key"


def test_desktop_routes_are_explicit_and_keys_remain_secret_in_diagnostics(monkeypatch):
    for key, value in LLM_ENV.items():
        monkeypatch.setenv(key, value)
    payload = {
        "chat": {
            "baseUrl": "https://chat.invalid/v1",
            "model": "character",
            "apiKey": "chat-private",
        },
        "reasoning": {
            "baseUrl": "https://reason.invalid/v1",
            "model": "flash",
            "apiKey": "reason-private",
        },
    }
    monkeypatch.setenv("FISH_MEMORY_DESKTOP_MODEL_ROUTES", json.dumps(payload))
    settings = Settings(_env_file=None)
    routes = settings.resolved_llm_routes()
    assert routes.chat.client_arguments()["model"] == "character"
    assert routes.reasoning.client_arguments()["api_key"] == "reason-private"
    for key in ("chat-private", "reason-private"):
        assert key not in settings.model_dump_json()
        assert key not in repr(routes)
