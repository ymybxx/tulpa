import os

import pytest
from pydantic import SecretStr

import fish_memory.agent.credentials as credentials_module
from fish_memory.agent.credentials import (
    AgentCredentialLookupError,
    AgentCredentialResolver,
)
from fish_memory.runtime_platform import private_file_permissions
from tests.private_file_helpers import allow_public_read, assert_private_file


def test_resolves_the_requested_secret_without_exposing_it_in_repr() -> None:
    resolver = AgentCredentialResolver({"hermes:local": SecretStr("private-key")})

    assert resolver.resolve("hermes:local") == "private-key"
    assert "private-key" not in repr(resolver)


def test_missing_credential_reference_is_a_secret_safe_lookup_error() -> None:
    resolver = AgentCredentialResolver({"hermes:local": SecretStr("private-key")})

    with pytest.raises(AgentCredentialLookupError) as exc_info:
        resolver.resolve("missing:credential")

    assert exc_info.value.credential_ref == "missing:credential"
    assert str(exc_info.value) == "missing:credential"
    assert "private-key" not in str(exc_info.value)


def test_persists_generated_credentials_in_a_private_secret_file(
    tmp_path,
    monkeypatch,
) -> None:
    protected = []

    def protect_empty(path, descriptor):
        assert path.read_bytes() == b""
        private_file_permissions(path, descriptor)
        protected.append(path)

    monkeypatch.setattr(credentials_module, "private_file_permissions", protect_empty)
    credentials_file = tmp_path / "agent-credentials.json"
    resolver = AgentCredentialResolver({}, credentials_file=credentials_file)

    resolver.store("fish:override", " private-override-key ")

    assert resolver.resolve("fish:override") == "private-override-key"
    assert_private_file(credentials_file)
    assert protected and all(path != credentials_file for path in protected)
    assert "private-override-key" not in repr(resolver)
    reloaded = AgentCredentialResolver({}, credentials_file=credentials_file)
    assert reloaded.resolve("fish:override") == "private-override-key"

    resolver.remove("fish:override")
    with pytest.raises(AgentCredentialLookupError):
        resolver.resolve("fish:override")


def test_clearing_global_keys_keeps_fish_overrides_but_revokes_every_global_ref(
    tmp_path,
) -> None:
    credentials_file = tmp_path / "agent-credentials.json"
    with credentials_file.open("w", encoding="utf-8") as stream:
        private_file_permissions(credentials_file, stream.fileno())
        stream.write(
            '{"version":1,"credentials":'
            '{"global:old-url":"stale-global-key","fish:override":"fish-key"}}'
        )

    resolver = AgentCredentialResolver({}, credentials_file=credentials_file)

    assert resolver.resolve("fish:override") == "fish-key"
    assert resolver.resolve("global:old-url") == "stale-global-key"
    resolver.store_global("global:new-url", "new-global-key")
    assert resolver.resolve("global:new-url") == "new-global-key"

    resolver.clear_global_credentials()

    assert resolver.resolve("fish:override") == "fish-key"
    with pytest.raises(AgentCredentialLookupError):
        resolver.resolve("global:old-url")
    with pytest.raises(AgentCredentialLookupError):
        resolver.resolve("global:new-url")
    with pytest.raises(ValueError, match="Fish override"):
        resolver.store("global:new-url", "new-global-key")


@pytest.mark.asyncio
async def test_transient_probe_key_does_not_replace_the_live_fish_key(
    tmp_path,
) -> None:
    resolver = AgentCredentialResolver(
        {},
        credentials_file=tmp_path / "agent-credentials.json",
    )
    resolver.store("fish:stable", "working-key")

    async with resolver.update_lock("fish:stable"):
        resolver.stage("probe:candidate", "candidate-key")
        assert resolver.resolve("fish:stable") == "working-key"
        assert resolver.resolve("probe:candidate") == "candidate-key"
        resolver.unstage("probe:candidate")

    assert resolver.resolve("fish:stable") == "working-key"
    with pytest.raises(AgentCredentialLookupError):
        resolver.resolve("probe:candidate")


def test_rejects_publicly_readable_credentials_before_reading_contents(tmp_path):
    path = tmp_path / "agent-credentials.json"
    resolver = AgentCredentialResolver({}, credentials_file=path)
    resolver.store("fish:override", "private-key")
    path.write_text("not JSON", encoding="utf-8")
    allow_public_read(path)

    with pytest.raises(PermissionError, match="accessible by other users"):
        AgentCredentialResolver({}, credentials_file=path)


def test_permission_failure_preserves_prior_key_and_removes_empty_temporary(
    tmp_path, monkeypatch
):
    path = tmp_path / "agent-credentials.json"
    resolver = AgentCredentialResolver({}, credentials_file=path)
    resolver.store("fish:override", "original-key")
    original = path.read_bytes()
    descriptors = []

    def fail_protection(temporary, descriptor):
        assert temporary.read_bytes() == b""
        descriptors.append(descriptor)
        raise PermissionError("cannot protect file")

    monkeypatch.setattr(credentials_module, "private_file_permissions", fail_protection)
    with pytest.raises(PermissionError, match="cannot protect file"):
        resolver.store("fish:override", "replacement-key")

    assert path.read_bytes() == original
    assert resolver.resolve("fish:override") == "original-key"
    assert list(tmp_path.iterdir()) == [path]
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
