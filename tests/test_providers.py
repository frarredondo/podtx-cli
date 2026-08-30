from __future__ import annotations

import httpx

from podtx.providers import (
    Provider,
    ProviderError,
    available_providers,
    build_provider,
    get_spec,
    normalize_backend,
    resolve_api_key,
)
from podtx.providers.anthropic import ANTHROPIC_VERSION, AnthropicProvider
from podtx.providers.base import ProviderError as ProviderErrorBase
from podtx.providers.openai import OpenAICompatibleProvider
from podtx.providers import registry as registry_mod
from podtx.providers.registry import (
    OPENAI_DEFAULT_BASE_URL,
    ProviderSpec,
)


def test_provider_error_base() -> None:
    assert issubclass(ProviderError, Exception)
    assert issubclass(ProviderErrorBase, Exception)


def test_provider_protocol_runtime_check() -> None:
    class Fake:
        name = "fake"

        def complete(self, messages, *, timeout=120.0, temperature=0.3):
            return "ok"

    assert isinstance(Fake(), Provider)


def test_normalize_backend() -> None:
    assert normalize_backend("local") == "lmstudio"
    assert normalize_backend("LOCAL") == "lmstudio"
    assert normalize_backend("openai-compatible") == "openai"
    assert normalize_backend("openrouter") == "openrouter"
    assert normalize_backend("  openai  ") == "openai"


def test_available_providers() -> None:
    names = available_providers()
    assert names == sorted(names)
    assert {
        "openrouter",
        "opencode",
        "openai",
        "anthropic",
        "lmstudio",
    }.issubset(names)


def test_get_spec_known() -> None:
    spec = get_spec("openrouter")
    assert spec.name == "openrouter"
    assert spec.default_model is not None
    assert spec.env_var == "OPENROUTER_API_KEY"
    assert spec.keychain_service == "podtx-openrouter"
    assert spec.requires_api_key


def test_get_spec_unknown() -> None:
    try:
        get_spec("bogus")
    except ValueError as exc:
        assert "Unknown provider" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_api_key_precedence(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with monkeypatch.context() as m:
        m.setattr(registry_mod, "_keychain_get", lambda svc, acct: "keychain-key")
        assert resolve_api_key("openrouter", api_key="flag") == "flag"
        assert resolve_api_key(
            "openrouter", api_key="flag", settings_api_key="settings"
        ) == "flag"


def test_resolve_api_key_settings_over_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    assert (
        resolve_api_key("openrouter", settings_api_key="settings-key")
        == "settings-key"
    )


def test_resolve_api_key_env_over_keychain(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    with monkeypatch.context() as m:
        m.setattr(registry_mod, "_keychain_get", lambda svc, acct: "keychain-key")
        assert resolve_api_key("openrouter") == "env-key"


def test_resolve_api_key_keychain_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with monkeypatch.context() as m:
        m.setattr(registry_mod, "_keychain_get", lambda svc, acct: "keychain-key")
        assert resolve_api_key(
            "openrouter", service="podtx-openrouter", account="api-key"
        ) == "keychain-key"


def test_resolve_api_key_none_for_keyless() -> None:
    assert resolve_api_key("lmstudio") is None


def test_keychain_get_direct() -> None:
    with __import__("unittest.mock").mock.patch("podtx.keychain.get_api_key", return_value="kc"):
        assert registry_mod._keychain_get("podtx-openrouter", "api-key") == "kc"


def test_build_provider_openrouter() -> None:
    provider = build_provider("openrouter", api_key="k")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "openrouter"
    assert provider.base_url.endswith("/v1")
    assert provider.model == get_spec("openrouter").default_model
    assert provider.api_key == "k"


def test_build_provider_openai_defaults() -> None:
    provider = build_provider("openai", model="gpt-4.1-mini", api_key="k")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == OPENAI_DEFAULT_BASE_URL
    assert provider.model == "gpt-4.1-mini"


def test_build_provider_anthropic() -> None:
    provider = build_provider("anthropic", model="claude-sonnet-4", api_key="k")
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-sonnet-4"


def test_build_provider_lmstudio_no_key() -> None:
    provider = build_provider("lmstudio", model="llama")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.api_key is None
    assert provider.base_url == "http://localhost:1234/v1"


def test_build_provider_requires_model() -> None:
    try:
        build_provider("openai", api_key="k")
    except ProviderError as exc:
        assert "requires --model" in str(exc)
    else:
        raise AssertionError("expected ProviderError")
    try:
        build_provider("anthropic", api_key="k")
    except ProviderError as exc:
        assert "requires --model" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_build_provider_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with monkeypatch.context() as m:
        m.setattr(registry_mod, "_keychain_get", lambda svc, acct: None)
        try:
            build_provider("openrouter")
        except ProviderError as exc:
            assert "requires an API key" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_build_provider_unknown() -> None:
    try:
        build_provider("bogus")
    except ProviderError as exc:
        assert "Unknown provider" in str(exc)
        assert "openrouter" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_build_provider_missing_base_url(monkeypatch) -> None:
    monkeypatch.setitem(
        registry_mod._REGISTRY,
        "nobase",
        ProviderSpec(
            name="nobase",
            provider_class=registry_mod.OpenAICompatibleProvider,
            base_url=None,
            default_model="m",
            env_var=None,
            keychain_service=None,
            requires_api_key=False,
        ),
    )
    try:
        try:
            build_provider("nobase")
        except ProviderError as exc:
            assert "requires --base-url" in str(exc)
        else:
            raise AssertionError("expected ProviderError")
    finally:
        registry_mod._REGISTRY.pop("nobase")


def test_build_provider_keychain_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with monkeypatch.context() as m:
        m.setattr(registry_mod, "_keychain_get", lambda svc, acct: "kc-key")
        provider = build_provider(
            "openrouter", service="podtx-openrouter", account="api-key"
        )
        assert provider.api_key == "kc-key"


def test_openai_client_success() -> None:
    import podtx.providers.openai as mod

    resp = httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = OpenAICompatibleProvider("openrouter", "https://x.example/v1/", "m", "k")
        out = provider.complete(
            [{"role": "user", "content": "hello"}], timeout=5.0, temperature=0.7
        )
        assert out == "hi"
        post = m.return_value.__enter__.return_value.post
        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        assert url == "https://x.example/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer k"
        assert kwargs["json"]["model"] == "m"
        assert kwargs["json"]["temperature"] == 0.7
        assert kwargs["json"]["response_format"] == {"type": "json_object"}


def test_openai_client_no_key_header() -> None:
    import podtx.providers.openai as mod

    resp = httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = OpenAICompatibleProvider("lmstudio", "http://localhost:1234/v1", "m")
        provider.complete([{"role": "user", "content": "hello"}], timeout=5.0)
        kwargs = m.return_value.__enter__.return_value.post.call_args.kwargs
        assert "Authorization" not in kwargs["headers"]


def test_openai_client_http_error() -> None:
    import podtx.providers.openai as mod

    resp = httpx.Response(401, text="bad key")
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = OpenAICompatibleProvider("openrouter", "https://x.example/v1", "m", "k")
        try:
            provider.complete([{"role": "user", "content": "hello"}])
        except ProviderError as exc:
            assert "401" in str(exc)
            assert "bad key" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_openai_client_request_error() -> None:
    import podtx.providers.openai as mod

    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError("boom")
        provider = OpenAICompatibleProvider("openrouter", "https://x.example/v1", "m", "k")
        try:
            provider.complete([{"role": "user", "content": "hello"}])
        except ProviderError as exc:
            assert "request failed" in str(exc)
            assert "boom" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_openai_client_invalid_json() -> None:
    import podtx.providers.openai as mod

    resp = httpx.Response(200, text="not json")
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = OpenAICompatibleProvider("openrouter", "https://x.example/v1", "m", "k")
        try:
            provider.complete([{"role": "user", "content": "hello"}])
        except ProviderError as exc:
            assert "invalid JSON" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_openai_client_missing_content() -> None:
    import podtx.providers.openai as mod

    resp = httpx.Response(200, json={"choices": [{"message": {}}]})
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = OpenAICompatibleProvider("openrouter", "https://x.example/v1", "m", "k")
        try:
            provider.complete([{"role": "user", "content": "hello"}])
        except ProviderError as exc:
            assert "missing choices/message/content" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_openai_client_non_string_content() -> None:
    import podtx.providers.openai as mod

    resp = httpx.Response(200, json={"choices": [{"message": {"content": {"a": 1}}}]})
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = OpenAICompatibleProvider("openrouter", "https://x.example/v1", "m", "k")
        out = provider.complete([{"role": "user", "content": "hello"}])
        assert out == '{"a": 1}'


def test_anthropic_client_success() -> None:
    import podtx.providers.anthropic as mod

    resp = httpx.Response(
        200,
        json={
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"},
            ]
        },
    )
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = AnthropicProvider("anthropic", "https://api.anthropic.com/v1", "claude", "k")
        out = provider.complete(
            [
                {"role": "system", "content": "sys-a"},
                {"role": "system", "content": "sys-b"},
                {"role": "user", "content": "hello"},
            ],
            timeout=9.0,
            temperature=0.4,
        )
        assert out == "line one\nline two"
        post = m.return_value.__enter__.return_value.post
        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        assert url.endswith("/messages")
        assert kwargs["headers"]["x-api-key"] == "k"
        assert kwargs["headers"]["anthropic-version"] == ANTHROPIC_VERSION
        assert kwargs["json"]["system"] == "sys-a\n\nsys-b"
        assert kwargs["json"]["messages"] == [{"role": "user", "content": "hello"}]
        assert kwargs["json"]["max_tokens"] == 8192
        assert kwargs["json"]["temperature"] == 0.4


def test_anthropic_client_no_api_key(tmp_path) -> None:
    provider = AnthropicProvider("anthropic", "https://api.anthropic.com/v1", "claude")
    try:
        provider.complete([{"role": "user", "content": "hello"}])
    except ProviderError as exc:
        assert "requires an API key" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_anthropic_client_no_user_message(tmp_path) -> None:
    provider = AnthropicProvider("anthropic", "https://api.anthropic.com/v1", "claude", "k")
    try:
        provider.complete([{"role": "system", "content": "sys"}])
    except ProviderError as exc:
        assert "requires at least one user message" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_anthropic_client_http_error() -> None:
    import podtx.providers.anthropic as mod

    resp = httpx.Response(429, text="rate limited")
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = AnthropicProvider("anthropic", "https://api.anthropic.com/v1", "claude", "k")
        try:
            provider.complete([{"role": "user", "content": "hello"}])
        except ProviderError as exc:
            assert "429" in str(exc)
            assert "rate limited" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_anthropic_client_request_error() -> None:
    import podtx.providers.anthropic as mod

    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError("boom")
        provider = AnthropicProvider("anthropic", "https://api.anthropic.com/v1", "claude", "k")
        try:
            provider.complete([{"role": "user", "content": "hello"}])
        except ProviderError as exc:
            assert "request failed" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_anthropic_client_invalid_json() -> None:
    import podtx.providers.anthropic as mod

    resp = httpx.Response(200, text="not json")
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = AnthropicProvider("anthropic", "https://api.anthropic.com/v1", "claude", "k")
        try:
            provider.complete([{"role": "user", "content": "hello"}])
        except ProviderError as exc:
            assert "invalid JSON" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_anthropic_client_missing_content() -> None:
    import podtx.providers.anthropic as mod

    resp = httpx.Response(200, json={"foo": "bar"})
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = AnthropicProvider("anthropic", "https://api.anthropic.com/v1", "claude", "k")
        try:
            provider.complete([{"role": "user", "content": "hello"}])
        except ProviderError as exc:
            assert "missing content" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_anthropic_client_empty_content() -> None:
    import podtx.providers.anthropic as mod

    resp = httpx.Response(200, json={"content": []})
    with __import__("unittest.mock").mock.patch.object(mod.httpx, "Client") as m:
        m.return_value.__enter__.return_value.post.return_value = resp
        provider = AnthropicProvider("anthropic", "https://api.anthropic.com/v1", "claude", "k")
        try:
            provider.complete([{"role": "user", "content": "hello"}])
        except ProviderError as exc:
            assert "empty text content" in str(exc)
        else:
            raise AssertionError("expected ProviderError")