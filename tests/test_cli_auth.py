from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from podtx.cli import app

runner = CliRunner()


def test_auth_set_with_key():
    with patch("podtx.keychain.save_api_key") as mock:
        result = runner.invoke(app, ["auth", "set", "openrouter", "--api-key", "sk-test"])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert "Saved" in result.stdout
        mock.assert_called_once()
        assert mock.call_args[0][0] == "podtx-openrouter"


def test_auth_set_prompt(monkeypatch):
    # monkeypatch prompt via typer.prompt patch? Use patch
    with patch("podtx.keychain.save_api_key") as mock_save, patch("typer.prompt", return_value="prompted-key"):
        result = runner.invoke(app, ["auth", "set", "opencode"])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert mock_save.call_args[0][2] == "prompted-key"


def test_auth_set_empty_prompt():
    with patch("typer.prompt", return_value="   "):
        result = runner.invoke(app, ["auth", "set", "openrouter"])
        assert result.exit_code != 0
        assert "No key" in result.stdout or "No key" in result.stderr


def test_auth_set_unknown_backend():
    result = runner.invoke(app, ["auth", "set", "bogus"])
    assert result.exit_code != 0
    assert "Unknown backend" in result.stdout or "Unknown backend" in result.stderr


def test_auth_set_failure():
    with patch("podtx.keychain.save_api_key", side_effect=RuntimeError("fail saving")):
        result = runner.invoke(app, ["auth", "set", "openrouter", "--api-key", "k"])
        assert result.exit_code != 0
        assert "fail saving" in result.stdout or "fail saving" in result.stderr


def test_auth_get_found():
    with patch("podtx.keychain.get_api_key", return_value="secret123"):
        result = runner.invoke(app, ["auth", "get", "openrouter"])
        assert result.exit_code == 0
        assert "Found" in result.stdout


def test_auth_get_not_found():
    with patch("podtx.keychain.get_api_key", return_value=None):
        result = runner.invoke(app, ["auth", "get", "openrouter"])
        assert result.exit_code != 0
        assert "No key" in result.stdout


def test_auth_get_unknown_backend():
    result = runner.invoke(app, ["auth", "get", "bogus"])
    assert result.exit_code != 0


def test_auth_delete_success():
    with patch("podtx.keychain.delete_api_key", return_value=True):
        result = runner.invoke(app, ["auth", "delete", "openrouter"])
        assert result.exit_code == 0
        assert "Deleted" in result.stdout


def test_auth_delete_not_found():
    with patch("podtx.keychain.delete_api_key", return_value=False):
        result = runner.invoke(app, ["auth", "delete", "openrouter"])
        assert result.exit_code != 0
        assert "No key" in result.stdout


def test_auth_delete_unknown_backend():
    result = runner.invoke(app, ["auth", "delete", "bogus"])
    assert result.exit_code != 0


def test_auth_delete_failure():
    with patch("podtx.keychain.delete_api_key", side_effect=RuntimeError("fail")):
        result = runner.invoke(app, ["auth", "delete", "openrouter"])
        assert result.exit_code != 0
        assert "fail" in result.stdout or "fail" in result.stderr


def test_auth_set_custom_service_account():
    with patch("podtx.keychain.save_api_key") as mock:
        result = runner.invoke(app, ["auth", "set", "openrouter", "--api-key", "k", "--service", "svc", "--account", "acct"])
        assert result.exit_code == 0
        assert mock.call_args[0][:2] == ("svc", "acct")
