from __future__ import annotations

from unittest.mock import MagicMock, patch

from podtx.keychain import delete_api_key, get_api_key, save_api_key


def test_get_api_key_success() -> None:
    mock = MagicMock(returncode=0, stdout=" secret123\n", stderr="")
    with patch("podtx.keychain.subprocess.run", return_value=mock) as run:
        assert get_api_key("svc", "acct") == "secret123"
        run.assert_called_once()
        assert run.call_args[0][0][:3] == ["security", "find-generic-password", "-s"]


def test_get_api_key_not_found() -> None:
    mock = MagicMock(returncode=44, stdout="", stderr="could not be found")
    with patch("podtx.keychain.subprocess.run", return_value=mock):
        assert get_api_key("svc", "acct") is None


def test_get_api_key_empty_stdout() -> None:
    mock = MagicMock(returncode=0, stdout="   \n", stderr="")
    with patch("podtx.keychain.subprocess.run", return_value=mock):
        assert get_api_key("svc", "acct") is None


def test_get_api_key_security_missing() -> None:
    with patch("podtx.keychain.subprocess.run", side_effect=FileNotFoundError):
        assert get_api_key("svc", "acct") is None


def test_get_api_key_oserror() -> None:
    with patch("podtx.keychain.subprocess.run", side_effect=OSError("boom")):
        assert get_api_key("svc", "acct") is None


def test_save_api_key_success() -> None:
    mock = MagicMock(returncode=0, stdout="", stderr="")
    with patch("podtx.keychain.subprocess.run", return_value=mock) as run:
        save_api_key("svc", "acct", "secret")
        assert run.call_args[0][0][:3] == ["security", "add-generic-password", "-U"]


def test_save_api_key_failure() -> None:
    mock = MagicMock(returncode=1, stdout="", stderr="error saving")
    with patch("podtx.keychain.subprocess.run", return_value=mock):
        try:
            save_api_key("svc", "acct", "secret")
            assert False
        except RuntimeError as exc:
            assert "Failed to save" in str(exc)


def test_save_api_key_missing_security() -> None:
    with patch("podtx.keychain.subprocess.run", side_effect=FileNotFoundError):
        try:
            save_api_key("svc", "acct", "secret")
            assert False
        except RuntimeError as exc:
            assert "security" in str(exc).lower()


def test_save_api_key_oserror() -> None:
    with patch("podtx.keychain.subprocess.run", side_effect=OSError("boom")):
        try:
            save_api_key("svc", "acct", "secret")
            assert False
        except RuntimeError as exc:
            assert "Failed to save" in str(exc)


def test_delete_api_key_found() -> None:
    mock = MagicMock(returncode=0, stdout="", stderr="")
    with patch("podtx.keychain.subprocess.run", return_value=mock):
        assert delete_api_key("svc", "acct") is True


def test_delete_api_key_not_found() -> None:
    mock = MagicMock(returncode=44, stdout="", stderr="could not be found")
    with patch("podtx.keychain.subprocess.run", return_value=mock):
        assert delete_api_key("svc", "acct") is False


def test_delete_api_key_not_found_variant() -> None:
    mock = MagicMock(returncode=44, stdout="not found", stderr="")
    with patch("podtx.keychain.subprocess.run", return_value=mock):
        assert delete_api_key("svc", "acct") is False


def test_delete_api_key_other_error_returns_false() -> None:
    mock = MagicMock(returncode=1, stdout="", stderr="some other error")
    with patch("podtx.keychain.subprocess.run", return_value=mock):
        assert delete_api_key("svc", "acct") is False
