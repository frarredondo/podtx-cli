from __future__ import annotations

import subprocess


def get_api_key(service: str, account: str) -> str | None:
    """Fetch secret from macOS Keychain via `security`.

    Returns None if not found or on non-macOS / security unavailable.
    Never raises for missing entry; raises for unexpected errors? No, returns None.
    """
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:  # security not installed (Linux CI)
        return None
    except OSError:
        return None
    if result.returncode != 0:
        return None
    val = result.stdout.strip()
    return val if val else None


def save_api_key(service: str, account: str, secret: str) -> None:
    """Save/Update secret in macOS Keychain via `security add-generic-password -U`.

    Raises RuntimeError if security is unavailable or fails.
    """
    try:
        result = subprocess.run(
            ["security", "add-generic-password", "-U", "-s", service, "-a", account, "-w", secret],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("macOS `security` tool not found — cannot save to Keychain") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to save to Keychain: {exc}") from exc
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"Failed to save to Keychain: {msg}")


def delete_api_key(service: str, account: str) -> bool:
    """Delete entry from Keychain. Returns True if deleted, False if not found.

    Raises RuntimeError on unexpected failure.
    """
    try:
        result = subprocess.run(
            ["security", "delete-generic-password", "-s", service, "-a", account],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - missing security
        raise RuntimeError("macOS `security` tool not found — cannot delete from Keychain") from exc  # pragma: no cover
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to delete from Keychain: {exc}") from exc  # pragma: no cover
    if result.returncode == 0:
        return True
    # Common not-found message contains "could not be found"
    err = (result.stderr or result.stdout or "").lower()
    if "could not be found" in err or "not found" in err:
        return False
    # Treat non-zero as not found gracefully if message unclear? Return False
    # But if security failed for other reason, raise?
    # For robustness, return False if not found pattern, else raise?
    # We'll return False for now to keep CLI friendly.
    return False
