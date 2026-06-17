"""Optional encryption support for snapshot data."""

from __future__ import annotations

import base64
import hashlib
import os


class Encryptor:
    """
    Symmetric encryption wrapper for snapshot data using Fernet (AES-128-CBC).

    Requires the ``cryptography`` package::

        pip install pyrecall[privacy]

    Keys are derived from a user-supplied passphrase via PBKDF2-HMAC-SHA256
    so no key is ever written to disk.  Use :meth:`from_passphrase` to
    construct an instance — direct instantiation with a raw key is still
    supported for internal use.
    """

    def __init__(self, key: bytes) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise ImportError(
                "Snapshot encryption requires the 'privacy' extra. "
                "Install it with: pip install pyrecall[privacy]"
            ) from exc

        if not key:
            raise ValueError("Encryption key must not be empty.")
        self.key = key
        self._fernet = Fernet(self.key)

    @classmethod
    def from_passphrase(cls, passphrase: str, salt: bytes | None = None) -> "Encryptor":
        """Derive a Fernet key from *passphrase* and return a ready Encryptor.

        If *salt* is None a fresh 16-byte salt is generated; retrieve it via
        ``encryptor.salt`` and store it alongside the ciphertext so decryption
        can reproduce the same key.
        """
        try:
            from cryptography.fernet import Fernet  # noqa: F401 — validates extra installed
        except ImportError as exc:
            raise ImportError(
                "Snapshot encryption requires the 'privacy' extra. "
                "Install it with: pip install pyrecall[privacy]"
            ) from exc

        if not passphrase:
            raise ValueError("passphrase must not be empty.")
        salt = salt if salt is not None else os.urandom(16)
        raw_key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, iterations=600_000)
        fernet_key = base64.urlsafe_b64encode(raw_key)
        enc = cls(fernet_key)
        enc.salt = salt
        return enc

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, value: str) -> str:
        try:
            return self._fernet.decrypt(value.encode()).decode()
        except Exception as exc:
            raise ValueError(
                "Failed to decrypt snapshot data. "
                "Wrong passphrase, or the snapshot may be corrupted."
            ) from exc
