"""Optional encryption support for snapshot data."""

from __future__ import annotations


class Encryptor:
    """
    Symmetric encryption wrapper for snapshot data using Fernet (AES-128-CBC).

    Requires the ``cryptography`` package::

        pip install pyrecall[privacy]

    A new random key is generated on instantiation.  To reuse a key across
    sessions, pass it explicitly::

        enc = Encryptor(key=my_key)
        enc2 = Encryptor(key=enc.key)   # same key → can decrypt enc's output

    The ``key`` attribute is a URL-safe base64-encoded 32-byte value.
    """

    def __init__(self, key: bytes | None = None) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise ImportError(
                "Snapshot encryption requires the 'privacy' extra. "
                "Install it with: pip install pyrecall[privacy]"
            ) from exc

        if key is None:
            self.key: bytes = Fernet.generate_key()
        else:
            if not key:
                raise ValueError("Encryption key must not be empty.")
            self.key = key
        self._fernet = Fernet(self.key)

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, value: str) -> str:
        try:
            return self._fernet.decrypt(value.encode()).decode()
        except Exception as exc:
            raise ValueError(
                "Failed to decrypt snapshot data. "
                "The snapshot may be corrupted or encrypted with a different key."
            ) from exc
