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

        self.key: bytes = key or Fernet.generate_key()
        self._fernet = Fernet(self.key)

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, value: str) -> str:
        return self._fernet.decrypt(value.encode()).decode()
