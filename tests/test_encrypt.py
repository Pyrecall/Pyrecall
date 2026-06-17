"""Tests for pyrecall.encrypt.Encryptor."""

from __future__ import annotations

import pytest

pytest.importorskip("cryptography", reason="cryptography not installed — skip encrypt tests")

from pyrecall.encrypt import Encryptor


def test_generates_key_on_init():
    enc = Encryptor()
    assert enc.key
    assert isinstance(enc.key, bytes)


def test_two_instances_have_different_keys():
    assert Encryptor().key != Encryptor().key


def test_round_trip():
    enc = Encryptor()
    plaintext = "hello world"
    assert enc.decrypt(enc.encrypt(plaintext)) == plaintext


def test_round_trip_unicode():
    enc = Encryptor()
    plaintext = "résumé 日本語 🔑"
    assert enc.decrypt(enc.encrypt(plaintext)) == plaintext


def test_reuse_key_can_decrypt():
    enc1 = Encryptor()
    ciphertext = enc1.encrypt("secret")
    enc2 = Encryptor(key=enc1.key)
    assert enc2.decrypt(ciphertext) == "secret"


def test_wrong_key_raises_value_error():
    enc1 = Encryptor()
    enc2 = Encryptor()
    ciphertext = enc1.encrypt("secret")
    with pytest.raises(ValueError, match="decrypt"):
        enc2.decrypt(ciphertext)


def test_garbage_ciphertext_raises_value_error():
    enc = Encryptor()
    with pytest.raises(ValueError, match="decrypt"):
        enc.decrypt("not-valid-ciphertext")


def test_empty_key_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        Encryptor(key=b"")


def test_encrypted_value_differs_each_call():
    enc = Encryptor()
    # Fernet uses a random IV so the same plaintext encrypts differently each time.
    assert enc.encrypt("same") != enc.encrypt("same")
