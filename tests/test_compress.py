"""Python Tests for pyrecall/compress.py."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from pyrecall.compress import compress_adapter_dir, decompressed_adapter


def _write_fake_adapter(directory: Path) -> None:
    """Write a minimal fake adapter dir with a weight file and a config file."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_model.bin").write_bytes(b"fake weights data")
    (directory / "adapter_config.json").write_text('{"r": 16}')


class TestCompressAdapterDir:
    def test_gzip_compresses_weight_file(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_fake_adapter(adapter_dir)

        compress_adapter_dir(adapter_dir, "gzip")

        assert (adapter_dir / "adapter_model.bin.gzip").exists()
        assert not (adapter_dir / "adapter_model.bin").exists()

    def test_non_weight_files_untouched(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_fake_adapter(adapter_dir)

        compress_adapter_dir(adapter_dir, "gzip")

        assert (adapter_dir / "adapter_config.json").exists()

    def test_none_codec_is_noop(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_fake_adapter(adapter_dir)

        compress_adapter_dir(adapter_dir, "none")

        assert (adapter_dir / "adapter_model.bin").exists()

    def test_unknown_codec_raises(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_fake_adapter(adapter_dir)

        with pytest.raises(ValueError, match="Unknown compression codec"):
            compress_adapter_dir(adapter_dir, "bz2")


class TestDecompressedAdapter:
    def test_round_trip_preserves_bytes(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_fake_adapter(adapter_dir)
        original = (adapter_dir / "adapter_model.bin").read_bytes()

        compress_adapter_dir(adapter_dir, "gzip")

        with decompressed_adapter(adapter_dir, "gzip") as decomp_dir:
            result = (decomp_dir / "adapter_model.bin").read_bytes()

        assert result == original

    def test_non_weight_files_copied_as_is(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_fake_adapter(adapter_dir)
        compress_adapter_dir(adapter_dir, "gzip")

        with decompressed_adapter(adapter_dir, "gzip") as decomp_dir:
            assert (decomp_dir / "adapter_config.json").exists()
            assert (decomp_dir / "adapter_config.json").read_text() == '{"r": 16}'

    def test_temp_dir_cleaned_up_after_exit(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_fake_adapter(adapter_dir)
        compress_adapter_dir(adapter_dir, "gzip")

        with decompressed_adapter(adapter_dir, "gzip") as decomp_dir:
            decomp_path = decomp_dir

        assert not decomp_path.exists()

    def test_none_codec_yields_original_dir(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_fake_adapter(adapter_dir)

        with decompressed_adapter(adapter_dir, "none") as decomp_dir:
            assert decomp_dir == adapter_dir

    def test_single_suffix_weight_file_no_index_error(self, tmp_path: Path) -> None:
        """Regression: files with only one suffix (e.g. weights.gzip) must not
        raise IndexError from src.suffixes[-2]."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        # Manually create a .bin.gzip file whose stem has no extra dot
        weight_data = b"minimal weights"
        compressed = gzip.compress(weight_data)
        (adapter_dir / "weights.bin.gzip").write_bytes(compressed)
        (adapter_dir / "adapter_config.json").write_text("{}")

        with decompressed_adapter(adapter_dir, "gzip") as decomp_dir:
            assert (decomp_dir / "weights.bin").read_bytes() == weight_data
