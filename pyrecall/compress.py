"""Adapter weight compression helpers for snapshot storage."""

from __future__ import annotations

import gzip
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

# File extensions that contain adapter weights (everything else is kept as-is).
_WEIGHT_SUFFIXES: frozenset[str] = frozenset({".bin", ".safetensors", ".pt", ".pth"})

SUPPORTED_CODECS: frozenset[str] = frozenset({"none", "gzip", "zstd", "lz4"})


def compress_adapter_dir(adapter_dir: Path, codec: str) -> None:
    """Compress all weight files in *adapter_dir* in-place using *codec*.

    Non-weight files (config JSON, index files, etc.) are left untouched.
    Already-compressed files are skipped silently.
    """
    if codec == "none":
        return
    if codec not in SUPPORTED_CODECS:
        raise ValueError(
            f"Unknown compression codec '{codec}'. Supported: {sorted(SUPPORTED_CODECS)}"
        )
    for src in list(adapter_dir.iterdir()):
        if src.suffix not in _WEIGHT_SUFFIXES:
            continue
        dst = src.with_suffix(src.suffix + f".{codec}")
        _compress_file(src, dst, codec)
        src.unlink()


def _compress_file(src: Path, dst: Path, codec: str) -> None:
    data = src.read_bytes()
    if codec == "gzip":
        with gzip.open(dst, "wb", compresslevel=6) as fh:
            fh.write(data)
    elif codec == "zstd":
        import zstandard as zstd  # optional dep

        cctx = zstd.ZstdCompressor(level=3)
        dst.write_bytes(cctx.compress(data))
    elif codec == "lz4":
        import lz4.frame as lz4  # optional dep

        dst.write_bytes(lz4.compress(data))
    else:
        raise ValueError(f"Unsupported codec: {codec}")


def _decompress_file(src: Path, dst: Path, codec: str) -> None:
    data = src.read_bytes()
    if codec == "gzip":
        dst.write_bytes(gzip.decompress(data))
    elif codec == "zstd":
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        dst.write_bytes(dctx.decompress(data))
    elif codec == "lz4":
        import lz4.frame as lz4

        dst.write_bytes(lz4.decompress(data))
    else:
        raise ValueError(f"Unsupported codec: {codec}")


@contextmanager
def decompressed_adapter(adapter_dir: Path, codec: str):
    """Context manager that yields a path to a decompressed copy of *adapter_dir*.

    If *codec* is ``"none"``, yields *adapter_dir* directly (no temp dir created).
    The temp dir is always cleaned up on exit.
    """
    if codec == "none":
        yield adapter_dir
        return

    tmp = Path(tempfile.mkdtemp(prefix="pyrecall_decomp_"))
    try:
        ext = f".{codec}"
        for src in adapter_dir.iterdir():
            if src.name.endswith(ext) and src.suffixes[-2] in (s for s in _WEIGHT_SUFFIXES):
                # e.g. adapter_model.bin.gzip → adapter_model.bin
                original_name = src.name[: -len(ext)]
                _decompress_file(src, tmp / original_name, codec)
            else:
                # Copy non-weight files (config, index, etc.) as-is.
                shutil.copy2(src, tmp / src.name)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
