"""Adapter weight compression helpers for snapshot storage."""

from __future__ import annotations

import gzip
import os
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
    weight_files = [f for f in adapter_dir.iterdir() if f.suffix in _WEIGHT_SUFFIXES]
    compressed: list[tuple[Path, Path]] = []
    try:
        for src in weight_files:
            dst = src.with_suffix(src.suffix + f".{codec}")
            _compress_file(src, dst, codec)
            compressed.append((src, dst))
    except Exception:
        for _, dst in compressed:
            dst.unlink(missing_ok=True)
        raise
    for src, _ in compressed:
        src.unlink()


# shutil.copyfileobj uses this as a buffer-size hint; actual I/O may vary by
# platform and codec. Peak memory per file is O(_CHUNK), not O(file size).
_CHUNK = 4 * 1024 * 1024  # 4 MiB


def _compress_file(src: Path, dst: Path, codec: str) -> None:
    """Write compressed *src* to *dst* atomically via a same-dir temp file.

    On failure the partial temp file is deleted and *dst* is never written,
    so a crash mid-compression never leaves a half-written output file.
    *dst* must not already exist.
    """
    tmp_fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=".compress_tmp_")
    # Close the fd immediately so codec libraries can open the same path.
    # On Windows an open fd blocks any other open() on the same file.
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        if codec == "gzip":
            with src.open("rb") as fsrc, gzip.open(tmp_path, "wb", compresslevel=6) as fdst:
                shutil.copyfileobj(fsrc, fdst, length=_CHUNK)
        elif codec == "zstd":
            try:
                import zstandard as zstd
            except ImportError as exc:
                raise ImportError(
                    "zstd compression requires the 'zstandard' package. "
                    "Install it with: pip install zstandard"
                ) from exc
            cctx = zstd.ZstdCompressor(level=3)
            with src.open("rb") as fsrc, tmp_path.open("wb") as fdst:
                cctx.copy_stream(fsrc, fdst, read_size=_CHUNK, write_size=_CHUNK)
        elif codec == "lz4":
            try:
                import lz4.frame as lz4
            except ImportError as exc:
                raise ImportError(
                    "lz4 compression requires the 'lz4' package. Install it with: pip install lz4"
                ) from exc
            with src.open("rb") as fsrc, lz4.open(tmp_path, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, length=_CHUNK)
        else:
            raise ValueError(f"Unsupported codec: {codec}")
        os.replace(tmp_name, dst)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _decompress_file(src: Path, dst: Path, codec: str) -> None:
    """Write decompressed *src* to *dst*.

    *dst* must not already exist — callers are responsible for ensuring
    the destination path is fresh (e.g. writing into a new temp directory).
    """
    if codec == "gzip":
        with gzip.open(src, "rb") as fsrc, dst.open("wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=_CHUNK)
    elif codec == "zstd":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "zstd decompression requires the 'zstandard' package. "
                "Install it with: pip install zstandard"
            ) from exc
        dctx = zstd.ZstdDecompressor()
        with src.open("rb") as fsrc, dst.open("wb") as fdst:
            dctx.copy_stream(fsrc, fdst, read_size=_CHUNK, write_size=_CHUNK)
    elif codec == "lz4":
        try:
            import lz4.frame as lz4
        except ImportError as exc:
            raise ImportError(
                "lz4 decompression requires the 'lz4' package. Install it with: pip install lz4"
            ) from exc
        with lz4.open(src, "rb") as fsrc, dst.open("wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=_CHUNK)
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
            if src.is_dir():
                shutil.copytree(src, tmp / src.name)
                continue
            original_name = src.name[: -len(ext)]
            if src.name.endswith(ext) and Path(original_name).suffix in _WEIGHT_SUFFIXES:
                # e.g. adapter_model.bin.gzip → adapter_model.bin
                _decompress_file(src, tmp / original_name, codec)
            else:
                # Copy non-weight files (config, index, etc.) as-is.
                shutil.copy2(src, tmp / src.name)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
