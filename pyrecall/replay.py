def _save(self) -> None:
    meta = {
        "total_seen": self._total_seen,
        "max_size": self._max_size,
        "seen_hashes": list(self._seen_hashes),
    }

    lines = [json.dumps(meta)]
    lines += [
        json.dumps(
            {
                "text": e["text"],
                "category": e.get("category"),
            }
        )
        for e in self._buffer
    ]

    content = "\n".join(lines) + "\n"

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=self._path.parent,
        prefix=".buffer_tmp_",
    )

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(content)

            # Ensure everything is written to disk before replacing
            fh.flush()
            os.fsync(fh.fileno())

        # Atomically replace the old file
        os.replace(tmp_path, self._path)

        # Sync the directory entry (extra crash safety on POSIX)
        if os.name != "nt":
            dir_fd = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise
