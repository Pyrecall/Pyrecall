### Describe the bug

`ReplayBuffer._save()` writes the buffer file non-atomically. If the process crashes mid-write, the meta line (containing `seen_hashes` and `total_seen` counts) can be corrupted, silently resetting the buffer state on next load.

### Current Behavior

The buffer writes directly to the target file. On crash:
- Partial write leaves truncated/corrupted file
- Next load fails to parse meta line
- Buffer silently resets `seen_hashes=0, total_seen=0`

### Expected Behavior

Use atomic write pattern:
1. Write to temporary file
2. `os.rename(tmp, target)` (atomic on POSIX)
3. On load, handle corrupted files gracefully

### Files to Fix

- `pyrecall/replay.py` - `ReplayBuffer._save()` method

### Acceptance Criteria

- [ ] Buffer writes use atomic rename
- [ ] Corrupted buffer files detected and handled (recover or clear with warning)
- [ ] No silent data loss on crash
- [ ] Test added for crash simulation