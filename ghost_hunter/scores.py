"""Read and update the local JSON history of completed human games."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


class InvalidScoreHistory(ValueError):
    """Keep the original bytes and any usable rows when reading fails."""

    def __init__(self, path: Path, original: bytes, usable_rows: list[dict]):
        super().__init__(path.name)
        self.path = path
        self.original = original
        self.usable_rows = usable_rows

    def __str__(self) -> str:
        return f"{self.path.name} contains unreadable score data."


def _usable(row: object) -> bool:
    if not isinstance(row, dict) or type(row.get("captured")) is not bool:
        return False
    if not row["captured"]:
        return True  # Earlier loss records may omit movement and turn counts.
    return all(type(row.get(key)) is int and row[key] >= 0 for key in ("turns", "distance"))


def read_scores(path: Path) -> list[dict]:
    try:
        original = path.read_bytes()
    except FileNotFoundError:
        return []
    try:
        rows = json.loads(original.decode("utf-8-sig"))
    except (ValueError, RecursionError) as error:
        raise InvalidScoreHistory(path, original, []) from error
    if not isinstance(rows, list):
        raise InvalidScoreHistory(path, original, [])
    usable = [row for row in rows if _usable(row)]
    if len(usable) != len(rows):
        raise InvalidScoreHistory(path, original, usable)
    return usable


@contextmanager
def _score_lock(path: Path):
    """Coordinate app instances on the same local score file."""
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield  # Closing the descriptor releases the lock, including on errors.


def append_score(path: Path, score: dict) -> Path | None:
    """Append one game. Return the backup path if the old file needed recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _score_lock(path.with_suffix(".lock")):
        backup = None
        try:
            rows = read_scores(path)
        except InvalidScoreHistory as error:
            with NamedTemporaryFile(dir=path.parent, prefix=path.stem + ".",
                                    suffix=".bak", delete=False) as saved:
                backup = Path(saved.name)
                saved.write(error.original)
                saved.flush()
                os.fsync(saved.fileno())
            rows = error.usable_rows
        payload = json.dumps([*rows, score], indent=2) + "\n"
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                    prefix=path.stem + ".", suffix=".tmp", delete=False) as new:
                temporary = Path(new.name)
                new.write(payload)
                new.flush()
                os.fsync(new.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return backup
