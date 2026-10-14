"""In-memory buffer for rule-file lines.

functions.storeRule used to open the rule file once per line; lines are now collected here
and written in one go by flush() before the file is read.
"""

import os

_DISABLED = os.environ.get("CHEFBOOST_NO_RULEBUFFER") == "1"

_BUFFER = {}


def reset(path: str) -> None:
    """Discard the unwritten buffer of a file (called when the file is re-created)."""
    _BUFFER.pop(path, None)


def append(path: str, content: str) -> None:
    if _DISABLED:
        with open(path, "a+", encoding="UTF-8") as f:
            f.write(content)
        return
    _BUFFER.setdefault(path, []).append(content)


def flush(path: str) -> None:
    """Append the buffered lines to the file and clear the buffer."""
    lines = _BUFFER.pop(path, None)
    if lines:
        with open(path, "a+", encoding="UTF-8") as f:
            f.write("".join(lines))


def flush_all() -> None:
    for path in list(_BUFFER):
        flush(path)
