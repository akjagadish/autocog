"""Project logger.

For now this is a thin wrapper around `print()`;
"""

from __future__ import annotations


def log(msg: str, *, level: str = "INFO") -> None:
    """Emit a single message tagged with `level` (defaults to INFO)."""
    print(f"[{level}] {msg}")


def info(msg: str) -> None:
    log(msg, level="INFO")


def warn(msg: str) -> None:
    log(msg, level="WARN")


def error(msg: str) -> None:
    log(msg, level="ERROR")


def debug(msg: str) -> None:
    log(msg, level="DEBUG")
