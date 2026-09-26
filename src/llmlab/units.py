"""Unit discipline: byte and time constants, parsing, and labelled formatting.

Bytes are ``int``; time is ``int`` nanoseconds internally. GiB-vs-GB labels
are applied only when displaying (book §3.1 conventions).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

KiB: int = 1024
MiB: int = 1024**2
GiB: int = 1024**3
KB: int = 10**3
MB: int = 10**6
GB: int = 10**9

_NS_PER_S: int = 1_000_000_000

_BYTES_RE = re.compile(r"^\s*(?P<num>[0-9][0-9_]*(?:\.[0-9]+)?)\s*(?P<unit>[A-Za-z]*)\s*$")

_UNIT_MULTIPLIERS: dict[str, int] = {
    "": 1,
    "B": 1,
    "KB": KB,
    "MB": MB,
    "GB": GB,
    "KIB": KiB,
    "MIB": MiB,
    "GIB": GiB,
    "K": KiB,
    "M": MiB,
    "G": GiB,
}


def parse_bytes(text: str) -> int:
    """Parse a byte quantity like ``"24GiB"``, ``"24 GB"`` or ``"57344"``.

    Formula: n_bytes = number × multiplier(unit). Book §3.1.
    """
    m = _BYTES_RE.match(text)
    if m is None:
        raise ValueError(f"cannot parse byte quantity: {text!r}")
    num = float(m.group("num").replace("_", ""))
    unit = m.group("unit").upper()
    if unit not in _UNIT_MULTIPLIERS:
        raise ValueError(f"unknown byte unit: {m.group('unit')!r}")
    return int(num * _UNIT_MULTIPLIERS[unit])


def fmt_bytes(n: int, system: str = "binary", digits: int = 2) -> str:
    """Format an int byte count with an explicit unit label.

    Formula: display = n / unit_base with unit label appended. Book §3.1.
    """
    if system not in ("binary", "decimal"):
        raise ValueError(f"unknown system: {system!r}")
    if system == "binary":
        units = [("GiB", GiB), ("MiB", MiB), ("KiB", KiB)]
    else:
        units = [("GB", GB), ("MB", MB), ("KB", KB)]
    for label, size in units:
        if abs(n) >= size:
            return f"{n / size:.{digits}f} {label}"
    return f"{n} B"


def ns_to_s(ns: int) -> float:
    """Convert integer nanoseconds to float seconds. Book §3.1."""
    return ns / _NS_PER_S


def s_to_ns(s: float) -> int:
    """Convert float seconds to integer nanoseconds. Book §3.1."""
    return round(s * _NS_PER_S)
