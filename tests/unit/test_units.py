"""Unit tests for llmlab.units."""

from __future__ import annotations

import pytest

from llmlab import units


def test_round_trip_binary() -> None:
    assert units.parse_bytes("24GiB") == 24 * units.GiB
    assert units.parse_bytes("24 GiB") == 24 * units.GiB
    assert units.parse_bytes("512MiB") == 512 * units.MiB
    assert units.parse_bytes("57344") == 57344
    assert units.parse_bytes("1.5GB") == int(1.5 * units.GB)


def test_fmt_bytes_labels() -> None:
    assert units.fmt_bytes(units.GiB).endswith("GiB")
    assert units.fmt_bytes(units.GB, system="decimal").endswith("GB")
    assert units.fmt_bytes(512) == "512 B"


def test_fmt_bytes_prevents_unit_mixup() -> None:
    # fmt_bytes always labels the system, so GiB vs GB confusion is visible.
    assert "GiB" in units.fmt_bytes(6_850_000_000)
    assert "GB" in units.fmt_bytes(6_850_000_000, system="decimal")


def test_ch4_free_slots_regression() -> None:
    """Regression: v8 case study free slots = 196,608 − 140,480 = 56,128.

    The wrong ~53,000 figure arises from the unit mix-up 6.85e9 / 128e3
    (decimal GB divided by a binary-ish per-token figure); explicit labelled
    units prevent it. Book Ch. 4.
    """
    total_slots = 196_608
    used_slots = 140_480
    assert total_slots - used_slots == 56_128
    wrong = int(6.85e9 / 128e3)
    assert 52_000 < wrong < 55_000  # documents the erroneous ~53k value
    assert total_slots - used_slots != wrong


def test_ns_conversion() -> None:
    assert units.ns_to_s(1_000_000_000) == pytest.approx(1.0)
    assert units.s_to_ns(1.5) == 1_500_000_000


def test_parse_errors() -> None:
    with pytest.raises(ValueError):
        units.parse_bytes("abc")
    with pytest.raises(ValueError):
        units.parse_bytes("10 XB")
    with pytest.raises(ValueError):
        units.fmt_bytes(10, system="furlongs")
