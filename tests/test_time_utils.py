# coding: utf-8
"""Unit tests for app.utils.time — UTC serialization edge cases."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from app.utils.time import serialize_utc_datetime, utc_now


def test_utc_now_returns_aware_utc():
    now = utc_now()
    assert now.tzinfo == timezone.utc


def test_serialize_none():
    assert serialize_utc_datetime(None) is None


def test_serialize_naive_datetime_gets_z_suffix():
    naive = datetime(2025, 6, 16, 8, 30, 0)
    assert serialize_utc_datetime(naive) == "2025-06-16T08:30:00Z"


def test_serialize_aware_utc_gets_z_suffix():
    aware = datetime(2025, 6, 16, 8, 30, 0, tzinfo=timezone.utc)
    assert serialize_utc_datetime(aware) == "2025-06-16T08:30:00Z"


def test_serialize_non_utc_offset_converts_to_utc():
    # 2025-06-16 16:30 in UTC+8 == 2025-06-16 08:30 UTC
    east8 = timezone(timedelta(hours=8))
    local = datetime(2025, 6, 16, 16, 30, 0, tzinfo=east8)
    assert serialize_utc_datetime(local) == "2025-06-16T08:30:00Z"


def test_serialize_preserves_microseconds():
    naive = datetime(2025, 6, 16, 8, 30, 0, 123456)
    assert serialize_utc_datetime(naive) == "2025-06-16T08:30:00.123456Z"
