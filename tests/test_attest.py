"""Tests for snapshot fingerprinting.

Two layers:
  1. Pure unit tests on the fingerprint() function with synthetic DataFileRef
     lists. No pyiceberg needed.
  2. Integration tests that create a real Iceberg table via pyiceberg's
     SqlCatalog (SQLite-backed, file-system warehouse) and round-trip a
     fingerprint through it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pytest

from silvermark.attest import (
    DataFileRef,
    fingerprint_data_files,
    from_iceberg_snapshot,
    verify,
)

# ============================================================================
# pure algorithm tests - no pyiceberg
# ============================================================================


def test_empty_data_files_returns_known_hash():
    # sha256 of empty input is a well-known constant
    h = fingerprint_data_files([])
    assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_single_file_is_deterministic():
    refs = [DataFileRef("s3://w/data/00001.parquet", 1024, 100)]
    a = fingerprint_data_files(refs)
    b = fingerprint_data_files(refs)
    assert a == b


def test_order_independence():
    refs_in_order = [
        DataFileRef("s3://w/00001.parquet", 100, 1),
        DataFileRef("s3://w/00002.parquet", 200, 2),
        DataFileRef("s3://w/00003.parquet", 300, 3),
    ]
    refs_reversed = list(reversed(refs_in_order))
    assert fingerprint_data_files(refs_in_order) == fingerprint_data_files(refs_reversed)


def test_different_path_different_hash():
    a = fingerprint_data_files([DataFileRef("s3://w/a.parquet", 100, 1)])
    b = fingerprint_data_files([DataFileRef("s3://w/b.parquet", 100, 1)])
    assert a != b


def test_different_size_different_hash():
    a = fingerprint_data_files([DataFileRef("s3://w/a.parquet", 100, 1)])
    b = fingerprint_data_files([DataFileRef("s3://w/a.parquet", 200, 1)])
    assert a != b


def test_different_record_count_different_hash():
    a = fingerprint_data_files([DataFileRef("s3://w/a.parquet", 100, 1)])
    b = fingerprint_data_files([DataFileRef("s3://w/a.parquet", 100, 99)])
    assert a != b


def test_duplicate_refs_collapse_via_set():
    # Same file listed twice in the manifest should not change the hash.
    # set() in fingerprint_data_files takes care of this.
    once = [DataFileRef("s3://w/a.parquet", 100, 1)]
    twice = [
        DataFileRef("s3://w/a.parquet", 100, 1),
        DataFileRef("s3://w/a.parquet", 100, 1),
    ]
    assert fingerprint_data_files(once) == fingerprint_data_files(twice)


def test_realistic_partition_set():
    # Mirror the shape of a real Iceberg partition: 32 files, varied sizes
    refs = [
        DataFileRef(
            file_path=f"s3://warehouse/bronze/sensor_readings/data/dt=2026-05-19/part-{i:05d}.parquet",
            file_size_in_bytes=268_000_000 + (i * 1000),
            record_count=100_000 + i,
        )
        for i in range(32)
    ]
    h = fingerprint_data_files(refs)
    assert len(h) == 64  # SHA-256 hex
    assert all(c in "0123456789abcdef" for c in h)


def test_fingerprint_matches_self():
    refs = [DataFileRef("s3://w/a.parquet", 100, 1)]
    h = fingerprint_data_files(refs)
    assert h == fingerprint_data_files(refs)


# ============================================================================
# integration tests - real pyiceberg sqlite catalog
# ============================================================================


@pytest.fixture
def catalog(tmp_path: Path):
    """A throwaway SQLite-backed Iceberg catalog with a file:// warehouse."""
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    db = tmp_path / "catalog.db"

    cat = SqlCatalog(
        "test",
        **{
            "uri": f"sqlite:///{db}",
            "warehouse": f"file://{warehouse}",
        },
    )
    cat.create_namespace("ns")
    yield cat
    # SqlCatalog can hold file handles; explicit cleanup keeps Windows CI happy.
    try:
        shutil.rmtree(warehouse)
    except FileNotFoundError:
        pass


@pytest.fixture
def small_table(catalog):
    """A 3-row table with one append committed. Returns (table, snapshot_id)."""
    from pyiceberg.schema import Schema
    from pyiceberg.types import IntegerType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "label", StringType(), required=False),
    )
    table = catalog.create_table("ns.t", schema=schema)
    data = pa.Table.from_pydict(
        {"id": [1, 2, 3], "label": ["a", "b", "c"]},
        schema=pa.schema(
            [pa.field("id", pa.int32(), nullable=False), pa.field("label", pa.string())]
        ),
    )
    table.append(data)
    return table, table.current_snapshot().snapshot_id


def test_iceberg_snapshot_fingerprint_is_deterministic(small_table):
    table, snap_id = small_table
    a = from_iceberg_snapshot(table, snap_id)
    b = from_iceberg_snapshot(table, snap_id)
    assert a.fingerprint == b.fingerprint
    assert a.snapshot_id == snap_id
    assert a.data_file_count >= 1


def test_iceberg_snapshot_fingerprint_has_correct_shape(small_table):
    table, snap_id = small_table
    fp = from_iceberg_snapshot(table, snap_id)
    assert isinstance(fp.fingerprint, str)
    assert len(fp.fingerprint) == 64
    assert fp.snapshot_id == snap_id
    assert fp.manifest_list  # non-empty path


def test_iceberg_verify_matches(small_table):
    table, snap_id = small_table
    fp = from_iceberg_snapshot(table, snap_id)
    assert verify(table, snap_id, fp) is True


def test_iceberg_verify_against_raw_hex(small_table):
    table, snap_id = small_table
    fp = from_iceberg_snapshot(table, snap_id)
    assert verify(table, snap_id, fp.fingerprint) is True
    assert verify(table, snap_id, "0" * 64) is False


def test_iceberg_appended_data_changes_fingerprint(small_table):
    table, original_snap = small_table
    fp_before = from_iceberg_snapshot(table, original_snap)

    new_data = pa.Table.from_pydict(
        {"id": [4, 5], "label": ["d", "e"]},
        schema=pa.schema(
            [pa.field("id", pa.int32(), nullable=False), pa.field("label", pa.string())]
        ),
    )
    table.append(new_data)
    new_snap = table.current_snapshot().snapshot_id

    fp_after = from_iceberg_snapshot(table, new_snap)
    assert new_snap != original_snap
    assert fp_after.fingerprint != fp_before.fingerprint
    assert fp_after.data_file_count > fp_before.data_file_count


def test_iceberg_old_snapshot_still_verifies_after_new_writes(small_table):
    """The whole point of fingerprints: an old snapshot's fingerprint
    keeps matching even after new appends. The history is immutable."""
    table, snap_id = small_table
    fp_at_time_t1 = from_iceberg_snapshot(table, snap_id)

    new_data = pa.Table.from_pydict(
        {"id": [99], "label": ["new"]},
        schema=pa.schema(
            [pa.field("id", pa.int32(), nullable=False), pa.field("label", pa.string())]
        ),
    )
    table.append(new_data)

    # at time T2, the table has moved on but our T1 snapshot is unchanged
    assert verify(table, snap_id, fp_at_time_t1) is True


def test_iceberg_missing_snapshot_raises(small_table):
    table, _ = small_table
    fake_snap_id = 999_999_999_999
    with pytest.raises(ValueError, match="not found"):
        from_iceberg_snapshot(table, fake_snap_id)


def test_snapshot_fingerprint_matches_method():
    from datetime import datetime, timezone

    from silvermark.attest import SnapshotFingerprint

    fp1 = SnapshotFingerprint(
        snapshot_id=1,
        manifest_list="x",
        data_file_count=0,
        fingerprint="abc123",
        computed_at=datetime.now(timezone.utc),
    )
    fp2 = SnapshotFingerprint(
        snapshot_id=2,
        manifest_list="y",
        data_file_count=99,
        fingerprint="abc123",
        computed_at=datetime.now(timezone.utc),
    )
    fp3 = SnapshotFingerprint(
        snapshot_id=1,
        manifest_list="x",
        data_file_count=0,
        fingerprint="different",
        computed_at=datetime.now(timezone.utc),
    )
    assert fp1.matches(fp2) is True  # same hex
    assert fp1.matches(fp3) is False
    assert fp1.matches("abc123") is True
    assert fp1.matches("different") is False
