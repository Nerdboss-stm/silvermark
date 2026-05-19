"""Deterministic fingerprints for Iceberg snapshots.

The promise: take an Iceberg snapshot, produce a hex string. Later, given the
same table and snapshot id, recompute the hex string. If they match, every
data file referenced by that snapshot is still where it was, and nothing
silently moved.

What goes into the fingerprint:
- The list of data files in the snapshot, sorted by path for stability
- For each data file: path, size in bytes, record count

Sorting matters because Iceberg's manifest order is not guaranteed across
catalogs or across pyiceberg versions. A fingerprint that depends on
manifest order would drift even when nothing real changed.

What is NOT in the fingerprint:
- The snapshot_id itself - we report it separately so two snapshots with
  the same data files can be detected as logically equivalent
- Timestamps - they change every time the manifest is rewritten
- Iceberg metadata json paths - we hash file_path because that's what
  carries actual data
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class DataFileRef:
    """One row in a snapshot's data-file list. Hashable so we can dedupe."""

    file_path: str
    file_size_in_bytes: int
    record_count: int


@dataclass(frozen=True)
class SnapshotFingerprint:
    """A computed fingerprint plus the metadata needed to verify it later."""

    snapshot_id: int
    manifest_list: str
    data_file_count: int
    fingerprint: str
    computed_at: datetime

    def matches(self, other: SnapshotFingerprint | str) -> bool:
        """Compare against another fingerprint or a raw hex string."""
        if isinstance(other, str):
            return self.fingerprint == other
        return self.fingerprint == other.fingerprint


def fingerprint_data_files(data_files: Iterable[DataFileRef]) -> str:
    """SHA-256 over a canonical-form data file list.

    Two snapshots referencing the same data files produce the same hex
    string, regardless of the order pyiceberg returned them in.
    """
    h = hashlib.sha256()
    # Sort by path so manifest-traversal order does not matter.
    sorted_refs = sorted(
        set(data_files), key=lambda d: (d.file_path, d.file_size_in_bytes, d.record_count)
    )
    for ref in sorted_refs:
        line = f"{ref.file_path}\t{ref.file_size_in_bytes}\t{ref.record_count}\n"
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def from_iceberg_snapshot(table: Any, snapshot_id: int) -> SnapshotFingerprint:
    """Fingerprint an Iceberg snapshot by reading its manifests.

    `table` is a pyiceberg Table. Raises ValueError if the snapshot is not
    in the table's history (e.g. it was expired or never existed).
    """
    snap = table.snapshot_by_id(snapshot_id)
    if snap is None:
        raise ValueError(
            f"snapshot {snapshot_id} not found in table. "
            f"It may have been expired by expire_snapshots."
        )

    refs: list[DataFileRef] = []
    for manifest in snap.manifests(table.io):
        for entry in manifest.fetch_manifest_entry(table.io):
            df = entry.data_file
            refs.append(
                DataFileRef(
                    file_path=str(df.file_path),
                    file_size_in_bytes=int(df.file_size_in_bytes),
                    record_count=int(df.record_count),
                )
            )

    return SnapshotFingerprint(
        snapshot_id=snapshot_id,
        manifest_list=str(snap.manifest_list),
        data_file_count=len(refs),
        fingerprint=fingerprint_data_files(refs),
        computed_at=datetime.now(timezone.utc),
    )


def verify(table: Any, snapshot_id: int, expected: SnapshotFingerprint | str) -> bool:
    """Recompute the fingerprint and compare against an expected value.

    Use this to assert a snapshot is unchanged since you last saw it. If
    Iceberg's expire_snapshots has dropped this snapshot since then, this
    raises ValueError instead of returning False, because that is a
    different and more serious problem than data drift.
    """
    actual = from_iceberg_snapshot(table, snapshot_id)
    return actual.matches(expected)
