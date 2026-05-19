"""Snapshot-level hash attestation for Iceberg tables."""

from silvermark.attest.snapshot import (
    DataFileRef,
    SnapshotFingerprint,
    fingerprint_data_files,
    from_iceberg_snapshot,
    verify,
)

__all__ = [
    "DataFileRef",
    "SnapshotFingerprint",
    "fingerprint_data_files",
    "from_iceberg_snapshot",
    "verify",
]
