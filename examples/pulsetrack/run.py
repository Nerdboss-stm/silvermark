"""silvermark on a tiny PulseTrack-shaped lakehouse.

PulseTrack is a streaming health-data lakehouse (Iceberg on S3, Spark
Structured Streaming, dbt + Snowflake). silvermark was extracted from it.
This script demonstrates how each module gets used on a PulseTrack-shaped
medallion, in 60-ish seconds, on a laptop, with no AWS and no Docker.

It builds two tiny tables via pyiceberg's SqlCatalog (SQLite + local
filesystem warehouse), then runs:

  1. dedup.minhash      - finds near-duplicate sensor rows in bronze
  2. contamination      - finds train/eval text leakage in silver
  3. attest             - fingerprints the bronze snapshot, then
                          proves an old fingerprint still matches after a
                          new write has landed
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, NestedField, StringType

from silvermark.attest import from_iceberg_snapshot, verify
from silvermark.contamination import ngram_overlap
from silvermark.dedup import jaccard_estimate, minhash_signature, shingle

# --------------------------------------------------------------------
# 1. catalog setup
# --------------------------------------------------------------------


def build_catalog(workdir: Path) -> SqlCatalog:
    warehouse = workdir / "warehouse"
    warehouse.mkdir(exist_ok=True)
    db = workdir / "catalog.db"
    cat = SqlCatalog(
        "pulsetrack",
        **{
            "uri": f"sqlite:///{db}",
            "warehouse": f"file://{warehouse}",
        },
    )
    cat.create_namespace_if_not_exists("bronze")
    cat.create_namespace_if_not_exists("silver")
    return cat


# --------------------------------------------------------------------
# 2. seed the bronze sensor_readings table with deliberate near-dups
# --------------------------------------------------------------------


def populate_sensor_readings(catalog: SqlCatalog):
    schema = Schema(
        NestedField(1, "reading_id", StringType(), required=True),
        NestedField(2, "device_id", StringType(), required=True),
        NestedField(3, "patient_email", StringType(), required=False),
        NestedField(4, "metric_name", StringType(), required=True),
        NestedField(5, "metric_value", DoubleType(), required=True),
    )
    table = catalog.create_table("bronze.sensor_readings", schema=schema)

    rows: list[dict] = []
    for i in range(80):
        rows.append(
            {
                "reading_id": f"r-{i:05d}",
                "device_id": f"WT-A{i % 10}F-{i:05d}",
                "patient_email": f"patient_{i % 25}@example.org",
                "metric_name": "heart_rate_bpm" if i % 2 == 0 else "spo2_pct",
                "metric_value": 72.5 + (i % 7) * 0.5,
            }
        )
    # 20 near-duplicates: same device/metric/patient, value drift of 0.01
    for i in range(20):
        src = rows[i]
        rows.append(
            {
                **src,
                "reading_id": f"r-dup-{i:05d}",
                "metric_value": src["metric_value"] + 0.01,
            }
        )

    arrow = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("reading_id", pa.string(), nullable=False),
                pa.field("device_id", pa.string(), nullable=False),
                pa.field("patient_email", pa.string()),
                pa.field("metric_name", pa.string(), nullable=False),
                pa.field("metric_value", pa.float64(), nullable=False),
            ]
        ),
    )
    table.append(arrow)
    return table, rows


# --------------------------------------------------------------------
# 3. seed silver.clinical_notes with train + eval splits and one leak
# --------------------------------------------------------------------


def populate_clinical_notes(catalog: SqlCatalog):
    schema = Schema(
        NestedField(1, "note_id", StringType(), required=True),
        NestedField(2, "split", StringType(), required=True),
        NestedField(3, "note_text", StringType(), required=True),
        NestedField(4, "patient_email", StringType(), required=False),
    )
    table = catalog.create_table("silver.clinical_notes", schema=schema)

    train_texts = [
        "patient presented with shortness of breath and elevated heart rate after exertion",
        "lab results indicate normal kidney and liver function",
        "discharged in stable condition after 48 hours of observation",
        "recommended follow-up in two weeks for routine checkup",
        "no acute distress noted on physical examination",
        "vital signs stable throughout the overnight stay",
        "ekg shows normal sinus rhythm without ectopy",
        "patient tolerated the procedure well without complications",
    ]
    eval_texts = [
        "patient admitted to the emergency room on tuesday morning",
        # this next one is verbatim from train, leaked
        "discharged in stable condition after 48 hours of observation",
        "vital signs were monitored continuously during the stay",
    ]

    rows = []
    for i, t in enumerate(train_texts):
        rows.append(
            {
                "note_id": f"train-{i:03d}",
                "split": "train",
                "note_text": t,
                "patient_email": f"patient_{i}@example.org",
            }
        )
    for i, t in enumerate(eval_texts):
        rows.append(
            {
                "note_id": f"eval-{i:03d}",
                "split": "eval",
                "note_text": t,
                "patient_email": f"patient_{i + 100}@example.org",
            }
        )

    arrow = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("note_id", pa.string(), nullable=False),
                pa.field("split", pa.string(), nullable=False),
                pa.field("note_text", pa.string(), nullable=False),
                pa.field("patient_email", pa.string()),
            ]
        ),
    )
    table.append(arrow)
    return table, rows


# --------------------------------------------------------------------
# 4. dedup demo
# --------------------------------------------------------------------


def demo_dedup(rows: list[dict]) -> list[tuple[str, str, float]]:
    """Build a minhash signature per row over (device, metric, rounded value)
    and report pairs above a Jaccard threshold. The 20 near-dups we seeded
    should all be found."""
    signatures = {}
    for r in rows:
        signature_input = (
            f"device={r['device_id']}|"
            f"metric={r['metric_name']}|"
            f"value~{round(r['metric_value'])}|"
            f"patient={r['patient_email']}"
        )
        signatures[r["reading_id"]] = minhash_signature(
            shingle(signature_input, k=5), num_perm=64, seed=1
        )

    ids = list(signatures.keys())
    found = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            est = jaccard_estimate(signatures[ids[i]], signatures[ids[j]])
            if est > 0.9:
                found.append((ids[i], ids[j], est))
    return found


# --------------------------------------------------------------------
# 5. contamination demo
# --------------------------------------------------------------------


def demo_contamination(notes_rows: list[dict]):
    train = [r["note_text"] for r in notes_rows if r["split"] == "train"]
    eval_ = [r["note_text"] for r in notes_rows if r["split"] == "eval"]
    return ngram_overlap(train=train, eval=eval_, n=8, examples=3)


# --------------------------------------------------------------------
# 6. attest demo: fingerprint, write more, re-verify the old fingerprint
# --------------------------------------------------------------------


def demo_attest(catalog: SqlCatalog, table):
    snap_id = table.current_snapshot().snapshot_id
    fp = from_iceberg_snapshot(table, snap_id)

    # simulate "weeks later, new data arrived"
    extra = pa.Table.from_pylist(
        [
            {
                "reading_id": "r-late-0001",
                "device_id": "WT-LATE-0001",
                "patient_email": "late@example.org",
                "metric_name": "heart_rate_bpm",
                "metric_value": 99.0,
            }
        ],
        schema=pa.schema(
            [
                pa.field("reading_id", pa.string(), nullable=False),
                pa.field("device_id", pa.string(), nullable=False),
                pa.field("patient_email", pa.string()),
                pa.field("metric_name", pa.string(), nullable=False),
                pa.field("metric_value", pa.float64(), nullable=False),
            ]
        ),
    )
    table.append(extra)
    new_snap = table.current_snapshot().snapshot_id

    return {
        "original_snapshot": snap_id,
        "original_fingerprint": fp.fingerprint,
        "data_file_count_at_t1": fp.data_file_count,
        "new_snapshot": new_snap,
        "old_fingerprint_still_verifies": verify(table, snap_id, fp.fingerprint),
    }


# --------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------


def main():
    workdir = Path(tempfile.mkdtemp(prefix="silvermark-pulsetrack-"))
    try:
        print("workdir:", workdir)
        print()

        catalog = build_catalog(workdir)
        sensor_table, sensor_rows = populate_sensor_readings(catalog)
        _, notes_rows = populate_clinical_notes(catalog)

        print("=== dedup: near-duplicate sensor rows ===")
        dups = demo_dedup(sensor_rows)
        print(f"checked {len(sensor_rows)} rows, found {len(dups)} near-duplicate pairs")
        for a, b, est in dups[:5]:
            print(f"  {a:>15s}  ~  {b:<15s}  jaccard={est:.3f}")
        if len(dups) > 5:
            print(f"  ... and {len(dups) - 5} more")
        print()

        print("=== contamination: train vs eval clinical notes ===")
        contam = demo_contamination(notes_rows)
        print(f"train n-grams: {contam.train_ngram_count}")
        print(f"eval  n-grams: {contam.eval_ngram_count}")
        print(f"overlap:       {contam.overlapping_ngrams} ({contam.overlap_rate:.2%})")
        print(f"is_contaminated(threshold=0.005): {contam.is_contaminated()}")
        print(f"example overlapping n-grams: {contam.sample_overlaps}")
        print()

        print("=== attest: snapshot fingerprint across a new write ===")
        result = demo_attest(catalog, sensor_table)
        print(f"snapshot at t1:  {result['original_snapshot']}")
        print(f"fingerprint:     {result['original_fingerprint']}")
        print(f"files at t1:     {result['data_file_count_at_t1']}")
        print(f"snapshot at t2:  {result['new_snapshot']}  (after extra append)")
        print(
            "old fingerprint still verifies against t1 snapshot: "
            f"{result['old_fingerprint_still_verifies']}"
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
