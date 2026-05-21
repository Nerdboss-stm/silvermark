# silvermark

[![tests](https://github.com/Nerdboss-stm/silvermark/actions/workflows/test.yml/badge.svg)](https://github.com/Nerdboss-stm/silvermark/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/Nerdboss-stm/silvermark)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Provenance and immutability checks for Iceberg lakehouses.

Three things in one library:

- **dedup** - MinHash-LSH near-duplicate detection
- **contamination** - n-gram overlap between two text corpora (train vs eval leakage)
- **attest** - snapshot-level hash attestation so you can prove an eval set hasn't moved

## why this exists

Last week I left a Spark Structured Streaming job running with `trigger(processingTime=5s)` overnight and no scheduled compaction. By morning the bronze Iceberg table had 850,212 files at 12 KB each. Every query stalled in the Catalyst planner.

The recovery took about 24 minutes of real `rewrite_data_files` work and 6 hours of YARN config archaeology. After that mess, I tried to run an eval against the silver layer. I couldn't prove the snapshot I was reading was the same snapshot I'd evaluated against the previous week. `expire_snapshots` had run between the two evals. The snapshot ID matched on paper. Whether the underlying files were identical, I had no way to check without manually reading manifests.

That's the gap silvermark fills. Save a fingerprint at eval time, verify it later. Run an n-gram check between train and eval and see how much overlap is hiding in your "clean" split. Find near-duplicate rows that slipped through the dedup you thought worked.

## install

```
pip install git+https://github.com/Nerdboss-stm/silvermark.git@main
```

PyPI publish is coming with v0.1. Needs Python 3.10+, pyiceberg 0.7+, duckdb 0.10+.

## quickstart

### dedup with minhash

```python
from silvermark.dedup import shingle, minhash_signature, jaccard_estimate

# Use a smaller k for short texts. Default is k=9, which means each shingle
# is 9 consecutive words; that's right for paragraph-length notes but too
# coarse for a single sentence.
a_sh = shingle("the patient was discharged in stable condition after 48 hours", k=3)
b_sh = shingle("patient was discharged in stable condition after 48 hours of observation", k=3)

a = minhash_signature(a_sh, num_perm=128, seed=1)
b = minhash_signature(b_sh, num_perm=128, seed=1)
print(f"{jaccard_estimate(a, b):.3f}")  # ~0.7 (true Jaccard 7/10 = 0.700)
```

For bucketing candidate pairs at scale, `lsh_bands(sig, bands, rows)` splits a signature into LSH bands. Pick `(bands, rows)` so `bands * rows == num_perm` and the s-curve threshold `(1 / bands) ** (1 / rows)` matches the Jaccard cutoff you want.

A larger end-to-end example (Iceberg tables + all three modules) is at `examples/pulsetrack/run.py`.

### contamination check between two corpora

```python
from silvermark.contamination import ngram_overlap

train_texts = ["the patient was discharged in stable condition", "..."]
eval_texts = ["patient admitted to ER on Tuesday", "the patient was discharged in stable condition"]

report = ngram_overlap(train=train_texts, eval=eval_texts, n=8)
print(f"{report.overlap_rate:.4f}")           # e.g. 0.07
print(report.overlapping_ngrams)              # e.g. 41
print(report.sample_overlaps[:3])             # first three overlapping n-grams
if report.is_contaminated(threshold=0.005):
    print("warning: leakage above 0.5% threshold")
```

If you'd rather hand it Iceberg tables directly (no manual scan + to_pylist), there's a wrapper:

```python
from pyiceberg.catalog import load_catalog
from silvermark.contamination import ngram_overlap_iceberg

catalog = load_catalog("default", **{"type": "glue", "glue.region": "us-east-1"})
train = catalog.load_table("warehouse.silver.training_v3")
eval_  = catalog.load_table("warehouse.silver.eval_v1")

report = ngram_overlap_iceberg(train, eval_, column="note", n=8)
```

The wrapper does projection on only the named column, so it doesn't materialize the rest of the row.

### snapshot attestation

```python
from silvermark.attest import from_iceberg_snapshot, verify

# at the time you run your eval, fingerprint the snapshot
fp = from_iceberg_snapshot(table, snapshot_id=4429181903)
print(fp.fingerprint)         # 64-char sha256 hex
print(fp.data_file_count)     # how many files this snapshot references

# write fp.fingerprint to your eval-run log alongside scores

# weeks later, prove the snapshot is unchanged
assert verify(table, snapshot_id=4429181903, expected=fp.fingerprint)
```

The fingerprint is deterministic over the snapshot's data file list (path, size, record count), sorted by path. It does not depend on pyiceberg's manifest-traversal order, so the same snapshot fingerprints the same way across runs and across catalogs.

If the snapshot has been expired (via `expire_snapshots`), `from_iceberg_snapshot` raises `ValueError`. That is intentionally a different error than "fingerprint mismatch" because the data is gone, not drifting.

## end-to-end example

`examples/pulsetrack/run.py` builds a tiny PulseTrack-shaped lakehouse (two Iceberg tables via pyiceberg + SqlCatalog), runs all three modules on it, and prints a report. About 30 seconds on a laptop. No AWS, no Docker. See `examples/pulsetrack/README.md` and `examples/pulsetrack/expected_output.txt`.

## what works today

- Read-only. silvermark never modifies tables.
- Pure Python. No Spark, no JVM. Tested on 8 GB laptops.
- 48 tests passing including real Iceberg round-trips via pyiceberg's SqlCatalog.

## what does not work yet

- Iceberg-table wrapper for `dedup`. The `attest` and `contamination` modules already take real Iceberg `Table`s; `dedup` still expects iterables of strings.
- Distributed MinHash. Single-node only, fine up to about 100M shingles. For bigger, use Spark.
- Snowflake-managed Iceberg tables (only externally-managed via Glue, Hive, REST, or SQL catalogs).

## design notes

- Why MinHash and not embeddings? Embeddings need a model and the model is a moving target across runs. MinHash is deterministic for a given seed and signature length, so two runs against the same data produce identical results.
- Why character n-grams for contamination? Because lakehouse text columns mix natural language and tokenless strings (URLs, error messages, codes), and character n-grams handle both. Word n-grams break on the second category.
- Why a 2^31 - 1 prime in the MinHash? Bigger primes (like the textbook 2^61 - 1) silently overflow numpy uint64 when multiplied by hash outputs. M31 keeps everything in safe arithmetic. The tests catch this.

## design partners wanted

If you have a real Iceberg lakehouse with eval-set hygiene concerns, I'd like to test silvermark on your data. Email saran@stmallela.com. The first three users get me on call while we shake bugs out.

## license

MIT. See LICENSE.
