# silvermark

Provenance and immutability checks for Iceberg lakehouses.

Three things in one library:

- **dedup** - MinHash-LSH near-duplicate detection
- **contamination** - n-gram overlap between two text corpora (train vs eval leakage)
- **attest** - snapshot-level hash attestation so you can prove an eval set hasn't moved

Built after I spent a week last quarter trying to convince myself that the eval set we shipped was actually the eval set we trained against. The answer was "probably, but I can't prove it." This is the proof.

## install

```
pip install silvermark
```

Needs Python 3.11+, pyiceberg 0.7+, duckdb 0.10+.

## quickstart

### dedup with minhash

```python
from silvermark.dedup import shingle, minhash_signature, jaccard_estimate

a = minhash_signature(shingle("the quick brown fox jumps over the lazy dog"), num_perm=128, seed=1)
b = minhash_signature(shingle("the quick brown fox jumped over a lazy dog"), num_perm=128, seed=1)
print(jaccard_estimate(a, b))  # estimated Jaccard, e.g. 0.62
```

For bucketing candidate pairs at scale, `lsh_bands(sig, bands, rows)` splits a signature into LSH bands. Pick `(bands, rows)` so `bands * rows == num_perm` and the s-curve threshold `(1 / bands) ** (1 / rows)` matches the Jaccard cutoff you want.

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

You load the text columns yourself with pyiceberg or duckdb and pass them in as iterables of strings. A wrapper that takes Iceberg table identifiers directly is on the v0.1 roadmap.

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

## what works today

- Read-only. silvermark never modifies tables.
- Pure Python. No Spark, no JVM. Tested on 8 GB laptops.
- 48 tests passing including real Iceberg round-trips via pyiceberg's SqlCatalog.

## what does not work yet

- Iceberg-table-identifier wrappers for dedup and contamination. v0 takes iterables of strings; you load text via pyiceberg or duckdb yourself. The `attest` module already takes a real Iceberg `Table`.
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
