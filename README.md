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

Coming in v0.1. The plan is `attest.snapshot_hash(table, snapshot_id) -> str`, returning a hash over the manifest list so you can prove a snapshot you ran against earlier hasn't shifted.

## what works today

- Read-only. silvermark never modifies tables.
- Pure Python. No Spark, no JVM. Tested on 8 GB laptops.
- 31 tests, all passing.

## what does not work yet

- Iceberg-table-identifier wrappers. v0 takes iterables of strings; v0.1 adds `from_iceberg(table_id)` helpers.
- Snapshot attestation module is stubbed (raises `NotImplementedError`). Coming in v0.1.
- Distributed MinHash. Single-node only, fine up to about 100M shingles. For bigger, use Spark.
- Snowflake-managed Iceberg tables.

## design notes

- Why MinHash and not embeddings? Embeddings need a model and the model is a moving target across runs. MinHash is deterministic for a given seed and signature length, so two runs against the same data produce identical results.
- Why character n-grams for contamination? Because lakehouse text columns mix natural language and tokenless strings (URLs, error messages, codes), and character n-grams handle both. Word n-grams break on the second category.
- Why a 2^31 - 1 prime in the MinHash? Bigger primes (like the textbook 2^61 - 1) silently overflow numpy uint64 when multiplied by hash outputs. M31 keeps everything in safe arithmetic. The tests catch this.

## design partners wanted

If you have a real Iceberg lakehouse with eval-set hygiene concerns, I'd like to test silvermark on your data. Email saran@stmallela.com. The first three users get me on call while we shake bugs out.

## license

MIT. See LICENSE.
