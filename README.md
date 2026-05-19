# silvermark

Provenance and immutability checks for Iceberg lakehouses.

Three things in one library:

- **dedup** — MinHash-LSH near-duplicate detection across partitions and snapshots
- **contamination** — n-gram overlap between any two tables (train vs eval splits, especially)
- **attest** — snapshot-level hash attestation so you can prove an eval set hasn't moved

Built after I spent a week last quarter trying to convince myself that the eval set we shipped was actually the eval set we trained against. The answer was "probably, but I can't prove it." This is the proof.

## install

```
pip install silvermark
```

Needs Python 3.11+, pyiceberg 0.7+, duckdb 0.10+.

## quickstart

```python
from silvermark import attest, contamination, dedup

# proof a snapshot hasn't changed since you ran your eval
fingerprint = attest.snapshot_hash(
    "warehouse.silver.eval_v3",
    snapshot_id=4429181903,
)

# is there train/eval leakage?
report = contamination.ngram_overlap(
    train="warehouse.silver.training_v3",
    eval="warehouse.silver.eval_v1",
    n=8,
    sample=1.0,
)
print(report.overlap_rate)  # 0.0023 means 0.23% of eval n-grams appear in training

# find near-duplicates in a partition
groups = dedup.minhash(
    "warehouse.bronze.events",
    where="ingest_date = '2026-05-18'",
    threshold=0.85,
    num_perm=128,
)
```

## what works today

- Read-side only. silvermark never modifies tables. You decide what to do with the report.
- Catalogs: Hive, Glue, Nessie, SQL (whatever pyiceberg supports).
- Backends: pyiceberg + duckdb for local. PyArrow for in-memory.

## what does not work yet

- Snowflake-managed Iceberg tables (only externally-managed). Coming.
- Streaming/online dedup. This is batch only.
- Distributed MinHash. Single-node, fine up to ~100M rows. For bigger, you want spark.
- Write-side anything. silvermark refuses to write.

## design notes

- Why MinHash and not embeddings? Embeddings need a model, and the model is a moving target across runs. MinHash is deterministic for a given seed and signature length.
- Why n-gram contamination and not LLM-as-judge? Because if you're proving train/eval cleanliness, you want a method that doesn't itself depend on an LLM. NVIDIA's text-data-processing guide and the LSHBloom paper both argue for n-gram as the baseline.
- Why snapshot hashes instead of full file checksums? Iceberg manifests already hash the data files. We hash the manifests themselves. O(snapshots) not O(files).

## design partners wanted

If you have a real Iceberg lakehouse with eval-set hygiene concerns, I'd like to test silvermark on your data (or have you test it on yours). Email saran@stmallela.com. The first three users get me as on-call support while we shake bugs out.

## license

MIT. See LICENSE.
