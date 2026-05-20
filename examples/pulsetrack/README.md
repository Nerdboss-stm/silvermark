# silvermark on PulseTrack

PulseTrack is a streaming health-data lakehouse (Iceberg on S3, Spark Structured Streaming, dbt + Snowflake). silvermark was extracted from it. This example shows what each silvermark module looks like in the context it came from.

## what this script does

`run.py` builds a tiny PulseTrack-shaped medallion entirely in a temp directory: two Iceberg tables (`bronze.sensor_readings` and `silver.clinical_notes`) via pyiceberg's SqlCatalog. No AWS. No Docker. Runs on an 8 GB laptop in under a minute.

Then it demonstrates each silvermark module on those tables:

1. **dedup.minhash** — the bronze table has 80 real sensor readings plus 20 deliberate near-duplicates (same device, same patient, same metric, value drift of 0.01). The script signs each row, compares signatures, and returns pairs above a Jaccard threshold of 0.9. Should find all 20 deliberate dups plus a few incidental collisions where two different rows happen to round to the same hash bucket.
2. **contamination.ngram_overlap** — the silver table has eight train notes and three eval notes. One eval row is verbatim from the train set (the same "discharged in stable condition" string). The script computes n-gram overlap between the two splits and reports the rate. With one leaked sentence, overlap lands around 35-40% which is above the default 0.5% contamination threshold.
3. **attest.from_iceberg_snapshot** — the script fingerprints the bronze snapshot, then appends another row to the table, then re-verifies the original fingerprint against the original snapshot ID. It still matches, because Iceberg snapshot history is immutable.

## run it

From the silvermark repo root:

```
pip install -e .
python examples/pulsetrack/run.py
```

Takes about 30 seconds. Cleans up its own tempdir.

## sample output

See `expected_output.txt`. The deterministic fields (counts, percentages, overlap examples) are stable across runs. The random fields (Iceberg snapshot IDs assigned at commit time, the tempdir name, the sha256 hex over those snapshot IDs) change every run.

## why this matters

The patterns this example shows are what silvermark is for in production. The relevant PulseTrack incident: in May 2026, a streaming bronze table accumulated 850K data files in 16 hours because nothing was running `rewrite_data_files`. Recovery worked. But during recovery, the question "is the eval set I trained against still the same set I'm evaluating on?" had no good answer. silvermark's three modules are the three things I would have needed:

- dedup to spot whether streaming retries had double-committed rows
- contamination to verify train and eval splits hadn't leaked across compaction passes
- attest to prove the eval snapshot I used last quarter still hashes the same

This example is silvermark's smallest end-to-end demonstration of all three.
