# Changelog

All notable changes go here. Format roughly follows Keep a Changelog.

## [Unreleased]

## [0.0.1] - 2026-05-21

First public release. Three modules, 49 tests, end-to-end example.

### added
- `dedup` - MinHash + LSH bands for near-duplicate detection
- `contamination` - n-gram overlap between two text corpora
- `attest` - SHA-256 snapshot fingerprint over Iceberg data file lists, with `verify` for later
- `examples/pulsetrack/` - one script that builds a small Iceberg lakehouse and runs all three modules
- GitHub Actions CI matrix on Python 3.10 / 3.11 / 3.12

### tested against
- pyiceberg 0.11.1, duckdb 0.10+, boto3 1.43+ (when using a Glue catalog)
- macOS arm64 + Linux x86_64 (CI)
- real AWS Glue catalog + S3 Iceberg tables (manual verification 2026-05-21, 5 tables fingerprinted in under 8 seconds total)

### known limitations
- No `pip install silvermark` from PyPI yet, install from git
- `dedup` and `contamination` take iterables of strings, not Iceberg table identifiers (you load text via pyiceberg or duckdb yourself). The `attest` module takes a real `Table`.
- Single-node MinHash. Fine up to about 100M shingles. For bigger, use Spark.
- Only externally-managed Iceberg tables (Glue, Hive, REST, SQL). Snowflake-managed tables are on the v0.1 list.

[0.0.1]: https://github.com/Nerdboss-stm/silvermark/releases/tag/v0.0.1
[Unreleased]: https://github.com/Nerdboss-stm/silvermark/compare/v0.0.1...HEAD
