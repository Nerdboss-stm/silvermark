"""Tests for the Iceberg-table convenience wrappers on the contamination side.

These build a small SqlCatalog-backed Iceberg table with a text column and
exercise the wrapper end-to-end. The pure-python path is already covered
in test_ngram.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pytest

from silvermark.contamination import (
    column_ngrams_from_table,
    ngram_overlap_iceberg,
)


@pytest.fixture
def catalog(tmp_path: Path):
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
    try:
        shutil.rmtree(warehouse)
    except FileNotFoundError:
        pass


def _make_text_table(catalog, name: str, texts: list[str]):
    from pyiceberg.schema import Schema
    from pyiceberg.types import IntegerType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "note", StringType(), required=False),
    )
    table = catalog.create_table(f"ns.{name}", schema=schema)
    data = pa.Table.from_pydict(
        {"id": list(range(len(texts))), "note": texts},
        schema=pa.schema(
            [
                pa.field("id", pa.int32(), nullable=False),
                pa.field("note", pa.string()),
            ]
        ),
    )
    table.append(data)
    return table


def test_column_ngrams_from_table_basic(catalog):
    table = _make_text_table(
        catalog,
        "tbl_a",
        ["the patient was discharged in stable condition after 48 hours of observation"],
    )
    ngs = column_ngrams_from_table(table, column="note", n=8)
    # The text is ~75 chars, so we expect a lot of 8-grams.
    assert len(ngs) >= 50
    assert all(isinstance(s, str) for s in ngs)
    assert all(len(s) == 8 for s in ngs)


def test_column_ngrams_from_table_unknown_column_raises(catalog):
    table = _make_text_table(catalog, "tbl_missing_col", ["hello world"])
    with pytest.raises(KeyError, match="not found in table schema"):
        column_ngrams_from_table(table, column="does_not_exist", n=4)


def test_column_ngrams_handles_nulls(catalog):
    # None values in a text column should be skipped, not crash.
    table = _make_text_table(
        catalog, "tbl_with_nulls", ["alpha beta gamma delta epsilon zeta", None, "iota kappa lambda mu nu"]
    )
    ngs = column_ngrams_from_table(table, column="note", n=5)
    assert "alpha" in ngs
    assert "iota " in ngs or "iota" in "".join(ngs)


def test_ngram_overlap_iceberg_identical_tables(catalog):
    text = ["the quick brown fox jumps over the lazy dog and runs back home"]
    train = _make_text_table(catalog, "train_identical", text)
    evals = _make_text_table(catalog, "eval_identical", text)

    report = ngram_overlap_iceberg(train, evals, column="note", n=6)
    assert report.overlap_rate == 1.0
    assert report.train_ngram_count == report.eval_ngram_count


def test_ngram_overlap_iceberg_disjoint_corpora(catalog):
    train = _make_text_table(
        catalog,
        "train_disjoint",
        ["foxtrot golf hotel india juliet kilo lima mike november oscar papa"],
    )
    evals = _make_text_table(
        catalog,
        "eval_disjoint",
        ["zulu yankee xray whiskey victor uniform tango sierra romeo quebec"],
    )

    report = ngram_overlap_iceberg(train, evals, column="note", n=8)
    assert report.overlap_rate == 0.0
    assert report.overlapping_ngrams == 0


def test_ngram_overlap_iceberg_partial_overlap_realistic(catalog):
    # A realistic-looking train-eval leakage scenario.
    train = _make_text_table(
        catalog,
        "train_realistic",
        [
            "patient discharged in stable condition after 48 hours of monitoring",
            "lab results within normal range; follow up in two weeks",
            "no acute distress, vitals stable, plan for outpatient follow up",
        ],
    )
    evals = _make_text_table(
        catalog,
        "eval_realistic",
        [
            "patient discharged in stable condition after 48 hours",  # near-duplicate of row 0
            "completely novel sentence about something unrelated to anything in train",
        ],
    )

    report = ngram_overlap_iceberg(train, evals, column="note", n=8)
    assert 0.1 < report.overlap_rate < 1.0
    assert report.is_contaminated(threshold=0.005)
    assert report.overlapping_ngrams > 0
    assert len(report.sample_overlaps) > 0


def test_ngram_overlap_iceberg_sample_is_reproducible(catalog):
    train = _make_text_table(
        catalog,
        "train_sample",
        [f"row {i} contains some text content unique to this row" for i in range(50)],
    )
    evals = _make_text_table(
        catalog,
        "eval_sample",
        [f"row {i} contains some text content unique to this row" for i in range(25, 75)],
    )

    a = ngram_overlap_iceberg(train, evals, column="note", n=6, sample=0.5, seed=42)
    b = ngram_overlap_iceberg(train, evals, column="note", n=6, sample=0.5, seed=42)
    assert a.overlap_rate == b.overlap_rate
    assert a.overlapping_ngrams == b.overlapping_ngrams
