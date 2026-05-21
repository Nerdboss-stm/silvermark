"""Iceberg-table convenience wrappers for the contamination check.

The pure-python `ngram_overlap` already does the work. This module is just
sugar so callers don't have to write the pyiceberg scan + arrow to_pylist
boilerplate every time they want to check two real tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from silvermark.contamination.ngram import (
    ContamReport,
    column_to_ngrams,
)

if TYPE_CHECKING:
    from pyiceberg.table import Table


def column_ngrams_from_table(
    table: Table,
    column: str,
    n: int = 8,
    sample: float = 1.0,
    seed: int = 1,
) -> set[str]:
    """Read a single text column of an Iceberg table and return its n-gram set.

    Uses pyiceberg projection so only the named column is materialized. For
    big tables, pass `sample` to subsample rows before n-gramming.

    Raises KeyError if the column does not exist in the table schema. (pyiceberg
    itself raises ValueError; we re-raise as KeyError so callers can write
    consistent `except KeyError` blocks regardless of pyiceberg version.)
    """
    try:
        arrow_tbl = table.scan(selected_fields=(column,)).to_arrow()
    except ValueError as exc:
        if "Could not find column" in str(exc) or "column" in str(exc).lower():
            raise KeyError(
                f"column {column!r} not found in table schema"
            ) from exc
        raise
    if arrow_tbl.num_rows == 0:
        return set()
    values = arrow_tbl.column(column).to_pylist()
    return column_to_ngrams(values, n=n, sample=sample, seed=seed)


def ngram_overlap_iceberg(
    train_table: Table,
    eval_table: Table,
    column: str,
    n: int = 8,
    sample: float = 1.0,
    seed: int = 1,
    examples: int = 5,
) -> ContamReport:
    """Compute n-gram overlap between the same column in two Iceberg tables.

    The classic train-vs-eval contamination check. Both tables must have a
    column named `column`. The function reads only that column from each
    table (no full-row materialization).
    """
    train_ngrams = column_ngrams_from_table(
        train_table, column, n=n, sample=sample, seed=seed
    )
    eval_ngrams = column_ngrams_from_table(
        eval_table, column, n=n, sample=sample, seed=seed
    )

    intersection = train_ngrams & eval_ngrams
    overlap_rate = (
        len(intersection) / len(eval_ngrams) if eval_ngrams else 0.0
    )
    sample_overlaps = sorted(intersection)[:examples] if examples else []

    return ContamReport(
        train_ngram_count=len(train_ngrams),
        eval_ngram_count=len(eval_ngrams),
        overlapping_ngrams=len(intersection),
        overlap_rate=overlap_rate,
        sample_overlaps=sample_overlaps,
        n=n,
    )


__all__ = ["column_ngrams_from_table", "ngram_overlap_iceberg"]
