"""n-gram contamination between two text corpora.

The check is: what fraction of n-grams in the eval corpus also appear in the
train corpus? Anything above ~0.5% is usually a leakage signal worth
investigating.

This is the baseline check that does not require an embedding model or any
fine-tuned judge. NVIDIA's "Mastering LLM Techniques: Text Data Processing"
walks through the same algorithm; LSHBloom (arXiv 2411.04257) is the
internet-scale variant. We do not implement the bloom-filter optimization
yet because most lakehouse eval sets are small enough that the naive set
intersection is fine.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field


def ngrams(text: str, n: int = 8) -> set[str]:
    """Character n-grams of `text`.

    Returns an empty set if the text is shorter than n. We use character
    n-grams (not word n-grams) because lakehouse text columns mix natural
    language and tokenless strings (URLs, error messages, codes) and
    character n-grams handle both.
    """
    if not text or len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def column_to_ngrams(
    values: Iterable[str | None],
    n: int = 8,
    sample: float = 1.0,
    seed: int | None = None,
) -> set[str]:
    """Union of n-grams across a column of strings, optionally sampled.

    For a 100M-row table, full extraction can OOM. Pass sample=0.01 to use
    1 in 100 rows. Sampling is per-row with a fixed seed; results are
    reproducible across runs.
    """
    if not 0.0 < sample <= 1.0:
        raise ValueError(f"sample must be in (0, 1], got {sample}")

    rng = random.Random(seed) if seed is not None else random.Random()
    result: set[str] = set()
    for v in values:
        if v is None:
            continue
        if sample < 1.0 and rng.random() > sample:
            continue
        result.update(ngrams(str(v), n))
    return result


@dataclass
class ContamReport:
    """Result of n-gram overlap between train and eval corpora.

    overlap_rate is (eval n-grams that also appear in train) / (eval n-grams).
    Near zero means clean. Above 0.005 typically indicates leakage.
    The threshold depends on domain; the default `is_contaminated()` uses
    0.005 as a starting point.
    """

    train_ngram_count: int
    eval_ngram_count: int
    overlapping_ngrams: int
    overlap_rate: float
    sample_overlaps: list[str] = field(default_factory=list)
    n: int = 8

    def is_contaminated(self, threshold: float = 0.005) -> bool:
        return self.overlap_rate > threshold


def ngram_overlap(
    train: Iterable[str | None],
    eval: Iterable[str | None],
    n: int = 8,
    sample: float = 1.0,
    seed: int = 1,
    examples: int = 5,
) -> ContamReport:
    """n-gram overlap between two text corpora.

    Pass iterables of strings, one for the train side and one for the eval
    side. Returns a ContamReport with the rate and a few example matches
    for manual inspection.

    For Iceberg-table inputs (passing table identifiers like
    "warehouse.silver.train_v3"), v0.1 adds a wrapper that reads with
    pyiceberg and forwards to this function. For v0 you load the data
    yourself with pyiceberg or duckdb and pass the columns in.
    """
    train_ngs = column_to_ngrams(train, n=n, sample=sample, seed=seed)
    eval_ngs = column_to_ngrams(eval, n=n, sample=sample, seed=seed)

    intersection = train_ngs & eval_ngs
    overlap_rate = len(intersection) / len(eval_ngs) if eval_ngs else 0.0

    sample_overlaps = sorted(intersection)[:examples] if examples else []

    return ContamReport(
        train_ngram_count=len(train_ngs),
        eval_ngram_count=len(eval_ngs),
        overlapping_ngrams=len(intersection),
        overlap_rate=overlap_rate,
        sample_overlaps=sample_overlaps,
        n=n,
    )
