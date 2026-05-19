"""Tests for n-gram contamination detection."""

import pytest

from silvermark.contamination import ngram


def test_ngrams_basic():
    result = ngram.ngrams("hello world", n=3)
    assert "hel" in result
    assert "llo" in result
    assert "lo " in result
    assert "o w" in result
    # "hello world" has 11 chars, so 11 - 3 + 1 = 9 trigrams
    assert len(result) == 9


def test_ngrams_short_text_returns_empty():
    assert ngram.ngrams("hi", n=8) == set()


def test_ngrams_empty_string():
    assert ngram.ngrams("", n=3) == set()


def test_ngrams_exact_length():
    result = ngram.ngrams("abcdef", n=6)
    assert result == {"abcdef"}


def test_column_to_ngrams_combines():
    values = ["hello", "world"]
    result = ngram.column_to_ngrams(values, n=3)
    assert "hel" in result
    assert "wor" in result


def test_column_to_ngrams_skips_none():
    values = ["hello", None, "world"]
    result = ngram.column_to_ngrams(values, n=3)
    assert "hel" in result
    assert "wor" in result


def test_column_to_ngrams_handles_non_string():
    # column_to_ngrams should coerce non-string values via str()
    values = [42, 3.14]
    result = ngram.column_to_ngrams(values, n=2)
    assert "42" in result
    assert "3." in result


def test_column_to_ngrams_sample_reproducible():
    values = [f"text_{i}_extra_padding" for i in range(100)]
    a = ngram.column_to_ngrams(values, n=5, sample=0.3, seed=42)
    b = ngram.column_to_ngrams(values, n=5, sample=0.3, seed=42)
    assert a == b


def test_column_to_ngrams_sample_reduces_count():
    values = [f"text_{i}_extra_padding" for i in range(200)]
    full = ngram.column_to_ngrams(values, n=5, sample=1.0)
    sampled = ngram.column_to_ngrams(values, n=5, sample=0.1, seed=42)
    assert len(sampled) < len(full)


def test_column_to_ngrams_invalid_sample_raises():
    with pytest.raises(ValueError, match="sample must be"):
        ngram.column_to_ngrams(["x"], sample=0.0)
    with pytest.raises(ValueError, match="sample must be"):
        ngram.column_to_ngrams(["x"], sample=1.5)


def test_identical_corpora_full_overlap():
    text = ["the quick brown fox jumps over the lazy dog"]
    report = ngram.ngram_overlap(train=text, eval=text, n=5)
    assert report.overlap_rate == 1.0
    assert report.is_contaminated()


def test_disjoint_corpora_zero_overlap():
    # 8-gram is long enough that these never collide
    train = ["aaaaaaaaaaaaaaaaaa"]
    eval_data = ["zzzzzzzzzzzzzzzzzz"]
    report = ngram.ngram_overlap(train=train, eval=eval_data, n=8)
    assert report.overlap_rate == 0.0
    assert not report.is_contaminated()


def test_partial_overlap_is_partial():
    train = ["the quick brown fox jumps over the lazy dog"]
    eval_data = ["the quick brown horse runs through the meadow"]
    report = ngram.ngram_overlap(train=train, eval=eval_data, n=5)
    assert 0.0 < report.overlap_rate < 1.0


def test_contam_report_threshold_default():
    report = ngram.ContamReport(
        train_ngram_count=1000,
        eval_ngram_count=500,
        overlapping_ngrams=10,
        overlap_rate=0.02,
        n=8,
    )
    assert report.is_contaminated() is True
    assert report.is_contaminated(threshold=0.05) is False


def test_contam_report_threshold_zero_overlap():
    report = ngram.ContamReport(
        train_ngram_count=1000,
        eval_ngram_count=500,
        overlapping_ngrams=0,
        overlap_rate=0.0,
        n=8,
    )
    assert report.is_contaminated() is False


def test_overlap_returns_sample_examples():
    train = ["foobar baz"]
    eval_data = ["foobar qux"]
    report = ngram.ngram_overlap(train=train, eval=eval_data, n=3, examples=3)
    assert len(report.sample_overlaps) > 0
    assert len(report.sample_overlaps) <= 3


def test_overlap_examples_disabled():
    train = ["hello world"]
    eval_data = ["hello world"]
    report = ngram.ngram_overlap(train=train, eval=eval_data, n=3, examples=0)
    assert report.sample_overlaps == []


def test_empty_corpora():
    report = ngram.ngram_overlap(train=[], eval=[], n=8)
    assert report.overlap_rate == 0.0
    assert report.train_ngram_count == 0
    assert report.eval_ngram_count == 0
    assert not report.is_contaminated()


def test_empty_eval_does_not_divide_by_zero():
    report = ngram.ngram_overlap(train=["hello world"], eval=[], n=3)
    assert report.overlap_rate == 0.0
    assert report.eval_ngram_count == 0


def test_realistic_contamination_scenario():
    # eval set has a few rows that appear verbatim in training - the kind of
    # leakage that quietly inflates eval scores. We expect overlap_rate to
    # be high enough that is_contaminated() trips.
    train = [
        "The patient presented with shortness of breath and elevated heart rate.",
        "Discharged in stable condition after 48 hours of observation.",
        "Lab results indicate normal kidney function.",
        "Recommended follow-up in two weeks for routine checkup.",
    ]
    eval_corpus = [
        "Patient was admitted to the ER on Tuesday morning.",
        # this next one is verbatim from train, leaked
        "Discharged in stable condition after 48 hours of observation.",
        "Vital signs were monitored continuously during the stay.",
    ]
    report = ngram.ngram_overlap(train=train, eval=eval_corpus, n=8)
    assert report.is_contaminated()
    assert report.overlapping_ngrams > 0
