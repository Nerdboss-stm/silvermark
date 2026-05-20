"""Tests for MinHash signatures and Jaccard estimation."""

import numpy as np
import pytest

from silvermark.dedup import minhash


def test_shingle_word_level():
    s = "the quick brown fox jumps over the lazy dog"
    sh = minhash.shingle(s, k=3)
    assert "the quick brown" in sh
    assert "over the lazy" in sh
    assert len(sh) == 7  # 9 words, k=3 -> 7 shingles


def test_shingle_char_level_for_idless_strings():
    sh = minhash.shingle("abcdef", k=3)
    assert sh == {"abc", "bcd", "cde", "def"}


def test_shingle_short_text_returns_singleton():
    assert minhash.shingle("hi there", k=5) == {"hi there"}


def test_shingle_empty():
    assert minhash.shingle("", k=3) == set()


def test_identical_text_jaccard_one():
    shingles = minhash.shingle("the quick brown fox jumps over", k=3)
    sig_a = minhash.minhash_signature(shingles, num_perm=128, seed=42)
    sig_b = minhash.minhash_signature(shingles, num_perm=128, seed=42)
    assert minhash.jaccard_estimate(sig_a, sig_b) == 1.0


def test_disjoint_text_jaccard_near_zero():
    a = minhash.shingle("the quick brown fox jumps", k=3)
    b = minhash.shingle("xyzzy plugh hodor garply spam", k=3)
    sig_a = minhash.minhash_signature(a, num_perm=256, seed=42)
    sig_b = minhash.minhash_signature(b, num_perm=256, seed=42)
    # MinHash is an estimator; tolerance for disjoint sets at 256 perms.
    assert minhash.jaccard_estimate(sig_a, sig_b) < 0.05


def test_partial_overlap_estimator_within_tolerance():
    a = {f"shingle_{i}" for i in range(100)}
    b = {f"shingle_{i}" for i in range(50, 150)}  # 50/150 overlap = 1/3
    sig_a = minhash.minhash_signature(a, num_perm=512, seed=7)
    sig_b = minhash.minhash_signature(b, num_perm=512, seed=7)
    estimate = minhash.jaccard_estimate(sig_a, sig_b)
    # True Jaccard is 50/150 = 0.3333. With 512 perms, stderr ~0.022.
    assert abs(estimate - 0.3333) < 0.08


def test_signature_shape_mismatch_raises():
    a = minhash.minhash_signature({"x"}, num_perm=64, seed=1)
    b = minhash.minhash_signature({"x"}, num_perm=128, seed=1)
    with pytest.raises(ValueError, match="signature shapes differ"):
        minhash.jaccard_estimate(a, b)


def test_lsh_bands_split():
    sig = np.arange(128, dtype=np.uint64)
    bands = minhash.lsh_bands(sig, bands=16, rows=8)
    assert len(bands) == 16
    assert all(isinstance(b, bytes) for b in bands)
    assert len(bands[0]) == 8 * 8  # 8 uint64 rows = 64 bytes


def test_lsh_bands_bad_split_raises():
    sig = np.arange(128, dtype=np.uint64)
    with pytest.raises(ValueError, match="must equal"):
        minhash.lsh_bands(sig, bands=10, rows=8)


def test_empty_shingles_returns_max_signature():
    sig = minhash.minhash_signature(set(), num_perm=64, seed=1)
    assert sig.shape == (64,)
    # All positions should be the sentinel "infinity" value used during init.
    assert (sig == sig[0]).all()
    assert sig[0] >= (1 << 30)  # something large enough to be unreachable


def test_readme_quickstart_estimate_is_close_to_true_jaccard():
    """The README claims this exact code produces an estimate near 0.7.
    If we change the algorithm or the constants, this test breaks loudly
    so we remember to update the README."""
    a_sh = minhash.shingle(
        "the patient was discharged in stable condition after 48 hours", k=3
    )
    b_sh = minhash.shingle(
        "patient was discharged in stable condition after 48 hours of observation", k=3
    )
    true_jaccard = len(a_sh & b_sh) / len(a_sh | b_sh)
    assert true_jaccard == pytest.approx(0.7, abs=0.01)

    a = minhash.minhash_signature(a_sh, num_perm=128, seed=1)
    b = minhash.minhash_signature(b_sh, num_perm=128, seed=1)
    estimate = minhash.jaccard_estimate(a, b)
    # 128 perms has stderr ~0.045 on a true Jaccard of 0.7. Allow 3x stderr.
    assert estimate == pytest.approx(true_jaccard, abs=0.15)
