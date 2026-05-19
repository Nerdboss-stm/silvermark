"""MinHash signatures and LSH bucketing.

Standard integer MinHash with a random permutation family (a*x + b) mod p.
Signatures collapse a set of shingles into a fixed-length vector where
the expected fraction of equal positions equals the Jaccard similarity
of the two sets. Banded LSH then buckets signatures so candidate pairs
can be found in sub-quadratic time.

Reference: Leskovec, Rajaraman, Ullman, *Mining of Massive Datasets*, ch 3.
"""

from __future__ import annotations

import mmh3
import numpy as np

# Mersenne prime M31 = 2^31 - 1 = 2147483647. The prime has to be < 2^32
# so that (a * h + b) stays within uint64 without overflow when both a and h
# are reduced mod p. A 2^61 prime overflows; tests will catch this.
_PRIME = np.uint64((1 << 31) - 1)
_MAX_HASH = np.uint64((1 << 31) - 1)


def shingle(text: str, k: int = 9) -> set[str]:
    """k-shingles for similarity.

    For text with whitespace, shingles are word-level (good for natural
    language and log lines). For tokenless strings, shingles are
    character-level (good for IDs, URLs, code fragments).
    """
    if not text:
        return set()
    if " " in text:
        words = text.split()
        if len(words) < k:
            return {" ".join(words)}
        return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def minhash_signature(
    shingles: set[str], num_perm: int = 128, seed: int = 1
) -> np.ndarray:
    """MinHash signature of a shingle set.

    Returns a uint64 array of length ``num_perm``. Two signatures with the
    same ``num_perm`` and ``seed`` can be compared position-wise; the
    fraction of equal positions is an estimate of Jaccard.
    """
    rng = np.random.default_rng(seed)
    p = int(_PRIME)
    a = rng.integers(1, p, size=num_perm, dtype=np.int64).astype(np.uint64)
    b = rng.integers(0, p, size=num_perm, dtype=np.int64).astype(np.uint64)

    sig = np.full(num_perm, _MAX_HASH, dtype=np.uint64)
    if not shingles:
        return sig

    for s in shingles:
        # Reduce hash into [0, p) so the (a * h + b) multiplication stays
        # within uint64. mmh3 returns up to 2^32 - 1, which is larger than p.
        h = np.uint64(mmh3.hash(s, signed=False) % p)
        candidate = (a * h + b) % _PRIME
        sig = np.minimum(sig, candidate)
    return sig


def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimated Jaccard from two MinHash signatures of equal length."""
    if sig_a.shape != sig_b.shape:
        raise ValueError(
            f"signature shapes differ: {sig_a.shape} vs {sig_b.shape}"
        )
    return float((sig_a == sig_b).mean())


def lsh_bands(sig: np.ndarray, bands: int = 16, rows: int = 8) -> list[bytes]:
    """Split a signature into LSH bands for candidate-pair bucketing.

    Each band is the byte representation of ``rows`` consecutive signature
    positions. Two signatures share a candidate-pair bucket if they share
    any band exactly. Choose (bands, rows) so that ``bands * rows == num_perm``
    and the implied s-curve threshold ``(1/bands)**(1/rows)`` matches the
    Jaccard cutoff you want.
    """
    if bands * rows != sig.size:
        raise ValueError(
            f"bands * rows ({bands * rows}) must equal signature length ({sig.size})"
        )
    return [sig[i * rows : (i + 1) * rows].tobytes() for i in range(bands)]
