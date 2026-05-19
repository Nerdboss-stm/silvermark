"""Near-duplicate detection via MinHash-LSH."""

from silvermark.dedup.minhash import (
    jaccard_estimate,
    lsh_bands,
    minhash_signature,
    shingle,
)

__all__ = ["shingle", "minhash_signature", "jaccard_estimate", "lsh_bands"]
