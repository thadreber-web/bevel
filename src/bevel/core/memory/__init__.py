"""Long-term memory: what bevel learned in an earlier session and should not relearn.

Two modules, one job each. :mod:`~bevel.core.memory.store` owns ``memories.db``
-- typed rows, a contentless FTS5 index sharing rowids with them, and the
supersede links. :mod:`~bevel.core.memory.recall` owns retrieval and the
token-budgeted block that gets injected into a conversation.

Retrieval is two-stage because the arithmetic says it must be. A full-corpus
cosine scan in pure Python is about 115 ms at 1 000 memories and 5.4 s at
50 000; FTS5 with ``bm25`` narrows to ~100 candidates in single-digit
milliseconds at 200 000 rows, and re-ranking only those costs about 14 ms. So
lexical first, cosine second, and cosine only when the user has opted in and
the endpoint actually serves ``/embeddings``.

No numpy and no vector database, which is what keeps ``uv tool install bevel``
working on Windows and ARM64 with nothing to compile. Nothing is ever deleted
either: a corrected memory is superseded, not removed, so the chain stays
auditable.

One limitation worth knowing before trusting a recall: the index is tokenised
with ``porter unicode61``, and Porter strips suffixes without a vocabulary. It
unifies ``index``/``indexes``/``indexed`` but not ``broke``/``broken``/``break``.
"""
