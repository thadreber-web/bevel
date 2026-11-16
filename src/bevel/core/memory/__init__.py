"""Long-term memory.

SQLite FTS5 from the standard library for lexical recall, plus optional
brute-force cosine over ``float32`` vectors from the server's ``/embeddings``
endpoint. No numpy, no vector database -- thousands of memories brute-force in
microseconds and it keeps installation trivial on Windows and ARM64.
"""
