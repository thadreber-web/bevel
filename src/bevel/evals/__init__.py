"""Headless eval harness -- **not implemented. This package is empty until m13.**

``bevel eval`` exits non-zero and says so. The plan: TOML task specs run against a
fixture repository with a seeded bug and failing tests. Assertions cover test
outcomes, file contents, diff must/must-not patterns, permitted tool-call
sequences, and token ceilings. CI runs against ``MockProvider`` so it needs
neither a GPU nor an API key. ``tests/evals/`` is empty for the same reason.
"""
