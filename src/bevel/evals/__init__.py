"""Headless eval harness.

TOML task specs run against a fixture repository with a seeded bug and failing
tests. Assertions cover test outcomes, file contents, diff must/must-not
patterns, permitted tool-call sequences, and token ceilings. CI runs against
``MockProvider`` so it needs neither a GPU nor an API key.
"""
