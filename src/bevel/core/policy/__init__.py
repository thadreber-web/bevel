"""Tiered policy engine and the approval protocol.

Seven risk tiers from READ (free) to EXEC_DESTRUCTIVE (always ask, typed
confirmation). Every decision is logged together with the rule that produced
it, so the TUI can explain itself.
"""
