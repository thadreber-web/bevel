"""Sandbox backends.

Three tiers, selected by capability probe and confirmed by self-test:

* **T0 policy-only** -- path jail plus the rule engine. Always on.
* **T1 in-process kernel** -- Linux Landlock + seccomp + rlimits; Windows
  AppContainer + restricted token + low integrity level + Job Object.
  Pure ``ctypes``, nothing to install.
* **T2 OS container** -- ``bwrap`` if already present, or Windows Sandbox via a
  generated ``.wsb``. Opt-in through ``--untrusted-repo`` only: hypervisor
  isolation is stronger but costs a 10-30s cold start and is disposable by
  design, which is the wrong shape for per-tool-call execution.

The UI always reports the level that was *verified*, never the level that was
requested.
"""
