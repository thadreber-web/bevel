"""Sandbox backends -- **not implemented. This package is empty until m12.**

Nothing here enforces anything. The only containment that runs today is the T0
path jail in :mod:`bevel.core.tools.base` and the rule engine in
:mod:`bevel.core.policy`. ``bevel doctor`` reports what your *operating system*
could enforce, which is not what bevel does.

The tiers below are the design m12 has to honour, not a description of this
package:

* **T0 policy-only** -- path jail plus the rule engine. Always on. **Ships.**
* **T1 in-process kernel** -- Linux Landlock + seccomp + rlimits; Windows
  AppContainer + restricted token + low integrity level + Job Object.
  Pure ``ctypes``, nothing to install.
* **T2 OS container** -- ``bwrap`` if already present, or Windows Sandbox via a
  generated ``.wsb``. Opt-in through ``--untrusted-repo`` only: hypervisor
  isolation is stronger but costs a 10-30s cold start and is disposable by
  design, which is the wrong shape for per-tool-call execution.

m12 also owes the self-test that makes the UI honest: it reports the level it
*verified* by attempting a forbidden operation and watching it get denied, never
the level that was requested. Until then there is no such test and no badge.
"""
