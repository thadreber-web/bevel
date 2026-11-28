"""Sandbox backends: bevel builds the argv, the kernel does the enforcing.

m12a ships exactly one backend. The old Landlock/seccomp/AppContainer/
Windows-Sandbox ``ctypes`` ladder was removed from the plan before any of it
was built, and this package does not revive it: there is no tier enum with
slots for backends that do not exist, no syscall probes, no ``.wsb`` files.

* :mod:`bevel.core.sandbox.base` -- the two enforcement levels that exist
  (``POLICY_ONLY``, ``BWRAP``) and the
  :class:`~bevel.core.sandbox.base.SandboxStatus` record the UI reports.
* :mod:`bevel.core.sandbox.bwrap` -- the bwrap argv prefix, the probe that
  **runs** bwrap (installed != working on Ubuntu 24.04), and the degrade
  ladder.
* :mod:`bevel.core.sandbox.selftest` -- the oracle: a real child denied a
  forbidden write and a forbidden connect by a real kernel.

What enforces what: the T0 path jail (:mod:`bevel.core.tools.base`) and the
rule engine (:mod:`bevel.core.policy`) are always on, both platforms. The
bwrap backend adds kernel enforcement on Linux when the probe and self-test
both pass; when they do not, the level is ``POLICY_ONLY`` and the reason is
rendered loudly -- the TUI badge and ``bevel doctor`` print the probe
command, its exit code, its stderr, and the named fix. Windows gets no new
behavior until m12b makes the Windows decision (T0 + a WSL2 pointer).
"""
