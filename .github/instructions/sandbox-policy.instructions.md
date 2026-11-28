---
applyTo: "src/bevel/core/sandbox/**/*.py,src/bevel/_capability.py,src/bevel/core/policy/**/*.py"
---

# Sandbox and policy code

This is the part of bevel that earns trust. The failure mode to avoid is not a crash — it's **claiming protection that isn't there**.

## The honesty rule

The UI reports the enforcement level that was **verified**, never the level that was requested or configured. `sandbox: bwrap` must mean a self-test actually spawned a real child and watched the **kernel** deny it a forbidden write and a forbidden connection. If it didn't, the badge says `sandbox: policy-only`.

Related: probes in `_capability.py` are **read-only and side-effect free**. They answer "would the OS honour this?" They never enforce and never mutate. Enforcement belongs in a backend, and verification belongs in `selftest.py`. Keep those three roles separate — it's what makes `bevel doctor` usable before m12 lands.

## Tiers

| Tier | Linux | Windows |
|---|---|---|
| **T0** policy-only | path jail + rule engine (always on, both platforms) | same |
| **T1** bwrap | spawn the tool through a `bwrap` argv prefix: `--ro-bind / /`, `--bind <workspace> <workspace>`, `--dev`, `--proc`, `--unshare-net`, `--unshare-user`, `--die-with-parent` | none — `bevel doctor` says so and names WSL2 |

There is no T2, and there is no `ctypes` syscall ladder. The Landlock/seccomp/AppContainer/Windows-Sandbox design this file used to specify was **removed from the plan before any of it was built**: it was the plan's own highest stall risk (`ctypes` probes fail silently), no shipped harness implements the Windows half, and the cheap path — bevel builds argv, `bwrap` enforces, the same division of labour as `gh` in m11b — is verified working on the Spark. Do not revive it. If a Windows kernel boundary is ever wanted, it arrives as a new slice with its own oracle.

`bwrap` is a user-installed distro package, joining `git` and `gh` as things bevel probes for rather than assumes. Bevel itself installs nothing and adds no runtime dependency.

## Rules for this code

- **Probe by running, never by searching.** `shutil.which("bwrap")` alone is a forbidden shortcut. On Ubuntu 24.04 *installed is not working*: the `uidmap` helpers (`newuidmap`/`newgidmap`) and an AppArmor profile granting `/usr/bin/bwrap` the `userns` permission are both required, and each has failed independently on the Spark while bwrap itself was "installed". The probe runs `bwrap --ro-bind / / --unshare-user --unshare-net -- /bin/echo ok` and requires exit 0 with the expected output.
- **Invoke bwrap by resolved path.** The AppArmor grant is by path, so a second copy earlier on `PATH` is not covered by it and will fail. Resolve, then use that path.
- **Degrade, never fail open — and never degrade silently.** Every tier transition goes down the ladder with a recorded reason, and every degrade is **loud**: the badge reads `sandbox: policy-only`, `bevel doctor` prints the probe command verbatim with its exit code and stderr, and the message names the missing prerequisite and the exact fix (`sudo apt install bwrap`, `sudo apt install uidmap`, the AppArmor profile path). Match the register of `gh auth login` in m11b and the missing `user.name` in the git tool. The old plan's "silently degrades to T0 and the badge says so" was self-contradicting; silence is the failure this rule exists to prevent.
- A probe that **errors** and a probe that finds bwrap **absent** are different cases — keep them distinct (`Capability.error`) and treat both as unavailable.
- **The self-test proves denial by the kernel, not by bevel.** Spawn a real child through the bwrap prefix and assert that `touch` outside the workspace and an outbound `connect` are refused by the kernel. A Python-side refusal that never reached the boundary proves nothing — that path is already covered by `test_tools_base.py`. If the probe passed but the self-test's write succeeded, the probe lied: drop to `POLICY_ONLY` and say so loudly.
- **Fail closed on ambiguity.** If you cannot determine whether an operation is permitted, it is not permitted.
- **No CPU/memory caps in this slice.** `bwrap` does not provide them and `setrlimit` in a `preexec_fn` is out of scope; the timeout and tree-kill remain the backstop. The module says so rather than letting the tier name imply protection that isn't there.
- **No `--untrusted-repo` flag, no reused container, no Windows Sandbox `.wsb`.** All three were attached to the deleted tier ladder. Declaring a flag the current milestone doesn't read is the failure mode AGENTS.md names.

## Policy engine

Seven tiers, and the behaviour is fixed by the tier — don't invent new semantics locally:

`READ` free · `WRITE_WORKSPACE` ask (auto in auto mode) · `WRITE_OUTSIDE` always ask · `EXEC_SAFE` free once allowlisted · `EXEC_UNSAFE` ask · `EXEC_DESTRUCTIVE` always ask + typed confirmation · `NETWORK_EGRESS` ask (Linux: enforced by `bwrap --unshare-net` once m12 lands).

The one exception to `WRITE_OUTSIDE` always-ask is `[memory] auto_accept = true`, which exempts the `remember` tool by name. It is not a general bypass and must not become one: it is read off the `MemoryStore` on the `ToolContext`, so a run with memory off cannot inherit it; it matches `remember` alone, so a second `WRITE_OUTSIDE` tool still asks; and it is checked *after* the path jail, so a call with an escaping path still asks. It exists because learned rules are keyed on argv and `remember` has none, so `a` at the prompt has nothing to learn. Do not extend it to other tools, and do not make it the default.

- Every decision is logged with **the rule that produced it**. "allowed" without a reason is a bug.
- Learned allowlist entries persist to `policy.db` with a timestamp and a hash. A learned rule must be inspectable and revocable.
- `EXEC_DESTRUCTIVE` requires *typed* confirmation, not a yes/no keypress.
- Checkpoints run even under `--yolo` (the m05 auto-approve flag). Auto-approval governs what the agent may do, not whether it can be undone.
- Path jail comparisons go through `Path.resolve()` before checking containment, and must handle symlink escape, `..` traversal, UNC paths, drive-letter ambiguity, and 8.3 short names on Windows.
