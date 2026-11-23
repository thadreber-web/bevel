---
applyTo: "src/bevel/core/sandbox/**/*.py,src/bevel/_capability.py,src/bevel/core/policy/**/*.py"
---

# Sandbox and policy code

This is the part of bevel that earns trust. The failure mode to avoid is not a crash — it's **claiming protection that isn't there**.

## The honesty rule

The UI reports the enforcement level that was **verified**, never the level that was requested or configured. `sandbox: landlock+seccomp` must mean a self-test actually attempted a forbidden operation and watched it get denied. If it didn't, the badge says `sandbox: policy-only`.

Related: probes in `_capability.py` are **read-only and side-effect free**. They answer "would the OS honour this?" They never enforce, never mutate, never create a ruleset or a profile. Enforcement belongs in a backend, and verification belongs in `selftest.py`. Keep those three roles separate — it's what makes `bevel doctor` usable before m12 lands.

## Tiers

| Tier | Linux | Windows |
|---|---|---|
| **T0** policy-only | path jail + rule engine (always on, both platforms) | same |
| **T1** in-process kernel | Landlock (ABI v4 → kernel's actual ABI) + seccomp-bpf + `setrlimit` | AppContainer (`userenv!CreateAppContainerProfile` + `kernelbase!CreateAppContainerToken`) + low integrity level + Job Object |
| **T2** OS container | `bwrap` **if already present** | Windows Sandbox via generated `.wsb` |

T1 must need **nothing installed** — pure `ctypes`, no changes to the host. This is a hard user requirement (a DGX Spark that must not be modified), not an aesthetic preference.

T2 is **opt-in only**, behind the `--untrusted-repo` flag (added by m12, alongside the other CLI flags — do not pre-declare flags the current milestone doesn't read). Do not promote it to a default or a per-profile setting, and do not build a long-lived reused container to hide the cold start — reuse is exactly what destroys the disposable guarantee that makes it worth having.

## Rules for this code

- **Degrade, never fail open.** Every tier transition goes down the ladder with a recorded reason. A probe that errors and a probe that finds the feature absent are different cases — keep them distinct (`Capability.error`) and treat both as unavailable.
- **Fail closed on ambiguity.** If you cannot determine whether an operation is permitted, it is not permitted.
- Landlock: request ABI v4, then mask the ruleset to the version the kernel actually reports. Never assume v4.
- seccomp: default action `SCMP_ACT_ERRNO`, allowlist what's needed. Never `SCMP_ACT_ALLOW` default with a deny list — a new syscall becomes an escape.
- `prctl(PR_SET_NO_NEW_PRIVS, 1)` before `seccomp()` or the filter install is refused by the kernel.
- AppContainer: the exports you need are `CreateAppContainerProfile` in **`userenv.dll`** and `CreateAppContainerToken` in **`kernelbase.dll`** — confirmed by parsing PE export tables on a real Win11 build. They are *not* in `ole32`, and `kernel32` does not forward `CreateAppContainerToken`, so probing either of those gives a false negative that would silently degrade a capable machine to T0. The `api-ms-win-core-appcontainer-*` API-set DLLs are not files on disk, so `CDLL` on those names fails with error 126. When you need to confirm an export, parse the export table from the file bytes — `LoadLibraryExW` with `DONT_RESOLVE_DLL_REFERENCES` does not resolve forwarders and reports false negatives.
- AppContainer: granting the `S-1-15-2-…` SID mutates ACLs on allowed paths. Scope grants to the workspace plus explicitly allowed paths, tag the ACE so it's identifiable, revoke on clean exit, and provide `bevel sandbox cleanup` to recover after a crash. It is on by default **only** because the self-test gates it — if the probe fails, degrade to T0 silently and say so.
- Windows Sandbox: `WindowsSandbox.exe` is staged on capable SKUs even when the `Containers-DisposableClientVM` feature is **off**, so the executable's existence proves nothing. Probe the `WindowsSandboxService` registry key instead, and report tier 2 only when the service is actually installed.
- Windows Sandbox mapped folders are *not* protected by the VM. The VM protects everything else. Document this; brokered shares also have weaker symlink/locking/ACL fidelity, so never route ordinary editing through them.
- `setrlimit` / Job Object caps must be set **before** the child starts, not after — a cap applied post-launch is a cap the child can race past.
- Self-tests spawn a real child that attempts the forbidden operation. Assert denial. If it succeeds, drop a tier. Never simulate the denial.

## Policy engine

Seven tiers, and the behaviour is fixed by the tier — don't invent new semantics locally:

`READ` free · `WRITE_WORKSPACE` ask (auto in auto mode) · `WRITE_OUTSIDE` always ask · `EXEC_SAFE` free once allowlisted · `EXEC_UNSAFE` ask · `EXEC_DESTRUCTIVE` always ask + typed confirmation · `NETWORK_EGRESS` ask.

The one exception to `WRITE_OUTSIDE` always-ask is `[memory] auto_accept = true`, which exempts the `remember` tool by name. It is not a general bypass and must not become one: it is read off the `MemoryStore` on the `ToolContext`, so a run with memory off cannot inherit it; it matches `remember` alone, so a second `WRITE_OUTSIDE` tool still asks; and it is checked *after* the path jail, so a call with an escaping path still asks. It exists because learned rules are keyed on argv and `remember` has none, so `a` at the prompt has nothing to learn. Do not extend it to other tools, and do not make it the default.

- Every decision is logged with **the rule that produced it**. "allowed" without a reason is a bug.
- Learned allowlist entries persist to `policy.db` with a timestamp and a hash. A learned rule must be inspectable and revocable.
- `EXEC_DESTRUCTIVE` requires *typed* confirmation, not a yes/no keypress.
- Checkpoints run even under `--yolo` (the m05 auto-approve flag). Auto-approval governs what the agent may do, not whether it can be undone.
- Path jail comparisons go through `Path.resolve()` before checking containment, and must handle symlink escape, `..` traversal, UNC paths, drive-letter ambiguity, and 8.3 short names on Windows.
