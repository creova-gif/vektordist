## Summary

<!-- Briefly describe what this PR changes and why. -->

## Related Issues & ADRs

- Closes #<!-- issue number -->
- ADR: <!-- Link to `docs/adr/NNN-*.md` if this PR introduces a non-trivial design, protocol, or storage format decision, or `N/A` -->

## Verification & Quality Gates

- [ ] `cargo fmt --all -- --check` passes with zero diffs.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes with zero warnings.
- [ ] `cargo test --workspace` passes deterministically (using `SeededRng` for synthetic datasets/schedules).
- [ ] `cargo audit --deny warnings` passes cleanly.
- [ ] No `.unwrap()` or `.expect()` calls were added on non-test code paths (`#![deny(clippy::unwrap_used, clippy::expect_used)]`).
- [ ] Durability / security invariants verified (`acknowledged_writes_lost == 0`, CRC32 corruption rejection, `OrgId` isolation) where applicable.
- [ ] Release benchmarks (`vektordist bench` / `vektordist chaos`) executed and recorded in `docs/BENCHMARKS.md` if modifying SIMD, indexing, quantization, or network hot paths.

## Measured Results / Test Output

```text
<!-- Paste relevant test or `--release` benchmark output here -->
```
