# Shared Engineering Standards (Antigravity, Claude, Devin, GStack)

1. **Architecture & Decisions:** Record every non-trivial design choice in `docs/adr/` using `docs/adr/000-template.md`.
2. **Code Quality & Static Analysis:**
   - **Python:** Python 3.12+, `mypy --strict`, `ruff check`, Pydantic v2. Never use bare `except Exception: pass`.
   - **Rust:** `rustfmt`, `cargo clippy -- -D warnings`, `cargo audit`. Never use `.unwrap()` or `.expect()` in production paths.
   - **Go:** `gofmt`, `go vet ./...`, `go test -race ./...`. Always check returned errors explicitly.
3. **Testing & Verification:** Every PR must include deterministic unit tests. Never claim benchmarks without running them.
4. **Security:** Never commit `.env` files, API keys, or PII/PHI fixtures.
