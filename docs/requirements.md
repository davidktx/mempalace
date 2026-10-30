# Fork requirements (davidktx/mempalace)

These are the behaviours any dependency bump, maintainer change or package
swap must preserve. `validation/run.py` checks them; see `validation/README.md`.

## Functional

| ID | Requirement | Checked by |
|----|-------------|------------|
| F1 | `mempalace mine DIR --mode convos` ingests session transcripts into a palace. | `mine_corpus` |
| F2 | `palace_query` search returns the expected passage for every golden query in the top-K (K and the hit-rate floor in `validation/config.json`). | `recall_golden` |
| F3 | The light MCP server in `--read-only` mode lists only `palace_query`, refuses `palace_exec`, and leaves palace files byte-identical (SQLite `-shm` excepted). | `readonly_mcp` |
| F4 | Concurrent mines and readers never corrupt a palace: a contending mine either succeeds or is refused by the palace lock; afterwards SQLite `integrity_check` is `ok`, nothing is quarantined, and recall is unchanged. | `concurrency` |
| F5 | Secrets matching the redaction patterns never reach palace storage. | `redaction` |
| F6 | Quarantined segment copies are capped (`MEMPALACE_QUARANTINE_KEEP`, default 1; negative = unbounded) so a crash loop cannot fill the disk. | `quarantine_cap` |
| F7 | The nightly re-mine (same command as F1 against an existing palace) is idempotent and keeps F2 passing. | `concurrency` (re-mine + recall) and `deploy` mode |
| F8 | Embeddings are computed locally with all-MiniLM-L6-v2 (ONNX); no network embedding service. | `no_telemetry_or_torch`, code review |
| F9 | No telemetry: posthog / analytics modules are never imported. | `no_telemetry_or_torch` |
| F10 | No server mode: nothing listens on a network port; MCP is stdio only. | code review, `readonly_mcp` uses stdio |

## Non-functional

- **N1 Security:** Snyk Open Source and Snyk Code issue counts must not
  increase versus the recorded baseline (`validation/run.py deps --baseline`).
  No secrets in logs, command arguments or committed files.
- **N2 Provenance:** no models or dependencies from Chinese-origin
  maintainers (e.g. BGE, Qwen, Milvus/Zilliz). Prefer US-maintained projects.
- **N3 Configuration:** no hard-coded numbers other than 0, 1 and infinity;
  every tunable comes from config or environment with a documented default.
- **N4 Deployability:** a candidate install passes `validation/run.py deploy`
  against a *copy* of the production palace before it replaces Amelia's venv.

## Known gaps (2026-10-30)

- **Chromadb backend violates F3.** Every read-only session rewrites
  `chroma.sqlite3` and the HNSW segment files (chromadb persists its index on
  open/close). The `sqlite_exact` backend passes F3.
- 19 Snyk OSS issues remain in chromadb (and torch/transformers, which only
  the optional `embeddinggemma2` / `multimodal` / `image` extras pull in).
