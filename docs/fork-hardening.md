# Fork hardening settings (davidktx/mempalace)

| Variable | Default | Effect |
|---|---|---|
| `MEMPALACE_READ_ONLY_OPEN` | `0` (auto `1` with `--read-only` / `MEMPALACE_MCP_READ_ONLY=1`) | Skip every pre-open repair/rename/quarantine pass so a read-only process never modifies palace files. Repairs happen in the writer, under `palace_db_lock`. |
| `MEMPALACE_QUARANTINE_KEEP` | `1` | Older quarantined copies (`*.drift-*`, `*.corrupt-*`) kept per segment; oldest pruned before each new quarantine. Negative = unbounded (upstream behaviour). `0` = keep none. |
| `MEMPALACE_REDACT` | `1` | Redact secrets in transcripts before chunking/indexing. |
| `MEMPALACE_REDACT_REPLACEMENT` | `[REDACTED]` | Replacement text. |
| `MEMPALACE_REDACT_PATTERNS_FILE` | unset | Extra literal strings to redact, one per line (matched literally, not as regexes, to rule out reDOS). |
| `MEMPALACE_REDACT_DISABLE_DEFAULTS` | `0` | Use only the patterns file. |

Default redaction patterns (`mempalace/redact.py` `DEFAULT_PATTERNS`): PEM private-key blocks, AWS access key IDs, GitHub tokens, Slack tokens, `sk-` style API keys, Google API keys, Stripe keys, JWTs, `Bearer` headers, URL credentials (`scheme://user:pass@`), and `password|secret|api_key|token|client_secret|private_key = value` assignments (label kept).

Redaction applies to new mining only; existing drawers need a re-mine. Chromadb 1.5.9 still opens its SQLite file read-write even in read-only mode; this fork only stops MemPalace's own repairs from writing.
Only files owned by you and not group/world-writable are loaded as patterns files.
