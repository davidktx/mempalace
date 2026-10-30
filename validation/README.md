# Validation harness

One entrypoint, three modes. Defaults live in `validation/config.json`; any key
can be overridden with `VALIDATION_<KEY>` (e.g. `VALIDATION_BACKEND=sqlite_exact`,
`VALIDATION_MIN_HIT_RATE=1`). Requirements it enforces: `docs/requirements.md`.

```bash
# Functional back-tests on the synthetic corpus (repo checkout)
uv run --frozen python validation/run.py functional

# Gate an install before deploying: copies the palace to a temp dir first,
# never touches the original
python3 validation/run.py deploy --palace /path/to/palace --install /path/to/venv

# Dependency-swap check: pytest + functional + Snyk OSS (uv export) + Snyk Code,
# diffed against a previous run
uv run --frozen python validation/run.py deps --baseline ../mempalace-validation/<stamp>
```

| Key | Default | Meaning |
|-----|---------|---------|
| backend | "" (package default) | storage backend under test |
| top_k | 5 | results per golden query |
| min_hit_rate | 1 | fraction of golden queries that must hit |
| concurrent_readers | 4 | MCP reader processes during the concurrency test |
| reader_rounds | 5 | queries per reader |
| concurrent_mines | 2 | simultaneous mines (lock refusals are allowed) |
| timeout_seconds | 600 | per subprocess |
| quarantine_keep | 1 | cap verified by `quarantine_cap` |

`VALIDATION_UV_EXPORT_ARGS` sets the extras for the Snyk OSS export (default
`--all-extras`; use an empty string for core-only). Snyk steps are skipped when
the `snyk` CLI is not authenticated. Exit code 0 means every check passed;
`--report FILE` writes JSON.

Corpus (`corpus/`) and golden answers (`golden/recall.json`) are synthetic;
the "secrets" in `s06-work.txt` are fake.
