"""Redact secrets from transcript text before it is indexed.

Transcripts often contain pasted tokens, passwords and keys. Anything mined
into the palace is stored verbatim and returned by search, so secrets are
replaced *before* chunking/embedding.

Configuration (environment):

``MEMPALACE_REDACT``
    ``1``/``0``. Default ``1`` (on).
``MEMPALACE_REDACT_REPLACEMENT``
    Text that replaces a match. Default ``[REDACTED]``.
``MEMPALACE_REDACT_PATTERNS_FILE``
    Path to a file of extra regexes, one per line (``#`` comments allowed).
    A pattern with a named group ``secret`` redacts only that group, so the
    label (e.g. ``password=``) stays readable.
``MEMPALACE_REDACT_DISABLE_DEFAULTS``
    ``1`` to use only the patterns file. Default ``0``.

Default patterns are listed in :data:`DEFAULT_PATTERNS` and documented in
``docs/redaction.md``.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Iterable

from ._env import env_bool, env_str

logger = logging.getLogger(__name__)

DEFAULT_REPLACEMENT = "[REDACTED]"

# (name, regex). Quantifier bounds describe the published token formats.
DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("private_key_block",
     r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----[\s\S]+?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    ("aws_access_key_id", r"\b(?P<secret>(?:AKIA|ASIA)[0-9A-Z]{16})\b"),
    ("github_token", r"\b(?P<secret>(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,}))"),
    ("slack_token", r"\b(?P<secret>xox[abposr]-[A-Za-z0-9-]{10,})"),
    ("openai_style_key", r"\b(?P<secret>sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,})"),
    ("google_api_key", r"\b(?P<secret>AIza[0-9A-Za-z_-]{35})\b"),
    ("stripe_key", r"\b(?P<secret>[rs]k_(?:live|test)_[0-9A-Za-z]{16,})"),
    ("jwt", r"\b(?P<secret>eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)"),
    ("bearer_header", r"(?i)\bbearer\s+(?P<secret>[A-Za-z0-9._~+/=-]{16,})"),
    ("url_credentials", r"(?i)\b[a-z][a-z0-9+.-]*://[^\s:/@]+:(?P<secret>[^\s@/]+)@"),
    ("assignment",
     r"(?i)\b(?:password|passwd|pwd|passphrase|secret|api[_-]?key|access[_-]?token|"
     r"auth[_-]?token|client[_-]?secret|private[_-]?key)\b[\"']?\s*[:=]\s*[\"']?"
     r"(?P<secret>[^\s\"',;]+)"),
)


def _load_extra(path: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    # Patterns are code-like config: only trust a file this user owns that
    # nobody else can write (a hostile regex could stall mining).
    try:
        st = os.stat(path)
        if (hasattr(os, "getuid") and st.st_uid != os.getuid()) or st.st_mode & 0o022:
            logger.warning("redaction patterns file %s is not private to this user; ignored", path)
            return out
    except OSError as exc:
        logger.warning("redaction patterns file unreadable (%s); using defaults only", exc)
        return out
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append((f"{path}:{lineno}", line))
    except OSError as exc:
        logger.warning("redaction patterns file unreadable (%s); using defaults only", exc)
    return out


@lru_cache(maxsize=None)
def _compiled(extra_path: str, disable_defaults: bool) -> tuple[re.Pattern, ...]:
    specs: list[tuple[str, str]] = [] if disable_defaults else list(DEFAULT_PATTERNS)
    if extra_path:
        specs.extend(_load_extra(extra_path))
    compiled = []
    for name, rx in specs:
        try:
            compiled.append(re.compile(rx))
        except re.error as exc:
            logger.warning("skipping invalid redaction pattern %s: %s", name, exc)
    return tuple(compiled)


def patterns() -> tuple[re.Pattern, ...]:
    return _compiled(
        env_str("MEMPALACE_REDACT_PATTERNS_FILE", "") or "",
        env_bool("MEMPALACE_REDACT_DISABLE_DEFAULTS", False),
    )


def enabled() -> bool:
    return env_bool("MEMPALACE_REDACT", True)


def redact(text: str, compiled: Iterable[re.Pattern] | None = None) -> str:
    """Return ``text`` with secrets replaced. No-op when redaction is off."""
    if not text or not enabled():
        return text
    replacement = env_str("MEMPALACE_REDACT_REPLACEMENT", DEFAULT_REPLACEMENT) or DEFAULT_REPLACEMENT

    def _sub(match: re.Match) -> str:
        if "secret" in match.re.groupindex and match.group("secret") is not None:
            start, end = match.span("secret")
            whole_start = match.start()
            whole = match.group(0)
            return whole[: start - whole_start] + replacement + whole[end - whole_start :]
        return replacement

    for rx in compiled if compiled is not None else patterns():
        text = rx.sub(_sub, text)
    return text
