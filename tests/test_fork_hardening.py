"""Tests for the davidktx fork hardening: read-only open, quarantine cap,
transcript redaction, endpoint scheme guard, pom DTD refusal."""

import os
import time

import pytest

from mempalace import redact as R
from mempalace.backends import chroma as C


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for k in list(os.environ):
        if k.startswith("MEMPALACE_REDACT") or k == "MEMPALACE_QUARANTINE_KEEP":
            monkeypatch.delenv(k, raising=False)
    R._compiled.cache_clear()
    prev = C.read_only_open()
    yield
    C.set_read_only_open(prev)
    R._compiled.cache_clear()


# --- read-only open -------------------------------------------------------

def test_read_only_open_skips_all_repairs(monkeypatch, tmp_path):
    calls = []
    for fn in ("_fix_missing_collection_type", "_fix_blob_seq_ids",
               "quarantine_invalid_hnsw_metadata", "quarantine_stale_hnsw"):
        monkeypatch.setattr(C, fn, lambda *a, _fn=fn, **k: calls.append(_fn))
    C.ChromaBackend._quarantined_paths.discard(str(tmp_path))
    C.set_read_only_open(True)
    C.ChromaBackend._prepare_palace_for_open(str(tmp_path))
    assert calls == []
    C.set_read_only_open(False)
    C.ChromaBackend._prepare_palace_for_open(str(tmp_path))
    assert len(calls) == 4


def test_server_read_only_flag_sets_policy(monkeypatch):
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server, "_READ_ONLY", False)
    C.set_read_only_open(False)
    mcp_server._apply_server_flags(read_only=True)
    assert C.read_only_open()


# --- quarantine retention -------------------------------------------------

def _mk(path, age):
    path.mkdir()
    t = time.time() - age
    os.utime(path, (t, t))


def test_prune_keeps_default_one(tmp_path):
    seg = tmp_path / "abcd-1234"
    for i in range(4):
        _mk(tmp_path / f"abcd-1234.drift-{i}", age=100 - i)
    removed = C._prune_quarantined_copies(str(seg), ".drift-")
    left = sorted(p.name for p in tmp_path.iterdir())
    assert len(removed) == 3
    assert left == ["abcd-1234.drift-3"]


def test_prune_negative_means_unbounded(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_QUARANTINE_KEEP", "-1")
    for i in range(3):
        _mk(tmp_path / f"s-1.corrupt-{i}", age=i)
    assert C._prune_quarantined_copies(str(tmp_path / "s-1"), ".corrupt-") == []
    assert len(list(tmp_path.iterdir())) == 3


def test_prune_zero_keeps_none_and_ignores_other_segments(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_QUARANTINE_KEEP", "0")
    _mk(tmp_path / "s-1.drift-a", age=1)
    _mk(tmp_path / "s-2.drift-a", age=1)
    C._prune_quarantined_copies(str(tmp_path / "s-1"), ".drift-")
    assert [p.name for p in tmp_path.iterdir()] == ["s-2.drift-a"]


# --- redaction ------------------------------------------------------------

@pytest.mark.parametrize("text,secret", [
    ("token ghp_" + "a" * 36, "ghp_" + "a" * 36),
    ("key AKIA" + "B" * 16 + " here", "AKIA" + "B" * 16),
    ("xoxb-" + "1" * 12 + "-abc", "xoxb-" + "1" * 12 + "-abc"),
    ("Authorization: Bearer " + "z" * 24, "z" * 24),
    ("password=hunter2hunter2", "hunter2hunter2"),
    ("db postgres://user:s3cr3t@host/db", "s3cr3t"),
    ("use sk-proj-" + "x" * 30, "sk-proj-" + "x" * 30),
])
def test_default_patterns_redact(text, secret):
    out = R.redact(text)
    assert secret not in out
    assert R.DEFAULT_REPLACEMENT in out


def test_label_is_preserved():
    assert R.redact("password: abcdefgh") == "password: [REDACTED]"


def test_private_key_block():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIB\n-----END RSA PRIVATE KEY-----"
    assert R.redact("x " + pem + " y") == "x [REDACTED] y"


def test_disabled_and_custom(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_REDACT", "0")
    assert R.redact("password=abc") == "password=abc"
    monkeypatch.setenv("MEMPALACE_REDACT", "1")
    f = tmp_path / "p.txt"
    f.write_text("# comment\ninternal-(?P<secret>[0-9]+)\n")
    f.chmod(0o600)
    monkeypatch.setenv("MEMPALACE_REDACT_PATTERNS_FILE", str(f))
    monkeypatch.setenv("MEMPALACE_REDACT_DISABLE_DEFAULTS", "1")
    monkeypatch.setenv("MEMPALACE_REDACT_REPLACEMENT", "***")
    R._compiled.cache_clear()
    assert R.redact("internal-42 password=abc") == "internal-*** password=abc"


def test_plain_text_unchanged():
    s = "We talked about the heat setpoint and the password policy doc."
    assert R.redact(s) == s


def test_normalize_redacts(tmp_path):
    from mempalace.normalize import normalize, normalize_conversations

    f = tmp_path / "t.txt"
    f.write_text("> hi\nmy api_key=SUPERSECRETVALUE1\n> a\n> b\n")
    assert "SUPERSECRETVALUE1" not in normalize(str(f))
    assert all("SUPERSECRETVALUE1" not in c for c in normalize_conversations(str(f)))


# --- endpoint scheme + pom ------------------------------------------------

def test_post_json_rejects_file_scheme():
    from mempalace.llm_client import LLMError, post_json

    with pytest.raises(LLMError):
        post_json("file:///etc/passwd", {}, {}, timeout=1)


def test_pom_with_doctype_refused(tmp_path):
    from mempalace.project_scanner import _parse_pom

    p = tmp_path / "pom.xml"
    p.write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e "v">]>'
                 "<project><artifactId>&e;</artifactId></project>")
    assert _parse_pom(p) is None
    p.write_text("<project><artifactId>ok</artifactId></project>")
    assert _parse_pom(p) == "ok"


def test_group_writable_patterns_file_ignored(monkeypatch, tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("internal-[0-9]+\n")
    f.chmod(0o666)
    monkeypatch.setenv("MEMPALACE_REDACT_PATTERNS_FILE", str(f))
    R._compiled.cache_clear()
    assert R.redact("internal-42") == "internal-42"
