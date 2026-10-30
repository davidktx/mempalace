#!/usr/bin/env python3
"""MemPalace validation harness (single entrypoint).

  python validation/run.py functional            # back-tests on synthetic corpus
  python validation/run.py deploy --palace P --install VENV
                                                  # gate: runs against a COPY of P
  python validation/run.py deps [--baseline DIR]  # pytest + functional + Snyk, diff

Exit code 0 = all checks passed. JSON report written with --report.
No real user data lives in this repo; the corpus is synthetic.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CORPUS = HERE / "corpus"
GOLDEN = HERE / "golden" / "recall.json"


def load_config() -> dict:
    cfg = json.loads((HERE / "config.json").read_text())
    cfg.pop("_doc", None)
    for key, default in list(cfg.items()):
        env = os.environ.get(f"VALIDATION_{key.upper()}")
        if env is not None:
            cfg[key] = type(default)(float(env)) if isinstance(default, (int, float)) else env
    return cfg


CFG = load_config()


class Install:
    """Binaries of the install under test (a venv dir, or the current env)."""

    def __init__(self, root: Path | None):
        bindir = (root / "bin") if root else Path(sys.executable).parent
        self.mempalace = str(bindir / "mempalace")
        self.light = str(bindir / "mempalace-light-mcp")
        self.python = str(bindir / "python")
        for exe in (self.mempalace, self.light, self.python):
            if not os.access(exe, os.X_OK):
                raise SystemExit(f"not executable: {exe}")


def isolated_env(work: Path) -> dict:
    env = dict(os.environ)
    env.update(
        MEMPALACE_HOME=str(work / "home"),
        MEMPALACE_CONFIG_DIR=str(work / "cfg"),
        ANONYMIZED_TELEMETRY="False",
        CHROMA_TELEMETRY="False",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
        MEMPALACE_QUARANTINE_KEEP=str(CFG["quarantine_keep"]),
    )
    if CFG.get("backend"):
        env["MEMPALACE_BACKEND"] = CFG["backend"]
    for k in ("MEMPALACE_PALACE_PATH", "MEMPALACE_MCP_READ_ONLY", "MEMPALACE_READ_ONLY_OPEN"):
        env.pop(k, None)
    return env


def run(cmd, env, check=True):
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=CFG["timeout_seconds"])
    if check and p.returncode:
        raise RuntimeError(f"{cmd[0]} failed rc={p.returncode}: {p.stderr[-2000:]}")
    return p


def mine(inst, palace, src, wing, env):
    return run([inst.mempalace, "--palace", str(palace), "mine", str(src), "--mode", "convos", "--wing", wing], env)


def mcp_session(inst, palace, requests, env):
    init = {"jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "validation", "version": "1"}}}
    lines = [json.dumps(r) for r in [init, *requests]]
    p = subprocess.run([inst.light, "--palace", str(palace), "--read-only"], input="\n".join(lines) + "\n",
                       env=env, capture_output=True, text=True, timeout=CFG["timeout_seconds"])
    out = {}
    for line in p.stdout.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        out[msg.get("id")] = msg
    return out


def query_req(i, q, wing=None):
    args = {"target": "search", "query": q, "limit": CFG["top_k"]}
    if wing:
        args["wing"] = wing
    return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": "palace_query", "arguments": {"query": json.dumps(args)}}}


def result_texts(msg) -> list[str]:
    try:
        payload = json.loads(msg["result"]["content"][0]["text"])
    except (KeyError, IndexError, TypeError, ValueError):
        return []
    return [r.get("text", "") for r in payload.get("results", [])]


def tree_state(path: Path) -> dict:
    state = {}
    for f in sorted(path.rglob("*")):
        if f.is_file():
            st = f.stat()
            state[str(f.relative_to(path))] = (st.st_size, st.st_mtime_ns, hashlib.sha256(f.read_bytes()).hexdigest())
    return state


def sqlite_files(palace: Path):
    return [f for f in palace.rglob("*.sqlite3") if f.is_file()]


# ---------------------------------------------------------------- checks
def check_recall(inst, palace, env, wing):
    golden = json.loads(GOLDEN.read_text())
    reqs = [query_req(i + 1, g["q"], wing) for i, g in enumerate(golden["queries"])]
    resp = mcp_session(inst, palace, reqs, env)
    misses = [g["q"] for i, g in enumerate(golden["queries"])
              if not any(g["expect"] in t for t in result_texts(resp.get(i + 1, {})))]
    rate = 1 - len(misses) / len(golden["queries"])
    return rate >= CFG["min_hit_rate"], {"hit_rate": round(rate, 3), "misses": misses}


def check_redaction(palace):
    golden = json.loads(GOLDEN.read_text())
    leaks = []
    for f in palace.rglob("*"):
        if f.is_file():
            data = f.read_bytes()
            leaks += [f"{f.name}:{s[:4]}..." for s in golden["must_not_appear"] if s.encode() in data]
    return not leaks, {"leaks": leaks}


def check_readonly_mcp(inst, palace, env):
    """Only palace_query exposed, writes refused, and a read-only session
    leaves the palace byte-identical (SQLite -shm excepted). Chromadb applies pending vector writes
    to its HNSW files on the first open after a mine; that first session is
    reported (first_open_changes) and must touch only HNSW segment files,
    never a SQLite database. The second session must change nothing."""
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        query_req(2, "furnace filter"),
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "palace_exec", "arguments": {}}},
    ]

    def session():
        before = tree_state(palace)
        resp = mcp_session(inst, palace, reqs, env)
        after = tree_state(palace)
        # SQLite's -shm file is a shared-memory index rebuilt by any reader;
        # it carries no palace data, so it is not a write.
        return resp, sorted(k for k in set(before) | set(after)
                            if before.get(k) != after.get(k) and not k.endswith("-shm"))

    resp, first = session()
    resp2, second = session()
    tools = [t["name"] for t in resp.get(1, {}).get("result", {}).get("tools", [])]
    detail = {"tools": tools, "query_ok": bool(result_texts(resp2.get(2, {}))),
              "exec_refused": "error" in resp.get(3, {}),
              "first_open_changes": first, "second_open_changes": second}
    sqlite_touched = [f for f in first if f.endswith((".sqlite3", ".sqlite3-wal", ".sqlite3-journal"))]
    ok = (tools == ["palace_query"] and detail["query_ok"] and detail["exec_refused"]
          and not second and not sqlite_touched)
    return ok, detail


def integrity(palace):
    bad = {}
    for db in sqlite_files(palace):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            res = con.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            con.close()
        if res != "ok":
            bad[db.name] = res
    return bad


def check_concurrency(inst, palace, env, work):
    srcs = []
    for n in range(int(CFG["concurrent_mines"])):
        d = work / f"extra{n}"
        shutil.copytree(CORPUS, d)
        for f in d.iterdir():
            f.rename(d / f"c{n}-{f.name}")
        srcs.append(d)
    errors, refused = [], []

    def reader(_):
        for r in range(int(CFG["reader_rounds"])):
            resp = mcp_session(inst, palace, [query_req(1, "furnace filter size")], env)
            if not result_texts(resp.get(1, {})):
                errors.append(f"empty read round {r}")

    with cf.ThreadPoolExecutor() as ex:
        mines = [ex.submit(mine, inst, palace, d, f"conc{i}", env) for i, d in enumerate(srcs)]
        readers = [ex.submit(reader, i) for i in range(int(CFG["concurrent_readers"]))]
        for fut in [*mines, *readers]:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 — report every failure
                msg = str(exc)
                if "is held by PID" in msg:
                    refused.append("mine refused: palace lock held (expected)")
                else:
                    errors.append(msg[-300:])
    bad = integrity(palace)
    drift = [p.name for p in palace.iterdir() if ".drift-" in p.name or ".corrupt-" in p.name]
    ok_recall, rec = check_recall(inst, palace, env, "val")
    ok = not errors and not bad and ok_recall
    return ok, {"errors": errors, "lock_refusals": refused, "integrity": bad or "ok", "quarantined": drift, "recall_after": rec}


def check_quarantine_cap(inst, env, work):
    d = work / "qcap"
    d.mkdir()
    keep = int(CFG["quarantine_keep"])
    for i in range(keep + int(CFG["concurrent_readers"])):
        p = d / f"seg-1.drift-{i}"
        p.mkdir()
        os.utime(p, (i, i))
    code = ("import sys;from mempalace.backends.chroma import _prune_quarantined_copies as p;"
            "p(sys.argv[1], '.drift-')")
    # Prune is backend-agnostic file handling; it lives in the chroma module.
    run([inst.python, "-c", code, str(d / "seg-1")], env)
    left = sorted(x.name for x in d.iterdir())
    return len(left) == keep, {"keep": keep, "left": left}


def check_no_telemetry_imports(inst, env):
    code = ("import sys,mempalace.backends.chroma, mempalace.mcp_light_server;"
            "bad=[m for m in ('posthog','torch','transformers','sentence_transformers') if m in sys.modules];"
            "print(','.join(bad))")
    out = run([inst.python, "-c", code], env).stdout.strip()
    return not out, {"loaded": out or "none"}


# ---------------------------------------------------------------- modes
def functional(inst, work, base_palace=None):
    env = isolated_env(work)
    palace = work / "palace"
    if base_palace:
        shutil.copytree(base_palace, palace, symlinks=True)
    results = {}

    def rec(name, fn, *a):
        t = time.time()
        try:
            ok, detail = fn(*a)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, {"exception": str(exc)[-500:]}
        results[name] = {"pass": ok, "seconds": round(time.time() - t, 1), **detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(detail)[:300]}", flush=True)

    rec("mine_corpus", lambda: (mine(inst, palace, CORPUS, "val", env).returncode == 0, {}))
    rec("recall_golden", check_recall, inst, palace, env, "val")
    rec("redaction", check_redaction, palace)
    rec("readonly_mcp", check_readonly_mcp, inst, palace, env)
    rec("concurrency", check_concurrency, inst, palace, env, work)
    rec("quarantine_cap", check_quarantine_cap, inst, env, work)
    rec("no_telemetry_or_torch", check_no_telemetry_imports, inst, env)
    return results


def snyk_scans(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    venv = Path(tempfile.mkdtemp(prefix="mp-snyk-venv-"))  # outside the scanned dir
    req = out / "requirements.txt"
    exp = subprocess.run(["uv", "export", "--frozen", "--no-hashes", "--no-emit-project",
                          *os.environ.get("VALIDATION_UV_EXPORT_ARGS", "").split()],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    skip = ("directml", "coreml", "cuda", "windows", "win32")
    req.write_text("\n".join(l for l in exp.splitlines() if l.strip() and not l.lstrip().startswith("#")
                             and not any(s in l.lower() for s in skip)) + "\n")
    # Snyk's pip plugin needs pip inside the venv, hence --seed.
    subprocess.run(["uv", "venv", "-q", "--seed", "--allow-existing", "-p", sys.executable, str(venv)], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "-p", str(venv / "bin" / "python"), "-r", str(req)], check=True)
    with open(out / "oss.json", "w") as fh:
        subprocess.run(["snyk", "test", "--file=requirements.txt", "--package-manager=pip",
                        f"--command={venv / 'bin' / 'python'}", "--skip-unresolved", "--json"], cwd=out, stdout=fh)
    with open(out / "code.json", "w") as fh:
        subprocess.run(["snyk", "code", "test", "--json"], cwd=REPO, stdout=fh)
    shutil.rmtree(venv, ignore_errors=True)
    return summarize(out)


def summarize(d: Path) -> dict:
    s = {"oss": {}, "code": {}}
    try:
        raw = json.loads((d / "oss.json").read_text())
        if raw.get("ok") is False and raw.get("error"):
            raise ValueError(raw["error"])
        vulns = {v["id"]: v for v in raw.get("vulnerabilities", [])}
        for v in vulns.values():
            s["oss"][v["severity"]] = s["oss"].get(v["severity"], 0) + 1
        s["oss_packages"] = sorted({f"{v['packageName']}@{v['version']}" for v in vulns.values()})
    except (OSError, ValueError) as exc:
        s["oss"] = f"missing: {str(exc)[:200]}"
    try:
        for r in json.loads((d / "code.json").read_text())["runs"][0]["results"]:
            s["code"][r["level"]] = s["code"].get(r["level"], 0) + 1
    except (OSError, ValueError, KeyError):
        s["code"] = "missing"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["functional", "deploy", "deps"])
    ap.add_argument("--install", type=Path, help="venv root of the install under test (default: current env)")
    ap.add_argument("--palace", type=Path, help="deploy: palace to COPY and test against (never modified)")
    ap.add_argument("--baseline", type=Path, help="deps: earlier scan dir to diff against")
    ap.add_argument("--out", type=Path, default=REPO.parent / "mempalace-validation")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--skip-pytest", action="store_true")
    a = ap.parse_args()
    inst = Install(a.install)
    report = {"mode": a.mode, "config": CFG, "install": inst.mempalace}
    with tempfile.TemporaryDirectory(prefix="mp-validate-") as tmp:
        work = Path(tmp)
        if a.mode == "deploy":
            if not a.palace or not a.palace.is_dir():
                raise SystemExit("deploy needs --palace DIR (it is copied, never touched)")
            report["checks"] = functional(inst, work, a.palace)
        else:
            report["checks"] = functional(inst, work)
    ok = all(c["pass"] for c in report["checks"].values())
    if a.mode == "deps":
        if not a.skip_pytest:
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=REPO)
            report["pytest_rc"] = p.returncode
            ok = ok and p.returncode == 0
        stamp = time.strftime("%Y%m%d-%H%M%S")
        report["snyk"] = snyk_scans(a.out / stamp)
        cur = report["snyk"]
        ok = ok and isinstance(cur["oss"], dict) and isinstance(cur["code"], dict)
        if a.baseline:
            base = report["baseline"] = summarize(a.baseline)
            regressions = {}
            for kind in ("oss", "code"):
                if isinstance(base[kind], dict) and isinstance(cur[kind], dict):
                    for sev, n in cur[kind].items():
                        if n > base[kind].get(sev, 0):
                            regressions[f"{kind}.{sev}"] = [base[kind].get(sev, 0), n]
            report["regressions"] = regressions
            ok = ok and not regressions
        print(json.dumps({k: report.get(k) for k in ("pytest_rc", "snyk", "baseline", "regressions")}, indent=2))
    if a.report:
        a.report.write_text(json.dumps(report, indent=2))
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
