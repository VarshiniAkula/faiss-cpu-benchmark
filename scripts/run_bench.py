"""Orchestrator: isolates every experiment into its own run directory.

Each run directory contains:
  manifest.json          what was planned, env/versions/git/dataset hashes
  machine.json           copied from inspect_machine.py
  run_order.json         actual order configs ran in, plus wall times
  per_call/*.json        raw per-config, per-rep records (one file each)
  status.json            completion state, missing/duplicates, finished_at

Modes:
  --mode full  : six configs x N reps. Rejects duplicates/missing at end.
  --mode smoke : one config (1,1,0). Writes a smoke manifest so analyze
                 won't confuse it with a full run.

Refuses to overwrite an existing full run (manifest present & mode=full
& any records present) unless --allow-overwrite is given explicitly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import validate as v  # noqa: E402

THREAD_CHOICES = [1, 2, 4]
BATCH_CHOICES = [1, 32]
CONFIGS = [(t, b) for b in BATCH_CHOICES for t in THREAD_CHOICES]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_meta(root: Path) -> dict:
    def _git(args: list[str]) -> str | None:
        try:
            r = subprocess.run(["git", *args], cwd=root,
                               capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    commit = _git(["rev-parse", "HEAD"])
    short = _git(["rev-parse", "--short", "HEAD"])
    status = _git(["status", "--porcelain"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "commit": commit, "commit_short": short,
        "dirty": bool(status) if status is not None else None,
        "branch": branch,
    }


def pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception as e:
        return f"__err__:{e}"


def make_env(threads: int) -> dict:
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[var] = str(threads)
    return env


def run_one(run_dir: Path, data_dir: Path, threads: int, batch_size: int,
            rep: int, seed: int, queries_per_rep: int, k: int,
            expected_hashes: dict) -> dict:
    per_call_dir = run_dir / "per_call"
    per_call_dir.mkdir(parents=True, exist_ok=True)
    out_path = per_call_dir / f"b{batch_size}_t{threads}_rep{rep}.json"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "worker.py"),
        "--threads", str(threads),
        "--batch-size", str(batch_size),
        "--rep", str(rep),
        "--seed", str(seed),
        "--queries-per-rep", str(queries_per_rep),
        "--k", str(k),
        "--data-dir", str(data_dir),
        "--out", str(out_path),
        "--expected-xb-sha", expected_hashes["xb_sha256"],
        "--expected-xq-sha", expected_hashes["xq_sha256"],
    ]
    env = make_env(threads)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        print("STDOUT:", r.stdout)
        print("STDERR:", r.stderr)
        raise SystemExit(
            f"worker failed t={threads} b={batch_size} rep={rep}"
        )
    print(f"[rep {rep}] b={batch_size} t={threads} wall={wall:.2f}s "
          f"-> {out_path.relative_to(run_dir)}")
    return {
        "threads": threads, "batch_size": batch_size, "rep": rep,
        "wall_seconds": wall, "out_path": str(out_path),
        "worker_stdout": r.stdout.strip(),
    }


def existing_run_blocks_overwrite(run_dir: Path) -> str | None:
    """Return reason string if there's an existing run we shouldn't clobber."""
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return None
    try:
        m = json.loads(manifest.read_text())
    except Exception:
        return None
    pc = run_dir / "per_call"
    has_any = pc.exists() and any(pc.glob("*.json"))
    if m.get("mode") == "full" and has_any:
        return (f"{run_dir} already contains a 'full' run with records; "
                f"refusing to overwrite (use --allow-overwrite).")
    if m.get("mode") == "smoke" and has_any:
        return (f"{run_dir} already contains a 'smoke' run; "
                f"use a different --run-dir.")
    return None


def write_manifest(run_dir: Path, mode: str, reps: int, seed: int,
                   queries_per_rep: int, k: int,
                   configs: list[tuple[int, int]],
                   data_dir: Path) -> dict:
    expected_entries = (
        [[t, b, r] for r in range(reps) for (t, b) in configs]
        if mode == "full" else [[1, 1, 0]]
    )
    hashes = {
        "xb_sha256": sha256_file(data_dir / "xb.npy"),
        "xq_sha256": sha256_file(data_dir / "xq.npy"),
    }
    manifest = {
        "schema_version": 2,
        "run_dir": str(run_dir),
        "mode": mode,
        "started_at_utc": utc_now_iso(),
        "seed": seed,
        "queries_per_rep": queries_per_rep,
        "k": k,
        "configs": [list(c) for c in configs],
        "reps": reps,
        "expected_entries": expected_entries,
        "dataset_sha256": hashes,
        "git": git_meta(ROOT),
        "python_version": sys.version.split()[0],
        "packages": {
            "faiss-cpu": pkg_version("faiss-cpu"),
            "numpy": pkg_version("numpy"),
            "matplotlib": pkg_version("matplotlib"),
            "threadpoolctl": pkg_version("threadpoolctl"),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def copy_machine_info(run_dir: Path) -> None:
    # Prefer a freshly-generated per-run machine.json from inspect_machine.
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "inspect_machine.py"),
             "--out", str(run_dir / "machine.json")],
            check=True, capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        (run_dir / "machine.json").write_text(json.dumps({
            "error": f"inspect_machine.py failed: {e}",
            "captured_at_utc": utc_now_iso(),
        }, indent=2))


def write_status(run_dir: Path, manifest: dict,
                 records: dict) -> dict:
    report = v.validate_manifest_vs_records(manifest, records)
    if manifest["mode"] == "smoke":
        state = (
            "completed" if not report["missing"] and not report["duplicates"]
            else "incomplete"
        )
    else:
        state = (
            "completed"
            if not report["missing"] and not report["duplicates"]
            and not report["extra"]
            else "incomplete"
        )
    status = {
        "state": state,
        "mode": manifest["mode"],
        "finished_at_utc": utc_now_iso(),
        "n_records_present": len(records),
        "n_entries_expected": len(manifest["expected_entries"]),
        **report,
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    return status


def gather_records(run_dir: Path) -> dict:
    """Map (t, b, rep) -> list of file paths (list for duplicate detection)."""
    out: dict = {}
    pc = run_dir / "per_call"
    for p in sorted(pc.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        c = rec.get("config", {})
        try:
            key = (int(c["threads"]), int(c["batch_size"]), int(c["rep"]))
        except Exception:
            continue
        out.setdefault(key, []).append(str(p))
    return out


def auto_run_id(mode: str, git: dict) -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = git.get("commit_short") or "no-git"
    return f"{ts}-{mode}-{short}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--run-dir", type=str, default=None,
                   help="output run directory. Default: runs/<auto id>.")
    p.add_argument("--data-dir", type=str,
                   default=str(ROOT / "data"))
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--queries-per-rep", type=int, default=512)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--allow-overwrite", action="store_true")
    args = p.parse_args()

    # Argument validation up-front (positive ints etc).
    reps = args.reps if args.mode == "full" else 1
    v.validate_positive_int("reps", reps)
    v.validate_positive_int("k", args.k, 1, v.VALID_K_RANGE[1])
    v.validate_positive_int("queries_per_rep", args.queries_per_rep)
    for b in BATCH_CHOICES:
        if args.queries_per_rep % b != 0:
            raise SystemExit(
                f"queries_per_rep={args.queries_per_rep} not divisible by "
                f"batch={b}; cannot run planned batch configurations"
            )

    data_dir = Path(args.data_dir)
    if not (data_dir / "xb.npy").exists() or not (data_dir / "xq.npy").exists():
        raise SystemExit(
            f"dataset .npy missing in {data_dir}; run scripts/make_data.py"
        )

    git = git_meta(ROOT)
    run_dir = (
        Path(args.run_dir) if args.run_dir else
        ROOT / "runs" / auto_run_id(args.mode, git)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    reason = existing_run_blocks_overwrite(run_dir)
    if reason and not args.allow_overwrite:
        raise SystemExit(reason)

    copy_machine_info(run_dir)
    configs = (
        [(1, 1)] if args.mode == "smoke" else list(CONFIGS)
    )
    manifest = write_manifest(run_dir, args.mode, reps, args.seed,
                              args.queries_per_rep, args.k, configs,
                              data_dir)

    expected_hashes = manifest["dataset_sha256"]
    print(f"run_dir={run_dir}")
    print(f"mode={args.mode} reps={reps} configs={configs}")

    order_log = []
    if args.mode == "smoke":
        rec = run_one(run_dir, data_dir, 1, 1, 0, args.seed,
                      args.queries_per_rep, args.k, expected_hashes)
        order_log.append(rec)
    else:
        for rep in range(reps):
            shuffled = list(configs)
            random.Random(f"{args.seed}:{rep}").shuffle(shuffled)
            print(f"=== rep {rep} order: {shuffled} ===")
            for (t, b) in shuffled:
                rec = run_one(run_dir, data_dir, t, b, rep, args.seed,
                              args.queries_per_rep, args.k, expected_hashes)
                order_log.append(rec)

    (run_dir / "run_order.json").write_text(json.dumps(order_log, indent=2))

    records = gather_records(run_dir)
    status = write_status(run_dir, manifest, records)
    print(f"status={status['state']} "
          f"present={status['n_records_present']}/"
          f"{status['n_entries_expected']}")
    if status["state"] != "completed":
        print("WARNING: run did not complete cleanly.")
        print(" missing   :", status["missing"])
        print(" duplicates:", status["duplicates"])
        print(" extra     :", status["extra"])
        raise SystemExit(2)


if __name__ == "__main__":
    main()
