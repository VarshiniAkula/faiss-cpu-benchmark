"""Orchestrator: spawn fresh worker subprocesses per (config, rep).

Each worker inherits a scrubbed env with thread limits set to the
requested count BEFORE importing numpy/faiss. We also record the exact
order configs ran in, for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PER_CALL = RESULTS / "per_call"

THREAD_CHOICES = [1, 2, 4]
BATCH_CHOICES = [1, 32]
CONFIGS = [(t, b) for b in BATCH_CHOICES for t in THREAD_CHOICES]


def make_env(threads: int) -> dict:
    env = os.environ.copy()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
              "BLIS_NUM_THREADS"):
        env[v] = str(threads)
    return env


def run_one(threads: int, batch_size: int, rep: int, seed: int,
            queries_per_rep: int, k: int) -> dict:
    out_path = PER_CALL / f"b{batch_size}_t{threads}_rep{rep}.json"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "worker.py"),
        "--threads", str(threads),
        "--batch-size", str(batch_size),
        "--rep", str(rep),
        "--seed", str(seed),
        "--queries-per-rep", str(queries_per_rep),
        "--k", str(k),
        "--out", str(out_path),
    ]
    env = make_env(threads)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        print("STDOUT:", r.stdout)
        print("STDERR:", r.stderr)
        raise SystemExit(f"worker failed t={threads} b={batch_size} rep={rep}")
    print(f"[rep {rep}] b={batch_size} t={threads} wall={wall:.2f}s "
          f"-> {out_path.name}")
    return {"threads": threads, "batch_size": batch_size, "rep": rep,
            "wall_seconds": wall, "out_path": str(out_path),
            "worker_stdout": r.stdout.strip()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--queries-per-rep", type=int, default=512)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--smoke", action="store_true",
                   help="single-config smoke run (t=1,b=1,1 rep)")
    args = p.parse_args()

    PER_CALL.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    order_log: list[dict] = []
    if args.smoke:
        rec = run_one(1, 1, 0, args.seed, args.queries_per_rep, args.k)
        order_log.append(rec)
    else:
        for rep in range(args.reps):
            configs = list(CONFIGS)
            random.Random(f"{args.seed}:{rep}").shuffle(configs)
            print(f"=== rep {rep} order: {configs} ===")
            for (t, b) in configs:
                rec = run_one(t, b, rep, args.seed,
                              args.queries_per_rep, args.k)
                order_log.append(rec)

    (RESULTS / "run_order.json").write_text(json.dumps(order_log, indent=2))
    print(f"wrote {RESULTS / 'run_order.json'} ({len(order_log)} entries)")


if __name__ == "__main__":
    main()
