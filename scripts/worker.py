"""Single-config benchmark worker.

Runs in a fresh process with thread limits applied via env vars by the
parent. Imports faiss and sets faiss.omp_set_num_threads(threads).

Only index.search() calls are timed. Data load, index build, warm-up,
and the (separate, untimed) pass that collects top-5 IDs for the
agreement check are all outside the timed region.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import faiss

ROOT = Path(__file__).resolve().parent.parent


def ids_from_untimed_pass(index: "faiss.Index", xq: np.ndarray,
                          batch_size: int, k: int) -> np.ndarray:
    """Collect top-k IDs for the exact queries used in the timed region.

    Done in a single pass outside timing to avoid polluting measurements
    with Python-side bookkeeping.
    """
    n = xq.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    if batch_size == 1:
        for i in range(n):
            _, I = index.search(xq[i:i + 1], k)
            out[i] = I[0]
    else:
        for start in range(0, n, batch_size):
            _, I = index.search(xq[start:start + batch_size], k)
            out[start:start + batch_size] = I
    return out


def run(threads: int, batch_size: int, rep: int, seed: int,
        queries_per_rep: int, k: int, out_path: Path) -> None:
    # Apply faiss thread count explicitly. Env vars were set by the parent.
    faiss.omp_set_num_threads(int(threads))

    # Load data and rebuild index. Outside timing.
    xb = np.load(ROOT / "data" / "xb.npy")
    xq_all = np.load(ROOT / "data" / "xq.npy")
    assert xb.dtype == np.float32 and xq_all.dtype == np.float32
    d = xb.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(xb)

    # Deterministic query order per rep: permute the first queries_per_rep
    # indices using (seed, rep). The same shuffle maps to the same order
    # across configs for a given rep, so batches align 1:1.
    rng = np.random.default_rng((seed, rep))
    nq_all = xq_all.shape[0]
    if queries_per_rep > nq_all:
        raise SystemExit(f"queries_per_rep={queries_per_rep} > nq={nq_all}")
    order = rng.permutation(nq_all)[:queries_per_rep]
    xq = np.ascontiguousarray(xq_all[order], dtype=np.float32)

    # Warm-up: enough calls to amortise BLAS first-call cost and OMP spin-up.
    warm_n = 10
    for i in range(warm_n):
        if batch_size == 1:
            index.search(xq[i % xq.shape[0]:i % xq.shape[0] + 1], k)
        else:
            s = (i * batch_size) % xq.shape[0]
            e = min(s + batch_size, xq.shape[0])
            index.search(xq[s:e], k)

    # Timed region.
    call_durations: list[float] = []
    queries_processed = 0
    t_total_start = time.perf_counter()
    if batch_size == 1:
        for i in range(xq.shape[0]):
            q = xq[i:i + 1]
            t0 = time.perf_counter()
            index.search(q, k)
            t1 = time.perf_counter()
            call_durations.append(t1 - t0)
            queries_processed += 1
    else:
        for start in range(0, xq.shape[0], batch_size):
            end = start + batch_size
            qb = xq[start:end]
            t0 = time.perf_counter()
            index.search(qb, k)
            t1 = time.perf_counter()
            call_durations.append(t1 - t0)
            queries_processed += qb.shape[0]
    t_total_end = time.perf_counter()
    total_timed = t_total_end - t_total_start

    # Untimed pass to capture top-k IDs for agreement check.
    top_k_ids = ids_from_untimed_pass(index, xq, batch_size, k)

    record = {
        "config": {"threads": int(threads), "batch_size": int(batch_size),
                   "rep": int(rep)},
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "faiss_omp_get_max_threads_after_set": int(
                faiss.omp_get_max_threads()
            ),
        },
        "dataset": {
            "nb": int(xb.shape[0]), "nq_total": int(xq_all.shape[0]),
            "d": int(d), "k": int(k),
            "queries_per_rep": int(queries_per_rep),
            "query_order_seed": [int(seed), int(rep)],
        },
        "warm_up_calls": warm_n,
        "queries_processed": int(queries_processed),
        "n_timed_calls": len(call_durations),
        "call_durations_s": [float(x) for x in call_durations],
        "total_timed_seconds": float(total_timed),
        "top5_ids": top_k_ids.tolist(),
        "query_order": order.tolist(),
        "faiss_version": faiss.__version__,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record))
    print(f"ok threads={threads} batch={batch_size} rep={rep} "
          f"n_calls={len(call_durations)} timed={total_timed:.3f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--queries-per-rep", type=int, default=512)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    run(args.threads, args.batch_size, args.rep, args.seed,
        args.queries_per_rep, args.k, Path(args.out))


if __name__ == "__main__":
    main()
