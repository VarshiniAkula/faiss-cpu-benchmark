"""Single-config benchmark worker.

Runs in a fresh process. Env vars for OMP/BLAS thread limits are set by
the orchestrator before this script imports numpy/faiss. After import,
the worker also calls faiss.omp_set_num_threads(threads) and records the
loaded BLAS/OpenMP runtimes via threadpoolctl.

Two timing fields, both recorded per run:
  sum_call_durations_s   -- sum of individual timed search() calls.
                            Used as the throughput denominator so Python
                            overhead between calls is excluded.
  elapsed_around_loop_s  -- wall time from the first call's t0 to the
                            last call's t1. Diagnostic only; strictly
                            >= sum_call_durations_s.

The pass that collects top-5 IDs for the agreement check is a SEPARATE,
UNTIMED pass after the timed region.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import faiss

try:
    from threadpoolctl import threadpool_info
except ImportError:  # pragma: no cover - fail loudly later
    threadpool_info = None  # type: ignore

# Allow running as `python scripts/worker.py` or `python -m scripts.worker`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import validate as v  # noqa: E402


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ids_from_untimed_pass(index: "faiss.Index", xq: np.ndarray,
                          batch_size: int, k: int) -> np.ndarray:
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


def pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception as e:  # pragma: no cover
        return f"__err__:{e}"


def run(threads: int, batch_size: int, rep: int, seed: int,
        queries_per_rep: int, k: int,
        data_dir: Path, out_path: Path,
        expected_dataset_sha: dict | None = None) -> None:
    v.validate_cli(threads, batch_size, reps=1, k=k,
                   queries_per_rep=queries_per_rep)

    faiss.omp_set_num_threads(int(threads))

    xb_path = data_dir / "xb.npy"
    xq_path = data_dir / "xq.npy"
    if not xb_path.exists() or not xq_path.exists():
        raise SystemExit(
            f"dataset .npy files missing in {data_dir}; "
            f"run scripts/make_data.py first."
        )
    # Hash BEFORE load so we fail fast on data drift.
    hashes = {"xb_sha256": sha256_file(xb_path),
              "xq_sha256": sha256_file(xq_path)}
    if expected_dataset_sha is not None:
        for key, got in hashes.items():
            want = expected_dataset_sha.get(key)
            if want and want != got:
                raise SystemExit(
                    f"dataset hash mismatch for {key}: "
                    f"got {got[:12]}..., expected {want[:12]}..."
                )

    xb = np.load(xb_path)
    xq_all = np.load(xq_path)
    if xb.dtype != np.float32 or xq_all.dtype != np.float32:
        raise SystemExit("dataset arrays must be float32")
    d = int(xb.shape[1])
    if xq_all.shape[1] != d:
        raise SystemExit(f"xq d={xq_all.shape[1]} != xb d={d}")
    index = faiss.IndexFlatL2(d)
    index.add(xb)

    rng = np.random.default_rng(np.array([seed, rep], dtype=np.int64))
    nq_all = int(xq_all.shape[0])
    if queries_per_rep > nq_all:
        raise SystemExit(f"queries_per_rep={queries_per_rep} > nq={nq_all}")
    order = rng.permutation(nq_all)[:queries_per_rep]
    xq = np.ascontiguousarray(xq_all[order], dtype=np.float32)

    warm_n = 10
    for i in range(warm_n):
        if batch_size == 1:
            index.search(xq[i % xq.shape[0]:i % xq.shape[0] + 1], k)
        else:
            s = (i * batch_size) % xq.shape[0]
            e = min(s + batch_size, xq.shape[0])
            index.search(xq[s:e], k)

    call_durations: list[float] = []
    queries_processed = 0
    t_loop_start = time.perf_counter()
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
    t_loop_end = time.perf_counter()
    elapsed_around_loop = t_loop_end - t_loop_start
    sum_calls = float(sum(call_durations))

    top5 = ids_from_untimed_pass(index, xq, batch_size, k)

    # Provenance: runtime BLAS/OMP info AFTER thread limits applied.
    tp_info = threadpool_info() if threadpool_info else {
        "error": "threadpoolctl not installed",
    }

    record = {
        "schema_version": 2,
        "config": {"threads": int(threads), "batch_size": int(batch_size),
                   "rep": int(rep)},
        "dataset": {
            "nb": int(xb.shape[0]), "nq_total": int(xq_all.shape[0]),
            "d": int(d), "k": int(k),
            "queries_per_rep": int(queries_per_rep),
            "query_order_seed": [int(seed), int(rep)],
            "xb_sha256": hashes["xb_sha256"],
            "xq_sha256": hashes["xq_sha256"],
        },
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "BLIS_NUM_THREADS": os.environ.get("BLIS_NUM_THREADS"),
            "faiss_omp_get_max_threads_after_set": int(
                faiss.omp_get_max_threads()
            ),
        },
        "runtime_pools": tp_info,
        "packages": {
            "faiss-cpu": pkg_version("faiss-cpu"),
            "numpy": pkg_version("numpy"),
            "matplotlib": pkg_version("matplotlib"),
            "threadpoolctl": pkg_version("threadpoolctl"),
        },
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "warm_up_calls": warm_n,
        "queries_processed": int(queries_processed),
        "n_timed_calls": len(call_durations),
        "call_durations_s": [float(x) for x in call_durations],
        "sum_call_durations_s": sum_calls,
        "elapsed_around_loop_s": float(elapsed_around_loop),
        # DERIVED convenience, documented: search-only throughput.
        "throughput_qps_search_only": (
            float(queries_processed / sum_calls) if sum_calls > 0 else None
        ),
        "top5_ids": top5.tolist(),
        "query_order": order.tolist(),
        "faiss_version": faiss.__version__,
        "started_at_utc": _utc_iso_offset_at_start,
        "finished_at_utc": _utc_iso(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record))
    print(
        f"ok threads={threads} batch={batch_size} rep={rep} "
        f"n_calls={len(call_durations)} "
        f"sum_calls={sum_calls:.4f}s "
        f"loop={elapsed_around_loop:.4f}s "
        f"throughput={record['throughput_qps_search_only']:.1f} qps"
    )


def _utc_iso() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_utc_iso_offset_at_start = _utc_iso()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--queries-per-rep", type=int, default=512)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--expected-xb-sha", type=str, default=None)
    p.add_argument("--expected-xq-sha", type=str, default=None)
    args = p.parse_args()
    expected = None
    if args.expected_xb_sha or args.expected_xq_sha:
        expected = {}
        if args.expected_xb_sha:
            expected["xb_sha256"] = args.expected_xb_sha
        if args.expected_xq_sha:
            expected["xq_sha256"] = args.expected_xq_sha
    run(args.threads, args.batch_size, args.rep, args.seed,
        args.queries_per_rep, args.k,
        Path(args.data_dir), Path(args.out), expected)


if __name__ == "__main__":
    main()
