"""Fixed-duration search workload for external profiling (e.g. Intel VTune).

Design:
  - Data load and index build happen BEFORE the measured region.
  - 10 warm-up search calls happen BEFORE the measured region.
  - Inside the measured region, the same 512 queries are run in batches
    of the requested batch size, in a tight loop, repeated across the
    full query set, for a configurable wall-clock duration (default 25s).
  - Queries are REPEATED, not freshly generated. This is intentional for
    a profiler run so the measured region is a single, stable workload.
    For headline performance numbers use scripts/run_bench.py instead.

The script optionally emits Intel ITT markers (via the `itt-py` package
if installed) to delimit the measured region, so VTune can filter on
"measured_region". If ITT is not available, we still delimit the region
with two sentinel prints that VTune's time-line view can anchor to.

If run with `--vtune-pause` the process will PAUSE itself (instead of
exiting) for a few seconds AFTER the measured region, giving a running
`vtune -collect` the chance to finalise its capture cleanly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Optional ITT API. Available when VTune's ITT SDK is installed and the
# Python binding `itt` is on PYTHONPATH. We treat absence as expected.
try:
    import itt  # type: ignore
    _HAS_ITT = True
except Exception:
    _HAS_ITT = False


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--duration-s", type=float, default=25.0,
                   help="target wall-clock duration of the measured region")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--out", type=str, required=True,
                   help="JSON summary output path")
    p.add_argument("--vtune-pause", type=float, default=0.0,
                   help="seconds to sleep AFTER the measured region "
                        "(gives external collectors time to flush)")
    args = p.parse_args()

    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be > 0")

    # Environment thread limits should have been set by the launcher;
    # we only enforce FAISS's OMP post-import.
    import numpy as np
    import faiss
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        threadpool_info = None  # type: ignore

    faiss.omp_set_num_threads(int(args.threads))

    data_dir = Path(args.data_dir)
    xb_path = data_dir / "xb.npy"
    xq_path = data_dir / "xq.npy"
    if not xb_path.exists() or not xq_path.exists():
        raise SystemExit(
            f"dataset .npy missing in {data_dir}; run scripts/make_data.py"
        )

    xb = np.load(xb_path)
    xq = np.load(xq_path)
    d = int(xb.shape[1])
    index = faiss.IndexFlatL2(d)
    index.add(xb)

    # Warm up (NOT inside the measured region).
    for i in range(10):
        s = (i * args.batch_size) % xq.shape[0]
        e = min(s + args.batch_size, xq.shape[0])
        index.search(xq[s:e], args.k)

    tp_info = threadpool_info() if threadpool_info else {
        "error": "threadpoolctl not installed",
    }

    # --- MEASURED REGION ---------------------------------------------------
    # Sentinel prints flush stdout so VTune's user-mode sampling timeline
    # can anchor to them even without ITT.
    print("PROFILE_REGION_START", flush=True)
    if _HAS_ITT:
        itt.resume()
        itt.task_begin("faiss_search_region")

    deadline = time.perf_counter() + args.duration_s
    batches_done = 0
    queries_done = 0
    sum_call_s = 0.0
    nq = xq.shape[0]
    bs = args.batch_size
    idx = 0
    t_region_start = time.perf_counter()
    while True:
        s = idx
        e = s + bs
        if e > nq:
            # Wrap at the end of the query set rather than slicing short,
            # so every batch has exactly `bs` queries.
            s = 0
            e = bs
            idx = 0
        qb = xq[s:e]
        t0 = time.perf_counter()
        index.search(qb, args.k)
        t1 = time.perf_counter()
        sum_call_s += (t1 - t0)
        batches_done += 1
        queries_done += bs
        idx = e
        if t1 >= deadline:
            break
    t_region_end = time.perf_counter()
    region_wall = t_region_end - t_region_start

    if _HAS_ITT:
        itt.task_end()
        itt.pause()
    print("PROFILE_REGION_END", flush=True)
    # ----------------------------------------------------------------------

    summary = {
        "schema_version": 1,
        "mode": "profile_workload",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "requested_threads": int(args.threads),
        "batch_size": int(args.batch_size),
        "k": int(args.k),
        "duration_target_s": float(args.duration_s),
        "region_wall_s": region_wall,
        "sum_call_s": sum_call_s,
        "batches_in_region": batches_done,
        "queries_in_region": queries_done,
        "throughput_qps_search_only":
            (queries_done / sum_call_s) if sum_call_s > 0 else None,
        "query_set_repeats_wrapped": True,
        "query_set_size": int(nq),
        "itt_available": _HAS_ITT,
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "faiss_omp_get_max_threads_after_set": int(
                faiss.omp_get_max_threads()
            ),
        },
        "runtime_pools": tp_info,
        "dataset": {
            "nb": int(xb.shape[0]), "d": int(d),
            "xb_sha256": sha256_file(xb_path),
            "xq_sha256": sha256_file(xq_path),
        },
        "faiss_version": faiss.__version__,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}  "
          f"batches={batches_done} queries={queries_done} "
          f"wall={region_wall:.3f}s "
          f"throughput={summary['throughput_qps_search_only']:.1f} qps")

    if args.vtune_pause > 0:
        time.sleep(args.vtune_pause)


if __name__ == "__main__":
    main()
