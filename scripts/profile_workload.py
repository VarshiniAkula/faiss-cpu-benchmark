"""Fixed-duration search workload for external profiling (e.g. Intel VTune).

Design:
  - Data load and index build happen BEFORE the measured region.
  - 10 warm-up search calls happen BEFORE the measured region.
  - Inside the measured region, the same 512 queries are run in batches
    of the requested batch size, in a tight loop, repeated across the
    full query set, for a configurable wall-clock duration.
  - Queries are REPEATED (wrapped) at the end of the batch stream.
    This is intentional for a profiler run so the measured region is a
    single, stable workload. Headline performance numbers must come
    from scripts/run_bench.py, not this script.

Region control for VTune:
  - If the Intel ITT notify DLL (libittnotify / libittnotify_collector)
    is loadable, this script calls __itt_resume() when entering the
    measured region and __itt_pause() when leaving it. When VTune is
    launched with `-start-paused`, data collection is enabled only
    while the region is active, so index load / build / warm-up are
    NOT part of the collected sample set. This is the "region
    filtering that excludes loading, index construction, and warm-up"
    requirement.
  - When ITT is not loadable, the script still emits
    PROFILE_REGION_START / PROFILE_REGION_END sentinel lines but
    records itt_available=false in its JSON summary. Console prints
    are NOT by themselves a filter; the JSON flag makes this explicit
    to downstream readers.
  - The sampled event counters and the "queries_in_region" count are
    produced from the same region start/stop points, so matched-work
    normalization downstream (cycles/query etc.) is well defined.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path


# ------------------- ITT loader -----------------------------------------
def _load_ittnotify() -> tuple[object | None, str]:
    """Try several standard VTune install locations for libittnotify."""
    if sys.platform != "win32":
        for name in ("libittnotify.so", "libittnotify_collector.so"):
            try:
                return ctypes.CDLL(name), name
            except OSError:
                pass
        return None, "not found (non-Windows)"
    candidate_dirs: list[Path] = []
    # 1. User-pointed override.
    env = os.environ.get("ITT_LIB_DIR") or os.environ.get("VTUNE_PROFILER_DIR")
    if env:
        candidate_dirs.append(Path(env))
    # 2. oneAPI standard install (any version).
    oneapi_vtune = Path(r"C:\Program Files (x86)\Intel\oneAPI\vtune")
    if oneapi_vtune.exists():
        for v in sorted(oneapi_vtune.iterdir(), reverse=True):
            if v.is_dir():
                candidate_dirs.extend([
                    v / "sdk" / "lib64",
                    v / "sdk" / "lib",
                    v / "bin64",
                ])
    # 3. Legacy VTune Amplifier.
    legacy = Path(r"C:\Program Files (x86)\IntelSWTools")
    if legacy.exists():
        for v in sorted(legacy.iterdir(), reverse=True):
            if v.is_dir() and v.name.lower().startswith("vtune"):
                candidate_dirs.extend([
                    v / "sdk" / "lib64",
                    v / "bin64",
                ])
    dll_names = (
        "libittnotify.dll", "libittnotify_collector.dll",
        "ittnotify_collector.dll",
    )
    tried: list[str] = []
    for d in candidate_dirs:
        for name in dll_names:
            full = d / name
            tried.append(str(full))
            if full.exists():
                try:
                    return ctypes.CDLL(str(full)), str(full)
                except OSError:
                    continue
    # 4. Plain name on PATH (if setvars has been sourced).
    for name in dll_names:
        tried.append(name)
        try:
            return ctypes.CDLL(name), name
        except OSError:
            continue
    return None, f"not found (tried: {len(tried)} paths)"


_ITT_LIB, _ITT_LIB_PATH = _load_ittnotify()
_HAS_ITT = False
if _ITT_LIB is not None:
    try:
        _ITT_LIB.__itt_resume.restype = None
        _ITT_LIB.__itt_pause.restype = None
        _HAS_ITT = True
    except AttributeError:
        _HAS_ITT = False


def itt_resume() -> None:
    if _HAS_ITT:
        _ITT_LIB.__itt_resume()


def itt_pause() -> None:
    if _HAS_ITT:
        _ITT_LIB.__itt_pause()


# ------------------- misc -----------------------------------------------
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
    p.add_argument("--data-dir", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "data"))
    p.add_argument("--out", type=str, required=True,
                   help="JSON summary output path")
    p.add_argument("--settling-s", type=float, default=2.0,
                   help="sleep after warm-up, before the measured region, "
                        "to let the collector ramp (default 2s)")
    p.add_argument("--vtune-pause-s", type=float, default=0.0,
                   help="seconds to sleep after the measured region "
                        "(gives external collectors time to flush)")
    args = p.parse_args()

    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be > 0")

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

    # With ITT, make sure the collector is PAUSED while we set up.
    # (Harmless if the collector isn't running.)
    itt_pause()

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

    if args.settling_s > 0:
        time.sleep(args.settling_s)

    # --- MEASURED REGION ---------------------------------------------------
    print("PROFILE_REGION_START", flush=True)
    itt_resume()

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

    itt_pause()
    print("PROFILE_REGION_END", flush=True)
    # ----------------------------------------------------------------------

    summary = {
        "schema_version": 2,
        "mode": "profile_workload",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "requested_threads": int(args.threads),
        "batch_size": int(args.batch_size),
        "k": int(args.k),
        "duration_target_s": float(args.duration_s),
        "settling_s": float(args.settling_s),
        "region_wall_s": region_wall,
        "sum_call_s": sum_call_s,
        "batches_in_region": batches_done,
        "queries_in_region": queries_done,
        "throughput_qps_search_only":
            (queries_done / sum_call_s) if sum_call_s > 0 else None,
        "query_set_repeats_wrapped": True,
        "query_set_size": int(nq),
        "itt_available": _HAS_ITT,
        "itt_library_path": _ITT_LIB_PATH,
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
          f"throughput={summary['throughput_qps_search_only']:.1f} qps "
          f"itt_available={_HAS_ITT}")

    if args.vtune_pause_s > 0:
        time.sleep(args.vtune_pause_s)


if __name__ == "__main__":
    main()
