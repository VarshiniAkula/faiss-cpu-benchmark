"""Controlled P-core vs E-core experiment.

Runs three isolated full-benchmark runs with identical workload
(batch=32, 4 threads) but different process-level CPU affinity:
  1. default_scheduler          -- no affinity restriction
  2. four_physical_p_cores      -- 4 P-cores, one logical CPU per core
  3. four_physical_e_cores      -- 4 E-cores

Each sub-run is a fresh run directory containing its own manifest,
per-call records, status, topology snapshot, and analysis.

Topology discovery uses scripts/topology.py (which calls
GetLogicalProcessorInformationEx). If P/E cannot be identified, the
experiment stops with a clear message.

Process-level affinity is applied by the worker BEFORE importing
numpy/faiss; this restricts the set of CPUs the OS scheduler may place
worker threads on. It is NOT per-thread pinning; thread-to-core
binding within the restricted set remains under the scheduler.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from topology import enumerate_topology, pick_one_logical_per_physical  # noqa

RUN_BENCH = ROOT / "scripts" / "run_bench.py"
ANALYZE = ROOT / "scripts" / "analyze.py"


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_short() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=ROOT, capture_output=True, text=True,
                           timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def run_one_subrun(label: str, affinity: list[int] | None,
                   base_dir: Path, reps: int, threads: int) -> dict:
    sub = base_dir / f"sub_{label}"
    sub.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUN_BENCH),
        "--mode", "full",
        "--run-dir", str(sub),
        "--reps", str(reps),
        "--threads-batch1", str(threads),      # single thread value
        "--threads-batch32", str(threads),
    ]
    if affinity is not None:
        cmd.extend(["--cpu-affinity", ",".join(str(c) for c in affinity)])
    print(f"\n=== sub-run: {label}  affinity={affinity} ===")
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        raise SystemExit(f"sub-run {label} failed (rc={r.returncode})")
    # Analyse.
    r2 = subprocess.run(
        [sys.executable, str(ANALYZE), "--run-dir", str(sub)],
        capture_output=True, text=True,
    )
    if r2.returncode != 0:
        raise SystemExit(
            f"sub-run {label} analyze failed: {r2.stderr[-500:]}"
        )
    recs = json.loads((sub / "recommendations.json").read_text())
    return {
        "label": label,
        "affinity": affinity,
        "sub_dir": str(sub),
        "throughput_batch32": recs["throughput_batch32_qps"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=None,
                   help="output base directory (holds three sub runs)")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--threads", type=int, default=4,
                   help="threads to request in every sub-run")
    p.add_argument("--n-cores", type=int, default=4,
                   help="physical cores to restrict to (per pool)")
    args = p.parse_args()

    topo = enumerate_topology()
    if (topo.get("p_core_efficiency_class") is None
            or topo.get("n_p_logical", 0) == 0
            or topo.get("n_e_logical", 0) == 0):
        print(json.dumps(topo, indent=2))
        raise SystemExit(
            "Topology does not expose both P- and E-cores; halting P/E "
            "experiment."
        )
    try:
        p_cpus = pick_one_logical_per_physical(topo, "p", args.n_cores)
        e_cpus = pick_one_logical_per_physical(topo, "e", args.n_cores)
    except ValueError as e:
        raise SystemExit(str(e))

    base_dir = (
        Path(args.base_dir) if args.base_dir else
        ROOT / "runs" / f"{utc_now_iso()}-pe-{git_short()}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    topo["chosen_p_core_cpus"] = p_cpus
    topo["chosen_e_core_cpus"] = e_cpus
    topo["experiment_config"] = {
        "threads": args.threads, "reps": args.reps,
        "n_cores_per_pool": args.n_cores,
    }
    (base_dir / "topology.json").write_text(json.dumps(topo, indent=2))

    results = []
    for label, affinity in (
        ("default_scheduler", None),
        (f"{args.n_cores}_physical_p_cores", p_cpus),
        (f"{args.n_cores}_physical_e_cores", e_cpus),
    ):
        results.append(run_one_subrun(
            label, affinity, base_dir, args.reps, args.threads,
        ))

    # Compile a tiny cross-run summary.
    comparison = {
        "experiment": "p_vs_e_core",
        "base_dir": str(base_dir),
        "threads_requested": args.threads,
        "n_physical_cores_per_pool": args.n_cores,
        "chosen_p_core_cpus": p_cpus,
        "chosen_e_core_cpus": e_cpus,
        "sub_runs": results,
    }
    (base_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(f"\nwrote {base_dir / 'comparison.json'}")
    print("\n=== P vs E comparison (median batch-32 throughput, qps) ===")
    for r in results:
        t = r["throughput_batch32"]
        if "best_observed" in t:
            m = t["best_observed"]["median"]
            mm = t["best_observed"]["min_max"]
        else:
            m = t["only_measured"]["median"]
            mm = t["only_measured"]["min_max"]
        print(f"  {r['label']:>30}: median {m:.1f} qps "
              f"[{mm[0]:.1f}, {mm[1]:.1f}]")


if __name__ == "__main__":
    main()
