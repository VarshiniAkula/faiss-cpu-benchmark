"""Controlled P-core vs E-core experiment with interleaved placement.

For each repetition, this orchestrator deterministically shuffles the
three placement modes and runs ONE measurement per placement before
moving to the next repetition. That breaks up back-to-back repeats on
the same affinity set, which the earlier five-rep version showed were
sensitive to cold OMP thread-pool recreation and background OS work.

Placements:
  1. default_scheduler      -- no --cpu-affinity passed to the worker
  2. four_physical_p_cores  -- 4 P-cores, one logical CPU per physical
  3. four_physical_e_cores  -- 4 E-cores

Protocol highlights:
  - Each measurement is a fresh subprocess invocation of
    scripts/worker.py.
  - Both (threads=1, batch=1) and (threads=`--threads`, batch=32)
    records are produced per rep per placement, so the per-rep
    agreement reference is always present.
  - Query ordering within a rep is deterministic from (seed, rep), so
    all three placements see the *same query order* in the same rep.
  - Rep count defaults to 10 (minimum requested by the review).
  - Execution order of placements in each rep is logged to
    execution_log.json.
  - Battery/power-plan state is captured at start and end of the run
    (best-effort; recorded as unknown if the tooling fails).
  - Observed process affinity is checked against the requested set in
    every per-call record (worker.py already captures this); this
    orchestrator additionally cross-checks and summarises.

Output layout (schema 2-compatible):
  base_dir/
    manifest.json          top-level manifest for the whole experiment
    topology.json          topology snapshot + chosen CPU ids
    execution_log.json     per-rep placement order and timing
    power_state.json       battery/plan at start and end
    comparison.json        cross-placement summary
    sub_<label>/
      manifest.json        per-placement sub-run manifest (for analyze.py)
      machine.json         copied once from inspect_machine.py
      per_call/*.json      N reps per (t,b) config
      run_order.json       entries for records produced by this sub
      status.json          completed when all expected records present
      ...analysis artefacts after analyze.py runs
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
from topology import enumerate_topology, pick_one_logical_per_physical  # noqa

ANALYZE = ROOT / "scripts" / "analyze.py"
WORKER = ROOT / "scripts" / "worker.py"
INSPECT = ROOT / "scripts" / "inspect_machine.py"


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ts_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_meta() -> dict:
    def _git(args):
        try:
            r = subprocess.run(["git", *args], cwd=ROOT,
                               capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    commit = _git(["rev-parse", "HEAD"])
    short = _git(["rev-parse", "--short", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    try:
        src_dirty = subprocess.run(
            ["git", "diff", "--quiet"], cwd=ROOT,
            capture_output=True, timeout=5,
        ).returncode != 0 or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT,
            capture_output=True, timeout=5,
        ).returncode != 0
    except Exception:
        src_dirty = None
    return {
        "commit": commit, "commit_short": short, "branch": branch,
        "source_dirty": src_dirty,
    }


def pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception as e:
        return f"__err__:{e}"


def capture_power_state() -> dict:
    """Best-effort battery/power-plan snapshot on Windows."""
    out = {"captured_at_utc": utc_now_iso()}
    if sys.platform != "win32":
        out["note"] = "non-Windows: not collected"
        return out
    def _ps(cmd):
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                capture_output=True, text=True, timeout=10,
            )
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception as e:
            return f"__err__:{e}"
    bat = _ps(
        "$b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue; "
        "if ($b) { $b | Select-Object -First 1 BatteryStatus,"
        "EstimatedChargeRemaining | ConvertTo-Json -Compress } else {'{}'}"
    )
    try:
        out["battery"] = json.loads(bat) if bat else {}
    except Exception:
        out["battery_raw"] = bat
    out["active_power_plan"] = _ps("(powercfg /getactivescheme) 2>&1")
    return out


def make_env(threads: int) -> dict:
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[var] = str(threads)
    return env


def write_sub_manifest(sub_dir: Path, placement_label: str,
                       affinity: list[int] | None, reps: int,
                       seed: int, queries_per_rep: int, k: int,
                       threads: int, dataset_hashes: dict,
                       git: dict) -> dict:
    configs = [[1, 1], [threads, 1], [1, 32], [threads, 32]]
    # Deduplicate (if threads==1, configs collapse).
    configs = [list(x) for x in dict.fromkeys(tuple(c) for c in configs)]
    expected_entries = [[t, b, r] for r in range(reps) for (t, b) in configs]
    manifest = {
        "schema_version": 2,
        "run_dir": str(sub_dir),
        "mode": "full",
        "started_at_utc": utc_now_iso(),
        "seed": seed,
        "queries_per_rep": queries_per_rep,
        "k": k,
        "configs": configs,
        "reps": reps,
        "expected_entries": expected_entries,
        "dataset_sha256": dataset_hashes,
        "git": git,
        "python_version": sys.version.split()[0],
        "packages": {
            "faiss-cpu": pkg_version("faiss-cpu"),
            "numpy": pkg_version("numpy"),
            "matplotlib": pkg_version("matplotlib"),
            "threadpoolctl": pkg_version("threadpoolctl"),
            "psutil": pkg_version("psutil"),
        },
        "cpu_affinity_requested": affinity,
        "pe_placement_label": placement_label,
    }
    (sub_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run_worker(sub_dir: Path, threads: int, batch_size: int, rep: int,
               seed: int, queries_per_rep: int, k: int,
               dataset_hashes: dict,
               cpu_affinity: list[int] | None,
               data_dir: Path) -> dict:
    per_call = sub_dir / "per_call"
    per_call.mkdir(parents=True, exist_ok=True)
    out = per_call / f"b{batch_size}_t{threads}_rep{rep}.json"
    cmd = [
        sys.executable, str(WORKER),
        "--threads", str(threads),
        "--batch-size", str(batch_size),
        "--rep", str(rep),
        "--seed", str(seed),
        "--queries-per-rep", str(queries_per_rep),
        "--k", str(k),
        "--data-dir", str(data_dir),
        "--out", str(out),
        "--expected-xb-sha", dataset_hashes["xb_sha256"],
        "--expected-xq-sha", dataset_hashes["xq_sha256"],
    ]
    if cpu_affinity is not None:
        cmd.extend(["--cpu-affinity", ",".join(str(c) for c in cpu_affinity)])
    env = make_env(threads)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        print("STDERR:", r.stderr, file=sys.stderr)
        raise SystemExit(
            f"worker failed (rc={r.returncode}) placement={sub_dir.name} "
            f"t={threads} b={batch_size} rep={rep}"
        )
    return {"threads": threads, "batch_size": batch_size, "rep": rep,
            "sub": sub_dir.name, "wall_seconds": wall,
            "out_path": str(out)}


def write_status(sub_dir: Path, manifest: dict) -> dict:
    expected = {tuple(e) for e in manifest["expected_entries"]}
    present = {}
    for p in (sub_dir / "per_call").glob("*.json"):
        try:
            rec = json.loads(p.read_text())
            c = rec["config"]
            key = (int(c["threads"]), int(c["batch_size"]), int(c["rep"]))
            present.setdefault(key, []).append(str(p))
        except Exception:
            continue
    missing = sorted(expected - set(present.keys()))
    extra = sorted(set(present.keys()) - expected)
    dups = sorted(k for k, files in present.items() if len(files) > 1)
    state = ("completed" if not missing and not extra and not dups
             else "incomplete")
    status = {
        "state": state, "mode": manifest["mode"],
        "finished_at_utc": utc_now_iso(),
        "n_records_present": len(present),
        "n_entries_expected": len(expected),
        "missing": [list(x) for x in missing],
        "extra": [list(x) for x in extra],
        "duplicates": [list(x) for x in dups],
    }
    (sub_dir / "status.json").write_text(json.dumps(status, indent=2))
    return status


def cross_check_observed_affinity(sub_dir: Path, expected: list[int] | None
                                  ) -> dict:
    """Sanity: every per-call record's observed affinity must match the
    expected set (or the full system set when expected is None)."""
    mismatches: list[dict] = []
    checked = 0
    for p in sorted((sub_dir / "per_call").glob("*.json")):
        rec = json.loads(p.read_text())
        obs = rec.get("cpu_affinity", {}).get("observed_cpu_ids")
        if obs is None:
            continue
        checked += 1
        if expected is None:
            # Any observed is fine; just record.
            continue
        if sorted(obs) != sorted(expected):
            mismatches.append({
                "file": str(p), "observed": obs, "expected": expected,
            })
    return {"records_checked": checked, "mismatches": mismatches}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--threads", type=int, default=4,
                   help="threads requested in every placement (batch=32)")
    p.add_argument("--n-cores", type=int, default=4,
                   help="physical cores per pool")
    p.add_argument("--queries-per-rep", type=int, default=512)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    args = p.parse_args()

    if args.reps < 10:
        raise SystemExit(
            f"this PE orchestrator requires --reps >= 10 "
            f"(review-mandated minimum); got {args.reps}"
        )

    git = git_meta()
    topo = enumerate_topology()
    if (topo.get("p_core_efficiency_class") is None
            or topo.get("n_p_logical", 0) == 0
            or topo.get("n_e_logical", 0) == 0):
        print(json.dumps(topo, indent=2))
        raise SystemExit("topology does not expose P+E split; halting")
    try:
        p_cpus = pick_one_logical_per_physical(topo, "p", args.n_cores)
        e_cpus = pick_one_logical_per_physical(topo, "e", args.n_cores)
    except ValueError as e:
        raise SystemExit(str(e))

    base_dir = (Path(args.base_dir) if args.base_dir
                else ROOT / "runs" / f"{ts_slug()}-pe2-"
                     f"{git.get('commit_short') or 'no-git'}")
    base_dir.mkdir(parents=True, exist_ok=True)

    topo["chosen_p_core_cpus"] = p_cpus
    topo["chosen_e_core_cpus"] = e_cpus
    topo["experiment_config"] = {
        "threads": args.threads, "reps": args.reps,
        "n_cores_per_pool": args.n_cores,
        "queries_per_rep": args.queries_per_rep, "k": args.k,
        "seed": args.seed,
        "protocol": "interleaved_placement_per_repetition_v2",
    }
    (base_dir / "topology.json").write_text(json.dumps(topo, indent=2))

    data_dir = Path(args.data_dir)
    dataset_hashes = {
        "xb_sha256": sha256_file(data_dir / "xb.npy"),
        "xq_sha256": sha256_file(data_dir / "xq.npy"),
    }

    placements = [
        ("default_scheduler", None),
        (f"{args.n_cores}_physical_p_cores", p_cpus),
        (f"{args.n_cores}_physical_e_cores", e_cpus),
    ]

    # One shared machine snapshot at experiment start.
    subprocess.run(
        [sys.executable, str(INSPECT), "--out",
         str(base_dir / "machine.json")],
        check=False, capture_output=True, text=True, timeout=60,
    )

    # Prepare each sub-run directory with its manifest.
    sub_dirs: dict[str, Path] = {}
    manifests: dict[str, dict] = {}
    for label, affinity in placements:
        sub = base_dir / f"sub_{label}"
        sub.mkdir(parents=True, exist_ok=True)
        # Copy machine info to each sub so analyze works self-contained.
        (sub / "machine.json").write_bytes(
            (base_dir / "machine.json").read_bytes()
        )
        manifests[label] = write_sub_manifest(
            sub, label, affinity, args.reps, args.seed,
            args.queries_per_rep, args.k, args.threads, dataset_hashes, git,
        )
        sub_dirs[label] = sub

    power_start = capture_power_state()

    # Interleaved execution: for each rep, shuffle placement order.
    exec_log: list[dict] = []
    run_order_by_sub: dict[str, list[dict]] = {lab: [] for lab, _ in placements}
    configs_per_placement = [
        (1, 1),                   # agreement reference
        (args.threads, 1),        # batch-1 at target threads
        (1, 32),                  # batch-32 baseline
        (args.threads, 32),       # batch-32 at target threads (headline)
    ]
    # Deduplicate if threads==1.
    configs_per_placement = list(dict.fromkeys(configs_per_placement))

    for rep in range(args.reps):
        order = list(placements)
        random.Random(f"pe:{args.seed}:{rep}").shuffle(order)
        order_log = [lbl for lbl, _ in order]
        print(f"=== rep {rep} placement order: {order_log} ===")
        for label, affinity in order:
            # Within a placement, configs in a deterministic shuffled order
            # keyed by (seed, rep, label) to prevent any config preferring
            # a particular placement due to run-order effects.
            cfgs = list(configs_per_placement)
            random.Random(f"pe-cfg:{args.seed}:{rep}:{label}").shuffle(cfgs)
            for (t, b) in cfgs:
                started = utc_now_iso()
                rec = run_worker(
                    sub_dirs[label], t, b, rep, args.seed,
                    args.queries_per_rep, args.k, dataset_hashes,
                    affinity, data_dir,
                )
                rec["placement_label"] = label
                rec["started_at_utc"] = started
                rec["finished_at_utc"] = utc_now_iso()
                exec_log.append(rec)
                run_order_by_sub[label].append(rec)
                print(f"  [{label}] b={b} t={t} rep={rep} "
                      f"wall={rec['wall_seconds']:.2f}s")

    # Per-sub run_order and status.
    for label in sub_dirs:
        (sub_dirs[label] / "run_order.json").write_text(
            json.dumps(run_order_by_sub[label], indent=2)
        )

    power_end = capture_power_state()
    (base_dir / "power_state.json").write_text(json.dumps({
        "start": power_start, "end": power_end,
    }, indent=2))
    (base_dir / "execution_log.json").write_text(json.dumps(exec_log, indent=2))

    # Compute status + affinity cross-check per sub.
    status_summary = {}
    affinity_checks = {}
    for label, affinity in placements:
        status = write_status(sub_dirs[label], manifests[label])
        status_summary[label] = status
        affinity_checks[label] = cross_check_observed_affinity(
            sub_dirs[label], affinity
        )

    # Analyze each sub.
    for label in sub_dirs:
        r = subprocess.run(
            [sys.executable, str(ANALYZE), "--run-dir",
             str(sub_dirs[label])],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise SystemExit(
                f"analyze failed for {label}:\n{r.stderr[-800:]}"
            )

    # Build comparison.json: pick each placement's (threads, batch=32) row.
    def pick_row(label: str) -> dict | None:
        recs = json.loads(
            (sub_dirs[label] / "recommendations.json").read_text()
        )
        rows = recs["throughput_batch32_qps"].get("all_configs", [])
        for row in rows:
            if row["threads"] == args.threads:
                return row
        return None

    comparison = {
        "experiment": "p_vs_e_core_v2",
        "protocol": "interleaved_placement_per_repetition",
        "base_dir": str(base_dir),
        "git": git,
        "threads_requested": args.threads,
        "n_physical_cores_per_pool": args.n_cores,
        "reps": args.reps,
        "queries_per_rep": args.queries_per_rep,
        "chosen_p_core_cpus": p_cpus,
        "chosen_e_core_cpus": e_cpus,
        "sub_runs": [
            {
                "label": label,
                "affinity_requested": affinity,
                "sub_dir": str(sub_dirs[label]),
                "status": status_summary[label],
                "affinity_check": affinity_checks[label],
                "throughput_batch32_at_target_threads": pick_row(label),
            }
            for label, affinity in placements
        ],
    }
    (base_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(f"\nwrote {base_dir / 'comparison.json'}")
    print(f"\n=== P/E v2 comparison (median batch-32 throughput, qps, "
          f"at {args.threads} requested threads; {args.reps} interleaved reps) ===")
    for sr in comparison["sub_runs"]:
        row = sr["throughput_batch32_at_target_threads"]
        if row is None:
            print(f"  {sr['label']:>30}: (no row)")
            continue
        m, mm = row["median"], row["min_max"]
        print(f"  {sr['label']:>30}: median {m:.1f} qps "
              f"[{mm[0]:.1f}, {mm[1]:.1f}]  "
              f"(status={sr['status']['state']}, "
              f"affinity_mismatches="
              f"{len(sr['affinity_check']['mismatches'])})")


if __name__ == "__main__":
    main()
