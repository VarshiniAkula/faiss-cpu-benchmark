"""Focused regression checks. Runnable as `python tests/test_regression.py`.

Each check prints PASS/FAIL. Non-zero exit if any fail.

Checks (as required in the review):
  1. Isolated smoke run analyzes successfully.
  2. Smoke and full runs remain separate (manifest/mode guard).
  3. Incomplete full run is detected and refused by analyze.
  4. Mismatched query/ID arrays are detected by validator.
  5. Throughput is computed from sum(call_durations_s), not elapsed-around-loop.
  6. Recommendation output is consistent with its documented decision rule.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import validate as v  # noqa: E402
from analyze import (  # noqa: E402
    aggregate, per_run_metric, recommendations, decide_clear_winner,
)

PY = sys.executable
FAIL: list[str] = []
PASS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"PASS  {name}")
        PASS.append(name)
    else:
        print(f"FAIL  {name}  {detail}")
        FAIL.append(f"{name}: {detail}")


def ensure_data() -> None:
    if not (ROOT / "data" / "xb.npy").exists():
        subprocess.run([PY, str(ROOT / "scripts" / "make_data.py")],
                       check=True)


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def test_smoke_analysis_success(tmp: Path) -> None:
    run_dir = tmp / "smoke"
    r = run_cmd([PY, str(ROOT / "scripts" / "run_bench.py"),
                 "--mode", "smoke", "--run-dir", str(run_dir)])
    check("1a smoke run succeeds", r.returncode == 0,
          f"rc={r.returncode}\n{r.stderr}")
    a = run_cmd([PY, str(ROOT / "scripts" / "analyze.py"),
                 "--run-dir", str(run_dir)])
    check("1b smoke analyze succeeds", a.returncode == 0,
          f"rc={a.returncode}\n{a.stderr}")
    # Smoke produced recommendations with status=='single_configuration_only'
    # for b=1, and insufficient_data for b=32.
    recs = json.loads((run_dir / "recommendations.json").read_text())
    check(
        "1c smoke recommendation: interactive status is single_configuration_only "
        "or no_clear_winner (not error)",
        recs["interactive_batch1_p95_ms"]["status"] in (
            "single_configuration_only", "no_clear_winner"),
        str(recs["interactive_batch1_p95_ms"]),
    )
    check(
        "1d smoke recommendation: throughput status is insufficient_data",
        recs["throughput_batch32_qps"]["status"] == "insufficient_data",
        str(recs["throughput_batch32_qps"]),
    )
    notes = json.loads((run_dir / "analysis_notes.json").read_text())
    check(
        "1e smoke notes lists comparisons_not_performed",
        bool(notes["comparisons_not_performed"]),
        str(notes),
    )


def test_smoke_full_separation(tmp: Path) -> None:
    # Create a smoke in run_dir, then try to run FULL into the SAME dir.
    run_dir = tmp / "shared"
    r1 = run_cmd([PY, str(ROOT / "scripts" / "run_bench.py"),
                  "--mode", "smoke", "--run-dir", str(run_dir)])
    check("2a smoke seeds a shared dir", r1.returncode == 0, r1.stderr)
    r2 = run_cmd([PY, str(ROOT / "scripts" / "run_bench.py"),
                  "--mode", "full", "--run-dir", str(run_dir),
                  "--reps", "1"])
    check("2b full refuses to overwrite smoke's dir",
          r2.returncode != 0 and "smoke" in (r2.stderr + r2.stdout).lower(),
          f"rc={r2.returncode}\n{r2.stderr}\n{r2.stdout}")


def test_incomplete_full_detected(tmp: Path) -> None:
    # Produce a "full" run with only a subset of records, then analyze.
    run_dir = tmp / "incomplete"
    # Start a full run with reps=1 so we only need 6 records; then delete one.
    r = run_cmd([PY, str(ROOT / "scripts" / "run_bench.py"),
                 "--mode", "full", "--run-dir", str(run_dir),
                 "--reps", "1"])
    # Even if r.returncode is 2 (incomplete), proceed; we want to test the
    # analyze path.
    # Delete one record to simulate an incomplete run.
    pc = run_dir / "per_call"
    files = sorted(pc.glob("*.json"))
    if files:
        files[0].unlink()
    a = run_cmd([PY, str(ROOT / "scripts" / "analyze.py"),
                 "--run-dir", str(run_dir)])
    check("3 analyze refuses incomplete full run",
          a.returncode != 0 and "incomplete" in (a.stderr + a.stdout).lower(),
          f"rc={a.returncode}\nstderr={a.stderr}\nstdout={a.stdout}")


def test_mismatched_query_id_arrays() -> None:
    # Construct a synthetic record where top5_ids length != queries_per_rep.
    rec = {
        "config": {"threads": 1, "batch_size": 1, "rep": 0},
        "dataset": {"nb": 10, "d": 4, "k": 5, "queries_per_rep": 3},
        "call_durations_s": [0.001, 0.001, 0.001],
        "sum_call_durations_s": 0.003,
        "elapsed_around_loop_s": 0.003,
        "queries_processed": 3,
        "top5_ids": [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],  # length 2 not 3
        "query_order": [0, 1, 2],
    }
    try:
        v.validate_record(rec)
        check("4a mismatched top5_ids detected", False,
              "validator did not raise")
    except ValueError as e:
        check("4a mismatched top5_ids detected", "top5_ids length" in str(e),
              str(e))

    rec2 = dict(rec); rec2["top5_ids"] = [[0, 1, 2, 3, 4]] * 3
    rec2["query_order"] = [0, 1]  # wrong length
    try:
        v.validate_record(rec2)
        check("4b mismatched query_order detected", False,
              "validator did not raise")
    except ValueError as e:
        check("4b mismatched query_order detected",
              "query_order" in str(e), str(e))


def test_throughput_from_sum_calls() -> None:
    # If elapsed-around-loop > sum of calls (always true in practice),
    # throughput_qps_search_only > queries / elapsed-around-loop.
    rec = {
        "config": {"threads": 1, "batch_size": 32, "rep": 0},
        "call_durations_s": [0.01] * 16,          # sum = 0.16
        "sum_call_durations_s": 0.16,
        "elapsed_around_loop_s": 0.20,           # 25% overhead
        "queries_processed": 512,
    }
    m = per_run_metric({**rec, "dataset": {"nb": 10, "d": 4, "k": 5,
                                            "queries_per_rep": 512}})
    expected_search_only = 512 / 0.16
    bad_elapsed = 512 / 0.20
    check(
        "5a throughput uses sum(call_durations_s)",
        abs(m["throughput_qps_search_only"] - expected_search_only) < 1e-6,
        f"got {m['throughput_qps_search_only']} vs {expected_search_only}",
    )
    check(
        "5b throughput is NOT queries/elapsed_around_loop",
        abs(m["throughput_qps_search_only"] - bad_elapsed) > 1.0,
        f"got {m['throughput_qps_search_only']} vs bad={bad_elapsed}",
    )


def test_decision_rule_consistency() -> None:
    # Case A: throughput best clearly above all other mins -> clear_winner.
    agg_clear = [
        {"threads": 1, "batch_size": 32, "n_reps": 5, "metric": "qps",
         "median": 1000.0, "min": 980.0, "max": 1020.0, "all_reps": []},
        {"threads": 2, "batch_size": 32, "n_reps": 5, "metric": "qps",
         "median": 1500.0, "min": 1450.0, "max": 1550.0, "all_reps": []},
        {"threads": 4, "batch_size": 32, "n_reps": 5, "metric": "qps",
         "median": 2500.0, "min": 2400.0, "max": 2600.0, "all_reps": []},
    ]
    r = recommendations(agg_clear)
    check(
        "6a clear winner reported when ranges don't overlap the best median",
        r["throughput_batch32_qps"]["status"] == "clear_winner",
        str(r["throughput_batch32_qps"]),
    )
    # Case B: overlapping ranges => no_clear_winner.
    agg_overlap = [
        {"threads": 1, "batch_size": 1, "n_reps": 5, "metric": "p95",
         "median": 4.0, "min": 3.7, "max": 4.1, "all_reps": []},
        {"threads": 2, "batch_size": 1, "n_reps": 5, "metric": "p95",
         "median": 3.95, "min": 3.8, "max": 4.3, "all_reps": []},
        {"threads": 4, "batch_size": 1, "n_reps": 5, "metric": "p95",
         "median": 3.9, "min": 3.8, "max": 4.2, "all_reps": []},
    ]
    r2 = recommendations(agg_overlap)
    check(
        "6b no clear winner reported when ranges overlap the best median",
        r2["interactive_batch1_p95_ms"]["status"] == "no_clear_winner",
        str(r2["interactive_batch1_p95_ms"]),
    )
    # Case C: decide_clear_winner is directly testable (sanity).
    best = agg_overlap[2]
    check(
        "6c decide_clear_winner: overlap => False",
        decide_clear_winner(agg_overlap, best, lower_better=True) is False,
    )
    best = agg_clear[2]
    check(
        "6d decide_clear_winner: non-overlap => True",
        decide_clear_winner(agg_clear, best, lower_better=False) is True,
    )


def main() -> int:
    ensure_data()
    with tempfile.TemporaryDirectory(prefix="faissbench_test_") as tmpd:
        tmp = Path(tmpd)
        test_smoke_analysis_success(tmp)
        test_smoke_full_separation(tmp)
        test_incomplete_full_detected(tmp)
    test_mismatched_query_id_arrays()
    test_throughput_from_sum_calls()
    test_decision_rule_consistency()
    print()
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(" -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
