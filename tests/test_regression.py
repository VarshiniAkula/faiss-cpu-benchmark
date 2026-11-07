"""Focused regression checks. Runnable as `python tests/test_regression.py`.

Each check prints PASS/FAIL. Non-zero exit if any fail.

Existing checks (from initial refactor):
  1. Isolated smoke run analyzes successfully.
  2. Smoke and full runs remain separate (manifest/mode guard).
  3. Incomplete full run is detected and refused by analyze.
  4. Mismatched query/ID arrays are detected by validator.
  5. Throughput is computed from sum(call_durations_s).
  6. Recommendation output is consistent with its documented decision rule.

New checks (this review):
  7.  Forged sum_call_durations_s (tenfold) is rejected by the validator.
  8.  Nonfinite or nonpositive sum_call_durations_s is rejected.
  9.  elapsed_around_loop_s < sum(call_durations_s) is rejected.
  10. Manifest queries_per_rep mismatch is rejected by analyze.
  11. Manifest k mismatch is rejected by analyze.
  12. Manifest dataset hash mismatch is rejected by analyze.
  13. Empty smoke run (manifest expects 1, no records) is rejected by analyze.
  14. Duplicate smoke record is rejected by analyze.
  15. Extra/unexpected smoke record is rejected by analyze.
  16. Overwrite without --allow-overwrite is refused.
  17. Overwrite WITH --allow-overwrite clears prior records (no mixing).
  18. Interrupted / failed run writes status=failed and does not mark completed.
  19. n_timed_calls inconsistent with call_durations_s length is rejected.
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


# ---------- small helpers for synthetic records ----------

def good_record(threads=1, batch_size=1, rep=0,
                queries_per_rep=4, k=5,
                nb=10, d=4,
                xb_sha=None, xq_sha=None) -> dict:
    """A validator-clean per-call record used as a base for tampering tests."""
    expected_calls = queries_per_rep if batch_size == 1 else queries_per_rep // batch_size
    durs = [0.001] * expected_calls
    return {
        "config": {"threads": threads, "batch_size": batch_size, "rep": rep},
        "dataset": {
            "nb": nb, "nq_total": queries_per_rep, "d": d, "k": k,
            "queries_per_rep": queries_per_rep,
            "xb_sha256": xb_sha, "xq_sha256": xq_sha,
        },
        "call_durations_s": durs,
        "sum_call_durations_s": float(sum(durs)),
        "elapsed_around_loop_s": float(sum(durs)) + 0.001,
        "n_timed_calls": len(durs),
        "queries_processed": queries_per_rep,
        "top5_ids": [[0, 1, 2, 3, 4]] * queries_per_rep,
        "query_order": list(range(queries_per_rep)),
    }


# ---------- existing regression cases (kept intact) ----------

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
    recs = json.loads((run_dir / "recommendations.json").read_text())
    check(
        "1c smoke interactive recommendation: single_configuration_only",
        recs["interactive_batch1_p95_ms"]["status"] in (
            "single_configuration_only", "no_clear_winner"),
        str(recs["interactive_batch1_p95_ms"]),
    )
    check(
        "1d smoke throughput recommendation: insufficient_data",
        recs["throughput_batch32_qps"]["status"] == "insufficient_data",
        str(recs["throughput_batch32_qps"]),
    )
    notes = json.loads((run_dir / "analysis_notes.json").read_text())
    check(
        "1e smoke notes lists comparisons_not_performed",
        bool(notes["comparisons_not_performed"]), str(notes),
    )


def test_smoke_full_separation(tmp: Path) -> None:
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
    run_dir = tmp / "incomplete"
    run_cmd([PY, str(ROOT / "scripts" / "run_bench.py"),
             "--mode", "full", "--run-dir", str(run_dir),
             "--reps", "1"])
    pc = run_dir / "per_call"
    files = sorted(pc.glob("*.json"))
    if files:
        files[0].unlink()
    a = run_cmd([PY, str(ROOT / "scripts" / "analyze.py"),
                 "--run-dir", str(run_dir)])
    out = a.stderr + a.stdout
    check("3 analyze refuses incomplete full run",
          a.returncode != 0 and "missing" in out.lower(),
          f"rc={a.returncode}\n{out}")


def test_mismatched_query_id_arrays() -> None:
    rec = good_record(queries_per_rep=3)
    rec["top5_ids"] = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]  # length 2 not 3
    try:
        v.validate_record(rec)
        check("4a mismatched top5_ids detected", False,
              "validator did not raise")
    except ValueError as e:
        check("4a mismatched top5_ids detected",
              "top5_ids length" in str(e), str(e))

    rec2 = good_record(queries_per_rep=3)
    rec2["query_order"] = [0, 1]  # wrong length
    try:
        v.validate_record(rec2)
        check("4b mismatched query_order detected", False,
              "validator did not raise")
    except ValueError as e:
        check("4b mismatched query_order detected",
              "query_order" in str(e), str(e))


def test_throughput_from_sum_calls() -> None:
    rec = {
        "config": {"threads": 1, "batch_size": 32, "rep": 0},
        "call_durations_s": [0.01] * 16,
        "sum_call_durations_s": 0.16,
        "elapsed_around_loop_s": 0.20,
        "queries_processed": 512,
        "dataset": {"nb": 10, "d": 4, "k": 5, "queries_per_rep": 512},
    }
    m = per_run_metric(rec)
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
    check(
        "6c decide_clear_winner: overlap => False",
        decide_clear_winner(agg_overlap, agg_overlap[2],
                            lower_better=True) is False,
    )
    check(
        "6d decide_clear_winner: non-overlap => True",
        decide_clear_winner(agg_clear, agg_clear[2],
                            lower_better=False) is True,
    )


# ---------- new regression cases ----------

def test_forged_sum_tenfold_rejected() -> None:
    rec = good_record()
    true_sum = sum(rec["call_durations_s"])
    rec["sum_call_durations_s"] = true_sum / 10.0  # fake 10x throughput
    try:
        v.validate_record(rec)
        check("7 forged sum_call_durations_s (1/10x) rejected", False,
              "validator did not raise")
    except ValueError as e:
        check("7 forged sum_call_durations_s (1/10x) rejected",
              "sum_call_durations_s" in str(e) and "tolerance" in str(e),
              str(e))


def test_nonfinite_or_nonpositive_sum_rejected() -> None:
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        rec = good_record()
        rec["sum_call_durations_s"] = bad
        try:
            v.validate_record(rec)
            check(f"8 sum_call_durations_s={bad!r} rejected", False,
                  "validator did not raise")
            return
        except ValueError:
            pass
    check("8 nonfinite/nonpositive sum_call_durations_s rejected", True)


def test_elapsed_less_than_sum_rejected() -> None:
    rec = good_record()
    rec["elapsed_around_loop_s"] = sum(rec["call_durations_s"]) / 2.0
    try:
        v.validate_record(rec)
        check("9 elapsed_around_loop_s < sum rejected", False,
              "validator did not raise")
    except ValueError as e:
        check("9 elapsed_around_loop_s < sum rejected",
              "elapsed_around_loop_s" in str(e) and "inconsistent" in str(e),
              str(e))


def _write_fake_run(tmp: Path, mode: str, manifest_overrides: dict,
                    rec_overrides_list: list[dict] | None = None,
                    expected_entries: list[list[int]] | None = None,
                    extra_record_keys: list[tuple[int, int, int]] | None = None
                    ) -> Path:
    """Create a run_dir with a crafted manifest and crafted per_call records."""
    run_dir = tmp / f"fake_{mode}_{len(list(tmp.iterdir()))}"
    (run_dir / "per_call").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "run_dir": str(run_dir), "mode": mode,
        "started_at_utc": "2026-11-07T00:00:00Z",
        "seed": 1, "queries_per_rep": 4, "k": 5,
        "configs": [[1, 1]], "reps": 1,
        "expected_entries": expected_entries or [[1, 1, 0]],
        "dataset_sha256": {"xb_sha256": "a" * 64, "xq_sha256": "b" * 64},
        "git": {}, "python_version": "3.11.9",
        "packages": {"faiss-cpu": "1.8.0", "numpy": "1.26.0"},
    }
    manifest.update(manifest_overrides)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    if rec_overrides_list is None:
        rec_overrides_list = [{}]
    for ro in rec_overrides_list:
        rec = good_record(
            queries_per_rep=manifest["queries_per_rep"],
            k=manifest["k"],
            xb_sha=manifest["dataset_sha256"]["xb_sha256"],
            xq_sha=manifest["dataset_sha256"]["xq_sha256"],
        )
        # Apply overrides (nested dict merge for simple cases).
        for k, v_ in ro.items():
            rec[k] = v_
        c = rec["config"]
        p = (run_dir / "per_call" /
             f"b{c['batch_size']}_t{c['threads']}_rep{c['rep']}.json")
        p.write_text(json.dumps(rec))
    if extra_record_keys:
        for (t, b, r_) in extra_record_keys:
            rec = good_record(
                threads=t, batch_size=b, rep=r_,
                queries_per_rep=manifest["queries_per_rep"],
                k=manifest["k"],
                xb_sha=manifest["dataset_sha256"]["xb_sha256"],
                xq_sha=manifest["dataset_sha256"]["xq_sha256"],
            )
            p = (run_dir / "per_call" / f"b{b}_t{t}_rep{r_}.json")
            p.write_text(json.dumps(rec))
    return run_dir


def _write_mismatch_run(tmp: Path, name: str,
                        manifest_fields: dict,
                        record_fields: dict) -> Path:
    """Write a run dir where manifest and record intentionally disagree."""
    run_dir = tmp / name
    (run_dir / "per_call").mkdir(parents=True, exist_ok=True)
    # Record with EXPLICIT fields (independent of manifest).
    rec = good_record(
        queries_per_rep=record_fields.get("queries_per_rep", 4),
        k=record_fields.get("k", 5),
        xb_sha=record_fields.get("xb_sha", "a" * 64),
        xq_sha=record_fields.get("xq_sha", "b" * 64),
    )
    (run_dir / "per_call" / "b1_t1_rep0.json").write_text(json.dumps(rec))
    manifest = {
        "schema_version": 2, "run_dir": str(run_dir), "mode": "smoke",
        "started_at_utc": "2026-11-07T00:00:00Z", "seed": 1,
        "queries_per_rep": manifest_fields.get("queries_per_rep", 4),
        "k": manifest_fields.get("k", 5),
        "configs": [[1, 1]], "reps": 1,
        "expected_entries": [[1, 1, 0]],
        "dataset_sha256": {
            "xb_sha256": manifest_fields.get("xb_sha", "a" * 64),
            "xq_sha256": manifest_fields.get("xq_sha", "b" * 64),
        },
        "git": {}, "python_version": "3.11.9", "packages": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def test_manifest_queries_per_rep_mismatch(tmp: Path) -> None:
    # Record has queries_per_rep=4; manifest claims 256.
    run_dir = _write_mismatch_run(
        tmp, "qpr_mismatch",
        manifest_fields={"queries_per_rep": 256},
        record_fields={"queries_per_rep": 4},
    )
    a = run_cmd([PY, str(ROOT / "scripts" / "analyze.py"),
                 "--run-dir", str(run_dir)])
    out = a.stderr + a.stdout
    check(
        "10 manifest queries_per_rep mismatch rejected",
        a.returncode != 0 and "queries_per_rep" in out,
        f"rc={a.returncode}\n{out}",
    )


def test_manifest_k_mismatch(tmp: Path) -> None:
    # Record has k=5 (top5 width 5); manifest claims k=9.
    run_dir = _write_mismatch_run(
        tmp, "k_mismatch",
        manifest_fields={"k": 9},
        record_fields={"k": 5},
    )
    a = run_cmd([PY, str(ROOT / "scripts" / "analyze.py"),
                 "--run-dir", str(run_dir)])
    out = a.stderr + a.stdout
    check(
        "11 manifest k mismatch rejected",
        a.returncode != 0 and ("record.k=5" in out or "manifest.k=9" in out),
        f"rc={a.returncode}\n{out}",
    )


def test_manifest_dataset_hash_mismatch(tmp: Path) -> None:
    # Record carries 'a'*64 / 'b'*64; manifest claims 'c'*64 / 'd'*64.
    run_dir = _write_mismatch_run(
        tmp, "hash_mismatch",
        manifest_fields={"xb_sha": "c" * 64, "xq_sha": "d" * 64},
        record_fields={"xb_sha": "a" * 64, "xq_sha": "b" * 64},
    )
    a = run_cmd([PY, str(ROOT / "scripts" / "analyze.py"),
                 "--run-dir", str(run_dir)])
    out = a.stderr + a.stdout
    check(
        "12 manifest dataset hash mismatch rejected",
        a.returncode != 0 and "dataset identity mismatch" in out,
        f"rc={a.returncode}\n{out}",
    )


def test_empty_smoke_rejected(tmp: Path) -> None:
    # Manifest for smoke, but NO per_call records.
    run_dir = tmp / "empty_smoke"
    (run_dir / "per_call").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2, "run_dir": str(run_dir), "mode": "smoke",
        "started_at_utc": "2026-11-07T00:00:00Z", "seed": 1,
        "queries_per_rep": 4, "k": 5, "configs": [[1, 1]], "reps": 1,
        "expected_entries": [[1, 1, 0]],
        "dataset_sha256": {"xb_sha256": "a" * 64, "xq_sha256": "b" * 64},
        "git": {}, "python_version": "3.11.9", "packages": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    a = run_cmd([PY, str(ROOT / "scripts" / "analyze.py"),
                 "--run-dir", str(run_dir)])
    out = a.stderr + a.stdout
    check(
        "13 empty smoke run rejected",
        a.returncode != 0 and "missing" in out.lower(),
        f"rc={a.returncode}\n{out}",
    )


def test_duplicate_smoke_record_rejected(tmp: Path) -> None:
    # Two per_call files for the same (1,1,0) key.
    run_dir = tmp / "dup_smoke"
    (run_dir / "per_call").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2, "run_dir": str(run_dir), "mode": "smoke",
        "started_at_utc": "2026-11-07T00:00:00Z", "seed": 1,
        "queries_per_rep": 4, "k": 5, "configs": [[1, 1]], "reps": 1,
        "expected_entries": [[1, 1, 0]],
        "dataset_sha256": {"xb_sha256": "a" * 64, "xq_sha256": "b" * 64},
        "git": {}, "python_version": "3.11.9", "packages": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    rec = good_record(queries_per_rep=4, xb_sha="a" * 64, xq_sha="b" * 64)
    (run_dir / "per_call" / "b1_t1_rep0.json").write_text(json.dumps(rec))
    (run_dir / "per_call" / "b1_t1_rep0_copy.json").write_text(json.dumps(rec))
    a = run_cmd([PY, str(ROOT / "scripts" / "analyze.py"),
                 "--run-dir", str(run_dir)])
    out = a.stderr + a.stdout
    check(
        "14 duplicate smoke record rejected",
        a.returncode != 0 and "duplicates" in out.lower(),
        f"rc={a.returncode}\n{out}",
    )


def test_extra_smoke_record_rejected(tmp: Path) -> None:
    # One expected entry (1,1,0) plus one unexpected (2,1,0).
    run_dir = _write_fake_run(
        tmp, "smoke", {}, rec_overrides_list=[{}],
        extra_record_keys=[(2, 1, 0)],
    )
    a = run_cmd([PY, str(ROOT / "scripts" / "analyze.py"),
                 "--run-dir", str(run_dir)])
    out = a.stderr + a.stdout
    check(
        "15 extra smoke record rejected",
        a.returncode != 0 and "extra" in out.lower(),
        f"rc={a.returncode}\n{out}",
    )


def test_overwrite_refused_without_flag(tmp: Path) -> None:
    run_dir = tmp / "ow"
    r1 = run_cmd([PY, str(ROOT / "scripts" / "run_bench.py"),
                  "--mode", "smoke", "--run-dir", str(run_dir)])
    check("16a prime run succeeds", r1.returncode == 0, r1.stderr)
    r2 = run_cmd([PY, str(ROOT / "scripts" / "run_bench.py"),
                  "--mode", "smoke", "--run-dir", str(run_dir)])
    check("16b second run refused without --allow-overwrite",
          r2.returncode != 0, f"rc={r2.returncode}\n{r2.stderr}\n{r2.stdout}")


def test_overwrite_clears_prior_records(tmp: Path) -> None:
    run_dir = tmp / "ow2"
    r1 = run_cmd([PY, str(ROOT / "scripts" / "run_bench.py"),
                  "--mode", "smoke", "--run-dir", str(run_dir)])
    check("17a prime run succeeds", r1.returncode == 0, r1.stderr)
    # Plant a stale extra record that must be cleared.
    stale = run_dir / "per_call" / "b32_t4_rep9.json"
    stale.write_text(json.dumps(good_record(threads=4, batch_size=32, rep=9,
                                             queries_per_rep=4)))
    r2 = run_cmd([PY, str(ROOT / "scripts" / "run_bench.py"),
                  "--mode", "smoke", "--run-dir", str(run_dir),
                  "--allow-overwrite"])
    check("17b overwrite rerun succeeds", r2.returncode == 0,
          f"rc={r2.returncode}\n{r2.stderr}")
    check("17c stale record removed by overwrite",
          not stale.exists(), f"{stale} still present")
    # And a fresh analyse should succeed (no extras, no duplicates).
    a = run_cmd([PY, str(ROOT / "scripts" / "analyze.py"),
                 "--run-dir", str(run_dir)])
    check("17d post-overwrite analysis succeeds",
          a.returncode == 0, f"rc={a.returncode}\n{a.stderr}")


def test_interrupted_run_writes_failed_status(tmp: Path) -> None:
    # The orchestrator permits any queries_per_rep that is divisible by the
    # planned batch sizes, but the worker fails if queries_per_rep exceeds
    # nq_all (512 in our dataset). This produces a worker failure after the
    # manifest has been written, which is exactly the "mid-run failure"
    # path we want to exercise.
    run_dir = tmp / "worker_fails"
    r = run_cmd([PY, str(ROOT / "scripts" / "run_bench.py"),
                 "--mode", "smoke", "--run-dir", str(run_dir),
                 "--queries-per-rep", "1024"])
    status_path = run_dir / "status.json"
    status = (json.loads(status_path.read_text())
              if status_path.exists() else None)
    check(
        "18 failed worker causes status=failed",
        r.returncode != 0 and status is not None
        and status["state"] == "failed",
        f"rc={r.returncode} status={status}",
    )


def test_ntimed_calls_consistency() -> None:
    rec = good_record()
    rec["n_timed_calls"] = rec["n_timed_calls"] + 1
    try:
        v.validate_record(rec)
        check("19 n_timed_calls mismatch rejected", False,
              "validator did not raise")
    except ValueError as e:
        check(
            "19 n_timed_calls mismatch rejected",
            "n_timed_calls" in str(e), str(e),
        )


def main() -> int:
    ensure_data()
    with tempfile.TemporaryDirectory(prefix="faissbench_test_") as tmpd:
        tmp = Path(tmpd)
        test_smoke_analysis_success(tmp)
        test_smoke_full_separation(tmp)
        test_incomplete_full_detected(tmp)
        test_manifest_queries_per_rep_mismatch(tmp)
        test_manifest_k_mismatch(tmp)
        test_manifest_dataset_hash_mismatch(tmp)
        test_empty_smoke_rejected(tmp)
        test_duplicate_smoke_record_rejected(tmp)
        test_extra_smoke_record_rejected(tmp)
        test_overwrite_refused_without_flag(tmp)
        test_overwrite_clears_prior_records(tmp)
        test_interrupted_run_writes_failed_status(tmp)
    test_mismatched_query_id_arrays()
    test_throughput_from_sum_calls()
    test_decision_rule_consistency()
    test_forged_sum_tenfold_rejected()
    test_nonfinite_or_nonpositive_sum_rejected()
    test_elapsed_less_than_sum_rejected()
    test_ntimed_calls_consistency()
    print()
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(" -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
