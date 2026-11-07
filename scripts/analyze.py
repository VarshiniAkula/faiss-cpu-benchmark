"""Analyse exactly one run directory. Smoke-safe. Validates first.

Writes INSIDE the run directory:
  summary.csv              per-run metrics
  summary_agg.csv          median + [min, max] per config
  agreement.json           per-run top-5 identity vs per-rep reference
  recommendations.json     machine-readable best-per-use-case
  figure.png               two-panel chart (if enough data)
  analysis_notes.json      what comparisons were/weren't performed
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import validate as v  # noqa: E402


# Decision rule for "no clear winner", documented in README:
#   Call the recommendation CLEAR only if the best config's median is
#   OUTSIDE the [min, max] observed range of every other config.
#   This is a descriptive overlap check, NOT a formal significance test.
def decide_clear_winner(items: list[dict], best: dict,
                        lower_better: bool) -> bool:
    for other in items:
        if other is best:
            continue
        if lower_better:
            if best["median"] >= other["min"]:
                return False
        else:
            if best["median"] <= other["max"]:
                return False
    return True


def load_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    per_call = run_dir / "per_call"
    records_by_key: dict = {}
    duplicate_paths: dict = {}
    for p in sorted(per_call.glob("*.json")):
        rec = json.loads(p.read_text())
        c = rec["config"]
        key = (int(c["threads"]), int(c["batch_size"]), int(c["rep"]))
        if key in records_by_key:
            duplicate_paths.setdefault(key, [records_by_key[key]["_path"]]).append(str(p))
            # Keep only the first; flag.
        else:
            rec["_path"] = str(p)
            records_by_key[key] = rec

    # Validate each record shape AND against manifest dataset.
    expect = {"nb": None, "d": None, "k": int(manifest["k"]),
              "queries_per_rep": int(manifest["queries_per_rep"])}
    for key, rec in records_by_key.items():
        ds = rec["dataset"]
        if expect["nb"] is None:
            expect["nb"] = ds["nb"]; expect["d"] = ds["d"]
        v.validate_record(rec, expect_dataset=expect)
        # Also enforce dataset hash agreement with manifest:
        for h_key in ("xb_sha256", "xq_sha256"):
            if ds.get(h_key) != manifest["dataset_sha256"].get(h_key):
                raise ValueError(
                    f"{rec['_path']}: dataset {h_key} differs from manifest"
                )
    return {
        "manifest": manifest,
        "records_by_key": records_by_key,
        "duplicate_paths": duplicate_paths,
    }


def per_run_metric(rec: dict) -> dict:
    cfg = rec["config"]
    b = cfg["batch_size"]
    durs = rec["call_durations_s"]
    sum_calls = float(rec["sum_call_durations_s"])
    around = float(rec["elapsed_around_loop_s"])
    nq = int(rec["queries_processed"])
    return {
        "threads": cfg["threads"], "batch_size": b, "rep": cfg["rep"],
        "n_calls": len(durs), "queries": nq,
        "sum_call_durations_s": sum_calls,
        "elapsed_around_loop_s": around,
        "mean_call_ms": float(np.mean(durs) * 1000),
        "median_call_ms": float(np.median(durs) * 1000),
        "p95_call_ms": float(np.percentile(durs, 95) * 1000),
        "throughput_qps_search_only":
            float(nq / sum_calls) if sum_calls > 0 else float("nan"),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["threads"], r["batch_size"]), []).append(r)
    out = []
    for (t, b), rs in groups.items():
        if b == 1:
            vals = [r["p95_call_ms"] for r in rs]
            metric_name = "p95_call_ms"
        else:
            vals = [r["throughput_qps_search_only"] for r in rs]
            metric_name = "throughput_qps_search_only"
        out.append({
            "threads": t, "batch_size": b, "n_reps": len(rs),
            "metric": metric_name,
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "all_reps": vals,
        })
    return sorted(out, key=lambda r: (r["batch_size"], r["threads"]))


def agreement_report(records_by_key: dict) -> dict:
    """Compare each (t, b, rep) to (1, 1, rep). Fail hard on missing refs
    when a non-reference record is present without its per-rep reference."""
    reps = sorted({rep for (_, _, rep) in records_by_key.keys()})
    per_run: list[dict] = []
    per_config: dict = {}
    missing_refs: list = []
    for key, rec in sorted(records_by_key.items()):
        t, b, rep = key
        ref = records_by_key.get((1, 1, rep))
        if ref is None:
            missing_refs.append([t, b, rep])
            continue
        v.validate_agreement_pair(ref, rec)
        ref_ids = [set(r) for r in ref["top5_ids"]]
        cmp_ids = [set(r) for r in rec["top5_ids"]]
        n = len(ref_ids)
        matches = sum(1 for i in range(n) if ref_ids[i] == cmp_ids[i])
        frac = matches / n
        per_run.append({
            "threads": t, "batch_size": b, "rep": rep,
            "n_queries_compared": n,
            "n_identical_top5_sets": matches,
            "fraction_identical": frac,
        })
        per_config.setdefault((t, b), []).append(frac)
    summary = [
        {
            "threads": t, "batch_size": b, "n_reps": len(vals),
            "min_fraction_identical": float(np.min(vals)),
            "median_fraction_identical": float(np.median(vals)),
            "max_fraction_identical": float(np.max(vals)),
        }
        for (t, b), vals in sorted(per_config.items())
    ]
    return {
        "reference_rule":
            "per-rep: (threads=1, batch=1, same rep). "
            "Missing reference for any non-reference record is an error.",
        "per_run": per_run,
        "per_config_summary": summary,
        "missing_references": missing_refs,
        "reps_covered": reps,
    }


def write_summary(rows: list[dict], path: Path) -> None:
    fields = ["threads", "batch_size", "rep", "n_calls", "queries",
              "sum_call_durations_s", "elapsed_around_loop_s",
              "mean_call_ms", "median_call_ms", "p95_call_ms",
              "throughput_qps_search_only"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["batch_size"], r["threads"],
                                             r["rep"])):
            w.writerow({k: r[k] for k in fields})


def write_summary_agg(agg: list[dict], path: Path) -> None:
    fields = ["threads", "batch_size", "metric", "n_reps", "median",
              "min", "max"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in agg:
            w.writerow({k: r[k] for k in fields})


def build_figure(agg: list[dict], path: Path) -> bool:
    b1 = sorted([r for r in agg if r["batch_size"] == 1],
                key=lambda r: r["threads"])
    b32 = sorted([r for r in agg if r["batch_size"] == 32],
                 key=lambda r: r["threads"])
    if not b1 and not b32:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    def panel(ax, data, title, ylab, color):
        if not data:
            ax.text(0.5, 0.5, "no data for this panel\n(smoke run?)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title); ax.set_xlabel("requested threads")
            ax.set_ylabel(ylab); return
        t = [r["threads"] for r in data]
        m = [r["median"] for r in data]
        lo = [r["median"] - r["min"] for r in data]
        hi = [r["max"] - r["median"] for r in data]
        ax.errorbar(t, m, yerr=[lo, hi], fmt="o-", capsize=4, color=color)
        ax.set_title(title); ax.set_xlabel("requested threads")
        ax.set_ylabel(ylab); ax.set_xticks(t); ax.grid(True, alpha=0.3)

    panel(axes[0], b1, "Interactive (batch=1): p95 latency",
          "p95 call latency (ms)", "tab:blue")
    panel(axes[1], b32, "Batch-32: throughput (search-only)",
          "throughput (queries/sec)", "tab:orange")
    fig.suptitle(
        "FAISS CPU retrieval benchmark (synthetic 50k x 384 f32, k=5)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def recommendations(agg: list[dict]) -> dict:
    """Produce a recommendation per use case or 'insufficient data'.

    Decision rule is the one implemented in decide_clear_winner().
    """
    out: dict = {
        "decision_rule":
            "Recommendation is CLEAR only if the best config's median is "
            "outside the [min, max] observed range of every other config "
            "measured in this run. This is a descriptive overlap check, "
            "not a formal significance test.",
    }
    b1 = sorted([r for r in agg if r["batch_size"] == 1],
                key=lambda r: r["threads"])
    b32 = sorted([r for r in agg if r["batch_size"] == 32],
                 key=lambda r: r["threads"])

    def summarize(data, lower_better, metric_label):
        if len(data) == 0:
            return {"status": "insufficient_data", "reason":
                    "no records for this batch size in this run"}
        best = (min(data, key=lambda r: r["median"]) if lower_better
                else max(data, key=lambda r: r["median"]))
        baseline = next((r for r in data if r["threads"] == 1), None)
        if baseline is None:
            base_info = {"status": "no_single_thread_baseline"}
        else:
            if lower_better:
                pct = ((best["median"] - baseline["median"])
                       / baseline["median"] * 100.0)
            else:
                pct = ((best["median"] - baseline["median"])
                       / baseline["median"] * 100.0)
            base_info = {
                "baseline_1thread_median": baseline["median"],
                "baseline_1thread_min_max": [baseline["min"], baseline["max"]],
                "pct_vs_1thread_median": pct,
            }
        if len(data) == 1:
            return {
                "status": "single_configuration_only",
                "only_measured": {"threads": best["threads"],
                                  "metric": metric_label,
                                  "median": best["median"],
                                  "min_max": [best["min"], best["max"]]},
                **base_info,
            }
        clear = decide_clear_winner(data, best, lower_better)
        return {
            "status": "clear_winner" if clear else "no_clear_winner",
            "lower_better": lower_better,
            "best_observed": {
                "threads": best["threads"],
                "median": best["median"],
                "min_max": [best["min"], best["max"]],
            },
            "default_if_no_winner": {"threads": 1},
            "all_configs": [
                {"threads": r["threads"], "median": r["median"],
                 "min_max": [r["min"], r["max"]], "n_reps": r["n_reps"]}
                for r in data
            ],
            **base_info,
        }

    out["interactive_batch1_p95_ms"] = summarize(b1, True, "p95_call_ms")
    out["throughput_batch32_qps"] = summarize(
        b32, False, "throughput_qps_search_only"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True,
                   help="run directory produced by run_bench.py")
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    if not (run_dir / "manifest.json").exists():
        raise SystemExit(
            f"{run_dir}/manifest.json missing. --run-dir must point to a run "
            f"directory produced by scripts/run_bench.py."
        )

    data = load_run(run_dir)
    manifest = data["manifest"]
    records = data["records_by_key"]
    # Manifest vs records check: smoke must have exactly its expected set;
    # full must have no missing/duplicates/extra.
    present = {k: [v["_path"]] for k, v in records.items()}
    for k, paths in data["duplicate_paths"].items():
        present[k] = paths
    report = v.validate_manifest_vs_records(manifest, present)

    if manifest["mode"] == "full":
        if report["missing"] or report["duplicates"] or report["extra"]:
            raise SystemExit(
                "Full run is incomplete or inconsistent; refusing to publish.\n"
                f" missing   : {report['missing']}\n"
                f" duplicates: {report['duplicates']}\n"
                f" extra     : {report['extra']}"
            )

    rows = [per_run_metric(r) for r in records.values()]
    agg = aggregate(rows)
    write_summary(rows, run_dir / "summary.csv")
    write_summary_agg(agg, run_dir / "summary_agg.csv")
    had_figure = build_figure(agg, run_dir / "figure.png")
    rep = agreement_report(records)
    if rep["missing_references"]:
        raise SystemExit(
            "Agreement check cannot complete; missing per-rep reference "
            f"(1-thread, batch=1) for: {rep['missing_references']}"
        )
    (run_dir / "agreement.json").write_text(json.dumps(rep, indent=2))
    recs = recommendations(agg)
    (run_dir / "recommendations.json").write_text(json.dumps(recs, indent=2))

    notes = {
        "mode": manifest["mode"],
        "configs_present": sorted({(k[0], k[1]) for k in records.keys()}),
        "reps_covered": sorted({k[2] for k in records.keys()}),
        "interactive_analyzed":
            any(r["batch_size"] == 1 for r in agg),
        "throughput_analyzed":
            any(r["batch_size"] == 32 for r in agg),
        "figure_rendered": had_figure,
        "comparisons_not_performed": (
            [] if manifest["mode"] == "full"
            else ["cross-thread p95 (smoke)", "batch-32 throughput (smoke)"]
        ),
    }
    (run_dir / "analysis_notes.json").write_text(json.dumps(notes, indent=2))

    print(f"\n=== Aggregated results from {run_dir} ===")
    print(f"{'batch':>5} {'threads':>8} {'metric':>30} "
          f"{'median':>12} {'min':>12} {'max':>12}")
    for r in agg:
        print(f"{r['batch_size']:>5} {r['threads']:>8} {r['metric']:>30} "
              f"{r['median']:>12.4f} {r['min']:>12.4f} {r['max']:>12.4f}")
    print("\nRecommendation:")
    print(json.dumps(recs, indent=2))
    print("\nNotes:")
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    main()
