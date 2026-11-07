"""Aggregate per-call results into a summary + figure + agreement report.

Writes:
  results/summary.csv       per-run metrics for every (config, rep)
  results/summary_agg.csv   median and [min,max] across reps per config
  results/agreement.json    fraction of queries with identical top-5 sets
                            vs reference (threads=1, batch=1, rep=0)
  results/figure.png        two-panel figure
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PER_CALL = RESULTS / "per_call"


def load_records() -> list[dict]:
    out = []
    for p in sorted(PER_CALL.glob("b*_t*_rep*.json")):
        out.append(json.loads(p.read_text()))
    return out


def p95(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(arr, 95))


def per_run_metric(rec: dict) -> dict:
    cfg = rec["config"]
    b = cfg["batch_size"]
    durs = rec["call_durations_s"]
    total = rec["total_timed_seconds"]
    nq = rec["queries_processed"]
    row = {
        "threads": cfg["threads"], "batch_size": b, "rep": cfg["rep"],
        "n_calls": len(durs), "queries": nq,
        "total_timed_s": total,
        "mean_call_ms": float(np.mean(durs) * 1000) if durs else float("nan"),
        "median_call_ms": (
            float(np.median(durs) * 1000) if durs else float("nan")
        ),
        "p95_call_ms": p95(durs) * 1000 if durs else float("nan"),
        "throughput_qps": float(nq / total) if total > 0 else float("nan"),
    }
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    key = lambda r: (r["threads"], r["batch_size"])
    rows_sorted = sorted(rows, key=key)
    groups: dict[tuple, list[dict]] = {}
    for r in rows_sorted:
        groups.setdefault(key(r), []).append(r)
    agg = []
    for (t, b), rs in groups.items():
        if b == 1:
            vals = [r["p95_call_ms"] for r in rs]
            metric_name = "p95_call_ms"
        else:
            vals = [r["throughput_qps"] for r in rs]
            metric_name = "throughput_qps"
        agg.append({
            "threads": t, "batch_size": b, "n_reps": len(rs),
            "metric": metric_name,
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "all_reps": vals,
        })
    return sorted(agg, key=lambda r: (r["batch_size"], r["threads"]))


def agreement_report(records: list[dict]) -> dict:
    """Compare each (config, rep) top-5 set against the SAME rep's
    1-thread, batch-size-1 run (same queries, same order)."""
    recs_by_key = {}
    for rec in records:
        c = rec["config"]
        recs_by_key[(c["threads"], c["batch_size"], c["rep"])] = rec

    report = {
        "reference_rule": "per-rep: (threads=1, batch=1, same rep)",
        "per_run": [],
    }
    overall: dict[tuple, list[float]] = {}
    for (t, b, rep), rec in sorted(recs_by_key.items()):
        ref = recs_by_key.get((1, 1, rep))
        if ref is None:
            continue
        if rec["query_order"] != ref["query_order"]:
            # Must be true by construction; log if not.
            report["per_run"].append({
                "threads": t, "batch_size": b, "rep": rep,
                "error": "query order mismatch vs reference",
            })
            continue
        ref_ids = [set(row) for row in ref["top5_ids"]]
        cmp_ids = [set(row) for row in rec["top5_ids"]]
        n = min(len(ref_ids), len(cmp_ids))
        matches = sum(1 for i in range(n) if ref_ids[i] == cmp_ids[i])
        frac = matches / n if n else float("nan")
        report["per_run"].append({
            "threads": t, "batch_size": b, "rep": rep,
            "n_queries_compared": n,
            "n_identical_top5_sets": matches,
            "fraction_identical": frac,
        })
        overall.setdefault((t, b), []).append(frac)

    report["per_config_summary"] = [
        {
            "threads": t, "batch_size": b, "n_reps": len(v),
            "min_fraction_identical": float(np.min(v)),
            "median_fraction_identical": float(np.median(v)),
            "max_fraction_identical": float(np.max(v)),
        }
        for (t, b), v in sorted(overall.items())
    ]
    return report


def write_summary(rows: list[dict], path: Path) -> None:
    fields = ["threads", "batch_size", "rep", "n_calls", "queries",
              "total_timed_s", "mean_call_ms", "median_call_ms",
              "p95_call_ms", "throughput_qps"]
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


def build_figure(agg: list[dict], path: Path) -> None:
    b1 = [r for r in agg if r["batch_size"] == 1]
    b32 = [r for r in agg if r["batch_size"] == 32]
    b1.sort(key=lambda r: r["threads"])
    b32.sort(key=lambda r: r["threads"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    t = [r["threads"] for r in b1]
    med = [r["median"] for r in b1]
    lo = [r["median"] - r["min"] for r in b1]
    hi = [r["max"] - r["median"] for r in b1]
    ax1.errorbar(t, med, yerr=[lo, hi], fmt="o-", capsize=4,
                 color="tab:blue")
    ax1.set_title("Interactive (batch=1): p95 latency")
    ax1.set_xlabel("requested threads"); ax1.set_ylabel("p95 call latency (ms)")
    ax1.set_xticks(t); ax1.grid(True, alpha=0.3)

    t2 = [r["threads"] for r in b32]
    med2 = [r["median"] for r in b32]
    lo2 = [r["median"] - r["min"] for r in b32]
    hi2 = [r["max"] - r["median"] for r in b32]
    ax2.errorbar(t2, med2, yerr=[lo2, hi2], fmt="o-", capsize=4,
                 color="tab:orange")
    ax2.set_title("Batch-32: throughput (32 queries waiting)")
    ax2.set_xlabel("requested threads"); ax2.set_ylabel("throughput (queries/sec)")
    ax2.set_xticks(t2); ax2.grid(True, alpha=0.3)
    fig.suptitle("FAISS CPU retrieval benchmark (synthetic 50k x 384 f32, k=5)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def print_table(agg: list[dict]) -> None:
    print("\n=== Aggregated results (median [min, max] across reps) ===")
    print(f"{'batch':>5} {'threads':>8} {'metric':>18} {'median':>12} "
          f"{'min':>12} {'max':>12}")
    for r in agg:
        print(f"{r['batch_size']:>5} {r['threads']:>8} {r['metric']:>18} "
              f"{r['median']:>12.4f} {r['min']:>12.4f} {r['max']:>12.4f}")


def recommendations(agg: list[dict]) -> dict:
    b1 = sorted([r for r in agg if r["batch_size"] == 1],
                key=lambda r: r["threads"])
    b32 = sorted([r for r in agg if r["batch_size"] == 32],
                 key=lambda r: r["threads"])
    base1 = next(r for r in b1 if r["threads"] == 1)
    base32 = next(r for r in b32 if r["threads"] == 1)
    best1 = min(b1, key=lambda r: r["median"])
    best32 = max(b32, key=lambda r: r["median"])
    return {
        "interactive_best": {
            "threads": best1["threads"],
            "p95_ms_median": best1["median"],
            "pct_vs_1thread": (
                (best1["median"] - base1["median"]) / base1["median"] * 100.0
            ),
            "baseline_1thread_p95_ms_median": base1["median"],
            "baseline_1thread_p95_ms_min_max":
                [base1["min"], base1["max"]],
            "best_min_max": [best1["min"], best1["max"]],
        },
        "throughput_best": {
            "threads": best32["threads"],
            "qps_median": best32["median"],
            "pct_vs_1thread": (
                (best32["median"] - base32["median"]) / base32["median"] * 100.0
            ),
            "baseline_1thread_qps_median": base32["median"],
            "baseline_1thread_qps_min_max":
                [base32["min"], base32["max"]],
            "best_min_max": [best32["min"], best32["max"]],
        },
    }


def main() -> None:
    records = load_records()
    print(f"loaded {len(records)} per-call records")
    rows = [per_run_metric(r) for r in records]
    agg = aggregate(rows)
    write_summary(rows, RESULTS / "summary.csv")
    write_summary_agg(agg, RESULTS / "summary_agg.csv")
    build_figure(agg, RESULTS / "figure.png")
    rep = agreement_report(records)
    (RESULTS / "agreement.json").write_text(json.dumps(rep, indent=2))
    recs = recommendations(agg)
    (RESULTS / "recommendations.json").write_text(json.dumps(recs, indent=2))
    print_table(agg)
    print("\nRecommendation:")
    print(json.dumps(recs, indent=2))


if __name__ == "__main__":
    main()
