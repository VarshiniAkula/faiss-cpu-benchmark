"""Shared validators. Pure functions so tests can import them.

Design rule: every validator either returns silently on success or raises
ValueError with an actionable message. No logging to stderr from here.
"""
from __future__ import annotations

import math
from typing import Iterable


VALID_K_RANGE = (1, 1024)

# Tolerance used when checking stored timing totals against a recomputed
# sum of float durations. Floating-point summation order and C-level
# vs Python-level accumulation can produce tiny differences; anything
# meaningfully larger is treated as tampering or corruption.
SUM_REL_TOL = 1e-9
SUM_ABS_TOL = 1e-12

# Allow a very small negative slack when comparing elapsed-around-loop
# to the search-time sum, to tolerate clock quantisation on short runs.
ELAPSED_VS_SUM_NEG_SLACK_S = 1e-6


def validate_positive_int(name: str, value: int, lo: int = 1,
                          hi: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name}: must be int, got {type(value).__name__}")
    if value < lo:
        raise ValueError(f"{name}={value} must be >= {lo}")
    if hi is not None and value > hi:
        raise ValueError(f"{name}={value} must be <= {hi}")


def validate_cli(threads: int, batch_size: int, reps: int, k: int,
                 queries_per_rep: int,
                 allowed_threads: Iterable[int] | None = None,
                 allowed_batch: Iterable[int] | None = None) -> None:
    """When allowed_* is None, any positive int is accepted. When a set is
    given, membership is enforced. This lets the orchestrator define its
    own experiment-specific thread/batch sets without baking them into
    the worker-level validator."""
    validate_positive_int("threads", threads)
    if allowed_threads is not None and threads not in set(allowed_threads):
        raise ValueError(
            f"threads={threads} not in allowed set {sorted(allowed_threads)}"
        )
    validate_positive_int("batch_size", batch_size)
    if allowed_batch is not None and batch_size not in set(allowed_batch):
        raise ValueError(
            f"batch_size={batch_size} not in allowed set {sorted(allowed_batch)}"
        )
    validate_positive_int("reps", reps)
    validate_positive_int("k", k, VALID_K_RANGE[0], VALID_K_RANGE[1])
    validate_positive_int("queries_per_rep", queries_per_rep)
    if queries_per_rep % batch_size != 0:
        raise ValueError(
            f"queries_per_rep={queries_per_rep} must be divisible by "
            f"batch_size={batch_size}"
        )


def _finite_positive(name: str, value) -> None:
    if value is None:
        raise ValueError(f"{name} must be present")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name}={value} must be finite")
    if value <= 0:
        raise ValueError(f"{name}={value} must be > 0")


def validate_record(record: dict, expect_dataset: dict | None = None) -> None:
    """Check a per-call worker record for shape, timing consistency, and
    (optionally) agreement with a manifest-derived expectation.

    expect_dataset, when given, may contain any of:
      nb, d, k, queries_per_rep       -- integer identity
      xb_sha256, xq_sha256            -- dataset identity hashes
    """
    if "config" not in record or "dataset" not in record:
        raise ValueError("record missing 'config' or 'dataset'")
    cfg = record["config"]
    ds = record["dataset"]
    for key in ("threads", "batch_size", "rep"):
        if key not in cfg:
            raise ValueError(f"record.config missing '{key}'")
    for key in ("nb", "d", "k", "queries_per_rep"):
        if key not in ds:
            raise ValueError(f"record.dataset missing '{key}'")

    # Integer identity checks against manifest.
    if expect_dataset is not None:
        for key in ("nb", "d", "k", "queries_per_rep"):
            if key in expect_dataset and ds[key] != expect_dataset[key]:
                raise ValueError(
                    f"dataset mismatch: record.{key}={ds[key]} but "
                    f"manifest.{key}={expect_dataset[key]}"
                )
        for key in ("xb_sha256", "xq_sha256"):
            if key in expect_dataset and expect_dataset[key] is not None:
                if ds.get(key) != expect_dataset[key]:
                    raise ValueError(
                        f"dataset identity mismatch: record.{key}="
                        f"{ds.get(key)} but manifest.{key}="
                        f"{expect_dataset[key]}"
                    )

    # Durations.
    durs = record.get("call_durations_s", None)
    if durs is None:
        raise ValueError("record missing 'call_durations_s'")
    if not isinstance(durs, list):
        raise ValueError("call_durations_s must be a list")

    expected_calls = (
        ds["queries_per_rep"] if cfg["batch_size"] == 1
        else ds["queries_per_rep"] // cfg["batch_size"]
    )
    if len(durs) != expected_calls:
        raise ValueError(
            f"call_durations_s length {len(durs)} != expected {expected_calls}"
            f" (queries_per_rep={ds['queries_per_rep']}, "
            f"batch_size={cfg['batch_size']})"
        )
    for i, x in enumerate(durs):
        if not isinstance(x, (int, float)):
            raise ValueError(f"call_durations_s[{i}] not numeric: {x!r}")
        if not math.isfinite(x) or x <= 0:
            raise ValueError(
                f"call_durations_s[{i}]={x} must be finite and positive"
            )

    # n_timed_calls, if present, must match len(durs) AND expected_calls.
    n_timed = record.get("n_timed_calls")
    if n_timed is not None:
        if n_timed != len(durs):
            raise ValueError(
                f"n_timed_calls={n_timed} != len(call_durations_s)={len(durs)}"
            )
        if n_timed != expected_calls:
            raise ValueError(
                f"n_timed_calls={n_timed} != expected_calls={expected_calls}"
            )

    # Timing totals: must be finite/positive and numerically consistent
    # with sum(call_durations_s).
    sum_declared = record.get("sum_call_durations_s")
    _finite_positive("sum_call_durations_s", sum_declared)
    sum_actual = float(sum(durs))
    tol = max(SUM_REL_TOL * sum_actual, SUM_ABS_TOL)
    if abs(float(sum_declared) - sum_actual) > tol:
        raise ValueError(
            f"sum_call_durations_s={sum_declared} does not match "
            f"sum(call_durations_s)={sum_actual} within tolerance {tol:.3e}"
        )

    elapsed = record.get("elapsed_around_loop_s")
    _finite_positive("elapsed_around_loop_s", elapsed)
    # elapsed must be >= sum_actual up to a tiny clock slack.
    if float(elapsed) < sum_actual - ELAPSED_VS_SUM_NEG_SLACK_S:
        raise ValueError(
            f"elapsed_around_loop_s={elapsed} < sum(call_durations_s)="
            f"{sum_actual}; timing totals are inconsistent"
        )

    # queries_processed must equal queries_per_rep.
    if "queries_processed" not in record:
        raise ValueError("record missing 'queries_processed'")
    if record["queries_processed"] != ds["queries_per_rep"]:
        raise ValueError(
            f"queries_processed={record['queries_processed']} != "
            f"queries_per_rep={ds['queries_per_rep']}"
        )

    # top5_ids: shape queries_per_rep x k.
    top5 = record.get("top5_ids", None)
    if top5 is None:
        raise ValueError("record missing 'top5_ids'")
    if len(top5) != ds["queries_per_rep"]:
        raise ValueError(
            f"top5_ids length {len(top5)} != queries_per_rep="
            f"{ds['queries_per_rep']}"
        )
    for i, row in enumerate(top5):
        if len(row) != ds["k"]:
            raise ValueError(
                f"top5_ids[{i}] length {len(row)} != k={ds['k']}"
            )

    # query_order: length queries_per_rep.
    qo = record.get("query_order", None)
    if qo is None or len(qo) != ds["queries_per_rep"]:
        raise ValueError(
            f"query_order missing or wrong length (got "
            f"{None if qo is None else len(qo)}, "
            f"expected {ds['queries_per_rep']})"
        )


def validate_manifest_vs_records(manifest: dict,
                                 records_by_key: dict) -> dict:
    """Return {'missing': [...], 'extra': [...], 'duplicates': [...]} given
    expected combos. records_by_key maps (threads, batch_size, rep) -> list
    of file paths (duplicates have list length > 1)."""
    expected = {tuple(e) for e in manifest["expected_entries"]}
    present = set(records_by_key.keys())
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    duplicates = sorted(
        k for k, files in records_by_key.items() if len(files) > 1
    )
    return {
        "missing": [list(x) for x in missing],
        "extra": [list(x) for x in extra],
        "duplicates": [list(x) for x in duplicates],
    }


def validate_planned_configs(manifest: dict,
                             allowed_threads: Iterable[int] | None = None,
                             allowed_batch: Iterable[int] | None = None
                             ) -> None:
    """Every planned config must be a positive (threads, batch) pair
    (and inside the explicit allowed sets, if provided). Every expected
    entry must reference a planned config."""
    planned = {(int(t), int(b)) for t, b in manifest.get("configs", [])}
    for t, b in planned:
        validate_positive_int("thread count", t)
        validate_positive_int("batch_size", b)
        if allowed_threads is not None and t not in set(allowed_threads):
            raise ValueError(
                f"manifest configs: thread count {t} not in "
                f"{sorted(allowed_threads)}"
            )
        if allowed_batch is not None and b not in set(allowed_batch):
            raise ValueError(
                f"manifest configs: batch_size {b} not in "
                f"{sorted(allowed_batch)}"
            )
    for entry in manifest.get("expected_entries", []):
        t, b, _ = entry
        if (int(t), int(b)) not in planned:
            raise ValueError(
                f"expected_entries contains {entry} not in planned configs"
            )


def validate_agreement_pair(ref: dict, cmp: dict) -> None:
    """Both must exist, same query order, same queries_per_rep."""
    if ref is None or cmp is None:
        raise ValueError("agreement: reference or comparison record is None")
    if ref["query_order"] != cmp["query_order"]:
        raise ValueError(
            "agreement: query_order differs between reference and comparison"
        )
    if len(ref["top5_ids"]) != len(cmp["top5_ids"]):
        raise ValueError(
            "agreement: top5_ids lengths differ between ref and cmp"
        )
