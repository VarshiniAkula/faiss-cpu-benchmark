"""Shared validators. Pure functions so tests can import them.

Design rule: every validator either returns silently on success or raises
ValueError with an actionable message. No logging to stderr from here.
"""
from __future__ import annotations

import math
from typing import Iterable


VALID_K_RANGE = (1, 1024)


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
                 allowed_threads: Iterable[int] = (1, 2, 4),
                 allowed_batch: Iterable[int] = (1, 32)) -> None:
    validate_positive_int("threads", threads)
    if threads not in set(allowed_threads):
        raise ValueError(
            f"threads={threads} not in allowed set {sorted(allowed_threads)}"
        )
    validate_positive_int("batch_size", batch_size)
    if batch_size not in set(allowed_batch):
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


def validate_record(record: dict, expect_dataset: dict | None = None) -> None:
    """Check a per-call worker record for shape and finite positive times."""
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
    if expect_dataset is not None:
        for key in ("nb", "d", "k"):
            if ds[key] != expect_dataset[key]:
                raise ValueError(
                    f"dataset mismatch: record.{key}={ds[key]} but "
                    f"manifest.{key}={expect_dataset[key]}"
                )

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
        )
    for i, x in enumerate(durs):
        if not isinstance(x, (int, float)):
            raise ValueError(f"call_durations_s[{i}] not numeric: {x!r}")
        if not math.isfinite(x) or x <= 0:
            raise ValueError(
                f"call_durations_s[{i}]={x} must be finite and positive"
            )

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

    qo = record.get("query_order", None)
    if qo is None or len(qo) != ds["queries_per_rep"]:
        raise ValueError(
            "query_order missing or wrong length"
        )

    if "queries_processed" not in record:
        raise ValueError("record missing 'queries_processed'")
    if record["queries_processed"] != ds["queries_per_rep"]:
        raise ValueError(
            f"queries_processed={record['queries_processed']} != "
            f"queries_per_rep={ds['queries_per_rep']}"
        )

    if "sum_call_durations_s" not in record:
        raise ValueError("record missing 'sum_call_durations_s'")
    if "elapsed_around_loop_s" not in record:
        raise ValueError("record missing 'elapsed_around_loop_s'")


def validate_manifest_vs_records(manifest: dict,
                                 records_by_key: dict) -> dict:
    """Return {'missing': [...], 'duplicates': [...]} given expected combos.

    Expected combos come from manifest['expected_entries']. records_by_key
    maps (threads, batch_size, rep) -> list of file paths (duplicates
    produce list length > 1).
    """
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
