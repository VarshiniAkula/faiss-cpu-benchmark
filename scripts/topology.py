"""Enumerate logical processors and classify P-core vs E-core on Windows.

Uses the documented Win32 API `GetLogicalProcessorInformationEx` with
`RelationProcessorCore`. Each returned structure describes one physical
core: its EfficiencyClass (0 is lowest; on Intel hybrid CPUs 0 means
E-core, >0 means P-core), SMT flag, and the GroupMask bitmask of its
logical processors.

This project assumes a single processor group (up to 64 logical CPUs),
which covers the i9-14900HX (32 logical). Multi-group machines are
explicitly out of scope.

Output format (also written to --out as JSON):
  {
    "schema_version": 1,
    "physical_cores": [
      {"core_index": 0, "efficiency_class": 1, "smt": true,
       "logical_cpu_ids": [0, 1]},
      ...
    ],
    "processor_groups_observed": 1,
    "p_cores_logical_ids": [...],
    "e_cores_logical_ids": [...],
    "all_logical_ids": [...],
    "notes": "...",
  }
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from ctypes import wintypes
from pathlib import Path

# Constants / structures from winnt.h / sysinfoapi.h
RelationProcessorCore = 0

ERROR_INSUFFICIENT_BUFFER = 122

LTP_PC_SMT = 0x1


class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [
        ("Mask", ctypes.c_size_t),
        ("Group", wintypes.WORD),
        ("Reserved", wintypes.WORD * 3),
    ]


class PROCESSOR_RELATIONSHIP(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.BYTE),
        ("EfficiencyClass", wintypes.BYTE),
        ("Reserved", wintypes.BYTE * 20),
        ("GroupCount", wintypes.WORD),
        # Trailing: GROUP_AFFINITY GroupMask[GroupCount]
        # Access via byte offset below.
    ]


class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
    _fields_ = [
        ("Relationship", wintypes.DWORD),
        ("Size", wintypes.DWORD),
        # Union follows; for RelationProcessorCore it is a
        # PROCESSOR_RELATIONSHIP.
    ]


def _enumerate_physical_cores() -> list[dict]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetLogicalProcessorInformationEx.argtypes = (
        wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL

    length = wintypes.DWORD(0)
    ok = kernel32.GetLogicalProcessorInformationEx(
        RelationProcessorCore, None, ctypes.byref(length)
    )
    if ok or ctypes.get_last_error() != ERROR_INSUFFICIENT_BUFFER:
        raise OSError(
            f"GetLogicalProcessorInformationEx (size query) unexpected: "
            f"ok={ok} err={ctypes.get_last_error()}"
        )

    buf = (ctypes.c_byte * length.value)()
    ok = kernel32.GetLogicalProcessorInformationEx(
        RelationProcessorCore, buf, ctypes.byref(length)
    )
    if not ok:
        raise OSError(
            f"GetLogicalProcessorInformationEx (fetch) failed: "
            f"err={ctypes.get_last_error()}"
        )

    cores: list[dict] = []
    offset = 0
    idx = 0
    while offset < length.value:
        header = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX.from_address(
            ctypes.addressof(buf) + offset
        )
        if header.Relationship != RelationProcessorCore:
            offset += header.Size
            continue
        # PROCESSOR_RELATIONSHIP starts right after the header.
        pr_addr = (ctypes.addressof(buf) + offset
                   + ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX))
        pr = PROCESSOR_RELATIONSHIP.from_address(pr_addr)
        # GroupMask array follows PROCESSOR_RELATIONSHIP.
        gm_addr = pr_addr + ctypes.sizeof(PROCESSOR_RELATIONSHIP)
        logical_ids: list[int] = []
        groups_seen: set[int] = set()
        for g in range(pr.GroupCount):
            ga = GROUP_AFFINITY.from_address(
                gm_addr + g * ctypes.sizeof(GROUP_AFFINITY)
            )
            groups_seen.add(int(ga.Group))
            m = int(ga.Mask)
            # Only single-group assumption; still enumerate bits.
            for i in range(64):
                if (m >> i) & 1:
                    logical_ids.append(i)
        cores.append({
            "core_index": idx,
            "efficiency_class": int(pr.EfficiencyClass),
            "smt": bool(pr.Flags & LTP_PC_SMT),
            "logical_cpu_ids": sorted(logical_ids),
            "groups_observed": sorted(groups_seen),
        })
        idx += 1
        offset += header.Size
    return cores


def enumerate_topology() -> dict:
    if sys.platform != "win32":
        return {
            "schema_version": 1,
            "error": "topology discovery implemented for Windows only",
            "physical_cores": [],
        }
    cores = _enumerate_physical_cores()
    groups_seen = set()
    for c in cores:
        groups_seen.update(c["groups_observed"])
    # EfficiencyClass: pick the maximum observed value. Cores with that
    # class are "high performance" (P-cores). Cores with class 0 are E.
    eff_values = sorted({c["efficiency_class"] for c in cores})
    p_class = max(eff_values) if eff_values else 0
    p_logical: list[int] = []
    e_logical: list[int] = []
    for c in cores:
        if c["efficiency_class"] == p_class and p_class > 0:
            p_logical.extend(c["logical_cpu_ids"])
        elif c["efficiency_class"] == 0:
            e_logical.extend(c["logical_cpu_ids"])
    # Sort for determinism
    all_logical = sorted({cid for c in cores for cid in c["logical_cpu_ids"]})
    notes = []
    if len(groups_seen) > 1:
        notes.append(
            "Multiple processor groups observed; this tool supports only a "
            "single group for affinity. Masks apply within group 0 only."
        )
    if len(eff_values) == 1:
        notes.append(
            "All cores share the same EfficiencyClass; CPU is either not "
            "hybrid or the kernel does not expose a split."
        )
    return {
        "schema_version": 1,
        "physical_cores": cores,
        "processor_groups_observed": sorted(groups_seen),
        "efficiency_class_values_seen": eff_values,
        "p_core_efficiency_class": p_class if p_class > 0 else None,
        "p_cores_logical_ids": sorted(p_logical),
        "e_cores_logical_ids": sorted(e_logical),
        "all_logical_ids": all_logical,
        "n_p_logical": len(p_logical),
        "n_e_logical": len(e_logical),
        "n_logical_total": len(all_logical),
        "notes": notes,
    }


def pick_one_logical_per_physical(topology: dict, pool: str,
                                  count: int) -> list[int]:
    """Choose `count` distinct physical cores from the requested pool
    ('p' or 'e') and return one logical CPU id per chosen core (the
    lowest-numbered SMT sibling)."""
    pool_cores = [
        c for c in topology["physical_cores"]
        if (c["efficiency_class"] ==
            (topology["p_core_efficiency_class"] if pool == "p" else 0))
    ]
    if len(pool_cores) < count:
        raise ValueError(
            f"only {len(pool_cores)} {pool.upper()}-cores available, need {count}"
        )
    # Deterministic: sort by core_index then take lowest logical id each.
    chosen = sorted(pool_cores, key=lambda c: c["core_index"])[:count]
    return sorted(min(c["logical_cpu_ids"]) for c in chosen)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None,
                   help="write topology JSON here (default: stdout)")
    args = p.parse_args()
    topo = enumerate_topology()
    s = json.dumps(topo, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(s)
        print(f"wrote {args.out}")
    else:
        print(s)


if __name__ == "__main__":
    main()
