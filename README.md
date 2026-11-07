# FAISS CPU Retrieval Benchmark (synthetic workload)

A small, reproducible FAISS CPU retrieval benchmark on a Lenovo Legion
laptop (Intel Core i9-14900HX, Windows 11 native). The workload is a
**synthetic** `IndexFlatL2` of 50,000 x 384-D float32 vectors with
seeded Gaussian queries (512 held-out queries, k=5) — described in
full under [Experiment protocol](#experiment-protocol). Headline
numbers below are search-only `time.perf_counter()` measurements from
per-process workers, 5 repetitions each, deterministically shuffled
order, manifest-validated.

## Benchmark questions

On this CPU, with that index and k:

1. **Interactive (batch=1)** — which of 1, 2, or 4 requested OpenMP
   threads gives the lowest p95 per-call search latency? (unchanged
   historical experiment)
2. **Throughput scaling (batch=32)** — which of 1, 2, 4, 8, or 16
   requested threads gives the highest throughput when 32 queries are
   already waiting? Does scaling continue, plateau, or regress?
3. **P-core vs E-core placement** — at 4 requested threads, how does
   restricting the process to 4 physical P-cores or 4 physical E-cores
   compare to the default scheduler?

## Headline results

![scaling chart: p95 latency and batch-32 throughput vs threads](runs/20261107T064707Z-full-8b401ae/figure.png)

### Thread scaling (batch=32, search-only throughput, qps)

From `runs/20261107T064707Z-full-8b401ae/` (commit `8b401ae`, source
clean, 40/40 records; agreement 1.000 across every rep):

| threads | median qps | min    | max    | speedup | parallel eff. | % vs 1t |
|---------|------------|--------|--------|---------|---------------|---------|
| 1       | 1600.4     | 1525.4 | 1625.4 | 1.00x   | 1.00          | —       |
| 2       | 2413.6     | 2255.5 | 2525.9 | 1.51x   | 0.75          | +50.8%  |
| 4       | 3138.0     | 3002.2 | 3349.6 | 1.96x   | 0.49          | +96.1%  |
| 8       | **3492.2** | 3172.9 | 3661.4 | 2.18x   | 0.27          | +118.2% |
| 16      | 2870.1     | 2318.1 | 2981.1 | 1.79x   | 0.11          | +79.3%  |

**Scaling judgment: `regression`.** Throughput rises from 1 to 8
threads, then drops at 16 threads. Scaling efficiency falls monotonically
from 1.00 to 0.11. The best observed configuration is **8 threads**.

### Interactive (batch=1): p95 per-call latency (ms)

| threads | median p95 | min    | max    | % vs 1-thread |
|---------|------------|--------|--------|---------------|
| 1       | 3.823      | 3.607  | 3.914  | —             |
| 2       | 3.960      | 3.743  | 4.126  | +3.6%         |
| 4       | 3.960      | 3.621  | 3.976  | +3.6%         |

**Decision: `no_clear_winner`.** The best median is t=1 (3.823 ms) but
t=1's own range `[3.607, 3.914]` overlaps both t=2 and t=4 medians.
With 5 reps the between-thread effect is within per-rep noise.

### P-core vs E-core placement (batch=32, 4 requested threads)

From `runs/20261107T071454Z-pe-9f2ab32/`, three isolated sub-runs with
identical workload but different process-level CPU affinity. Topology
(from `scripts/topology.py` via `GetLogicalProcessorInformationEx`):
8 physical P-cores exposed as logical CPUs 0..15 (SMT), 16 physical
E-cores as logical CPUs 16..31 (no SMT), `EfficiencyClass` 1 vs 0.

| placement             | median qps | min    | max    | per-rep qps         |
|-----------------------|------------|--------|--------|---------------------|
| default scheduler     | 3087.0     | 2686.4 | 3351.7 | 3087, 2686, 2968, 3199, 3352 |
| 4 physical P-cores    | **3290.7** | 397.3  | 3395.5 | 397, 3291, 3131, 3395, 3308 |
| 4 physical E-cores    | 1235.0     | 342.2  | 1276.5 | 1277, 1235, 1245, 342, 418 |

Headline ratio (medians): **P-cores / E-cores ≈ 2.66x**. The default
scheduler median lands between the two. Whether the OS actually placed
most default-scheduler threads on P-cores was **not measured** here;
VTune thread/core placement evidence (Stage 3) would be needed to
support such a claim.

Two outlier reps (one in P-cores, two in E-cores) are 3–8x slower than
the rest of their sub-run. Observed affinity still matches requested
(verified in each per-call record's `cpu_affinity.observed_mask_hex`);
cause of the drops is **not measured** (candidate hypotheses: cold OMP
thread-pool recreation when CPU set is restricted, kernel/driver work
landing on the affinity set, thermal settling — none verified without
hardware counters or VTune).

**Preliminary status.** The 2.66x figure is from 5-rep sub-runs executed
back-to-back per placement. Follow-up work (`scripts/pe_experiment.py`
rewritten for this stage) interleaves placements per repetition and
raises the rep count; see the new PE run and comparison for the current
figure that supersedes this preliminary value.

### Why 16 threads regressed — mechanism unresolved

**Observation (scaling):** throughput peaks at 8 threads and drops
~18% at 16 threads.

**Observation (P/E, preliminary):** on 4 physical E-cores the same
workload runs ~2.66x slower than on 4 physical P-cores. This is a
property of the per-core work rate, not a direct explanation of the
scaling curve.

**Important correction to earlier wording.** This machine's 8 P-cores
expose **16 logical processors via SMT**, so 16 OpenMP threads can
fit on P-cores alone (two threads per physical core, SMT siblings).
"16 threads necessarily spill onto E-cores" is **not** supported by
the available topology data and has been removed. Candidate
mechanisms for the 16-thread regression that remain live:

- SMT contention: two logical threads sharing one P-core's physical
  execution units can contend for L1/L2 cache, FP units, and memory
  controllers.
- E-core placement by the OS scheduler: under default affinity the
  kernel may still place some workers on E-cores even when SMT
  capacity exists.
- OMP thread-management overhead (barriers, work distribution) at
  higher thread counts.

**Which of these actually dominates is unresolved.** FAISS's internal
work-partitioning strategy for `IndexFlatL2` is not characterised by
this experiment and should not be assumed. Confirming the mechanism
requires VTune Microarchitecture Exploration + thread/core placement
evidence at 16 threads (and comparisons at 1, 4, 8 threads), which
is the Stage 2/3 work still blocked on VTune install. Commands are
in `docs/VTUNE_COMMANDS.md`.

## Historical results preserved

| commit    | run dir                                        | note                                                |
|-----------|------------------------------------------------|-----------------------------------------------------|
| 6604f72   | `results/`                                     | original run, legacy schema                         |
| 6604f72   | `runs/20261107T032705Z-full-6604f72/`          | re-analysed under schema 2 (same data)              |
| 2f23831   | `runs/20261107T042639Z-full-2f23831/`          | validated schema-2 full run, 3-thread set           |
| 8b401ae   | `runs/20261107T064707Z-full-8b401ae/`          | **new 5-thread scaling (1,2,4,8,16)**               |
| 9f2ab32   | `runs/20261107T071454Z-pe-9f2ab32/`            | **new P-core vs E-core experiment**                 |

The historical 3-thread runs are still accepted by the current
analyser without modification; new runs add two threads at the top
and the P/E experiment.

## Stage 2/3: Intel VTune collection — *blocked, awaiting install*

VTune Profiler is **not installed** on this Legion as of commit
`9f2ab32`. Everything needed to collect the three requested profiles
(Hotspots + Microarchitecture Exploration at 1, 4, and 8 threads) is
implemented and documented:

- **Workload:** `scripts/profile_workload.py` runs the same batched
  search in a tight loop for a configurable wall-clock duration
  (default 25 s). Data load, index build, and 10 warm-up calls are
  **outside** the measured region. The region is delimited with
  sentinel prints (`PROFILE_REGION_START` / `PROFILE_REGION_END`)
  and, if the ITT API is importable, with
  `itt.task_begin/task_end` markers.
- **Baseline-throughput verification:** `profile_workload.py` runs
  without any profiler produce throughput numbers that match the
  per-process benchmark (1 thread: 1603 qps vs 1600, 4 threads:
  3174 vs 3138, 8 threads: 3408 vs 3492; see
  `runs/20261107T070000Z-profile-8b401ae/profile_t*.json`). The
  workload is therefore stable and ready for profiler attachment.
- **Exact collection commands** for Hotspots, Microarchitecture
  Exploration, and Performance Snapshot are in
  `docs/VTUNE_COMMANDS.md`, including matched-work interpretation
  guidance (fixed-duration runs process different query counts; totals
  must be normalised per query before cross-run comparison).

To unblock Stages 2/3, install VTune (free), then run the commands in
`docs/VTUNE_COMMANDS.md`. The expected outputs for commit comparison
are `summary.csv`, `hotspots.csv`, `callstacks.csv`, and
`workload_summary.json` per thread count.

## Experiment protocol

- **Dataset.** `scripts/make_data.py` writes `data/xb.npy`
  (50,000 x 384 f32, seed 20261106) and `data/xq.npy` (512 x 384 f32,
  seed 20261107). Dataset identity is enforced per-run: the manifest
  records SHA-256 of both files and every worker re-hashes and refuses
  to run on drift.
- **Index.** `faiss.IndexFlatL2(384)` with `index.add(xb)`, built fresh
  in every worker (outside the timed region).
- **Configurations (default full run).** `(threads, batch_size)` =
  `{(1,1),(2,1),(4,1), (1,32),(2,32),(4,32),(8,32),(16,32)}`, 5 reps each.
  `--threads-batch1` and `--threads-batch32` make these sets
  configurable; manifests enumerate exactly the chosen set.
- **Queries.** Same 512 distinct queries used in every (config, rep).
  Query order per rep is a deterministic permutation keyed by
  `(seed, rep)`; the same order is applied by every config within a
  rep. Query count was **not increased** for this extension — the
  observed per-rep ranges at every thread count stayed within the same
  envelope as the 3-thread historical runs, so the measured duration
  is sufficient for the comparisons we are making. Had we increased
  the query count, it would have been applied uniformly and labelled
  as a new protocol.
- **Process isolation.** Every (config, rep) runs in a fresh Python
  subprocess spawned by `scripts/run_bench.py`. Before the worker
  imports NumPy/FAISS, the orchestrator sets `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, and `BLIS_NUM_THREADS` to the requested
  count. After import, the worker calls
  `faiss.omp_set_num_threads(threads)` and records `threadpool_info()`
  from `threadpoolctl` for the loaded BLAS/OMP runtimes.
- **CPU affinity (P/E experiment).** `--cpu-affinity` is parsed by the
  worker **before** numpy/faiss import and applied via Windows
  `SetProcessAffinityMask`. Observed affinity (process-level mask) is
  read back via `GetProcessAffinityMask` and recorded per record.
  This is a **process-level restriction** on the set of CPUs the OS
  scheduler may use; it is not per-thread pinning (thread-to-core
  binding within the restricted set is still OS-controlled).
- **Timing.** Only `index.search()` calls are timed with
  `time.perf_counter()`. Two timing fields are recorded:
  - `sum_call_durations_s` — sum of per-call durations.
    **Throughput denominator.** Python overhead between calls is
    excluded.
  - `elapsed_around_loop_s` — wall from the first call's `t0` to the
    last call's `t1`. Diagnostic only; always >=
    `sum_call_durations_s`. Not used in headline metrics.
- **Agreement.** After the timed region, each worker makes a
  separate, untimed pass over the same queries to collect top-5 IDs.
  Analysis compares every record to the same-rep `(threads=1,
  batch=1)` run and fails if any required reference is missing.
- **Validation before publishing.** `analyze.py` runs all validation
  (manifest-vs-records, dataset identity hashes, timing-total
  consistency, nonfinite guards, agreement) **before any report
  artefact is written**. See `scripts/validate.py` and
  `tests/test_regression.py`.

## Hardware and software

Recorded in `runs/<run_id>/machine.json` and per-worker records:

- **CPU:** Intel(R) Core(TM) i9-14900HX, 24 physical cores / 32
  logical processors. **Hybrid: 8 P-cores (SMT, EfficiencyClass 1,
  logical CPUs 0–15), 16 E-cores (no SMT, EfficiencyClass 0, logical
  CPUs 16–31)**, verified by `scripts/topology.py`. WMI base clock
  2200 MHz; actual boost not measured.
- **OS:** Microsoft Windows 11 Home, build 10.0.26300. Native
  Windows, not WSL.
- **Memory:** ~15.73 GiB visible.
- **Power:** AC, plan "Balanced" during all runs.
- **Python:** 3.11.9.
- **Packages:** faiss-cpu 1.8.0, numpy 1.26.0, matplotlib 3.10.3,
  threadpoolctl 3.6.0, psutil 5.9.8.
- **BLAS/OMP runtimes (threadpoolctl at 4 threads):** OpenBLAS 0.3.23,
  two libomp libraries loaded by FAISS, all respecting the configured
  thread limit.
- **Explicitly not measured:** actual achieved CPU frequency and
  boost residency, cache utilisation, memory bandwidth,
  energy/power in watts, achieved CPU utilisation percentage, whether
  the OS scheduler placed workers on P- or E-cores under the
  **default** (unrestricted) sub-run. Any mechanism claim invoking
  these is a hypothesis. **No energy-per-query or
  performance-per-watt claims are made.**

## Reproducing

### 1. Isolated environment and install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If pip fails with an SSL certificate error on Windows Store Python:

```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### 2. Build the seeded dataset (once)

```powershell
python scripts\make_data.py
```

### 3. Scaling run (batch-1 at 1/2/4 threads + batch-32 at 1/2/4/8/16)

```powershell
python scripts\run_bench.py --mode full --reps 5
python scripts\analyze.py --run-dir (Get-ChildItem runs\*-full-* -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
```

To tune the thread sets:

```powershell
python scripts\run_bench.py --mode full --reps 5 `
    --threads-batch1 "1,2,4" --threads-batch32 "1,2,4,8,16"
```

### 4. Topology dump

```powershell
python scripts\topology.py --out runs\topology.json
```

### 5. P-core vs E-core experiment (three sub-runs, 5 reps each)

```powershell
python scripts\pe_experiment.py --reps 5 --threads 4
```

Writes a new `runs/<timestamp>-pe-<git short>/` with three sub-runs
and `comparison.json`.

### 6. Regression tests

```powershell
python tests\test_regression.py
```

All 47 checks must pass.

### 7. VTune profile collection (blocked on install — see `docs/VTUNE_COMMANDS.md`)

## Repository layout (additions from this extension)

```
scripts/
  profile_workload.py    fixed-duration VTune-ready workload
  topology.py            P/E-core enumeration (GetLogicalProcessorInformationEx)
  pe_experiment.py       3-sub-run P/E comparison orchestrator
docs/
  VTUNE_COMMANDS.md      exact install + collect commands
runs/
  <ts>-full-<git>/       as before (scaling runs)
  <ts>-pe-<git>/         P/E experiment (topology.json, comparison.json,
                         three sub-runs)
  <ts>-profile-<git>/    unprofiled fixed-duration runs as a baseline
```

## Limitations and reproduction boundary

- Synthetic Gaussian data; `IndexFlatL2`. Approximate indexes
  (`HNSW`, `IndexIVFPQ`, …) are out of scope.
- Only threads in {1, 2, 4, 8, 16} and batch sizes in {1, 32}, 5 reps
  each (plus the P/E sub-runs at 4 threads). The decision rule is a
  descriptive range-overlap check, not a significance test.
- **Variability.** The P/E sub-runs show per-rep outliers (3–8x slower
  than typical). The medians support the P-core / E-core ratio, but
  min/max are wide. Likely candidates include OMP thread-pool
  recreation cost under affinity restriction and background OS load;
  **not measured**.
- **Scheduler placement under default (unrestricted) runs** is not
  observed. Interpretations like "the default scheduler puts threads
  on P-cores" are consistent with the data but not verified.
- Cross-platform byte-identical regeneration of `xb.npy` from seeds is
  **not established** (an independent reviewer on Linux reports a
  different xb SHA-256 while xq matches). On this Legion, regenerating
  from seeds produces byte-identical files to the committed hashes.
  The benchmark is reproducible given the original Legion-generated
  `.npy` files identified by the committed hashes (the `.npy` files
  themselves are gitignored and not distributed in the repo); every
  worker refuses to run if the dataset drifts.
- No hardware counters, cache/bandwidth, RAPL energy, or per-core
  frequency were collected. VTune collection is Stage 2/3 and is
  blocked on install.

## What this extension verified

- **Thread scaling, 1→16 threads at batch=32:** peak at 8 threads
  (+118% median vs 1 thread), regression at 16 threads (+79%,
  down from +118%). Scaling judgment labelled `regression`.
- **P-core vs E-core affinity at 4 threads (preliminary):** medians
  3291 qps (P-cores) vs 1235 qps (E-cores), ratio ~2.66x; default
  scheduler median between them. **Preliminary** — placements were
  not interleaved and reps=5. A re-run with interleaved placements
  and 10 reps is the authoritative PE number; see the `*-pe2-*` run
  dir in the headline P/E section once published. Observed affinity
  verified to match requested in every per-call record.
- **Topology discovery:** `GetLogicalProcessorInformationEx` cleanly
  separates 8 P-cores from 16 E-cores on this hybrid CPU; result is
  deterministic and testable.
- **Validation hardening preserved:** all existing failure modes
  (forged sums, manifest mismatches, incomplete runs, missing
  references) still rejected; 47 / 47 regression tests pass.

## What this extension did NOT verify

- The 16-thread regression's **mechanism** (E-core spill vs SMT
  contention vs scheduler ping-pong) — requires VTune profiles
  (Stage 2/3, blocked).
- Cause of the 3–8x outlier reps in the P/E sub-runs.
- Boost clock residency on P- vs E-cores during the restricted runs.
- Energy or performance-per-watt comparisons. **No such claim is
  made.**
