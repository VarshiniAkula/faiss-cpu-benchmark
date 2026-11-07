# FAISS CPU Retrieval Benchmark (synthetic mode)

A small, reproducible FAISS CPU retrieval benchmark run on a Lenovo Legion
laptop. This is synthetic-mode: no HyArg artifacts were supplied, so the
dataset is seeded synthetic.

## Benchmark question

On this CPU, with an `IndexFlatL2` of 50,000 x 384-D float32 vectors and
`k=5`:

1. **Interactive use case (batch=1):** which of 1, 2, or 4 requested
   threads gives the lowest p95 per-call search latency?
2. **Throughput use case (batch=32):** which gives the highest throughput
   when 32 queries are already waiting?

Two different questions, same index, same queries, same k.

## Hardware and software

Collected by `scripts/inspect_machine.py` into `results/machine.json`.

- CPU: **Intel(R) Core(TM) i9-14900HX**, 24 physical cores / 32 logical
  processors (hybrid P-core + E-core). Max base clock reported by WMI:
  2200 MHz. **Boost clock not directly measured here.**
- OS: Microsoft Windows 11 Home, build 10.0.26300. Native Windows, not WSL.
- Memory: ~15.73 GiB visible.
- Python: 3.11.9 (Windows Store distribution).
- NumPy: 1.26.0.
- FAISS: `faiss-cpu` 1.8.0 (CPU, no GPU use). Default `omp_get_max_threads`
  on this machine without env overrides: 32.
- Power: on AC power, active plan "Balanced". Battery state was not
  constant across the run (charging from ~32% to ~37% while benchmarking).
  **Boost/throttling behavior not independently measured; see limitations.**

## Dataset (synthetic)

Seeded standard-normal float32 arrays. Built once by `scripts/make_data.py`
and saved to `data/`:

| name | shape       | dtype   | seed       |
|------|-------------|---------|------------|
| xb   | 50000 x 384 | float32 | 20261106   |
| xq   | 512   x 384 | float32 | 20261107   |

Index: `faiss.IndexFlatL2(384)` with `index.add(xb)`. Exact search, no
approximation. k=5.

## Timing boundaries

What is included in the timed region, per worker process:

- Only the `index.search(xq_slice, k)` calls.
- For batch=1: 512 individual calls, one per query.
- For batch=32: 16 batch calls (512 queries total).

What is **excluded** from timing:

- Synthetic data generation and `.npy` load.
- `IndexFlatL2` construction and `index.add(xb)`.
- 10 warm-up search calls per worker (not recorded).
- The separate, untimed pass that collects top-5 IDs for the agreement
  check.
- Python startup, FAISS import, file writes.

Timer: `time.perf_counter()` (monotonic, high resolution).

## Experiment protocol

- 6 configurations: `(threads, batch_size) in {(1,1),(2,1),(4,1),(1,32),(2,32),(4,32)}`.
- 5 measured repetitions per configuration.
- Every (config, rep) runs in a fresh Python subprocess spawned by
  `scripts/run_bench.py`. Before the worker imports NumPy/FAISS, the
  orchestrator sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`,
  and `BLIS_NUM_THREADS` to the requested thread count. After import, the
  worker also calls `faiss.omp_set_num_threads(threads)` explicitly and
  records `faiss.omp_get_max_threads()` after the call.
- Per repetition, the configuration order is a deterministic shuffle keyed
  by `(seed, rep)`. The actual order used is logged to
  `results/run_order.json`.
- Within a rep, all six configs process the same 512 queries in the same
  shuffled order (the per-rep order is also deterministic given `(seed,
  rep)`), so results align 1:1 across configs for the same rep.
- Metric per rep:
  - `batch=1`: `p95` across the 512 per-call durations.
  - `batch=32`: `queries_processed / total_timed_seconds`.
- Aggregation across the 5 reps: **median** and `[min, max]` range.

"Configured thread limits" (what we set) and "observed CPU utilization"
(what the OS scheduler actually did) are not the same thing. We only
record and control the former.

## Results

Aggregated across 5 reps per configuration (full per-rep data in
`results/summary.csv`):

### Interactive (batch=1, metric = p95 per-call latency, ms)

| threads | median p95 (ms) | min    | max    | % vs 1-thread median |
|---------|-----------------|--------|--------|----------------------|
| 1       | 3.921           | 3.712  | 4.045  | —                    |
| 2       | 3.962           | 3.757  | 4.600  | +1.1%                |
| 4       | 3.883           | 3.803  | 4.115  | -1.0%                |

### Throughput (batch=32, metric = queries/sec)

| threads | median qps | min    | max    | % vs 1-thread median |
|---------|------------|--------|--------|----------------------|
| 1       | 1508.6     | 1476.3 | 1540.1 | —                    |
| 2       | 2228.0     | 2148.0 | 2422.8 | +47.7%               |
| 4       | 2755.9     | 2643.0 | 3100.7 | +82.7%               |

Figure: `results/figure.png` (two panels, latency + throughput vs
threads, error bars show `[min, max]` across the 5 reps).

### Agreement with reference

Agreement rule: for each `(config, rep)`, compare the top-5 ID sets to
the `(threads=1, batch=1, same rep)` run on the same queries, outside
the timed region.

| threads | batch | n_reps | min frac | median frac | max frac |
|---------|-------|--------|----------|-------------|----------|
| 1       | 1     | 5      | 1.000    | 1.000       | 1.000    |
| 1       | 32    | 5      | 1.000    | 1.000       | 1.000    |
| 2       | 1     | 5      | 1.000    | 1.000       | 1.000    |
| 2       | 32    | 5      | 1.000    | 1.000       | 1.000    |
| 4       | 1     | 5      | 1.000    | 1.000       | 1.000    |
| 4       | 32    | 5      | 1.000    | 1.000       | 1.000    |

All configurations return identical top-5 sets. This is expected for
exact search with `IndexFlatL2`; threading does not change the computed
L2 distances, and there are no tie-break differences at k=5 on this
dataset. This is **agreement with a per-rep reference**, not Recall@5
against a labeled relevance dataset.

## Practical recommendations

- **Interactive (one query at a time).** On this machine, with this
  index and k, the p95 differences between 1, 2, and 4 threads are
  smaller than the run-to-run range. The experiment **did not establish
  a clear winner for the interactive use case**; 1 thread is a reasonable
  default. Thread-count tuning here is unlikely to move p95 by more than
  a few percent.
- **Throughput (32 queries waiting).** Use the **4-thread** setting:
  median ~2756 qps, ~+82.7% vs 1 thread. 2 threads already gives most
  of the win (+47.7%); the jump from 2 to 4 is a further ~+23.7%.

These recommendations are specific to this CPU, this index type and
size, this dataset, this k, and the "Balanced" power plan on AC power
during the run.

## Limitations

- Synthetic standard-normal data only. No HyArg artifacts were provided
  to the run. A real HyArg index may have different access patterns,
  cardinality, and dimensionality.
- `IndexFlatL2` is an exact brute-force index. Conclusions do not
  transfer directly to `IndexIVFPQ`, `HNSW`, or other approximate
  indexes.
- Only thread counts 1, 2, 4 were tested. 8 and 16 threads may change
  the picture, especially on a 24-core CPU.
- Only two batch sizes (1 and 32) were tested.
- Battery state was not constant during the run (on AC, charging). I
  did not measure or constrain P-core vs E-core placement or CPU
  frequency; the OS scheduler controlled both. Explanations about cache,
  memory bandwidth, or P/E scheduling would be **hypotheses, not
  measured results**.
- Hardware performance counter data was not collected (not available
  without additional setup on native Windows for this user).
- WMI's `MaxClockSpeed` field reports a nominal base, not the boost
  clock in effect during the run.

## Reproducing

All commands below are PowerShell on Windows (what was used to produce
these results). They assume `faiss-cpu==1.8.0`, `numpy==1.26.0`, and
`matplotlib==3.10.3` are installed into the Python interpreter named
`python`.

Install dependencies:

```powershell
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

Smoke test (one config, one rep):

```powershell
python scripts\inspect_machine.py
python scripts\make_data.py
python scripts\run_bench.py --smoke
python scripts\analyze.py
```

Full run and figure (what produced the numbers above):

```powershell
python scripts\inspect_machine.py
python scripts\make_data.py
python scripts\run_bench.py --reps 5
python scripts\analyze.py
```

Repository layout:

```
faiss-cpu-benchmark/
  requirements.txt
  README.md
  scripts/
    inspect_machine.py   hardware/software metadata
    make_data.py         seeded synthetic dataset
    worker.py            single-config benchmark worker
    run_bench.py         orchestrator (spawns fresh workers)
    analyze.py           aggregate + figure + agreement + table
  data/                  xb.npy, xq.npy, dataset.json
  results/
    machine.json
    run_order.json
    per_call/            one JSON per (config, rep) with raw timings
    summary.csv          per-rep metrics
    summary_agg.csv      median [min, max] per config
    agreement.json       per-rep top-5 identity vs reference
    recommendations.json machine-readable best-per-use-case
    figure.png           two-panel chart
```

## What was and was not verified

- Verified by this run: 30 per-call JSON files written (6 configs x 5
  reps); every file contains 512 queries; the agreement check ran against
  per-rep references; the figure's medians match `summary_agg.csv`; the
  configured `OMP_NUM_THREADS` matches `faiss.omp_get_max_threads()`
  after `faiss.omp_set_num_threads` in every worker record.
- Not verified: actual achieved CPU frequency during the run,
  P-core vs E-core placement of worker threads, memory bandwidth
  utilization, cache behavior, and OS scheduler decisions. Any claim
  about those mechanisms would be a hypothesis.
