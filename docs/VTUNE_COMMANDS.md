# Running Intel VTune Profiler against this benchmark

VTune Profiler is not installed on this Legion as of commit `8b401ae`.
These are the exact PowerShell commands to install it and collect the
three comparison profiles the review asks for (1 thread, 4 threads, and
the best-observed `>4` thread configuration — on this machine, 8
threads).

The workload is `scripts/profile_workload.py` configured for a fixed
wall-clock duration so each collection spans roughly the same real time.
Headline throughput numbers still come from `scripts/run_bench.py`;
profiles are used as diagnostic evidence only.

## 1. Install VTune

Download from Intel (free, no license):

- Standalone **Intel VTune Profiler** installer:
  <https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html>
- Or install as part of the **Intel oneAPI Base Toolkit**:
  <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>

Install with defaults. After installation, open a fresh PowerShell and
initialise the VTune environment (the exact path reflects the installed
version; `2024.X` is a placeholder):

```powershell
& "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
vtune --version
```

`vtune --version` must succeed before running any collection.

## 2. Hotspots: three thread counts

Collect a 25-second Hotspots profile at **1**, **4**, and **8**
threads. The thread-env block applies BEFORE the Python import so
OpenMP picks up the restricted set.

```powershell
cd C:\Users\sdas8\faiss-cpu-benchmark
$RUN = "runs\vtune-hotspots"
New-Item -ItemType Directory -Force -Path $RUN | Out-Null

foreach ($t in 1,4,8) {
    $env:OMP_NUM_THREADS       = "$t"
    $env:MKL_NUM_THREADS       = "$t"
    $env:OPENBLAS_NUM_THREADS  = "$t"
    $env:VECLIB_MAXIMUM_THREADS = "$t"
    $env:NUMEXPR_NUM_THREADS   = "$t"
    $env:BLIS_NUM_THREADS      = "$t"
    $resdir = "$RUN\hs_t$t"
    vtune -collect hotspots `
        -result-dir $resdir `
        -finalization-mode=full `
        -- python scripts\profile_workload.py `
            --threads $t --duration-s 25 `
            --out "$resdir\workload_summary.json"
}
```

Then export per-collection summaries (CSV) into the same directories:

```powershell
foreach ($t in 1,4,8) {
    $resdir = "$RUN\hs_t$t"
    vtune -report summary          -r $resdir -format=csv -report-output "$resdir\summary.csv"
    vtune -report hotspots         -r $resdir -format=csv -report-output "$resdir\hotspots.csv"
    vtune -report callstacks       -r $resdir -format=csv -report-output "$resdir\callstacks.csv"
}
```

## 3. Microarchitecture Exploration (hardware events)

Only run this after Hotspots works. Requires that VTune's sampling
drivers are installed; on a laptop you may be asked to allow driverless
sampling or, for deeper events, install the kernel-mode collector.

```powershell
$RUN2 = "runs\vtune-uarch"
New-Item -ItemType Directory -Force -Path $RUN2 | Out-Null

foreach ($t in 1,4,8) {
    $env:OMP_NUM_THREADS = "$t"; $env:MKL_NUM_THREADS = "$t"
    $env:OPENBLAS_NUM_THREADS = "$t"
    $resdir = "$RUN2\uarch_t$t"
    vtune -collect uarch-exploration `
        -result-dir $resdir `
        -finalization-mode=full `
        -- python scripts\profile_workload.py `
            --threads $t --duration-s 25 `
            --out "$resdir\workload_summary.json"
    vtune -report summary -r $resdir -format=csv -report-output "$resdir\summary.csv"
}
```

If `uarch-exploration` reports "metric not available," run the minimal
alternative first:

```powershell
vtune -collect performance-snapshot `
    -result-dir "$RUN2\snap_t4" `
    -- python scripts\profile_workload.py --threads 4 --duration-s 25 `
        --out "$RUN2\snap_t4\workload_summary.json"
```

## 4. What to save in Git

**Do commit:**
- `summary.csv`, `hotspots.csv`, `callstacks.csv` per thread count
  (text, small).
- `workload_summary.json` per run (small; this file comes from
  `profile_workload.py`, not from VTune).
- Optional: readable screenshots of the Hotspots top-functions view and
  the Microarchitecture Exploration summary.

**Do NOT commit:**
- The `*.vtune` or `r000hs` capture directories (can be hundreds of MB).
  Keep them under `runs/vtune-*` but add that path to `.gitignore` if
  the repository gets heavy.

## 5. Matched-work interpretation

The profile runs are fixed **duration**, so the number of queries per
run differs across thread counts. Do NOT compare raw totals (e.g. total
cycles) directly. Normalise by `queries_in_region` from
`workload_summary.json` before comparing across thread counts.

Example (compute cycles per query across runs):

```powershell
foreach ($t in 1,4,8) {
    $sum = Get-Content "runs\vtune-uarch\uarch_t$t\summary.csv"
    $work = Get-Content "runs\vtune-uarch\uarch_t$t\workload_summary.json" | ConvertFrom-Json
    # pull "Clockticks" (or "CPU Time") from $sum and divide by $work.queries_in_region
}
```

## 6. Known issues on consumer Windows laptops

- Some hardware counters require administrator privileges and the VTune
  driver. If a metric is unavailable, VTune will say so — record that
  as "unavailable on this host" rather than inferring the value.
- Power-plan and boost-clock behaviour during a 25-second collection is
  not under VTune's control. Keep the laptop on AC and avoid
  background GPU workloads during collection.
