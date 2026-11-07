# Running Intel VTune Profiler against this benchmark

Intel VTune Profiler is **not installed** on this Legion as of commit
`97132e8`. These are the exact PowerShell commands to install it and
collect the four comparison profiles the review requires
(1 / 4 / 8 / 16 threads). The 16-thread profile is necessary to
investigate the observed regression.

The workload is `scripts/profile_workload.py` (fixed wall-clock
duration). Headline throughput still comes from
`scripts/run_bench.py`; profiles are diagnostic evidence only.

## 1. Install VTune

Download from Intel (free, no license):

- **Standalone Intel VTune Profiler**:
  <https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html>
- or as part of the **Intel oneAPI Base Toolkit**:
  <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>

Accept the defaults. Standalone installs to roughly:

```
C:\Program Files (x86)\Intel\oneAPI\vtune\<version>\bin64\vtune.exe
C:\Program Files (x86)\Intel\oneAPI\vtune\<version>\sdk\lib64\libittnotify.dll
```

## 2. Initialise the VTune environment (PowerShell, official Intel path)

Intel ships **`setvars.ps1`** for PowerShell (and `setvars.bat` for cmd).
The official way to initialise VTune in a PowerShell session is:

```powershell
# Option A: oneAPI (sets env for VTune and every other oneAPI tool)
. "C:\Program Files (x86)\Intel\oneAPI\setvars.ps1"

# Option B: VTune standalone, explicit (no setvars dependency)
$VTUNE_BIN = (Get-ChildItem "C:\Program Files (x86)\Intel\oneAPI\vtune\*" -Directory |
              Sort-Object Name -Descending | Select-Object -First 1).FullName + "\bin64"
$VTUNE_ITT = (Get-ChildItem "C:\Program Files (x86)\Intel\oneAPI\vtune\*" -Directory |
              Sort-Object Name -Descending | Select-Object -First 1).FullName + "\sdk\lib64"
$env:PATH         = "$VTUNE_BIN;$env:PATH"
$env:ITT_LIB_DIR  = $VTUNE_ITT   # read by scripts/profile_workload.py
```

> **Do not** use `& "...setvars.bat"` — a `.bat` run under `&` sets
> env vars only in the child process, not in your PowerShell session.
> Use `setvars.ps1` with dot-sourcing or set `$env:PATH` directly.

Verify:

```powershell
vtune --version
# -> Intel(R) VTune(TM) Profiler <X.Y.Z>
```

## 3. Region filtering (ITT): only the measured region is profiled

`scripts/profile_workload.py` loads `libittnotify.dll` and calls
`__itt_resume()` at the start of the measured loop and `__itt_pause()`
at the end. When VTune is launched with **`-start-paused`**, data
collection begins only when `__itt_resume` fires. That excludes:

- Python startup
- `IndexFlatL2` construction and `index.add(xb)`
- 10 warm-up search calls
- The post-region sleep / shutdown

The workload JSON summary records `itt_available=true` and
`itt_library_path=<full path>` when the DLL loaded. If
`itt_available=false` in the JSON, VTune will have collected the whole
process lifetime — fix the install or set `$env:ITT_LIB_DIR` and
re-run before interpreting any numbers.

## 4. Hotspots at four thread counts

```powershell
cd C:\Users\sdas8\faiss-cpu-benchmark
$RUN = "runs\vtune-hotspots-$(Get-Date -Format yyyyMMdd-HHmmss)"
New-Item -ItemType Directory -Force -Path $RUN | Out-Null

foreach ($t in 1,4,8,16) {
    $env:OMP_NUM_THREADS       = "$t"
    $env:MKL_NUM_THREADS       = "$t"
    $env:OPENBLAS_NUM_THREADS  = "$t"
    $env:VECLIB_MAXIMUM_THREADS = "$t"
    $env:NUMEXPR_NUM_THREADS   = "$t"
    $env:BLIS_NUM_THREADS      = "$t"
    $resdir = "$RUN\hs_t$t"
    vtune -collect hotspots `
        -start-paused `
        -result-dir $resdir `
        -finalization-mode=full `
        -- python scripts\profile_workload.py `
            --threads $t --duration-s 25 `
            --out "$resdir\workload_summary.json"
}
```

Export tables (CSV) per thread count:

```powershell
foreach ($t in 1,4,8,16) {
    $resdir = (Get-ChildItem "$RUN\hs_t$t" -Directory | Select-Object -First 1).FullName
    vtune -report summary          -r $resdir -format=csv -report-output "$resdir\summary.csv"
    vtune -report hotspots         -r $resdir -format=csv -report-output "$resdir\hotspots.csv"
    vtune -report callstacks       -r $resdir -format=csv -report-output "$resdir\callstacks.csv"
    # Thread/core placement over time (Hotspots with threading analysis):
    vtune -report hw-events -group-by=cpu -r $resdir -format=csv `
          -report-output "$resdir\by_cpu.csv" 2>&1 | Out-String
}
```

The `-report hw-events -group-by=cpu` call lists per-CPU event totals
and will tell you whether 16-thread execution actually touched E-core
CPU ids (16..31). This is the direct test of the "spills to E-cores"
hypothesis.

## 5. Microarchitecture Exploration (hardware events)

Only run this after Hotspots works. Needs the VTune sampling driver.
On a laptop the installer offers a "sampling driver" step; accept it.

```powershell
$RUN2 = "runs\vtune-uarch-$(Get-Date -Format yyyyMMdd-HHmmss)"
New-Item -ItemType Directory -Force -Path $RUN2 | Out-Null

foreach ($t in 1,4,8,16) {
    $env:OMP_NUM_THREADS = "$t"; $env:MKL_NUM_THREADS = "$t"
    $env:OPENBLAS_NUM_THREADS = "$t"
    $resdir = "$RUN2\uarch_t$t"
    vtune -collect uarch-exploration `
        -start-paused `
        -result-dir $resdir `
        -finalization-mode=full `
        -- python scripts\profile_workload.py `
            --threads $t --duration-s 25 `
            --out "$resdir\workload_summary.json"
    vtune -report summary -r $resdir -format=csv `
          -report-output "$resdir\summary.csv"
    vtune -report hw-events -group-by=cpu -r $resdir -format=csv `
          -report-output "$resdir\by_cpu.csv" 2>&1 | Out-String
}
```

Fallback if `uarch-exploration` reports metrics unavailable
(happens on some consumer SKUs without the kernel driver):

```powershell
vtune -collect performance-snapshot -start-paused `
    -result-dir "$RUN2\snap_t4" `
    -- python scripts\profile_workload.py --threads 4 --duration-s 25 `
        --out "$RUN2\snap_t4\workload_summary.json"
```

## 6. Matched-work normalisation

The profile runs are fixed **duration**, so the number of queries per
run differs across thread counts. Do NOT compare raw totals (e.g.
total cycles, total instructions) across thread counts directly.
Normalise by `queries_in_region` from `workload_summary.json`. Example:

```powershell
foreach ($t in 1,4,8,16) {
    $sum = Import-Csv "runs\vtune-uarch\uarch_t$t\summary.csv"
    $work = Get-Content "runs\vtune-uarch\uarch_t$t\workload_summary.json" | ConvertFrom-Json
    $queries = $work.queries_in_region
    # Divide "Clockticks" and "Instructions Retired" by $queries before compare.
}
```

Both the hw-event total and `queries_in_region` cover the **same**
ITT-delimited region, so the ratio is a clean per-query metric.

## 7. What to save in Git

**Do commit** (text, small):
- `summary.csv`, `hotspots.csv`, `callstacks.csv`, `by_cpu.csv`
  per thread count.
- `workload_summary.json` per run.
- Optional readable screenshots of the Hotspots top-functions view
  and the Microarchitecture Exploration summary.

**Do NOT commit** the `*.vtune` / `r000hs` raw capture directories
(can be hundreds of MB). Keep them in `runs/vtune-*` and add that
exact path to `.gitignore` if needed.

## 8. Known blockers on consumer Windows laptops

- Some metrics require the VTune kernel-mode sampling driver and
  administrator rights. If VTune says "metric unavailable," record
  that verbatim in your writeup rather than inferring the value.
- Boost-clock and thermal behaviour over a 25-second collection is
  outside VTune's control; keep the laptop on AC and avoid GPU work
  during collection.
- Hybrid CPU (P + E cores): VTune 2024.0+ shows core-type in the
  CPU-usage view. Older versions may not. Check
  `vtune --version` and fall back to `by_cpu.csv` + the topology JSON
  (`scripts/topology.py --out runs\topology.json`) for a manual
  P/E-vs-CPU-id mapping.
