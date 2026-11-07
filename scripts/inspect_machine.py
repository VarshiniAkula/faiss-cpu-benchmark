"""Collect hardware and software metadata; write results/machine.json.

No FAISS import here on purpose — the orchestrator writes a separate
runtime record once thread limits have been applied.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "machine.json"


def _ps(cmd: str) -> str:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=20,
        )
        return (r.stdout or "").strip()
    except Exception as e:
        return f"__err__:{e}"


def detect_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version") as f:
            v = f.read().lower()
        return "microsoft" in v or "wsl" in v
    except Exception:
        return False


def main() -> None:
    info: dict = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_string": platform.processor(),
        "wsl": detect_wsl(),
        "native_windows": sys.platform == "win32",
    }

    if sys.platform == "win32":
        cpu_json = _ps(
            "Get-CimInstance Win32_Processor | Select-Object -First 1 "
            "Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,"
            "MaxClockSpeed,L2CacheSize,L3CacheSize | ConvertTo-Json -Compress"
        )
        os_json = _ps(
            "Get-CimInstance Win32_OperatingSystem | Select-Object "
            "Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize "
            "| ConvertTo-Json -Compress"
        )
        bat_json = _ps(
            "$b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue; "
            "if ($b) { $b | Select-Object -First 1 BatteryStatus,"
            "EstimatedChargeRemaining | ConvertTo-Json -Compress } else { '{}' }"
        )
        plan = _ps("(powercfg /getactivescheme) 2>&1")
        try:
            info["cpu_wmi"] = json.loads(cpu_json) if cpu_json else {}
        except Exception:
            info["cpu_wmi_raw"] = cpu_json
        try:
            info["os_wmi"] = json.loads(os_json) if os_json else {}
        except Exception:
            info["os_wmi_raw"] = os_json
        try:
            info["battery_wmi"] = json.loads(bat_json) if bat_json else {}
        except Exception:
            info["battery_wmi_raw"] = bat_json
        info["active_power_plan"] = plan
    info["env_thread_vars_at_inspect"] = {
        k: os.environ.get(k)
        for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        )
    }
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except Exception as e:
        info["numpy_version"] = f"__err__:{e}"
    try:
        import faiss
        info["faiss_version"] = faiss.__version__
        info["faiss_omp_max_threads_default"] = int(faiss.omp_get_max_threads())
    except Exception as e:
        info["faiss_version"] = f"__err__:{e}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(info, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
