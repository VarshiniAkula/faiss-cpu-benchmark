"""Build seeded synthetic dataset for the benchmark.

Database: 50,000 x 384 float32 standard normal, fixed seed.
Queries : 512 x 384 float32 standard normal, different fixed seed.

We save the raw arrays as .npy. Each worker rebuilds IndexFlatL2 from xb;
rebuild time is outside the timed region, so this is fine and keeps the
data files simple and inspectable.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

NB = 50_000
NQ = 512
D = 384
DB_SEED = 20261106
Q_SEED = 20261106 + 1


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng_db = np.random.default_rng(DB_SEED)
    xb = rng_db.standard_normal((NB, D), dtype=np.float32)
    rng_q = np.random.default_rng(Q_SEED)
    xq = rng_q.standard_normal((NQ, D), dtype=np.float32)
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    np.save(DATA / "xb.npy", xb)
    np.save(DATA / "xq.npy", xq)
    meta = {
        "nb": NB, "nq": NQ, "d": D,
        "dtype": "float32", "metric": "L2", "index_type": "IndexFlatL2",
        "db_seed": DB_SEED, "query_seed": Q_SEED,
    }
    (DATA / "dataset.json").write_text(__import__("json").dumps(meta, indent=2))
    print(f"wrote xb={xb.shape} xq={xq.shape} -> {DATA}")


if __name__ == "__main__":
    main()
