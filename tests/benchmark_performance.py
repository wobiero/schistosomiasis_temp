from __future__ import annotations

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schisto_tool.parameters import HaematobiumInputs, MansoniInputs
from schisto_tool.simulation import run_monte_carlo_haematobium, run_monte_carlo_mansoni


def _clear_cache(func) -> None:
    clear = getattr(func, "clear", None)
    if callable(clear):
        clear()


def _bench(label: str, func, *args) -> float:
    _clear_cache(func)
    start = time.perf_counter()
    df = func(*args)
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.4f}s | shape={df.shape}")
    return elapsed


def main() -> None:
    for n in (1_000, 10_000):
        _bench(
            f"S. mansoni PSA, n={n:,}",
            run_monte_carlo_mansoni,
            n,
            100_000,
            20.0,
            MansoniInputs(at_risk_pop=100_000),
            0.75,
            42,
        )
        _bench(
            f"S. haematobium PSA, n={n:,}",
            run_monte_carlo_haematobium,
            n,
            100_000,
            20.0,
            0.50,
            60.0,
            HaematobiumInputs(at_risk_pop=100_000),
            0.75,
            43,
        )


if __name__ == "__main__":
    main()
