"""Benchmark target-agnostic ECFP4 tranching: intra-tranche docking-score
variance and top-1% enrichment, per target.

Usage:
    python scripts/03_tranching_benchmark.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranching import run_benchmark  # noqa: E402

if __name__ == "__main__":
    output_path = run_benchmark()
    results = pd.read_parquet(output_path)
    print(f"Benchmark results written to {output_path}\n")
    pivot = results.pivot(index="target", columns="metric", values="value")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
