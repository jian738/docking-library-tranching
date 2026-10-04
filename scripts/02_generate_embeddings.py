"""Generate ECFP4 and ECFP6 embeddings for the unique molecules in the frozen
corpus subset.

Usage:
    python scripts/02_generate_embeddings.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embeddings import build_embeddings  # noqa: E402

if __name__ == "__main__":
    output_paths = build_embeddings()
    for name, path in output_paths.items():
        print(f"{name} embeddings written to {path}")
