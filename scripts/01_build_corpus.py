"""Build the frozen corpus subset from the raw pre-docked data.

Usage:
    python scripts/01_build_corpus.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import build_corpus_subset  # noqa: E402

if __name__ == "__main__":
    output_path = build_corpus_subset()
    print(f"Frozen corpus subset written to {output_path}")
