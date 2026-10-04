"""Tests for src/tranching.py: tranche assignment and per-target metrics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranching import assign_tranches, compute_target_metrics  # noqa: E402


def test_compute_target_metrics_known_values():
    # Two tranches of 3 molecules each. Tranche 0 has the better mean score
    # and contains the single best-scoring molecule (-10), so a top-1
    # selection should land entirely inside tranche 0.
    target_df = pd.DataFrame({
        "mol_id": [f"M{i}" for i in range(6)],
        "docking_score": [-10, -9, -8, -1, 0, 1],
        "tranche_id": [0, 0, 0, 1, 1, 1],
    })

    metrics = compute_target_metrics(target_df, top_pct=1 / 6)

    assert metrics["intra_tranche_variance"] == pytest.approx(2 / 3)
    assert metrics["top1pct_enrichment"] == pytest.approx(1.0)
    assert metrics["top1pct_enrichment_random_baseline"] == pytest.approx(0.5)


def test_compute_target_metrics_worse_than_random_is_detectable():
    # Best-mean tranche (0) deliberately excludes the single top-scoring
    # molecule, which sits in tranche 1 instead -> enrichment should be 0,
    # strictly below the random baseline (tranche 0's size share).
    target_df = pd.DataFrame({
        "mol_id": [f"M{i}" for i in range(6)],
        "docking_score": [-9, -8, -7, -10, 0, 1],
        "tranche_id": [0, 0, 0, 1, 1, 1],
    })

    metrics = compute_target_metrics(target_df, top_pct=1 / 6)

    assert metrics["top1pct_enrichment"] == pytest.approx(0.0)
    assert metrics["top1pct_enrichment_random_baseline"] == pytest.approx(0.5)
    assert metrics["top1pct_enrichment"] < metrics["top1pct_enrichment_random_baseline"]


def test_assign_tranches_deterministic_with_seed():
    rng = np.random.default_rng(0)
    embeddings_df = pd.DataFrame(
        rng.integers(0, 2, size=(40, 16)),
        columns=[f"bit_{i}" for i in range(16)],
    )
    embeddings_df.insert(0, "mol_id", [f"M{i}" for i in range(40)])

    out1 = assign_tranches(embeddings_df, k=4, seed=7)
    out2 = assign_tranches(embeddings_df, k=4, seed=7)

    assert set(out1["tranche_id"].unique()) <= set(range(4))
    pd.testing.assert_frame_equal(out1, out2)
