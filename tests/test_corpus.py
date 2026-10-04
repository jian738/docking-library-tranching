"""Tests for the per-target stratified sampling in src/corpus.py.

These assert the properties the sampler is supposed to guarantee, not just
that it runs: balanced per-target counts, full decile coverage per target,
and seeded reproducibility.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import stratified_sample_by_group  # noqa: E402


def _make_df(n_per_group: int, n_groups: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(n_groups):
        frames.append(pd.DataFrame({
            "target_id": [f"T{g}"] * n_per_group,
            "docking_score": rng.normal(loc=g, scale=5.0, size=n_per_group),
            "row_id": np.arange(n_per_group) + g * 100_000,
        }))
    return pd.concat(frames, ignore_index=True)


def test_equal_counts_per_target_when_groups_have_enough_rows():
    df = _make_df(n_per_group=300, n_groups=4)
    out = stratified_sample_by_group(
        df, group_column="target_id", score_column="docking_score",
        quota_per_group=50, n_bins=10, ascending=True, seed=42,
    )
    counts = out["target_id"].value_counts()
    assert set(counts.index) == {"T0", "T1", "T2", "T3"}
    assert (counts == 50).all()


def test_group_smaller_than_quota_keeps_all_its_rows_not_more_not_padded():
    df = _make_df(n_per_group=300, n_groups=1)
    small = pd.DataFrame({
        "target_id": ["SMALL"] * 5,
        "docking_score": np.linspace(-3, 3, 5),
        "row_id": np.arange(5),
    })
    df = pd.concat([df, small], ignore_index=True)

    out = stratified_sample_by_group(
        df, group_column="target_id", score_column="docking_score",
        quota_per_group=50, n_bins=10, ascending=True, seed=1,
    )
    counts = out["target_id"].value_counts()
    assert counts["T0"] == 50
    assert counts["SMALL"] == 5  # fewer rows than quota -> keep all, nothing dropped or duplicated


def test_remainder_not_dropped_when_quota_not_divisible_by_bin_count():
    df = _make_df(n_per_group=200, n_groups=1)
    quota = 53  # 53 % 10 != 0
    out = stratified_sample_by_group(
        df, group_column="target_id", score_column="docking_score",
        quota_per_group=quota, n_bins=10, ascending=True, seed=7,
    )
    assert len(out) == quota


def test_every_decile_represented_per_target():
    df = _make_df(n_per_group=500, n_groups=2)
    n_bins = 10
    out = stratified_sample_by_group(
        df, group_column="target_id", score_column="docking_score",
        quota_per_group=100, n_bins=n_bins, ascending=True, seed=3,
    )
    for target, full_group in df.groupby("target_id"):
        bin_labels = pd.qcut(full_group["docking_score"], q=n_bins, labels=False, duplicates="drop")
        sampled_idx = out.loc[out["target_id"] == target].index
        sampled_bins = bin_labels.loc[bin_labels.index.intersection(sampled_idx)]
        assert sampled_bins.nunique() == bin_labels.nunique(), (
            f"target {target}: only {sampled_bins.nunique()} of {bin_labels.nunique()} deciles represented"
        )


def test_same_seed_is_reproducible():
    df = _make_df(n_per_group=200, n_groups=3)
    out1 = stratified_sample_by_group(
        df, group_column="target_id", score_column="docking_score",
        quota_per_group=40, n_bins=10, ascending=True, seed=99,
    )
    out2 = stratified_sample_by_group(
        df, group_column="target_id", score_column="docking_score",
        quota_per_group=40, n_bins=10, ascending=True, seed=99,
    )
    pd.testing.assert_frame_equal(
        out1.sort_values("row_id").reset_index(drop=True),
        out2.sort_values("row_id").reset_index(drop=True),
    )


def test_different_seed_changes_the_subset():
    df = _make_df(n_per_group=200, n_groups=3)
    out1 = stratified_sample_by_group(
        df, group_column="target_id", score_column="docking_score",
        quota_per_group=40, n_bins=10, ascending=True, seed=1,
    )
    out2 = stratified_sample_by_group(
        df, group_column="target_id", score_column="docking_score",
        quota_per_group=40, n_bins=10, ascending=True, seed=2,
    )
    assert set(out1["row_id"]) != set(out2["row_id"])


def test_build_corpus_subset_matches_config_quota_on_real_data():
    from corpus import DEFAULT_CONFIG_PATH, build_corpus_subset, load_config

    config = load_config(DEFAULT_CONFIG_PATH)
    output_path = build_corpus_subset(DEFAULT_CONFIG_PATH)
    df = pd.read_parquet(output_path)

    quota = config["selection"]["top_n_per_target"]
    counts = df[config["selection"]["group_column"]].value_counts()
    assert (counts == quota).all(), counts.to_dict()
