"""Build a frozen corpus subset from the raw pre-docked compound data."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported raw data format: {path.suffix}")


def _derived_seed(base_seed: int, *parts: object) -> int:
    """Deterministic seed derived from base_seed and arbitrary parts.

    Independent of groupby iteration order, so re-running with the same
    base_seed reproduces the same subset regardless of processing order.
    """
    digest = hashlib.sha256(f"{base_seed}|{'|'.join(str(p) for p in parts)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def stratified_sample_by_group(
    df: pd.DataFrame,
    group_column: str,
    score_column: str,
    quota_per_group: int,
    n_bins: int = 10,
    ascending: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample up to `quota_per_group` rows from each group in `group_column`,
    stratified by that group's OWN `score_column` distribution.

    Each group's rows are binned into `n_bins` quantile bins (deciles for
    n_bins=10) computed from that group's scores alone -- score scales differ
    between groups (e.g. per-target docking scores), so a global binning
    would be meaningless. An equal share of the quota is drawn from each bin,
    guaranteeing the strong-scoring tail is represented rather than left to
    chance, as a plain top-N or head() would risk.

    Edge cases:
      - Group has fewer rows than quota_per_group: all of its rows are kept
        (there is nothing more to sample); it contributes fewer than
        quota_per_group rows to the result.
      - quota_per_group does not divide evenly across a group's bins: the
        remainder is assigned one-at-a-time to the best-scoring bins first
        (per `ascending`), so no rows are silently dropped.
    """
    sampled_parts: list[pd.DataFrame] = []

    for group_value, group_df in df.groupby(group_column, sort=True):
        quota = min(quota_per_group, len(group_df))

        try:
            bin_labels = pd.qcut(group_df[score_column], q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            bin_labels = pd.Series(0, index=group_df.index)

        bins = sorted(bin_labels.dropna().unique())
        if not bins:
            bin_labels = pd.Series(0, index=group_df.index)
            bins = [0]

        base, remainder = divmod(quota, len(bins))
        bin_order = bins if ascending else list(reversed(bins))
        bin_quotas = {b: base for b in bins}
        for b in bin_order[:remainder]:
            bin_quotas[b] += 1

        bin_chunks = []
        for b in bins:
            bin_df = group_df[bin_labels == b]
            take = min(bin_quotas[b], len(bin_df))
            if take <= 0:
                continue
            bin_seed = _derived_seed(seed, group_value, b)
            bin_chunks.append(bin_df.sample(n=take, random_state=np.random.default_rng(bin_seed)))

        sampled = pd.concat(bin_chunks) if bin_chunks else group_df.iloc[0:0]

        # A bin can fall short of its quota (e.g. tied scores collapse bins
        # unevenly); top up from the group's remaining rows so the group
        # still reaches `quota` instead of silently returning fewer rows.
        shortfall = quota - len(sampled)
        if shortfall > 0:
            remaining = group_df.drop(sampled.index)
            topup_seed = _derived_seed(seed, group_value, "topup")
            topup = remaining.sample(n=min(shortfall, len(remaining)), random_state=np.random.default_rng(topup_seed))
            sampled = pd.concat([sampled, topup])

        sampled_parts.append(sampled)

    return pd.concat(sampled_parts) if sampled_parts else df.iloc[0:0]


def build_corpus_subset(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Load the raw pre-docked data, apply the subset selection rules from
    config.yaml, and write the frozen subset to data/corpus_subset/.

    Returns the path to the written parquet file.
    """
    config = load_config(config_path)

    raw_path = REPO_ROOT / config["input"]["raw_file"]
    df = _read_table(raw_path)

    sel = config["selection"]
    seed = config.get("seed", 0)

    if sel.get("drop_decoys") and "is_decoy" in df.columns:
        df = df[~df["is_decoy"]]

    required_columns = sel.get("required_columns") or []
    if required_columns:
        df = df.dropna(subset=required_columns)

    score_column = sel.get("score_column")
    ascending = sel.get("ascending", True)

    if score_column and sel.get("score_threshold") is not None:
        threshold = sel["score_threshold"]
        df = df[df[score_column] <= threshold] if ascending else df[df[score_column] >= threshold]

    top_n_per_target = sel.get("top_n_per_target")
    group_column = sel.get("group_column")

    if top_n_per_target and group_column and score_column:
        df = stratified_sample_by_group(
            df,
            group_column=group_column,
            score_column=score_column,
            quota_per_group=top_n_per_target,
            n_bins=sel.get("score_bins", 10),
            ascending=ascending,
            seed=seed,
        )
    elif top_n_per_target and group_column:
        logger.warning(
            "top_n_per_target is set but no score_column is configured: falling back "
            "to row-order head() per group. This is NOT stratified by score -- its "
            "output depends on the raw file's row order, not on sampling."
        )
        df = df.groupby(group_column, group_keys=False).head(top_n_per_target)
    else:
        if score_column:
            df = df.sort_values(score_column, ascending=ascending)
        if sel.get("top_n"):
            df = df.head(sel["top_n"])

    if sel.get("random_sample_n"):
        df = df.sample(n=sel["random_sample_n"], random_state=seed)

    columns_to_keep = config.get("columns_to_keep")
    if columns_to_keep:
        df = df[columns_to_keep]

    output_dir = REPO_ROOT / config["output"]["subset_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config["output"]["subset_filename"]

    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)

    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_corpus_subset()
