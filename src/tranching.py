"""Target-agnostic tranching benchmark: k-means on molecule embeddings, then
per-target intra-tranche docking-score variance and top-1% enrichment.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from corpus import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config


def assign_tranches(embeddings_df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """K-means cluster molecules by their embedding vectors into k tranches.

    Clustering is target-agnostic: it uses only the embedding columns, so
    the same tranche assignment is reused across all targets.

    Returns a DataFrame with columns [mol_id, tranche_id].
    """
    feature_cols = [c for c in embeddings_df.columns if c != "mol_id"]
    X = embeddings_df[feature_cols].to_numpy()
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    return pd.DataFrame({"mol_id": embeddings_df["mol_id"].values, "tranche_id": labels})


def _select_top_tranches(tranche_stats: pd.DataFrame, n_top: int) -> set:
    """Select the fewest best-mean-score tranches whose cumulative molecule
    count is >= n_top -- a selection budget comparable to picking the top-N
    molecules individually. `tranche_stats` must already be sorted
    best-score-first.
    """
    selected = set()
    cumulative = 0
    for tranche_id, count in zip(tranche_stats["tranche_id"], tranche_stats["count"]):
        selected.add(tranche_id)
        cumulative += count
        if cumulative >= n_top:
            break
    return selected


def compute_target_metrics(target_df: pd.DataFrame, top_pct: float = 0.01) -> dict:
    """Compute intra-tranche docking-score variance and top-N% enrichment for
    one target's rows. `target_df` must already have a `tranche_id` column
    and be restricted to a single target.

    Returns a dict with:
      - intra_tranche_variance: size-weighted (pooled) within-tranche
        variance of docking_score.
      - top1pct_enrichment: fraction of the target's true top-top_pct
        molecules (by docking_score) that fall inside the tranche(s) with
        the best mean docking_score.
      - top1pct_enrichment_random_baseline: the fraction expected under
        random tranche assignment of the same selection size, i.e. the
        threshold enrichment must clear to beat chance.
    """
    n = len(target_df)
    n_top = max(1, math.ceil(top_pct * n))

    grouped = target_df.groupby("tranche_id")["docking_score"]
    tranche_var = grouped.var(ddof=0).fillna(0.0)
    tranche_count = grouped.size()
    intra_tranche_variance = float((tranche_var * tranche_count).sum() / tranche_count.sum())

    tranche_mean = grouped.mean().sort_values(ascending=True)  # ascending score = best first
    tranche_stats = pd.DataFrame({
        "tranche_id": tranche_mean.index,
        "count": tranche_count.loc[tranche_mean.index].values,
    })

    top_mol_ids = set(target_df.nsmallest(n_top, "docking_score")["mol_id"])

    selected_tranches = _select_top_tranches(tranche_stats, n_top)
    selected_rows = target_df[target_df["tranche_id"].isin(selected_tranches)]
    selected_mol_ids = set(selected_rows["mol_id"])

    hits = len(top_mol_ids & selected_mol_ids)
    enrichment = hits / n_top
    random_baseline = len(selected_rows) / n

    ratio = enrichment / random_baseline

    return {
        "intra_tranche_variance": intra_tranche_variance,
        "top1pct_enrichment": enrichment,
        "top1pct_enrichment_random_baseline": random_baseline,
        "top1pct_ratio": ratio,
    }


def run_benchmark(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Run the tranching benchmark across every target in the frozen corpus
    subset and write results/benchmark_results.parquet in long format
    (representation, task, target, metric, value).

    Returns the path to the written parquet file.
    """
    config = load_config(config_path)
    seed = config.get("seed", 0)
    tconf = config["tranching"]

    subset_dir = REPO_ROOT / config["output"]["subset_dir"]
    corpus_df = pd.read_parquet(subset_dir / config["output"]["subset_filename"])
    embeddings_df = pd.read_parquet(REPO_ROOT / "data" / "embeddings" / "ecfp4.parquet")

    k = max(1, round(len(embeddings_df) / tconf["target_mean_tranche_size"]))
    tranches = assign_tranches(embeddings_df, k=k, seed=seed)
    merged = corpus_df.merge(tranches, on="mol_id", how="inner")

    rows = []
    for target_id, target_df in merged.groupby("target_id"):
        metrics = compute_target_metrics(target_df, top_pct=tconf["top_pct"])
        for metric_name, value in metrics.items():
            rows.append({
                "representation": tconf["representation"],
                "task": tconf["task"],
                "target": target_id,
                "metric": metric_name,
                "value": value,
            })

    results = pd.DataFrame(rows)

    output_path = REPO_ROOT / tconf["results_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_path, index=False)

    return output_path


if __name__ == "__main__":
    run_benchmark()
