"""Generate a synthetic "molecules x targets" dataset for practicing a docking-library
analysis pipeline before real docking data arrives.

The sandbox exercises two tasks:

  Task 1 (tranching).  Each molecule carries a latent `binding_cluster` label that
    drives its binding-relevant features. Good clustering of molecules should
    recover tranches whose members share docking behaviour, so intra-tranche
    variance and top-tranche enrichment become meaningful. A tunable
    `interaction_weight` controls how much of the score variance comes from the
    molecule-target INTERACTION versus molecule-intrinsic "stickiness":
      - interaction_weight ~ 0  -> binding is molecule-intrinsic; a TARGET-AGNOSTIC
        tranching scheme should generalise across targets.
      - interaction_weight large -> binding is target-specific; target-agnostic
        tranching should FAIL and TARGET-DEPENDENT tranching should win.
    Sweeping this one knob gives you sandboxes where you already KNOW which
    strategy ought to win, so you can check that the tranching metrics detect it.

  Task 2 (score inference).  A known nonlinear function of a few feature columns,
    plus a per-scaffold offset, plus power-law-imbalanced scaffold sizes, so a
    scaffold-style split genuinely differs from a random split and exposes
    leakage. `scaffold_id` (leakage/generalisation axis) is kept DISTINCT from
    `binding_cluster` (tranche-quality axis) on purpose: conflating them would let
    a model cheat by using scaffold as a proxy for binding.

The score model is a low-rank bilinear ("coupling Hamiltonian") form:

    s(m, t) = a(m) + b(t) + interaction_weight * <phi(m), psi(t)> + f_nl(m) + noise

where phi(m) is the molecule's latent binding vector (driven by its binding
cluster), psi(t) is the target's latent pocket-preference vector, a(m) is
molecule-intrinsic stickiness, b(t) a target baseline, and f_nl is the known
nonlinear term used for the inference task. Scores follow the docking convention
(more negative = better) and are heavy-tailed so a meaningful "top 1%" exists.

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --n-molecules 5000 --n-targets 8 --seed 7
    python scripts/generate_synthetic.py --interaction-weight 0.0   # agnostic wins
    python scripts/generate_synthetic.py --interaction-weight 3.0   # dependent wins
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

_SMILES_TEMPLATES = [
    "CCO", "c1ccccc1", "Cc1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1", "c1ccc2[nH]ccc2c1", "c1ccncc1", "c1cnc[nH]1",
    "c1ccc2ccccc2c1", "Oc1ccccc1", "Nc1ccccc1", "CC(C)=O", "CC(=O)O",
    "CCN(CC)CC", "c1ccsc1", "c1ccoc1", "C1CCCCC1", "C1CCNCC1",
    "C1COCCN1", "CC(N)C(=O)O", "NC(CS)C(=O)O", "NC(Cc1ccccc1)C(=O)O",
    "c1ccc(cc1)-c1ccccc1", "CCOC(=O)c1ccccc1", "CC(C)(C)c1ccccc1",
    "Clc1ccccc1", "Fc1ccccc1", "Brc1ccccc1", "COc1ccccc1",
    "CC(=O)c1ccccc1", "N#Cc1ccccc1", "O=C(O)c1ccccc1",
    "Nc1ccc(cc1)S(=O)(=O)N", "CCCCCC", "CCCCCCCC", "c1ccc2ncccc2c1",
]


def _valid_smiles_pool() -> list[str]:
    invalid = [s for s in _SMILES_TEMPLATES if Chem.MolFromSmiles(s) is None]
    if invalid:
        raise ValueError(f"Invalid SMILES in placeholder template pool: {invalid}")
    return _SMILES_TEMPLATES


_F_SQUARE = 0
_F_INTERACT_A, _F_INTERACT_B = 1, 2
_F_SIN = 3
_F_LOG = 4

_COEFS = {
    "square": 1.5,
    "interact": 2.0,
    "sin": 3.0,
    "log": 1.0,
    "intercept": 0.5,
}


def nonlinear_term(X: np.ndarray) -> np.ndarray:
    """Known nonlinear function of a few feature columns (molecule-intrinsic).
    This is the signal the score-inference regressors must learn. It depends only on the
    molecule's observed features, so it is fully learnable from X alone.
    """
    return (
        _COEFS["intercept"]
        + _COEFS["square"] * X[:, _F_SQUARE] ** 2
        + _COEFS["interact"] * X[:, _F_INTERACT_A] * X[:, _F_INTERACT_B]
        + _COEFS["sin"] * np.sin(2.0 * X[:, _F_SIN])
        + _COEFS["log"] * np.log1p(np.abs(X[:, _F_LOG]))
    )


def make_group_sizes(n_items: int, n_groups: int, rng: np.random.Generator,
                     concentration: float = 0.3) -> np.ndarray:
    """Power-law-imbalanced group sizes (a few large groups, many small ones).

    Used for both scaffolds and binding clusters. A low Dirichlet concentration
    yields the heavy imbalance seen in real chemical series.
    """
    weights = rng.dirichlet(np.full(n_groups, concentration))
    sizes = np.floor(weights * n_items).astype(int)
    sizes[sizes < 1] = 1
    diff = n_items - sizes.sum()
    idx_order = np.argsort(-sizes)
    i = 0
    while diff != 0:
        j = idx_order[i % n_groups]
        if diff > 0:
            sizes[j] += 1
            diff -= 1
        elif sizes[j] > 1:
            sizes[j] -= 1
            diff += 1
        i += 1
    return sizes


def generate(
    n_molecules: int,
    n_features: int,
    n_scaffolds: int,
    n_targets: int,
    n_binding_clusters: int,
    latent_dim: int,
    interaction_weight: float,
    noise_std: float,
    scaffold_offset_std: float,
    scaffold_spread: float,
    stickiness_std: float,
    target_baseline_std: float,
    heavy_tail_df: float,
    decoy_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)

    scaffold_sizes = make_group_sizes(n_molecules, n_scaffolds, rng)
    scaffold_ids = np.repeat(np.arange(n_scaffolds), scaffold_sizes)
    rng.shuffle(scaffold_ids)

    cluster_sizes = make_group_sizes(n_molecules, n_binding_clusters, rng)
    binding_clusters = np.repeat(np.arange(n_binding_clusters), cluster_sizes)
    rng.shuffle(binding_clusters)

    scaffold_centers = rng.normal(0.0, 1.0, size=(n_scaffolds, n_features))
    scaffold_target_offsets = rng.normal(0.0, scaffold_offset_std, size=n_scaffolds)
    X = scaffold_centers[scaffold_ids] + rng.normal(
        0.0, scaffold_spread, size=(n_molecules, n_features)
    )

    cluster_phi_centers = rng.normal(0.0, 1.0, size=(n_binding_clusters, latent_dim))
    phi = cluster_phi_centers[binding_clusters] + rng.normal(
        0.0, 0.25, size=(n_molecules, latent_dim)
    )

    n_leak = min(latent_dim, max(1, n_features // 4))
    X[:, -n_leak:] += phi[:, :n_leak]

    psi = rng.normal(0.0, 1.0, size=(n_targets, latent_dim))
    target_baselines = rng.normal(0.0, target_baseline_std, size=n_targets)

    stickiness = rng.normal(0.0, stickiness_std, size=n_molecules)

    nl = nonlinear_term(X)  
    interaction = interaction_weight * (phi @ psi.T) 

    base = (
        stickiness[:, None]
        + nl[:, None]
        + scaffold_target_offsets[scaffold_ids][:, None]
        + target_baselines[None, :]
    )
    raw = base + interaction

    heavy = rng.standard_t(df=heavy_tail_df, size=raw.shape) * noise_std
    score_matrix = -(raw + heavy)

    mol_index = np.repeat(np.arange(n_molecules), n_targets)
    tgt_index = np.tile(np.arange(n_targets), n_molecules)
    long = pd.DataFrame({
        "mol_id": [f"MOL_{i:05d}" for i in mol_index],
        "scaffold_id": [f"SCAFFOLD_{scaffold_ids[i]:04d}" for i in mol_index],
        "binding_cluster": [f"BCLUSTER_{binding_clusters[i]:03d}" for i in mol_index],
        "target_id": [f"TARGET_{t:02d}" for t in tgt_index],
        "docking_score": score_matrix[mol_index, tgt_index],
    })

    feat_cols = [f"feat_{i}" for i in range(n_features)]
    feat_df = pd.DataFrame(X[mol_index], columns=feat_cols)
    long = pd.concat([long.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    smiles_pool = np.array(_valid_smiles_pool())
    templates_per_cluster = min(2, len(smiles_pool))
    cluster_subpools = [
        rng.choice(smiles_pool, size=templates_per_cluster, replace=False)
        for _ in range(n_binding_clusters)
    ]
    molecule_smiles = np.array([
        rng.choice(cluster_subpools[binding_clusters[m]]) for m in range(n_molecules)
    ])
    long["smiles"] = molecule_smiles[mol_index]

    n_decoy = int(decoy_frac * len(long))
    if n_decoy > 0:
        decoy_rows = rng.choice(len(long), size=n_decoy, replace=False)
        long.loc[decoy_rows, feat_cols] = np.nan
        long.loc[decoy_rows, "is_decoy"] = True
    long["is_decoy"] = long.get("is_decoy", pd.Series(False, index=long.index)).fillna(False)

    intrinsic_var = float(np.var(base))
    interaction_var = float(np.var(interaction))
    frac_interaction = interaction_var / (intrinsic_var + interaction_var + 1e-12)

    metadata = {
        "seed": seed,
        "n_molecules": n_molecules,
        "n_targets": n_targets,
        "n_rows": int(len(long)),
        "n_features": n_features,
        "n_scaffolds": n_scaffolds,
        "n_binding_clusters": n_binding_clusters,
        "latent_dim": latent_dim,
        "interaction_weight": interaction_weight,
        "noise_std": noise_std,
        "heavy_tail_df": heavy_tail_df,
        "decoy_frac": decoy_frac,
        "smiles_pool_size": len(smiles_pool),
        "smiles_templates_per_cluster": templates_per_cluster,
        "smiles_note": (
            "Placeholder structures, NOT real chemistry: each binding_cluster "
            "draws from its own small sub-pool of templates, so structure "
            "correlates with binding_cluster (and therefore with docking "
            "behaviour via phi) but is NOT itself an input to the score model."
        ),
        "score_convention": "more_negative_is_better (docking-style)",
        "score_model": (
            "docking_score = -(stickiness(m) + f_nl(m) + scaffold_offset(m) "
            "+ target_baseline(t) + interaction_weight*<phi(m),psi(t)>) + heavy_tail_noise"
        ),
        "true_nonlinear_function": (
            "f_nl = intercept + square*feat_{sq}**2 + interact*feat_{ia}*feat_{ib} "
            "+ sin*sin(2*feat_{si}) + log*log1p(abs(feat_{lg}))"
        ).format(sq=_F_SQUARE, ia=_F_INTERACT_A, ib=_F_INTERACT_B, si=_F_SIN, lg=_F_LOG),
        "coefficients": _COEFS,
        "feature_indices_used": {
            "square": _F_SQUARE, "interact_a": _F_INTERACT_A,
            "interact_b": _F_INTERACT_B, "sin": _F_SIN, "log": _F_LOG,
        },
        "known_answer_diagnostics": {
            "intrinsic_variance": intrinsic_var,
            "interaction_variance": interaction_var,
            "fraction_variance_from_interaction": frac_interaction,
            "expected_winner": (
                "target_dependent" if frac_interaction > 0.5 else "target_agnostic"
            ),
            "note": (
                "If fraction_variance_from_interaction is high, target-agnostic "
                "tranching should generalise poorly and target-dependent should win. "
                "Use this to check the tranching metrics detect the right strategy."
            ),
        },
    }
    return long, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-molecules", type=int, default=3000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-scaffolds", type=int, default=150)
    parser.add_argument("--n-targets", type=int, default=6,
                        help="Number of protein targets (enables the tranching task).")
    parser.add_argument("--n-binding-clusters", type=int, default=40,
                        help="Latent binding groups that good tranching should recover.")
    parser.add_argument("--latent-dim", type=int, default=8,
                        help="Rank of the molecule/target latent binding vectors.")
    parser.add_argument("--interaction-weight", type=float, default=1.0,
                        help="0 -> target-agnostic wins; large -> target-dependent wins.")
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--scaffold-offset-std", type=float, default=2.0)
    parser.add_argument("--scaffold-spread", type=float, default=0.5)
    parser.add_argument("--stickiness-std", type=float, default=2.0,
                        help="Std of molecule-intrinsic binding propensity a(m).")
    parser.add_argument("--target-baseline-std", type=float, default=1.0)
    parser.add_argument("--heavy-tail-df", type=float, default=3.0,
                        help="Student-t dof for score noise; lower = heavier tail.")
    parser.add_argument("--decoy-frac", type=float, default=0.01,
                        help="Fraction of rows with NaN features (failed featurisation).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="data/raw/synthetic_molecules.csv")
    args = parser.parse_args()

    max_used_idx = max(_F_SQUARE, _F_INTERACT_A, _F_INTERACT_B, _F_SIN, _F_LOG)
    if args.n_features <= max_used_idx:
        raise ValueError(f"--n-features must be > {max_used_idx} for the true function")
    if args.latent_dim < 1:
        raise ValueError("--latent-dim must be >= 1")

    df, metadata = generate(
        n_molecules=args.n_molecules,
        n_features=args.n_features,
        n_scaffolds=args.n_scaffolds,
        n_targets=args.n_targets,
        n_binding_clusters=args.n_binding_clusters,
        latent_dim=args.latent_dim,
        interaction_weight=args.interaction_weight,
        noise_std=args.noise_std,
        scaffold_offset_std=args.scaffold_offset_std,
        scaffold_spread=args.scaffold_spread,
        stickiness_std=args.stickiness_std,
        target_baseline_std=args.target_baseline_std,
        heavy_tail_df=args.heavy_tail_df,
        decoy_frac=args.decoy_frac,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    diag = metadata["known_answer_diagnostics"]
    print(f"Wrote {len(df)} rows "
          f"({args.n_molecules} molecules x {args.n_targets} targets) to {output_path}")
    print(f"Scaffolds: {args.n_scaffolds} | binding clusters: {args.n_binding_clusters} "
          f"| latent_dim: {args.latent_dim}")
    print(f"Fraction of score variance from interaction: "
          f"{diag['fraction_variance_from_interaction']:.3f} "
          f"-> expected winner: {diag['expected_winner']}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()