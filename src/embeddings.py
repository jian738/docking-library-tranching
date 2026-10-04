"""Molecule embeddings for the corpus subset.

A representation is any function Sequence[Chem.Mol] -> np.ndarray of shape
(n_mols, dim); dim and dtype are its own business (a fingerprint's bits, a
descriptor vector's floats, a learned embedding's floats -- whatever it
returns is written as-is). To add a new one, write the function and add it
to _REPRESENTATIONS; build_embeddings() picks up the rest automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from corpus import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config

Representation = Callable[[Sequence[Chem.Mol]], np.ndarray]

FINGERPRINT_N_BITS = 2048

# ECFPn names the diameter, RDKit's Morgan generator takes the radius (n/2).
_ECFP4_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_N_BITS)
_ECFP6_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=FINGERPRINT_N_BITS)


def _morgan_fingerprints(mols: Sequence[Chem.Mol], generator, fn_name: str) -> np.ndarray:
    mols = list(mols)
    fps = np.zeros((len(mols), FINGERPRINT_N_BITS), dtype=np.uint8)
    for i, mol in enumerate(mols):
        if mol is None:
            raise ValueError(f"{fn_name}() received a None Mol at index {i}")
        fps[i] = generator.GetFingerprintAsNumPy(mol)
    return fps


def ecfp4(mols: Sequence[Chem.Mol]) -> np.ndarray:
    """Compute ECFP4 (Morgan, radius=2) fingerprints for a sequence of RDKit
    Mol objects.

    Returns a dense uint8 array of shape (len(mols), FINGERPRINT_N_BITS).
    """
    return _morgan_fingerprints(mols, _ECFP4_GENERATOR, "ecfp4")


def ecfp6(mols: Sequence[Chem.Mol]) -> np.ndarray:
    """Compute ECFP6 (Morgan, radius=3) fingerprints for a sequence of RDKit
    Mol objects.

    Returns a dense uint8 array of shape (len(mols), FINGERPRINT_N_BITS).
    """
    return _morgan_fingerprints(mols, _ECFP6_GENERATOR, "ecfp6")


# name -> representation. Any function with the Representation signature can be
# registered here (descriptors, MACCS keys, learned embeddings, ...).
_REPRESENTATIONS: dict[str, Representation] = {
    "ecfp4": ecfp4,
    "ecfp6": ecfp6,
}


def build_embeddings(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Path]:
    """Compute every registered representation for the unique molecules in
    the frozen corpus subset and write each to
    data/embeddings/<name>.parquet, keyed by mol_id.

    Embeddings depend only on molecular structure, not on which target a
    molecule was docked against, so the corpus subset (one row per
    molecule-target pair) is deduplicated down to one row per mol_id first.

    Returns {representation_name: path_to_written_parquet}.
    """
    config = load_config(config_path)

    subset_dir = REPO_ROOT / config["output"]["subset_dir"]
    subset_path = subset_dir / config["output"]["subset_filename"]
    df = pd.read_parquet(subset_path)

    molecules = (
        df[["mol_id", "smiles"]]
        .drop_duplicates(subset="mol_id")
        .sort_values("mol_id")
        .reset_index(drop=True)
    )

    mols = [Chem.MolFromSmiles(s) for s in molecules["smiles"]]
    invalid_ids = [mol_id for mol_id, mol in zip(molecules["mol_id"], mols) if mol is None]
    if invalid_ids:
        raise ValueError(f"Could not parse SMILES for mol_id(s): {invalid_ids}")

    output_dir = REPO_ROOT / "data" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}
    for name, represent in _REPRESENTATIONS.items():
        vectors = represent(mols)
        out = pd.DataFrame(vectors, columns=[f"dim_{i}" for i in range(vectors.shape[1])])
        out.insert(0, "mol_id", molecules["mol_id"].values)
        output_path = output_dir / f"{name}.parquet"
        out.to_parquet(output_path, index=False)
        output_paths[name] = output_path

    return output_paths


if __name__ == "__main__":
    build_embeddings()
