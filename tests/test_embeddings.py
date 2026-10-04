"""Tests for src/embeddings.py: ECFP4 fingerprinting and the corpus-keyed
embeddings build.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embeddings import FINGERPRINT_N_BITS, build_embeddings, ecfp4, ecfp6  # noqa: E402


def test_ecfp4_shape_and_dtype():
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(=O)O"]]
    fps = ecfp4(mols)
    assert fps.shape == (3, FINGERPRINT_N_BITS)
    assert fps.dtype == np.uint8
    assert set(np.unique(fps).tolist()) <= {0, 1}


def test_ecfp4_identical_molecules_give_identical_fingerprints():
    mols = [Chem.MolFromSmiles("c1ccccc1"), Chem.MolFromSmiles("c1ccccc1")]
    fps = ecfp4(mols)
    np.testing.assert_array_equal(fps[0], fps[1])


def test_ecfp4_rejects_unparseable_mol():
    with pytest.raises(ValueError):
        ecfp4([None])


def test_ecfp6_shape_and_dtype():
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(=O)O"]]
    fps = ecfp6(mols)
    assert fps.shape == (3, FINGERPRINT_N_BITS)
    assert fps.dtype == np.uint8
    assert set(np.unique(fps).tolist()) <= {0, 1}


def test_ecfp6_differs_from_ecfp4_for_same_molecule():
    mols = [Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")]  # aspirin: enough structure to differ at radius 2 vs 3
    assert not np.array_equal(ecfp4(mols)[0], ecfp6(mols)[0])


def test_build_embeddings_shape_and_id_alignment_against_real_corpus():
    from corpus import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config

    config = load_config(DEFAULT_CONFIG_PATH)
    subset_path = REPO_ROOT / config["output"]["subset_dir"] / config["output"]["subset_filename"]
    corpus_df = pd.read_parquet(subset_path)
    expected_mol_ids = set(corpus_df["mol_id"].unique())

    output_paths = build_embeddings(DEFAULT_CONFIG_PATH)
    assert set(output_paths) == {"ecfp4", "ecfp6"}

    for name, output_path in output_paths.items():
        emb_df = pd.read_parquet(output_path)

        # One embedding row per unique molecule -- embeddings are
        # structure-only, not per molecule-target row, so this is NOT
        # len(corpus_df).
        assert len(emb_df) == len(expected_mol_ids), name
        assert emb_df["mol_id"].is_unique, name
        assert set(emb_df["mol_id"]) == expected_mol_ids, name

        feature_cols = [c for c in emb_df.columns if c != "mol_id"]
        assert len(feature_cols) == FINGERPRINT_N_BITS, name
