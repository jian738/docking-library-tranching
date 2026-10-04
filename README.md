# Target-Agnostic Tranching of a Docking Library

A tested pipeline for one question in ultra-large virtual screening: if you cluster molecules by structure alone, do the clusters ("tranches") group molecules with similar docking behaviour across protein targets? Built on synthetic data with a known ground truth, ahead of real docking data.

## Approach
1. **Synthetic data** (`scripts/generate_synthetic.py`): molecules × targets with a low-rank bilinear score model,
   s(m, t) = a(m) + b(t) + w·⟨φ(m), ψ(t)⟩ + f_nl(m) + heavy-tailed noise.
   The interaction weight w decides whether binding is mostly molecule-intrinsic or target-specific, so you know in advance which tranching strategy should win. Scaffold labels (for leakage) are kept separate from binding clusters (for tranche quality).
2. **Corpus subset** (`src/corpus.py`): keeps up to N rows per target, sampled evenly across that target's own score deciles, so the strong-binder tail is always represented. Seeding is deterministic and independent of iteration order.
3. **Embeddings** (`src/embeddings.py`): ECFP4 and ECFP6 fingerprints with RDKit, through a registry that accepts any representation.
4. **Tranching benchmark** (`src/tranching.py`): k-means on the embeddings (k = n_molecules / 50), then for each target the intra-tranche score variance and the top-1% enrichment of the best tranches compared with a random baseline.

## Results (default config, seed 42)
- The generator attributes 24.1% of score variance to molecule–target interaction, so the target-agnostic strategy is the expected winner.
- Top-1% enrichment is 14–21× the random baseline for 3 of 6 targets, and 0 for the other 3.

## Running
```bash
conda env create -f environment.yml && conda activate compbio-practice
python scripts/generate_synthetic.py
python scripts/01_build_corpus.py
python scripts/02_generate_embeddings.py
python scripts/03_tranching_benchmark.py
pytest
```
Generate the data before running `pytest`: two tests read `data/raw/synthetic_molecules.csv`.

## Known limitations
- The placeholder SMILES pool has only 38 structures, so many molecules share identical fingerprints, and k-means finds fewer distinct clusters than requested (28 of 31).
- All data is synthetic. Scores do not come from the molecules' actual chemistry.

## Acknowledgments
Based off of general approaches of the Arthanari Lab, but implemented independently.
Developed with AI coding assistance (Claude).
