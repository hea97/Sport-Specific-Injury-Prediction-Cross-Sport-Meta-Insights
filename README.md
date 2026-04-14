# Sport-Specific Injury Prediction Cross-Sport Meta Insights

This repository studies recall-first injury decision support across three sports datasets:

- `NBA`: injury report text and calendar context
- `Football`: player injury impact records with pre-injury match context
- `Multimodal`: wearable-like physiological and biomechanical sensor data

The project intent is unchanged: compare multiple model families on each dataset, prioritize recall for missed-injury risk reduction, and keep the original script entrypoints runnable.

## What Changed

The codebase has been refactored for reproducibility, maintainability, and more defensible analysis:

- deterministic feature engineering replaced prior random `Load_Score` generation
- data loading and schema checks now live under `src/data/`
- feature engineering now lives under `src/features/`
- CV-safe model training and standardized metrics now live under `src/train/`
- explainability placeholders now live under `src/xai/`
- dataset-specific experiment settings now live under `configs/*.yaml`
- insight plots and `results/MODEL_COMPARISON.md` are regenerated from standardized artifacts instead of hard-coded values

## Repository Layout

```text
configs/                  dataset-specific YAML configs
data/                     raw CSV datasets
insights/                 generated plots and visualization entrypoint
notebooks/                backward-compatible runnable scripts
results/                  standardized result tables and generated reports
src/data/                 loaders and schema validation
src/features/             deterministic feature builders and normalization helpers
src/train/                config parsing, pipelines, metrics, reporting, CLI
src/xai/                  SHAP/PDP placeholders
tests/                    smoke tests
```

## Dataset Notes

Each config documents the dataset path, label rule, feature list, model settings, imbalance handling, and seed.

| Dataset | Raw file | Label rule | Current intent |
| --- | --- | --- | --- |
| NBA | `data/injuries_2010-2020.csv` | keyword-derived target from `Notes` | injury-flag proxy from injury report text |
| Football | `data/player_injuries_impact.csv` | `Days_Out >= 30` after date cleaning | serious injury duration proxy |
| Multimodal | `data/sports_multimodal_data.csv` | existing `injury_risk` column | binary risk classification |

## Reproducible Runs

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run a single dataset with the new entrypoint:

```bash
python -m src.train.run --config configs/nba.yaml
python -m src.train.run --config configs/football.yaml
python -m src.train.run --config configs/multimodal.yaml
```

Legacy notebook-style commands still work and now call the same pipeline:

```bash
python notebooks/NBA.py
python notebooks/Football.py
python notebooks/Multimodal.py
```

Regenerate plots and comparison markdown from the latest standardized artifacts:

```bash
python insights/visualization.py
```

Run smoke tests:

```bash
python -m unittest discover -s tests -v
```

## Standardized Outputs

Each run writes:

- `results/<dataset>_results.csv`: per-model metrics with recall, precision, F1, PR-AUC, threshold, and confusion-matrix counts
- `results/artifacts/<dataset>/models/`: serialized model artifacts
- `results/artifacts/<dataset>/predictions/`: per-model prediction tables for plotting and audit
- `results/artifacts/<dataset>/run_metadata.json`: run config, notes, feature list, and label balance
- `results/MODEL_COMPARISON.md`: latest generated cross-dataset comparison
- `insights/*.png`: regenerated visual outputs from latest artifacts

Tree-based models are saved with `joblib`. The PyTorch MLP is saved with `torch.save`.

## Analytical Rigor Notes

The refactor intentionally addresses several legacy issues:

- prior Football and Multimodal narrative metrics in `README.md`, `results/MODEL_COMPARISON.md`, and `insights/visualization.py` did not match the CSV artifacts in `results/`
- prior NBA and Football scripts used `np.random.uniform(...)` to create `Load_Score`, which made reruns non-deterministic
- prior visualization code hard-coded recall values instead of reading result artifacts
- the raw Football CSV on disk is much larger than the old narrative claimed, and contains at least one invalid negative injury-duration row that is now dropped during feature building

These limitations still matter:

- NBA labels remain proxy labels derived from injury-report text because the source file does not contain a cleaner binary target
- Football remains a severity-style proxy task because the available file records injury events rather than full injury-free exposure history
- SHAP and partial dependence outputs are stubbed as placeholders unless those capabilities are implemented later

## Recommended Workflow

1. Run one or more dataset configs with `python -m src.train.run --config ...`
2. Inspect `results/<dataset>_results.csv` and `results/artifacts/<dataset>/run_metadata.json`
3. Regenerate plots with `python insights/visualization.py` if needed
4. Review `results/MODEL_COMPARISON.md` for the current cross-dataset summary

## Reproducibility Defaults

- global seed is controlled per config
- train/test splitting is stratified
- SMOTE is applied inside an `imblearn` pipeline so resampling stays inside the training path
- thresholding is centralized and configurable
- optional stratified cross-validation is available in config

## Next Extensions

- implement full SHAP and PDP generation under `src/xai/`
- add calibration-specific threshold selection if stricter recall targets are needed
- add richer domain features where higher-fidelity workload data becomes available
