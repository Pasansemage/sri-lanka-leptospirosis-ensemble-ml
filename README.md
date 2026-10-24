# Sri Lanka Leptospirosis - Ensemble ML Research

Reproducible comparison of bagging, boosting and stacking using admission features from a Sri Lankan febrile-patient study. This repository contains the research code, an output-free modelling notebook, methodology notes and tests. Patient data, fitted models, generated results, presentations and the demo application are not distributed here.

## Data source and access

**Source:** Suneth Agampodi (2022), *Demographic, exposure, clinical, biochemical and diagnostic data of febrile patients recruited for the largest field study on leptospirosis in Sri Lanka*, Mendeley Data, **version 4**, published 31 May 2022.

- Dataset and download: [Mendeley Data v4](https://data.mendeley.com/datasets/zmxmp42g9p/4)
- Dataset DOI: [10.17632/zmxmp42g9p.4](https://doi.org/10.17632/zmxmp42g9p.4)
- Associated data paper: [10.1016/j.dib.2022.108378](https://doi.org/10.1016/j.dib.2022.108378)
- Publisher-listed data licence: CC BY 4.0. Cite the original dataset and follow its terms when conducting your research.

**No patient data is committed or redistributed.** Obtain the dataset directly from the source and keep your copy outside the repository. Supply its path with `--data`; there is no automatic download or upload. An optional local `data/` folder is also ignored by Git.

The recorded experiment used a CSV named `Leptospirosis clinical data.csv` with SHA-256 `8f15e42cf808f6c342b868995633eedac86a7648cc73f553c852191424e7e880`. This identifies the local experiment input; its equality to a publisher download has not been independently verified. A different file/version can produce different results. Required columns are `Serial`, `Final` and the predictors explicitly listed in [src/data.py](src/data.py).

## Reproduce the experiment

Use **Python 3.12**. Install the pinned research dependencies in an isolated environment:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python train.py --data "/path/to/Leptospirosis clinical data.csv"
python -m unittest discover -s tests -v
python scripts/check_notebook.py
```

Open [notebooks/modelling.ipynb](notebooks/modelling.ipynb) with the same environment. First run the training command above, or set `DATA_PATH` to your local CSV and `RUN_TRAINING = True` in the notebook. Training is explicit; no source data or fitted model is needed to inspect the notebook or run the synthetic-data checks.

Training writes only local generated files under ignored `outputs/`:

- `outputs/reports/`: all candidate/fold results, feature ablation, frozen selection, holdout metrics, plots, permutation importance and synthetic explanations.
- `outputs/artifacts/`: fitted estimator, synthetic reference and manifest containing source, package, dataset and output checksums.

Use `--output /another/local/directory` to change the destination; set `OUTPUT_DIR` in the notebook to the same directory. After a reporting interruption, `python train.py --data /path/to/file.csv --finalize` verifies the existing source/model metrics and rebuilds explanations without rerunning selection. For an ordinary new experiment, use a full training run.

## Methodology and interpretation

The benchmark declares **36 recipes across 13 methods**, two feature representations and shared five-fold development splits. It includes bagged trees, random forest, Extra Trees, six boosting methods, two stacking recipes, logistic regression and a prior baseline. Stacking fits complete preprocessing pipelines inside three inner folds.

[Feature engineering](docs/FEATURE_ENGINEERING.md) explains each operation, reason, assumption and leakage boundary. Median imputation, missing indicators and encoders are fitted only on training folds. Four derived inputs are tested through development-only ablation.

Selection maximizes **mean development ROC AUC**. Exact ties use development average precision and deterministic ordering. The internal holdout does not select the model, features or threshold. Evaluation additionally reports average precision, accuracy, balanced accuracy, precision, recall, F1, Brier score, log loss and a confusion matrix at threshold 0.50. Global permutation importance and local Monte Carlo Shapley contributions explain the chosen predictor after selection.

The [historical model card](docs/MODEL_CARD.md) records the original experiment: Extra Trees with base features, development ROC AUC **0.6054**, internal holdout ROC AUC **0.6612**, and recall **0.2021** at 0.50. These are reference results, not guarantees for another run. The historical holdout was used in earlier project work; it is internal evaluation, not independent external validation.

**Research limitations:** numerical source outcome meanings, measurement units and the 99 missing-value convention remain provisional. Do not equate source code 1 with a verified clinical diagnosis. Scores are uncalibrated. Low recall, cohort selection, hospital/time shifts, correlated predictors and selection optimism limit interpretation. Verify the source codebook and obtain independent validation before making stronger substantive claims.

## Extend the research

Change declared models/grids in `src/models.py`, the admission feature contract in `src/data.py`, or explanations in `src/explain.py`. Keep learned preprocessing inside folds and stacking base pipelines. Choose features, hyperparameters and any operating threshold using development data; reserve an independent evaluation set for final assessment. Use a separate output directory for each new experiment and record source version, assumptions and changes alongside the generated manifest.

## Checks without patient data

```sh
python -m unittest discover -s tests -v
python scripts/check_notebook.py --validate-only
python -m compileall -q src train.py
```

Synthetic-data tests cover input validation, derived-feature handling, preprocessing isolation, stacking boundaries, development-only ranking and explanation reconstruction. Three checks of fitted outputs run after local training and otherwise report as skipped. GitHub Actions runs the data-free checks; it does not fetch or publish the dataset. Notebook execution does not save cell outputs unless `--save-outputs` is explicitly requested; clear all outputs before committing.

`.gitignore` excludes data formats, fitted models, generated output directories, archives and local environments. Review staged files before publishing, especially when adding new data formats or notebooks.
