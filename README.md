# Sri Lanka Leptospirosis - Ensemble ML Research

Reproducible comparison of the approved classification models using admission features from a Sri Lankan febrile-patient study. This repository contains the research code, modelling notebook, methodology notes, tests and the Streamlit demo under [demo-app/](demo-app/README.md). The demo includes a reviewed fitted model and aggregate or synthetic release reports. Source patient data, generated research outputs and presentations remain outside version control.

## Deploy the demo from this repository

The app lives at **[demo-app/app.py](demo-app/app.py)**. Commit and push the repository, then use these Streamlit Community Cloud settings:

| Setting | Value |
|---|---|
| Repository | `Pasansemage/sri-lanka-leptospirosis-ensemble-ml` |
| Branch | `main` |
| Main file path | `demo-app/app.py` |
| Python version | `3.12` |
| Secrets | None |

Keep `.streamlit/config.toml` at the repository root. App dependencies are in `demo-app/requirements.txt`, which Community Cloud reads before the root research requirements. See the [demo deployment guide](demo-app/README.md).

For local inference, from this repository's root run:

```sh
python -m pip install -r demo-app/requirements.txt
python -m streamlit run demo-app/app.py
```

The deployed release should show Random Forest, version `6fe8ec8491c4`, with default synthetic example score approximately `0.0726`.

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

Training writes local generated files under ignored `outputs/`. The reviewed release in `demo-app/` is packaged separately for serving:

- `outputs/reports/`: all candidate/fold results, feature ablation, frozen selection, holdout metrics, the current model card, plots, permutation importance and synthetic explanations.
- `outputs/artifacts/`: fitted estimator, synthetic reference and manifest containing source, package, dataset and output checksums.

Use `--output /another/local/directory` to change the destination; set `OUTPUT_DIR` in the notebook to the same directory. After a reporting interruption, `python train.py --data /path/to/file.csv --finalize` verifies the existing source/model metrics and rebuilds explanations without rerunning selection. For an ordinary new experiment, use a full training run.

## Notebook setup in VS Code

Select the repository's `.venv/Scripts/python.exe` with **Select Kernel**, then restart the kernel and choose **Run All**. The setup cell finds the project from `notebooks/`, the repository root, `ML Assignment`, or the broader `Applied ML` workspace. It prints the resolved project folder and Python executable so you can check which environment is in use.

If the kernel starts somewhere unrelated, set `PROJECT_DIR` in the first code cell to your research repository path. `RUN_TRAINING = False` reads existing local `outputs/`; for a fresh clone, first run the training command above using your separately obtained dataset.

## Methodology and interpretation

The benchmark declares **36 recipes across 10 classification methods**, two feature representations and shared five-fold development splits: **Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, AdaBoost, Gradient Boosting, LightGBM and CatBoost**. The approved project list also includes **Linear Regression**, reserved for continuous-target tasks; it is not fitted to the current binary target. No other algorithms or stacking combinations enter selection. A constant development-prevalence score is retained only as an evaluation reference, not as a candidate estimator. SVM uses an RBF kernel and its built-in probability estimation; numeric scaling is fitted within each development fold.

[Feature engineering](docs/FEATURE_ENGINEERING.md) explains each operation, reason, assumption and leakage boundary. Median imputation, missing indicators and encoders are fitted only on training folds. Four derived inputs are tested through development-only ablation.

Selection maximizes **mean development ROC AUC**. Exact ties use development average precision and deterministic ordering. The internal holdout does not select the model, features or threshold. Evaluation additionally reports average precision, accuracy, balanced accuracy, precision, recall, F1, Brier score, log loss and a confusion matrix at threshold 0.50. Global permutation importance and local Monte Carlo Shapley contributions explain the chosen predictor after selection.

The [historical model card](docs/MODEL_CARD.md) records a superseded experiment outside the current approved model scope: Extra Trees with base features, development ROC AUC **0.6054**, internal holdout ROC AUC **0.6612**, and recall **0.2021** at 0.50. These are archived reference results, not results of the restricted benchmark. Retrain to obtain the current winner and metrics. The historical holdout was used in earlier project work; it is internal evaluation, not independent external validation.

**Research limitations:** numerical source outcome meanings, measurement units and the 99 missing-value convention remain provisional. Do not equate source code 1 with a verified clinical diagnosis. Scores are uncalibrated. Low recall, cohort selection, hospital/time shifts, correlated predictors and selection optimism limit interpretation. Verify the source codebook and obtain independent validation before making stronger substantive claims.

## Extend the research

Change declared models/grids in `src/models.py`, the admission feature contract in `src/data.py`, or explanations in `src/explain.py`. Keep learned preprocessing inside folds and keep model additions within the approved list. Choose features, hyperparameters and any operating threshold using development data; reserve an independent evaluation set for final assessment. Use a separate output directory for each new experiment and record source version, assumptions and changes alongside the generated manifest.

## Checks without patient data

```sh
python -m unittest discover -s tests -v
python scripts/check_notebook.py --validate-only
python -m compileall -q src train.py
```

Synthetic-data tests cover input validation, derived-feature handling, preprocessing isolation, the approved model scope, development-only ranking and explanation reconstruction. Three checks of fitted outputs run after local training and otherwise report as skipped. GitHub Actions runs the data-free checks; it does not fetch or publish the dataset. Notebook execution does not save cell outputs unless `--save-outputs` is explicitly requested; clear all outputs before committing.

`.gitignore` excludes source data formats, generated research outputs, archives and local environments. Narrow exceptions and the app allowlist include only the reviewed model and aggregate or synthetic release reports under `demo-app/`. Review staged files before publishing, especially when adding new data formats or notebooks.
