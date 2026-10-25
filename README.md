# Sri Lanka Leptospirosis - Ensemble ML Research

Extra Trees (Extremely Randomized Trees) as the focal algorithm, with ten comparison classifiers using admission features from a Sri Lankan febrile-patient study. This repository contains the research code, modelling notebook, methodology notes, tests and the Streamlit demo under [demo-app/](demo-app/README.md). The demo includes a reviewed fitted model and aggregate or synthetic release reports. Source patient data, generated research outputs and presentations remain outside version control.

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

The deployed release should show Extra Trees, version `503bc9aeb6f8`, with saved reference example score approximately `0.0772` (Age=35, Fever=Yes (assumed label for code 1), all other categorical inputs Not recorded). The live form defaults to Male for sex and Yes for symptoms and exposures (source code 1), which is a different example. Symptom Yes/No labels are a provisional demo assumption pending codebook verification.

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

The updated assignment motivates **Extra Trees as the focal algorithm**, extending the originally supplied model list. The benchmark declares **38 configurations across 11 classifiers** on two feature representations and shared five-fold development splits. Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, AdaBoost, Gradient Boosting, LightGBM and CatBoost provide comparisons. Linear Regression remains reserved for continuous targets. No deep learning or stacking is used. The constant development-prevalence score is an evaluation reference rather than a candidate estimator.

[Extra Trees rationale](docs/EXTRA_TREES.md) explains random split thresholds, how the method differs from Random Forest, the declared 150-tree configuration and its limitations. Check actual lecture coverage before claiming the algorithm was not taught. The existing taught-model comparison remains useful evidence rather than being the assignment's focal contribution.

[Feature engineering](docs/FEATURE_ENGINEERING.md) explains each operation, reason, assumption and leakage boundary. Median imputation, missing indicators and encoders are fitted only on training folds. Four derived inputs are tested through development-only ablation.

Selection maximizes **mean development ROC AUC**. Exact ties use development average precision and deterministic ordering. The internal holdout does not select the model, features or threshold. Evaluation additionally reports average precision, accuracy, balanced accuracy, precision, recall, F1, Brier score, log loss and a confusion matrix at threshold 0.50. Global permutation importance and local Monte Carlo Shapley contributions explain the chosen predictor after selection.

The [current model card](docs/MODEL_CARD.md) describes the regenerated experiment. **Extra Trees with base features** has the highest mean development ROC AUC, **0.6054**, versus Random Forest **0.6028**. Its internal holdout ROC AUC is **0.6612**, average precision **0.5335**, and recall **0.2021** at 0.50. These close CV means do not establish statistical superiority. The historical holdout was already observed and is not independent external validation.

**Research limitations:** numerical source outcome meanings, measurement units and the 99 missing-value convention remain provisional. Do not equate source code 1 with a verified clinical diagnosis. Scores are uncalibrated. Low recall, cohort selection, hospital/time shifts, correlated predictors and selection optimism limit interpretation. Verify the source codebook and obtain independent validation before making stronger substantive claims.

## Extend the research

Change declared models/grids in `src/models.py`, the admission feature contract in `src/data.py`, or explanations in `src/explain.py`. Keep learned preprocessing inside folds and declare focal methods and comparisons before evaluating the holdout. Choose features, hyperparameters and any operating threshold using development data; reserve an independent evaluation set for final assessment. Use a separate output directory for each new experiment and record source version, assumptions and changes alongside the generated manifest.

## Checks without patient data

```sh
python -m unittest discover -s tests -v
python scripts/check_notebook.py --validate-only
python -m compileall -q src train.py
```

Synthetic-data tests cover input validation, derived-feature handling, preprocessing isolation, the focal Extra Trees comparison scope, development-only ranking and explanation reconstruction. Three checks of fitted outputs run after local training and otherwise report as skipped. GitHub Actions runs the data-free checks; it does not fetch or publish the dataset. Notebook execution does not save cell outputs unless `--save-outputs` is explicitly requested; review saved outputs before committing; include only aggregate tables, plots and synthetic examples. The current notebook refreshes these outputs from the local run.

`.gitignore` excludes source data formats, generated research outputs, archives and local environments. Narrow exceptions and the app allowlist include only the reviewed model and aggregate or synthetic release reports under `demo-app/`. Review staged files before publishing, especially when adding new data formats or notebooks.
