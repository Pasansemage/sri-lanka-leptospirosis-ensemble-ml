# Sri Lanka Leptospirosis — Ensemble ML

A reproducible comparison of bagging, boosting and stacking on admission features from a Sri Lankan febrile-patient study, with an interactive Streamlit app and explanations of the selected model.

**Research demonstration only.** Source outcome codes and missing-value conventions are provisional. Scores are uncalibrated model outputs, not clinical risks. The existing project's historical holdout is reused for internal evaluation; this is not external validation.

## Project folder

The working project is `D:\Masters\Second Semester\Applied ML\sri-lanka-leptospirosis-ensemble-ml`. The earlier `ML Project` folder is preserved as a backup; make future changes here.

## Start here

- [`leptospirosis_project/modelling.ipynb`](leptospirosis_project/modelling.ipynb): feature reasoning, ensemble comparison, selection metrics, explainability and MLOps.
- [`reports/FEATURE_ENGINEERING.md`](reports/FEATURE_ENGINEERING.md): exact decisions, assumptions and ablation design.
- [`reports/MODEL_CARD.md`](reports/MODEL_CARD.md): measured results and limitations.
- [`PRESENTATION_SCRIPT.md`](PRESENTATION_SCRIPT.md): a spoken demonstration script using the actual results.
- [`DEPLOYMENT.md`](DEPLOYMENT.md): free public hosting and release/rollback instructions.
- `leptospirosis_project/analysis.ipynb`: earlier broader course analysis, retained with outputs cleared. Its original `results.zip` stays local.

## Results

The benchmark evaluates **36 model/feature/hyperparameter combinations** across **13 methods**, with the same five development folds. Methods include three bagging methods (bagged trees, random forest, Extra Trees), six boosting methods (AdaBoost, gradient boosting, histogram boosting, XGBoost, LightGBM, CatBoost), two stacking recipes, logistic regression and a prior baseline. Stacking uses three inner folds, with preprocessing inside each base pipeline.

**Extra Trees with the base features** wins by mean development ROC AUC **0.6054**, with fold SD **0.0500**. The engineered representation scores **0.6000** for Extra Trees, so the derived inputs are excluded from the deployed predictor. Exact AUC ties use development average precision and then deterministic ordering; test scores never choose the winner.

| Internal holdout metric | Selected model | Prior baseline |
|---|---:|---:|
| ROC AUC | 0.6612 | 0.5000 |
| Average precision | 0.5335 | 0.3884 |
| Accuracy | 0.6322 | 0.6116 |
| Balanced accuracy | 0.5538 | 0.5000 |
| Precision, code 1 | 0.5758 | 0.0000 |
| Recall, code 1 | 0.2021 | 0.0000 |
| F1, code 1 | 0.2992 | 0.0000 |
| Brier score (lower is better) | 0.2212 | 0.2376 |
| Log loss (lower is better) | 0.6342 | 0.6680 |

The fixed threshold is 0.50. Low code-1 recall is a material limitation. The fixed-model bootstrap ROC AUC interval is approximately **0.5902–0.7313**; it does not include model-selection uncertainty. The source has 1,734 records; 1,206 meet the provisional cohort contract, with 964 development and 242 holdout records.

## Run the app

Use Python **3.12** and the exact versions in `requirements.txt` to load the saved model. The app works without the source CSV and does not retrain on startup.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_dashboard.ps1
```

Or, from an activated environment:

```sh
python -m streamlit run app.py
```

The app includes immediate hypothetical-record inference, original-feature explanations, development comparisons, held-out metrics/confusion matrix, reliability plots, feature ablation and release provenance. Application code does not persist or log submitted inputs. Use hypothetical inputs.

## Reproduce training

Obtain the dataset separately and place `Leptospirosis clinical data.csv` at the repository root. It is ignored by Git. Its recorded SHA-256 is `8f15e42cf808f6c342b868995633eedac86a7648cc73f553c852191424e7e880`.

```sh
python train.py
python -m unittest discover -s tests -v
```

Training fits all candidates, freezes the development-selected recipe, evaluates only the winner and prior baseline on the holdout, computes explanations, and exports `artifacts/` and aggregate `reports/`. It can take several minutes. `python train.py --finalize` resumes report generation for an already evaluated local artifact after source/metric parity checks, without rerunning selection.

For the notebook, install `requirements-dev.txt` and select the same environment. Its saved-results path works without the CSV; retraining is explicit and disabled by default.

## Release checks

```sh
python -m unittest discover -s tests -v
python -m compileall -q src app.py train.py
```

Checks cover invalid inputs, derived features, fold-local preprocessing, stacking boundaries, development-only selection, release integrity, explanation reconstruction and the interactive prediction/explanation flow. GitHub Actions runs them after publishing; remote CI has not run locally.

`.gitattributes` preserves exact release/report bytes so manifest checks work across Windows and Linux. Checksums detect changes; they do not make arbitrary pickle files trustworthy. Load only the project's own model artifact.

## Publish this local repository

Git is initialized on `main`. Publish this existing folder through GitHub Desktop; the public repository and live hosting are the remaining account-side steps.

Suggested public name: **`sri-lanka-leptospirosis-ensemble-ml`** under **`Pasansemage`**.

In GitHub Desktop choose **File → Add local repository**, select this folder, then **Publish repository**. Set the suggested name and clear **Keep this code private** for a public repo. No remote is configured and no push is performed automatically.

Alternatively, create an empty public repository on GitHub, then run:

```sh
git remote add origin https://github.com/Pasansemage/sri-lanka-leptospirosis-ensemble-ml.git
git push -u origin main
```

## Data attribution and local-only files

Suneth Agampodi (2022), *Demographic, exposure, clinical, biochemical and diagnostic data of febrile patients recruited for the largest field study on leptospirosis in Sri Lanka*, Mendeley Data v4, DOI [10.17632/zmxmp42g9p.4](https://data.mendeley.com/datasets/zmxmp42g9p/4), CC BY 4.0. Data paper: [10.1016/j.dib.2022.108378](https://doi.org/10.1016/j.dib.2022.108378). The local CSV's equality to a publisher download has not been independently verified here.

Raw data, historical patient-containing bundles, temporary files, credentials and the virtual environment stay local. The tracked artifact is the fitted estimator, without the experiment dictionaries containing patient tables. Reference values are synthetic/aggregate; published reports are aggregate or synthetic. Supporting questionnaires are included as supplied and do not replace the CSV codebook. Outcome meanings, units and the 99 convention require verification before substantive clinical interpretation.
