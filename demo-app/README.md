# Sri Lanka Leptospirosis demo

The demo lives in `demo-app/` inside the `sri-lanka-leptospirosis-ensemble-ml` repository. It serves the selected Random Forest with engineered features, using a saved model and aggregate or synthetic reports. It requires no source patient CSV or startup training.

## Streamlit Community Cloud settings

Commit and push the whole repository, including this directory, to GitHub. In Streamlit Community Cloud, use:

| Setting | Value |
|---|---|
| Repository | `Pasansemage/sri-lanka-leptospirosis-ensemble-ml` |
| Branch | `main` |
| Main file path | `demo-app/app.py` |
| Python version in Advanced settings | `3.12` |
| Secrets | None |

The app dependencies are in `demo-app/requirements.txt`. Community Cloud searches the entrypoint directory before the repository root, so these serving dependencies take precedence over the research requirements. Keep `.streamlit/config.toml` at the repository root.

1. In GitHub Desktop, select the existing `sri-lanka-leptospirosis-ensemble-ml` repository.
2. Commit the updated research files, `demo-app/`, root Streamlit configuration and demo CI workflow. Push to `main`.
3. Open https://share.streamlit.io/, sign in with GitHub and choose **Create app**.
4. Choose the repository and settings above, then click **Deploy**.
5. Verify **Random Forest**, model version **6fe8ec8491c4**, and the default hypothetical example score **0.0726** for Age=35, Feverad=1 and other inputs missing. Request its explanation as well.

Official instructions: [deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization), and [dependency selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). Public deployment has not been verified locally.

## Repository layout

```text
sri-lanka-leptospirosis-ensemble-ml/
  .streamlit/config.toml
  .github/workflows/demo.yml
  notebooks/modelling.ipynb
  requirements.txt                 Research dependencies
  train.py
  src/                             Research modules
  demo-app/
    app.py
    requirements.txt               Serving dependencies
    artifacts/                     Model, reference and manifest
    reports/                       Aggregate and synthetic release reports
    src/                           Serving validation, loading and explanations
    tests/test_deployment.py
```

The app directory is an ordinary folder tracked by the parent repository. It has no nested Git repository or submodule. Its allowlist admits only the reviewed serving files. Patient data, environments, temporary files and generated research outputs remain ignored.

## Run locally

From the repository root, use the Python 3.12 environment and install the app dependencies:

```sh
python -m pip install -r demo-app/requirements.txt
python -m streamlit run demo-app/app.py
```

Run from the repository root to match Community Cloud's working directory. The app resolves its model and reports relative to its own files.

## Validation and updates

Run the demo checks in a separate Python process from the app directory:

```sh
cd demo-app
python -m unittest discover -s tests -v
```

The root `.github/workflows/demo.yml` runs these checks using the app's serving dependencies, alongside the research workflow. The checks cover release hashes, the input contract, approved models, explanations and interactive inference. Confirm actual remote CI results after pushing. Keep model, reports, code and pinned dependencies together when updating or rolling back the release.

## Model evidence

The benchmark includes Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, AdaBoost, Gradient Boosting, LightGBM and CatBoost. Linear Regression is reserved for continuous outcomes. Selection uses mean stratified five-fold development ROC AUC. Random Forest with engineered features scored **0.6028** in development and **0.6328** on the internal holdout. Recall at threshold 0.50 was **0.2021**. The constant prior reference is not a candidate estimator.

Research demonstration only. Final code meanings, measurement units and the 99 missing convention remain provisional. Scores are uncalibrated and the historical holdout is not independent external validation. Use hypothetical inputs. The app does not persist submitted records.

Source: Suneth Agampodi (2022), [Mendeley Data v4](https://data.mendeley.com/datasets/zmxmp42g9p/4), DOI 10.17632/zmxmp42g9p.4, publisher-listed CC BY 4.0. Obtain original data from the publisher; the app does not redistribute patient records.
