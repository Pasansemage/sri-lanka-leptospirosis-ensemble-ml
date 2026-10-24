# Free public hosting and basic MLOps

## Streamlit Community Cloud

Streamlit's [official documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud) describes free hosting linked to GitHub. This app serves the saved Python model directly; GitHub Pages alone cannot execute its inference pipeline.

1. Push this folder to a public repository, suggested name `Pasansemage/sri-lanka-leptospirosis-ensemble-ml`.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with the GitHub account that has admin access to the repository.
3. Choose **Create app**, repository `Pasansemage/sri-lanka-leptospirosis-ensemble-ml`, branch `main`, entrypoint **`app.py`** at the repository root.
4. In Advanced settings select **Python 3.12**. `requirements.txt` supplies the tested versions. No application secrets are required.
5. Optionally request an available subdomain such as `sri-lanka-leptospirosis-ml`, then Deploy. Availability is not guaranteed.
6. Confirm successful startup, compare the displayed model version with `artifacts/manifest.json`, and run the default hypothetical example followed by **Explain this prediction**.
7. Check the evidence and explanation tabs, then record the actual public URL in README. No public URL is provisioned by this local setup.

The source CSV is not needed for hosting. Keep it and historical patient-containing bundles out of the public repository.

Official references: [deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [resources/app sleep](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status). Checked 24 October 2026. Apps without traffic for 12 hours sleep and visitors can wake them; platform limits can change. This is a free academic demo with possible cold starts.

## Implemented MLOps

| Stage | Implementation |
|---|---|
| Reproducibility | Explicit source contract, seed 42, shared folds, declared compact grids |
| Selection | Mean development ROC AUC, exact ties by AP; choice persisted before test predictions |
| Evaluation | Winner and prior baseline, fixed threshold, fixed-model bootstrap intervals |
| Explainability | Raw-feature permutation importance and local Monte Carlo Shapley game |
| Versioning | Model-hash version, data/source/package/report hashes in manifest |
| Serving | Cached predictor, integrity/package checks, validated inputs, no startup training |
| Checks | Data contract, leakage boundaries, selection, explanations and interactive app tests |
| CI | GitHub Actions configuration for pushes and pull requests; execution requires publishing |
| History | Local append-only aggregate run history |

Retraining and promotion are manual. The app shows example inference latency; live drift/outcome monitoring and automatic retraining are not implemented.

## Release and rollback

1. Train locally with `python train.py` and inspect comparison, feature ablation, metrics and explanations. Keep the provisional codebook assumptions explicit.
2. Run `python -m unittest discover -s tests -v`. Confirm hypothetical inference and explanations work.
3. Commit code, requirements, artifacts, aggregate reports and documentation together. Inspect staged files for raw data or secrets.
4. Push and verify CI. Community Cloud follows the configured branch; it can redeploy before CI finishes, so this is **not a CI-gated promotion pipeline**. For a manual gate, deploy a dedicated release branch and update it only after checks pass on main.
5. Verify the public model version and example prediction after deployment.

Rollback by reverting the release commit (`git revert <commit>`) and pushing the revert, keeping code, requirements, reports and model in sync. Use Community Cloud logs for failures and its app controls for restarts. No application-level raw input logging is included. Future monitoring could track aggregate latency, errors and missingness shifts; meaningful model-performance monitoring requires independently obtained outcomes. Do not repeatedly tune against this historical holdout and call it independent validation.
