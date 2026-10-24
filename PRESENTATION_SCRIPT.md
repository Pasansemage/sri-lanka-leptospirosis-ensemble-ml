# Presentation script — Sri Lanka Leptospirosis ML

Suggested duration: 8–10 minutes. Bracketed cues describe what to show, not words to read aloud. Source labels remain neutral throughout. Values below come from the recorded run; regenerate this script if the model changes.

## 0:00–0:50 — Project purpose

[Open the modelling notebook introduction.]

This project compares ensemble classifiers using admission features from a Sri Lankan hospital-based febrile-patient study. My goal is to compare bagging, boosting and stacking fairly, choose a model using explicit evidence, explain what it learned, and serve it through an interactive application.

The source contains 1734 records and 806 columns. I use Final code 1 versus Final code 2 as numerical labels. Their clinical meanings and the missing-value convention still need verification against the publisher's CSV codebook. I therefore present this as an academic modelling demonstration, not a clinical diagnostic tool.

## 0:50–2:10 — Feature engineering and leakage

[Show the feature-engineering decision table and ablation results.]

I define the prediction point as admission. The allowlist retains 54 original inputs: age, pulse, two blood-pressure fields, sex codes, 29 symptom fields and 20 exposure-history fields. Identifiers, hospital and collection context, diagnostic tests, follow-up measurements, and uncertain-time laboratory values are excluded. This reduces direct outcome leakage and avoids memorizing patient identifiers or recruitment context.

I preserve the earlier cohort rule: a record needs a supported outcome code and at least one recorded admission symptom. That leaves 1206 records. This selects a subgroup, so it limits generalization.

Blank or invalid values and source 99 are provisionally treated as missing. This is an explicit assumption: a real measurement equal to 99 could be lost, which is why codebook verification matters. Broad fixed numeric bounds catch gross input errors. I use median imputation and missing indicators for numerics, plus a Missing category and one-hot encoding for categorical codes. I do not assume code 1 means presence. All learned preprocessing is fitted using the fitting fold only.

I also test four derived inputs: pulse pressure, approximate mean arterial pressure, and the fraction of symptom and exposure fields that were recorded. The first two expose combinations of numeric inputs; the fractions describe documentation completeness, not symptom burden. These features might help shallow models, but they are correlated and may capture hospital recording practices.

I test the base and engineered representations on identical development folds. For the winning Extra Trees model, engineering lowers mean AUC from 0.6054 to 0.6000. I therefore deploy the base representation. Feature engineering is evaluated rather than assumed beneficial.

## 2:10–3:40 — Bagging, boosting and stacking

[Show the three comparison sections and the combined chart.]

The comparison includes 13 methods and 36 complete recipes. I preserve a stratified 80/20 split with seed 42: 964 development records and 242 internal holdout records. The development set has five identical stratified folds for every candidate.

The bagging comparison includes bagged decision trees, random forest and Extra Trees. Bagged trees bootstrap training subsets and average predictions. Random forest also randomizes feature choices at splits. Extra Trees adds random split thresholds. Its configuration here uses the full fitting fold rather than bootstrap samples, so it is more precisely a randomized averaging ensemble grouped with the bagging comparison.

Boosting fits learners sequentially to improve an objective. I compare AdaBoost, gradient boosting, histogram gradient boosting, XGBoost, LightGBM and CatBoost. Their capacity and regularization settings are intentionally modest. CatBoost receives the shared one-hot representation here, so this is a controlled common-preprocessing comparison, not a demonstration of its native categorical mode.

Stacking learns a combiner from predictions of logistic regression, random forest and gradient boosting. I compare a regularized logistic combiner and a shallow nonlinear boosting combiner. Each uses three inner folds to generate out-of-fold predictions. Complete base pipelines, including imputers and encoders, are fitted inside those inner folds. Fitting preprocessing outside the stack would leak information into the meta-training procedure.

The prior classifier and logistic regression provide reference points. Every model uses equal class weights in this new comparison. These small grids do not establish a globally optimal architecture.

## 3:40–5:00 — Choosing the winner and interpreting metrics

[Show selection.json, the development leaderboard and held-out metrics.]

My primary selection metric is mean development ROC AUC. It evaluates how well code-1 records are ranked above code-2 records across thresholds. Exact AUC ties use development average precision, then a deterministic ordering. The winner is written to disk before held-out predictions are evaluated.

Extra trees with base features wins with mean development ROC AUC 0.6054. Its fold standard deviation is 0.0500. The scores are close, so I do not claim statistical superiority from the winning mean.

On the internal holdout, ROC AUC is 0.6612. Average precision is 0.5335, compared with code-1 prevalence 0.3884. Accuracy is 0.6322 versus 0.6116 for the prior baseline, so accuracy alone would exaggerate what the model achieves.

At the fixed 0.50 threshold, precision is 0.5758, recall is 0.2021, and F1 is 0.2992. The confusion matrix contains 134 correctly predicted code-2 records, 14 code-2 records predicted as code 1, 75 missed code-1 records, and 19 correctly predicted code-1 records. The recall limitation is substantial even though AUC is above chance.

Brier score is 0.2212 and log loss is 0.6342; lower is better. I show a reliability plot but do not call the probabilities calibrated. A fixed-model bootstrap AUC interval is approximately 0.5902 to 0.7313. It does not capture model-selection uncertainty.

This is the same historical holdout used earlier in the project, so it is internal evaluation. Independent temporal or external validation remains future work. I do not choose a new threshold by inspecting this holdout.

## 5:00–6:30 — Explain the chosen model

[Open the global importance chart, then run the synthetic example's local explanation.]

I explain the selected Extra Trees predictor after model selection. For global importance, I shuffle one original input at a time on the holdout and measure the ROC AUC decrease. Ten shuffles estimate sensitivity to the shuffle. Using the original columns makes the result easier to interpret and preserves the dependency between an original input and any derived descendants.

The top measured inputs by mean drop are Neckstiffnessad, Cnsuffusionad, Headachead. The error bars are shuffle standard deviations, not evidence of causal effects. Correlated inputs can mask each other, and unimportant-looking inputs for this model are not necessarily unimportant in the population.

For an individual prediction, I use a Monte Carlo Shapley substitution game. The reference is synthetic: development medians for numeric inputs and Missing for the categorical codes. Across 24 random feature orders, I replace reference inputs with the submitted values and average the marginal score changes.

Positive contributions move the model toward source code 1 relative to this reference. The reference score plus all contributions reconstructs the prediction. This is an approximate order average against one reference, not conditional population SHAP. It is reference-dependent, and some substitutions can be unrealistic. It explains how this fitted predictor behaves, not what causes an outcome.

The example uses age 35 and Feverad code 1, with other fields missing. It is a hypothetical record, not an exposed patient row. The separate age-response plot changes age in one fixed synthetic context and must not be interpreted as a population effect.

## 6:30–7:30 — Interactive application

[Open the app's Live prediction tab. Submit the default example; click Explain this prediction.]

The application loads the same saved fitted model that was evaluated. It checks the artifact and report hashes and the key package versions. The source CSV is not required for serving and the app does not retrain at startup.

I can enter hypothetical admission inputs and get an immediate code-1 score, predicted source code and inference latency. Missing inputs use the same contract as training. The example requires at least one recorded symptom code and rejects invalid codes or inconsistent supplied blood pressures.

The threshold slider changes the displayed label for this example only. Published benchmark metrics remain at 0.50. The app also shows the development comparisons, holdout evidence, engineering ablation, global explanations and release provenance. The application does not persist submitted inputs.

## 7:30–9:00 — MLOps and free public hosting

[Show the repository, workflow file, manifest and DEPLOYMENT.md.]

The MLOps work makes this experiment reproducible and deployable. Source code and documentation live in Git. Training records the dataset hash, source hashes, package versions, seed, selected recipe and aggregate results. A SHA-256 prefix identifies the fitted model version.

Tests verify the data contract, fold-local preprocessing, stacking boundaries, development-only selection, finite predictions, explanation reconstruction and the interactive inference flow. GitHub Actions is configured to run these checks after the repository is pushed. Local configuration is not proof that remote CI has passed.

For free hosting, I will use Streamlit Community Cloud with the public GitHub repository. The deployment settings are branch main, entrypoint app.py and Python 3.12. The host installs the pinned requirements and serves the committed artifact. The raw CSV and patient-containing historical experiment bundles remain local.

Once deployed, I will confirm the public app's displayed model hash and run the hypothetical example. Community Cloud follows the configured GitHub branch; it can redeploy before CI finishes. A dedicated release branch updated after tests pass would provide a stronger manual promotion gate. A rollback must restore matching code, requirements, model and reports together.

Free hosting can sleep when idle, so cold starts are expected. Runtime logs help diagnose errors, while actual population drift and outcome monitoring are future work. No automatic retraining or performance monitoring is claimed.

The main findings are that a simple randomized ensemble wins this declared comparison, plausible engineered inputs do not automatically help, and above-chance ranking does not remove the low-recall limitation. The result is a transparent academic demo with reproducible evidence and a concrete public deployment path.
