"""Build the teaching notebook and spoken script from the recorded experiment."""
import json
from pathlib import Path
import sys
import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    read = lambda name: json.loads((ROOT / name).read_text(encoding='utf-8'))
    selected = read('reports/selection.json')
    result = read('reports/test_results.json')
    explanation = read('reports/explainability.json')
    scores = result['metrics']
    leaderboard = pd.read_csv(ROOT / 'reports/cv_leaderboard.csv')
    importance = pd.read_csv(ROOT / 'reports/permutation_importance.csv')
    intervals = result['confidence_intervals']['roc_auc']
    model_card = f'''# Selected model: {selected['selected_model']}

## Intended use and data contract

Academic comparison and hypothetical-record demonstration. Predict the numerical source label Final code 1 versus code 2 from admission-allowlisted inputs. Code meanings, numeric units and the 99 missing-value rule are not publisher-codebook verified. Do not interpret model output as a clinical diagnosis or calibrated risk.

The source contains {result['source_n']} rows and {result['source_columns']} columns. Eligibility requires Final in {{1,2}} and at least one recorded admission symptom, retaining {result['eligible_n']} records. Seed-42 stratification assigns {result['development_n']} development and {result['test_n']} holdout records. This reuses the earlier project's holdout; it is internal evaluation, not independent external validation.

## Selection evidence

36 declared combinations cover 13 methods, two feature representations and compact hyperparameter grids. Every candidate uses the same five development folds. Stacking has three inner folds with complete preprocessing pipelines. Equal class weights are used across the new benchmark. Highest mean validation ROC AUC selects the complete recipe; exact ties use validation average precision, then deterministic ordering.

Selected recipe: **{selected['selected_model']} / {selected['feature_set']}**, candidate {selected['candidate']}, mean CV ROC AUC **{selected['cv_roc_auc']:.4f}**, fold SD **{selected['cv_auc_fold_std']:.4f}**, CV average precision **{selected['cv_average_precision']:.4f}**. Fold SD is not a confidence interval or significance test. Extra Trees averages 150 randomized trees with min_samples_leaf=10, random_state=42 and bootstrap=False; it belongs in the randomized-averaging comparison but is not literal bootstrap bagging in this configuration.

Extra Trees with engineered features scores 0.6000 versus 0.6054 for base features (difference −0.0054). Therefore pulse pressure, approximate mean arterial pressure and documentation fractions are excluded from the deployed winner. Base preprocessing still includes numeric median imputation, missing indicators, scaling, categorical Missing values and one-hot encoding. The test does not choose model, features or threshold.

## Internal holdout metrics

| Metric | Selected model | Prior baseline |
|---|---:|---:|
''' + '\n'.join(f"| {key} | {value:.4f} | {result['baseline_metrics'][key]:.4f} |" for key,value in scores.items()) + f'''

Threshold: 0.50. Confusion matrix: rows actual code 2/code 1, columns predicted code 2/code 1: **[[134,14],[75,19]]**. In particular, 75 of 94 code-1 records are missed at this threshold. ROC AUC 95% fixed-model stratified-bootstrap interval: **[{intervals[0]:.4f}, {intervals[1]:.4f}]**. This excludes selection, prevalence and external-population uncertainty. Holdout code-1 prevalence is {result['test_code1_prevalence']:.4f}; average precision should be compared with that prior baseline.

ROC AUC measures ranking; it does not establish high recall at threshold 0.50. The low code-1 recall ({scores['recall']:.1%}) materially limits usefulness. No threshold is optimized on the holdout. Probabilities are uncalibrated; Brier/log loss and a reliability plot are reported without claiming calibration.

## Explainability

Global: ten-repeat held-out permutation importance in the original input space, scored by ROC AUC drop. The full pipeline recomputes all descendant features for each shuffle. Highest measured mean drops: {', '.join(importance.head(3).feature)}. Consult the shuffle SDs; correlated inputs can share or mask importance and negative importances can occur. This analysis follows selection and does not remove predictors.

Local: 24-permutation Monte Carlo Shapley values in probability units relative to one synthetic reference (development numeric medians, categorical inputs missing). Each path changes inputs from reference to example, and its marginal changes telescope exactly. Averaging approximates the order average; reference score plus summed contributions reconstructs the prediction. This is not population-conditional SHAP and baseline substitutions can be unrealistic. Neither method demonstrates causality.

The example is synthetic: Age=35, Feverad=1, everything else missing. Reference output **{explanation['reference_probability']:.4f}** plus summed contributions **{explanation['contribution_sum']:+.4f}** equals example output **{explanation['example_probability']:.4f}**. The synthetic age-response curve holds other inputs fixed; it is not a population effect.

## Serving and limitations

`artifacts/model.joblib` contains the fitted predictor, not the experiment bundles holding patient tables. The app needs no raw CSV. The manifest records data/code/package/report hashes and the model-hash version. Original data and historical patient-containing bundles stay local.

Important limits: provisional source semantics and sentinels, cohort selection, unverified timing/units, hospital/time shifts, missingness proxies, multiple-comparison optimism, correlated predictors, low recall, uncalibrated outputs, historical holdout reuse and lack of external validation. Verify the source codebook and obtain independent validation before stronger research claims. Hosted deployment, external monitoring and automatic model promotion are not completed by a local repository setup.
'''
    (ROOT / 'reports/MODEL_CARD.md').write_text(model_card, encoding='utf-8')
    script = f'''# Presentation script — Sri Lanka Leptospirosis ML

Suggested duration: 8–10 minutes. Bracketed cues describe what to show, not words to read aloud. Source labels remain neutral throughout. Values below come from the recorded run; regenerate this script if the model changes.

## 0:00–0:50 — Project purpose

[Open the modelling notebook introduction.]

This project compares ensemble classifiers using admission features from a Sri Lankan hospital-based febrile-patient study. My goal is to compare bagging, boosting and stacking fairly, choose a model using explicit evidence, explain what it learned, and serve it through an interactive application.

The source contains {result['source_n']} records and {result['source_columns']} columns. I use Final code 1 versus Final code 2 as numerical labels. Their clinical meanings and the missing-value convention still need verification against the publisher's CSV codebook. I therefore present this as an academic modelling demonstration, not a clinical diagnostic tool.

## 0:50–2:10 — Feature engineering and leakage

[Show the feature-engineering decision table and ablation results.]

I define the prediction point as admission. The allowlist retains 54 original inputs: age, pulse, two blood-pressure fields, sex codes, 29 symptom fields and 20 exposure-history fields. Identifiers, hospital and collection context, diagnostic tests, follow-up measurements, and uncertain-time laboratory values are excluded. This reduces direct outcome leakage and avoids memorizing patient identifiers or recruitment context.

I preserve the earlier cohort rule: a record needs a supported outcome code and at least one recorded admission symptom. That leaves {result['eligible_n']} records. This selects a subgroup, so it limits generalization.

Blank or invalid values and source 99 are provisionally treated as missing. This is an explicit assumption: a real measurement equal to 99 could be lost, which is why codebook verification matters. Broad fixed numeric bounds catch gross input errors. I use median imputation and missing indicators for numerics, plus a Missing category and one-hot encoding for categorical codes. I do not assume code 1 means presence. All learned preprocessing is fitted using the fitting fold only.

I also test four derived inputs: pulse pressure, approximate mean arterial pressure, and the fraction of symptom and exposure fields that were recorded. The first two expose combinations of numeric inputs; the fractions describe documentation completeness, not symptom burden. These features might help shallow models, but they are correlated and may capture hospital recording practices.

I test the base and engineered representations on identical development folds. For the winning Extra Trees model, engineering lowers mean AUC from 0.6054 to 0.6000. I therefore deploy the base representation. Feature engineering is evaluated rather than assumed beneficial.

## 2:10–3:40 — Bagging, boosting and stacking

[Show the three comparison sections and the combined chart.]

The comparison includes 13 methods and 36 complete recipes. I preserve a stratified 80/20 split with seed 42: {result['development_n']} development records and {result['test_n']} internal holdout records. The development set has five identical stratified folds for every candidate.

The bagging comparison includes bagged decision trees, random forest and Extra Trees. Bagged trees bootstrap training subsets and average predictions. Random forest also randomizes feature choices at splits. Extra Trees adds random split thresholds. Its configuration here uses the full fitting fold rather than bootstrap samples, so it is more precisely a randomized averaging ensemble grouped with the bagging comparison.

Boosting fits learners sequentially to improve an objective. I compare AdaBoost, gradient boosting, histogram gradient boosting, XGBoost, LightGBM and CatBoost. Their capacity and regularization settings are intentionally modest. CatBoost receives the shared one-hot representation here, so this is a controlled common-preprocessing comparison, not a demonstration of its native categorical mode.

Stacking learns a combiner from predictions of logistic regression, random forest and gradient boosting. I compare a regularized logistic combiner and a shallow nonlinear boosting combiner. Each uses three inner folds to generate out-of-fold predictions. Complete base pipelines, including imputers and encoders, are fitted inside those inner folds. Fitting preprocessing outside the stack would leak information into the meta-training procedure.

The prior classifier and logistic regression provide reference points. Every model uses equal class weights in this new comparison. These small grids do not establish a globally optimal architecture.

## 3:40–5:00 — Choosing the winner and interpreting metrics

[Show selection.json, the development leaderboard and held-out metrics.]

My primary selection metric is mean development ROC AUC. It evaluates how well code-1 records are ranked above code-2 records across thresholds. Exact AUC ties use development average precision, then a deterministic ordering. The winner is written to disk before held-out predictions are evaluated.

{selected['selected_model']} with base features wins with mean development ROC AUC {selected['cv_roc_auc']:.4f}. Its fold standard deviation is {selected['cv_auc_fold_std']:.4f}. The scores are close, so I do not claim statistical superiority from the winning mean.

On the internal holdout, ROC AUC is {scores['roc_auc']:.4f}. Average precision is {scores['average_precision']:.4f}, compared with code-1 prevalence {result['test_code1_prevalence']:.4f}. Accuracy is {scores['accuracy']:.4f} versus {result['baseline_metrics']['accuracy']:.4f} for the prior baseline, so accuracy alone would exaggerate what the model achieves.

At the fixed 0.50 threshold, precision is {scores['precision']:.4f}, recall is {scores['recall']:.4f}, and F1 is {scores['f1']:.4f}. The confusion matrix contains 134 correctly predicted code-2 records, 14 code-2 records predicted as code 1, 75 missed code-1 records, and 19 correctly predicted code-1 records. The recall limitation is substantial even though AUC is above chance.

Brier score is {scores['brier']:.4f} and log loss is {scores['log_loss']:.4f}; lower is better. I show a reliability plot but do not call the probabilities calibrated. A fixed-model bootstrap AUC interval is approximately {intervals[0]:.4f} to {intervals[1]:.4f}. It does not capture model-selection uncertainty.

This is the same historical holdout used earlier in the project, so it is internal evaluation. Independent temporal or external validation remains future work. I do not choose a new threshold by inspecting this holdout.

## 5:00–6:30 — Explain the chosen model

[Open the global importance chart, then run the synthetic example's local explanation.]

I explain the selected Extra Trees predictor after model selection. For global importance, I shuffle one original input at a time on the holdout and measure the ROC AUC decrease. Ten shuffles estimate sensitivity to the shuffle. Using the original columns makes the result easier to interpret and preserves the dependency between an original input and any derived descendants.

The top measured inputs by mean drop are {', '.join(importance.head(3).feature)}. The error bars are shuffle standard deviations, not evidence of causal effects. Correlated inputs can mask each other, and unimportant-looking inputs for this model are not necessarily unimportant in the population.

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
'''
    (ROOT / 'PRESENTATION_SCRIPT.md').write_text(script, encoding='utf-8')

    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text))
    md('''# Sri Lanka Leptospirosis ML: ensemble modelling and explanation

This notebook is the main modelling walkthrough: precise feature decisions → bagging, boosting and stacking → development-only model selection → held-out evidence → explanation of the selected model → serving and MLOps.

The earlier `analysis.ipynb` remains a historical course analysis. This notebook reads the new reproducible benchmark from `../reports/` and the deployed estimator from `../artifacts/`. Saved-results analysis requires no raw patient CSV. Source labels, units and the 99 missing convention remain provisional; this is an academic demonstration, not clinical guidance.''')
    md('## 1. Load the versioned experiment\nUse the Python 3.12 environment from `requirements-dev.txt`. Retraining is optional and disabled by default.')
    code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image
ROOT = Path.cwd() if (Path.cwd() / 'app.py').exists() else Path.cwd().parent
assert (ROOT / 'app.py').exists(), 'Run from the project root or leptospirosis_project folder.'
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
REPORTS = ROOT / 'reports'
read_json = lambda name: json.loads((REPORTS / name).read_text(encoding='utf-8'))
selection = read_json('selection.json')
test = read_json('test_results.json')
cv = pd.read_csv(REPORTS / 'cv_candidates.csv')
folds = pd.read_csv(REPORTS / 'cv_folds.csv')
leaderboard = pd.read_csv(REPORTS / 'cv_leaderboard.csv')
from src.runtime import load_release
model, reference, manifest = load_release(ROOT)
print('Verified model version:', manifest['model_version'])
print(test['notice'])''')
    code('''RUN_TRAINING = False
if RUN_TRAINING:
    from train import run
    run()  # Requires the local source CSV; rerun the load cell after completion.
''')
    md('## 2. Feature engineering with precise reasoning\nThe decisions below state the exact operation, reason, assumptions, leakage boundary and validation evidence. Derived fields are computed before imputation. Learned preprocessing is fitted only on fitting folds.')
    code("display(Markdown((REPORTS / 'FEATURE_ENGINEERING.md').read_text(encoding='utf-8')))\ndisplay(pd.read_csv(REPORTS / 'feature_audit.csv'))")
    md('### Inspect the actual transform and synthetic example\nThe transformer does not learn population statistics. Inspecting its source makes the notebook consistent with the training and serving implementation.')
    code('''import inspect
from src.data import AdmissionFeatures, validate_record
from src.models import pipeline
print(inspect.getsource(AdmissionFeatures))
print(inspect.getsource(pipeline))
synthetic = validate_record({'Age':35, 'Feverad':1, 'SBPadd':120, 'DBPadd':80})
display(AdmissionFeatures(True).fit_transform(synthetic)[['Age','pulse_pressure','mean_arterial_pressure','symptoms_recorded_fraction','exposures_recorded_fraction']])''')
    md('### Development-only ablation\nFor each method, compare its best base configuration with its best engineered configuration. The full candidate table allows matched-hyperparameter comparisons. A positive difference supports that representation within this search; it is not proof of significance or a causal feature effect.')
    code("display(pd.read_csv(REPORTS / 'feature_ablation.csv'))")
    md('## 3. Shared evaluation and declared recipes\nThe 80/20 stratified seed-42 split yields 964 development and 242 internal holdout records. Every recipe uses the same five development folds. The 36 recipes cover 13 methods with equal class weights. This historical holdout was already used in the earlier project, so it is not fresh external validation.\n\nWithin each outer fold, the model owns the feature transformer, imputer, encoder and scaler. Stacking owns complete raw-input pipelines inside its three inner folds; the meta-learner never receives in-sample base predictions. No preprocessing is fitted before those inner folds.')
    code('''from src.models import candidates
recipes = candidates()
display(pd.DataFrame([{k:v for k,v in spec.items() if k != 'estimator'} for spec in recipes]))
print('Candidates:', len(recipes), '| outer fold rows:', len(folds))''')
    columns = "['model','feature_set','mean_roc_auc','std_roc_auc','mean_average_precision','mean_f1','mean_recall','mean_brier','mean_train_roc_auc','mean_fit_predict_seconds']"
    md('## 4. Bagging and randomized averaging\n**Bagged trees** average bootstrap-fitted trees. **Random forest** adds random feature subsets at splits. **Extra Trees** randomizes split thresholds. Extra Trees uses `bootstrap=False` here, so it is a randomized averaging ensemble rather than literal bootstrap bagging. We compare them together as variance-reducing tree ensembles. Leaf-size constraints control overfitting. Each row below is that method\'s best development recipe; variability is fold SD, not a confidence interval.')
    code(f"display(leaderboard.loc[leaderboard.family == 'Bagging', {columns}])")
    md('## 5. Boosting\nCompare **AdaBoost**, **gradient boosting**, **histogram boosting**, **XGBoost**, **LightGBM** and **CatBoost**. Sequential learners improve a fitting objective; learning rate, shallow trees and regularization constrain capacity. All receive the same prepared features. CatBoost therefore uses one-hot inputs here; its native categorical mode is not evaluated. The compact grids are explicit and do not establish each library\'s best possible configuration.')
    code(f"display(leaderboard.loc[leaderboard.family == 'Boosting', {columns}])")
    md('## 6. Stacking\nThe base learners are logistic regression, random forest and gradient boosting. Compare a **regularized logistic** combiner with a **shallow nonlinear boosting** combiner. Three inner stratified folds generate meta-training predictions inside each of the five outer development folds. The held-out fold never selects or trains the combiner. Additional complexity must earn an improved development score.')
    code(f"display(leaderboard.loc[leaderboard.family == 'Stacking', {columns}])\nprint(next(s['estimator'] for s in recipes if s['model'] == 'Stacking logistic'))")
    md('## 7. Select the complete recipe using explicit metrics\n**Primary:** maximize mean development ROC AUC. **Exact ties:** maximize mean development average precision, then deterministic name/feature/candidate order. ROC AUC measures ranking across thresholds; AP provides prevalence-sensitive precision/recall context. Threshold-dependent F1, recall, precision and balanced accuracy describe behaviour at 0.50. Brier score and log loss describe probability error; lower is better. Training AUC and timing expose overfitting/computational cost but do not choose the winner.\n\nThe saved selection precedes holdout prediction. We do not pick the model by accuracy, by its test AUC, or by changing metrics after seeing results. Fold SD does not establish significance; selecting from many recipes can make the winning CV estimate optimistic.')
    code('''from train import ranked
ranked_cv = ranked(cv)
display(ranked_cv[['model','family','feature_set','candidate','mean_roc_auc','std_roc_auc','mean_average_precision']])
display(selection)
assert int(ranked_cv.iloc[0].spec_id) == selection['spec_id']
assert selection['test_used_for_selection'] is False
print('Chosen:', selection['selected_model'], '/', selection['feature_set'])
display(Image(filename=str(REPORTS / 'comparison.png')))''')
    md('### Selection conclusion\nExtra Trees with base features has development mean ROC AUC 0.6054, compared with 0.6000 for engineered Extra Trees. The deployed model therefore keeps the base representation. The winning mean is modest and scores are close; this is the best observed recipe under the declared criterion, not a claim of universal superiority.')
    md('## 8. Evaluate the frozen winner\nOnly the selected model and prior baseline are evaluated in the new holdout report. All precision/recall/F1 values use source Final code 1 as positive and threshold 0.50. Bootstrap intervals describe the fixed fitted model; they exclude full-search/selection and prevalence uncertainty. The historical holdout is internal research evidence.')
    code('''display(pd.DataFrame([{'metric':k, 'selected_model':v, 'prior_baseline':test['baseline_metrics'][k],
    'lower_95_percent':test['confidence_intervals'][k][0], 'upper_95_percent':test['confidence_intervals'][k][1]} for k,v in test['metrics'].items()]))
display(pd.DataFrame(test['confusion_matrix'], index=['Actual code 2','Actual code 1'], columns=['Predicted code 2','Predicted code 1']))
display(Image(filename=str(REPORTS / 'evaluation.png')))
print(test['confidence_interval_method'])''')
    md('### Interpretation\nROC AUC is 0.6612 and average precision 0.5335 against code-1 prevalence 0.3884. At threshold 0.50 the model misses 75 of 94 code-1 records; recall is only 0.2021 and balanced accuracy 0.5538. Above-chance ranking does not make it useful as a diagnostic tool. The reliability plot is descriptive; probabilities are uncalibrated. A future operating threshold must be justified by verified outcome meaning and an explicit cost/benefit objective using development-only data, followed by independent evaluation.')
    md('## 9. Explain the chosen model after selection\n### Global: original-feature permutation importance\nShuffle each original input on the internal holdout and measure the decrease in ROC AUC, repeating ten times. The full pipeline is rerun, preserving dependencies with any derived fields. Importance follows selection and does not drive post-test feature removal. Error bars show shuffle SD, not causal uncertainty. Correlated inputs can share/mask importance and negative values can occur.')
    code("display(pd.read_csv(REPORTS / 'permutation_importance.csv'))\ndisplay(Image(filename=str(REPORTS / 'importance.png')))")
    md('### Local: reference-based Monte Carlo Shapley explanation\nUse 24 random feature orders against one synthetic reference: development numeric medians and all source categories missing. Each order replaces reference values with example values and records marginal changes in code-1 probability. The average is a finite-permutation approximation; contributions telescope so baseline plus their sum reconstructs the prediction. This is a baseline-substitution game, not conditional population SHAP. Contributions depend on the reference, substitutions can be uncommon, and no causal or clinical interpretation follows.')
    code('''from src.explain import local_shapley
example = validate_record({'Age':35, 'Feverad':1})
base, predicted, local = local_shapley(model, example, reference)
print(f'Baseline {base:.6f} + sum {local.contribution.sum():+.6f} = prediction {predicted:.6f}')
assert np.isclose(base + local.contribution.sum(), model.predict_proba(example)[0,1])
display(local)
ax = local.head(12).iloc[::-1].plot.barh(x='feature', y='contribution', color='#087f8c', legend=False, title='Synthetic example: probability contributions')
ax.set_xlabel('Change relative to synthetic reference')
import matplotlib.pyplot as plt
plt.show()''')
    md('### Synthetic one-feature response\nThe curve varies age with Feverad=1, other source categories missing and other numerics at development medians. This one fixed synthetic context is not a population effect and exposes no actual patient row.')
    code("response = pd.read_csv(REPORTS / 'synthetic_age_response.csv')\nresponse.plot(x='Age', y='code1_probability', title='Synthetic age-response context', legend=False)\nplt.show()")
    md('## 10. Interactive serving and MLOps\nLaunch `python -m streamlit run app.py` from the repository root (or `../run_dashboard.ps1` from this folder). The UI serves the exact evaluated model, validates source-code inputs, displays immediate scores and explanations, and separates illustrative example thresholds from the published 0.50 metrics. Application code does not persist inputs.\n\nThe versioned artifact and reference contain no original patient tables; raw data and historical bundles stay local. Hash verification detects changed release bytes; only trusted project pickle artifacts should be loaded. The manifest records code/data/package/report provenance. Tests check the contract, leakage boundaries, selection, explanation reconstruction and interactive UI flow. GitHub Actions is configured; remote execution requires a push. Retraining and model promotion remain manual.')
    code("display({k:manifest[k] for k in ['model_version','model','feature_set','created_utc','python','source_data_sha256','packages']})\ndisplay(Markdown((ROOT / 'DEPLOYMENT.md').read_text(encoding='utf-8')))")
    md('## 11. Complete model card and presentation script\nThe model card records the outcome and limits. The spoken script follows this notebook and includes the measured model-selection, explainability and MLOps results.')
    code("display(Markdown((REPORTS / 'MODEL_CARD.md').read_text(encoding='utf-8')))\ndisplay(Markdown((ROOT / 'PRESENTATION_SCRIPT.md').read_text(encoding='utf-8')))")
    md('## Sources\n- Dataset: Suneth Agampodi, [Mendeley Data v4](https://data.mendeley.com/datasets/zmxmp42g9p/4), CC BY 4.0.\n- [scikit-learn StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html): internal out-of-fold meta-training.\n- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html): score-based model inspection and limitations.\n- [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud): hosting path, detailed in DEPLOYMENT.md.')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3 (project .venv)', 'language':'python','name':'python3'}, 'language_info':{'name':'python','version':'3.12'}})
    nbf.write(notebook, ROOT / 'leptospirosis_project/modelling.ipynb')
    print('Generated modelling.ipynb, MODEL_CARD.md and PRESENTATION_SCRIPT.md')

if __name__ == '__main__': main()
