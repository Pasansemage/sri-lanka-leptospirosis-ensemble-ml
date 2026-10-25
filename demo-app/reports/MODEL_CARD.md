# Selected model: Random forest

Academic comparison using admission inputs to predict numerical Final code 1 versus code 2.
Research demonstration only. Final code 1 and code 2 have unverified clinical meanings. The 99 missing-value convention and measurement units are provisional. Scores are uncalibrated model outputs, not clinical risks.

## Approved scope and selection

Only Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, AdaBoost,
Gradient Boosting, LightGBM and CatBoost are fitted. Linear Regression is approved
for continuous-target tasks but is not fitted to this binary target. The prior
reference is a constant development-prevalence score, not a candidate estimator.

36 configurations compare base and engineered features on
the same five stratified development folds. All learned preprocessing is fold-local.
Selection maximizes mean development ROC AUC, using average precision for exact ties,
then deterministic ordering. The holdout selects neither model nor features nor threshold.

Selected representation: **engineered**, candidate 0.
Development ROC AUC: **0.6028**; fold SD:
**0.0477**; average precision: **0.4847**.
Fold SD is not a confidence interval. Consult cv_candidates.csv and feature_ablation.csv
for the full comparison and representation evidence.

## Internal holdout evaluation

Eligible records: 1206; development: 964;
holdout: 242. Threshold: 0.50.

| Metric | Selected model | Constant prior reference |
|---|---:|---:|
| roc_auc | 0.6328 | 0.5000 |
| average_precision | 0.5108 | 0.3884 |
| accuracy | 0.6116 | 0.6116 |
| balanced_accuracy | 0.5369 | 0.5000 |
| precision | 0.5000 | 0.0000 |
| recall | 0.2021 | 0.0000 |
| f1 | 0.2879 | 0.0000 |
| brier | 0.2238 | 0.2376 |
| log_loss | 0.6372 | 0.6680 |

Confusion matrix (actual rows, predicted columns; code 2, code 1):
**[[129, 19], [75, 19]]**. ROC AUC 95% fixed-model bootstrap interval:
**[0.5653, 0.7037]**. This excludes selection uncertainty.

## Explanations and limitations

Ten-repeat raw-feature permutation importance follows selection. Local explanations
use 24 random feature orders against a synthetic median/missing reference; reference
score plus contributions reconstructs the prediction. These are model explanations,
not causal effects. See explainability.json and the synthetic explanation CSV.

Historical internal holdout already used in the earlier project; not independent external validation.
Outcome semantics, units and missing-value codes remain provisional. Low recall,
uncalibrated scores, cohort selection, correlated predictors, recording practices,
selection optimism and hospital/time shifts limit interpretation. Independent validation
is required before stronger claims. Patient data are not included in the saved release.
