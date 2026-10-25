# Selected model: Extra trees

Academic comparison using admission inputs to predict numerical Final code 1 versus code 2.
Research demonstration only. Final code 1 and code 2 have unverified clinical meanings. The 99 missing-value convention and measurement units are provisional. Scores are uncalibrated model outputs, not clinical risks.

## Focal algorithm and comparison scope

**Extra Trees (Extremely Randomized Trees)** is the focal extension beyond the
originally supplied model list. It samples candidate split thresholds at random,
then chooses the best of those candidates. This run uses 150 trees, minimum leaf
size 10, square-root feature sampling and bootstrap=False. Confirm its lecture
coverage against the course syllabus before claiming it was not taught.

Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, AdaBoost,
Gradient Boosting, LightGBM and CatBoost provide comparisons. Linear Regression
is reserved for continuous targets and is not fitted to this binary target. The prior
reference is a constant development-prevalence score, not a candidate estimator.

38 configurations compare base and engineered features on
the same five stratified development folds. All learned preprocessing is fold-local.
Selection maximizes mean development ROC AUC, using average precision for exact ties,
then deterministic ordering. The holdout selects neither model nor features nor threshold.

Selected representation: **base**, candidate 0.
Development ROC AUC: **0.6054**; fold SD:
**0.0500**; average precision: **0.4920**.
Fold SD is not a confidence interval. Consult cv_candidates.csv and feature_ablation.csv
for the full comparison and representation evidence.

## Internal holdout evaluation

Eligible records: 1206; development: 964;
holdout: 242. Threshold: 0.50.

| Metric | Selected model | Constant prior reference |
|---|---:|---:|
| roc_auc | 0.6612 | 0.5000 |
| average_precision | 0.5335 | 0.3884 |
| accuracy | 0.6322 | 0.6116 |
| balanced_accuracy | 0.5538 | 0.5000 |
| precision | 0.5758 | 0.0000 |
| recall | 0.2021 | 0.0000 |
| f1 | 0.2992 | 0.0000 |
| brier | 0.2212 | 0.2376 |
| log_loss | 0.6342 | 0.6680 |

Confusion matrix (actual rows, predicted columns; code 2, code 1):
**[[134, 14], [75, 19]]**. ROC AUC 95% fixed-model bootstrap interval:
**[0.5902, 0.7313]**. This excludes selection uncertainty.

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
