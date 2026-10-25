# Historical reference model: Extra Trees

This archived experiment is **superseded and outside the current approved model scope**. Extra Trees, bagged trees, histogram boosting and stacking are excluded from new training. These recorded metrics must not be presented as results of the restricted benchmark. Reproduce locally and inspect outputs/reports/ for the current approved-model winner and results.

## Intended use and data contract

Academic comparison and hypothetical-record demonstration. Predict the numerical source label Final code 1 versus code 2 from admission-allowlisted inputs. Code meanings, numeric units and the 99 missing-value rule are not publisher-codebook verified. Do not interpret model output as a clinical diagnosis or calibrated risk.

The source contains 1734 rows and 806 columns. Eligibility requires Final in {1,2} and at least one recorded admission symptom, retaining 1206 records. Seed-42 stratification assigns 964 development and 242 holdout records. This reuses the earlier project's holdout; it is internal evaluation, not independent external validation.

## Selection evidence

36 declared combinations cover 13 methods, two feature representations and compact hyperparameter grids. Every candidate uses the same five development folds. Stacking has three inner folds with complete preprocessing pipelines. Equal class weights were used across that historical benchmark. Highest mean validation ROC AUC selects the complete recipe; exact ties use validation average precision, then deterministic ordering.

Selected recipe: **Extra trees / base**, candidate 0, mean CV ROC AUC **0.6054**, fold SD **0.0500**, CV average precision **0.4920**. Fold SD is not a confidence interval or significance test. Extra Trees averages 150 randomized trees with min_samples_leaf=10, random_state=42 and bootstrap=False; it belongs in the randomized-averaging comparison but is not literal bootstrap bagging in this configuration.

Extra Trees with engineered features scores 0.6000 versus 0.6054 for base features (difference −0.0054). Therefore pulse pressure, approximate mean arterial pressure and documentation fractions are excluded from the deployed winner. Base preprocessing still includes numeric median imputation, missing indicators, scaling, categorical Missing values and one-hot encoding. The test does not choose model, features or threshold.

## Internal holdout metrics

| Metric | Selected model | Prior baseline |
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

Threshold: 0.50. Confusion matrix: rows actual code 2/code 1, columns predicted code 2/code 1: **[[134,14],[75,19]]**. In particular, 75 of 94 code-1 records are missed at this threshold. ROC AUC 95% fixed-model stratified-bootstrap interval: **[0.5902, 0.7313]**. This excludes selection, prevalence and external-population uncertainty. Holdout code-1 prevalence is 0.3884; average precision should be compared with that prior baseline.

ROC AUC measures ranking; it does not establish high recall at threshold 0.50. The low code-1 recall (20.2%) materially limits usefulness. No threshold is optimized on the holdout. Probabilities are uncalibrated; Brier/log loss and a reliability plot are reported without claiming calibration.

## Explainability

Global: ten-repeat held-out permutation importance in the original input space, scored by ROC AUC drop. The full pipeline recomputes all descendant features for each shuffle. Highest measured mean drops: Neckstiffnessad, Cnsuffusionad, Headachead. Consult the shuffle SDs; correlated inputs can share or mask importance and negative importances can occur. This analysis follows selection and does not remove predictors.

Local: 24-permutation Monte Carlo Shapley values in probability units relative to one synthetic reference (development numeric medians, categorical inputs missing). Each path changes inputs from reference to example, and its marginal changes telescope exactly. Averaging approximates the order average; reference score plus summed contributions reconstructs the prediction. This is not population-conditional SHAP and baseline substitutions can be unrealistic. Neither method demonstrates causality.

The example is synthetic: Age=35, Feverad=1, everything else missing. Reference output **0.2074** plus summed contributions **-0.1302** equals example output **0.0772**. The synthetic age-response curve holds other inputs fixed; it is not a population effect.

## Reproducibility and limitations

Training generates `outputs/artifacts/model.joblib` containing the fitted predictor, with data/code/package/report hashes in a local manifest. No raw data, fitted artifacts or generated outputs are distributed in this research repository. Obtain the dataset directly from the source in README.md.

Important limits: provisional source semantics and sentinels, cohort selection, unverified timing/units, hospital/time shifts, missingness proxies, multiple-comparison optimism, correlated predictors, low recall, uncalibrated outputs, historical holdout reuse and lack of external validation. Verify the source codebook and obtain independent validation before stronger research claims.
