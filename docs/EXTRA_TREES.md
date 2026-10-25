# Extra Trees as the focal algorithm

The updated assignment asks for a non-deep-learning algorithm outside the taught syllabus, evaluation and XAI. Extra Trees (Extremely Randomized Trees) extends the original model list supplied for this project. Its absence from that list does not independently prove lecture coverage; check the actual syllabus before making that claim in the final submission.

## Why investigate this method?

Admission predictors include sparse categorical codes, missing measurements and possible interactions. Averaging randomized trees can represent nonlinear relationships without a deep network. This is a reason to investigate Extra Trees, not a guarantee that it will perform well. The same five development folds compare it with ten familiar classifiers.

A Decision Tree fits one set of split rules. Logistic Regression models an additive log-odds relationship in the prepared inputs. Random Forest averages trees, using random candidate features and usually bootstrap samples, while seeking the best split threshold. Extra Trees instead samples random thresholds for candidate features and chooses the best of those random candidates. Additional randomization changes the bias/variance trade-off.

## Declared configuration and fair comparison

- 150 trees, `min_samples_leaf=10`, `max_features='sqrt'`, `bootstrap=False`, `random_state=42` and one worker.
- Every tree uses the whole fitting fold, rather than bootstrap sampling.
- Base and engineered inputs each define one Extra Trees configuration. This restores the pre-existing configuration rather than tuning it against the observed holdout.
- Ten comparison classifiers retain their previous compact grids. The complete search has 38 configurations across 11 methods.
- Median imputation, missing indicators, scaling and one-hot encoding fit within each training fold.
- A stratified 80/20 split separates development from internal holdout. Shared stratified five-fold development ROC AUC selects the complete recipe, with average precision for exact ties.

Extra Trees is the focal study method. The code still selects the highest development-scoring recipe without forcing Extra Trees to win. All tables and the app identify the actual selected predictor. The earlier holdout has already been observed, so its evaluation remains historical internal validation.

## Explainability and critical discussion

Use raw-feature permutation importance and reference-based Monte Carlo Shapley contributions. Correlation, missingness proxies and unrealistic reference substitutions limit their interpretation. The synthetic age-response plot is not a population partial-dependence estimate. Discuss low recall, uncertain outcome labels, the provisional missing-code convention, selection optimism and potential age/sex/site recording biases. Subgroup fairness and independent validation have not been established.

Symptom Yes/No labels in the interface are a user-authorized display assumption. They do not change training codes or verify clinical source meanings. Preserve that distinction in the report and presentation.

## References

- [scikit-learn: Extremely Randomized Trees](https://scikit-learn.org/stable/modules/ensemble#extremely-randomized-trees)
- [ExtraTreesClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
- Geurts, Ernst and Wehenkel (2006), *Extremely randomized trees*, Machine Learning 63, 3-42, DOI [10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1).
