# Feature engineering: decisions and evidence

The prediction point is admission. Source labels and units remain provisional. The new benchmark uses 54 original inputs (4 numeric and 50 categorical codes). The engineered variant adds four derived inputs. No step uses the target to create a feature.

| Decision | Exact implementation | Reason | Limitation / evidence |
|---|---|---|---|
| Admission-only allowlist | Age, Sex, four admission-vital fields including age, 29 admission symptom fields, 20 exposure fields; see `src/data.py` for exact names | Features should be available at the stated prediction time | Exposure history is assumed available at admission; timing still needs source verification |
| Exclude IDs and collection context | Exclude Serial, Hospital, Sample, Year, Month, ICU/OPD and all other unlisted fields | IDs encourage memorization; site/sample/time can proxy recruitment or outcomes; ICU may be downstream | Site and temporal external validation remain necessary; excluding site alone does not remove site effects |
| Exclude diagnostic/follow-up fields | Final, MAT, PCR, serial-day measurements and uncertain-time laboratory fields are never predictors | Tests defining or revealing the outcome and measurements obtained later cause leakage | A verified availability/codebook audit could enable a different admission-only contract |
| Eligible cohort | Only Final in {1,2} and at least one recorded admission symptom | Match the prior project cohort; avoid predicting from wholly absent admission symptom records | Excludes patients systematically; results apply only to this selected cohort |
| Missing source values | Parse numerics; convert 99, blank, unparsable entries to missing | 99 dominates missing blocks; this preserves the earlier project's provisional convention | Not codebook-confirmed. A real value of 99 (e.g. pulse or age) would be lost; verify before substantive research interpretation |
| Numeric bounds | Age [0,110], PRad [20,250], SBPadd [40,300], DBPadd [20,200]; outside values become missing | Deterministic protection against invalid source codes and gross data-entry errors | Broad project validation bounds, not clinical reference ranges. No clipping or learned outlier removal |
| Categorical representation | Only 1/2 retained; missing becomes an explicit Missing category; one-hot encoding handles unseen codes | Avoid assuming code 2 is twice code 1 or that code 1 means symptom present | Keep neutral codes until codebook verification. Encoding is fitted inside each fold |
| Numeric imputation | Fold-training median with missing indicators; retain empty features | Robust to extreme numeric values and preserves the fact that an input was unrecorded | Missing indicators may learn collection practices; no population-wide median is fitted |
| Numeric scaling | Standardize after imputation using fold-training means/SDs | Useful for logistic regression, SVM and KNN; gives every family the same preprocessing contract | Tree models do not require scaling; it does not create information |
| Pulse pressure candidate | SBPadd − DBPadd, only if both observed and SBP ≥ DBP | Exposes a deterministic interaction that shallow/additive learners might otherwise need to discover | Assumes both pressures share source units; correlated with original pressures and often missing |
| Mean arterial pressure candidate | DBPadd + (SBPadd − DBPadd)/3, same validity mask | Makes a weighted summary of two existing numeric inputs available to low-depth models | An approximation, not a new measured value; units/semantics unverified; kept only as a candidate representation |
| Symptom recording fraction candidate | Number of nonmissing symptom codes divided by 29, before imputation | Distinguish sparse documentation from fully recorded symptom history | Counts documentation, not symptoms present; can encode site practices |
| Exposure recording fraction candidate | Number of nonmissing exposure codes divided by 20, before imputation | Distinguish unrecorded history from recorded source codes | Counts documentation, not exposures present; shift-sensitive |
| Reject positive-code counts | No symptom burden / exposure burden based on code 1 | The presence/absence mapping is unverified | A codebook is required before such aggregations are meaningful |
| Reject arbitrary feature removal | No whole-dataset correlation/importance screening; no post-test feature dropping | Preserve leakage boundaries and avoid test-driven choices | 54 original inputs with correlated fields can dilute individual importance |
| Reject synthetic oversampling here | Original class proportions; equal class weights across all new candidates | Moderate code imbalance; consistent objective across ensemble families; avoid creating mixtures of sparse categorical records | ROC AUC alone does not solve imbalance; report AP, class prevalence, recall, precision, F1 and calibration metrics |

Generated comparison tables are written locally under `outputs/reports/`; no patient rows or fitted artifacts are distributed. The static historical outcome is described in `docs/MODEL_CARD.md`.

## How feature engineering is retained or rejected

Both `base` and `engineered` representations are declared before the new run. They use exactly the same development folds and every estimator refits all learned preprocessing inside its fitting fold. Only the ten approved classification methods enter selection; no stacking is used.

`feature_ablation.csv` compares each model's best development configuration under both representations. Its `engineered_minus_base` column reports the change in mean validation ROC AUC. Hyperparameters can differ between these best configurations, so it is a representation-plus-tuning comparison rather than a controlled single-parameter causal effect. `cv_candidates.csv` also permits matched-candidate comparisons.

The overall selection chooses the complete model/feature-set/candidate combination with the highest development mean ROC AUC. The exact-AUC tie-breaker is mean average precision, followed by deterministic name/feature/candidate order. Therefore engineered inputs are not retained merely because they sound plausible. The selected representation is recorded in `selection.json`; the generated reports state the current measured outcome. `MODEL_CARD.md` preserves an earlier, superseded experiment outside the approved scope.

No further predictor is added or removed after seeing test performance or explainability results. The historical holdout has already been examined in the previous analysis; this is internal research validation. Multiple model/feature comparisons can make the winning CV estimate optimistic; fresh external or temporal evaluation is future work.
