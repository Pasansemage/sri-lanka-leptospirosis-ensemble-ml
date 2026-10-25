"""Run the predeclared development comparison, freeze a winner, evaluate, explain, export."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import time
import warnings

import joblib
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,
    log_loss, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold
from threadpoolctl import threadpool_limits

from src.data import ROOT, DATA, OUTPUT, FEATURES, NUMERIC, NOTICE, SEED, load_data, sha256
from src.models import FOCAL_MODEL, candidates
from src.explain import local_shapley

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def metrics(y, p):
    pred = np.asarray(p) >= .5
    return dict(roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
        accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)), brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])))

def intervals(y, p, repeats=1000):
    """Fixed-model, class-stratified percentile bootstrap; excludes selection uncertainty."""
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    indices = [np.flatnonzero(y == c) for c in [0, 1]]
    rows = []
    for _ in range(repeats):
        selected = np.concatenate([rng.choice(i, len(i), replace=True) for i in indices])
        rows.append(metrics(y[selected], p[selected]))
    return {k: np.quantile([r[k] for r in rows], [.025, .975]).tolist() for k in rows[0]}

def ranked(frame):
    # Exact mean AUC ties only. No retrospective metric switching or test-based tie-breaking.
    return frame.sort_values(['mean_roc_auc', 'mean_average_precision', 'model', 'feature_set', 'candidate'],
                             ascending=[False, False, True, True, True], kind='stable').reset_index(drop=True)

@threadpool_limits.wrap(limits=1)
def run(data_path=DATA, output=OUTPUT):
    start = time.perf_counter()
    output = Path(output)
    reports, artifacts = output / 'reports', output / 'artifacts'
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    raw, X, y = load_data(data_path)
    train_ids, test_ids = train_test_split(X.index, test_size=.2, random_state=SEED, stratify=y)
    Xdev, ydev = X.loc[train_ids], y.loc[train_ids]
    Xtest, ytest = X.loc[test_ids], y.loc[test_ids]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xdev, ydev))
    recipes = candidates()
    rows, summaries = [], []
    for spec_id, spec in enumerate(recipes):
        fold_rows = []
        for fold, (a, b) in enumerate(folds, 1):
            before = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = clone(spec['estimator']).fit(Xdev.iloc[a], ydev.iloc[a])
                p = model.predict_proba(Xdev.iloc[b])[:, 1]
            values = metrics(ydev.iloc[b], p)
            row = {k: v for k, v in spec.items() if k != 'estimator'}
            row.update(spec_id=spec_id, fold=fold, train_roc_auc=float(roc_auc_score(ydev.iloc[a], model.predict_proba(Xdev.iloc[a])[:, 1])),
                fit_predict_seconds=time.perf_counter() - before, warnings=' | '.join(sorted({str(w.message) for w in caught})), **values)
            rows.append(row)
            fold_rows.append(row)
        summary = {k: v for k, v in spec.items() if k != 'estimator'}
        summary.update(spec_id=spec_id, parameters=str(spec['estimator']))
        for k in values:
            summary['mean_' + k] = float(np.mean([r[k] for r in fold_rows]))
            summary['std_' + k] = float(np.std([r[k] for r in fold_rows], ddof=1))
        summary['mean_train_roc_auc'] = float(np.mean([r['train_roc_auc'] for r in fold_rows]))
        summary['mean_fit_predict_seconds'] = float(np.mean([r['fit_predict_seconds'] for r in fold_rows]))
        summaries.append(summary)
        print(f"{spec_id+1}/{len(recipes)} {spec['family']} | {spec['model']} | {spec['feature_set']} | candidate {spec['candidate']}: AUC {summary['mean_roc_auc']:.4f}", flush=True)
    comparison = ranked(pd.DataFrame(summaries))
    comparison.to_csv(reports / 'cv_candidates.csv', index=False)
    pd.DataFrame(rows).to_csv(reports / 'cv_folds.csv', index=False)
    leaders = comparison.drop_duplicates('model').reset_index(drop=True)
    leaders.to_csv(reports / 'cv_leaderboard.csv', index=False)
    # The feature ablation compares best hyperparameters within each feature set, using development only.
    ablation = comparison.drop_duplicates(['model', 'feature_set']).pivot(index='model', columns='feature_set', values='mean_roc_auc')
    ablation['engineered_minus_base'] = ablation.engineered - ablation.base
    ablation.to_csv(reports / 'feature_ablation.csv')
    best = comparison.iloc[0]
    selection = dict(selected_model=str(best.model), family=str(best.family), feature_set=str(best.feature_set),
        candidate=int(best.candidate), spec_id=int(best.spec_id), selection_metric='mean development 5-fold ROC AUC (higher is better)',
        tie_breaker='mean development average precision for exact AUC ties; then deterministic name/feature/candidate order',
        cv_roc_auc=float(best.mean_roc_auc), cv_auc_fold_std=float(best.std_roc_auc),
        cv_average_precision=float(best.mean_average_precision), threshold=.5, test_used_for_selection=False,
        n_candidates=len(recipes), seed=SEED, outer_folds=5, focal_model=FOCAL_MODEL,
        study_scope='Extra Trees as the focal extension beyond the supplied model list, with ten comparison classifiers; lecture coverage requires confirmation.',
        validation_status='Historical internal holdout already used in the earlier project; not independent external validation.')
    # Persist the choice before reading any held-out predictions.
    write_json(reports / 'selection.json', selection)
    winner = clone(recipes[int(best.spec_id)]['estimator']).fit(Xdev, ydev)
    p = winner.predict_proba(Xtest)[:, 1]
    prior = np.repeat(float(ydev.mean()), len(ytest))
    result = dict(model=selection['selected_model'], feature_set=selection['feature_set'], threshold=.5,
        metrics=metrics(ytest, p), baseline_metrics=metrics(ytest, prior),
        confidence_intervals=intervals(ytest, p),
        confidence_interval_method='1000 fixed-model class-stratified bootstrap samples; does not cover model selection or prevalence uncertainty',
        confusion_matrix=confusion_matrix(ytest, p >= .5, labels=[0, 1]).tolist(),
        confusion_matrix_order=['Final code 2 (0)', 'Final code 1 (1)'],
        development_n=len(Xdev), test_n=len(Xtest), eligible_n=len(X), source_n=len(raw), source_columns=len(raw.columns),
        development_code1_prevalence=float(ydev.mean()), test_code1_prevalence=float(ytest.mean()), source_sha256=sha256(data_path), notice=NOTICE)
    write_json(reports / 'test_results.json', result)
    # Deployed artifact holds the fitted predictor only, never the original X/y or patient IDs.
    joblib.dump(winner, artifacts / 'model.joblib', compress=3)
    reference = {c: (float(Xdev[c].median()) if Xdev[c].notna().any() else None) if c in NUMERIC else None for c in FEATURES}
    write_json(artifacts / 'reference.json', reference)
    pd.DataFrame({'feature': FEATURES, 'development_missing_fraction': Xdev.isna().mean(),
                  'type': ['numeric' if c in NUMERIC else 'categorical source code' for c in FEATURES]}).to_csv(reports / 'feature_audit.csv', index=False)
    return finalize_release(output, data_path, winner, reference, Xtest, ytest, p, prior, selection, start)

@threadpool_limits.wrap(limits=1)
def finalize_release(output, data_path, winner, reference, Xtest, ytest, p, prior, selection, start):
    reports, artifacts = Path(output) / 'reports', Path(output) / 'artifacts'
    result = json.loads((reports / 'test_results.json').read_text())
    leaders = pd.read_csv(reports / 'cv_leaderboard.csv')
    # Permute RAW columns to preserve derived-feature dependencies and jointly perturb their descendants.
    importance = permutation_importance(winner, Xtest, ytest, scoring='roc_auc', n_repeats=10, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({'feature': FEATURES, 'mean_auc_drop': importance.importances_mean, 'shuffle_std': importance.importances_std}).sort_values('mean_auc_drop', ascending=False)
    imp.to_csv(reports / 'permutation_importance.csv', index=False)
    example = pd.DataFrame([{c: (35.0 if c == 'Age' else '1' if c == 'Feverad' else np.nan) for c in FEATURES}])
    base, probability, local = local_shapley(winner, example, reference)
    local.to_csv(reports / 'synthetic_local_explanation.csv', index=False)
    write_json(reports / 'explainability.json', dict(model=selection['selected_model'], global_method='10-repeat held-out raw-feature permutation importance (ROC AUC drop)',
        local_method='24-permutation Monte Carlo Shapley baseline-substitution game in code-1 probability units',
        reference='Synthetic reference: development numeric medians and all categorical inputs missing; not an observed patient.',
        example='Synthetic example: Age=35, Feverad=1, all other features missing.',
        reference_probability=base, example_probability=probability, contribution_sum=float(local.contribution.sum()),
        limitations=['Correlated predictors can share or mask permutation importance.', 'Local contributions depend on the synthetic reference.',
            'Baseline substitutions may be outside the observed joint distribution.', 'Contributions explain the fitted model, not causality or clinical effects.',
            'Order-sampling standard deviations are not confidence intervals.']))
    curves_and_plots(reports, leaders, ytest, p, prior, imp, winner, reference)
    write_model_card(reports, selection, result)
    (reports / 'FEATURE_ENGINEERING.md').write_text(
        (ROOT / 'docs/FEATURE_ENGINEERING.md').read_text(encoding='utf-8'), encoding='utf-8')
    source_paths = [ROOT / 'train.py', *sorted((ROOT / 'src').glob('*.py')), ROOT / 'requirements.txt']
    source_hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in source_paths}
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        model=selection['selected_model'], feature_set=selection['feature_set'], source_data_sha256=sha256(data_path), seed=SEED,
        python=platform.python_version(), packages={pkg: importlib.metadata.version(pkg) for pkg in ['numpy','pandas','scikit-learn','scipy','joblib','xgboost','lightgbm','catboost']},
        code_sha256=source_hashes, artifacts={str(path.relative_to(output)).replace('\\', '/'): sha256(path) for path in sorted(artifacts.iterdir()) if path.is_file() and path.name != 'manifest.json'},
        report_sha256={str(path.relative_to(output)).replace('\\', '/'): sha256(path) for path in sorted(reports.iterdir()) if path.is_file() and path.name != 'run_history.jsonl'},
        elapsed_seconds=time.perf_counter() - start, notice=NOTICE)
    manifest['model_version'] = manifest['artifacts']['artifacts/model.joblib'][:12]
    write_json(artifacts / 'manifest.json', manifest)
    with (reports / 'run_history.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({k: manifest[k] for k in ['created_utc','model','feature_set','model_version','source_data_sha256','elapsed_seconds']}) + '\n')
    print(json.dumps(dict(selection=selection, test_metrics=result['metrics'], model_version=manifest['model_version']), indent=2), flush=True)
    return selection

def write_model_card(reports, selection, result):
    """Keep the generated release documentation aligned with the selected run."""
    interval = result['confidence_intervals']['roc_auc']
    table = '\n'.join(f"| {key} | {value:.4f} | {result['baseline_metrics'][key]:.4f} |"
                      for key, value in result['metrics'].items())
    text = f"""# Selected model: {selection['selected_model']}

Academic comparison using admission inputs to predict numerical Final code 1 versus code 2.
{result['notice']}

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

{selection['n_candidates']} configurations compare base and engineered features on
the same five stratified development folds. All learned preprocessing is fold-local.
Selection maximizes mean development ROC AUC, using average precision for exact ties,
then deterministic ordering. The holdout selects neither model nor features nor threshold.

Selected representation: **{selection['feature_set']}**, candidate {selection['candidate']}.
Development ROC AUC: **{selection['cv_roc_auc']:.4f}**; fold SD:
**{selection['cv_auc_fold_std']:.4f}**; average precision: **{selection['cv_average_precision']:.4f}**.
Fold SD is not a confidence interval. Consult cv_candidates.csv and feature_ablation.csv
for the full comparison and representation evidence.

## Internal holdout evaluation

Eligible records: {result['eligible_n']}; development: {result['development_n']};
holdout: {result['test_n']}. Threshold: {selection['threshold']:.2f}.

| Metric | Selected model | Constant prior reference |
|---|---:|---:|
{table}

Confusion matrix (actual rows, predicted columns; code 2, code 1):
**{result['confusion_matrix']}**. ROC AUC 95% fixed-model bootstrap interval:
**[{interval[0]:.4f}, {interval[1]:.4f}]**. This excludes selection uncertainty.

## Explanations and limitations

Ten-repeat raw-feature permutation importance follows selection. Local explanations
use 24 random feature orders against a synthetic median/missing reference; reference
score plus contributions reconstructs the prediction. These are model explanations,
not causal effects. See explainability.json and the synthetic explanation CSV.

{selection['validation_status']}
Outcome semantics, units and missing-value codes remain provisional. Low recall,
uncalibrated scores, cohort selection, correlated predictors, recording practices,
selection optimism and hospital/time shifts limit interpretation. Independent validation
is required before stronger claims. Patient data are not included in the saved release.
"""
    (reports / 'MODEL_CARD.md').write_text(text, encoding='utf-8')

@threadpool_limits.wrap(limits=1)
def resume_finalization(data_path=DATA, output=OUTPUT):
    """Rebuild explanations after a reporting failure without repeating model selection."""
    output = Path(output)
    result = json.loads((output / 'reports/test_results.json').read_text())
    if sha256(data_path) != result['source_sha256']:
        raise ValueError('Source changed; a full training run is required.')
    _, X, y = load_data(data_path)
    a, b = train_test_split(X.index, test_size=.2, random_state=SEED, stratify=y)
    winner = joblib.load(output / 'artifacts/model.joblib')
    p = winner.predict_proba(X.loc[b])[:,1]
    for key, value in metrics(y.loc[b], p).items():
        if not np.isclose(value, result['metrics'][key]):
            raise ValueError('Saved model differs from the evaluated model.')
    reference = json.loads((output / 'artifacts/reference.json').read_text())
    selection = json.loads((output / 'reports/selection.json').read_text())
    if selection['selected_model'] not in {spec['model'] for spec in candidates()} or selection.get('focal_model') != FOCAL_MODEL:
        raise ValueError('Saved model belongs to a different study scope; a full training run is required.')
    return finalize_release(output, data_path, winner, reference, X.loc[b], y.loc[b], p,
        np.repeat(float(y.loc[a].mean()), len(b)), selection, time.perf_counter())

def curves_and_plots(reports, leaders, y, p, prior, imp, model, reference):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    fraction, mean = calibration_curve(y, p, n_bins=8, strategy='quantile')
    write_json(reports / 'curves.json', dict(fpr=fpr.tolist(), tpr=tpr.tolist(), precision=precision.tolist(), recall=recall.tolist(),
        calibration_mean=mean.tolist(), calibration_fraction=fraction.tolist()))
    axes[0].plot(fpr, tpr, color='#087f8c'); axes[0].plot([0,1], [0,1], '--', color='gray')
    axes[0].set(title='Winner: internal holdout ROC', xlabel='False positive rate', ylabel='True positive rate')
    axes[1].plot(recall, precision, color='#087f8c'); axes[1].axhline(np.mean(y), linestyle='--', color='gray')
    axes[1].set(title='Precision–recall', xlabel='Recall', ylabel='Precision')
    axes[2].plot(mean, fraction, 'o-', color='#087f8c'); axes[2].plot([0,1], [0,1], '--', color='gray')
    axes[2].set(title='Reliability (not calibrated)', xlabel='Mean predicted code-1 score', ylabel='Observed code-1 fraction')
    fig.tight_layout(); fig.savefig(reports / 'evaluation.png', dpi=150); plt.close(fig)
    plot = leaders.sort_values('mean_roc_auc')
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {'Linear':'#9ba7b5','Tree':'#677b91','Bagging':'#087f8c',
              'Kernel':'#7755ab','Neighbors':'#bc658d','Boosting':'#de8b35','Randomized trees':'#164e5b'}
    ax.barh(plot.model, plot.mean_roc_auc, xerr=plot.std_roc_auc, color=[colors[f] for f in plot.family], capsize=3)
    ax.axvline(.5, linestyle='--', color='gray'); ax.set(xlim=(.35,.8), xlabel='Mean development ROC AUC ± fold SD (not CI)', title='Extra Trees and comparison classifiers')
    fig.tight_layout(); fig.savefig(reports / 'comparison.png', dpi=150); plt.close(fig)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature, top.mean_auc_drop, xerr=top.shuffle_std, color='#087f8c', capsize=2)
    ax.axvline(0, color='gray'); ax.set(xlabel='Held-out ROC AUC decrease ± shuffle SD', title='Selected model: raw-feature permutation importance')
    fig.tight_layout(); fig.savefig(reports / 'importance.png', dpi=150); plt.close(fig)
    # Synthetic one-feature response; uses no patient rows and makes the fixed context explicit.
    points = []
    for age in range(18, 86):
        row = {c: reference[c] for c in FEATURES}; row['Age'] = float(age); row['Feverad'] = '1'
        points.append(row)
    frame = pd.DataFrame(points).where(pd.DataFrame(points).notna(), np.nan)
    for c in NUMERIC: frame[c] = frame[c].astype(float)
    pd.DataFrame({'Age': range(18,86), 'code1_probability': model.predict_proba(frame)[:,1]}).to_csv(reports / 'synthetic_age_response.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--finalize', action='store_true', help='Resume explanation/export for an already evaluated local model.')
    args = parser.parse_args()
    (resume_finalization if args.finalize else run)(args.data, args.output)
