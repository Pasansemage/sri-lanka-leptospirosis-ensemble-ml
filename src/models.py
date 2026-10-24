"""Predeclared modest candidate grids; inner stacking folds own their preprocessing."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from .data import AdmissionFeatures, NUMERIC, CATEGORICAL, DERIVED, SEED

def pipeline(estimator, engineered=False):
    num = NUMERIC + (DERIVED if engineered else [])
    prepare = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                              ('scale', StandardScaler())]), num),
        ('category', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing', keep_empty_features=True)),
                               ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL)])
    return Pipeline([('features', AdmissionFeatures(engineered)), ('prepare', prepare), ('model', estimator)])

def candidates():
    """All weights equal: isolate algorithm effects under a consistent objective."""
    result = []
    def add(name, family, models):
        for variant in [False, True]:
            for i, model in enumerate(models):
                result.append(dict(model=name, family=family, feature_set='engineered' if variant else 'base',
                                   candidate=i, estimator=pipeline(model, variant)))
    add('Prior baseline', 'Baseline', [DummyClassifier(strategy='prior')])
    add('Logistic regression', 'Baseline', [LogisticRegression(C=c, max_iter=2000, random_state=SEED) for c in [0.1, 1]])
    add('Bagged trees', 'Bagging', [BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED),
        n_estimators=100, max_samples=.8, n_jobs=1, random_state=SEED)])
    add('Random forest', 'Bagging', [RandomForestClassifier(n_estimators=150, min_samples_leaf=l, max_features='sqrt', n_jobs=1, random_state=SEED) for l in [5, 15]])
    add('Extra trees', 'Bagging', [ExtraTreesClassifier(n_estimators=150, min_samples_leaf=10, n_jobs=1, random_state=SEED)])
    add('AdaBoost', 'Boosting', [AdaBoostClassifier(n_estimators=100, learning_rate=r, random_state=SEED) for r in [.1, .5]])
    add('Gradient boosting', 'Boosting', [GradientBoostingClassifier(n_estimators=100, max_depth=d, learning_rate=.05, min_samples_leaf=10, random_state=SEED) for d in [1, 2]])
    add('Histogram boosting', 'Boosting', [HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, learning_rate=.05, l2_regularization=5, early_stopping=False, random_state=SEED)])
    add('XGBoost', 'Boosting', [XGBClassifier(n_estimators=120, max_depth=d, learning_rate=.05, subsample=.8,
        colsample_bytree=.8, reg_lambda=5, min_child_weight=5, tree_method='hist', n_jobs=1, random_state=SEED, eval_metric='logloss') for d in [2, 3]])
    add('LightGBM', 'Boosting', [LGBMClassifier(n_estimators=120, num_leaves=7, learning_rate=.05, reg_lambda=5, verbosity=-1, n_jobs=1, random_state=SEED)])
    add('CatBoost', 'Boosting', [CatBoostClassifier(iterations=120, depth=3, learning_rate=.05, l2_leaf_reg=5,
        verbose=False, allow_writing_files=False, thread_count=1, random_seed=SEED)])
    for variant in [False, True]:
        bases = [('lr', pipeline(LogisticRegression(C=.1, max_iter=2000, random_state=SEED), variant)),
                 ('rf', pipeline(RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=1, random_state=SEED), variant)),
                 ('gb', pipeline(GradientBoostingClassifier(n_estimators=80, max_depth=1, learning_rate=.05, random_state=SEED), variant))]
        # The stack consumes RAW predictors; no outside fitted imputer/encoder can leak into inner OOF folds.
        for name, final in [('Stacking logistic', LogisticRegression(C=.1, max_iter=2000, random_state=SEED)),
                            ('Stacking nonlinear', GradientBoostingClassifier(n_estimators=50, max_depth=1, learning_rate=.03, min_samples_leaf=15, random_state=SEED))]:
            result.append(dict(model=name, family='Stacking', feature_set='engineered' if variant else 'base', candidate=0,
                estimator=StackingClassifier(estimators=bases, final_estimator=final,
                    cv=StratifiedKFold(3, shuffle=True, random_state=SEED), stack_method='predict_proba', passthrough=False, n_jobs=1)))
    return result
