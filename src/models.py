"""Extra Trees focal study and predeclared comparison classifiers.

Linear regression is approved for regression tasks, but the current target is binary.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from .data import AdmissionFeatures, NUMERIC, CATEGORICAL, DERIVED, SEED

FOCAL_MODEL = 'Extra trees'

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
                result.append(dict(model=name, family=family,
                                   role='Focal algorithm' if name == FOCAL_MODEL else 'Comparison',
                                   feature_set='engineered' if variant else 'base',
                                   candidate=i, estimator=pipeline(model, variant)))
    # Extra Trees samples split thresholds; this configuration fits every tree on
    # the whole fitting fold. It extends the originally supplied taught-model list.
    add(FOCAL_MODEL, 'Randomized trees', [ExtraTreesClassifier(n_estimators=150, min_samples_leaf=10,
        max_features='sqrt', bootstrap=False, n_jobs=1, random_state=SEED)])
    add('Logistic regression', 'Linear', [LogisticRegression(C=c, max_iter=2000, random_state=SEED) for c in [0.1, 1]])
    add('Decision tree', 'Tree', [DecisionTreeClassifier(max_depth=d, min_samples_leaf=10, random_state=SEED) for d in [3, 5]])
    add('Random forest', 'Bagging', [RandomForestClassifier(n_estimators=150, min_samples_leaf=l, max_features='sqrt', n_jobs=1, random_state=SEED) for l in [5, 15]])
    add('SVM', 'Kernel', [SVC(C=c, kernel='rbf', probability=True, random_state=SEED) for c in [.1, 1]])
    add('KNN', 'Neighbors', [KNeighborsClassifier(n_neighbors=k, n_jobs=1) for k in [15, 25]])
    add('AdaBoost', 'Boosting', [AdaBoostClassifier(n_estimators=100, learning_rate=r, random_state=SEED) for r in [.1, .5]])
    add('Gradient boosting', 'Boosting', [GradientBoostingClassifier(n_estimators=100, max_depth=d, learning_rate=.05, min_samples_leaf=10, random_state=SEED) for d in [1, 2]])
    add('XGBoost', 'Boosting', [XGBClassifier(n_estimators=120, max_depth=d, learning_rate=.05, subsample=.8,
        colsample_bytree=.8, reg_lambda=5, min_child_weight=5, tree_method='hist', n_jobs=1, random_state=SEED, eval_metric='logloss') for d in [2, 3]])
    add('LightGBM', 'Boosting', [LGBMClassifier(n_estimators=120, num_leaves=7, learning_rate=.05, reg_lambda=5, verbosity=-1, n_jobs=1, random_state=SEED)])
    add('CatBoost', 'Boosting', [CatBoostClassifier(iterations=120, depth=3, learning_rate=.05, l2_leaf_reg=5,
        verbose=False, allow_writing_files=False, thread_count=1, random_seed=SEED)])
    return result
