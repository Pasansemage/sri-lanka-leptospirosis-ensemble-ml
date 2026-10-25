"""Synthetic-data contract checks and optional checks on locally trained research outputs."""
import json
import unittest
import numpy as np
import pandas as pd
from src.data import FEATURES, NUMERIC, SYMPTOMS, OUTPUT, AdmissionFeatures, validate_record
from src.models import FOCAL_MODEL, pipeline, candidates
from sklearn.linear_model import LogisticRegression
from src.runtime import load_release
from src.explain import local_shapley

class ContractTests(unittest.TestCase):
    def test_reject_invalid_input(self):
        for record in [{'Age':35}, {'Age':-1,'Feverad':1}, {'Sex':3,'Feverad':1},
                       {'Age':float('inf'),'Feverad':1}, {'SBPadd':70,'DBPadd':80,'Feverad':1},
                       {'Final':1,'Feverad':1}]:
            with self.assertRaises(ValueError): validate_record(record)

    def test_features_before_imputation(self):
        row = validate_record({'Age':35,'Feverad':1,'SBPadd':120,'DBPadd':80})
        engineered = AdmissionFeatures(True).fit_transform(row)
        self.assertAlmostEqual(engineered.pulse_pressure.iloc[0], 40)
        self.assertAlmostEqual(engineered.mean_arterial_pressure.iloc[0], 80 + 40/3)
        self.assertAlmostEqual(engineered.symptoms_recorded_fraction.iloc[0], 1/len(SYMPTOMS))
        row['DBPadd'] = np.nan
        self.assertTrue(AdmissionFeatures(True).fit_transform(row).pulse_pressure.isna().all())

    def test_fold_preprocessing_isolation(self):
        a = pd.concat([validate_record({'Age':30+i,'PRad':70,'SBPadd':120,'DBPadd':80,'Feverad':1}) for i in range(8)], ignore_index=True)
        pipe = pipeline(LogisticRegression(), True).fit(a, [0,1]*4)
        imputer = pipe.named_steps['prepare'].named_transformers_['numeric'].named_steps['impute']
        before = imputer.statistics_.copy()
        b = a.copy(); b['Age'] = 100
        pipe.predict_proba(b)
        np.testing.assert_array_equal(before, imputer.statistics_)
        self.assertEqual(before[0], 33.5)
        self.assertFalse(set(FEATURES) & {'Final','Serial','Hospital','Platelet2','PCR','MAT'})

    def test_focal_extra_trees_and_comparison_models_are_declared(self):
        recipes = candidates()
        expected = {'Logistic regression', 'Decision tree', 'Random forest', 'SVM', 'KNN',
                    'XGBoost', 'AdaBoost', 'Gradient boosting', 'LightGBM', 'CatBoost', 'Extra trees'}
        self.assertEqual({spec['model'] for spec in recipes}, expected)
        for name in expected:
            self.assertEqual({s['feature_set'] for s in recipes if s['model'] == name}, {'base', 'engineered'})
        for spec in recipes:
            self.assertEqual(list(spec['estimator'].named_steps), ['features','prepare','model'])
            self.assertFalse(hasattr(spec['estimator'].named_steps['prepare'], 'transformers_'))
        focal = [s for s in recipes if s['role'] == 'Focal algorithm']
        self.assertEqual({s['model'] for s in focal}, {FOCAL_MODEL})
        self.assertEqual(len(focal), 2)

    def test_explanation_reconstructs_synthetic_prediction(self):
        records = pd.concat([validate_record({'Age':25 + i, 'PRad':65 + i, 'Feverad':1 if i % 2 else 2})
                             for i in range(12)], ignore_index=True)
        model = pipeline(LogisticRegression(), False).fit(records, [0,1] * 6)
        reference = {c: float(records[c].median()) if c in NUMERIC and records[c].notna().any() else None
                     for c in FEATURES}
        base, prediction, local = local_shapley(model, records.iloc[[3]], reference, permutations=3)
        self.assertTrue(np.isfinite(prediction))
        self.assertAlmostEqual(base + local.contribution.sum(), prediction)
        self.assertAlmostEqual(prediction, model.predict_proba(records.iloc[[3]])[0,1])

    def test_selection_ignores_test_scores_and_uses_declared_ties(self):
        from train import ranked
        rows = pd.DataFrame([
            dict(model='A', feature_set='base', candidate=0, mean_roc_auc=.60, mean_average_precision=.9, test_auc=.99),
            dict(model='B', feature_set='base', candidate=0, mean_roc_auc=.70, mean_average_precision=.5, test_auc=.50),
            dict(model='C', feature_set='base', candidate=0, mean_roc_auc=.70, mean_average_precision=.6, test_auc=.40),
        ])
        self.assertEqual(ranked(rows).model.tolist(), ['C','B','A'])
        rows['test_auc'] = [0, 1, 1]
        self.assertEqual(ranked(rows).model.tolist(), ['C','B','A'])

@unittest.skipUnless((OUTPUT / 'artifacts/manifest.json').is_file(), 'Run training locally to validate generated outputs.')
class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model, cls.reference, cls.manifest = load_release()

    def test_selection_uses_development_only(self):
        from train import ranked
        frame = ranked(pd.read_csv(OUTPUT / 'reports/cv_candidates.csv'))
        selected = json.loads((OUTPUT / 'reports/selection.json').read_text())
        self.assertIn(selected['selected_model'], {spec['model'] for spec in candidates()})
        self.assertEqual(set(frame.model), {spec['model'] for spec in candidates()})
        self.assertEqual(selected['focal_model'], FOCAL_MODEL)
        self.assertFalse(selected['test_used_for_selection'])
        self.assertEqual(int(frame.iloc[0].spec_id), selected['spec_id'])
        self.assertEqual(frame.iloc[0].model, selected['selected_model'])
        self.assertEqual(frame.iloc[0].feature_set, selected['feature_set'])

    def test_prediction_and_explanation_consistency(self):
        row = validate_record({'Age':35,'Feverad':1})
        p = self.model.predict_proba(row)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1)
        base, prediction, local = local_shapley(self.model, row, self.reference, permutations=3)
        self.assertAlmostEqual(prediction, p[0,1])
        self.assertAlmostEqual(base + local.contribution.sum(), prediction)
        self.assertEqual(set(local.feature), set(FEATURES))

    def test_artifact_contains_estimator_only(self):
        self.assertTrue(hasattr(self.model, 'predict_proba'))
        self.assertNotIsInstance(self.model, dict)
        self.assertFalse(hasattr(self.model, 'X_train'))
        self.assertEqual(len(self.reference), len(FEATURES))

if __name__ == '__main__': unittest.main(verbosity=2)
