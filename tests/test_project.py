"""Material checks for serving parity, leakage boundaries, release integrity, and app behaviour."""
import json
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.data import FEATURES, NUMERIC, SYMPTOMS, ROOT, AdmissionFeatures, validate_record
from src.models import pipeline, candidates
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

    def test_stack_inner_folds_include_preprocessing(self):
        stacks = [c for c in candidates() if c['family'] == 'Stacking']
        self.assertEqual(len(stacks), 4)
        for spec in stacks:
            self.assertEqual(spec['estimator'].cv.n_splits, 3)
            for _, base in spec['estimator'].estimators:
                self.assertEqual(list(base.named_steps), ['features','prepare','model'])
                self.assertFalse(hasattr(base.named_steps['prepare'], 'transformers_'))

class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model, cls.reference, cls.manifest = load_release()

    def test_selection_uses_development_only(self):
        from train import ranked
        frame = ranked(pd.read_csv(ROOT / 'reports/cv_candidates.csv'))
        selected = json.loads((ROOT / 'reports/selection.json').read_text())
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

    def test_streamlit_inference_and_explanation(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
        self.assertFalse(app.exception, str(app.exception))
        next(b for b in app.button if b.label == 'Run inference').click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertIn('prediction', app.session_state)
        next(b for b in app.button if b.label == 'Explain this prediction').click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertIn('local_explanation', app.session_state)
        next(s for s in app.slider if s.label == 'Illustrative inference threshold').set_value(.75).run()
        self.assertFalse(app.exception, str(app.exception))

if __name__ == '__main__': unittest.main(verbosity=2)
