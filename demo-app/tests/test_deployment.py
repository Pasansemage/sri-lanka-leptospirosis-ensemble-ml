"""Check a packaged serving release using hypothetical inputs only."""
import json
import unittest
import numpy as np
from src.data import ROOT, CATEGORICAL, validate_record
from src.explain import local_shapley
from src.runtime import load_release


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.reference, cls.manifest = load_release()

    def test_release_and_reference_prediction(self):
        self.assertFalse((ROOT / 'Leptospirosis clinical data.csv').exists())
        self.assertFalse((ROOT / 'data').exists())
        expected = json.loads((ROOT / 'reports/explainability.json').read_text(encoding='utf-8'))
        row = validate_record({'Age': 35, 'Feverad': 1})
        probability = self.model.predict_proba(row)
        self.assertTrue(np.isfinite(probability).all())
        self.assertAlmostEqual(float(probability[0, 1]), expected['example_probability'])

    def test_input_contract(self):
        for record in [{'Age': 35}, {'Age': -1, 'Feverad': 1}, {'Sex': 3, 'Feverad': 1}]:
            with self.assertRaises(ValueError):
                validate_record(record)

    def test_comparison_includes_focal_extra_trees(self):
        import pandas as pd
        expected = {'Logistic regression', 'Decision tree', 'Random forest', 'SVM', 'KNN',
                    'XGBoost', 'AdaBoost', 'Gradient boosting', 'LightGBM', 'CatBoost', 'Extra trees'}
        leaderboard = pd.read_csv(ROOT / 'reports/cv_leaderboard.csv')
        self.assertEqual(set(leaderboard.model), expected)
        self.assertIn(self.manifest['model'], expected)
        self.assertEqual(set(leaderboard.loc[leaderboard.role == 'Focal algorithm', 'model']), {'Extra trees'})

    def test_explanation_reconstructs_prediction(self):
        base, predicted, local = local_shapley(
            self.model, validate_record({'Age': 35, 'Feverad': 1}), self.reference, permutations=3)
        self.assertAlmostEqual(base + local.contribution.sum(), predicted)

    def test_interactive_inference_and_explanation(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertEqual(set(app.multiselect[0].options),
                         {'Linear', 'Tree', 'Bagging', 'Kernel', 'Neighbors', 'Boosting', 'Randomized trees'})
        self.assertEqual(len(app.selectbox), len(CATEGORICAL))
        self.assertTrue(all(widget.value == '1' for widget in app.selectbox))
        self.assertEqual(app.selectbox(key='Sex').options, ['Male', 'Female', 'Not recorded'])
        self.assertEqual(app.selectbox(key='Feverad').label, 'Fever')
        for widget in app.selectbox:
            if widget.key != 'Sex':
                self.assertEqual(widget.options, ['Yes', 'No', 'Not recorded'])
        self.assertEqual(app.selectbox(key='Wetsoilexposure').label, 'Contact with wet soil')
        self.assertEqual(app.selectbox(key='Marshlandexposure').options, ['Yes', 'No', 'Not recorded'])
        app.selectbox(key='Sex').select('2')
        app.selectbox(key='Feverad').select('2')
        app.selectbox(key='Marshlandexposure').select('2')
        app.selectbox(key='Wetsoilexposure').select('Missing')
        next(button for button in app.button if button.label == 'Run inference').click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertIn('prediction', app.session_state)
        record = app.session_state['prediction']['row']
        self.assertEqual(record['Sex'].iloc[0], '2')
        self.assertEqual(record['Feverad'].iloc[0], '2')
        self.assertEqual(record['Marshlandexposure'].iloc[0], '2')
        self.assertTrue(record['Wetsoilexposure'].isna().iloc[0])
        next(button for button in app.button if button.label == 'Explain this prediction').click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertIn('local_explanation', app.session_state)
        next(button for button in app.button if button.label == 'Load reference example').click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertEqual(app.selectbox(key='Feverad').value, '1')
        self.assertTrue(all(widget.value == 'Missing' for widget in app.selectbox if widget.key != 'Feverad'))
        next(button for button in app.button if button.label == 'Run inference').click().run()
        self.assertFalse(app.exception, str(app.exception))
        expected = json.loads((ROOT / 'reports/explainability.json').read_text(encoding='utf-8'))
        self.assertAlmostEqual(app.session_state['prediction']['probability'], expected['example_probability'])


if __name__ == '__main__':
    unittest.main()
