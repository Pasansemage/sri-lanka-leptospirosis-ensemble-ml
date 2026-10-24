"""Public academic demo. streamlit run app.py"""
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from src.data import ROOT, FEATURES, NUMERIC, SYMPTOMS, EXPOSURES, BOUNDS, NOTICE, validate_record
from src.explain import local_shapley
from src.runtime import load_release

st.set_page_config(page_title='Sri Lanka Leptospirosis ML | Ensemble ML', page_icon='◈', layout='wide')
st.markdown('''<style>
.stApp {background:#f6f8fb;color:#172a3a}
.block-container {max-width:1280px;padding-top:2.5rem}
h1 {letter-spacing:-.04em;color:#123b4a} h2,h3 {color:#164e5b}
[data-testid="stMetric"] {background:white;border:1px solid #dbe4eb;border-radius:12px;padding:16px}
[data-testid="stSidebar"] {background:#eaf1f5}
</style>''', unsafe_allow_html=True)
st.caption('APPLIED MACHINE LEARNING  /  SRI LANKAN FEBRILE-PATIENT STUDY')
st.title('Sri Lanka Leptospirosis ML')
st.write('Compare ensembles. Inspect the evidence. Explore a prediction.')
st.warning(NOTICE)
if not (ROOT / 'artifacts/manifest.json').is_file():
    st.error('No trained release found. Run python train.py before starting the app.')
    st.stop()

@st.cache_resource
def release(fingerprint):
    return load_release()

def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

try:
    model, reference, manifest = release((ROOT / 'artifacts/manifest.json').read_text())
except (ValueError, RuntimeError, OSError) as error:
    st.error(str(error)); st.stop()
selection = read_json('reports/selection.json')
test = read_json('reports/test_results.json')
with st.sidebar:
    st.subheader('The deployed model')
    st.write(f"**{selection['selected_model']}**")
    st.caption(f"{selection['family']} · {selection['feature_set']} features")
    st.code(f"version {manifest['model_version']}", language=None)
    st.write(f"{test['development_n']:,} development records · {test['test_n']:,} internal holdout records")
    st.caption('One patient per row. Source Final code 1 is the numerical positive class.')
    st.caption('Inputs remain in this session. The app does not persist or log patient inputs. Use hypothetical examples.')

cards = st.columns(4)
for col, label, value in zip(cards,
    ['Development ROC AUC', 'Internal holdout ROC AUC', 'Internal holdout average precision', 'Holdout code-1 prevalence'],
    [selection['cv_roc_auc'], test['metrics']['roc_auc'], test['metrics']['average_precision'], test['test_code1_prevalence']]):
    col.metric(label, f'{value:.3f}')
predict_tab, comparison_tab, explanation_tab, ops_tab = st.tabs(['Live prediction', 'Model evidence', 'Explainability', 'MLOps & provenance'])

with predict_tab:
    st.subheader('Try a hypothetical admission record')
    st.write('Select source codes exactly as recorded: 1, 2, or Missing. Their clinical meanings have not been verified. Blank numeric inputs remain missing.')
    with st.form('inference'):
        values = {}
        columns = st.columns(4)
        labels = {'Age':'Age (source years)', 'PRad':'Admission pulse (source units)', 'SBPadd':'Admission SBP (source units)', 'DBPadd':'Admission DBP (source units)'}
        for col, name in zip(columns, NUMERIC):
            with col:
                values[name] = st.number_input(labels[name], min_value=float(BOUNDS[name][0]), max_value=float(BOUNDS[name][1]),
                    value=35.0 if name == 'Age' else None, step=1.0, key=name, help='99 is provisionally treated as missing, consistently with training.')
        values['Sex'] = st.selectbox('Sex (source code)', ['Missing','1','2'])
        st.markdown('**Admission symptoms**')
        common = SYMPTOMS[:8]
        for i, name in enumerate(common):
            with columns[i % 4]:
                values[name] = st.selectbox(name, ['Missing','1','2'], index=1 if name == 'Feverad' else 0, key=name)
        with st.expander('Additional admission symptoms'):
            extra_cols = st.columns(3)
            for i, name in enumerate(SYMPTOMS[8:]):
                with extra_cols[i % 3]: values[name] = st.selectbox(name, ['Missing','1','2'], key=name)
        with st.expander('Exposure history'):
            exp_cols = st.columns(3)
            for i, name in enumerate(EXPOSURES):
                with exp_cols[i % 3]: values[name] = st.selectbox(name, ['Missing','1','2'], key=name)
        submitted = st.form_submit_button('Run inference', type='primary')
    if submitted:
        try:
            cleaned = validate_record({k: None if v == 'Missing' else v for k,v in values.items()})
            before = time.perf_counter()
            probability = float(model.predict_proba(cleaned)[0,1])
            st.session_state['prediction'] = dict(row=cleaned, probability=probability, latency_ms=(time.perf_counter()-before)*1000)
            st.session_state.pop('local_explanation', None)
        except ValueError as error:
            st.session_state.pop('prediction', None)
            st.error(str(error))
    if 'prediction' in st.session_state:
        result = st.session_state.prediction
        st.caption('Result for the last submitted record. Submit again after editing inputs.')
        threshold = st.slider('Illustrative inference threshold', .05, .95, .5, .05,
            help='Changes this example only. Published evaluation always uses the fixed 0.50 threshold.')
        out = st.columns(3)
        out[0].metric('Uncalibrated code-1 score', f"{result['probability']:.1%}")
        out[1].metric('Predicted source Final code', '1' if result['probability'] >= threshold else '2')
        out[2].metric('Model inference time', f"{result['latency_ms']:.1f} ms")
        missing = int(result['row'].isna().sum(axis=1).iloc[0])
        st.caption(f'{len(FEATURES)-missing}/{len(FEATURES)} original inputs recorded. Missingness can strongly affect the output.')
        if st.button('Explain this prediction'):
            with st.spinner('Calculating baseline-based Shapley contributions…'):
                st.session_state['local_explanation'] = local_shapley(model, result['row'], reference)
        if 'local_explanation' in st.session_state:
            base, probability, local = st.session_state.local_explanation
            st.write(f'Baseline {base:.4f} + contributions {local.contribution.sum():+.4f} = prediction {probability:.4f}')
            st.bar_chart(local.head(12).set_index('feature')['contribution'], horizontal=True, color='#087f8c')
            st.caption('Positive contributions move toward source code 1 relative to a synthetic median/missing reference. These are approximate permutation Shapley values, not causal effects or clinical advice. The Explainability tab describes the method.')
            st.dataframe(local, hide_index=True)

with comparison_tab:
    st.subheader('Why this model was chosen')
    st.write(f"**{selection['selected_model']} ({selection['feature_set']})** had the highest mean development ROC AUC: **{selection['cv_roc_auc']:.4f}**. "
             'All candidates used the same five stratified development folds. Average precision breaks exact AUC ties. The holdout did not select the winner.')
    st.info(selection['validation_status'])
    leaderboard = pd.read_csv(ROOT / 'reports/cv_leaderboard.csv')
    family = st.multiselect('Compare model families', ['Baseline','Bagging','Boosting','Stacking'], default=['Baseline','Bagging','Boosting','Stacking'])
    st.dataframe(leaderboard.loc[leaderboard.family.isin(family), ['model','family','feature_set','mean_roc_auc','std_roc_auc','mean_average_precision','mean_f1','mean_recall','mean_brier','mean_train_roc_auc','mean_fit_predict_seconds']], hide_index=True)
    st.image(str(ROOT / 'reports/comparison.png'))
    st.caption('Each row is the best development configuration for that model. Fold standard deviations describe variability; they are not confidence intervals or significance tests.')
    st.subheader('Selected model: internal holdout')
    rows = [{'metric':k,'selected_model':v,'prior_baseline':test['baseline_metrics'][k],
             'lower_95_percent':test['confidence_intervals'][k][0], 'upper_95_percent':test['confidence_intervals'][k][1]} for k,v in test['metrics'].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(test['confidence_interval_method'] + '. Precision, recall and F1 use Final code 1 as positive and threshold 0.50. Brier and log loss are lower-is-better.')
    st.dataframe(pd.DataFrame(test['confusion_matrix'], index=['Actual code 2','Actual code 1'], columns=['Predicted code 2','Predicted code 1']))
    st.image(str(ROOT / 'reports/evaluation.png'))
    with st.expander('Feature engineering evidence'):
        st.markdown((ROOT / 'reports/FEATURE_ENGINEERING.md').read_text(encoding='utf-8'))
        st.dataframe(pd.read_csv(ROOT / 'reports/feature_ablation.csv'), hide_index=True)
    st.download_button('Download development comparison', (ROOT / 'reports/cv_candidates.csv').read_bytes(), 'cv_candidates.csv', 'text/csv')

with explanation_tab:
    st.subheader('Explain the selected model')
    st.write('Global permutation importance shuffles each original input on the holdout and measures the decrease in ROC AUC. Every shuffle reruns the full pipeline, including derived features. Ten shuffles quantify sensitivity to the random permutation.')
    st.image(str(ROOT / 'reports/importance.png'))
    st.caption('Negative or near-zero importance is possible. Correlated inputs can mask one another. These explanations were generated after selection and did not drive feature removal.')
    st.subheader('How an individual explanation works')
    st.write('The live prediction uses 24 random feature orders. Along each order, inputs are changed from a synthetic reference to the submitted record. The average marginal changes give Monte Carlo Shapley contributions in probability units. The baseline plus all contributions reconstructs the prediction.')
    st.write('The reference uses development medians for numeric inputs and Missing for all source categories. This is a reference-dependent substitution game, not conditional population SHAP. Substitutions can create uncommon combinations. Contributions explain model behaviour and do not establish causes.')
    response = pd.read_csv(ROOT / 'reports/synthetic_age_response.csv')
    st.line_chart(response.set_index('Age'), color='#087f8c')
    st.caption('Synthetic age response: Feverad=1, other categorical inputs missing, numeric inputs fixed to development medians. This is one fixed context, not a population effect or a real patient.')

with ops_tab:
    st.subheader('A reproducible release')
    st.write('Train locally → compare development folds → freeze selection → evaluate → explain → checksum the fitted artifact → test → publish → deploy.')
    st.json({k:manifest[k] for k in ['model_version','created_utc','model','feature_set','source_data_sha256','python','packages']})
    st.write('The app verifies model and report hashes and package versions before loading. It runs inference from the saved model; it does not retrain at startup or require the original patient CSV.')
    st.write('GitHub Actions checks the data contract, model integrity, explanations and interactive app. Streamlit Community Cloud can host the public demo for free; deployments follow the chosen GitHub branch. Free apps may sleep when idle.')
    st.caption('CI configuration is included. Remote checks and public deployment status must be verified after publishing. Model retraining and promotion are manual; live drift/outcome monitoring is future work.')
    st.markdown('[Dataset attribution: Suneth Agampodi, Mendeley Data v4, CC BY 4.0](https://data.mendeley.com/datasets/zmxmp42g9p/4) · [Free hosting documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud)')
