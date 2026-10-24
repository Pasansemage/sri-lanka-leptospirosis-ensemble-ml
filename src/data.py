"""Explicit, provisional source-code contract. No diagnostic semantics inferred."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'Leptospirosis clinical data.csv'
OUTPUT = ROOT / 'outputs'
SEED = 42
SYMPTOMS = ['Feverad', 'Headachead', 'Chillsad', 'Rigorsad', 'Musclepainad', 'Muscletendernessad', 'Nauseaad', 'Vomitingadmission', 'Cnsuffusionad', 'Skinrashad', 'Mucosalrashad', 'Prostrationad', 'Diarrhoeaad', 'OliguriaAd', 'Anuriaad', 'Jaundicead', 'Hepatictendernessad', 'Hepatomegalyad', 'Spleenimegalyad', 'Lympadenopathyad', 'Photophobiaad', 'Neckstiffnessad', 'Psychoticsymptomsad', 'Confusionad', 'Coughad', 'Haemoptasisad', 'SOBadd', 'Chestpainad', 'Bleedingad']
EXPOSURES = ['Marshlandexposure', 'Wetsoilexposure', 'Floodexposure', 'Forestexposure', 'BushesExposure', 'Otheragriexposure', 'Gardeningexposure', 'Otheroutdoorworkexposure', 'Naturalrunningwaterexposure', 'Stagnantwaterexposure', 'Manmaderunningwaterexposure', 'Drainsexposure', 'Otherwaterexposure', 'Paddyfieldexposure', 'CattleHandle', 'BuffaloHandle', 'Goathandle', 'Pighandle', 'Cathandle', 'Doghandle']
NUMERIC = ['Age', 'PRad', 'SBPadd', 'DBPadd']
CATEGORICAL = ['Sex'] + SYMPTOMS + EXPOSURES
FEATURES = NUMERIC + CATEGORICAL
DERIVED = ['pulse_pressure', 'mean_arterial_pressure', 'symptoms_recorded_fraction', 'exposures_recorded_fraction']
BOUNDS = {'Age': (0, 110), 'PRad': (20, 250), 'SBPadd': (40, 300), 'DBPadd': (20, 200)}
NOTICE = ('Research demonstration only. Final code 1 and code 2 have unverified clinical meanings. '
          'The 99 missing-value convention and measurement units are provisional. '
          'Scores are uncalibrated model outputs, not clinical risks.')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean_features(raw):
    """Deterministic cleaning, shared by training and inference."""
    missing = set(FEATURES) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing predictors: {sorted(missing)}')
    X = pd.DataFrame(index=raw.index)
    for c in FEATURES:
        v = pd.to_numeric(raw[c], errors='coerce').replace(99, np.nan)
        if c in CATEGORICAL:
            X[c] = v.where(v.isin([1, 2])).map(lambda x: str(int(x)) if pd.notna(x) else np.nan).astype(object)
        else:
            lo, hi = BOUNDS[c]
            X[c] = v.where(v.between(lo, hi)).astype(float)
    return X

def load_data(path=DATA):
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {'Serial', 'Final', *FEATURES}.issubset(raw.columns):
        raise ValueError('CSV does not satisfy the required source schema.')
    if raw.Serial.str.strip().eq('').any() or raw.Serial.str.strip().duplicated().any():
        raise ValueError('Expected one nonempty, unique Serial per patient.')
    X = clean_features(raw)
    outcome = pd.to_numeric(raw.Final, errors='coerce')
    eligible = outcome.isin([1, 2]) & X[SYMPTOMS].notna().any(axis=1)
    return raw, X.loc[eligible], (outcome.loc[eligible] == 1).astype(int)

def validate_record(values):
    if set(values) - set(FEATURES):
        raise ValueError('Unexpected predictor names.')
    for c, v in values.items():
        if v is None or pd.isna(v) or v == '':
            continue
        try:
            number = float(v)
        except (TypeError, ValueError):
            raise ValueError(f'{c}: enter a number or leave missing.') from None
        if number == 99:
            continue
        if not np.isfinite(number):
            raise ValueError(f'{c}: enter a finite number.')
        if c in CATEGORICAL and number not in (1, 2):
            raise ValueError(f'{c}: only source codes 1, 2 or missing are supported.')
        if c in NUMERIC and not BOUNDS[c][0] <= number <= BOUNDS[c][1]:
            raise ValueError(f'{c}: outside the project bounds {BOUNDS[c]}.')
    X = clean_features(pd.DataFrame([{c: values.get(c, np.nan) for c in FEATURES}]))
    if not X[SYMPTOMS].notna().any(axis=1).iloc[0]:
        raise ValueError('Enter at least one admission symptom source code, as required by the training cohort.')
    if X.SBPadd.notna().iloc[0] and X.DBPadd.notna().iloc[0] and X.SBPadd.iloc[0] < X.DBPadd.iloc[0]:
        raise ValueError('SBP must be at least DBP when both are supplied.')
    return X

class AdmissionFeatures(TransformerMixin, BaseEstimator):
    """No fitted statistics; derived values calculated before fold-local imputation."""
    def __init__(self, engineered=False):
        self.engineered = engineered

    def fit(self, X, y=None):
        self.n_features_in_ = len(FEATURES)
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        return self

    def transform(self, X):
        out = X[FEATURES].copy()
        if self.engineered:
            valid = out.SBPadd.notna() & out.DBPadd.notna() & (out.SBPadd >= out.DBPadd)
            out['pulse_pressure'] = (out.SBPadd - out.DBPadd).where(valid)
            out['mean_arterial_pressure'] = (out.DBPadd + out['pulse_pressure'] / 3).where(valid)
            out['symptoms_recorded_fraction'] = out[SYMPTOMS].notna().mean(axis=1)
            out['exposures_recorded_fraction'] = out[EXPOSURES].notna().mean(axis=1)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES + (DERIVED if self.engineered else []), dtype=object)
