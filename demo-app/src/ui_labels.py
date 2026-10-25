"""Readable input labels, separate from the fitted model's source-code contract.

Sex and exposure choices follow Questionnaire.pdf, pages 1, 5 and 7.
Symptom choices provisionally assume 1=Yes and 2=No for this binary-input demo,
as authorized by the user. This is not a verified CSV-codebook interpretation:
the supplied symptom checklist uses severity levels. Do not infer target labels.
"""
from .data import EXPOSURES, SYMPTOMS

FIELD_LABELS = {
    'Sex': 'Sex',
    'Feverad': 'Fever',
    'Headachead': 'Headache',
    'Chillsad': 'Chills',
    'Rigorsad': 'Shaking chills (rigors)',
    'Musclepainad': 'Muscle pain',
    'Muscletendernessad': 'Muscle tenderness',
    'Nauseaad': 'Nausea',
    'Vomitingadmission': 'Vomiting',
    'Cnsuffusionad': 'Conjunctival suffusion',
    'Skinrashad': 'Skin rash',
    'Mucosalrashad': 'Mucosal rash',
    'Prostrationad': 'Extreme weakness (prostration)',
    'Diarrhoeaad': 'Diarrhoea',
    'OliguriaAd': 'Reduced urine output (oliguria)',
    'Anuriaad': 'No urine output (anuria)',
    'Jaundicead': 'Jaundice',
    'Hepatictendernessad': 'Liver tenderness',
    'Hepatomegalyad': 'Enlarged liver',
    'Spleenimegalyad': 'Enlarged spleen',
    'Lympadenopathyad': 'Enlarged lymph nodes',
    'Photophobiaad': 'Sensitivity to light',
    'Neckstiffnessad': 'Neck stiffness',
    'Psychoticsymptomsad': 'Psychotic symptoms',
    'Confusionad': 'Confusion',
    'Coughad': 'Cough',
    'Haemoptasisad': 'Coughing up blood (haemoptysis)',
    'SOBadd': 'Shortness of breath',
    'Chestpainad': 'Chest pain',
    'Bleedingad': 'Bleeding',
    'Marshlandexposure': 'Marshland exposure',
    'Wetsoilexposure': 'Contact with wet soil',
    'Floodexposure': 'Flood exposure',
    'Forestexposure': 'Forest exposure',
    'BushesExposure': 'Bush exposure',
    'Otheragriexposure': 'Other agricultural land exposure',
    'Gardeningexposure': 'Gardening',
    'Otheroutdoorworkexposure': 'Other outdoor work',
    'Naturalrunningwaterexposure': 'Natural running water exposure',
    'Stagnantwaterexposure': 'Stagnant water exposure',
    'Manmaderunningwaterexposure': 'Man-made running water exposure',
    'Drainsexposure': 'Drain exposure',
    'Otherwaterexposure': 'Other water exposure',
    'Paddyfieldexposure': 'Paddy field exposure',
    'CattleHandle': 'Handling cattle',
    'BuffaloHandle': 'Handling buffalo',
    'Goathandle': 'Handling goats',
    'Pighandle': 'Handling pigs',
    'Cathandle': 'Handling cats',
    'Doghandle': 'Handling dogs',
}

CATEGORY_OPTIONS = ['1', '2', 'Missing']
SEX_LABELS = {'1': 'Male', '2': 'Female', 'Missing': 'Not recorded'}
EXPOSURE_LABELS = {'1': 'Yes', '2': 'No', 'Missing': 'Not recorded'}
# Display assumption only. Never remap or reinterpret the fitted feature values.
SYMPTOM_LABELS = {'1': 'Yes', '2': 'No', 'Missing': 'Not recorded'}


def option_label(field, value):
    if field == 'Sex':
        return SEX_LABELS[value]
    if field in EXPOSURES:
        return EXPOSURE_LABELS[value]
    if field in SYMPTOMS:
        return SYMPTOM_LABELS[value]
    raise KeyError(f'No display labels declared for {field}')
