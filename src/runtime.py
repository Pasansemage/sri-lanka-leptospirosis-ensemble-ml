"""Minimal serving loader with model/report integrity and package compatibility checks."""
import importlib.metadata
import json
from pathlib import Path
import joblib
from .data import ROOT, sha256

def load_release(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'artifacts/manifest.json').read_text(encoding='utf-8'))
    for name, expected in {**manifest['artifacts'], **manifest['report_sha256']}.items():
        if sha256(root / name) != expected:
            raise ValueError(f'Artifact integrity check failed: {name}')
    for package in ['scikit-learn', 'numpy', 'pandas', 'xgboost', 'lightgbm', 'catboost']:
        actual = importlib.metadata.version(package)
        if actual != manifest['packages'][package]:
            raise RuntimeError(f'Package mismatch: {package} {actual}; expected {manifest["packages"][package]}. Install requirements.txt.')
    # This local, checksum-verified artifact is trusted project code. Never accept uploaded pickle files.
    model = joblib.load(root / 'artifacts/model.joblib')
    reference = json.loads((root / 'artifacts/reference.json').read_text(encoding='utf-8'))
    return model, reference, manifest
