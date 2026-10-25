"""Explanations of the deployed predictor in original feature space."""
import numpy as np
import pandas as pd
from .data import FEATURES, SEED

def local_shapley(model, row, reference, permutations=24, seed=SEED):
    """Monte Carlo permutation Shapley against ONE synthetic reference, probability units.

    Paths telescope exactly; finite-path estimates are order-sampling approximations.
    This is a baseline substitution game, not conditional population SHAP.
    """
    row = row[FEATURES].iloc[0]
    ref = pd.Series(reference, dtype=object)[FEATURES]
    rng = np.random.default_rng(seed)
    paths, orders = [], []
    for _ in range(permutations):
        order = rng.permutation(len(FEATURES))
        current = ref.copy()
        paths.append(current.copy())
        for j in order:
            current.iloc[j] = row.iloc[j]
            paths.append(current.copy())
        orders.append(order)
    frame = pd.DataFrame(paths, columns=FEATURES)
    for c in FEATURES[:4]:
        frame[c] = pd.to_numeric(frame[c], errors='coerce').astype(float)
    frame = frame.where(frame.notna(), np.nan)
    p = model.predict_proba(frame)[:, 1].reshape(permutations, len(FEATURES) + 1)
    contributions = np.zeros((permutations, len(FEATURES)))
    for k, order in enumerate(orders):
        contributions[k, order] = np.diff(p[k])
    result = pd.DataFrame({'feature': FEATURES, 'contribution': contributions.mean(axis=0),
                           'order_sampling_std': contributions.std(axis=0)})
    base, prediction = float(p[0, 0]), float(p[0, -1])
    if not np.isclose(base + result.contribution.sum(), prediction, atol=1e-8):
        raise RuntimeError('Explanation does not reconstruct model output.')
    return base, prediction, result.sort_values('contribution', key=abs, ascending=False)
