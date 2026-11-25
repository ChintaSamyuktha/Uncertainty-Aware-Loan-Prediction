"""Raw-feature recourse constraints, bootstrap uncertainty and local stability."""
import numpy as np
import pandas as pd


def complete(frame, reference):
    out = frame.copy()
    for col in out:
        value = reference[col].median() if pd.api.types.is_numeric_dtype(reference[col]) else reference[col].mode().iloc[0]
        out[col] = out[col].fillna(value)
    return out


def bounds(reference, config):
    return {c: [float(reference[c].min()), float(reference[c].max())] for c in config['mutable']}


def actionable(original, candidate, reference, config):
    limits = bounds(reference, config)
    for c in original.index:
        if c not in config['mutable']:
            if candidate[c] != original[c]:
                return False
        else:
            if not np.isfinite(float(candidate[c])):
                return False
            lo, hi = limits[c]
            # Existing outliers are preserved; changes must lie in training bounds.
            if candidate[c] != original[c] and not lo <= float(candidate[c]) <= hi:
                return False
            direction = config['mutable'][c]['direction']
            if direction == 'increase' and candidate[c] < original[c]:
                return False
            if direction == 'decrease' and candidate[c] > original[c]:
                return False
    return True


def perturb(original, candidate, reference, config, seed):
    rng = np.random.default_rng(seed)
    out = pd.DataFrame([candidate.to_dict()] * config['perturbations'])
    for c, policy in config['mutable'].items():
        lo, hi = bounds(reference, config)[c]
        if policy['direction'] == 'increase':
            lo = max(lo, float(original[c]))
        elif policy['direction'] == 'decrease':
            hi = min(hi, float(original[c]))
        if lo > hi:
            out[c] = candidate[c]
        else:
            scale = config['perturbation_fraction'] * (reference[c].quantile(.95) - reference[c].quantile(.05))
            out[c] = np.clip(float(candidate[c]) + rng.normal(0, scale, len(out)), lo, hi)
    return out


def stability(predictions):
    predictions = np.asarray(predictions)
    if predictions.size == 0:
        raise ValueError('Stability requires at least one perturbation')
    return float(np.mean(predictions == 1))


def assess(original, candidate, model, ensemble, reference, config, seed=42):
    frame = pd.DataFrame([candidate.to_dict()])[reference.columns]
    probability = float(model.predict_proba(frame)[0, 1])
    uncertainty = float(np.std([m.predict_proba(frame)[0, 1] for m in ensemble], ddof=1))
    distance = sum(config['mutable'][c]['cost'] * abs(float(candidate[c]) - float(original[c])) / max(float(reference[c].max()-reference[c].min()), 1) for c in config['mutable'])
    sparse = sum(candidate[c] != original[c] for c in original.index)
    valid = bool(model.predict(frame)[0] == 1)
    feasible = actionable(original, candidate, reference, config)
    nearby = perturb(original, candidate, reference, config, seed)
    nearby['prediction'] = model.predict(nearby[reference.columns])
    stable = stability(nearby.prediction)
    rule = config['reliability']
    reliable = valid and feasible and uncertainty <= rule['max_uncertainty'] and stable >= rule['min_stability'] and distance <= rule['max_proximity'] and sparse <= rule['max_sparsity']
    return dict(probability=probability, uncertainty=uncertainty, proximity=distance, sparsity=int(sparse), validity=valid, actionability=feasible, stability=stable, reliable=bool(reliable)), nearby


def dice_generator(reference, y, model):
    import dice_ml
    data = reference.copy()
    data['outcome'] = np.asarray(y, dtype=int)
    continuous = list(reference.select_dtypes(include='number').columns)
    return dice_ml.Dice(dice_ml.Data(dataframe=data, continuous_features=continuous, outcome_name='outcome'), dice_ml.Model(model=model, backend='sklearn'), method='random')


def generate(generator, original, reference, config, seed):
    ranges = bounds(reference, config)
    for c, policy in config['mutable'].items():
        if policy['direction'] == 'increase':
            ranges[c][0] = max(ranges[c][0], float(original[c]))
        elif policy['direction'] == 'decrease':
            ranges[c][1] = min(ranges[c][1], float(original[c]))
    if any(lo > hi for lo, hi in ranges.values()):
        return pd.DataFrame(), 'Original outside feasible training bounds'
    try:
        result = generator.generate_counterfactuals(pd.DataFrame([original]), total_CFs=config['candidate_count'], desired_class=1, features_to_vary=list(config['mutable']), permitted_range=ranges, sample_size=4000, random_seed=seed, verbose=False)
        candidates = result.cf_examples_list[0].final_cfs_df
        if candidates is None:
            return pd.DataFrame(), 'DiCE found no candidate'
        return candidates[reference.columns].drop_duplicates(), ''
    except Exception as exc:
        # Fail visibly, preserving other evaluation cases; never substitute fake CFs.
        return pd.DataFrame(), f'{type(exc).__name__}: {exc}'
