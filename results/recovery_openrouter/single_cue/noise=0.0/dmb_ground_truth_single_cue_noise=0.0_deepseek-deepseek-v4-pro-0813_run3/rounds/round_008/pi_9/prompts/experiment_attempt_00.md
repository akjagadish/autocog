# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** People make binary-feature choices by a primary unweighted signed tally. When the tally is clear, choice follows the tally stochastically with a small lapse. When the tally is tied or near-tied, people parse the validity display into a single contiguous terminal perceptual chunk only when the validity sequence contains a structural boundary. In addition to large jumps, moderate positive and negative end discontinuities, and cumulative short-display contrasts, short displays have a fallback parsing rule: if no other boundary rule has opened a chunk, the largest adjacent validity discontinuity defines the terminal chunk. A targeted forced late-pool rule is added for very short displays: when a flat high-validity early triplet is followed by a lower, homogeneous late triplet, that late block is opened with bounded decisiveness, producing recency-weighted choices against the early high-validity pool. Recency inside any chunk remains a weak linear ramp with a cap, not an exponential validity-times-position re-ranking, and large displays with small tallies remain low-confidence.

**Parameters:**
- beta_tally_small: [0.50, 0.75]
- beta_tally_large: [0.85, 1.25]
- epsilon_tally_small: [0.02, 0.06]
- epsilon_tally_large: [0.02, 0.06]
- beta_near: [0.18, 0.38]
- epsilon_near: [0.75, 0.95]
- near_bound: {1.5}
- n_low_conf: {15}
- tau_tie: {0.5}
- gap_hi: [0.34, 0.45]
- gap_lo: [0.06, 0.10]
- gap_pos_max: [0.16, 0.26]
- drop_lo: [0.10, 0.14]
- drop_hi: [0.22, 0.45]
- flat_tol: [0.12, 0.20]
- block_max_large: {6}
- block_max_small: {3}
- moderate_max_features: {12}
- short_max_features: {8}
- short_drop_min: [0.08, 0.12]
- cum_lo: [0.25, 0.35]
- cum_hi: [0.60, 0.85]
- cum_late_flat_tol: [0.10, 0.20]
- cum_prev_flat_tol: [0.04, 0.12]
- forced_max_features: {6}
- forced_contrast_lo: {0.01}
- forced_contrast_hi: [1.00, 1.50]
- forced_late_flat_tol: {0.60}
- forced_prev_flat_tol: {0.15}
- forced_altitude_lo: {0.01}
- forced_gain: [0.65, 0.95]
- forced_recency_slope: {3.0}
- forced_recency_cap: {4.0}
- forced_beta_mult: [1.00, 1.35]
- forced_eps: [0.10, 0.17]
- strength_min: [0.12, 0.22]
- strength_high_extreme: [0.40, 0.65]
- strength_contrast: [0.70, 1.10]
- strength_flat: [0.12, 0.25]
- contrast_ref: [0.30, 0.45]
- contrast_power: [2.5, 4.0]
- altitude_power: [0.8, 1.2]
- flat_ref: [0.15, 0.30]
- size_penalty: [0.04, 0.12]
- gain_floor: [0.12, 0.18]
- gain_cap: [1.20, 1.40]
- recency_slope: [0.15, 0.40]
- recency_cap: [0.10, 0.25]
- beta_sub: [0.45, 0.65]
- epsilon_sub: [0.08, 0.16]
- epsilon_sub_size: [0.04, 0.09]
- validities: validities

**`predict source code`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length mismatch.')

    a = stim[0]
    b = stim[1]
    diff = a - b
    tally = float(np.sum(diff))
    positions = np.arange(n_features, dtype=float)

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        return e / np.sum(e)

    def lapse_mix(core, eps):
        probs = (1.0 - eps) * core + eps * np.array([0.5, 0.5])
        probs = np.clip(probs, 0.005, 0.995)
        total = float(np.sum(probs))
        if total <= 1e-12:
            return np.array([0.5, 0.5])
        return probs / total

    def find_block():
        n = n_features
        v = validities
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        spread = max(vmax - vmin, 1e-9)
        candidates = []

        gap_hi = float(parameters['gap_hi'])
        gap_lo = float(parameters['gap_lo'])
        gap_pos_max = float(parameters['gap_pos_max'])
        drop_lo = float(parameters['drop_lo'])
        drop_hi = float(parameters['drop_hi'])
        flat_tol = float(parameters['flat_tol'])
        block_max_large = int(parameters['block_max_large'])
        block_max_small = int(parameters['block_max_small'])
        moderate_allowed = n <= int(parameters['moderate_max_features'])

        for size in range(2, min(block_max_large, n - 1) + 1):
            i = n - size - 1
            if i < 0:
                continue
            jump = float(v[i + 1] - v[i])
            block_vals = v[i + 1:]
            block_range = float(np.max(block_vals) - np.min(block_vals))
            if block_range > flat_tol:
                continue

            kind = None
            direction = 0
            if moderate_allowed and size <= block_max_small:
                if gap_lo <= jump <= gap_pos_max:
                    kind = 'positive_end'
                    direction = 1
                elif -drop_hi <= jump <= -drop_lo:
                    kind = 'negative_end'
                    direction = -1
            if abs(jump) >= gap_hi:
                kind = 'large_jump'
                direction = 1 if jump > 0.0 else -1

            if kind is not None:
                candidates.append({
                    'start': i + 1,
                    'size': size,
                    'jump': jump,
                    'contrast': abs(jump),
                    'block_vals': block_vals,
                    'block_range': block_range,
                    'direction': direction,
                    'kind': kind
                })

        if n <= int(parameters['short_max_features']):
            cum_lo = float(parameters['cum_lo'])
            cum_hi = float(parameters['cum_hi'])
            cum_late_flat = float(parameters['cum_late_flat_tol'])
            cum_prev_flat = float(parameters['cum_prev_flat_tol'])
            for k in (3, 2):
                if n < 2 * k:
                    continue
                prev = v[n - 2 * k:n - k]
                late = v[n - k:]
                late_range = float(np.max(late) - np.min(late))
                prev_range = float(np.max(prev) - np.min(prev))
                contrast = float(np.mean(prev) - np.mean(late))
                if (cum_lo <= contrast <= cum_hi and
                        late_range <= cum_late_flat and
                        prev_range <= cum_prev_flat):
                    candidates.append({
                        'start': n - k,
                        'size': k,
                        'jump': contrast,
                        'contrast': contrast,
                        'block_vals': late,
                        'block_range': late_range,
                        'direction': -1,
                        'kind': 'cumulative_late'
                    })

        if n <= int(parameters['forced_max_features']):
            forced_lo = float(parameters['forced_contrast_lo'])
            forced_hi = float(parameters['forced_contrast_hi'])
            forced_late_flat = float(parameters['forced_late_flat_tol'])
            forced_prev_flat = float(parameters['forced_prev_flat_tol'])
            forced_alt = float(parameters['forced_altitude_lo'])
            forced_gain = float(parameters['forced_gain'])
            k = 3
            if n >= 2 * k:
                prev = v[n - 2 * k:n - k]
                late = v[n - k:]
                contrast = float(np.mean(prev) - np.mean(late))
                late_range = float(np.max(late) - np.min(late))
                prev_range = float(np.max(prev) - np.min(prev))
                if (forced_lo <= contrast <= forced_hi and
                        late_range <= forced_late_flat and
                        prev_range <= forced_prev_flat and
                        float(vmax - np.mean(late)) >= forced_alt):
                    candidates.append({
                        'start': n - k,
                        'size': k,
                        'jump': contrast,
                        'contrast': contrast,
                        'block_vals': late,
                        'block_range': late_range,
                        'direction': -1,
                        'kind': 'forced_late_pool',
                        'gain_override': forced_gain
                    })

        if not candidates and n <= int(parameters['short_max_features']):
            short_drop_min = float(parameters['short_drop_min'])
            jumps = np.diff(v)
            order = np.argsort(-np.abs(jumps), kind='stable')
            for jpos in order:
                jpos = int(jpos)
                size = n - (jpos + 1)
                if size < 2 or size > block_max_large:
                    continue
                jump = float(v[jpos + 1] - v[jpos])
                if abs(jump) < short_drop_min:
                    continue
                late_vals = v[jpos + 1:]
                candidates.append({
                    'start': jpos + 1,
                    'size': size,
                    'jump': jump,
                    'contrast': abs(jump),
                    'block_vals': late_vals,
                    'block_range': float(np.max(late_vals) - np.min(late_vals)),
                    'direction': 1 if jump > 0.0 else -1,
                    'kind': 'largest_drop_short'
                })
                break

        if not candidates:
            return None

        strength_min = float(parameters['strength_min'])
        strength_high_extreme = float(parameters['strength_high_extreme'])
        strength_contrast = float(parameters['strength_contrast'])
        strength_flat = float(parameters['strength_flat'])
        contrast_ref = float(parameters['contrast_ref'])
        contrast_power = float(parameters['contrast_power'])
        altitude_power = float(parameters['altitude_power'])
        flat_ref = float(parameters['flat_ref'])
        size_penalty = float(parameters['size_penalty'])
        gain_floor = float(parameters['gain_floor'])
        gain_cap = float(parameters['gain_cap'])

        for c in candidates:
            if c.get('gain_override') is not None:
                c['gain'] = float(c['gain_override'])
                continue

            flat_norm = float(np.clip(1.0 - c['block_range'] / flat_ref, 0.0, 1.0))
            if c['direction'] > 0:
                altitude = float(np.clip((float(np.mean(c['block_vals'])) - vmin) / spread, 0.0, 1.0))
                contrast_term = strength_high_extreme * (altitude ** altitude_power)
            else:
                base = float(np.clip(c['contrast'] / contrast_ref, 0.0, 1.0))
                contrast_term = strength_contrast * (base ** contrast_power)
            gain = strength_min + contrast_term + strength_flat * flat_norm
            if c['size'] > 3:
                gain = gain / (1.0 + size_penalty * (c['size'] - 3))
            c['gain'] = float(np.clip(gain, gain_floor, gain_cap))

        best = max(candidates, key=lambda c: (c['kind'] == 'forced_late_pool', c['gain'], c['contrast'], -c['size']))
        return best

    tau_tie = float(parameters['tau_tie'])

    if abs(tally) <= tau_tie:
        block = find_block()
        if block is not None and block['size'] >= 2:
            idx = np.arange(block['start'], block['start'] + block['size'], dtype=int)
            size = int(block['size'])
            rel = (positions[idx] - float(np.min(positions[idx]))) / max(1.0, float(size - 1))

            is_forced = block.get('kind') == 'forced_late_pool'
            if is_forced:
                forced_slope = float(parameters['forced_recency_slope'])
                forced_cap = float(parameters['forced_recency_cap'])
                ramp = 1.0 + np.minimum(forced_slope * rel, forced_cap)
            else:
                slope = float(parameters['recency_slope'])
                cap = float(parameters['recency_cap'])
                ramp = 1.0 + np.minimum(slope * rel, cap)

            w = ramp
            denom = float(np.sum(w))
            if denom <= 1e-12:
                w = np.ones_like(w)
            else:
                w = w / denom * float(size)

            sub_tally = float(np.dot(w, diff[idx]))

            beta_sub = float(parameters['beta_sub'])
            beta_eff = float(np.clip(beta_sub * block['gain'], 0.05, 2.0))

            if is_forced:
                beta_eff = float(np.clip(beta_sub * block['gain'] * float(parameters['forced_beta_mult']), 0.05, 2.0))
                eps_eff = float(np.clip(float(parameters['forced_eps']), 0.0, 0.90))
            else:
                eps_sub = float(parameters['epsilon_sub'])
                eps_sub_size = float(parameters['epsilon_sub_size'])
                eps_eff = float(np.clip(eps_sub + eps_sub_size * max(0.0, float(size - 3)), 0.0, 0.90))

            if abs(sub_tally) > 1e-12:
                core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_eff)
                return lapse_mix(core, eps_eff)
            return np.array([0.5, 0.5])
        return np.array([0.5, 0.5])

    beta_small = float(parameters['beta_tally_small'])
    beta_large = float(parameters['beta_tally_large'])
    epsilon_small = float(parameters['epsilon_tally_small'])
    epsilon_large = float(parameters['epsilon_tally_large'])
    frac = float(np.clip((n_features - 8.0) / 7.0, 0.0, 1.0))
    beta_tally = beta_small + frac * (beta_large - beta_small)
    epsilon_tally = epsilon_small + frac * (epsilon_large - epsilon_small)

    near_bound = float(parameters['near_bound'])
    n_low_conf = int(parameters['n_low_conf'])
    low_conf = (n_features >= n_low_conf) and (abs(tally) <= near_bound)

    if low_conf:
        beta = float(parameters['beta_near'])
        eps = float(parameters['epsilon_near'])
    else:
        beta = beta_tally
        eps = epsilon_tally

    core = stable_softmax(np.array([tally, -tally]), beta)
    return lapse_mix(core, eps)
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** People make binary-feature choices by a primary unweighted signed tally. When the tally is clear, choice follows the tally through a softmax with a small lapse. When the tally is tied or near-tied, people parse the advertised validity display into a single contiguous terminal chunk only when that chunk is coherent, i.e. its validity profile is flat and bounded by a clear validity discontinuity. Inside an opened terminal chunk the choice is based on an unweighted or validity-ordered sub-tally, never on positive exponential recency. When no coherent terminal chunk opens on a short display, people pool the earliest/largest coherent preceding block and choose by its sub-tally, producing directional anti-recency on late-block metrics. A targeted forced late-pool route is retained for very short displays in which an early homogeneous high-validity block is followed by a lower homogeneous late block.

**Parameters:**
- beta_tally_small: [0.50, 0.75]
- beta_tally_large: [0.85, 1.25]
- epsilon_tally_small: [0.02, 0.06]
- epsilon_tally_large: [0.02, 0.06]
- beta_near: [0.18, 0.38]
- epsilon_near: [0.75, 0.95]
- near_bound: {1.5}
- n_low_conf: {15}
- tau_tie: {0.5}
- gap_hi: [0.34, 0.45]
- gap_lo: [0.06, 0.10]
- gap_pos_max: [0.16, 0.26]
- drop_lo: [0.10, 0.14]
- drop_hi: [0.22, 0.45]
- flat_tol: [0.10, 0.18]
- block_max_large: {6}
- block_max_small: {3}
- moderate_max_features: {12}
- short_max_features: {8}
- short_drop_min: [0.08, 0.12]
- cum_lo: [0.25, 0.35]
- cum_hi: [0.60, 0.85]
- cum_late_flat_tol: [0.08, 0.18]
- cum_prev_flat_tol: [0.04, 0.12]
- forced_max_features: {6}
- forced_contrast_lo: {0.01}
- forced_contrast_hi: [0.90, 1.50]
- forced_late_flat_tol: {0.60}
- forced_prev_flat_tol: {0.15}
- forced_altitude_lo: {0.01}
- forced_gain: [0.65, 0.95]
- forced_beta_mult: [1.00, 1.35]
- forced_eps: [0.10, 0.17]
- strength_min: [0.12, 0.22]
- strength_high_extreme: [0.40, 0.65]
- strength_contrast: [0.70, 1.10]
- strength_flat: [0.12, 0.25]
- contrast_ref: [0.30, 0.45]
- contrast_power: [2.5, 4.0]
- altitude_power: [0.8, 1.2]
- flat_ref: [0.15, 0.30]
- size_penalty: [0.04, 0.12]
- gain_floor: [0.12, 0.18]
- gain_cap: [1.20, 1.40]
- beta_sub: [0.45, 0.65]
- epsilon_sub: [0.08, 0.16]
- epsilon_sub_size: [0.04, 0.09]
- val_weight_power: [0.5, 2.0]
- val_weight_floor: [0.02, 0.08]
- pool_flat_tol: [0.10, 0.20]
- pool_min_size: {2}
- pool_rank: {earliest, largest}
- pool_beta: [0.40, 0.80]
- pool_eps: [0.18, 0.32]
- fallback_min_features: {8}
- fallback_max_features: {11}
- validities: validities

**`predict source code`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length mismatch.')

    a = stim[0]
    b = stim[1]
    diff = a - b
    tally = float(np.sum(diff))

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        denom = float(np.sum(e))
        if denom <= 1e-12:
            return np.ones_like(scores, dtype=float) / float(len(scores))
        return e / denom

    def lapse_mix(core, eps):
        probs = (1.0 - eps) * core + eps * np.array([0.5, 0.5])
        probs = np.clip(probs, 0.005, 0.995)
        total = float(np.sum(probs))
        if total <= 1e-12:
            return np.array([0.5, 0.5])
        return probs / total

    vmin = float(np.min(validities))
    vmax = float(np.max(validities))
    spread = max(vmax - vmin, 1e-9)

    def find_block():
        candidates = []
        gap_hi = float(parameters['gap_hi'])
        gap_lo = float(parameters['gap_lo'])
        gap_pos_max = float(parameters['gap_pos_max'])
        drop_lo = float(parameters['drop_lo'])
        drop_hi = float(parameters['drop_hi'])
        flat_tol = float(parameters['flat_tol'])
        block_max_large = int(parameters['block_max_large'])
        block_max_small = int(parameters['block_max_small'])
        moderate_allowed = n_features <= int(parameters['moderate_max_features'])

        for size in range(2, min(block_max_large, n_features - 1) + 1):
            start = n_features - size
            if start < 1:
                continue
            jump = float(validities[start] - validities[start - 1])
            block_vals = validities[start:]
            block_range = float(np.max(block_vals) - np.min(block_vals))
            if block_range > flat_tol:
                continue

            kind = None
            direction = 0
            if moderate_allowed and size <= block_max_small:
                if gap_lo <= jump <= gap_pos_max:
                    kind = 'positive_end'
                    direction = 1
                elif -drop_hi <= jump <= -drop_lo:
                    kind = 'negative_end'
                    direction = -1
            if abs(jump) >= gap_hi:
                kind = 'large_jump'
                direction = 1 if jump > 0.0 else -1

            if kind is not None:
                candidates.append({
                    'start': start,
                    'size': size,
                    'jump': jump,
                    'contrast': abs(jump),
                    'block_vals': block_vals,
                    'block_range': block_range,
                    'direction': direction,
                    'kind': kind,
                    'gain_override': None
                })

        if n_features <= int(parameters['short_max_features']):
            cum_lo = float(parameters['cum_lo'])
            cum_hi = float(parameters['cum_hi'])
            cum_late_flat = float(parameters['cum_late_flat_tol'])
            cum_prev_flat = float(parameters['cum_prev_flat_tol'])
            for k in (3, 2):
                if n_features < 2 * k:
                    continue
                prev = validities[n_features - 2 * k:n_features - k]
                late = validities[n_features - k:]
                late_range = float(np.max(late) - np.min(late))
                prev_range = float(np.max(prev) - np.min(prev))
                contrast = float(np.mean(prev) - np.mean(late))
                if (cum_lo <= contrast <= cum_hi and
                        late_range <= cum_late_flat and
                        prev_range <= cum_prev_flat):
                    candidates.append({
                        'start': n_features - k,
                        'size': k,
                        'jump': contrast,
                        'contrast': contrast,
                        'block_vals': late,
                        'block_range': late_range,
                        'direction': -1,
                        'kind': 'cumulative_late',
                        'gain_override': None
                    })

        if n_features <= int(parameters['forced_max_features']):
            forced_lo = float(parameters['forced_contrast_lo'])
            forced_hi = float(parameters['forced_contrast_hi'])
            forced_late_flat = float(parameters['forced_late_flat_tol'])
            forced_prev_flat = float(parameters['forced_prev_flat_tol'])
            forced_alt = float(parameters['forced_altitude_lo'])
            forced_gain = float(parameters['forced_gain'])
            k = 3
            if n_features >= 2 * k:
                prev = validities[n_features - 2 * k:n_features - k]
                late = validities[n_features - k:]
                contrast = float(np.mean(prev) - np.mean(late))
                late_range = float(np.max(late) - np.min(late))
                prev_range = float(np.max(prev) - np.min(prev))
                if (forced_lo <= contrast <= forced_hi and
                        late_range <= forced_late_flat and
                        prev_range <= forced_prev_flat and
                        float(vmax - np.mean(late)) >= forced_alt):
                    candidates.append({
                        'start': n_features - k,
                        'size': k,
                        'jump': contrast,
                        'contrast': contrast,
                        'block_vals': late,
                        'block_range': late_range,
                        'direction': -1,
                        'kind': 'forced_late_pool',
                        'gain_override': forced_gain
                    })

        if not candidates and n_features <= int(parameters['short_max_features']):
            short_drop_min = float(parameters['short_drop_min'])
            jumps = np.abs(np.diff(validities))
            order = np.argsort(-jumps, kind='stable')
            for jpos in order:
                jpos = int(jpos)
                size = n_features - (jpos + 1)
                if size < 2 or size > block_max_large:
                    continue
                jump = float(validities[jpos + 1] - validities[jpos])
                if abs(jump) < short_drop_min:
                    continue
                late_vals = validities[jpos + 1:]
                block_range = float(np.max(late_vals) - np.min(late_vals))
                if block_range > flat_tol:
                    continue
                candidates.append({
                    'start': jpos + 1,
                    'size': size,
                    'jump': jump,
                    'contrast': abs(jump),
                    'block_vals': late_vals,
                    'block_range': block_range,
                    'direction': 1 if jump > 0.0 else -1,
                    'kind': 'largest_drop_short',
                    'gain_override': None
                })
                break

        if not candidates:
            return None

        strength_min = float(parameters['strength_min'])
        strength_high_extreme = float(parameters['strength_high_extreme'])
        strength_contrast = float(parameters['strength_contrast'])
        strength_flat = float(parameters['strength_flat'])
        contrast_ref = float(parameters['contrast_ref'])
        contrast_power = float(parameters['contrast_power'])
        altitude_power = float(parameters['altitude_power'])
        flat_ref = float(parameters['flat_ref'])
        size_penalty = float(parameters['size_penalty'])
        gain_floor = float(parameters['gain_floor'])
        gain_cap = float(parameters['gain_cap'])

        for c in candidates:
            if c.get('gain_override') is not None:
                c['gain'] = float(c['gain_override'])
                continue

            flat_norm = float(np.clip(1.0 - c['block_range'] / flat_ref, 0.0, 1.0))
            if c['direction'] > 0:
                altitude = float(np.clip((float(np.mean(c['block_vals'])) - vmin) / spread, 0.0, 1.0))
                contrast_term = strength_high_extreme * (altitude ** altitude_power)
            else:
                base = float(np.clip(c['contrast'] / contrast_ref, 0.0, 1.0))
                contrast_term = strength_contrast * (base ** contrast_power)
            gain = strength_min + contrast_term + strength_flat * flat_norm
            if c['size'] > 3:
                gain = gain / (1.0 + size_penalty * (c['size'] - 3))
            c['gain'] = float(np.clip(gain, gain_floor, gain_cap))

        return max(candidates, key=lambda c: (c['kind'] == 'forced_late_pool', c['gain'], c['contrast'], -c['size']))

    def find_fallback_pool():
        pool_flat_tol = float(parameters['pool_flat_tol'])
        pool_min_size = int(parameters['pool_min_size'])
        runs = []
        start = 0
        for i in range(1, n_features + 1):
            if i == n_features:
                length = i - start
                if length >= pool_min_size:
                    runs.append((start, i))
                break
            if max(validities[start:i + 1]) - min(validities[start:i + 1]) > pool_flat_tol:
                length = i - start
                if length >= pool_min_size:
                    runs.append((start, i))
                start = i

        preceding = [r for r in runs if r[1] < n_features]
        if not preceding:
            return None
        rank = str(parameters['pool_rank'])
        if rank == 'earliest':
            return min(preceding, key=lambda r: (r[0], -(r[1] - r[0])))
        return max(preceding, key=lambda r: (r[1] - r[0], -r[0]))

    tau_tie = float(parameters['tau_tie'])

    if abs(tally) <= tau_tie:
        block = find_block()
        if block is not None and block['size'] >= 2:
            idx = np.arange(block['start'], block['start'] + block['size'], dtype=int)
            size = int(block['size'])
            vals = validities[idx]
            if float(np.max(vals) - np.min(vals)) > 1e-9:
                val_floor = float(parameters['val_weight_floor'])
                val_power = float(parameters['val_weight_power'])
                w = np.asarray((vals - vmin + val_floor) ** val_power, dtype=float)
            else:
                w = np.ones(size, dtype=float)
            denom = float(np.sum(w))
            if denom <= 1e-12:
                w = np.ones(size, dtype=float)
            else:
                w = w / denom * float(size)

            sub_tally = float(np.dot(w, diff[idx]))
            beta_sub = float(parameters['beta_sub'])
            beta_eff = float(np.clip(beta_sub * block['gain'], 0.05, 2.0))

            is_forced = block.get('kind') == 'forced_late_pool'
            if is_forced:
                beta_eff = float(np.clip(beta_sub * block['gain'] * float(parameters['forced_beta_mult']), 0.05, 2.0))
                eps_eff = float(np.clip(float(parameters['forced_eps']), 0.0, 0.90))
            else:
                eps_sub = float(parameters['epsilon_sub'])
                eps_sub_size = float(parameters['epsilon_sub_size'])
                eps_eff = float(np.clip(eps_sub + eps_sub_size * max(0.0, float(size - 3)), 0.0, 0.90))

            if abs(sub_tally) > 1e-12:
                core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_eff)
                return lapse_mix(core, eps_eff)
            return np.array([0.5, 0.5])

        fallback_min = int(parameters['fallback_min_features'])
        fallback_max = int(parameters['fallback_max_features'])
        if fallback_min <= n_features <= fallback_max:
            pool = find_fallback_pool()
            if pool is not None:
                idx = np.arange(pool[0], pool[1], dtype=int)
                sub_tally = float(np.sum(diff[idx]))
                if abs(sub_tally) > 1e-12:
                    core = stable_softmax(np.array([sub_tally, -sub_tally]), float(parameters['pool_beta']))
                    return lapse_mix(core, float(parameters['pool_eps']))
                return np.array([0.5, 0.5])

        return np.array([0.5, 0.5])

    beta_small = float(parameters['beta_tally_small'])
    beta_large = float(parameters['beta_tally_large'])
    eps_small = float(parameters['epsilon_tally_small'])
    eps_large = float(parameters['epsilon_tally_large'])
    frac = float(np.clip((n_features - 8.0) / 7.0, 0.0, 1.0))
    beta_tally = beta_small + frac * (beta_large - beta_small)
    epsilon_tally = eps_small + frac * (eps_large - eps_small)

    low_conf = (n_features >= int(parameters['n_low_conf'])) and (abs(tally) <= float(parameters['near_bound']))
    if low_conf:
        beta = float(parameters['beta_near'])
        eps = float(parameters['epsilon_near'])
    else:
        beta = beta_tally
        eps = epsilon_tally

    core = stable_softmax(np.array([tally, -tally]), beta)
    return lapse_mix(core, eps)
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design uses six binary cues with widely spread validities. It places TTB's first discriminating cue at position 0, 1, or 2 in the validity-sorted order, while packing the later cues to oppose or tie the tallying score. In half the unique pairs the first discriminating cue favors A and in half it favors B. TTB therefore predicts p(A) is uniformly high for all six TTB-A trials and uniformly low for all six TTB-B trials, because later cues are never consulted after the first discriminating cue. Tallying instead ignores validity and counts all features: in the six oppose pairs the tally winner is opposite to TTB (count differences of 1 vs 5, 1 vs 4, and 1 vs 3), producing a crossover; in the six tie pairs tallying predicts 0.5 regardless of beta. The 12 unique pairs are each repeated 8 times for roughly 96 trials per subject. This gives a quantitative dissociation: TTB choice probability should be invariant within first-cue winner type, whereas tallying choice probability should move from near 0 to 0.5 to near 1 across the tally-difference levels.

[1] Eight experts with validities spread from .50 to .92; the highest validity is placed at display index 4 so TTB must re-sort and cannot be mimicked by left-to-right scanning. In every unique pair the top cue discriminates: trials 1-8 have top cue A, trials 9-16 top cue B. TTB therefore predicts a constant p(A) within each block (sigmoid(beta) for A-top, sigmoid(-beta) for B-top, plus lapse), because lower cues are never consulted. Tallying instead counts all eight cues, so p(A) = sigmoid(beta*d) with d = #A wins - #B wins. The d sequence is +8,+6,+4,+2,0,-2,-4,-6 for A-top and -8,-6,-4,-2,0,+2,+4,+6 for B-top. This yields a full psychometric curve for Tallying within each TTB-constant block, including d=0 pairs where Tallying must guess but TTB remains confident, and crossover pairs where the top cue and the majority of cues point in opposite directions. Mirrored blocks control for A/B response bias. 16 unique pairs x 6 repeats = 96 trials.

[2] This design dissociates validity-graded compensatory integration from pure Tallying by crossing tally difference d with a validity-composition score S. Six low-validity (0.5) and six high-validity (1.0) experts give centered validities of -0.25 and +0.25. For each pair define d = #A wins - #B wins and S = 2 * sum of centered validities of features A wins. Tallying depends only on d, so it predicts a flat p(A)=0.5 for all eight d=0 pairs, one common value for the two d=+1 pairs, and one common value for the two d=-1 pairs. Validity-graded integration predicts D = d + kappa*S, so within each d block p(A) should vary with S. The d=0 pairs span S = +3, +2.5, +2, +1, -1, -2, -2.5, -3 (the +/-2.5 pairs include tied experts), producing a psychometric curve where Tallying is flat at 0.5. The d=+1 pairs have S=+3 and -3 (D=1.3 vs 0.7 at kappa=0.1), and the d=-1 pairs have S=+2.5 and -2.5 (D=-0.75 vs -1.25), so Tallying predicts identical p(A) within each d block while the advocated theory predicts a validity-composition shift. This is a quantitative dissociation: Tallying yields a three-level step function in d, whereas the advocated theory yields a continuous S effect within each d level. Because kappa may vary across subjects, the d=0 S curve also predicts overdispersion in individual choice proportions that Tallying cannot produce.

[3] This design dissociates pure Tallying from validity-graded compensatory integration by crossing the unweighted tally difference d = #A wins - #B wins with a validity-composition score S. Ten experts have graded validities [0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0] (mean 0.75), giving centered validities [-0.25, -0.25, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.25, 0.25]. Every pair is complementary (B = 1 - A), so there are no ties, d = sum(A) - sum(B), and S = 2 * sum of centered validities of features where A = 1. The eight d = 0 pairs (A has five 1s) span S = +1.9, +1.3, +0.7, +0.1, -0.1, -0.7, -1.3, -1.9. Tallying depends only on d, so it predicts p(A) = 0.5 for all eight d = 0 pairs at every beta. Validity-graded integration predicts advantage = d + kappa * S, so for any subject with kappa != 0, p(A) must vary monotonically with S across the d = 0 pairs: high-S pairs favor A and low-S pairs favor B when kappa > 0, with the reverse when kappa < 0. This yields a quantitative dissociation: a flat line at 0.5 for Tallying versus a psychometric S curve for the competing theory, plus overdispersion in individual d = 0 choice proportions that Tallying cannot produce. The four d = +/-2 pairs (two with S = +1.8, two with S = -1.8) serve as controls to estimate beta and the d effect. Twelve unique pairs x 8 repeats = 96 trials.

[4] Sixteen experts: eight with validity 0.50 and eight with validity 1.00, so centered validities are -0.25 and +0.25. This binary spread maximizes the validity-composition score S = sum_j (v_j - 0.75)*(A_j - B_j) for a fixed tally difference and makes the competing fallback weights (v_j - 0.5)^gamma exactly zero for low-validity experts. The 16 unique pairs are organized into three blocks. (1) Six d=+1 pairs (A wins one more feature than B) with S = -3.75, -2.75, -1.75, 0.25, 1.25, 3.75. (2) Six d=-1 pairs, the A/B swaps of the first block, with S = 3.75, 2.75, 1.75, -0.25, -1.25, -3.75. (3) Four d=0 tally-tie pairs in which the high-validity tally H = #high A wins - #high B wins is +8, +4, -4, -8. On the non-tie pairs the competing validity-gated tallying rule never opens its validity gate because |d| > tau, so it predicts a single p(A) for all six d=+1 pairs and a single p(A) for all six d=-1 pairs, regardless of S. The advocated validity-graded integration predicts advantage D = d + kappa*S, so within each d block p(A) must vary with S for any subject with kappa != 0 (e.g., D shifts by +/-0.375 at kappa=0.1), giving a within-block S psychometric curve and overdispersion. On the d=0 pairs the competing rule opens its validity gate and, with high beta_v, predicts p(A) strongly follows the high-validity tally (high for H=+8/+4, low for H=-4/-8, scaled by fallback_p), whereas the advocated rule predicts D = kappa*S and, because kappa has a symmetric zero-centered prior, aggregate p(A)=0.5 for all four tie pairs. Thus the design produces a double dissociation: non-tie trials test continuous validity reweighting against a closed gate, and tie trials test the gated validity fallback against a zero-average compensatory rule. Swapped pairs control for A/B response bias; 16 unique pairs x 6 repeats = 96 trials.

[5] Twelve experts with graded validities (mean .75) give centered validities [-.25,-.25,-.25,-.25,-.15,-.05,.05,.15,.25,.25,.25,.25]. The four d=0 pairs are constructed so the centered-validity score S=sum c_j(A_j-B_j) is exactly 0. For validity-graded compensatory integration, advantage D=d+kappa*S=0, so p(A)=.5 for every kappa. For validity-gated tallying, d=0 opens the validity gate; V=sum (v_j-.5)^gamma*(A_j-B_j) is dominated by the 1.0 experts. T1/T2 have V>0 (A favored) and T3/T4 have V<0 (B favored), so p(A) is pushed above/below .5 with strength scaled by fallback_p. This is a flat-line vs high-low dissociation. The four d=+/-1 non-tie pairs cross d with S=+/-2.35. Since |d|=1>tau, gated tallying predicts p(A) depends only on d (identical within d=+1 and within d=-1), while compensatory integration predicts D=d+kappa*S, so p(A) shifts with S within each d block. Mirrored pairs control A/B response bias. 8 unique pairs x 12 repeats = 96 trials.

[6] This design uses 11 graded experts with mean validity 0.75 and centered validities from -0.25 to +0.25. Eight of the twelve unique pairs are tally ties, d = sum(A - B) = 0, constructed to have large absolute linear validity evidence on tie trials (0.75 to 1.50). The competing context-gated tallying rule therefore sees tie prevalence c = 8/12 = 0.667 and mean tie evidence far above its consistency floor, so after warm-up its validity gate opens to lambda near lambda_max (0.93 to 0.96). The four remaining pairs are critical non-tie crossovers. In C1, d = +3 and S = sum centered_validities * (A - B) = -1.40. In C3, d = +1 and S = -1.50. For the advocated validity-graded integration, D = d + kappa * S with absolute kappa at most 0.10, so C1 gives D between 2.86 and 3.14 and C3 gives D between 0.85 and 1.15; p(A) is therefore above 0.5 for both, and higher for C1 than C3, regardless of kappa sign. For the competing rule with the gate open, the top-validity experts favor B in C1 and C3, so V = sum (v_j - 0.5)^gamma * (A_j - B_j) is negative; p_valid(A) is below 0.5 across the specified gamma and beta_v ranges, and mixing that with the near-1 tally route at lambda around 0.94 keeps overall p(A) below 0.5, typically well below. Thus C1 and C3 produce a sign reversal: advocated theory predicts A-favoring choices, competing theory predicts B-favoring choices once the gate is open. C2 and C4 are exact A/B swaps and give mirrored predictions. The eight d = 0 filler pairs add a second dissociation: advocated theory predicts only small kappa * S shifts around chance, while the gated competing rule predicts strong validity-driven near-1 or near-0 choices after gate opening. Finally, because the competing gate is history-dependent, it predicts a temporal change on the critical crossover pairs from early tally-like choices to later validity-like choices, whereas advocated theory predicts no history effect.

[7] This 20-feature design uses 10 experts with validity 1.00 and 10 experts with validity 0.50, so mean validity is 0.75 and centered validities are +0.25 for the first 10 features and -0.25 for the last 10 features. All unique pairs have tally difference d = +1 or -1, so there are zero tally ties. The context-gated tallying theory therefore stays in its unweighted tallying mode throughout: tie prevalence is zero, the validity gate never opens, and p(A) depends only on d. Within the d=+1 block, the validity-composition score S = sum_j (v_j - 0.75) * (a_j - b_j) takes values +4.75, +2.75, +0.75, -0.75, -2.75, and -4.75; the d=-1 block is the exact A/B swap. The advocated theory predicts one flat p(A) across all six d=+1 pairs and one flat p(A) across all six d=-1 pairs. The competing validity-graded integration predicts D = d + kappa * S, so for any subject with kappa not equal to zero p(A) must be monotone in S within each d block; across subjects with a symmetric kappa prior this appears as S-linked slope or overdispersion rather than the advocated flat S-independent curve. Mirrored swaps control response bias and create a quantitative double dissociation: two flat d-based levels versus two S-ordered psychometric gradients.

[8] This design isolates the competing theory's large-array near-tally regularization from the advocated validity-graded integration. The validity set has 20 experts: nine at 1.00, nine at 0.50, and two at 0.75. The mean validity is 0.75, so centered validities are +0.25 for the 1.00 experts, -0.25 for the 0.50 experts, and 0 for the 0.75 experts. The top tier contains nine experts with validity 1.00, the next validity is 0.75, and the gap is 0.25. That gap is below the competing theory's minimum salience_gap_hi of 0.30, and the top block is size 9 rather than a small elite of 2-3, so the validity tie-break never becomes active. All 16 unique pairs have unweighted tally d = sum(a_i - b_i) equal to +1 or -1, never zero, so there is no exact-tie validity fallback either. Because n_features = 20, the competing theory places every pair in the low-confidence near-tally branch. With beta_near between 0.3 and 0.8 and epsilon_near between 0.78 and 0.97, p(A) for d = +1 is only about 0.51 to 0.57, and p(A) for d = -1 is about 0.43 to 0.49, with no S effect across pairs. The advocated theory predicts D = d + kappa*S, where S = sum((v_i - 0.75)*(a_i - b_i)). The first eight pairs have d = +1 and S values -4.5, -3.0, -1.5, -0.5, +0.5, +1.5, +3.0, +4.5; the last eight are A/B swaps with d = -1 and opposite S values. For kappa in [-0.10, 0.10], D remains positive on all d = +1 pairs and negative on all d = -1 pairs. With a moderate beta, advocated choice probabilities are therefore far from chance, and within each d block they vary with S in opposite directions for positive versus negative kappa subjects. The competing theory instead predicts near-chance choices in both blocks with a nearly flat S curve. The mirrored pairs control for A/B response bias, and 16 unique pairs x 6 repeats gives 96 trials.

[9] This experiment uses six experts with validities [1.0, 1.0, 1.0, 0.9, 0.9, 0.5]. The mean validity is 0.8833, so the centered validities are [0.1167, 0.1167, 0.1167, 0.0167, 0.0167, -0.3833]. The top tier is exactly the three 1.0 experts; the second highest validity is 0.9, giving a validity gap of 0.10 and a top size of 3, which opens the advocated theory's small-elite tie-break gate.

Every unique pair is a tally tie: d = sum(A - B) = 0. On these pairs the advocated theory's default tallying signal is silent, and its gated tie-break consults only the three top experts. Let T be the top sub-tally, the number of top experts favoring A minus the number favoring B, and let S be the centered-validity score, S = sum c*(A - B). The construction uses the two 0.9 experts and the one 0.5 expert to vary S independently of T; in this validity set S = 0.5*T + 0.4*U, where U is the difference sum of the two 0.9 experts.

Trials 1-6 have T = +1, +1, +1, +2, +2, +3 with S values -0.30, 0.10, 0.50, 0.20, 0.60, 0.70. Trials 7-12 are the exact A/B mirrors with T = -1, -1, -1, -2, -2, -3 and S values 0.30, -0.10, -0.50, -0.20, -0.60, -0.70.

The advocated theory predicts p(A) follows T only: the three T=+1 pairs are indistinguishable around 0.57-0.61, the two T=+2 pairs around 0.645-0.71, and the T=+3 pair around 0.71-0.79, with mirrored values below 0.5. In particular, within T=+1, the large S change from -0.30 to +0.50 should not move p(A).

The competing validity-graded integrator has d = 0 on every trial, so its only advantage term is kappa*S. Because kappa has a symmetric zero-centered prior, the expected p(A) is exactly 0.5 for every one of the 12 pairs, regardless of T. Positive-kappa subjects would order the choices by S, with P1 below 0.5 and P3 above 0.5, while negative-kappa subjects would reverse that order. Thus the design gives a quantitative dissociation: a T-ordered elite sub-tally psychometric function for the advocated theory versus a flat-at-chance S-only pattern for the competing theory. The 12 unique pairs are repeated K = 8 times to give 96 trials.

[10] All 12 unique pairs are unweighted tally ties, d = sum(A - B) = 0, so both fallback gates open. The validities define a three-cue elite: the first three experts have validity 1.0 and the remaining nine have 0.8, giving a gap of 0.20 and top_size = 3, which meets both theories' small-elite conditions. The first three features define T = #top A wins - #top B wins among positions 0-2; the last three define L = #late A wins - #late B wins among positions 9-11. Middle features 3-8 are tally-balancing fillers. For the advocated model, combined relevance = validity * exp(rho * (position - max_position)) makes the final three positions dominate every earlier 0.8 cue for rho in [0.60, 1.00], so its restricted block is positions 9-11 and its weighted sub-tally is exactly +3 on L=+3 trials and -3 on L=-3 trials. The competing model ignores position and uses the three 1.0 experts, so its sub-tally equals T. The design is a 4 x 2 factorial over T in {+3,+1,-1,-3} and L in {+3,-3}; trials 1-8 are sign-reversal cells. Critical predictions: in T=+3,L=-3 cells, competing p(A) is roughly 0.85-0.91 while advocated p(A) is roughly 0.03-0.07; in T=-3,L=+3 cells these reverse. Same-sign cells provide magnitude controls. A validity-only fallback cannot produce an L effect within a fixed T, and a recency-weighted fallback cannot produce a T effect within a fixed L, giving a quantitative double dissociation. The 12 unique pairs are repeated 8 times for 96 trials.

[11] Eight experts are used. The first four validities (1.00, 0.996, 0.993, 0.991) fall within the advocated model's top_tol of 0.01, so it groups them into a top block of size 4. The competing model's top_tol of 0.001 leaves only the 1.00 expert in its top block, so its top_size is 1. With the next highest validity at 0.50, the advocated model sees a large validity gap of 0.50 with a bounded top block of size 4, which activates its block_clear gated tie-break on every trial. The competing model instead sees a validity gap of only 0.004 and no eligible small elite, so its validity gate stays closed. All 12 unique pairs have an exact unweighted tally tie, d = sum(A - B) = 0, so the competing model remains in default tally mode and predicts p(A) = 0.5 for every pair. The advocated model uses the gated top-block sub-tally T over the first four experts. The unique T values are +4, +3, +3, +2, +2, +1, -1, -2, -2, -3, -3, -4. With beta_sub between 0.35 and 0.50 and the top_size=4 extra lapse of about 0.24 to 0.34, advocated p(A) is ordered by T: roughly 0.61 to 0.68 for T = +1, rising to roughly 0.79 to 0.87 for T = +4, with mirrored values below 0.5 for negative T. The competing model is flat at 0.5 for all 12 pairs. Thus the design produces a quantitative T-ordered psychometric function for the advocated theory against a flat chance prediction for the competing theory. The 12 unique pairs are repeated 8 times for 96 trials per subject.

[12] This design uses 10 features. The first three experts have validity 1.00 and the remaining seven have validity 0.50, so the top validity block has size 3 with a validity gap of 0.50. This opens the advocated theory's small-elite/block-clear tie-break gate. With rho in [0.60, 1.00], the advocated theory re-ranks cues by validity * exp(rho * (position - max_position)), so its restricted block of size 3 is always the final three display positions 7-9, even though those cues have only 0.50 validity. The competing theory cannot open a late block here: its only validity jump is the 1.00 to 0.50 drop at position 2, which creates a later block of size 7, exceeding its maximum block size of 6; the positive end-boundary, negative-drop, cumulative-contrast, and six-cue routes never apply for these 10 features. All eight unique pairs are exact unweighted tally ties, d = sum(A - B) = 0. The final three cues vary from all favoring A, to A at only the latest position, to mirrored all-favoring-B patterns. The advocated theory therefore predicts an ordered recency-weighted late sub-tally psychometric function: trials 1-4 have decreasing positive sub-tally and p(A) above 0.5, while trials 5-8 are sign mirrors and p(A) below 0.5. The competing theory predicts p(A) = 0.5 for every pair because no perceptual late block is formed. Eight unique pairs repeated 12 times gives 96 trials.

[13] This eight-feature design uses validities [0.80 five times, 0.65 three times]. The last three equal low-validity cues are created by a moderate negative drop from 0.80 to 0.65. For the advocated theory this drop is within the negative end-drop segmentation route, so on exact tally ties it opens the late block at display positions 5-7 and uses a recency-weighted sub-tally there. For the competing theory the top validity tier has size 5, not 2-3, and the validity gap is only 0.15, below 0.30, so its validity gate stays closed; with an exact tally tie and only eight features it predicts p(A)=0.50 for every pair. Each pair sums the early five features against the late three so that total tally is exactly zero, but the late-block difference patterns span a symmetric six-level weighted sub-tally: +++ gives about +3, -++ about +1.7 to +2.4, +-+ about +1.1 to +1.5, -+- the mirrored negative, +-- more negative, and --- about -3. The advocated model therefore predicts ordered p(A) from roughly 0.12 to 0.88 across the six pairs, while the competing model predicts a flat 0.50 line. Mirrored A/B pairs control response bias.

[14] This design isolates a flatness requirement that differs between the two segmentation codes. The validity sequence contains a single large upward jump (0.50) into a six-cue terminal block that spans 1.00 down to 0.50, so its validity range is 0.50. The competing theory's large-jump route does not require the terminal block to be flat; it opens positions 4-9 on every tied trial and uses recency-weighted sub-tallies there. The advocated theory's large-jump route requires the terminal block's validity range to be within flat_tol (0.12-0.20), so this non-flat block is rejected. Because n_features=10, the advocated theory's short-display cumulative/forced/fallback routes are disabled, and the remaining internal jumps are all small (<=0.20), so no advocated chunk opens. All twelve unique pairs are exact unweighted tally ties (sum A-B = 0). The first six late-block patterns place later, recency-weighted positions in favor of A while earlier late positions oppose, producing a positive recency-weighted sub-tally; the last six are exact A/B mirrors. Thus the advocated theory predicts a flat p(A)=0.5 for every pair, whereas the competing theory predicts a graded positive-then-negative recency psychometric function. Twelve unique pairs x 8 repeats = 96 trials.

[15] The validity sequence has seven 0.50 cues followed by 0.60, 0.76, 0.50. Only moderate positive end discontinuities are present: +0.10 into the final three cues and +0.16 into the final two cues. On exact tally ties, the advocated theory's positive-end route opens the final two-cue block at positions 8 and 9 because that route has no flatness requirement, then applies positive recency weighting. The competing theory's positive-end route requires the opened terminal block to be flat within flat_tol 0.12 to 0.20. Both eligible terminal blocks are non-flat: positions 7 to 9 have validities 0.60, 0.76, 0.50 with range 0.26, and positions 8 to 9 have validities 0.76, 0.50 with range 0.26. With n_features equal to 10, the competing theory's cumulative, forced, and largest-jump fallback routes are disabled, and no jump reaches its large-jump threshold. Thus on tied trials the advocated theory predicts a graded recency-weighted late-block psychometric function, with sub-tallies +2, +w9, +w8, and +(w9 - w8) for the first four pairs and the exact A/B mirrors for the last four, producing p(A) values that spread between roughly 0.58 and 0.85 on the positive side. The competing theory finds no terminal block and predicts p(A) equal to 0.50 for all eight pairs. This gives a quantitative dissociation: an ordered recency-weighted curve versus a flat chance line. The eight unique pairs repeated 12 times give 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Six-feature design targeting the forced late-pool difference. The validity vector [1.0, 1.0, 1.0, 0.9, 0.5, 0.5] has a flat high early triplet and a lower late triplet with mean 0.633 and range 0.40, satisfying both models' forced late-pool opening conditions on every trial. All 12 unique pairs are exact unweighted tally ties, so the primary tally is silent and both models enter the forced terminal-chunk branch at display positions 3-5. Inside that opened chunk the advocated model uses its forced recency ramp with normalized position weights 0.4, 1.0, 1.6, so the last late feature dominates. The competing model instead uses validity-ordered weights; the first late feature has validity 0.9 while the last two are 0.5, so across its full parameter ranges the first late feature receives more weight than the middle and last late features combined. Every pair puts the first and last late features in conflict, with the middle late feature varied, while the early features only balance the total tally. Thus the advocated model predicts p(A) follows the last late feature, whereas the competing model predicts p(A) follows the first late feature: a per-trial sign reversal. Trials 1-6 have the first late feature favoring B and the last late feature favoring A; trials 7-12 are their A/B mirrors for bias control. Twelve unique pairs repeated eight times gives 96 trials.",
  "validities": [
    1.0,
    1.0,
    1.0,
    0.9,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 16482,
  "output_tokens": 30337,
  "reasoning_tokens": 29492,
  "provider": "Baidu"
}
```
