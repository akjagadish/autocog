# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=10):** [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 0, 1, 1]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1]
  trial 5: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]

**Rationale:** This design isolates a structural difference in the no-terminal-chunk, tied-tally path. The validity ladder uses adjacent steps of only 0.05, below both theories' smallest boundary thresholds (positive-end lower bound 0.06, negative-drop lower bound 0.10, large-jump lower bound 0.34), and n_features=10 disables the short-display cumulative, forced, and largest-drop routes. Therefore neither theory opens a terminal chunk, and the competing theory predicts p(A)=0.50 on every tied trial. The advocated theory instead has a preceding-block pooling route for 8-11 feature displays when no terminal chunk opens. With this validity ladder, its pool parser cuts off the earliest coherent run after the first 3, 4, or 5 positions depending on pool_flat_tol, and both pool_rank settings (earliest and largest) select that first run. All trial pairs are exact unweighted tally ties, so the primary tally is silent. Trials 1-4 put that early pooled prefix in favor of A, giving advocated p(A) above 0.5 (ranging from about 0.63 to 0.90 depending on the opened prefix and pool parameters), while trials 5-8 are A/B mirrors and give p(A) below 0.5. The competing theory is flat at chance for all eight pairs, producing a mirrored high/low versus flat-at-chance quantitative dissociation. Eight unique pairs repeated 12 times gives 96 trials per subject and the mirrored block controls response bias.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "import numpy as np\n\ndef metric(data):\n    a = np.array([np.asarray(x, dtype=float) for x in data['option_a_ratings']])\n    b = np.array([np.asarray(x, dtype=float) for x in data['option_b_ratings']])\n    early = a[:, :3] - b[:, :3]\n    early_tally = early.sum(axis=1)\n    resp = data['response'].to_numpy(dtype=float)\n    positive = early_tally > 0.0\n    negative = early_tally < 0.0\n    if not np.any(positive) or not np.any(negative):\n        return 0.0\n    return float(resp[negative].mean() - resp[positive].mean())\n",
  "rationale": "All eight trial pairs have an exact unweighted tally tie, so the competing theory predicts chance choices for every pair. The advocated theory predicts that its fallback earliest-coherent-pool route weights the first 3 to 5 validity positions, and the first three positions are always unanimously aligned with one option in every trial pair. This metric therefore classifies each trial by the first three features: if those positions favor A, early_tally is positive; if they favor B, early_tally is negative. It returns the B-choice rate on early-B trials minus the B-choice rate on early-A trials. Under the advocated theory this is large and positive because early-B trials should yield mostly B choices and early-A trials should yield mostly A choices. Under the competing theory both rates remain near 0.5, giving a value near zero. Using 48 trials per early direction per subject makes the per-subject estimate tight."
}
```

## Usage

```json
{
  "input_tokens": 11211,
  "output_tokens": 11337,
  "reasoning_tokens": 10952,
  "provider": "Baidu"
}
```
