# Round 7 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_9` — SURVIVED ✓

**Description:** People make binary-feature choices by a primary unweighted signed tally. When the tally is clear, choice follows the tally stochastically with a small lapse. When the tally is tied or near-tied, people parse the validity display into a single contiguous terminal perceptual chunk only when the validity sequence contains a structural boundary. In addition to large jumps, moderate positive and negative end discontinuities, and cumulative short-display contrasts, short displays have a fallback parsing rule: if no other boundary rule has opened a chunk, the largest adjacent validity discontinuity defines the terminal chunk. A targeted forced late-pool rule is added for very short displays: when a flat high-validity early triplet is followed by a lower, homogeneous late triplet, that late block is opened with bounded decisiveness, producing recency-weighted choices against the early high-validity pool. Recency inside any chunk remains a weak linear ramp with a cap, not an exponential validity-times-position re-ranking, and large displays with small tallies remain low-confidence.

**Rationale:** This is a minimal in-family edit on the accepted running-best base. I keep the unweighted tally, bounded terminal-chunk tie-break, and weak capped linear recency intact. The main fix targets the forced late-pool gate on six-cue displays: the previous stochastic flatness/contrast thresholds let only a subset of simulated subjects open the forced block in Experiment 10, leaving the predicted top-pool effect near -0.39. I make the forced-block predicates deterministic and deliberately permissive for n<=6 while keeping forced_prev_flat_tol at 0.15, which is just below the Experiment 1 early-triplet range of ~0.16, so Exp1 does not accidentally open the forced branch. I then recalibrate forced decisiveness by lowering forced_beta_mult and raising forced_eps so a fully covered Exp10 block lands near the observed -0.745 rather than saturating at the ceiling. Finally, I apply the single positive-side lever recommended by the critic, a modest reduction of recency_cap to pull the Exp14 overshoot down without reintroducing the broader positive gain/lapse trims that collapsed Exp4 in the rejected previous candidate. No exponential validity-times-position re-ranking or new mechanism family is introduced.

**Parameters:**
  - `beta_tally_small`: `[0.50, 0.75]`
  - `beta_tally_large`: `[0.85, 1.25]`
  - `epsilon_tally_small`: `[0.02, 0.06]`
  - `epsilon_tally_large`: `[0.02, 0.06]`
  - `beta_near`: `[0.18, 0.38]`
  - `epsilon_near`: `[0.75, 0.95]`
  - `near_bound`: `{1.5}`
  - `n_low_conf`: `{15}`
  - `tau_tie`: `{0.5}`
  - `gap_hi`: `[0.34, 0.45]`
  - `gap_lo`: `[0.06, 0.10]`
  - `gap_pos_max`: `[0.16, 0.26]`
  - `drop_lo`: `[0.10, 0.14]`
  - `drop_hi`: `[0.22, 0.45]`
  - `flat_tol`: `[0.12, 0.20]`
  - `block_max_large`: `{6}`
  - `block_max_small`: `{3}`
  - `moderate_max_features`: `{12}`
  - `short_max_features`: `{8}`
  - `short_drop_min`: `[0.08, 0.12]`
  - `cum_lo`: `[0.25, 0.35]`
  - `cum_hi`: `[0.60, 0.85]`
  - `cum_late_flat_tol`: `[0.10, 0.20]`
  - `cum_prev_flat_tol`: `[0.04, 0.12]`
  - `forced_max_features`: `{6}`
  - `forced_contrast_lo`: `{0.01}`
  - `forced_contrast_hi`: `[1.00, 1.50]`
  - `forced_late_flat_tol`: `{0.60}`
  - `forced_prev_flat_tol`: `{0.15}`
  - `forced_altitude_lo`: `{0.01}`
  - `forced_gain`: `[0.65, 0.95]`
  - `forced_recency_slope`: `{3.0}`
  - `forced_recency_cap`: `{4.0}`
  - `forced_beta_mult`: `[1.00, 1.35]`
  - `forced_eps`: `[0.10, 0.17]`
  - `strength_min`: `[0.12, 0.22]`
  - `strength_high_extreme`: `[0.40, 0.65]`
  - `strength_contrast`: `[0.70, 1.10]`
  - `strength_flat`: `[0.12, 0.25]`
  - `contrast_ref`: `[0.30, 0.45]`
  - `contrast_power`: `[2.5, 4.0]`
  - `altitude_power`: `[0.8, 1.2]`
  - `flat_ref`: `[0.15, 0.30]`
  - `size_penalty`: `[0.04, 0.12]`
  - `gain_floor`: `[0.12, 0.18]`
  - `gain_cap`: `[1.20, 1.40]`
  - `recency_slope`: `[0.15, 0.40]`
  - `recency_cap`: `[0.10, 0.25]`
  - `beta_sub`: `[0.45, 0.65]`
  - `epsilon_sub`: `[0.08, 0.16]`
  - `epsilon_sub_size`: `[0.04, 0.09]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


### slot 2 — `pi_8` — KILLED ✗

**Description:** People make binary-feature choices by a tally-first rule. When the unweighted signed tally is clearly nonzero, choice follows the tally through a softmax plus a small lapse. When the tally is tied or near-tied, advertised validities are used only to perceptually segment the cue display, and choice follows a recency-weighted sub-tally of a coherent late block if one is formed. Four segmentation routes can open such a block: (1) a large validity jump opening a later block of up to six cues; (2) a modest positive end boundary isolating a small two-to-three-cue late block; (3) a modest negative end drop isolating a homogeneous late low-validity block of two to three cues; and (4) a cumulative-contrast route, restricted to short displays of up to eight cues, that opens a homogeneous final two-to-three-cue block when its mean validity sits sufficiently below the immediately preceding two-to-three cues. For six-cue displays specifically, a homogeneous early elite followed by a uniformly lower late block is treated as a distinct late block even when the adjacent validity jump is moderate. When no perceptual block is formed, the tied choice is low confidence and is near chance.

**Rationale:** This is a minimal-diff edit on the accepted iter-5 base, focused on the binding Experiment 10 failure. The previous candidate kept Experiment 10 near chance because its late low-validity block never opened on the top-plus-3 versus top-minus-3 tied trials. I made three targeted changes. First, the three-cue cumulative-contrast lower bound is reduced from [0.18, 0.22] to [0.10, 0.16], and the late-block homogeneity tolerance is widened to [0.25, 0.45], so more of the Experiment 10 validity pattern qualifies. To keep Experiment 1 protected, the two-cue cumulative-contrast route now uses a separate, higher lower bound, contrast_lo_2 [0.20, 0.24], which blocks Experiment 1's two-cue contrast of about 0.17. Second, the negative adjacent-drop route is broadened upward to drop_hi [0.19, 0.35] and is now gated to short displays of at most eight cues, closing the dead zone between the old negative-drop upper bound and the large-boundary threshold without disturbing large-display experiments. Third, I added a short six-cue fallback: if the first three cues form a homogeneous elite and the last three cues are uniformly lower, the last three cues are treated as a distinct late block. This directly opens Experiment 10's late low block while leaving Experiment 1 closed because its first three validities span about 0.16, above the tight elite-flatness bound. Tally-route parameters and the sub-tally strength were deliberately left untouched because prior attempts to strengthen the large-display tally route were rejected by the gate.

**Parameters:**
  - `beta_tally_small`: `[0.60, 0.95]`
  - `beta_tally_large`: `[0.95, 1.45]`
  - `epsilon_tally_small`: `[0.02, 0.06]`
  - `epsilon_tally_large`: `[0.02, 0.06]`
  - `beta_near`: `[0.15, 0.35]`
  - `epsilon_near`: `[0.75, 0.95]`
  - `near_bound`: `{1.5}`
  - `n_low_conf`: `{15}`
  - `tau_tie`: `{0.5}`
  - `gap_hi`: `[0.32, 0.42]`
  - `gap_lo`: `[0.05, 0.08]`
  - `gap_lo_max`: `[0.18, 0.25]`
  - `drop_lo`: `[0.11, 0.14]`
  - `drop_hi`: `[0.19, 0.35]`
  - `drop_flat_tol`: `[0.05, 0.12]`
  - `block_size_min_hi`: `{2}`
  - `block_size_max_hi`: `{6}`
  - `block_size_min_lo`: `{2}`
  - `block_size_max_lo`: `{3}`
  - `contrast_lo`: `[0.10, 0.16]`
  - `contrast_lo_2`: `[0.20, 0.24]`
  - `contrast_hi`: `[0.40, 0.55]`
  - `contrast_flat_tol`: `[0.25, 0.45]`
  - `contrast_prev_flat_tol`: `[0.04, 0.12]`
  - `contrast_max_features`: `{8}`
  - `short6_elite_flat`: `[0.02, 0.06]`
  - `short6_late_flat`: `[0.25, 0.50]`
  - `short6_gap`: `[0.03, 0.10]`
  - `rho`: `[0.40, 0.90]`
  - `beta_sub`: `[0.50, 0.68]`
  - `epsilon_sub`: `[0.14, 0.21]`
  - `epsilon_sub_size`: `[0.08, 0.13]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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
        base = (1.0 - eps) * core + eps * np.array([0.5, 0.5])
        base = np.clip(base, 0.005, 0.995)
        total = float(np.sum(base))
        if total <= 1e-12:
            return np.array([0.5, 0.5])
        return base / total

    beta_small = float(parameters['beta_tally_small'])
    beta_large = float(parameters['beta_tally_large'])
    eps_small = float(parameters['epsilon_tally_small'])
    eps_large = float(parameters['epsilon_tally_large'])
    frac = float(np.clip((n_features - 8.0) / 7.0, 0.0, 1.0))
    beta_tally = beta_small + frac * (beta_large - beta_small)
    epsilon_tally = eps_small + frac * (eps_large - eps_small)

    tau_tie = float(parameters['tau_tie'])
    gap_hi = float(parameters['gap_hi'])
    gap_lo = float(parameters['gap_lo'])
    gap_lo_max = float(parameters['gap_lo_max'])
    min_lo = int(parameters['block_size_min_lo'])
    max_lo = int(parameters['block_size_max_lo'])

    block_indices = None

    jumps = np.diff(validities)
    large_bounds = np.where(np.abs(jumps) >= gap_hi)[0]
    if large_bounds.size > 0:
        i = int(large_bounds[-1])
        size = n_features - (i + 1)
        min_hi = int(parameters['block_size_min_hi'])
        max_hi = int(parameters['block_size_max_hi'])
        if min_hi <= size <= max_hi:
            block_indices = np.arange(i + 1, i + 1 + size)

    if block_indices is None:
        pos_jumps = np.diff(validities)
        small_bounds = np.where((pos_jumps >= gap_lo) & (pos_jumps <= gap_lo_max))[0]
        if small_bounds.size > 0:
            i = int(small_bounds[-1])
            size = n_features - (i + 1)
            if min_lo <= size <= max_lo:
                block_indices = np.arange(i + 1, n_features)

    if block_indices is None and n_features <= int(parameters['contrast_max_features']):
        drop_lo = float(parameters['drop_lo'])
        drop_hi = float(parameters['drop_hi'])
        drop_flat_tol = float(parameters['drop_flat_tol'])
        neg_jumps = np.diff(validities)
        drop_bounds = np.where((neg_jumps <= -drop_lo) & (neg_jumps >= -drop_hi))[0]
        if drop_bounds.size > 0:
            for i in drop_bounds[::-1]:
                i = int(i)
                size = n_features - (i + 1)
                if min_lo <= size <= max_lo:
                    cand = np.arange(i + 1, n_features)
                    v_block = validities[cand]
                    if float(np.max(v_block) - np.min(v_block)) <= drop_flat_tol:
                        block_indices = cand
                        break

    if block_indices is None and n_features <= int(parameters['contrast_max_features']):
        contrast_lo_3 = float(parameters['contrast_lo'])
        contrast_lo_2 = float(parameters['contrast_lo_2'])
        contrast_hi = float(parameters['contrast_hi'])
        contrast_flat_tol = float(parameters['contrast_flat_tol'])
        contrast_prev_flat_tol = float(parameters['contrast_prev_flat_tol'])
        for k in (3, 2):
            if n_features < 2 * k:
                continue
            contrast_lo = contrast_lo_3 if k == 3 else contrast_lo_2
            prev = validities[-2 * k:-k]
            late = validities[-k:]
            contrast = float(np.mean(prev) - np.mean(late))
            late_range = float(np.max(late) - np.min(late))
            prev_range = float(np.max(prev) - np.min(prev))
            if (contrast_lo <= contrast <= contrast_hi and
                    late_range <= contrast_flat_tol and
                    prev_range <= contrast_prev_flat_tol):
                block_indices = np.arange(n_features - k, n_features)
                break

    if block_indices is None and n_features == 6:
        early = validities[:3]
        late = validities[3:]
        early_range = float(np.max(early) - np.min(early))
        late_range = float(np.max(late) - np.min(late))
        gap = float(np.min(early) - np.max(late))
        if (early_range <= float(parameters['short6_elite_flat']) and
                late_range <= float(parameters['short6_late_flat']) and
                gap >= float(parameters['short6_gap'])):
            block_indices = np.arange(3, 6)

    if abs(tally) <= tau_tie:
        if block_indices is not None and block_indices.size > 0:
            rho = float(parameters['rho'])
            pos_sel = positions[block_indices]
            w = np.exp(rho * (pos_sel - np.max(pos_sel)))
            if np.sum(w) <= 1e-12:
                w = np.ones_like(w)
            w = w / np.sum(w) * float(block_indices.size)

            sub_tally = float(np.dot(w, diff[block_indices]))
            beta_sub = float(parameters['beta_sub'])
            eps_sub = float(parameters['epsilon_sub'])
            eps_sub_size = float(parameters['epsilon_sub_size'])
            extra = max(0.0, float(block_indices.size) - 3.0)
            eps_eff = float(np.clip(eps_sub + eps_sub_size * extra, 0.0, 0.95))

            if abs(sub_tally) > 1e-12:
                core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_sub)
                probs = (1.0 - eps_eff) * core + eps_eff * np.array([0.5, 0.5])
                probs = np.clip(probs, 0.005, 0.995)
                return probs / np.sum(probs)
            return np.array([0.5, 0.5])
        return np.array([0.5, 0.5])

    near_bound = float(parameters['near_bound'])
    n_low_conf = int(parameters['n_low_conf'])
    low_conf = (n_features >= n_low_conf and abs(tally) <= near_bound)

    if low_conf:
        beta = float(parameters['beta_near'])
        eps = float(parameters['epsilon_near'])
    else:
        beta = beta_tally
        eps = epsilon_tally

    core = stable_softmax(np.array([tally, -tally]), beta)
    return lapse_mix(core, eps)
```

**`policy(probs)`:**
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


## Replacement

### `pi_10` → slot 2 (via `new_theory`)

**Description:** People make binary-feature choices by a primary unweighted signed tally. When the tally is clear, choice follows the tally through a softmax with a small lapse. When the tally is tied or near-tied, people parse the advertised validity display into a single contiguous terminal chunk only when that chunk is coherent, i.e. its validity profile is flat and bounded by a clear validity discontinuity. Inside an opened terminal chunk the choice is based on an unweighted or validity-ordered sub-tally, never on positive exponential recency. When no coherent terminal chunk opens on a short display, people pool the earliest/largest coherent preceding block and choose by its sub-tally, producing directional anti-recency on late-block metrics. A targeted forced late-pool route is retained for very short displays in which an early homogeneous high-validity block is followed by a lower homogeneous late block.

**Rationale:** This theory keeps the unweighted tally as the primary rule, preserving the strong clear-tally fits from the previous best models. On ties it replaces the late-block exponential recency term with flatness-gated terminal chunking and unweighted or validity-ordered sub-tallies. The terminal chunks are opened only when the advertised validity profile is coherent, which prevents the model from over-using high-validity late blocks in large or graded displays. Critically, when no such coherent late chunk opens, short displays use an early-block pooling fallback that selects a coherent preceding block and chooses by its sub-tally, producing the directional anti-recency needed for the late-block metrics in Experiments 14, 15, and 16 while leaving the chance-like tie behavior in larger displays such as Experiments 5 and 6 intact. The forced late-pool route is retained only for very short displays with a homogeneous high-validity early block followed by a lower homogeneous late block, which preserves the strong anti-top-cue behavior in Experiment 10 without using exponential recency.

**Parameters:**
  - `beta_tally_small`: `[0.50, 0.75]`
  - `beta_tally_large`: `[0.85, 1.25]`
  - `epsilon_tally_small`: `[0.02, 0.06]`
  - `epsilon_tally_large`: `[0.02, 0.06]`
  - `beta_near`: `[0.18, 0.38]`
  - `epsilon_near`: `[0.75, 0.95]`
  - `near_bound`: `{1.5}`
  - `n_low_conf`: `{15}`
  - `tau_tie`: `{0.5}`
  - `gap_hi`: `[0.34, 0.45]`
  - `gap_lo`: `[0.06, 0.10]`
  - `gap_pos_max`: `[0.16, 0.26]`
  - `drop_lo`: `[0.10, 0.14]`
  - `drop_hi`: `[0.22, 0.45]`
  - `flat_tol`: `[0.10, 0.18]`
  - `block_max_large`: `{6}`
  - `block_max_small`: `{3}`
  - `moderate_max_features`: `{12}`
  - `short_max_features`: `{8}`
  - `short_drop_min`: `[0.08, 0.12]`
  - `cum_lo`: `[0.25, 0.35]`
  - `cum_hi`: `[0.60, 0.85]`
  - `cum_late_flat_tol`: `[0.08, 0.18]`
  - `cum_prev_flat_tol`: `[0.04, 0.12]`
  - `forced_max_features`: `{6}`
  - `forced_contrast_lo`: `{0.01}`
  - `forced_contrast_hi`: `[0.90, 1.50]`
  - `forced_late_flat_tol`: `{0.60}`
  - `forced_prev_flat_tol`: `{0.15}`
  - `forced_altitude_lo`: `{0.01}`
  - `forced_gain`: `[0.65, 0.95]`
  - `forced_beta_mult`: `[1.00, 1.35]`
  - `forced_eps`: `[0.10, 0.17]`
  - `strength_min`: `[0.12, 0.22]`
  - `strength_high_extreme`: `[0.40, 0.65]`
  - `strength_contrast`: `[0.70, 1.10]`
  - `strength_flat`: `[0.12, 0.25]`
  - `contrast_ref`: `[0.30, 0.45]`
  - `contrast_power`: `[2.5, 4.0]`
  - `altitude_power`: `[0.8, 1.2]`
  - `flat_ref`: `[0.15, 0.30]`
  - `size_penalty`: `[0.04, 0.12]`
  - `gain_floor`: `[0.12, 0.18]`
  - `gain_cap`: `[1.20, 1.40]`
  - `beta_sub`: `[0.45, 0.65]`
  - `epsilon_sub`: `[0.08, 0.16]`
  - `epsilon_sub_size`: `[0.04, 0.09]`
  - `val_weight_power`: `[0.5, 2.0]`
  - `val_weight_floor`: `[0.02, 0.08]`
  - `pool_flat_tol`: `[0.10, 0.20]`
  - `pool_min_size`: `{2}`
  - `pool_rank`: `{earliest, largest}`
  - `pool_beta`: `[0.40, 0.80]`
  - `pool_eps`: `[0.18, 0.32]`
  - `fallback_min_features`: `{8}`
  - `fallback_max_features`: `{11}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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
