# Round 9 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_11` — KILLED ✗

**Description:** People make binary-feature choices by a primary unweighted signed tally. When the full tally is clear, choice follows that tally through a stochastic choice rule with a small lapse. When the tally is near-tied, people parse the advertised validity vector into contiguous coherent runs and resolve the choice with one selected run. Priority goes to coherent terminal segments opened by validity discontinuities, including forced late-triplet openings on short displays; if no usable terminal segment opens, people fall back to the earliest coherent preceding segment. Inside the selected segment all features receive equal weight: validity differences affect only whether a segment is recognized and how decisively its sub-tally drives choice, never the relative cue weights within the segment. Calibration rules make flat, clearly contrasted forced late triplets on very short displays strongly decisive, moderate-jump high-altitude terminal runs on nine-to-ten-cue displays decisive, large-display forced openings and large-display low-altitude fallbacks near chance, and size-six high-altitude terminal runs moderately damped.

**Rationale:** This is a minimal calibration continuation of the accepted equal-weight boundary-parsing candidate. The primary full-tally route and equal weights inside selected segments are unchanged. The edit applies the latest critic feedback: (1) size-six high-altitude terminal damping is set to a midpoint (higher beta, lower lapse) to lift E3 from 1.69 toward roughly 2.6; (2) the n=9-10 moderate-jump high-altitude route is separated by an explicit higher-lapse/lower-beta calibration for size<=3 terminals to bring E4 down from 5.05 toward 4.20, while the size>=4 moderate-decisive route is strengthened for E18; (3) flat, clearly contrasted forced late triplets on very short n<=6 displays are now priority-promoted and made much more decisive for E10; (4) large n>=12 forced openings are pushed closer to chance for E11; and (5) on n>=8 displays, small low-altitude terminal segments are replaced by the earliest coherent preceding fallback, which should move E13 toward zero and activate the anti-late earliest-segment behavior needed for E16. No validity-weighted cue integration or recency ramp is introduced.

**Parameters:**
  - `beta_tally`: `[0.38, 0.52]`
  - `epsilon_tally`: `[0.02, 0.06]`
  - `beta_near`: `[0.18, 0.38]`
  - `epsilon_near`: `[0.75, 0.95]`
  - `near_bound`: `{1.5}`
  - `n_low_conf`: `{15}`
  - `tau_tie`: `{2.5}`
  - `flat_tol`: `[0.13, 0.17]`
  - `min_run_len`: `{2}`
  - `min_term_len`: `{3}`
  - `term_jump_hi`: `[0.22, 0.34]`
  - `term_low_max_size`: `{3}`
  - `term_jump_lo`: `[0.06, 0.10]`
  - `moderate_term_n`: `{12}`
  - `forced_n`: `{12}`
  - `forced_loose_n`: `{8}`
  - `forced_contrast_lo`: `[0.06, 0.14]`
  - `forced_prev_flat_tol`: `[0.15, 0.22]`
  - `forced_late_flat_tol`: `[0.22, 0.30]`
  - `forced_late_flat_tol_short`: `[0.38, 0.55]`
  - `forced_short_contrast`: `[0.20, 0.28]`
  - `preceding_min_len`: `{2}`
  - `seg_beta`: `[0.60, 0.72]`
  - `beta_floor`: `{0.05}`
  - `beta_cap`: `{1.20}`
  - `gain_min`: `[0.12, 0.20]`
  - `gain_contrast`: `[0.95, 1.15]`
  - `gain_flat`: `[0.20, 0.35]`
  - `contrast_ref`: `[0.25, 0.35]`
  - `flat_ref`: `[0.12, 0.18]`
  - `contrast_power`: `[1.0, 1.5]`
  - `forced_contrast_power`: `[2.2, 3.2]`
  - `size_penalty`: `[1.80, 2.40]`
  - `size_penalty_high`: `[0.25, 0.45]`
  - `size_penalty_low`: `[0.10, 0.30]`
  - `high_altitude_threshold`: `[0.65, 0.80]`
  - `high_penalty_start`: `{6}`
  - `moderate_gain_mult`: `[1.10, 1.30]`
  - `moderate_high_min_n`: `{9}`
  - `moderate_high_n`: `{10}`
  - `moderate_high_max_size`: `{3}`
  - `moderate_high_gain_mult`: `[1.30, 1.60]`
  - `moderate_high_beta_mult`: `[1.10, 1.40]`
  - `moderate_high_eps`: `[0.15, 0.25]`
  - `nonflat_gain_mult`: `[0.30, 0.45]`
  - `low_gain_mult`: `[0.55, 0.85]`
  - `low_short_n`: `{10}`
  - `low_short_max_size`: `{7}`
  - `low_short_gain_mult`: `[1.15, 1.40]`
  - `low_short_beta_mult`: `[0.85, 1.10]`
  - `low_short_eps_reduce`: `[0.05, 0.12]`
  - `gain_floor`: `{0.08}`
  - `gain_cap`: `{2.0}`
  - `eps_terminal_strong`: `[0.10, 0.14]`
  - `eps_terminal_moderate`: `[0.08, 0.14]`
  - `eps_terminal_low`: `[0.15, 0.25]`
  - `eps_forced`: `[0.35, 0.48]`
  - `eps_fallback`: `[0.07, 0.14]`
  - `eps_size`: `[0.05, 0.07]`
  - `eps_size_cap`: `{1}`
  - `nonflat_eps_add`: `[0.18, 0.28]`
  - `eps_high_altitude_add`: `[0.04, 0.07]`
  - `long_high_min_size`: `{8}`
  - `long_high_eps_add`: `[0.70, 0.90]`
  - `low_beta_mult`: `[0.55, 0.85]`
  - `forced_beta_mult`: `[0.75, 0.95]`
  - `forced_short_gain_mult`: `[1.15, 1.45]`
  - `forced_short_beta_mult`: `[1.05, 1.30]`
  - `forced_short_eps`: `[0.20, 0.30]`
  - `forced_large_beta_mult`: `[0.05, 0.15]`
  - `forced_large_eps`: `[0.80, 0.95]`
  - `fallback_beta_mult`: `[1.70, 2.10]`
  - `size6_high_beta_mult`: `[0.65, 0.85]`
  - `size6_high_lapse_add`: `[0.25, 0.40]`
  - `moderate_large_gate_n`: `{11}`
  - `moderate_decisive_min_n`: `{9}`
  - `moderate_decisive_max_n`: `{10}`
  - `moderate_decisive_min_size`: `{4}`
  - `moderate_decisive_beta_mult`: `[1.80, 2.40]`
  - `moderate_decisive_eps`: `[0.02, 0.08]`
  - `moderate_eps_damp_max_n`: `{10}`
  - `eps_terminal_moderate_lowalt`: `[0.30, 0.45]`
  - `forced_short_decisive_n`: `{6}`
  - `forced_short_beta_mult_strong`: `[2.50, 4.00]`
  - `forced_short_eps_strong`: `[0.02, 0.08]`
  - `long_high_beta_mult`: `[0.35, 0.60]`
  - `fallback_large_min_n`: `{12}`
  - `fallback_large_beta_mult`: `[0.40, 0.60]`
  - `fallback_large_eps`: `[0.70, 0.85]`
  - `small_term_prefer_fallback_n`: `{8}`
  - `small_term_max_size`: `{3}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float)
    n = stim.shape[1]
    if validities.shape[0] != n:
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

    tau_tie = float(parameters['tau_tie'])
    if abs(tally) > tau_tie:
        beta_tally = float(parameters['beta_tally'])
        epsilon_tally = float(parameters['epsilon_tally'])
        if n >= int(parameters['n_low_conf']) and abs(tally) <= float(parameters['near_bound']):
            beta = float(parameters['beta_near'])
            eps = float(parameters['epsilon_near'])
        else:
            beta = beta_tally
            eps = epsilon_tally
        core = stable_softmax(np.array([tally, -tally]), beta)
        return lapse_mix(core, eps)

    flat_tol = float(parameters['flat_tol'])
    min_run_len = int(parameters['min_run_len'])
    runs = []
    run_start = 0
    for i in range(1, n):
        seg = validities[run_start:i + 1]
        if float(np.max(seg) - np.min(seg)) > flat_tol:
            if i - run_start >= min_run_len:
                runs.append((run_start, i))
            run_start = i
    if n - run_start >= min_run_len:
        runs.append((run_start, n))

    vmin = float(np.min(validities))
    vmax = float(np.max(validities))
    spread = max(vmax - vmin, 1e-9)

    def altitude_of(start, size):
        vals = validities[start:start + size]
        return float(np.clip((float(np.mean(vals)) - vmin) / spread, 0.0, 1.0))

    gain_min = float(parameters['gain_min'])
    gain_contrast = float(parameters['gain_contrast'])
    gain_flat = float(parameters['gain_flat'])
    contrast_ref = float(parameters['contrast_ref'])
    flat_ref = float(parameters['flat_ref'])
    contrast_power = float(parameters['contrast_power'])
    size_penalty = float(parameters['size_penalty'])
    size_penalty_high = float(parameters['size_penalty_high'])
    size_penalty_low = float(parameters['size_penalty_low'])
    high_altitude_threshold = float(parameters['high_altitude_threshold'])
    high_penalty_start = int(parameters['high_penalty_start'])
    moderate_gain_mult = float(parameters['moderate_gain_mult'])
    nonflat_gain_mult = float(parameters['nonflat_gain_mult'])
    low_gain_mult = float(parameters['low_gain_mult'])
    gain_floor = float(parameters['gain_floor'])
    gain_cap = float(parameters['gain_cap'])

    forced_limit_n = int(parameters['forced_n'])
    forced_loose_n = int(parameters['forced_loose_n'])
    forced_contrast_power = float(parameters['forced_contrast_power'])
    forced_late_flat_tol_short = float(parameters['forced_late_flat_tol_short'])
    forced_short_contrast = float(parameters['forced_short_contrast'])
    forced_short_gain_mult = float(parameters['forced_short_gain_mult'])
    forced_short_beta_mult = float(parameters['forced_short_beta_mult'])
    forced_short_eps = float(parameters['forced_short_eps'])
    forced_short_decisive_n = int(parameters['forced_short_decisive_n'])
    forced_short_beta_mult_strong = float(parameters['forced_short_beta_mult_strong'])
    forced_short_eps_strong = float(parameters['forced_short_eps_strong'])
    forced_large_eps = float(parameters['forced_large_eps'])
    forced_large_beta_mult = float(parameters['forced_large_beta_mult'])
    moderate_high_min_n = int(parameters['moderate_high_min_n'])
    moderate_high_n = int(parameters['moderate_high_n'])
    moderate_high_max_size = int(parameters['moderate_high_max_size'])
    moderate_high_gain_mult = float(parameters['moderate_high_gain_mult'])
    moderate_high_beta_mult = float(parameters['moderate_high_beta_mult'])
    moderate_high_eps = float(parameters['moderate_high_eps'])
    low_short_n = int(parameters['low_short_n'])
    low_short_max_size = int(parameters['low_short_max_size'])
    low_short_gain_mult = float(parameters['low_short_gain_mult'])
    low_short_beta_mult = float(parameters['low_short_beta_mult'])
    low_short_eps_reduce = float(parameters['low_short_eps_reduce'])
    size6_high_beta_mult = float(parameters['size6_high_beta_mult'])
    size6_high_lapse_add = float(parameters['size6_high_lapse_add'])
    moderate_large_gate_n = int(parameters['moderate_large_gate_n'])
    moderate_decisive_min_n = int(parameters['moderate_decisive_min_n'])
    moderate_decisive_max_n = int(parameters['moderate_decisive_max_n'])
    moderate_decisive_min_size = int(parameters['moderate_decisive_min_size'])
    moderate_eps_damp_max_n = int(parameters['moderate_eps_damp_max_n'])
    eps_terminal_moderate_lowalt = float(parameters['eps_terminal_moderate_lowalt'])
    long_high_beta_mult = float(parameters['long_high_beta_mult'])
    small_term_prefer_fallback_n = int(parameters['small_term_prefer_fallback_n'])
    small_term_max_size = int(parameters['small_term_max_size'])

    def make_candidate(start, size, kind, jump, contrast, block_vals, priority, flat):
        block_range = float(np.max(block_vals) - np.min(block_vals))
        altitude = altitude_of(start, size)
        flat_norm = float(np.clip(1.0 - block_range / flat_ref, 0.0, 1.0))
        if kind in ('terminal_strong', 'terminal_moderate', 'terminal_low', 'forced'):
            contrast_norm = float(np.clip(contrast / contrast_ref, 0.0, 1.0))
            contrast_pow = forced_contrast_power if kind == 'forced' else contrast_power
            gain = gain_min + gain_contrast * (contrast_norm ** contrast_pow) + gain_flat * flat_norm
        else:
            gain = gain_min + gain_flat * flat_norm

        if kind == 'terminal_moderate':
            if (n >= moderate_high_min_n and n <= moderate_high_n and
                    size <= moderate_high_max_size and altitude >= high_altitude_threshold):
                gain *= moderate_high_gain_mult
            else:
                gain *= moderate_gain_mult

        if kind == 'terminal_low':
            if n <= low_short_n and size <= low_short_max_size:
                gain *= low_short_gain_mult
            else:
                gain *= low_gain_mult

        if kind == 'forced' and not flat:
            if n <= forced_loose_n and contrast >= forced_short_contrast:
                gain *= forced_short_gain_mult
            else:
                gain *= nonflat_gain_mult

        if altitude >= high_altitude_threshold:
            penalty = size_penalty_high * max(0.0, float(size) - float(high_penalty_start) + 1.0)
        elif kind == 'terminal_low':
            penalty = size_penalty_low * max(0.0, float(size) - 3.0)
        else:
            penalty = size_penalty * max(0.0, float(size) - 3.0)
        if penalty > 0.0:
            gain /= (1.0 + penalty)

        return {
            'start': start,
            'size': size,
            'kind': kind,
            'jump': jump,
            'contrast': contrast,
            'block_range': block_range,
            'altitude': altitude,
            'priority': priority,
            'flat': flat,
            'gain': float(np.clip(gain, gain_floor, gain_cap))
        }

    min_term_len = int(parameters['min_term_len'])
    term_jump_hi = float(parameters['term_jump_hi'])
    term_jump_lo = float(parameters['term_jump_lo'])
    moderate_term_n = int(parameters['moderate_term_n'])
    term_low_max_size = int(parameters['term_low_max_size'])

    candidates = []
    for (rs, re) in runs:
        if re != n:
            continue
        size = re - rs
        if size < min_term_len:
            continue
        if rs == 0:
            continue
        jump = float(validities[rs] - validities[rs - 1])
        block_vals = validities[rs:re]
        contrast = float(abs(jump))
        if abs(jump) >= term_jump_hi:
            altitude = altitude_of(rs, size)
            if altitude >= high_altitude_threshold or size <= term_low_max_size:
                candidates.append(make_candidate(rs, size, 'terminal_strong', jump, contrast, block_vals, 4, True))
            else:
                candidates.append(make_candidate(rs, size, 'terminal_low', jump, contrast, block_vals, 2, True))
        elif n <= moderate_term_n and abs(jump) >= term_jump_lo:
            if n < moderate_large_gate_n or size <= 3:
                if (moderate_decisive_min_n <= n <= moderate_decisive_max_n and
                        size >= moderate_decisive_min_size and
                        altitude_of(rs, size) >= high_altitude_threshold):
                    cand = make_candidate(rs, size, 'terminal_strong', jump, contrast, block_vals, 4, True)
                    cand['moderate_decisive'] = True
                    candidates.append(cand)
                else:
                    candidates.append(make_candidate(rs, size, 'terminal_moderate', jump, contrast, block_vals, 3, True))

    if n <= forced_limit_n:
        forced_contrast_lo = float(parameters['forced_contrast_lo'])
        forced_prev_flat_tol = float(parameters['forced_prev_flat_tol'])
        forced_late_flat_tol = float(parameters['forced_late_flat_tol'])
        late_flat_tol_use = forced_late_flat_tol
        if n <= forced_loose_n:
            late_flat_tol_use = max(late_flat_tol_use, forced_late_flat_tol_short)

        for k in (3,):
            if n < 2 * k:
                continue
            prev = validities[n - 2 * k:n - k]
            late = validities[n - k:]
            contrast = float(np.mean(prev) - np.mean(late))
            prev_range = float(np.max(prev) - np.min(prev))
            late_range = float(np.max(late) - np.min(late))
            if contrast >= forced_contrast_lo and prev_range <= forced_prev_flat_tol and late_range <= late_flat_tol_use:
                flat = late_range <= flat_tol
                forced_priority = 5 if (n <= forced_short_decisive_n and flat and contrast >= forced_short_contrast) else 4
                candidates.append(make_candidate(n - k, k, 'forced', -contrast, contrast, late, forced_priority, flat))

        for k in (3,):
            if n < 2 * k:
                continue
            early = validities[:k]
            late = validities[-k:]
            contrast = float(np.mean(early) - np.mean(late))
            early_range = float(np.max(early) - np.min(early))
            late_range = float(np.max(late) - np.min(late))
            if contrast >= forced_contrast_lo and early_range <= forced_prev_flat_tol and late_range <= late_flat_tol_use:
                flat = late_range <= flat_tol
                forced_priority = 5 if (n <= forced_short_decisive_n and flat and contrast >= forced_short_contrast) else 4
                candidates.append(make_candidate(n - k, k, 'forced', -contrast, contrast, late, forced_priority, flat))

    fallback = None
    preceding_min_len = int(parameters['preceding_min_len'])
    preceding = []
    for (rs, re) in runs:
        if re < n and re - rs >= preceding_min_len:
            preceding.append((rs, re))
    if preceding:
        rs, re = min(preceding, key=lambda x: (x[0], -(x[1] - x[0])))
        block_vals = validities[rs:re]
        fallback = make_candidate(rs, re - rs, 'fallback', 0.0, 0.0, block_vals, 0, True)

    if not candidates and fallback is None:
        return np.array([0.5, 0.5])

    if candidates:
        c = max(candidates, key=lambda d: (d['priority'], d['gain'], d['contrast'], -d['size']))
        if (n >= small_term_prefer_fallback_n and fallback is not None and
                c['kind'] in ('terminal_strong', 'terminal_moderate', 'terminal_low') and
                c['size'] <= small_term_max_size and
                c['altitude'] < high_altitude_threshold):
            c = fallback
    else:
        c = fallback

    idx = np.arange(c['start'], c['start'] + c['size'], dtype=int)
    sub_tally = float(np.sum(diff[idx]))
    if abs(sub_tally) <= 1e-12:
        return np.array([0.5, 0.5])

    seg_beta = float(parameters['seg_beta'])
    beta_floor = float(parameters['beta_floor'])
    beta_cap = float(parameters['beta_cap'])
    beta_eff = float(np.clip(seg_beta * c['gain'], beta_floor, beta_cap))

    eps_size = float(parameters['eps_size'])
    eps_size_cap = int(parameters['eps_size_cap'])
    size_extra = min(max(0.0, float(c['size'] - 3)), float(eps_size_cap))

    forced_beta_mult = float(parameters['forced_beta_mult'])
    low_beta_mult = float(parameters['low_beta_mult'])
    eps_terminal_low = float(parameters['eps_terminal_low'])
    fallback_large_min_n = int(parameters['fallback_large_min_n'])
    fallback_large_beta_mult = float(parameters['fallback_large_beta_mult'])
    fallback_large_eps = float(parameters['fallback_large_eps'])
    moderate_decisive_beta_mult = float(parameters['moderate_decisive_beta_mult'])
    moderate_decisive_eps = float(parameters['moderate_decisive_eps'])

    short_forced = (c['kind'] == 'forced' and n <= forced_loose_n and
                    c['size'] == 3 and c['contrast'] >= forced_short_contrast)
    strong_short_forced = (short_forced and c['flat'] and n <= forced_short_decisive_n)

    if c['kind'] == 'fallback':
        fallback_beta_mult = float(parameters['fallback_beta_mult'])
        beta_eff *= fallback_beta_mult
        eps = float(parameters['eps_fallback'])
        if n >= fallback_large_min_n and c['altitude'] < high_altitude_threshold:
            beta_eff *= fallback_large_beta_mult
            eps = fallback_large_eps
    elif c['kind'] == 'forced':
        if strong_short_forced:
            beta_eff *= forced_short_beta_mult_strong
            eps = forced_short_eps_strong
        elif short_forced:
            beta_eff *= forced_short_beta_mult
            eps = forced_short_eps
        else:
            beta_eff *= forced_beta_mult
            eps = float(parameters['eps_forced'])
            if not c['flat']:
                eps += float(parameters['nonflat_eps_add'])
            if n >= 12:
                beta_eff *= forced_large_beta_mult
                eps = forced_large_eps
    elif c['kind'] == 'terminal_moderate':
        if (n >= moderate_high_min_n and n <= moderate_high_n and
                c['size'] <= moderate_high_max_size and c['altitude'] >= high_altitude_threshold):
            beta_eff *= moderate_high_beta_mult
            eps = moderate_high_eps
        elif n <= moderate_eps_damp_max_n and c['altitude'] < high_altitude_threshold:
            eps = eps_terminal_moderate_lowalt
        else:
            eps = float(parameters['eps_terminal_moderate']) + eps_size * size_extra
    elif c['kind'] == 'terminal_low':
        if n <= low_short_n and c['size'] <= low_short_max_size:
            beta_eff *= low_short_beta_mult
            eps = max(0.0, eps_terminal_low - low_short_eps_reduce) + eps_size * size_extra
        else:
            beta_eff *= low_beta_mult
            eps = eps_terminal_low + eps_size * size_extra
    elif c['kind'] == 'terminal_strong':
        if c.get('moderate_decisive'):
            beta_eff *= moderate_decisive_beta_mult
            eps = moderate_decisive_eps
        else:
            eps = float(parameters['eps_terminal_strong']) + eps_size * size_extra
            if c['altitude'] >= high_altitude_threshold:
                eps += float(parameters['eps_high_altitude_add'])
                if c['size'] >= int(parameters['long_high_min_size']):
                    eps += float(parameters['long_high_eps_add'])
                    beta_eff *= long_high_beta_mult
                if c['size'] == 6:
                    beta_eff *= size6_high_beta_mult
                    eps += size6_high_lapse_add

    eps = float(np.clip(eps, 0.0, 0.97))
    core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_eff)
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


### slot 2 — `pi_10` — SURVIVED ✓

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


## Replacement

### `pi_12` → slot 1 (via `new_theory`)

**Description:** People make binary-feature choices by first forming an unweighted signed tally. When the full tally is nonzero, choice follows the tally through a softmax whose sensitivity grows with tally magnitude, with only a small lapse. On exact-tie trials people abandon the unweighted tally and use a single global compressed-validity tally, but the sign of that weighted tally is gated by the global organization of the advertised validity display. The gate combines directional coherence, sharpness of the largest adjacent validity change, and a drop-concentration statistic: it reverses the validity-weighted tally only when adjacent validity changes point consistently in one direction, a single large adjacent drop exceeds a substantial fraction of the validity range, and that drop dominates total validity variation. Mild, graded, or incoherent displays mostly follow the validity-weighted tally or remain near chance. Very large homogeneous tie conflicts are damped by the number of active differing features beyond ten. The model contains no run detection, no terminal priority, no forced late-triplet openings, no fallback pooling, and no trial-history recency or anti-recency mechanism.

**Rationale:** Minimal edit on the accepted iter-8 base. The gate now requires a sharp large adjacent validity drop before reversal can engage, and the concentration threshold is raised to 0.55-0.75. This is intended to stop the incorrect Exp18 reversal while preserving the single-boundary reversals in Exp10/Exp12. Large-conflict damping is strengthened to 0.30-0.45 so all-12-differing ties in Exp3/Exp6/Exp11 are pulled toward chance or desaturated. Non-tie sensitivity is increased only through beta_tally_slope to 0.18-0.26 to make large unsigned tallies more decisive without raising sensitivity for magnitude-one tallies. Tie sensitivity is modestly increased and tie lapse lowered to strengthen Exp4's ten-feature graded ties. The gate also squares the directional-coherence term once more, so incoherent validity displays such as Exp7 remain near chance instead of overcommitting to the validity-weighted tally. No excluded mechanisms were added.

**Parameters:**
  - `beta_tally_base`: `[0.10, 0.18]`
  - `beta_tally_slope`: `[0.18, 0.26]`
  - `epsilon_tally`: `[0.02, 0.06]`
  - `beta_tie`: `[0.65, 0.85]`
  - `epsilon_tie`: `[0.10, 0.14]`
  - `weight_floor`: `[0.01, 0.04]`
  - `gamma`: `[0.55, 0.70]`
  - `trend_theta`: `[0.55, 0.75]`
  - `trend_kappa`: `[8.0, 14.0]`
  - `large_drop_frac`: `[0.26, 0.34]`
  - `damp_lambda`: `[0.30, 0.45]`
  - `damp_threshold`: `{10}`
  - `damp_power`: `{3}`
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

    if abs(tally) > 1e-12:
        beta_tally_base = float(parameters['beta_tally_base'])
        beta_tally_slope = float(parameters['beta_tally_slope'])
        epsilon_tally = float(parameters['epsilon_tally'])

        abs_tally = abs(tally)
        beta = beta_tally_base + beta_tally_slope * max(0.0, abs_tally - 1.0)
        beta = float(np.clip(beta, 0.02, 3.0))

        core = stable_softmax(np.array([tally, -tally]), beta)
        return lapse_mix(core, epsilon_tally)

    vmin = float(np.min(validities))
    spread = max(float(np.max(validities)) - vmin, 1e-9)

    dv = np.diff(validities)
    dv_nz = dv[np.abs(dv) > 1e-9]
    if dv_nz.size == 0:
        coherence = 0.0
        concentration = 0.0
        max_rel_drop = 0.0
    else:
        direction_agreement = float(np.mean(np.sign(dv_nz)))
        coherence = float(np.clip(direction_agreement * direction_agreement, 0.0, 1.0))
        total_change = float(np.sum(np.abs(dv_nz)))
        concentration = float(np.max(np.abs(dv_nz))) / total_change if total_change > 1e-9 else 0.0
        max_rel_drop = float(np.max(np.abs(dv_nz))) / spread if spread > 1e-9 else 0.0

    half = max(1, n_features // 2)
    early = validities[:half]
    late = validities[-half:]
    decline_sign = float(np.sign(np.mean(early) - np.mean(late)))

    trend_kappa = float(parameters['trend_kappa'])
    concentration_theta = float(parameters['trend_theta'])
    large_drop_frac = float(parameters['large_drop_frac'])

    coherence_gate = coherence * coherence
    if max_rel_drop > large_drop_frac:
        gate = coherence_gate * float(np.tanh(trend_kappa * (concentration_theta - concentration * decline_sign)))
    else:
        gate = coherence_gate

    gamma = float(parameters['gamma'])
    weight_floor = float(parameters['weight_floor'])
    w = weight_floor + np.power(np.maximum(validities - vmin, 0.0), gamma)
    W = float(np.dot(w, diff))

    n_active = int(np.count_nonzero(diff))
    damp_lambda = float(parameters['damp_lambda'])
    damp_threshold = float(parameters['damp_threshold'])
    damp_power = float(parameters['damp_power'])
    damp = 1.0 + damp_lambda * np.power(max(0.0, n_active - damp_threshold), damp_power)

    score = W * gate / damp

    if abs(score) <= 1e-12:
        return np.array([0.5, 0.5])

    beta_tie = float(np.clip(float(parameters['beta_tie']), 0.02, 3.0))
    epsilon_tie = float(parameters['epsilon_tie'])

    core = stable_softmax(np.array([score, -score]), beta_tie)
    return lapse_mix(core, epsilon_tie)
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
