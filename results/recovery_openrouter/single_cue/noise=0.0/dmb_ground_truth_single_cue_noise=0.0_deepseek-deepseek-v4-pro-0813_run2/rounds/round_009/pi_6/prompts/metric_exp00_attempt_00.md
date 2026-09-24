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
**Validities (n_features=6):** [0.9, 0.8, 0.72, 0.62, 0.55, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1]
  trial 5: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 8: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 9: A=[1, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1, 0]
  trial 12: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 14: A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 15: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 16: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 17: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 18: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 19: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 20: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 21: A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 22: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 23: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 24: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 25: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 26: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 27: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 28: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 29: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 30: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 31: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 32: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0]

**Rationale:** This design isolates a trigger-definition dissociation between the advocated frequency-trigger theory and the competing leaky-accumulator theory. Validities are spread with a unique highest cue (feature 0, 0.90) and a unique lowest cue (feature 5, 0.50). The 8 critical pairs are fully diagnostic (all six experts disagree) and near-tied (|tally| = 2), with the highest and lowest cues in conflict and the tally opposing the highest cue: top-A d=-2 and top-B d=+2. These trials satisfy the advocated trigger exactly (n_discrim >= n and |tally| <= tie_margin), so with 8/32 = 25% qualifying unique pairs the trailing 24-trial qualifying fraction sits in the peak band and the backward amplification s reaches approximately 0.84-0.92. The competing accumulator assigns zero evidence to these same trials because its evidence gate requires the tally to align with the highest cue (the special 5-of-6 opposing branch does not apply to fully diagnostic trials). The 24 fillers are all non-qualifying for the advocated trigger (at least one expert tied) and zero-evidence for the competing accumulator (hi-lo aligned, or 5-of-6 conflict with |tally|=3 opposing the top cue, or 4-of-6), so competing S remains 0 throughout. On the critical probes the advocated model predicts P(A) approximately 0.03-0.11 for top-A d=-2 and approximately 0.89-0.97 for top-B d=+2, whereas the competing model predicts approximately 0.20-0.28 and 0.72-0.80. This is a non-overlapping quantitative dissociation driven solely by whether tally-opposing fully diagnostic conflict trials are treated as context triggers.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People choose between two products through a stable, subject-level mixture of three routes: a weak forward lexicographic scan, a moderate backward/recency lexicographic scan, and feature-win tallying. Choice is modulated by a context-sensitive amplification of a higher-precision backward lexicographic route. The decision maker monitors the trailing 24-trial frequency of fully diagnostic, near-tied-tally comparisons, ignores isolated occurrences via a minimum qualifying count, and maps the qualifying fraction through a band-limited trapezoidal sensitivity curve. Moderate densities of such trials produce the strongest backward amplification, which generates the strong negative top-cue effects seen in dense complementary designs; very high densities attenuate to a lower floor, protecting all-complementary blocks from over-reversal. Sparse designs that never contain fully diagnostic near-tied trials keep the trigger at exactly zero, so their behavior stays governed by the pi_4-like baseline mixture, preserving the near-zero active-inactive contrasts in those experiments.

**Parameters:**
- w_forward: [0.22, 0.32]
- w_backward: [0.42, 0.54]
- beta_forward: [0.08, 0.20]
- beta_backward: [1.30, 1.70]
- beta_tally: [1.80, 2.60]
- beta_backward_strong: [2.50, 4.00]
- context_window: {24}
- tie_margin: [2.0, 3.0]
- min_qualifying_count: {2, 3}
- rise_start_frac: [0.03, 0.06]
- peak_start_frac: [0.08, 0.14]
- peak_end_frac: [0.26, 0.34]
- fall_end_frac: [0.40, 0.48]
- s_max: [0.84, 0.92]
- s_floor: [0.40, 0.50]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n = int(stim.shape[1])

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n:
        validities = np.linspace(0.9, 0.5, n)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.full(2, 0.5, dtype=float)
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    tie_margin = float(parameters['tie_margin'])

    def complementarity_signal(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = x - y
        n_discrim = int(np.sum(d != 0.0))
        tally = float(np.sum(d > 0.0) - np.sum(d < 0.0))
        return 1.0 if (n_discrim >= n and abs(tally) <= tie_margin) else 0.0

    signals = []
    past_a = history.get('option_a_ratings', [])
    past_b = history.get('option_b_ratings', [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        signals.append(complementarity_signal(past_a[i], past_b[i]))
    signals.append(complementarity_signal(a, b))

    context_window = int(parameters['context_window'])
    if len(signals) > context_window:
        signals = signals[-context_window:]

    n_recent = max(1, len(signals))
    n_qual = float(np.sum(signals))
    frac = n_qual / n_recent

    min_count = int(parameters['min_qualifying_count'])
    rise_start = float(parameters['rise_start_frac'])
    peak_start = float(parameters['peak_start_frac'])
    peak_end = float(parameters['peak_end_frac'])
    fall_end = float(parameters['fall_end_frac'])
    s_max = float(parameters['s_max'])
    s_floor = min(float(parameters['s_floor']), s_max)

    if n_qual < min_count:
        s = 0.0
    else:
        if frac <= rise_start:
            s = 0.0
        elif frac <= peak_start:
            s = s_max * (frac - rise_start) / (peak_start - rise_start + 1e-9)
        elif frac <= peak_end:
            s = s_max
        elif frac <= fall_end:
            s = s_max - (s_max - s_floor) * (frac - peak_end) / (fall_end - peak_end + 1e-9)
        else:
            s = s_floor
    s = float(np.clip(s, 0.0, 1.0))

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_backward_strong = float(parameters['beta_backward_strong'])

    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)
    p_backward_strong = lex_probabilities(ascending, beta_backward_strong)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / np.sum(weights)

    p_base = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally
    p = (1.0 - s) * p_base + s * p_backward_strong
    p = np.clip(p, 0.0, None)
    return p / np.sum(p)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return 0
    probs = probs / probs.sum()
    return int(np.random.choice(probs.size, p=probs))
```


## COMPETING THEORY
**Description:** People choose between two products through a baseline mixture of a weak forward lexicographic scan, a moderate lowest-validity-first lexicographic scan, and feature-win tallying. Decision makers continuously accumulate local evidence about cue reliability from recent discriminating trials. Evidence is graded by discrimination density: near-tied trials with only a few discriminating features contribute little or nothing, while dense five-of-six or fully discriminating trials drive a leaky accumulator. The accumulator has asymmetric rise and decay, a minimum qualifying count, hysteresis, and smooth activation of a strengthened lowest-validity-first route. When fully discriminating trials dominate the local evidence stream, the reversed route is gently capped below its maximum so strongly complementary designs reverse strongly but not excessively. When dense conflict is sparse, evidence decays back to baseline mixture weights, preserving near-zero active-inactive contrasts in sparse designs.

**Parameters:**
- w_forward: [0.24, 0.32]
- w_backward: [0.40, 0.48]
- beta_forward: [0.12, 0.20]
- beta_backward: [1.30, 1.55]
- beta_tally: [2.00, 2.50]
- beta_backward_strong: [3.00, 3.40]
- tie_margin: {3}
- base_evidence: [0.60, 0.72]
- hi_lo_bonus: [0.20, 0.32]
- near_full_evidence_scale: [0.82, 0.90]
- opposing_evidence_scale: [0.20, 0.35]
- evidence_rise: [0.30, 0.42]
- evidence_decay: [0.06, 0.12]
- zero_decay_boost: [1.40, 2.40]
- conflict_threshold: [0.03, 0.05]
- saturation_evidence: [0.22, 0.32]
- hysteresis: [0.02, 0.05]
- activation_rise: [0.20, 0.30]
- activation_decay: [0.04, 0.08]
- min_qualifying_count: {3, 4}
- qualifying_window: {16}
- full_frac_window: {16}
- s_max: [0.80, 0.88]
- full_penalty_start: [0.28, 0.40]
- full_penalty_depth: [0.38, 0.50]
- full_penalty_width: [0.12, 0.20]
- full_penalty_floor: [0.20, 0.30]
- epsilon: {0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(x, y, order, beta):
        winner = None
        for j in order:
            if x[j] > y[j]:
                winner = 0
                break
            if y[j] > x[j]:
                winner = 1
                break
        if winner is None:
            return np.full(2, 0.5, dtype=float)
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_backward_strong = float(parameters['beta_backward_strong'])
    beta_tally = float(parameters['beta_tally'])

    p_forward = lex_probabilities(a, b, descending, beta_forward)
    p_backward = lex_probabilities(a, b, ascending, beta_backward)
    p_shift = lex_probabilities(a, b, ascending, beta_backward_strong)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / np.sum(weights)
    p_base = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    tie_margin = float(parameters['tie_margin'])
    hi_idx = int(descending[0])
    lo_idx = int(ascending[0])

    if n_features == 6:
        min_discrim_count = 5
    elif n_features == 5:
        min_discrim_count = 4
    else:
        min_discrim_count = int(np.ceil(0.75 * max(1, n_features)))

    base_evidence = float(parameters['base_evidence'])
    hi_lo_bonus = float(parameters['hi_lo_bonus'])
    near_full_evidence_scale = float(parameters['near_full_evidence_scale'])
    opposing_evidence_scale = float(parameters['opposing_evidence_scale'])

    def trial_evidence(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = x - y
        n_discrim = int(np.sum(d != 0.0))
        if n_discrim < min_discrim_count:
            return 0.0
        tally = float(np.sum(d))
        hi_dir = int(np.sign(d[hi_idx]))
        lo_dir = int(np.sign(d[lo_idx]))
        if hi_dir == 0 or lo_dir == 0 or hi_dir == lo_dir:
            return 0.0

        # Narrow special branch for six-feature 5-of-6 trials where the
        # highest and lowest cues disagree but the tally mildly opposes the
        # highest cue (e.g. E17 target patterns).  Exclude strong opposition
        # such as E1 patterns with tally +-3.
        if (n_features == 6 and n_discrim == 5 and abs(tally) <= 2.0
                and tally * hi_dir < 0.0):
            return float(min(1.0, opposing_evidence_scale * (base_evidence + hi_lo_bonus)))

        if abs(tally) > tie_margin:
            return 0.0
        if tally * hi_dir < 0.0:
            return 0.0

        if n_discrim == n_features:
            density_gain = 1.0
        else:
            density_gain = near_full_evidence_scale
        return float(min(1.0, (base_evidence + hi_lo_bonus) * density_gain))

    evidence_seq = []
    full_flags = []
    past_a = history.get('option_a_ratings', [])
    past_b = history.get('option_b_ratings', [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        x = np.asarray(past_a[i], dtype=float)
        y = np.asarray(past_b[i], dtype=float)
        evidence_seq.append(trial_evidence(x, y))
        full_flags.append(1.0 if np.all((x - y) != 0.0) else 0.0)
    evidence_seq.append(trial_evidence(a, b))
    full_flags.append(1.0 if np.all((a - b) != 0.0) else 0.0)

    evidence_rise = float(parameters['evidence_rise'])
    evidence_decay = float(parameters['evidence_decay'])
    zero_decay_boost = float(parameters['zero_decay_boost'])
    threshold = float(parameters['conflict_threshold'])
    saturation = float(parameters['saturation_evidence'])
    hysteresis = float(parameters['hysteresis'])
    activation_rise = float(parameters['activation_rise'])
    activation_decay = float(parameters['activation_decay'])
    min_qual = int(parameters['min_qualifying_count'])
    qualifying_window = int(parameters['qualifying_window'])
    full_frac_window = int(parameters['full_frac_window'])
    s_max = float(parameters['s_max'])
    full_penalty_start = float(parameters['full_penalty_start'])
    full_penalty_depth = float(parameters['full_penalty_depth'])
    full_penalty_width = float(parameters['full_penalty_width'])
    full_penalty_floor = float(parameters['full_penalty_floor'])

    X = 0.0
    S = 0.0

    for idx, u in enumerate(evidence_seq):
        if u >= X:
            X += evidence_rise * (u - X)
        else:
            decay_mult = zero_decay_boost if u <= 0.0 else 1.0
            X -= evidence_decay * decay_mult * (X - u)
        X = float(np.clip(X, 0.0, 1.0))

        q_start = max(0, idx - qualifying_window + 1)
        recent_q = np.asarray(evidence_seq[q_start:idx + 1], dtype=float)
        n_recent_qual = int(np.sum(recent_q > 0.0))

        f_start = max(0, idx - full_frac_window + 1)
        recent_flags = np.asarray(full_flags[f_start:idx + 1], dtype=float)
        window_len = idx - f_start + 1
        full_frac = float(np.sum(recent_flags > 0.5)) / max(1, window_len)

        center = full_penalty_start + full_penalty_width
        if full_frac <= full_penalty_start:
            full_penalty = 0.0
        elif full_frac <= center:
            full_penalty = full_penalty_depth * (full_frac - full_penalty_start) / full_penalty_width
        else:
            fall_end = center + full_penalty_width
            if full_frac >= fall_end:
                full_penalty = full_penalty_floor
            else:
                full_penalty = full_penalty_depth - (full_penalty_depth - full_penalty_floor) * (full_frac - center) / full_penalty_width
        full_penalty = float(np.clip(full_penalty, 0.0, 1.0))
        s_max_eff = s_max * (1.0 - full_penalty)

        if n_recent_qual < min_qual:
            target = 0.0
        else:
            threshold_eff = threshold - hysteresis if S > 0.5 else threshold
            if X <= threshold_eff:
                target = 0.0
            elif X >= saturation:
                target = s_max_eff
            else:
                target = s_max_eff * ((X - threshold_eff) / (saturation - threshold_eff + 1e-9))

        target = float(np.clip(target, 0.0, max(s_max_eff, 1e-9)))
        if target > S:
            S += activation_rise * (target - S)
        else:
            S -= activation_decay * (S - target)
        S = float(np.clip(S, 0.0, s_max_eff))

    s = float(np.clip(S, 0.0, s_max))
    p = (1.0 - s) * p_base + s * p_shift

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.full(2, 0.5, dtype=float)

    p = np.clip(p, 0.0, None)
    return p / np.sum(p)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return 0
    probs = probs / probs.sum()
    return int(np.random.choice(probs.size, p=probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    matches = []\n    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):\n        a = np.asarray(a, dtype=int)\n        b = np.asarray(b, dtype=int)\n        n = len(a)\n        predicted = None\n        for j in range(n - 1, -1, -1):\n            if a[j] > b[j]:\n                predicted = 0\n                break\n            if b[j] > a[j]:\n                predicted = 1\n                break\n        if predicted is None:\n            continue\n        matches.append(1.0 if int(r) == predicted else 0.0)\n    if not matches:\n        return 0.5\n    return float(np.mean(matches))",
  "rationale": "This metric measures the fraction of trials on which the chosen option is the one that wins the first discriminating feature when scanning from lowest validity to highest validity, i.e. feature indices 5, 4, 3, 2, 1, 0. For each trial it finds the first j in that reverse order with A[j] != B[j], predicts A if A[j] > B[j], and B otherwise, then scores whether the observed response matches that prediction. In the advocated theory the context trigger becomes active because the critical fully diagnostic near-tied trials qualify after a short transient, so the strong backward route with beta 2.50-4.00 pulls this low-validity agreement rate close to 0.93-0.95 across most of the 96 trials. In the competing theory the evidence accumulator receives zero evidence on all trial types in this design, so only the moderate baseline low-validity route is active and the same agreement rate stays near 0.72-0.80. Using all 96 trials per subject rather than only critical probes keeps the per-subject variance small, producing a large and stable between-theory gap at N=25."
}
```

## Usage

```json
{
  "input_tokens": 8617,
  "output_tokens": 29593,
  "reasoning_tokens": 29134,
  "provider": "Baidu"
}
```
