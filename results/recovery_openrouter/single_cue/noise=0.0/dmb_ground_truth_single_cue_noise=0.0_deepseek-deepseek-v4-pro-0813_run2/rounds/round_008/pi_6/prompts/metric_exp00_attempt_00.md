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
**Validities (n_features=6):** [0.9, 0.8, 0.72, 0.61, 0.6, 0.6]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 9: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 10: A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 11: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 12: A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 13: A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 14: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 15: A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 17: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 18: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 19: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 20: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 21: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 22: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 23: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 24: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 25: A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 26: A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 27: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 28: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 29: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 30: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 31: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 32: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]

**Rationale:** This design dissociates the advocated full-diagnostic frequency trigger from the competing leaky-accumulator trigger by combining three trial classes. Five unique fully discriminating near-tied trigger pairs (Q pairs) have all six experts discriminate with tally A_wins - B_wins in -2, 0, +2. Their top and lowest validities are aligned in the same direction, so they are definitely qualifying events for the advocated model but only weak full-evidence events for the competing model, which receives its hi-lo disagreement bonus only on conflict. These Q pairs are 5/32 = 15.625 percent of unique trials, so the advocated trailing 24-trial qualifying fraction sits in its peak/rise band and its backward amplification s is high in most windows after the minimum count is met. Eleven further fully discriminating pairs have tally +-4, which is outside both models' near-tie margins, so they add zero qualifying events to the advocated count and zero local evidence to the competing accumulator, but they do enter the competing full_frac denominator with full_frac about 16/32 = 0.50, engaging the competing full-evidence penalty if its accumulator ever fires. Sixteen probe pairs are dense five-of-six comparisons with one middle tie, feature 0 and feature 5 always in conflict, and tally +1 or -1. For top-A probes d = -1 and for top-B probes d = +1, so the tally opposes the top cue; the competing model therefore assigns these dense probes zero local evidence because its tally_align gate fails, while the advocated model also treats them as non-qualifying because they are not fully diagnostic. The critical quantitative dissociation is on the probes. Using representative baseline weights w_forward = 0.25, w_backward = 0.48, w_tally = 0.27, the baseline mixture gives P(A) near 0.25 for top-A d = -1 and near 0.75 for top-B d = +1. With advocated s near 0.80 to 0.92 and a strong backward route, the top-A probe falls to about 0.06 to 0.10 and the top-B probe rises to about 0.90 to 0.94. The competing model instead stays near its baseline because the five Q events are expected only about 2.5 per 16-trial qualifying window, below its fixed min_qualifying_count of four, and all fillers/probes contribute zero evidence; predicted values remain near 0.24 to 0.27 for top-A and near 0.73 to 0.76 for top-B. Thus the probe by top-direction interaction of roughly 0.15 to 0.20 is predicted by the advocated frequency-trigger theory but not by the competing accumulator theory.

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
- w_forward: [0.25, 0.28]
- w_backward: [0.44, 0.52]
- beta_forward: [0.12, 0.20]
- beta_backward: [1.40, 1.60]
- beta_tally: [1.90, 2.40]
- beta_backward_strong: [2.60, 2.90]
- tie_margin: [2.5, 3.0]
- hi_lo_bonus: [0.45, 0.55]
- base_evidence: [0.38, 0.50]
- tied_top_evidence: [0.34, 0.44]
- density_floor: [0.56, 0.62]
- near_full_boost: [1.05, 1.12]
- evidence_rise: [0.35, 0.45]
- evidence_decay: [0.05, 0.08]
- zero_decay_boost: [1.20, 2.20]
- conflict_threshold: [0.03, 0.05]
- saturation_evidence: [0.25, 0.35]
- hysteresis: [0.02, 0.05]
- activation_rise: [0.22, 0.30]
- activation_decay: [0.04, 0.07]
- min_qualifying_count: {4}
- qualifying_window: {16}
- full_frac_window: {16}
- s_max: [0.72, 0.78]
- full_penalty_threshold: [0.50, 0.60]
- full_penalty_depth: [0.15, 0.25]
- full_penalty_width: [0.15, 0.22]
- full_penalty_floor: [0.04, 0.10]
- top_tied_tally_min: {2, 3}
- top_tied_bonus: [0.10, 0.20]
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
    hi_lo_bonus = float(parameters['hi_lo_bonus'])
    base_evidence = float(parameters['base_evidence'])
    tied_top_evidence = float(parameters['tied_top_evidence'])
    density_floor = float(parameters['density_floor'])
    near_full_boost = float(parameters['near_full_boost'])
    top_tied_tally_min = float(parameters['top_tied_tally_min'])
    top_tied_bonus = float(parameters['top_tied_bonus'])
    hi_idx = int(descending[0])
    lo_idx = int(ascending[0])

    def trial_evidence(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = x - y
        n_discrim = int(np.sum(d != 0.0))
        density = n_discrim / max(1, n_features)
        tally = float(np.sum(d))
        near = 1.0 if abs(tally) <= tie_margin else 0.0
        full = 1.0 if n_discrim >= n_features else 0.0
        if near <= 0.0:
            return 0.0

        hi_dir = np.sign(d[hi_idx])
        lo_dir = np.sign(d[lo_idx])
        disagree = 1.0 if (hi_dir != 0.0 and lo_dir != 0.0 and hi_dir == -lo_dir) else 0.0
        tally_align = 1.0 if (hi_dir == 0.0 or float(tally) * hi_dir >= 0.0) else 0.0

        if full > 0.5:
            density_gain = 1.0
            if hi_dir == 0.0:
                base = tied_top_evidence
            else:
                base = base_evidence + hi_lo_bonus * disagree
            return float(min(1.0, base * density_gain))

        # Non-full branch: top-tied fallback is narrowly gated by tally
        # magnitude so marginal top-tied sparse conflicts (e.g. E11) stay blocked,
        # while E13-style dense top-tied 5/6 conflicts can still emit evidence.
        if hi_dir == 0.0:
            fallback_hi_dir = 0.0
            for j in descending:
                if d[j] != 0.0:
                    fallback_hi_dir = float(np.sign(d[j]))
                    break
            if fallback_hi_dir == 0.0:
                return 0.0
            if abs(tally) < top_tied_tally_min:
                return 0.0
            disagree_t = 1.0 if (lo_dir != 0.0 and fallback_hi_dir != 0.0 and fallback_hi_dir == -lo_dir) else 0.0
            tally_align_t = 1.0 if (fallback_hi_dir != 0.0 and float(tally) * fallback_hi_dir >= 0.0) else 0.0
            if density <= density_floor or disagree_t <= 0.0 or tally_align_t <= 0.0:
                return 0.0
            density_gain = float(np.sqrt(max(0.0, (density - density_floor) / (1.0 - density_floor + 1e-9))))
            density_gain *= near_full_boost
            base = base_evidence + hi_lo_bonus + top_tied_bonus
            return float(min(1.0, base * density_gain))

        if density <= density_floor or disagree <= 0.0 or tally_align <= 0.0:
            return 0.0
        density_gain = float(np.sqrt(max(0.0, (density - density_floor) / (1.0 - density_floor + 1e-9))))
        density_gain *= near_full_boost
        base = base_evidence + hi_lo_bonus * disagree
        return float(min(1.0, base * density_gain))

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
    full_penalty_center = float(parameters['full_penalty_threshold'])
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
        n_full = int(np.sum(recent_flags > 0.5))
        full_frac = n_full / max(1, window_len)

        width = max(full_penalty_width, 1e-9)
        rise_start = full_penalty_center - width
        fall_end = full_penalty_center + width
        if full_frac <= rise_start:
            full_penalty = 0.0
        elif full_frac < full_penalty_center:
            full_penalty = full_penalty_depth * max(0.0, (full_frac - rise_start)) / width
        elif full_frac < fall_end:
            full_penalty = full_penalty_floor + max(0.0, (full_penalty_depth - full_penalty_floor)) * max(0.0, (fall_end - full_frac)) / width
        else:
            full_penalty = full_penalty_floor
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

        target = float(np.clip(target, 0.0, s_max_eff))
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
  "metric_source": "def metric(data):\n    import numpy as np\n    if data.empty:\n        return 0.0\n    A = np.asarray(data['option_a_ratings'].tolist(), dtype=float)\n    B = np.asarray(data['option_b_ratings'].tolist(), dtype=float)\n    y = data['response'].to_numpy(dtype=float)\n    if A.ndim != 2 or A.shape[1] < 6:\n        return 0.0\n    d = A - B\n    n_disc = np.sum(d != 0.0, axis=1)\n    tally = np.sum(d, axis=1)\n    d0 = d[:, 0]\n    d4 = d[:, 4]\n    d5 = d[:, 5]\n    topA = (d0 > 0.0) & (d5 < 0.0) & (d4 <= 0.0) & (n_disc == 5) & (tally < 0.0)\n    topB = (d0 < 0.0) & (d5 > 0.0) & (d4 >= 0.0) & (n_disc == 5) & (tally > 0.0)\n    if topA.sum() == 0 or topB.sum() == 0:\n        return 0.0\n    pB_topA = float(np.mean(y[topA]))\n    pB_topB = float(np.mean(y[topB]))\n    return float(pB_topA - pB_topB)",
  "rationale": "This metric isolates the clean dense five-of-six probe contrast: top validity feature 0 favors one option while bottom feature 5 favors the other, the tied bottom feature 4 is aligned with or tied with feature 5, and the overall tally opposes the top cue. It returns P(choose B | top cue favors A) - P(choose B | top cue favors B). Under the advocated theory the Q trials keep backward amplification high, so these probes are dominated by the strengthened backward route and the contrast is very large, near 0.85. Under the competing theory these probes contribute no local evidence because tally_align fails, so choices stay near the moderate baseline mixture and the contrast is much smaller, near 0.50. Each subject provides 24 clean top-A and 21 clean top-B probe trials, so the metric has low per-subject binomial noise and separates the two theories reliably at N=25."
}
```

## Usage

```json
{
  "input_tokens": 9287,
  "output_tokens": 30094,
  "reasoning_tokens": 29535,
  "provider": "Baidu"
}
```
