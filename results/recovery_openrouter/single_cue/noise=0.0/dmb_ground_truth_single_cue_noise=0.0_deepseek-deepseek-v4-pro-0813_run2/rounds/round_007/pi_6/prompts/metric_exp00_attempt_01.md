# metric_exp00_attempt_01

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
  trial 1: A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 2: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 6: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 9: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 10: A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 12: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 13: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 14: A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 15: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 17: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 18: A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 19: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 20: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 21: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 22: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 23: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 24: A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 25: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 26: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 27: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 28: A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  trial 29: A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 30: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 31: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 32: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]

**Rationale:** This design isolates the density boundary between the two context-trigger definitions. Every unique pair is a dense five-of-six discriminating, near-tied comparison with the highest-validity cue (feature 0, validity 0.90) conflicting with the lowest-validity cue (feature 5, validity 0.50), exactly one of the four middle cues tied, and a feature-win tally difference of exactly +1 or -1. Under the competing accumulator model these trials are strong evidence-positive events: density 5/6 exceeds the density floor 0.80, abs(tally) = 1 is within tie_margin, and the top-low disagreement earns the hi_lo bonus. The accumulator therefore crosses threshold after the minimum qualifying count and S stays near s_max, with no full-evidence penalty because no fully discriminating trial ever appears. Under the advocated model the trigger requires a fully diagnostic comparison (all six experts disagree), so every trial is non-qualifying; the qualifying count stays zero and the backward-amplification term s stays exactly 0, leaving only the baseline forward/backward/tally mixture. Thus the probe contrast is bottom-cue direction by tally sign. For top-A/low-B pairs the backward scanner stops at feature 5 favoring B; for top-B/low-A pairs it stops at feature 5 favoring A. With representative mid-range parameters (w_forward = 0.25, w_backward = 0.48, w_tally = 0.27, beta_forward = 0.14, beta_backward = 1.5, beta_tally = 2.2, beta_backward_strong = 3.25, advocated s = 0, competing S about 0.80), top-A d = -1 gives advocated P(A) about 0.25 versus competing about 0.08; top-A d = +1 gives advocated about 0.46 versus competing about 0.12; top-B d = -1 gives advocated about 0.54 versus competing about 0.88; and top-B d = +1 gives advocated about 0.75 versus competing about 0.92. The advocated model therefore retains a d-sensitive response, whereas the competing model flattens the response near the bottom-cue winner. This isolates the trigger-density criterion: five-of-six near-ties should activate the competing reversed route but leave the advocated route completely unamplified.

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
- w_forward: [0.22, 0.32]
- w_backward: [0.42, 0.54]
- beta_forward: [0.08, 0.20]
- beta_backward: [1.30, 1.70]
- beta_tally: [1.80, 2.60]
- beta_backward_strong: [2.40, 4.00]
- tie_margin: [2.0, 3.0]
- hi_lo_bonus: [0.30, 0.55]
- base_evidence: [0.62, 0.78]
- tied_top_evidence: [0.50, 0.72]
- density_floor: {0.80}
- evidence_rise: [0.22, 0.42]
- evidence_decay: [0.02, 0.12]
- zero_decay_boost: [1.20, 2.50]
- conflict_threshold: [0.03, 0.05]
- hysteresis: [0.02, 0.05]
- activation_rise: [0.08, 0.30]
- activation_decay: [0.01, 0.08]
- min_qualifying_count: {2, 3}
- s_max: [0.75, 0.85]
- full_penalty_threshold: [0.85, 0.95]
- full_penalty_depth: [0.10, 0.18]
- epsilon: [0.0, 0.02]
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
        if near <= 0.0 or density <= density_floor:
            return 0.0

        density_gain = (density - density_floor) / (1.0 - density_floor)
        hi_dir = np.sign(d[hi_idx])
        lo_dir = np.sign(d[lo_idx])
        disagree = 1.0 if (hi_dir != 0.0 and lo_dir != 0.0 and hi_dir == -lo_dir) else 0.0

        if hi_dir == 0.0:
            base = tied_top_evidence
        else:
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
    hysteresis = float(parameters['hysteresis'])
    activation_rise = float(parameters['activation_rise'])
    activation_decay = float(parameters['activation_decay'])
    min_qual = int(parameters['min_qualifying_count'])
    s_max = float(parameters['s_max'])
    full_penalty_threshold = float(parameters['full_penalty_threshold'])
    full_penalty_depth = float(parameters['full_penalty_depth'])

    X = 0.0
    S = 0.0
    n_seen = 0
    for idx, u in enumerate(evidence_seq):
        if u >= X:
            X += evidence_rise * (u - X)
        else:
            decay_mult = zero_decay_boost if u <= 0.0 else 1.0
            X -= evidence_decay * decay_mult * (X - u)

        if u > 0.0:
            n_seen += 1

        n_evidence = 0
        n_full = 0
        for k in range(idx + 1):
            if evidence_seq[k] > 0.0:
                n_evidence += 1
                if full_flags[k] > 0.5:
                    n_full += 1
        full_frac = (n_full / max(1, n_evidence)) if n_evidence > 0 else 0.0

        if full_frac > full_penalty_threshold:
            full_penalty = full_penalty_depth * ((full_frac - full_penalty_threshold) / (1.0 - full_penalty_threshold + 1e-9))
        else:
            full_penalty = 0.0
        s_max_eff = s_max * (1.0 - full_penalty)

        if n_seen < min_qual:
            target = 0.0
        else:
            threshold_eff = threshold - hysteresis if S > 0.5 else threshold
            target = s_max_eff if X >= threshold_eff else 0.0

        if target > S:
            S += activation_rise * (target - S)
        else:
            S -= activation_decay * (S - target)

    s = float(np.clip(S, 0.0, s_max))
    p = (1.0 - s) * p_base + s * p_shift

    epsilon = float(parameters['epsilon'])
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
[0] rationale: This metric uses a fixed discriminant score equal to the difference between the strong backward-lexicographic choice probability and the baseline mixture probability for each trial. It then tracks the advocated theory's trigger: a rolling 24-trial count of fully diagnostic, near-tied comparisons. In trigger-active windows the advocated model amplifies the strong backward route, so responses should covary with this score; outside active windows the score should be near zero. The competing model amplifies the same route from dense five-of-six trials regardless of full diagnostics, so the active-inactive contrast should be much smaller. Comparing within-subject condition-specific covariances isolates the trigger-density criterion while controlling overall response bias.
metric_source:
import numpy as np

def metric(data):
    validities = np.array([0.9, 0.8, 0.72, 0.61, 0.6, 0.6])
    descending = np.array([0, 1, 2, 3, 4, 5])
    ascending = np.array([3, 4, 5, 2, 1, 0])
    bf = 0.14
    bb = 1.5
    bt = 2.2
    bs = 3.25
    wf = 0.25
    wb = 0.48
    wt = 0.27

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def trial_d(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        winner = None
        for j in descending:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner == 0:
            pbf = sigmoid(-bf)
        elif winner == 1:
            pbf = sigmoid(bf)
        else:
            pbf = 0.5

        winner = None
        for j in ascending:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner == 0:
            pbb = sigmoid(-bb)
            pbs = sigmoid(-bs)
        elif winner == 1:
            pbb = sigmoid(bb)
            pbs = sigmoid(bs)
        else:
            pbb = 0.5
            pbs = 0.5

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        pbt = sigmoid(bt * (b_wins - a_wins))
        pbase = wf * pbf + wb * pbb + wt * pbt
        return pbs - pbase

    def full_flag(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b
        n_disc = int(np.sum(diff != 0.0))
        tally = float(np.sum(diff))
        return 1.0 if (n_disc >= len(a) and abs(tally) <= 3.0) else 0.0

    dvals = data.apply(trial_d, axis=1).to_numpy(dtype=float)
    y = data['response'].to_numpy(dtype=float)
    full = data.apply(full_flag, axis=1).to_numpy(dtype=float)
    subj = data['subject_id'].to_numpy()
    n = data.shape[0]

    active = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        j = i + 1
        while j < n and subj[j] == subj[i]:
            j += 1
        m = j - i
        prefix = np.zeros(m + 1)
        for t in range(m):
            prefix[t + 1] = prefix[t] + full[i + t]
        for t in range(1, m + 1):
            lo = t - 24 if t > 24 else 0
            cnt = prefix[t] - prefix[lo]
            active[i + t - 1] = cnt >= 2.0
        i = j

    if np.sum(active) == 0 or np.sum(~active) == 0:
        return 0.0

    d_active_mean = float(np.mean(dvals[active]))
    d_inactive_mean = float(np.mean(dvals[~active]))
    c = np.where(active, dvals - d_active_mean, dvals - d_inactive_mean)
    score = c * y
    return float(np.mean(score[active]) - np.mean(score[~active]))
outcome: self_sim=-0.0162 (var=0.0010) adversary_sim=-0.0185 (var=0.0005) welch_t=+0.296 p=0.769 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "import numpy as np\n\ndef metric(data):\n    validities = np.array([0.9, 0.8, 0.72, 0.61, 0.6, 0.6])\n    ascending = np.argsort(validities, kind='stable')\n    n_features = len(validities)\n    window = 24\n    low_cut = 2\n    high_cut = 3\n    min_trial = 30\n\n    subject_scores = []\n    for subject_id, subj in data.groupby('subject_id', sort=False):\n        subj = subj.reset_index(drop=True)\n        n = len(subj)\n        if n <= min_trial:\n            continue\n\n        is_full = np.zeros(n, dtype=bool)\n        back_cons = np.zeros(n, dtype=float)\n        has_winner = np.zeros(n, dtype=bool)\n\n        for i in range(n):\n            a = np.asarray(subj.loc[i, 'option_a_ratings'], dtype=float)\n            b = np.asarray(subj.loc[i, 'option_b_ratings'], dtype=float)\n            diff = a - b\n            n_disc = int(np.sum(diff != 0.0))\n            tally = float(np.sum(diff))\n            if n_disc >= n_features and abs(tally) <= 3.0:\n                is_full[i] = True\n\n            winner = -1\n            for j in ascending:\n                if a[j] > b[j]:\n                    winner = 0\n                    break\n                if b[j] > a[j]:\n                    winner = 1\n                    break\n            if winner < 0:\n                continue\n            has_winner[i] = True\n            back_cons[i] = 1.0 if int(subj.loc[i, 'response']) == winner else 0.0\n\n        prefix = np.concatenate((np.zeros(1), np.cumsum(is_full.astype(float))))\n        low_mask = np.zeros(n, dtype=bool)\n        high_mask = np.zeros(n, dtype=bool)\n        eligible = np.zeros(n, dtype=bool)\n\n        for i in range(n):\n            if i < min_trial or i < window - 1:\n                continue\n            start = i - window + 1\n            full_count = prefix[i + 1] - prefix[start]\n            eligible[i] = True\n            if full_count < low_cut:\n                low_mask[i] = True\n            elif full_count >= high_cut:\n                high_mask[i] = True\n\n        non_full = ~is_full\n        low_sel = eligible & low_mask & non_full & has_winner\n        high_sel = eligible & high_mask & non_full & has_winner\n\n        if not low_sel.any() or not high_sel.any():\n            continue\n\n        low_mean = float(np.mean(back_cons[low_sel]))\n        high_mean = float(np.mean(back_cons[high_sel]))\n        subject_scores.append(high_mean - low_mean)\n\n    if not subject_scores:\n        return 0.0\n    return float(np.mean(subject_scores))",
  "rationale": "This metric targets the proposed trigger-density difference. The advocated model switches on a strengthened lowest-validity-first route only after at least 2 or 3 fully discriminating near-tied trials appear in the trailing 24-trial window; with fewer than 2 such trials its amplification s remains zero and choices stay at the baseline mixture. The competing model accumulates evidence from every dense five-of-six trial, so its strengthened reversed route should be active during both the low and high full-trial-count states. I classify each trial by its backward-lexicographic winner and score a response as backward-consistent if the subject chose that winner. After the first 30 trials, I count fully discriminating near-tied trials in a rolling 24-trial window and define low state as fewer than 2 and high state as at least 3. Fully discriminating trials themselves are excluded from the outcome means so composition is not dominated by the trigger events. The returned value is mean backward-consistency in the high state minus mean backward-consistency in the low state, averaged over subjects. Under the advocated model this contrast is positive because low-state trials show baseline responding while high-state trials show strong backward responding. Under the competing model both states have strong backward responding, so the same contrast should be near zero."
}
```

## Usage

```json
{
  "input_tokens": 9323,
  "output_tokens": 44460,
  "reasoning_tokens": 43454,
  "provider": "Baidu"
}
```
