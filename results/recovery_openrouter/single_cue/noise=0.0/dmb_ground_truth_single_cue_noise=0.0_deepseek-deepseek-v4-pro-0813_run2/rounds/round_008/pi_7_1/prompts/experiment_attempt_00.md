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


## COMPETING THEORY
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design makes the highest-validity cue disagree with the majority of the remaining cues on every trial. All six validities are ordered so feature 1 is the top TTB cue; every critical pair has that top cue discriminate in one direction while at least three of the five lower-validity cues discriminate in the opposite direction. Take The Best stops at the top cue and predicts the top-cue winner; Tallying counts all feature-wise wins and predicts the opposite majority. The 32 unique pairs contain 16 top-cue-favors-A cases and 16 top-cue-favors-B cases to balance any position bias. For a top-A pair where m lower cues favor B (here m = 3, 4, or 5), TTB predicts choice A with probability about 1/(1+exp(-beta)) before lapse, while Tallying predicts choice B with probability about 1/(1+exp(-beta*(m-1))), so the theories are separated in direction at moderate and high beta. The 32 unique pairs are repeated K = 3 times for 96 total trials.

[1] This design isolates a second-cue stopping signature. The highest-validity cue (feature 0, validity 0.90) is tied on every trial, so Take The Best skips it and stops at the second cue (feature 1, validity 0.78). Feature 1 always discriminates, so TTB never consults the three lower cues; it predicts the same choice probability for all feature-1-favors-A trials and the opposite for all feature-1-favors-B trials. Tallying, by contrast, treats the top tie as uninformative and counts wins on all remaining cues. The three lower cues are varied so that Tallying's tally difference d = A_wins - B_wins takes values -2, -1, and 0 for feature-1-A trials, and 0, +1, and +2 for feature-1-B trials. Thus TTB predicts a flat choice curve across lower-cue patterns within each direction, whereas Tallying predicts a logistic curve in d, producing opposite choices on the d=-1/-2 and d=+1/+2 patterns and guessing on d=0. The 32 unique pairs are balanced over top-tie value (0/0 vs 1/1) and second-cue direction, and are repeated K=3 for 96 total trials.

[2] Six features with a spread of validities. Feature 0 (validity 0.90) is tied on every trial, so Take The Best skips it and stops at feature 1 (validity 0.80), which always discriminates. The four lower features are arranged to set the tally difference d = A_wins - B_wins to -2, -1, 0, +1, or +2. Critically, for each d there are trials where feature 1 favors A and trials where feature 1 favors B. Pure Tallying depends only on d and therefore must predict identical choice probabilities for the two feature-1 directions within each d. The TTB/Tallying mixture instead predicts a within-d split of about q≈0.25: e.g. at d=0, pure Tallying predicts 0.5 for all trials, while the mixture predicts ≈0.62 when feature 1 favors A and ≈0.38 when it favors B; at d=+1, pure Tallying predicts near 1.0, while the mixture predicts ≈1.0 vs ≈0.75. This within-d split cannot be produced by any beta or epsilon in pure Tallying. The 32 unique pairs are repeated K=3 for 96 total trials.

[3] This design isolates a signature of pure Tallying that the TTB/Tallying mixture cannot reproduce. Pure Tallying depends only on the tally difference d = A_wins - B_wins. When d = 0, its softmax over equal tallies is exactly uniform and the lapse term also preserves uniformity, so it must predict P(A) = 0.5 for every such pair, regardless of which individual cue discriminates. The mixture model instead adds a TTB component (q ≈ 0.25) that stops at the first discriminating cue in descending-validity order. By tying the top two or top three cues on every trial, the first discriminator is pushed to feature 2 or feature 3. All 32 unique pairs are constructed to have d = 0, so the Tallying component of the mixture also gives 0.5, but the TTB component shifts choice probabilities to about 0.62 when the first discriminator favors A and about 0.38 when it favors B (using q ≈ 0.25 and beta_ttb ≈ 5.25). The critical contrast is therefore first-discriminator direction: pure Tallying predicts 0.5 vs 0.5 (difference 0), while the mixture predicts approximately 0.62 vs 0.38 (difference about 0.25). The design balances first-discriminator direction, top-tie value (0/0 vs 1/1), and discriminator position (feature 2 vs feature 3) across the 32 unique pairs, which are repeated K = 3 times for 96 trials.

[4] This design isolates a direction-of-lexicographic-scanning dissociation. All 32 unique pairs hold the feature-win tally difference at d = 0, so both theories' tallying routes are neutral and cannot drive the choice. On every trial the highest-validity cue (feature 0, validity 0.90) discriminates, and the lowest-validity cue (feature 5, validity 0.50) discriminates in the opposite direction. The advocated TTB/Tally mixture has only a descending lexicographic route, so its TTB component stops at feature 0; with q near 0.25 it predicts P(A) ≈ 0.62 when feature 0 favors A and ≈ 0.38 when feature 0 favors B. The competing model instead has a strong backward (ascending) lexicographic route (weight ≈ 0.42-0.54, beta ≈ 1.3-1.7) plus a weak forward route and neutral tallying, so on the same conflict trials it predicts the mirror pattern: P(A) ≈ 0.32-0.40 when feature 0 favors A/feature 5 favors B, and ≈ 0.60-0.68 when feature 0 favors B/feature 5 favors A. The middle four features are varied with equal A/B wins or ties to keep d = 0 while generating 32 unique pairs, 16 in each top-cue direction, repeated K = 3 for 96 trials.

[5] All 32 unique pairs are top-vs-bottom conflict trials: feature 0 (validity 0.90) and feature 5 (validity 0.50) always discriminate in opposite directions. Therefore the descending lexicographic route (TTB) stops immediately at feature 0, while the ascending lexicographic route (backward) stops immediately at feature 5. The four middle cues are varied only to set the feature-win tally difference d = A_wins - B_wins. In the 16 top-A/low-B trials d ranges 0..+4; in the 16 top-B/low-A trials d ranges -3..0. The competing model (q*TTB + tally, no backward route) predicts a step-like curve: for top-A/low-B it gives P(A) near 0.62 at d=0 and near 1.0 for d>=1; for top-B/low-A it gives P(A) near 0.38 at d=0 and near 0.0 for d<=-1. The advocated model has a strong backward route (w_backward about 0.48, beta about 1.5) that cancels the top-cue signal: for top-A/low-B P(A) stays in a compressed band near 0.36-0.48 across d, and for top-B/low-A P(A) stays near 0.52-0.64. The key quantitative dissociations are therefore top-A/low-B d=+1 (competing about 1.0 vs advocated about 0.46) and top-B/low-A d=-1 (competing about 0.005 vs advocated about 0.54). This d-response pattern cannot be produced by any q or beta in the competing model because the competing model lacks the backward lexicographic route.

[6] This design isolates the conflict-gate discontinuity that separates the advocated model from the stable-mixture model. Six experts have validities [0.90, 0.80, 0.70, 0.60, 0.55, 0.52]; feature 0 is the highest-validity cue and feature 5 is the lowest. On every trial these two cues discriminate in opposite directions (top-A/low-B or top-B/low-A). The four middle cues set the feature-win tally difference d = A_wins - B_wins to -2, -1, 0, +1, +2, with both signs present in both top-cue directions. For top-A/low-B trials the advocated model enters its conflict regime whenever d >= 0: tallying is suspended and the backward scan is boosted, so P(A) drops to about 0.15 and stays flat across d = 0, +1, +2. The competing stable mixture instead predicts a smooth monotonic increase from about 0.24 at d = -2 to about 0.48 at d = +2. For top-B/low-A trials the advocated conflict regime is active for d <= 0, giving P(A) near 0.85 for d = -2, -1, 0, whereas the competing model predicts only about 0.52-0.64. The non-conflict cells (top-A d < 0 and top-B d > 0) serve as controls where the two models nearly coincide. This drop/inversion at d = 0 for top-A trials and high plateau for top-B trials cannot be produced by any stable mixture of forward, backward, and tally routes, because that model is smooth and monotonic in d. The 32 unique pairs are balanced over top direction and d and repeated K = 3 for 96 total trials.

[7] This design uses a tied highest validity (features 0 and 1 both 0.90) to decouple the competing model's conflict-gate trigger from the first discriminating cue. On every trial the lowest-validity cue (feature 7, 0.50) discriminates opposite the top-direction cue. In gate-active trials, feature 0 itself discriminates in the top direction, so the competing model's conflict gate is live: highest and lowest cues conflict and the tally does not oppose the highest cue. In matched gate-inactive trials, feature 0 is tied and feature 1—the other tied-highest cue—discriminates in the same top direction. For the advocated stable mixture, each matched pair is identical: forward lexicographic stops at the same top-direction winner (feature 0 or feature 1), backward lexicographic stops at feature 7, and the tally difference d is held constant by adjusting feature 2. The advocated model therefore predicts exactly equal P(A) for matched gate-active and gate-inactive trials. The competing model instead predicts a large gate-condition gap: on gate-active trials tallying is suspended and a strong backward scan dominates, whereas on gate-inactive trials the baseline tally route remains active. With mid-range parameters, top-A d=+1/+2 matched pairs give advocated P(A)≈0.46/0.48 in both gate conditions, whereas competing gives ≈0.46/0.48 gate-inactive but ≈0.16/0.17 gate-active. The top-B d=-1/-2 mirror gives advocated ≈0.54/0.52 in both, while competing gives ≈0.54/0.52 gate-inactive versus ≈0.84/0.84 gate-active. The 32 unique pairs are balanced over top direction, d, and gate condition, and repeated K=3 for 96 trials.

[8] Six experts with validities [0.90, 0.78, 0.66, 0.58, 0.52, 0.52]. Features 4 and 5 are tied for lowest validity; because both theories use stable argsort, feature 4 is the first cue in ascending/backward scans. Every trial sets feature 0 (highest) and feature 4 (first-lowest) to discriminate in opposite directions. The middle features fix the tally difference at +1 for top-A trials and -1 for top-B trials. The critical manipulation is feature 5, the second tied-lowest cue: in gate-active trials it is tied, so the advocated model's low_dir is nonzero and its conflict gate is live (tally does not oppose the top cue); in matched gate-inactive trials feature 5 discriminates in the same direction as the top cue, making low_dir mixed and switching the gate off. The competing stable-mixture model is blind to this manipulation: forward lex stops at feature 0, backward lex stops at feature 4 before feature 5, and tally sees the same d, so it predicts identical P(A) across gate-active and gate-inactive pairs within each top direction. With representative mid-range parameters (advocated: w_forward=.25, w_backward=.48, beta_forward=.14, beta_backward=1.5, beta_tally=2.2, w_conflict_back=.80, beta_conflict_back=2.5; competing: w_forward=.27, w_backward=.48, beta_forward=.14, beta_backward=1.5, beta_tally=2.2), top-A gate-active P(A)=.168 versus gate-inactive P(A)=.464 for the advocated model, while the competing model gives .457 for both; top-B gate-active P(A)=.832 versus gate-inactive P(A)=.536 for the advocated model, while the competing model gives .543 for both. This yields a gate-condition by top-direction interaction unique to the advocated conflict-gated architecture.

[9] This design isolates a route-invariant conflict-gate toggle using three tied lowest-validity cues. Features 5-7 all have validity 0.50. Feature 5 is always tied, so the backward lexicographic scan skips it and stops at feature 6, which always discriminates opposite the highest-validity cue (feature 0). Feature 7 is the critical gate trigger: in gate-active trials it is tied, so the lowest-validity set has a uniform direction and the competing model's conflict gate is live; in gate-inactive trials feature 7 discriminates in the same direction as the top cue, making the lowest-validity set mixed and switching the gate off. A middle feature is adjusted to hold the feature-win tally difference constant (d=+1 for top-A, d=-1 for top-B) across the gate manipulation. The advocated stable-mixture model is blind to feature 7 because forward lex stops at feature 0, backward lex stops at feature 6, and tally d is unchanged; it predicts identical P(A) for matched gate-active and gate-inactive pairs. The competing conflict-gated model predicts a large gate-condition gap: with mid-range parameters, top-A gate-active P(A)=0.168 versus gate-inactive P(A)=0.457; top-B gate-active P(A)=0.832 versus gate-inactive P(A)=0.543, while the advocated model gives 0.464 and 0.536 in both gate conditions. The 32 unique pairs are balanced over top direction, gate condition, and middle-cue patterns, repeated K=3 for 96 trials.

[10] This design isolates a conflict-gate toggle that is invisible to the stable-mixture model. Six experts have a unique highest validity (0.90) and a unique lowest validity (0.50). On every trial the unique lowest cue discriminates opposite the lexicographic top-direction winner, and the middle cues fix the feature-win tally difference at +1 for top-A trials and -1 for top-B trials. The critical manipulation is the highest cue's rating: in gate-active trials feature 0 discriminates in the top direction, so the advocated model's conflict gate is live (tally does not oppose the top cue); in matched gate-inactive trials feature 0 is tied and feature 1 (validity 0.80) discriminates in the same top direction, so the gate is off. Forward lexicographic winner, backward lexicographic winner, and tally difference are identical across matched pairs, so the stable-mixture model must predict identical P(A). The advocated model instead suspends tallying and boosts backward scanning on gate-active trials, producing a large gate-condition by top-direction interaction: with representative mid-range parameters, top-A gate-active P(A) is about 0.17 versus gate-inactive about 0.46; top-B gate-active P(A) is about 0.83 versus gate-inactive about 0.54, while the stable mixture gives about 0.46 and 0.54 in both gate conditions. Unlike earlier tied-validity designs, the top validity is unique; the gate is toggled by tying the top cue's rating, not by changing the validity structure. The 32 unique pairs are balanced over top direction, gate condition, and top-tie value in the gate-inactive cells, and are repeated K=3 for 96 total trials.

[11] This design crosses two factors that the conflict-gated model treats as interacting but the stable-mixture model treats as irrelevant: (i) a gate toggle achieved by tying the highest-validity cue's rating while the second-highest cue carries the same direction, and (ii) the feature-win tally difference d. Validities are [0.90, 0.80, 0.72, 0.61, 0.55, 0.50]; feature 0 is the unique highest cue and feature 5 is the unique lowest cue. On every trial feature 5 discriminates opposite the top-direction winner. In gate-active trials feature 0 itself discriminates in the top direction; in matched gate-inactive trials feature 0 is tied and feature 1 discriminates in the same top direction. This keeps the forward lexicographic winner, backward lexicographic winner, and tally difference d identical across matched pairs, so the stable-mixture model must predict identical P(A). The conflict-gated model, however, is live only when the top cue discriminates and tally does not oppose it: top-A d>=0 and top-B d<=0. There it suspends tallying and boosts backward scanning, producing a large gate gap; when d opposes the top cue (top-A d=-1, top-B d=+1), the gate is off and the gap disappears. With representative parameters (advocated: w_forward=.27, w_backward=.48, beta_forward=.14, beta_backward=1.5, beta_tally=2.2; competing: w_forward=.25, w_backward=.48, beta_forward=.30, beta_backward=1.5, beta_tally=2.2, w_conflict_back=.85, beta_conflict_back=2.5), top-A d=+1 gives advocated P(A)=.457 in both gate conditions, whereas competing gives .474 gate-inactive but .151 gate-active; top-B d=-1 gives advocated .543 in both, competing .526 gate-inactive but .849 gate-active. Top-A d=-1 and top-B d=+1 are controls where both models predict no gate gap. The 32 unique pairs are balanced over top direction, gate condition, d, and tie value, repeated K=3 for 96 trials.

[12] This design drives the advocated model's context-sensitive backward amplification to its peak while critical stimuli stay outside the trigger set, creating a quantitative dissociation from the stable-mixture model. Six experts have validities [0.90, 0.80, 0.72, 0.62, 0.55, 0.50]; feature 5 is the unique lowest-validity cue and is therefore the first cue in every backward/ascending scan. Eight of the 32 unique pairs are inducers: fully diagnostic complementary pairs with tally difference -2, 0, or +2. These satisfy the advocated model's trigger condition (all n features discriminate and abs(tally) <= 2). Because 8 of 32 unique pairs are inducers and each pair is repeated K = 3 times, about 25% of trials qualify, so the trailing 24-trial qualifying fraction sits inside the peak band of the trapezoid and the backward amplification s is near 0.84-0.92 for most of the experiment. The 24 critical pairs are deliberately non-qualifying because at least one feature is tied; they therefore do not change their own trigger status. Every critical pair has the lowest-validity cue discriminating in one direction while m = 1, 2, 3, or 4 other non-tied features discriminate in the opposite direction. With representative base parameters (w_forward = 0.25, w_backward = 0.48, w_tally = 0.27, beta_forward = 0.14, beta_backward = 1.5, beta_tally = 2.2), the stable competing model gives P(A) around 0.51-0.64 for low-cue-A conflicts (graded across m), while the advocated model shifts to about 0.91-0.92 and flattens the m-response curve. The low-cue-B mirror gives competing P(A) around 0.36-0.49, while the advocated model gives about 0.08-0.09. The critical quantitative prediction is therefore a near-flat, low-cue-dominant m-response profile under a qualifying-context background, which a stable mixture cannot produce because its backward weight and beta are capped.

[13] Six experts with validities [0.90, 0.80, 0.72, 0.61, 0.55, 0.50]. Feature 0 is the unique highest validity and feature 5 is the unique lowest, so forward lexicographic scans stop at feature 0 and backward scans stop at feature 5 whenever both discriminate. Every pair is fully complementary, and feature 0 and feature 5 always discriminate in opposite directions. The four middle features set the tally difference d = A_wins - B_wins to -4, -2, 0, +2, or +4. The d = -2, 0, +2 pairs satisfy the context model's trigger condition: all six features discriminate and |d| <= tie_margin. The d = -4 and d = +4 pairs do not qualify themselves because |d| is too large. There are 28 qualifying unique pairs and 4 nonqualifying unique pairs; each unique pair is repeated K = 3, so 84 of 96 trials qualify and only 12 nonqualifying trials exist in the whole session. Therefore any trailing 24-trial window contains at least 12 qualifying trials, so the qualifying fraction can never fall below 0.50, above the fall_end_frac upper bound of 0.48. After the first few qualifying trials the competing context model is locked at its lower amplification floor s_floor around 0.40-0.50, rather than at zero or at the peak. The advocated stable-mixture model has no context term and uses the same fixed forward, backward, and tally mixture throughout. Using representative mid-range values (w_forward = 0.25, w_backward = 0.48, w_tally = 0.27, beta_forward = 0.14, beta_backward = 1.5, beta_tally = 2.2, s_floor = 0.45, beta_backward_strong = 3.25), the critical low-A d = -4 cell, where feature 5 favors A but the other five features favor B, gives advocated P(A) about 0.51, while the competing model predicts P(A) about 0.71. The full parameter-range gap is roughly 0.43-0.61 for the advocated model versus 0.63-0.80 for the competing model. The mirrored low-B d = +4 cell gives advocated P(A) about 0.49 versus competing about 0.29. The d-response slope is also flatter under the context model, producing a quantitative dissociation rather than a main effect. The 32 unique pairs are balanced over low-cue direction and tally difference.

[14] This design isolates the density boundary between the two context-trigger definitions. Every unique pair is a dense five-of-six discriminating, near-tied comparison with the highest-validity cue (feature 0, validity 0.90) conflicting with the lowest-validity cue (feature 5, validity 0.50), exactly one of the four middle cues tied, and a feature-win tally difference of exactly +1 or -1. Under the competing accumulator model these trials are strong evidence-positive events: density 5/6 exceeds the density floor 0.80, abs(tally) = 1 is within tie_margin, and the top-low disagreement earns the hi_lo bonus. The accumulator therefore crosses threshold after the minimum qualifying count and S stays near s_max, with no full-evidence penalty because no fully discriminating trial ever appears. Under the advocated model the trigger requires a fully diagnostic comparison (all six experts disagree), so every trial is non-qualifying; the qualifying count stays zero and the backward-amplification term s stays exactly 0, leaving only the baseline forward/backward/tally mixture. Thus the probe contrast is bottom-cue direction by tally sign. For top-A/low-B pairs the backward scanner stops at feature 5 favoring B; for top-B/low-A pairs it stops at feature 5 favoring A. With representative mid-range parameters (w_forward = 0.25, w_backward = 0.48, w_tally = 0.27, beta_forward = 0.14, beta_backward = 1.5, beta_tally = 2.2, beta_backward_strong = 3.25, advocated s = 0, competing S about 0.80), top-A d = -1 gives advocated P(A) about 0.25 versus competing about 0.08; top-A d = +1 gives advocated about 0.46 versus competing about 0.12; top-B d = -1 gives advocated about 0.54 versus competing about 0.88; and top-B d = +1 gives advocated about 0.75 versus competing about 0.92. The advocated model therefore retains a d-sensitive response, whereas the competing model flattens the response near the bottom-cue winner. This isolates the trigger-density criterion: five-of-six near-ties should activate the competing reversed route but leave the advocated route completely unamplified.

[15] This design separates the advocated accumulator theory from the competing full-diagnostic trigger theory using a 50/50 mixture of fully discriminating near-tied trials and dense five-of-six near-tied trials. Validities are [0.88, 0.78, 0.70, 0.60, 0.54, 0.50], so feature 0 is the unique highest cue and feature 5 is the unique lowest cue; on every pair those two cues discriminate in opposite directions. Sixteen of the 32 unique pairs are full complements with tally difference -2, 0, or +2. The other sixteen are five-of-six pairs with exactly one middle tie and tally difference +1 (top-A/low-B) or -1 (top-B/low-A). Under the advocated theory every trial has density greater than 0.80, so all 32 trials feed the evidence accumulator. Full trials produce evidence near 1.0 and five-of-six top-low conflicts produce roughly 0.15 to 0.22. Because full trials are only 50 percent of the evidence-positive stream, the full-evidence penalty threshold around 0.85 is not approached, and the strengthened lowest-validity-first route reaches s near 0.75 to 0.85. Under the competing theory only the 16 fully discriminating pairs qualify as triggers. With a 50 percent qualifying rate in the trailing 24-trial window, the fraction is above the fall_end_frac maximum of 0.48, so the competing route is locked at its floor s_floor of about 0.40 to 0.50. The critical probes are the five-of-six pairs. For a top-A/low-B probe with tally difference +1, the shared baseline mixture gives P(A) near 0.46 and the strong backward route gives P(A) near 0.05. The advocated model therefore predicts P(A) around 0.13, whereas the competing model predicts around 0.27. The mirrored top-B/low-A probe with tally difference -1 gives advocated P(A) around 0.87 versus competing around 0.73. This produces a stable quantitative dissociation in probe choice probability while keeping the baseline routes identical.

[16] This design dissociates the advocated full-diagnostic frequency trigger from the competing leaky-accumulator trigger by combining three trial classes. Five unique fully discriminating near-tied trigger pairs (Q pairs) have all six experts discriminate with tally A_wins - B_wins in -2, 0, +2. Their top and lowest validities are aligned in the same direction, so they are definitely qualifying events for the advocated model but only weak full-evidence events for the competing model, which receives its hi-lo disagreement bonus only on conflict. These Q pairs are 5/32 = 15.625 percent of unique trials, so the advocated trailing 24-trial qualifying fraction sits in its peak/rise band and its backward amplification s is high in most windows after the minimum count is met. Eleven further fully discriminating pairs have tally +-4, which is outside both models' near-tie margins, so they add zero qualifying events to the advocated count and zero local evidence to the competing accumulator, but they do enter the competing full_frac denominator with full_frac about 16/32 = 0.50, engaging the competing full-evidence penalty if its accumulator ever fires. Sixteen probe pairs are dense five-of-six comparisons with one middle tie, feature 0 and feature 5 always in conflict, and tally +1 or -1. For top-A probes d = -1 and for top-B probes d = +1, so the tally opposes the top cue; the competing model therefore assigns these dense probes zero local evidence because its tally_align gate fails, while the advocated model also treats them as non-qualifying because they are not fully diagnostic. The critical quantitative dissociation is on the probes. Using representative baseline weights w_forward = 0.25, w_backward = 0.48, w_tally = 0.27, the baseline mixture gives P(A) near 0.25 for top-A d = -1 and near 0.75 for top-B d = +1. With advocated s near 0.80 to 0.92 and a strong backward route, the top-A probe falls to about 0.06 to 0.10 and the top-B probe rises to about 0.90 to 0.94. The competing model instead stays near its baseline because the five Q events are expected only about 2.5 per 16-trial qualifying window, below its fixed min_qualifying_count of four, and all fillers/probes contribute zero evidence; predicted values remain near 0.24 to 0.27 for top-A and near 0.73 to 0.76 for top-B. Thus the probe by top-direction interaction of roughly 0.15 to 0.20 is predicted by the advocated frequency-trigger theory but not by the competing accumulator theory.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design isolates the density criterion that separates the advocated accumulator theory from the competing fully-diagnostic frequency trigger. Six experts have distinct validities [0.90, 0.80, 0.70, 0.60, 0.55, 0.50]; feature 0 is the unique highest-validity cue and feature 5 is the unique lowest-validity cue, so forward lexicographic scans stop at feature 0 and backward scans stop at feature 5 whenever both discriminate. On every trial those two cues discriminate in opposite directions, making the forward and backward routes disagree. Exactly four of the six experts discriminate on every trial; the remaining two middle experts are tied. This makes the discrimination density 4/6 = 0.667, which is above the advocated density floor range of 0.56 to 0.62 but below the competing fully-diagnostic criterion of all six experts discriminating. The tally difference among the four discriminating experts is +2, 0, or -2, all within both models' near-tie margins. Because no trial is fully diagnostic (two experts are always tied), the competing context signal is exactly zero throughout the experiment, so its backward amplification s stays zero and choice remains at the baseline forward/backward/tally mixture. The advocated accumulator, in contrast, treats every one of these dense four-of-six top-low conflict trials as evidence-positive: density exceeds the floor, the highest and lowest cues disagree, and the tally does not oppose the top cue. The evidence values are moderate, roughly 0.35 to 0.50, so after the minimum qualifying count of four evidence-positive trials and saturation of the accumulator, S rises toward s_max near 0.72 to 0.78, with no full-diagnostic penalty because full_frac is zero. As a result, on top-A / low-B trials with tally +2, the competing baseline predicts P(A) around 0.48 whereas the advocated model shifts strongly to the lowest-validity-first route and predicts P(A) around 0.16 to 0.18. On the mirrored top-B / low-A tally -2 trials, the competing baseline predicts P(A) around 0.52 whereas the advocated model predicts P(A) around 0.82 to 0.84. The d = 0 cells produce smaller but same-direction gaps of about 0.22. This produces a large low-cue-direction by tally-sign interaction that cannot be produced by any parameter settings of the competing theory because its trigger never fires for non-fully-diagnostic trials. The 32 unique pairs are balanced over top-cue direction, tally difference, which middle experts discriminate, and tied-expert display values, and are repeated K = 3 for 96 trials.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.75,
    0.73
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
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
      1,
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      1,
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
      0,
      0,
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
      0,
      1
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
      0,
      0,
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
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1,
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
      1,
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
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
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
      0,
      1
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
      0,
      0,
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
      0,
      1
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
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
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
      1,
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
      1,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
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
      1,
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 13893,
  "output_tokens": 27174,
  "reasoning_tokens": 25315,
  "provider": "Baidu"
}
```
