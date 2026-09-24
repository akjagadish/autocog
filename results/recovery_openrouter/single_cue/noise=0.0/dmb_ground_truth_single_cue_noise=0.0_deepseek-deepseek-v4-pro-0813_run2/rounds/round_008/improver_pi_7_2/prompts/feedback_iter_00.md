# feedback_iter_00

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_7_1`
- The recommendation below acts on THEORY 2 (= `pi_7_1`).

Keep pi_7_1's prose theory unchanged and regenerate the predict function and parameter ranges. First, make trial_evidence require at least n_features - 1 discriminating features for six features, i.e. only 5-of-6 or fully discriminating trials can supply evidence, and require the highest and lowest validity cues to disagree with tally alignment. This blocks the 4-of-6 trials in Experiment 2 and should move the prediction from about 0.749 toward the observed 0.694 while preserving full-conflict amplifications. Second, remove or heavily suppress the non-full top-tied fallback branch: set top_tied_bonus near zero or require extremely strict density and tally conditions, because Experiments 11, 13, and 14 show near-zero active-inactive contrasts that this fallback currently over-triggers. Third, raise the reversed-route ceiling modestly, for example s_max in roughly 0.82 to 0.92 and/or a stronger beta_backward_strong, to match the strong reversal in Experiment 8, where pi_7_1 predicted -0.542 but observed was -0.671, and the high lowest-cue adherence near 0.84 in Experiments 15 and 16. Fourth, jointly recalibrate the full_fraction penalty so high proportions of fully discriminating trials attenuate S more than at present: Experiment 18 was over-predicted at 0.711 versus observed 0.641, while Experiment 7 must stay near its observed -0.658 and Experiments 15 and 16 must not lose their strong reversal. Fifth, retain the asymmetric rise/decay accumulator, hysteresis, minimum qualifying count, and stochastic choice policy because Experiment 17 strongly supports leaky accumulation over pi_6's frequency trigger. Finally, validate the new parameter ranges against the full set of 18 experiments so the repaired model does not reintroduce the E2 or E13 over-amplification.


## CANDIDATE THEORY
People choose between two products through a baseline mixture of a weak forward lexicographic scan, a moderate lowest-validity-first lexicographic scan, and feature-win tallying. Decision makers continuously accumulate local evidence about cue reliability from recent discriminating trials. Evidence is graded by discrimination density: near-tied trials with only a few discriminating features contribute little or nothing, while dense five-of-six or fully discriminating trials drive a leaky accumulator. The accumulator has asymmetric rise and decay, a minimum qualifying count, hysteresis, and smooth activation of a strengthened lowest-validity-first route. When fully discriminating trials dominate the local evidence stream, the reversed route is gently capped below its maximum so strongly complementary designs reverse strongly but not excessively. When dense conflict is sparse, evidence decays back to baseline mixture weights, preserving near-zero active-inactive contrasts in sparse designs.

`predict(parameters, state, history) -> np.ndarray`:
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

    def trial_evidence(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = x - y
        n_discrim = int(np.sum(d != 0.0))
        if n_discrim < min_discrim_count:
            return 0.0
        tally = float(np.sum(d))
        if abs(tally) > tie_margin:
            return 0.0
        hi_dir = int(np.sign(d[hi_idx]))
        lo_dir = int(np.sign(d[lo_idx]))
        if hi_dir == 0 or lo_dir == 0 or hi_dir == lo_dir:
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


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return 0
    probs = probs / probs.sum()
    return int(np.random.choice(probs.size, p=probs))


`parameters`:
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

`rationale`:
This revision keeps the leaky-accumulator architecture but hardens the evidence gate so only dense near-complete conflicts can activate the reversed route. For six-feature designs it requires at least five discriminating features, directly blocking the four-of-six trials that previously inflated Experiment 18 toward 0.749. The highest- and lowest-validity cues must disagree and the tally must align with the highest cue; there is no top-tied fallback branch, which removes the residual over-triggering in Experiments 11, 12, and similar sparse top-tied designs. While the reversed route still has a raised ceiling via higher s_max and beta_backward_strong to accommodate strong reversals in dense complementary designs, the full-fraction penalty is now deeper and its floor is substantially higher, attenuation high proportions of fully discriminating trials so E16/E18 are pulled down. The asymmetric rise/decay, hysteresis, minimum qualifying count, and stochastic choice policy are retained, preserving the leaky-accumulator dynamics that Experiment 17 supports.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.0586 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.0586.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = None
        # Cues are in descending-validity order, so the first discriminating
        # feature is the Take The Best cue.
        for j in range(len(a)):
            if a[j] > b[j]:
                pred = 0
                break
            if b[j] > a[j]:
                pred = 1
                break
        if pred is not None:
            matches.append(1.0 if r == pred else 0.0)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2487 (var=0.0065)
**Candidate (simulated) value:** 0.2390 (var=0.0023)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8723 (var=0.0096)
- pi_2: 0.1175 (var=0.0068)
- pi_3: 0.2483 (var=0.0023)
- pi_4: 0.2306 (var=0.0024)
- pi_5: 0.2258 (var=0.0024)
- pi_5_1: 0.2360 (var=0.0031)
- pi_5_2: 0.2146 (var=0.0032)
- pi_6: 0.2292 (var=0.0019)
- pi_7: 0.2331 (var=0.0017)
- pi_7_1: 0.2379 (var=0.0032)

### Experiment 2
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        response_a = 1.0 if int(row["response"]) == 0 else 0.0

        # Take The Best style second-cue prediction.
        # By design, feature 0 is tied on every trial, so feature 1 is
        # the first discriminating cue.
        if a[1] > b[1]:
            second_wins_a = True
        elif b[1] > a[1]:
            second_wins_a = False
        else:
            second_wins_a = True

        second_score = response_a if second_wins_a else (1.0 - response_a)

        # Tallying total score. Feature-specific ties contribute zero.
        wins_a = float(np.sum(a > b))
        wins_b = float(np.sum(b > a))
        tally_d = wins_a - wins_b

        if tally_d > 0.0:
            tally_score = response_a
        elif tally_d < 0.0:
            tally_score = 1.0 - response_a
        else:
            # Tallying has no preference on a tied tally.
            tally_score = 0.5

        scores.append(tally_score - second_score)

    return float(np.mean(scores)) if scores else 0.0
```

**Observed (real) value:** 0.4196 (var=0.0128)
**Candidate (simulated) value:** 0.4360 (var=0.0090)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6304 (var=0.0462)
- pi_1: -0.6075 (var=0.0314)
- pi_3: 0.4169 (var=0.0101)
- pi_4: 0.4335 (var=0.0093)
- pi_5: 0.4321 (var=0.0103)
- pi_5_1: 0.4519 (var=0.0082)
- pi_5_2: 0.4796 (var=0.0123)
- pi_6: 0.4558 (var=0.0077)
- pi_7: 0.4373 (var=0.0096)
- pi_7_1: 0.4431 (var=0.0090)

### Experiment 3
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    a1 = data["option_a_ratings"].apply(lambda x: x[1])
    b1 = data["option_b_ratings"].apply(lambda x: x[1])
    f1_a = a1 > b1
    f1_b = b1 > a1
    chose_a = data["response"].astype(int) == 0
    if f1_a.sum() == 0 or f1_b.sum() == 0:
        return 0.0
    return float(chose_a[f1_a].mean() - chose_a[f1_b].mean())
```

**Observed (real) value:** -0.1300 (var=0.0159)
**Candidate (simulated) value:** -0.1346 (var=0.0063)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2375 (var=0.0045)
- pi_2: -0.0158 (var=0.0048)
- pi_1: 0.6592 (var=0.0472)
- pi_4: -0.1325 (var=0.0113)
- pi_5: -0.1446 (var=0.0066)
- pi_5_1: -0.1329 (var=0.0102)
- pi_5_2: -0.1471 (var=0.0086)
- pi_6: -0.1254 (var=0.0091)
- pi_7: -0.1158 (var=0.0055)
- pi_7_1: -0.1113 (var=0.0052)

### Experiment 4
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.92, 0.88, 0.80, 0.72, 0.65, 0.60, 0.55, 0.52])
    order = np.argsort(-validities)
    pos = []
    neg = []
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        first = 'A'
        found = False
        for j in order:
            if a[j] > b[j]:
                first = 'A'
                found = True
                break
            if b[j] > a[j]:
                first = 'B'
                found = True
                break
        if not found:
            continue
        chose_a = 1.0 if row.response == 0 else 0.0
        if first == 'A':
            pos.append(chose_a)
        else:
            neg.append(chose_a)
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(np.mean(pos) - np.mean(neg))

```

**Observed (real) value:** -0.2742 (var=0.0069)
**Candidate (simulated) value:** -0.2363 (var=0.0086)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0075 (var=0.0109)
- pi_3: 0.2375 (var=0.0114)
- pi_1: 0.6925 (var=0.0418)
- pi_4: -0.2879 (var=0.0081)
- pi_5: -0.2883 (var=0.0135)
- pi_5_1: -0.3025 (var=0.0121)
- pi_5_2: -0.3025 (var=0.0138)
- pi_6: -0.2946 (var=0.0138)
- pi_7: -0.2704 (var=0.0106)
- pi_7_1: -0.2825 (var=0.0100)

### Experiment 5
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    a0 = np.array([int(r[0]) for r in data['option_a_ratings']])
    b0 = np.array([int(r[0]) for r in data['option_b_ratings']])
    choose_a = (np.asarray(data['response'], dtype=int) == 0).astype(float)
    top_a = a0 > b0
    top_b = ~top_a
    if not np.any(top_a) or not np.any(top_b):
        return 0.0
    return float(choose_a[top_a].mean() - choose_a[top_b].mean())
```

**Observed (real) value:** -0.6583 (var=0.0598)
**Candidate (simulated) value:** -0.6842 (var=0.0074)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2613 (var=0.0080)
- pi_4: -0.2833 (var=0.0124)
- pi_1: 0.6971 (var=0.0436)
- pi_2: -0.0012 (var=0.0086)
- pi_5: -0.6717 (var=0.0368)
- pi_5_1: -0.5975 (var=0.0258)
- pi_5_2: -0.7096 (var=0.0160)
- pi_6: -0.6721 (var=0.0051)
- pi_7: -0.6529 (var=0.0087)
- pi_7_1: -0.6750 (var=0.0079)

### Experiment 6
**Design**
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    if len(data) == 0:
        return 0.0

    a = np.array([list(x) for x in data['option_a_ratings']], dtype=int)
    b = np.array([list(x) for x in data['option_b_ratings']], dtype=int)
    diff = a - b

    top_cue = diff[:, 0]
    tally_diff = (diff > 0).sum(axis=1) - (diff < 0).sum(axis=1)

    chose_a = (data['response'].to_numpy() == 0).astype(float)

    top_a_plus = (top_cue > 0) & (tally_diff >= 1)
    top_b_minus = (top_cue < 0) & (tally_diff <= -1)

    mean_top_a = np.mean(chose_a[top_a_plus]) if np.any(top_a_plus) else 0.0
    mean_top_b = np.mean(chose_a[top_b_minus]) if np.any(top_b_minus) else 0.0

    return float(mean_top_a - mean_top_b)
```

**Observed (real) value:** -0.6711 (var=0.0489)
**Candidate (simulated) value:** -0.6517 (var=0.0097)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0722 (var=0.0187)
- pi_3: 0.9950 (var=0.0001)
- pi_1: 0.6939 (var=0.0333)
- pi_2: 0.7450 (var=0.0289)
- pi_5: -0.6683 (var=0.0378)
- pi_5_1: -0.6233 (var=0.0227)
- pi_5_2: -0.7394 (var=0.0104)
- pi_6: -0.6406 (var=0.0120)
- pi_7: -0.5633 (var=0.0084)
- pi_7_1: -0.5422 (var=0.0126)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    choices = []
    for _, row in data.iterrows():
        a = [int(x) for x in row['option_a_ratings']]
        b = [int(x) for x in row['option_b_ratings']]
        if a[0] == b[0]:
            continue
        top = 0 if a[0] > b[0] else 1
        top_vec = a if top == 0 else b
        other_vec = b if top == 0 else a
        d_top = 0
        for i in range(len(top_vec)):
            if top_vec[i] > other_vec[i]:
                d_top += 1
            elif other_vec[i] > top_vec[i]:
                d_top -= 1
        if d_top >= 0:
            choices.append(1.0 if int(row['response']) == top else 0.0)
    if not choices:
        return 0.5
    return float(np.mean(choices))
```

**Observed (real) value:** 0.1553 (var=0.0159)
**Candidate (simulated) value:** 0.2437 (var=0.0027)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1807 (var=0.0115)
- pi_4: 0.4160 (var=0.0041)
- pi_1: 0.8197 (var=0.0124)
- pi_2: 0.7093 (var=0.0045)
- pi_3: 0.8537 (var=0.0015)
- pi_5_1: 0.1960 (var=0.0071)
- pi_5_2: 0.1453 (var=0.0037)
- pi_6: 0.1447 (var=0.0028)
- pi_7: 0.1667 (var=0.0021)
- pi_7_1: 0.1767 (var=0.0022)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0
    validities = np.array([0.9, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5], dtype=float)
    j_hi = int(np.argmax(validities))
    j_lo = int(np.argmin(validities))

    a_arr = np.array([np.asarray(x, dtype=float) for x in data['option_a_ratings']])
    b_arr = np.array([np.asarray(x, dtype=float) for x in data['option_b_ratings']])
    resp = np.asarray(data['response'], dtype=float)

    hi_diff = a_arr[:, j_hi] - b_arr[:, j_hi]
    lo_diff = a_arr[:, j_lo] - b_arr[:, j_lo]
    d = (a_arr > b_arr).sum(axis=1) - (b_arr > a_arr).sum(axis=1)

    hi_dir = np.where(hi_diff > 0, 1, np.where(hi_diff < 0, -1, 0))
    lo_dir = np.where(lo_diff > 0, 1, np.where(lo_diff < 0, -1, 0))
    live = (hi_dir != 0) & (lo_dir != 0) & (hi_dir == -lo_dir) & (d * hi_dir >= 0)

    y = np.where(lo_dir > 0, (resp == 0).astype(float),
                 np.where(lo_dir < 0, (resp == 1).astype(float), 0.5))

    if np.sum(live) == 0 or np.sum(~live) == 0:
        return 0.0
    return float(np.mean(y[live]) - np.mean(y[~live]))
```

**Observed (real) value:** 0.0150 (var=0.0027)
**Candidate (simulated) value:** -0.0292 (var=0.0030)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0162 (var=0.0110)
- pi_5: 0.2750 (var=0.0251)
- pi_1: -0.0038 (var=0.0040)
- pi_2: 0.0008 (var=0.0059)
- pi_3: 0.0004 (var=0.0001)
- pi_5_1: -0.0021 (var=0.0064)
- pi_5_2: -0.0021 (var=0.0144)
- pi_6: 0.0238 (var=0.0106)
- pi_7: 0.0208 (var=0.0062)
- pi_7_1: 0.0188 (var=0.0073)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.90, 0.78, 0.66, 0.58, 0.52, 0.52])
    top_idx = int(np.argmax(validities))
    low_idx = np.where(validities == np.min(validities))[0]

    subject_effects = []
    for sid, sub in data.groupby('subject_id'):
        groups = {1: {'active': [], 'inactive': []},
                  -1: {'active': [], 'inactive': []}}
        for _, row in sub.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            top_dir = int(np.sign(a[top_idx] - b[top_idx]))
            if top_dir == 0:
                continue
            lows = [int(np.sign(a[j] - b[j])) for j in low_idx]
            nonzero = [d for d in lows if d != 0]
            if len(nonzero) > 0 and all(d == nonzero[0] for d in nonzero):
                low_dir = nonzero[0]
            else:
                low_dir = 0
            tally_d = int(np.sum(a - b))
            gate = (low_dir != 0 and top_dir == -low_dir and tally_d * top_dir >= 0)
            y_a = 1.0 - float(row['response'])
            signed_y = float(top_dir) * y_a
            groups[top_dir]['active' if gate else 'inactive'].append(signed_y)

        dir_effects = []
        for d in (1, -1):
            active = groups[d]['active']
            inactive = groups[d]['inactive']
            if len(active) > 0 and len(inactive) > 0:
                dir_effects.append(float(np.mean(inactive) - np.mean(active)))
        if dir_effects:
            subject_effects.append(float(np.mean(dir_effects)))

    if not subject_effects:
        return 0.0
    return float(np.mean(subject_effects))
```

**Observed (real) value:** -0.0283 (var=0.0032)
**Candidate (simulated) value:** -0.0096 (var=0.0035)
**Other theories' values on this metric (for reference):**
- pi_5_1: 0.2417 (var=0.0120)
- pi_4: -0.0012 (var=0.0120)
- pi_1: -0.0150 (var=0.0031)
- pi_2: 0.0033 (var=0.0040)
- pi_3: -0.0004 (var=0.0002)
- pi_5: -0.0029 (var=0.0043)
- pi_5_2: 0.0038 (var=0.0084)
- pi_6: -0.0358 (var=0.0108)
- pi_7: -0.0013 (var=0.0043)
- pi_7_1: 0.0021 (var=0.0056)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0, 1]

**Metric**
```python
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a0 = data['option_a_ratings'].apply(lambda x: x[0]).astype(int)
    b0 = data['option_b_ratings'].apply(lambda x: x[0]).astype(int)
    a7 = data['option_a_ratings'].apply(lambda x: x[7]).astype(int)
    b7 = data['option_b_ratings'].apply(lambda x: x[7]).astype(int)

    top_is_a = a0 > b0
    gate_active = a7 == b7

    choose_top = ((top_is_a & (data['response'] == 0)) |
                  (~top_is_a & (data['response'] == 1))).astype(float)

    if gate_active.sum() == 0 or (~gate_active).sum() == 0:
        return 0.0

    return float(choose_top[~gate_active].mean() - choose_top[gate_active].mean())

```

**Observed (real) value:** -0.0008 (var=0.0091)
**Candidate (simulated) value:** -0.0162 (var=0.0097)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0100 (var=0.0111)
- pi_5_1: 0.2717 (var=0.0167)
- pi_1: -0.0088 (var=0.0044)
- pi_2: -0.0075 (var=0.0048)
- pi_3: 0.0021 (var=0.0002)
- pi_5: -0.0029 (var=0.0062)
- pi_5_2: -0.0125 (var=0.0074)
- pi_6: 0.0096 (var=0.0118)
- pi_7: -0.0071 (var=0.0128)
- pi_7_1: 0.0033 (var=0.0095)

### Experiment 11
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.5], dtype=float)
    descending = np.argsort(-validities)

    A = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    B = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
    R = np.asarray(data['response'].tolist(), dtype=float)
    chose_a = (R == 0).astype(float)

    n = A.shape[0]
    if n == 0:
        return 0.0

    top_dir = np.full(n, -1, dtype=int)
    for j in descending:
        mask = (top_dir == -1) & (A[:, j] != B[:, j])
        top_dir[mask] = (B[mask, j] > A[mask, j]).astype(int)

    gate_active = (A[:, 0] != B[:, 0])
    cond = np.full(n, -1, dtype=int)
    cond[(top_dir == 0) & gate_active] = 0
    cond[(top_dir == 0) & (~gate_active)] = 1
    cond[(top_dir == 1) & gate_active] = 2
    cond[(top_dir == 1) & (~gate_active)] = 3

    cell_means = []
    for k in range(4):
        rows = (cond == k)
        if rows.sum() == 0:
            return 0.0
        cell_means.append(float(chose_a[rows].mean()))

    return float(cell_means[1] - cell_means[0] + cell_means[2] - cell_means[3])
```

**Observed (real) value:** -0.0050 (var=0.0198)
**Candidate (simulated) value:** -0.0350 (var=0.0247)
**Other theories' values on this metric (for reference):**
- pi_5_2: 0.6500 (var=0.0457)
- pi_4: 0.0250 (var=0.0283)
- pi_1: -0.0150 (var=0.0273)
- pi_2: 0.0342 (var=0.0199)
- pi_3: -0.0058 (var=0.0006)
- pi_5: 0.5325 (var=0.0892)
- pi_5_1: 0.5525 (var=0.0595)
- pi_6: 0.0367 (var=0.0440)
- pi_7: 0.0408 (var=0.0280)
- pi_7_1: 0.0367 (var=0.0284)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
import numpy as np

def metric(data):
    if data is None or len(data) == 0:
        return 0.0
    validities = np.array([0.9, 0.8, 0.72, 0.61, 0.55, 0.5])
    order = np.argsort(-validities, kind='stable')
    hi_idx = int(order[0])
    second_idx = int(order[1])
    lo_idx = int(order[-1])
    active_low = []
    inactive_low = []
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        resp = row.response
        if a.shape != validities.shape or b.shape != validities.shape:
            continue
        if a[lo_idx] == b[lo_idx]:
            continue
        low_dir = 1 if a[lo_idx] > b[lo_idx] else -1
        low_chosen = 1.0 if ((low_dir == 1 and resp == 0) or (low_dir == -1 and resp == 1)) else 0.0
        tally_d = int(np.sum(a > b) - np.sum(b > a))
        if a[hi_idx] != b[hi_idx]:
            hi_dir = 1 if a[hi_idx] > b[hi_idx] else -1
            if hi_dir == -low_dir and tally_d * hi_dir >= 0:
                active_low.append(low_chosen)
        else:
            if a[second_idx] != b[second_idx]:
                hi_dir = 1 if a[second_idx] > b[second_idx] else -1
                if hi_dir == -low_dir and tally_d * hi_dir >= 0:
                    inactive_low.append(low_chosen)
    if not active_low or not inactive_low:
        return 0.0
    return float(np.mean(active_low) - np.mean(inactive_low))
```

**Observed (real) value:** -0.0178 (var=0.0089)
**Candidate (simulated) value:** 0.0050 (var=0.0119)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0067 (var=0.0117)
- pi_5_2: 0.2894 (var=0.0132)
- pi_1: -0.0050 (var=0.0090)
- pi_2: 0.0017 (var=0.0129)
- pi_3: -0.0067 (var=0.0049)
- pi_5: 0.2600 (var=0.0188)
- pi_5_1: 0.2211 (var=0.0170)
- pi_6: -0.0206 (var=0.0128)
- pi_7: -0.0200 (var=0.0134)
- pi_7_1: -0.0478 (var=0.0131)

### Experiment 13
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.9, 0.8, 0.72, 0.62, 0.61, 0.6], dtype=float)
    low_idx = int(np.argmin(validities))
    a = np.array(data['option_a_ratings'].tolist(), dtype=float)
    b = np.array(data['option_b_ratings'].tolist(), dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0:
        return 0.5
    a_low = a[:, low_idx]
    b_low = b[:, low_idx]
    valid = np.not_equal(a_low, b_low)
    if not np.any(valid):
        return 0.5
    low_cue_choice = np.where(a_low > b_low, 0, 1)
    response = data['response'].to_numpy(dtype=float)
    match = np.equal(response, low_cue_choice).astype(float)
    return float(np.mean(match[valid]))
```

**Observed (real) value:** 0.8425 (var=0.0101)
**Candidate (simulated) value:** 0.8006 (var=0.0026)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8048 (var=0.0022)
- pi_4: 0.5792 (var=0.0037)
- pi_1: 0.1235 (var=0.0079)
- pi_2: 0.3046 (var=0.0038)
- pi_3: 0.1665 (var=0.0008)
- pi_5: 0.7015 (var=0.0047)
- pi_5_1: 0.6865 (var=0.0051)
- pi_5_2: 0.7231 (var=0.0028)
- pi_7: 0.7425 (var=0.0023)
- pi_7_1: 0.7629 (var=0.0021)

### Experiment 14
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data):
    import numpy as np

    # Feature index 5 is the unique lowest-validity expert and discriminates in every pair.
    a5 = data['option_a_ratings'].apply(lambda x: int(x[5])).to_numpy()
    resp = data['response'].to_numpy(dtype=float)

    # A choice is feature-5-consistent if it picks the option whose feature 5 equals 1.
    match = np.where(a5 == 1, 1.0 - resp, resp)
    return float(np.mean(match))

```

**Observed (real) value:** 0.8413 (var=0.0122)
**Candidate (simulated) value:** 0.8250 (var=0.0014)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6440 (var=0.0025)
- pi_6: 0.7856 (var=0.0023)
- pi_1: 0.1590 (var=0.0096)
- pi_2: 0.4971 (var=0.0017)
- pi_3: 0.3656 (var=0.0013)
- pi_5: 0.7935 (var=0.0075)
- pi_5_1: 0.7956 (var=0.0031)
- pi_5_2: 0.8431 (var=0.0024)
- pi_7: 0.8350 (var=0.0017)
- pi_7_1: 0.8244 (var=0.0014)

### Experiment 15
**Design**
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]

**Metric**
```python
import numpy as np

def metric(data):
    validities = np.array([0.9, 0.8, 0.72, 0.61, 0.6, 0.6])
    ascending = np.argsort(validities, kind='stable')
    n_features = len(validities)
    window = 24
    low_cut = 2
    high_cut = 3
    min_trial = 30

    subject_scores = []
    for subject_id, subj in data.groupby('subject_id', sort=False):
        subj = subj.reset_index(drop=True)
        n = len(subj)
        if n <= min_trial:
            continue

        is_full = np.zeros(n, dtype=bool)
        back_cons = np.zeros(n, dtype=float)
        has_winner = np.zeros(n, dtype=bool)

        for i in range(n):
            a = np.asarray(subj.loc[i, 'option_a_ratings'], dtype=float)
            b = np.asarray(subj.loc[i, 'option_b_ratings'], dtype=float)
            diff = a - b
            n_disc = int(np.sum(diff != 0.0))
            tally = float(np.sum(diff))
            if n_disc >= n_features and abs(tally) <= 3.0:
                is_full[i] = True

            winner = -1
            for j in ascending:
                if a[j] > b[j]:
                    winner = 0
                    break
                if b[j] > a[j]:
                    winner = 1
                    break
            if winner < 0:
                continue
            has_winner[i] = True
            back_cons[i] = 1.0 if int(subj.loc[i, 'response']) == winner else 0.0

        prefix = np.concatenate((np.zeros(1), np.cumsum(is_full.astype(float))))
        low_mask = np.zeros(n, dtype=bool)
        high_mask = np.zeros(n, dtype=bool)
        eligible = np.zeros(n, dtype=bool)

        for i in range(n):
            if i < min_trial or i < window - 1:
                continue
            start = i - window + 1
            full_count = prefix[i + 1] - prefix[start]
            eligible[i] = True
            if full_count < low_cut:
                low_mask[i] = True
            elif full_count >= high_cut:
                high_mask[i] = True

        non_full = ~is_full
        low_sel = eligible & low_mask & non_full & has_winner
        high_sel = eligible & high_mask & non_full & has_winner

        if not low_sel.any() or not high_sel.any():
            continue

        low_mean = float(np.mean(back_cons[low_sel]))
        high_mean = float(np.mean(back_cons[high_sel]))
        subject_scores.append(high_mean - low_mean)

    if not subject_scores:
        return 0.0
    return float(np.mean(subject_scores))
```

**Observed (real) value:** 0.0553 (var=0.0224)
**Candidate (simulated) value:** -0.0100 (var=0.0172)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2099 (var=0.0205)
- pi_7: 0.0339 (var=0.0118)
- pi_1: 0.0181 (var=0.0166)
- pi_2: 0.1104 (var=0.0222)
- pi_3: 0.1243 (var=0.0156)
- pi_4: 0.0258 (var=0.0268)
- pi_5: 0.0293 (var=0.0291)
- pi_5_1: 0.0363 (var=0.0247)
- pi_5_2: 0.0421 (var=0.0316)
- pi_7_1: 0.0529 (var=0.0354)

### Experiment 16
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    low_order = [3, 4, 5]
    burn = 24

    sub_diffs = []
    for _, sub in data.groupby('subject_id', sort=False):
        sub = sub.reset_index(drop=True)
        infos = []
        full_flags = []

        for _, row in sub.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            d = a - b
            full_flags.append(1.0 if np.all(d != 0.0) else 0.0)

            if d[0] == 0:
                infos.append(None)
                continue

            top_dir = 1.0 if d[0] > 0 else -1.0
            low_dir = 0.0
            for j in low_order:
                if d[j] != 0.0:
                    low_dir = 1.0 if d[j] > 0 else -1.0
                    break

            if low_dir == 0.0:
                infos.append(None)
                continue

            top_choice = 0 if top_dir > 0 else 1
            matched = 1.0 if int(row['response']) == top_choice else 0.0
            aligned = top_dir * low_dir > 0
            infos.append((matched, aligned))

        aligned_vals = []
        conflict_vals = []

        for i in range(burn, len(sub)):
            window = full_flags[i - 23:i + 1]
            frac = float(np.mean(window))
            if frac < 0.5:
                continue

            info = infos[i]
            if info is None:
                continue

            matched, aligned = info
            if aligned:
                aligned_vals.append(matched)
            else:
                conflict_vals.append(matched)

        if len(aligned_vals) == 0 or len(conflict_vals) == 0:
            continue

        sub_diffs.append(float(np.mean(aligned_vals) - np.mean(conflict_vals)))

    if not sub_diffs:
        return 0.0
    return float(np.mean(sub_diffs))

```

**Observed (real) value:** 0.6412 (var=0.0315)
**Candidate (simulated) value:** 0.6407 (var=0.0166)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7974 (var=0.0100)
- pi_6: 0.6026 (var=0.0164)
- pi_1: -0.0026 (var=0.0106)
- pi_2: 0.2110 (var=0.0313)
- pi_3: 0.2314 (var=0.0135)
- pi_4: 0.3880 (var=0.0187)
- pi_5: 0.5215 (var=0.0194)
- pi_5_1: 0.4401 (var=0.0172)
- pi_5_2: 0.4013 (var=0.0236)
- pi_7_1: 0.7112 (var=0.0135)

### Experiment 17
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    if data.empty:
        return 0.0
    A = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    B = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
    y = data['response'].to_numpy(dtype=float)
    if A.ndim != 2 or A.shape[1] < 6:
        return 0.0
    d = A - B
    n_disc = np.sum(d != 0.0, axis=1)
    tally = np.sum(d, axis=1)
    d0 = d[:, 0]
    d4 = d[:, 4]
    d5 = d[:, 5]
    topA = (d0 > 0.0) & (d5 < 0.0) & (d4 <= 0.0) & (n_disc == 5) & (tally < 0.0)
    topB = (d0 < 0.0) & (d5 > 0.0) & (d4 >= 0.0) & (n_disc == 5) & (tally > 0.0)
    if topA.sum() == 0 or topB.sum() == 0:
        return 0.0
    pB_topA = float(np.mean(y[topA]))
    pB_topB = float(np.mean(y[topB]))
    return float(pB_topA - pB_topB)
```

**Observed (real) value:** 0.5710 (var=0.0228)
**Candidate (simulated) value:** 0.4688 (var=0.0260)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7742 (var=0.0080)
- pi_7_1: 0.5890 (var=0.0153)
- pi_1: -0.6544 (var=0.0529)
- pi_2: 0.7242 (var=0.0366)
- pi_3: 0.4645 (var=0.0136)
- pi_4: 0.4962 (var=0.0247)
- pi_5: 0.5211 (var=0.0189)
- pi_5_1: 0.5061 (var=0.0183)
- pi_5_2: 0.4869 (var=0.0207)
- pi_7: 0.7837 (var=0.0089)

### Experiment 18
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.75, 0.73], dtype=float)
    ascending = np.argsort(validities, kind='stable')
    A = np.array(data['option_a_ratings'].tolist(), dtype=float)
    B = np.array(data['option_b_ratings'].tolist(), dtype=float)
    resp = data['response'].to_numpy()
    diff = A - B
    scores = []
    for i in range(diff.shape[0]):
        bwd = -1
        for j in ascending:
            if diff[i, j] > 0.0:
                bwd = 0
                break
            if diff[i, j] < 0.0:
                bwd = 1
                break
        if bwd != -1:
            scores.append(1.0 if int(resp[i]) == bwd else 0.0)
    if len(scores) == 0:
        return 0.5
    return float(np.mean(scores))
```

**Observed (real) value:** 0.6942 (var=0.0025)
**Candidate (simulated) value:** 0.6940 (var=0.0019)
**Other theories' values on this metric (for reference):**
- pi_7_1: 0.7494 (var=0.0014)
- pi_6: 0.6875 (var=0.0016)
- pi_1: 0.6135 (var=0.0015)
- pi_2: 0.6225 (var=0.0032)
- pi_3: 0.6806 (var=0.0012)
- pi_4: 0.7056 (var=0.0015)
- pi_5: 0.7137 (var=0.0020)
- pi_5_1: 0.7254 (var=0.0025)
- pi_5_2: 0.7427 (var=0.0023)
- pi_7: 0.6952 (var=0.0018)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate stays inside the prescribed leaky-accumulator family and gets many targets right, most impressively E18 (0.6940 vs 0.6942), E16 (0.6407 vs 0.6412), E2 (0.4360 vs 0.4196), E3 and E5. The hardened evidence gate clearly fixed the previous E18 overprediction without destroying the basic behavior. However, several experiments show the joint calibration is not yet right. E7 is overpredicted by a large amount: candidate 0.2437 vs observed 0.1553, meaning the model still picks the top or tally winner too often where humans strongly follow the lowest-validity cue. E15 reverses sign: candidate -0.0100 vs observed +0.0553, so the model currently predicts slightly weaker backward adherence in high full-conflict windows when humans show stronger backward adherence. E17 is also too weak: candidate 0.4688 vs observed 0.5710, a 0.10 miss on the key dense-conflict reversal metric that the arbiter specifically wanted protected. E13 remains 0.04 too low on lowest-cue adherence (0.8006 vs 0.8425), and E6 is still slightly weak (0.6517 vs 0.6711). These are not random noise; they cluster around the reversed route being under-activated in exactly the dense or near-dense 5-of-6 conflict situations where the theory says it should be strongest, while the base forward/tally route leaks too much in E7.",
  "verdict": "regenerate",
  "rationale": "Regenerate the predict function and parameter ranges, but keep the same prose theory and the asymmetric rise/decay, hysteresis, minimum qualifying count, and stochastic policy. The current gate and full_fraction penalty are over-controlling the reversed route. First, make 5-of-6 evidence nearly as strong as fully discriminating evidence: raise near_full_evidence_scale to about 0.95 to 1.0 and/or lower saturation_evidence to about 0.18 to 0.25, so E17's 5-of-6 top-vs-lowest conflicts drive S harder. Second, soften the full_fraction penalty shape: move full_penalty_start upward and reduce full_penalty_depth, or raise full_penalty_floor, so the high-full-fraction windows in E15 produce the observed positive high-minus-low difference instead of the current negative value. E18 should remain near 0.694 mainly through the 5-of-6 gating already added, not through heavy blanket attenuation. Third, reduce the forward/tally contribution in conflict states to bring E7 down from 0.244 toward the observed 0.155: lower w_forward and beta_forward, and/or modestly raise the baseline backward weight, while checking that E1 does not fall below about 0.24. Fourth, raise the reversed-route ceiling moderately, e.g. s_max near 0.86 to 0.92 and beta_backward_strong near 3.4 to 3.8, to lift E13/E14 lowest-cue adherence and E6/E8, but only after the penalty is softened so E18 does not overshoot. Finally, validate all 18 experiments together to ensure the repairs for E7, E15 and E17 do not reintroduce the previous E18 or sparse-design over-amplification."
}
```

## Usage

```json
{
  "input_tokens": 40000,
  "output_tokens": 8013,
  "reasoning_tokens": 7283,
  "provider": "Baidu"
}
```
