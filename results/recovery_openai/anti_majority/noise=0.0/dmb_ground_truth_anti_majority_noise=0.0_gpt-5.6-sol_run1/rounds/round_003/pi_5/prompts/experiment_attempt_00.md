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
**Description:** Accessibility-Gated Validity Cascade: Communicated validities create a stable, descending cue hierarchy rather than a noisy priority race. On each trial, however, a cue can control choice only if its identity, stated validity, and displayed value are successfully accessible. Accessibility varies stably across people and cues and declines with list length and absolute display position, with a modest accelerating cost for cues near the end of long displays. Search follows the advertised-validity hierarchy among accessible cues and stops at the first accessible cue that discriminates. Lower-ranked evidence is never integrated. If no discriminating cue is accessible, the decision maker guesses. Retrieval failures are distinct from response execution errors: after forming an intended choice, the person can accidentally reverse it or lapse to a random response.

**Parameters:**
- n_features: n_features
- validities: validities
- cue_accessibility_offsets: [(-0.15, 0.15)] * n_features
- baseline_accessibility: [3.2, 3.8]
- list_length_gradient: [0.035, 0.055]
- position_gradient: [0.015, 0.035]
- late_position_cost: [0.003, 0.0055]
- execution_flip: [0.09, 0.13]
- lapse_rate: [0.0, 0.02]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Accessibility-Gated Validity Cascade expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but "
            f"n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    cue_offsets = np.asarray(
        parameters["cue_accessibility_offsets"], dtype=np.float64
    )
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got "
            f"shape {validities.shape}."
        )
    if cue_offsets.ndim != 1 or cue_offsets.size != n_features:
        raise ValueError(
            f"cue_accessibility_offsets must have length {n_features}; "
            f"got shape {cue_offsets.shape}."
        )

    baseline = float(parameters["baseline_accessibility"])
    list_gradient = float(parameters["list_length_gradient"])
    position_gradient = float(parameters["position_gradient"])
    late_position_cost = float(parameters["late_position_cost"])
    execution_flip = float(parameters["execution_flip"])
    lapse_rate = float(parameters["lapse_rate"])

    # Absolute zero-based display position. The quadratic component is a
    # restrained terminal-display cost: it has little effect on early cues
    # but makes cues near the end of long lists appreciably harder to encode
    # and retrieve. Cue offsets represent stable identity/complexity effects.
    positions = np.arange(n_features, dtype=np.float64)
    access_logits = (
        baseline
        - list_gradient * max(n_features - 1, 0)
        - position_gradient * positions
        - late_position_cost * positions * positions
        + cue_offsets
    )

    # Numerically stable logistic transformation.
    access_prob = np.empty(n_features, dtype=np.float64)
    positive = access_logits >= 0.0
    access_prob[positive] = 1.0 / (1.0 + np.exp(-access_logits[positive]))
    exp_x = np.exp(access_logits[~positive])
    access_prob[~positive] = exp_x / (1.0 + exp_x)
    access_prob = np.clip(access_prob, 1e-9, 1.0 - 1e-9)

    differences = stim[0] - stim[1]
    discriminating = differences != 0.0
    if not np.any(discriminating):
        return np.array([0.5, 0.5], dtype=np.float64)

    # Advertised validity fixes the search hierarchy. Stable sorting means
    # equal-validity cues retain their displayed order. Accessibility changes
    # whether a cue participates, never its priority relative to another cue.
    cue_order = np.argsort(-validities, kind="stable")

    # Marginalize exactly over independent trial-level accessibility events.
    # reach_probability is the probability that every earlier discriminating
    # cue in the validity hierarchy was inaccessible. Accessible tying cues do
    # not terminate search and therefore leave this quantity unchanged.
    reach_probability = 1.0
    intended_a = 0.0
    intended_b = 0.0

    for j in cue_order:
        if not discriminating[j]:
            continue

        stop_probability = reach_probability * access_prob[j]
        if differences[j] > 0.0:
            intended_a += stop_probability
        else:
            intended_b += stop_probability

        reach_probability *= 1.0 - access_prob[j]

    # If every discriminating cue is inaccessible, no evidence is available.
    intended_a += 0.5 * reach_probability
    intended_b += 0.5 * reach_probability

    intended_total = intended_a + intended_b
    if not np.isfinite(intended_total) or intended_total <= 0.0:
        intended_a = 0.5
    else:
        intended_a /= intended_total

    # Execution reversal is separate from retrieval failure. A final lapse
    # replaces the executed response with a random choice.
    executed_a = (
        intended_a * (1.0 - execution_flip)
        + (1.0 - intended_a) * execution_flip
    )
    p_a = (1.0 - lapse_rate) * executed_a + lapse_rate * 0.5
    p_a = float(np.clip(p_a, 0.0, 1.0))

    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The 12 unique pairs form six diagnostic comparison patterns plus exact A/B reversals, yielding 96 trials with eight repetitions per pair. In every pattern, the highest-validity cue that discriminates favors the option designated by Take The Best, while the unweighted tally either favors the other option or is tied. The first three pattern types hold TTB's prediction constant while making the opposing tally margin equal to 1, 3, or 5 cues. Thus TTB predicts the same choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 across these margins, whereas Tallying predicts probability (1-epsilon)*sigmoid(-beta*m)+epsilon/2 for the TTB-designated option, with m in {1,3,5}. A fourth pattern gives an exact tally tie, so Tallying predicts 0.5 while TTB remains directional. The fifth and sixth patterns make the highest-validity cue tie and place the first discriminating cue at validity rank 2 or 3; TTB still follows that cue, whereas the majority of lower-ranked cues favors the opposite option by three. Exact reversals balance response side and should reverse TTB's preference without changing effect magnitudes. The broad, strictly ordered validities make the cue cascade salient and unambiguous.

[1] This design tests a quantitative confidence signature rather than a directional conflict. In every pair, the highest-validity discriminating cue and the unweighted majority favor the same option, but the tally margin varies systematically from 1 through 6. The six base patterns have A-minus-B win-count margins of 1, 2, 3, 4, 5, and 6; each also contains at least one cue favoring B, avoiding trivial unanimous dominance. Tallying predicts a graded increase in choice probability for A: (1-epsilon)*sigmoid(beta*m)+epsilon/2, where m is the tally margin. Take The Best always stops at Expert 1 and therefore predicts the same probability, (1-epsilon)*sigmoid(beta)+epsilon/2, at every margin. Margin 1 provides a useful probability anchor because the two rules have the same functional prediction there, whereas margins 2-6 reveal the nonlinear accumulation unique to Tallying. Exact A/B reversals control response-side preferences while preserving each margin. With 12 unique pairs repeated eight times, the experiment contains exactly 96 trials. Unlike previously explored cue-majority conflict designs, this congruent design asks whether additional agreeing comparative evidence changes choice reliability after the top cue has already selected the same winner.

[2] This design isolates a quantitative confidence dissociation between the two cascade theories. The six base pairs place the first discriminating expert at validity ranks 1 through 6; their exact A/B reversals balance response side. Take The Best assigns the same binary winner score on every pair, so it predicts one rank-invariant probability of following the first discriminating cue: q=(1-epsilon)*sigmoid(beta)+epsilon/2. In contrast, every advertised validity is guaranteed to exceed the competing theory's confidence threshold throughout its parameter range: even the weakest expert has minimum subjective confidence 0.75*log(0.57/0.43)=0.211, above the maximum threshold of 0.20. Consequently, Confidence-Capped Cascade Integration also stops at the first discriminating cue, but its evidence magnitude equals sensitivity times that cue's log odds. It therefore predicts a strictly decreasing choice probability as the first discrimination moves from the 95% expert to the 57% expert: p_r=(1-epsilon)*sigmoid(beta*sensitivity*log(v_r/(1-v_r)))+epsilon/2. Thus, direction of choice is matched across theories while the predicted rank-by-confidence gradient is present only under the competing theory. Lower-ranked ratings alternate in direction where possible, preventing most displays from being simple unanimous dominance while remaining irrelevant after stopping. The 12 unique pairs are each repeated eight times, yielding exactly 96 trials.

[3] The 12 pairs comprise six weak-cue conflict levels and their exact A/B reversals, giving 96 trials with eight repetitions per pair. Expert 1 is highly valid (95%) but always ties, so it cannot determine a choice. Experts 2–8 all have validity 50.8%. Even at the advocated theory's maximum validity sensitivity, their subjective confidence is only 1.5*log(0.508/0.492)≈0.048, below its minimum confidence threshold of 0.05. Consequently, no discriminating cue can trigger cascade stopping, and Confidence-Capped Cascade Integration must combine all discriminating weak cues. Within each base pair, Expert 2 favors A and is the first discriminating cue under stable validity-tie ordering, while M of the remaining experts favor B, for M=1,...,6. Take The Best therefore predicts the same above-chance preference for A at every conflict level. The advocated theory instead predicts exact indifference when M=1 and an increasingly strong preference for B as M rises: its integrated evidence is proportional to (1-M)*[log(0.508/0.492)]^integration_exponent. Thus the critical signature is a reversal plus a graded conflict-level effect, neither of which TTB can generate with its single condition-invariant winner probability. Exact option reversals balance response-side biases while preserving these predictions. The tied 95% expert also supplies the required broad validity spread without accidentally activating the advocated theory's stopping rule.

[4] The six base pairs create a cumulative priority-race manipulation and are followed by exact A/B reversals, yielding 12 unique pairs repeated eight times for 96 trials. Expert 1 (95%) and Expert 2 (55%) always tie. The last-displayed expert has 72% validity and favors A in every base pair, whereas one through six earlier-displayed 70% experts favor B. Take The Best sorts cues strictly by advertised validity, skips the tied 95% cue, and always stops at the 72% cue. It therefore predicts the same above-chance probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 of choosing A at every base conflict level, irrespective of how many 70% cues oppose it. Noisy Priority Race Search instead gives the earlier 70% cues a serial-position advantage. Conflict level 1 activates only the 70% cue immediately before the 72% cue; each subsequent level adds the next-earlier 70% cue. Adding a B-favoring competitor strictly increases the race probability that B supplies the stopping cue, so the competing theory predicts a monotonic decline in A choices across levels. The highest conflict level also provides a parameter-robust directional reversal: for the earliest active 70% cue, its minimum priority advantage over the 72% cue is 6(0.15)-5[logit(0.72)-logit(0.70)]-0.16, approximately 0.254. Even at retrieval noise 0.08, that cue alone has over 23 times the race weight of the 72% cue, and all other discriminating race cues also favor B. Thus the competing theory predicts a strong B preference at level 6, while TTB must continue to prefer A. Exact option reversals balance response-side biases and reverse directions without changing the predicted conflict-level profile.

[5] The six base pairs isolate conflicts between two discriminating experts separated by exactly two display positions; all other experts tie and therefore cannot terminate either search rule. In every base pair, the earlier expert has validity 0.55 and favors B, whereas the later expert has slightly to substantially higher validity and favors A. The later-minus-earlier validity-log-odds gaps are approximately 0.01, 0.15, 0.35, 0.60, 0.90, and 1.30. Take The Best always ranks the later expert above the 0.55 expert, so it predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 at all six levels, regardless of the size of the validity gap. Noisy Priority Race Search instead predicts a validity-gap-dependent crossover because the earlier B-favoring expert receives a constant two-position serial advantage. Its later-minus-earlier priority difference is sensitivity*D - 2*serial_priority + (bias_later-bias_earlier). At D≈0.01 this difference is at most 5(0.01)-2(0.15)+0.16=-0.09 across the entire parameter range, so the advocated theory robustly predicts that the nominally less-valid, earlier expert wins the race and B is chosen more often. At D≈1.30 the difference is at least 1(1.30)-2(0.45)-0.16=0.24, so it robustly predicts that the more-valid expert wins and A is chosen more often. Intermediate gaps locate each subject's stochastic priority crossover and should produce a rising psychometric profile, potentially with stable cue-specific deviations from perfect monotonicity. Thus the critical signature is both a directional reversal at the smallest gap and a quantitative increase in following the higher-validity expert as the log-odds gap grows; TTB can only fit one condition-invariant response probability. The intervening tied experts preserve the same two-position separation, while tied 95% and 50% experts provide a broad advertised-validity spread without affecting choices. Each base pair has an exact A/B reversal, balancing response-side preferences. The 12 unique pairs are repeated eight times, yielding exactly 96 trials.

[6] The six base pairs hold Take The Best's prediction constant while moving the sole higher-validity A-favoring discriminating cue through display positions 3, 8, 14, 20, 26, and 32 (zero-based positions 2, 7, 13, 19, 25, and 31). Two lower-validity cues at the first two display positions always favor B; every other cue ties. TTB skips the higher-validity tying cues, stops at the designated target cue, and therefore predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 in all six conditions. The accessibility-gated cascade instead predicts a steep decline in A choices with target position. If a_t is target accessibility and R=(1-a_0)(1-a_1) is the probability that both early B cues are inaccessible, its intended A probability is a_t+0.5*(1-a_t)*R. The linear and quadratic position costs make a_t decrease strongly across conditions, while the two early fallback cues remain highly accessible. This produces a parameter-robust crossover: at position 2, target accessibility is at least about 0.78, whereas at position 31 it is at most about 0.38. Even using the least-accessible permissible fallback cues, intended A probability in the final condition is at most about 0.40, so execution flips and lapses attenuate but do not reverse the predicted B preference. Thus the competitor predicts a declining psychometric profile from a clear A preference to a B preference, while TTB predicts rank- and position-invariant choice reliability. Exact A/B reversals balance response-side preferences. The 12 unique pairs are repeated eight times, yielding 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "The six base conditions manipulate same-direction redundancy within the validity cascade while holding Take The Best's decisive cue fixed. In every base pair, the 95%-valid Expert 24 favors A, so TTB always stops there and predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2. Experts 1 and 2 favor B but have lower validities (72% and 70%). Condition 1 contains only the late-displayed 95% A cue; conditions 2 through 6 successively add A-favoring backup cues with validities 93%, 91%, 89%, 87%, and 85%. All remaining cues tie. Under the Accessibility-Gated Validity Cascade, each added backup supplies another opportunity for an A-favoring cue to be accessible before search reaches the B-favoring cues. If Q_m is the probability that all m active A cues are inaccessible and S is the probability that both B cues are inaccessible, intended P(A)=1-Q_m+0.5*Q_m*S. Thus A choice must increase monotonically as backups are added, with a characteristic diminishing-returns profile from the multiplicative accessibility failures. The manipulation is especially diagnostic because all A cues occupy terminal display positions 19\u201324, where the advocated theory predicts appreciable accessibility failures in a 24-cue list, whereas the two B fallbacks occupy the highly accessible first positions. Execution flips and lapses attenuate but preserve the positive redundancy gradient. TTB cannot produce this condition effect because its 95% cue already discriminates in every condition and lower-validity cues are never consulted. The final six pairs are exact A/B reversals, balancing response-side preferences while preserving the predicted redundancy profile. Twelve unique pairs repeated eight times yield 96 trials.",
  "validities": [
    0.72,
    0.7,
    0.66,
    0.65,
    0.64,
    0.63,
    0.62,
    0.61,
    0.6,
    0.59,
    0.58,
    0.57,
    0.56,
    0.55,
    0.54,
    0.53,
    0.52,
    0.5,
    0.85,
    0.87,
    0.89,
    0.91,
    0.93,
    0.95
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 6143,
  "output_tokens": 3578
}
```
