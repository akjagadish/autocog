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
**Description:** People compare options through a conflict-gated dual-lexicographic/tallying system. A baseline mixture combines a weak forward Take-The-Best route, a moderate backward lexicographic route, and feature-win tallying. When the highest-validity and lowest-validity cues discriminate in opposite directions and the tally does not oppose the highest cue, the decision maker enters a conflict regime: tallying is suspended and the choice is made with high probability by a strong backward lexicographic scan over ascending validities. Subjects differ substantially in how strongly and deterministically they reverse on such conflict trials.

**Parameters:**
- w_forward: [0.17, 0.30]
- w_backward: [0.36, 0.60]
- beta_forward: [0.06, 0.22]
- beta_backward: [1.20, 1.80]
- beta_tally: [1.50, 2.90]
- w_conflict_back: [0.60, 0.95]
- beta_conflict_back: [1.00, 4.00]
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
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

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
            return np.ones(2, dtype=float) / 2.0
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    def set_direction(indices):
        signs = []
        for j in indices:
            if a[j] > b[j]:
                signs.append(1)
            elif b[j] > a[j]:
                signs.append(-1)
        if not signs:
            return 0
        if all(s == signs[0] for s in signs):
            return signs[0]
        return 0

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_conflict_backward = float(parameters['beta_conflict_back'])

    p_forward = lex_probabilities(descending, beta_forward)
    p_backward_base = lex_probabilities(ascending, beta_backward)
    p_backward_conflict = lex_probabilities(ascending, beta_conflict_backward)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally_base = 1.0 - w_forward - w_backward
    if w_tally_base < 0.0:
        w_tally_base = 0.0
        denom = w_forward + w_backward
        w_forward /= denom
        w_backward /= denom

    v_max = np.max(validities)
    T = np.where(validities == v_max)[0]
    v_min = np.min(validities)
    L = np.where(validities == v_min)[0]

    top_dir = set_direction(T)
    low_dir = set_direction(L)
    tally_d = int(a_wins - b_wins)

    live_conflict = (
        top_dir != 0 and
        low_dir != 0 and
        top_dir == -low_dir and
        tally_d * top_dir >= 0
    )

    if live_conflict:
        w_forward_eff = 1.0 - float(parameters['w_conflict_back'])
        w_backward_eff = float(parameters['w_conflict_back'])
        w_tally_eff = 0.0
        p_backward = p_backward_conflict
    else:
        w_forward_eff = w_forward
        w_backward_eff = w_backward
        w_tally_eff = w_tally_base
        p_backward = p_backward_base

    weights = np.array([w_forward_eff, w_backward_eff, w_tally_eff], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2, dtype=float) / 2.0

    p = np.clip(p, 0.0, None)
    return p / p.sum()
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People choose between two options through a stable, subject-level mixture of three comparison routes: (1) forward lexicographic scanning, inspecting cues in descending validity order and stopping at the first discriminating cue; (2) backward lexicographic scanning, inspecting cues in ascending validity order and stopping at the first discriminating cue in that reversed order; and (3) feature-win tallying, where each option receives one point per feature on which it strictly beats the other and ties contribute nothing. The forward and backward lexicographic weights jointly act as a subject-level polarity parameter for first-validity-cue effects: forward weight supports a positive first-discriminator effect, while backward weight supports a negative one. Each route's preferred option is passed through its own softmax choice rule, so responding is noisy but not controlled by a uniform lapse.

**Parameters:**
- w_forward: [0.22, 0.32]
- w_backward: [0.42, 0.54]
- beta_forward: [0.08, 0.20]
- beta_backward: [1.30, 1.70]
- beta_tally: [1.80, 2.60]
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
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Preferred validity vector comes from the experiment.  If the
    # parameter resolver ever leaves it unresolved, fall back to a
    # descending validity vector so the model is still scorable.
    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

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
            return np.ones(2, dtype=float) / 2.0

        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)

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
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2, dtype=float) / 2.0

    p = np.clip(p, 0.0, None)
    return p / p.sum()
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Six experts with validities [0.90, 0.78, 0.66, 0.58, 0.52, 0.52]. Features 4 and 5 are tied for lowest validity; because both theories use stable argsort, feature 4 is the first cue in ascending/backward scans. Every trial sets feature 0 (highest) and feature 4 (first-lowest) to discriminate in opposite directions. The middle features fix the tally difference at +1 for top-A trials and -1 for top-B trials. The critical manipulation is feature 5, the second tied-lowest cue: in gate-active trials it is tied, so the advocated model's low_dir is nonzero and its conflict gate is live (tally does not oppose the top cue); in matched gate-inactive trials feature 5 discriminates in the same direction as the top cue, making low_dir mixed and switching the gate off. The competing stable-mixture model is blind to this manipulation: forward lex stops at feature 0, backward lex stops at feature 4 before feature 5, and tally sees the same d, so it predicts identical P(A) across gate-active and gate-inactive pairs within each top direction. With representative mid-range parameters (advocated: w_forward=.25, w_backward=.48, beta_forward=.14, beta_backward=1.5, beta_tally=2.2, w_conflict_back=.80, beta_conflict_back=2.5; competing: w_forward=.27, w_backward=.48, beta_forward=.14, beta_backward=1.5, beta_tally=2.2), top-A gate-active P(A)=.168 versus gate-inactive P(A)=.464 for the advocated model, while the competing model gives .457 for both; top-B gate-active P(A)=.832 versus gate-inactive P(A)=.536 for the advocated model, while the competing model gives .543 for both. This yields a gate-condition by top-direction interaction unique to the advocated conflict-gated architecture.",
  "validities": [
    0.9,
    0.78,
    0.66,
    0.58,
    0.52,
    0.52
  ],
  "trial_a_ratings": [
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
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0
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
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
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
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
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
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
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
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
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
      0,
      1,
      0,
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
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0
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
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      1,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 6849,
  "output_tokens": 25758,
  "reasoning_tokens": 17288,
  "provider": "CoreWeave"
}
```
