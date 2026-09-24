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
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
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


## COMPETING THEORY
**Description:** Noisy Validity-weighted evidence integration with capacity-interference and one-reason lapses (NVW). People do not use a single fixed heuristic; they pool all the expert ratings they can, but each cue's contribution is weighted by its validity raised to a steepness exponent gamma (gamma = 0 recovers Tallying, gamma -> infinity recovers Take-The-Best, so both classic heuristics are limiting cases). Integration is capacity-limited: beyond a working-memory span of about four cues, every additional cue that must be pooled multiplies the reliability (effective inverse temperature) of the integrated comparison by an interference factor delta < 1, so choice consistency is design-dependent and falls sharply as feature count grows. Each cue is additionally attended only with probability alpha (lapses of attention). On a minority of trials — with a probability omega that is modulated by how strongly the most valid discriminating cue's validity dominates the validity of the opposing side's cues — the agent abandons integration and makes a one-reason decision based solely on that top cue. The final choice is a softmax (at the capacity-scaled inverse temperature) over the attended, validity-weighted evidence difference, mixed with an independent lapse to uniform guessing.

**Parameters:**
- gamma: [0.7, 1.3]
- alpha: [0.9, 1.0]
- beta: [5.0, 11.0]
- delta: [0.18, 0.30]
- epsilon: [0.0, 0.06]
- omega_base: [0.0, 0.04]
- omega_dom: [-0.1, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # NVW: Noisy Validity-weighted integration with capacity-interference
    # and one-reason lapses.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"NVW expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n}."
        )

    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    delta = float(parameters["delta"])
    epsilon = float(parameters["epsilon"])
    omega_base = float(parameters["omega_base"])
    omega_dom = float(parameters["omega_dom"])

    # (1) Validity-steepness weights: gamma = 0 -> Tallying (equal weights),
    # gamma -> infinity -> Take-The-Best (top cue only).
    w = np.power(v, gamma)

    # (3) Attention: each cue independently attended with probability alpha.
    att = np.random.random(n) < alpha
    a_win = (a > b) & att
    b_win = (b > a) & att
    disc = a_win | b_win

    # (3) Capacity / interference: beyond a working-memory span of ~4 cues,
    # each additional cue that must be pooled multiplies the reliability
    # (effective inverse temperature) of the integrated evidence by delta.
    # This makes decision noise grow with feature count.
    excess = max(0, n - 4)
    beta_eff = beta * (delta ** excess)

    def _sig(z):
        # Numerically stable logistic.
        return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))

    if not np.any(disc):
        # No attended discriminating cue: pure guess.
        p_a = 0.5
    else:
        # (1) Integration: softmax over the weighted evidence difference.
        d = float(np.sum(w[a_win]) - np.sum(w[b_win]))
        p_int = float(_sig(beta_eff * d))

        # (2) One-reason mode: the single most valid attended
        # discriminating cue decides (TTB-like), with confidence scaled
        # by that cue's weight and the effective sensitivity.
        idx = np.where(disc)[0]
        top = int(idx[np.argmax(v[idx])])
        s = 1.0 if a[top] > b[top] else -1.0
        p_1r = float(_sig(beta_eff * w[top] * s))

        # (2) Validity-dominance coupling for the one-reason lapse rate:
        # dominance of the top cue over the opposing side's cue validities.
        opp = b_win if s > 0 else a_win
        if np.any(opp):
            dom = float(v[top] - np.mean(v[opp]))
        else:
            dom = float(v[top] - 0.5)
        omega_t = float(np.clip(omega_base + omega_dom * dom, 0.0, 0.95))

        p_a = (1.0 - omega_t) * p_int + omega_t * p_1r

    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent lapse toward uniform guessing.
    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design exploits the core structural difference between Take-The-Best (TTB) and Tallying: TTB's choice (and its predicted choice probability, sigmoid(beta)) is determined solely by the single highest-validity discriminating cue and is therefore INVARIANT to how many lower-validity cues oppose it, whereas Tallying's predicted probability is a softmax over win counts and is therefore GRADED in the tally margin. I use 5 features with a wide, strictly ordered validity spread [0.95, 0.85, 0.75, 0.65, 0.55] so the cue hierarchy is unambiguous. The critical trials (T1-T4 and their mirrors T5-T8) hold the TTB-relevant structure fixed — the top cue (validity 0.95) always discriminates in favor of the same option — while the tally margin against that option grows from 0 (tie) to 1, 2, and 3. TTB predicts a FLAT probability of choosing the top-cue winner across all four margin levels (identical sigmoid(beta) - epsilon mixture), while Tallying predicts a monotonically DECLINING curve: 0.5, sigmoid(-beta), sigmoid(-2beta), sigmoid(-3beta). This is a quantitative, parametric signature: the slope of P(top-cue winner) vs. tally margin is exactly zero under TTB and strongly negative under Tallying. Trials T9-T12 repeat the same logic with the top cue tied and the second cue (0.85) discriminating, and T15-T16 with the third cue (0.75) discriminating, testing whether the dissociation persists deeper in the cue hierarchy (TTB predicts the same flat sigmoid(beta) at every cue level; Tallying's predictions depend only on counts, not on which cue discriminates). Trials T13-T14 are agreement trials (both heuristics favor the same option) serving as a baseline. Every conflict type appears in both A-favored and B-favored versions to absorb side/position bias. With 16 unique pairs, K = 96 // 16 = 6 repetitions yields exactly 96 trials per subject. Predicted dissociation summary: (1) on tally-tie trials with a discriminating top cue (T1, T5, T9, T11), TTB predicts above-chance preference for the cue winner while Tallying predicts exactly 50/50 regardless of beta; (2) across the margin gradient (T1->T4), TTB predicts zero slope, Tallying predicts a steep negative slope; (3) TTB is indifferent to which cue in the hierarchy discriminates (same probability at f0, f1, f2 levels for matched tally structure), while any validity-sensitive weighting would break this invariance.

[1] This design's core novelty is a cue-allegiance flip at FIXED tally counts — the mirror image of the classic margin-gradient design. Tallying's predicted probability depends ONLY on the win counts through a softmax, which is shift-invariant (softmax over [3,2] equals softmax over [1,0]); it is therefore EXACTLY invariant to which features each option wins on and to cue depth. TTB's choice, in contrast, is determined solely by the allegiance of the highest-validity discriminating cue. The factorial core (trials 1-8) crosses tally margin {1, 2} x top-cue allegiance {pro-tally (agree), anti-tally (conflict)}: (i) C+ vs C- (and H+ vs H-) hold the win counts and even the tie placement identical (f0 tied, counts 3-2) while permuting cue allegiances so the top discriminating cue flips from favoring the tally winner to opposing it. Tallying predicts a precisely zero change in P(A) (identical sigma(beta)); TTB predicts a full flip from sigma(beta) to sigma(-beta). (ii) C vs H holds allegiance fixed and doubles the margin: TTB predicts exactly zero change (its probability is sigma(beta) for the top-cue winner regardless of how many cues oppose it), while Tallying predicts sigma(beta) -> sigma(2beta). The two theories thus occupy orthogonal cells of the 2x2: TTB shows a main effect of allegiance only; Tallying a main effect of margin only. The depth series (trials 9-14) places the anti-tally win at f0 (v=.95), f2 (v=.85), or f3 (v=.65), all at margin 1 with counts (3,2) or (2,1): both theories predict FLAT behavior here (TTB's probability is cue-depth invariant; Tallying's softmax is shift-invariant across (3,2)/(2,1)), so any validity-weighted hybrid would break this double invariance — a sharp signature that neither target theory but only mixtures would produce. The extreme anchors (trials 15-16) are 1-vs-5 splits where the top cue alone opposes five cues: TTB predicts P(cue winner) = sigma(beta) — numerically IDENTICAL to its prediction on the margin-1 conflict trials (invariance to 4 extra opposing cues), while Tallying predicts sigma(4beta) for the tally winner, near-deterministic at any reasonable beta. Quantitative dissociation summary: (1) on the allegiance flip (C+/C-, H+/H-), Tallying's predicted slope is exactly 0 and TTB's is -tanh(beta/2); (2) on the margin manipulation (C vs H), TTB's predicted slope is exactly 0 and Tallying's is positive; (3) across the depth series both are flat, jointly excluding weighted hybrids; (4) E1 vs C- isolates TTB's cue-count invariance against Tallying's steep margin gradient. Mirrors of every trial type absorb side/position bias. With 16 unique pairs, K = 96 // 16 = 6, yielding exactly 96 trials per subject.

[2] This design dissociates NVW (validity-weighted, capacity-limited integration with one-reason lapses) from Tallying by exploiting the fact that Tallying's predicted P(A) depends ONLY on the win-count margin: for any beta and epsilon, Tallying assigns exactly ONE probability to every trial with the same (a_wins, b_wins), and exactly 0.5 to every tie trial. NVW instead assigns a graded probability ordered by the validity-weighted evidence difference d = sum(v^gamma over A-won cues) - sum(v^gamma over B-won cues), plus a small one-reason tilt toward the highest-validity discriminating cue. I use n=5 (one cue beyond the working-memory span, so the interference factor delta is active but integration remains informative: beta_eff = beta*delta in [0.9, 3.3]) with a wide validity spread [0.95, 0.85, 0.75, 0.65, 0.55]. The design contains four dissociation axes, each with mirrored (A/B-swapped) versions to absorb side bias: (1) MARGIN-0 VALIDITY GRADIENT (types 1-5): 2-2 tie trials where the validity allocation varies from strongly pro-A (d=+0.60: A wins the two most valid cues) through d=+0.40, +0.20, 0.00, to a top-cue-conflict probe (d=+0.10 with the top cue favoring B). Tallying predicts EXACTLY 0.5 on all five types for every parameter value; NVW predicts a monotone deviation from 0.5 whose sign is parameter-free (because v^gamma preserves validity order for any gamma>0, A winning {f0,f1} vs B winning {f3,f4} yields d>0 at any gamma in [0.7,1.3]), with magnitude sigma(beta_eff*d) roughly 0.61-0.93 at d=0.60. (2) WITHIN-MARGIN-1 SPREAD (types 6-14): nine 3-2 trials spanning d from 1.35 down to 0.15. Tallying predicts an identical probability sigma(beta) on ALL nine; NVW predicts a monotone spread of at least ~0.12 (and up to ~0.5) across the aligned subseries (types 6-10, where the top cue also favors A, so both the integration and one-reason components agree). (3) ORDER REVERSAL: NVW predicts P(type 1; margin 0, d=0.60) > P(type 14; margin 1, d=0.15) strictly for every admissible parameter setting (the d-gap is positive at all gamma, and the one-reason lapse favors A on type 1 but B on type 14), whereas Tallying predicts the strict opposite ordering (0.5 < sigma(beta)). The sign of P(margin-0, high-d) - P(margin-1, low-d) is therefore a decisive, parameter-free test. (4) MATCHED-d ALLEGIANCE FLIP: types 9 vs 11 (both d=0.75) and 10 vs 12 (both d=0.55) hold counts AND the validity-weighted evidence identical while flipping which option wins the top cue; Tallying predicts exactly zero difference, NVW predicts a small positive difference (the one-reason lapse signature, magnitude up to ~0.07). Types 15-16 are margin-2 anchors probing the steepness of the margin gradient (Tallying: sigma(2beta); NVW: sigma(beta_eff*1.90) vs sigma(beta_eff*1.00), itself a within-margin spread Tallying cannot produce). In total Tallying predicts only THREE distinct choice probabilities across all 32 pairs (margin 0, 1, 2 and their complements), while NVW predicts a graded profile ordered by d with allegiance modulations - a sharp structural signature. With 32 unique pairs, K = 96 // 32 = 3, giving exactly 96 trials per subject.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design attacks NVW's central and most falsifiable structural claim: capacity interference. NVW asserts that beyond a working-memory span of ~4 cues, each additional pooled cue multiplies the effective inverse temperature by delta < 1, so choice consistency collapses as feature count grows. Tallying has no capacity limit: its softmax inverse temperature beta is design-independent, and its predicted probability depends only on win counts. I therefore set n_features = 8 (four cues beyond the span), which drives NVW's effective sensitivity to beta_eff = beta * delta^4 <= 11 * 0.30^4 = 0.089. Since the maximum possible validity-weighted evidence difference across all trials is |d| <= sum(v^gamma) <= 6.34 (at gamma = 0.7), NVW's predicted P(A) is HARD-BOUNDED to roughly [0.37, 0.64] on every single trial, for every admissible parameter setting (the one-reason lapse uses the same tiny beta_eff, and epsilon/omega only shrink deviations further; attention lapses at alpha <= 1 compress it even more). Tallying, in contrast, predicts a steep, margin-graded ladder: sigmoid(beta), sigmoid(2*beta), sigmoid(4*beta), sigmoid(6*beta) across the margin gradient G1->G4, and near-deterministic choice on the anchor trials E1/E2 (e.g., sigmoid(8*beta) >= 0.98 for beta >= 0.5). The design thus yields three decisive, parameter-free signatures: (1) CAPACITY CLIFF: observed choice proportions on unambiguous anchors (E1: 8-0 split; E2: 7-1 split) exceeding ~0.7 are impossible for NVW at ANY admissible parameter combination, while trivially accommodated by Tallying; likewise, a spread across the margin ladder (G1 vs G4) exceeding ~0.12 is impossible for NVW (its maximum ladder spread is sigmoid(0.089*4.73) - sigmoid(0.089*0.98) ~ 0.08) but natural for Tallying. (2) FIXED-COUNT VALIDITY GRADIENT: trials S1, S2, S3 all have identical win counts (4,3) but validity-weighted evidence d = +1.63, +0.99, -0.37 (order preserved for all gamma in [0.7, 1.3], verified at both endpoints, since v^gamma preserves validity order). Tallying predicts EXACTLY identical probabilities on all three; NVW predicts a strictly decreasing series with S3 below 0.5. Similarly, the tie trials Z1 (d = +1.11), Z2 (d = +0.25), Z3 (d = -0.19) all have counts (4,4): Tallying predicts exactly 0.5 on all three; NVW predicts a strictly ordered series crossing 0.5 between Z2 and Z3. (3) MATCHED-COUNT ALLEGIANCE: G4 and E2 both have counts (7,1) \u2014 Tallying predicts identical probabilities; NVW predicts P(G4) > P(E2) strictly, because both the integrated evidence (d = 4.73 vs 3.81) and the one-reason component (top cue favors the tally winner in G4, opposes it in E2) align. Conversely, if NVW is correct, subjects should be near-chance everywhere with only a small d-ordered ripple \u2014 a profile Tallying can only mimic by taking beta near 0, which would then destroy the margin ladder Tallying needs elsewhere; the joint pattern (near-chance + d-ordering at fixed counts) is uniquely NVW, while (steep ladder + exact flatness at fixed counts) is uniquely Tallying. Every trial type appears in A/B-mirrored pairs to absorb side and position bias. With 24 unique pairs, K = 96 // 24 = 4, giving exactly 96 trials per subject.",
  "validities": [
    0.98,
    0.9,
    0.82,
    0.74,
    0.66,
    0.6,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
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
      0
    ],
    [
      1,
      1,
      1,
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
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
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
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
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
      0,
      0,
      0
    ],
    [
      0,
      0,
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
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
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
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
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
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
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
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
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
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
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
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
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
      1
    ],
    [
      1,
      1,
      1,
      1,
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
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
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
      1,
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
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
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
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
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
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
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
      1,
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
  "input_tokens": 5062,
  "output_tokens": 16140,
  "reasoning_tokens": 13997,
  "provider": "Sail Research"
}
```
