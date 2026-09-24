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
[0] This design exploits the core structural difference between Take-The-Best (TTB) and Tallying: TTB's choice (and its predicted choice probability, sigmoid(beta)) is determined solely by the single highest-validity discriminating cue and is therefore INVARIANT to how many lower-validity cues oppose it, whereas Tallying's predicted probability is a softmax over win counts and is therefore GRADED in the tally margin. I use 5 features with a wide, strictly ordered validity spread [0.95, 0.85, 0.75, 0.65, 0.55] so the cue hierarchy is unambiguous. The critical trials (T1-T4 and their mirrors T5-T8) hold the TTB-relevant structure fixed — the top cue (validity 0.95) always discriminates in favor of the same option — while the tally margin against that option grows from 0 (tie) to 1, 2, and 3. TTB predicts a FLAT probability of choosing the top-cue winner across all four margin levels (identical sigmoid(beta) - epsilon mixture), while Tallying predicts a monotonically DECLINING curve: 0.5, sigmoid(-beta), sigmoid(-2beta), sigmoid(-3beta). This is a quantitative, parametric signature: the slope of P(top-cue winner) vs. tally margin is exactly zero under TTB and strongly negative under Tallying. Trials T9-T12 repeat the same logic with the top cue tied and the second cue (0.85) discriminating, and T15-T16 with the third cue (0.75) discriminating, testing whether the dissociation persists deeper in the cue hierarchy (TTB predicts the same flat sigmoid(beta) at every cue level; Tallying's predictions depend only on counts, not on which cue discriminates). Trials T13-T14 are agreement trials (both heuristics favor the same option) serving as a baseline. Every conflict type appears in both A-favored and B-favored versions to absorb side/position bias. With 16 unique pairs, K = 96 // 16 = 6 repetitions yields exactly 96 trials per subject. Predicted dissociation summary: (1) on tally-tie trials with a discriminating top cue (T1, T5, T9, T11), TTB predicts above-chance preference for the cue winner while Tallying predicts exactly 50/50 regardless of beta; (2) across the margin gradient (T1->T4), TTB predicts zero slope, Tallying predicts a steep negative slope; (3) TTB is indifferent to which cue in the hierarchy discriminates (same probability at f0, f1, f2 levels for matched tally structure), while any validity-sensitive weighting would break this invariance.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design's core novelty is a cue-allegiance flip at FIXED tally counts \u2014 the mirror image of the classic margin-gradient design. Tallying's predicted probability depends ONLY on the win counts through a softmax, which is shift-invariant (softmax over [3,2] equals softmax over [1,0]); it is therefore EXACTLY invariant to which features each option wins on and to cue depth. TTB's choice, in contrast, is determined solely by the allegiance of the highest-validity discriminating cue. The factorial core (trials 1-8) crosses tally margin {1, 2} x top-cue allegiance {pro-tally (agree), anti-tally (conflict)}: (i) C+ vs C- (and H+ vs H-) hold the win counts and even the tie placement identical (f0 tied, counts 3-2) while permuting cue allegiances so the top discriminating cue flips from favoring the tally winner to opposing it. Tallying predicts a precisely zero change in P(A) (identical sigma(beta)); TTB predicts a full flip from sigma(beta) to sigma(-beta). (ii) C vs H holds allegiance fixed and doubles the margin: TTB predicts exactly zero change (its probability is sigma(beta) for the top-cue winner regardless of how many cues oppose it), while Tallying predicts sigma(beta) -> sigma(2beta). The two theories thus occupy orthogonal cells of the 2x2: TTB shows a main effect of allegiance only; Tallying a main effect of margin only. The depth series (trials 9-14) places the anti-tally win at f0 (v=.95), f2 (v=.85), or f3 (v=.65), all at margin 1 with counts (3,2) or (2,1): both theories predict FLAT behavior here (TTB's probability is cue-depth invariant; Tallying's softmax is shift-invariant across (3,2)/(2,1)), so any validity-weighted hybrid would break this double invariance \u2014 a sharp signature that neither target theory but only mixtures would produce. The extreme anchors (trials 15-16) are 1-vs-5 splits where the top cue alone opposes five cues: TTB predicts P(cue winner) = sigma(beta) \u2014 numerically IDENTICAL to its prediction on the margin-1 conflict trials (invariance to 4 extra opposing cues), while Tallying predicts sigma(4beta) for the tally winner, near-deterministic at any reasonable beta. Quantitative dissociation summary: (1) on the allegiance flip (C+/C-, H+/H-), Tallying's predicted slope is exactly 0 and TTB's is -tanh(beta/2); (2) on the margin manipulation (C vs H), TTB's predicted slope is exactly 0 and Tallying's is positive; (3) across the depth series both are flat, jointly excluding weighted hybrids; (4) E1 vs C- isolates TTB's cue-count invariance against Tallying's steep margin gradient. Mirrors of every trial type absorb side/position bias. With 16 unique pairs, K = 96 // 16 = 6, yielding exactly 96 trials per subject.",
  "validities": [
    0.95,
    0.9,
    0.85,
    0.65,
    0.6,
    0.5
  ],
  "trial_a_ratings": [
    [
      0,
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
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
      1,
      0,
      0,
      1,
      1,
      1
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
      1
    ],
    [
      0,
      1,
      1,
      1,
      0,
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
      0,
      0,
      1,
      1
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
      0,
      0,
      1,
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
      0
    ],
    [
      1,
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
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      1,
      0,
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
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
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
      0,
      0,
      1,
      1,
      1
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
      1,
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
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
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
      0,
      0,
      1,
      1
    ],
    [
      0,
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 3077,
  "output_tokens": 19401,
  "reasoning_tokens": 18038,
  "provider": "Sail Research"
}
```
