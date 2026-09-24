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
[0] This design creates two orthogonal quantitative dissociations between Take-The-Best (TTB) and Tallying, using 5 cues with well-spread validities (0.95, 0.85, 0.75, 0.65, 0.55). DISSOCIATION 1 (graded vs. flat dose-response curve): Trials C1-C3 hold the TTB-relevant structure constant (the top cue v=0.95 favors A, so TTB always picks A with probability (1-eps)*sigmoid(beta)+eps/2) while systematically increasing the number of lower-validity cues favoring B (tally margins 1-vs-2, 1-vs-3, 1-vs-4). TTB predicts a FLAT p(A) across these three trial types — the number of opposing low-validity cues is irrelevant to a one-reason rule. Tallying predicts a strictly MONOTONE DECREASING p(A) = sigmoid(-k*beta) for k=1,2,3 — a graded dose-response in tally margin. Trials C5-C7 mirror this with B favored by the top cue (TTB flat-high p(B); Tallying increasingly favors A). This is a pure quantitative dissociation: the slope of p(choice) against tally margin is exactly 0 for TTB and strictly negative for Tallying at any beta > 0. DISSOCIATION 2 (parameter-free point prediction): Trials T1-T4 have equal tallies (1 win each) but a clear validity-ordered discriminator (e.g., T1: A wins the v=0.95 cue, B wins only the v=0.55 cue). Tallying MUST predict exactly p(A)=0.50 on these trials for ALL parameter values (the softmax over tied scores is uniform, and the lapse also injects uniform). TTB predicts p(A) = (1-eps)*sigmoid(beta)+eps/2, which is strictly above 0.50 for any beta>0. Thus, if subjects choose systematically (not at chance) on T1/T3 (toward A) or T2/T4 (toward B), Tallying is falsified outright with no free parameters to rescue it — an adversarial knife. Trials G1-G2 are agreement controls (both theories pick the same option) that anchor the noise parameters. 12 unique pairs x K=8 = 96 trials per subject. Note the design also avoids any all-tie (fully degenerate) trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design implements a fully crossed TWO-FACTOR double dissociation between Tallying and Take-The-Best (TTB), which is structurally stronger than a one-dimensional dose-response: each theory predicts a main effect of exactly one factor and a PARAMETER-FREE ZERO effect of the other. Factor 1 is the tally margin d = (A's cue wins) - (B's cue wins), the only thing Tallying's prediction depends on: p_Tally(A) = (1-eps)*sigmoid(beta*d) + eps/2. Factor 2 is the direction s of the highest-validity discriminating cue, the only thing TTB's prediction depends on: p_TTB(A) = (1-eps)*sigmoid(beta*s) + eps/2. The 16 pairs cross d in {-4,-3,-2,-1,0,+1,+2,+3,+4} with s in {+1,-1}. KEY QUANTITATIVE PREDICTIONS: (1) WITHIN-COLUMN FLATNESS falsifies TTB with no free parameters: trials P1-P6, P13, P15 all have s=+1 (top cue v=0.92 favors A), so TTB predicts the IDENTICAL p(A) on all eight for every (beta, eps). Yet d ranges from +4 (P15: A wins 5 of 6 cues) to -4 (P6: A wins ONLY the top cue, B wins the other five). Tallying predicts p(A) falling from ~(1-eps) to ~eps/2 across these same trials. If subjects' p(A) on P6 is any lower than on P15, TTB is falsified outright -- no parameters can rescue a flat function that is observed to slope. The s=-1 column (P7-P12, P14, P16) mirrors this. (2) WITHIN-ROW FLATNESS falsifies Tallying with no free parameters: matched pairs (P2,P11), (P3,P10), (P4,P9), (P5,P8) share the same d with s flipped, so Tallying predicts EXACTLY equal p(A) within each pair for all (beta, eps), while TTB predicts a swing of (1-eps)*(2*sigmoid(beta)-1) -- up to nearly 1.0. The d=0 knife pair (P13, P14) is even sharper: Tallying must predict exactly 0.50 on both (tied tallies give a uniform softmax for any beta, and the lapse also injects uniform), while TTB predicts symmetric deviations toward the top-cue winner. (3) GRADED DOSE-RESPONSE: within each column, Tallying predicts a strictly monotone p(A) in d (sigmoid(beta*d) for d = -4..+4), giving a 9-point psychometric curve at fixed TTB output; TTB predicts a horizontal line. Conversely, across columns at fixed d, TTB predicts the full sigmoid swing while Tallying predicts zero. (4) EVEN THE AGREEMENT TRIALS SEPARATE QUANTITATIVELY: on P15 (d=+4, s=+1) both theories choose A, but Tallying predicts sigmoid(4*beta) vs TTB's sigmoid(beta) -- e.g., at beta=1, 0.982 vs 0.731 -- so confidence on unambiguous trials discriminates the theories too. As an adversarial touch against my own advocated theory, cues are displayed in descending-validity order (0.92 first), which maximally favors TTB's serial search (display order matches its search order) while being irrelevant to Tallying; if Tallying still wins under TTB's best-case display, the evidence is stronger. 16 unique pairs x K=6 = 96 trials; A/B sides are balanced (8 trials TTB-favors-A, 8 TTB-favors-B; 7 tally-favors-each-side, 2 tied), P15/P16 and P6/P12 are label-swapped mirrors that control side bias, no all-tie trials, and every pair requires genuine cue comparison.",
  "validities": [
    0.92,
    0.84,
    0.76,
    0.68,
    0.6,
    0.62
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      1,
      1,
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
      0,
      1,
      1
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
      0,
      1,
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
      1,
      1,
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
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1,
      0,
      0,
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
      1,
      1,
      0,
      0,
      1
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
      1,
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
      1
    ],
    [
      1,
      1,
      0,
      1,
      1,
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
      1,
      1,
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
      0
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
      1,
      1,
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
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 3015,
  "output_tokens": 12596,
  "reasoning_tokens": 11044,
  "provider": "Sail Research"
}
```
