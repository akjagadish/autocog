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
[0] Goal: dissociate Take-The-Best (one-reason, cue-order-driven, magnitude-blind) from Tallying (equal-weight vote count, margin-graded) both in *direction* and in *graded magnitude* of choice probabilities.

Five experts with clearly spread validities (0.92, 0.80, 0.70, 0.60, 0.53) so the cue cascade is unambiguous and subjects are told expert 1 is much better than experts 3-5.

The 16 unique pairs (each repeated 6x, 96 trials) fall into five diagnostic families, each mirrored so that neither response key is systematically correct:

(1) DIRECTIONAL CONFLICT WITH GRADED TALLY MARGIN. Pairs 1-6: the option winning the top (92%) cue loses on all remaining cues, with the tally margin varied at 3 (1 vs 4 wins), 2 (1 vs 3 wins, one tie) and 1 (2 vs 3 wins). TTB predicts the SAME high probability of choosing the top-cue option in all three, independent of margin; Tallying predicts the OPPOSITE option with a probability that increases monotonically with margin (sigmoid(beta*margin)). This is the core quantitative dissociation: a flat-versus-graded profile across margin, plus opposite signs.

(2) TALLY-TIE / TTB-DISCRIMINATES. Pairs 7-8: A=[1,0,1,0,0] vs B=[0,1,0,1,0]. Feature wins are 2-2, so Tallying must guess (exactly 50/50 regardless of beta), while TTB is driven deterministically by the 92% cue. Any systematic departure from 50% here is fatal to Tallying.

(3) TTB-STRONG / TALLY-WEAK ON A SINGLE LOW CUE. Pairs 9-10: the options tie on the four top cues and differ only on the 53% cue. TTB is fully committed (its stopping rule gives the winner probability sigmoid(beta), the same as for a top-cue win), whereas Tallying only has a margin of 1, its weakest signal. Comparing accuracy here against family (5) below separates the two models' beta scalings.

(4) DEEP-CUE CASCADE CONFLICT. Pairs 13-14: options tie on the 92% cue, the 80% cue favors one option, and the three lower cues all favor the other. TTB stops at cue 2 and chooses against a 3-vs-1 tally. This checks that the cascade (not just the single best cue) governs choice.

(5) AGREEMENT ANCHORS. Pairs 11-12 (4 wins vs 1 win, same direction for both models) and 15-16 (only the top cue discriminates, all else tied) provide low-conflict baselines that pin down epsilon and beta for both models so the conflict trials are interpreted against a calibrated noise level.

Predicted pattern if TTB is right: near-ceiling choice of the top-cue option in families 1-4 with NO modulation by tally margin, and above-chance choice in the tally-tie pairs. If Tallying is right: choice reverses in family 1, scales with margin (m=1 < m=2 < m=3), is at chance in family 7-8, and is weak in pairs 9-10.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate Tallying (advocated) from Take-The-Best (competing) by exploiting the fact that TTB's choice probability is *constant* (sigmoid(beta)) whenever ANY cue discriminates, whereas Tallying's is a graded function of the vote margin, sigmoid(beta*m). The previously explored design put the gradation manipulation only inside directional-conflict trials; here the core diagnostic is a *no-conflict* margin ladder, which cleanly separates the two models on magnitude alone (no direction confound), and it is cross-validated against conflict and tally-tie families that separate them on direction.\n\nSix experts with widely spread, deliberately non-monotonic validities: f0=.93, f1=.58, f2=.86, f3=.51, f4=.74, f5=.65. TTB's cue cascade is therefore f0 > f2 > f4 > f5 > f1 > f3, which is NOT the display order, so cue-identity effects cannot be mimicked by a simple left-to-right reading strategy. 16 unique pairs, each repeated 6x (96 trials), fully key-balanced (every pair appears with A and B swapped).\n\nFAMILY C \trairairs 1-8): MARGIN LADDER WITHOUT CONFLICT. The 93% cue always favors the same option that wins the tally, and the tally margin takes the values m = 1, 2, 3, 4 (each in both key directions). TTB predicts EXACTLY the same accuracy at m=1 as at m=4 (flat line, sigmoid(beta) after lapse). Tallying predicts a steeply rising curve, e.g. at beta=0.6 roughly .65 / .77 / .86 / .92. This is the central quantitative signature: a flat-versus-monotone accuracy profile over margin with the direction of choice held constant, so no directional confound can explain it.\n\nFAMILY D (pairs 9-12): TALLY TIES, TTB DISCRIMINATES. Pairs 9-10 are 2-2 splits where the 93% cue discriminates; pairs 11-12 are 2-2 splits where f0 and f2 TIE and the first discriminating cue is the 74% expert (with the 65%, 58% and 51% experts split against it). Tallying must be exactly 50/50 in all four, irrespective of beta and of which cue is on top. TTB predicts systematic above-chance choice of the top-discriminating-cue option, and \u2014 importantly \u2014 the SAME magnitude for the f0-driven and f4-driven pairs. Any deviation from chance here falsifies Tallying; equality of the f0- and f4-driven deviations is a further TTB-specific prediction.\n\nFAMILY E (pairs 13-16): GRADED DIRECTIONAL CONFLICT. The 93% cue favors one option while the tally favors the other, with tally margin 4 (pair 13/14: 5 wins vs 1) or 2 (pair 15/16: 3 wins vs 1). TTB predicts the same probability of choosing the f0-winner in both, and this probability must equal its Family-C and Family-D accuracy (all are sigmoid(beta)). Tallying predicts the opposite option, with a much larger effect at m=4 than at m=2, and with beta already pinned down by the Family-C ladder \u2014 so the conflict trials are a genuine out-of-sample prediction rather than a free fit.\n\nDECISION RULE: If Tallying is correct, accuracy in Family C rises monotonically with margin, Family D is at chance, and Family E reverses against the 93% expert more strongly at m=4 than m=2, with all effects captured by a single (beta, epsilon). If TTB is correct, one flat probability level sigmoid(beta) accounts for Families C, D (top-cue side) and E (top-cue side) simultaneously, with zero margin modulation anywhere.",
  "validities": [
    0.93,
    0.58,
    0.86,
    0.51,
    0.74,
    0.65
  ],
  "trial_a_ratings": [
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
      1,
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
      1,
      1,
      1,
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
      0,
      0,
      1
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
      0,
      1,
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
      1,
      1,
      0,
      0,
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
      1,
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
      1,
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
      1,
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
      1,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
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
      0,
      0
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
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 5967,
  "output_tokens": 7849
}
```
