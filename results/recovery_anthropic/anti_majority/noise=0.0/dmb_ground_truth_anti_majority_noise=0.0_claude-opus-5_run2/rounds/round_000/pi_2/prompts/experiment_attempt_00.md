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
[0] Five experts with well-spread validities (0.90, 0.80, 0.70, 0.60, 0.55) so that the validity ordering is unambiguous and TTB's cue cascade is well defined. The design contains three diagnostic trial classes plus controls, each presented in both A/B mirror forms to remove side bias.

(1) CONFLICT trials (pairs 1-6): the highest discriminating cue favors one option while the majority of remaining cues favors the other. E.g. A=[1,0,0,0,0] vs B=[0,1,1,1,1]: TTB stops on expert 1 and picks A with probability ~sigmoid(beta), whereas Tallying counts 1 vs 4 and picks B with probability ~sigmoid(3*beta) \u2014 the two theories predict opposite responses, and Tallying's tally margin (3) is large so its prediction is nearly deterministic for any moderate beta. Pairs 3-4 use a smaller tally margin (1 vs 2), so Tallying predicts a *weaker* (margin-1 softmax) preference for the option TTB rejects; the graded difference across margin sizes is a quantitative signature that separates the models beyond mere direction.

(2) TALLY-TIE trials (pairs 7-10): the feature-wise wins are exactly balanced, so Tallying is forced to 50/50 regardless of beta, while TTB still discriminates decisively. Pairs 7-8 tie 2-2 with the top cue discriminating (TTB confident); pairs 9-10 tie 1-1 with all three high-validity cues equal so TTB is driven by the low-validity expert 4. If subjects show systematic (non-chance) preferences on these trials, Tallying cannot account for it at any parameter setting.

(3) CONTROL trials (pairs 11-14) where both theories agree, providing clean leverage on beta and epsilon so that the diagnostic trials are interpreted with well-identified noise parameters.

Pair 5-6 additionally makes the top cue non-discriminating (both options rated 1 by expert 1), forcing TTB down to expert 2 while Tallying reads a 1-vs-3 margin the other way \u2014 testing whether people truly stop at the first *discriminating* cue rather than integrating.

Advocated theory (TTB) predicts: choice of the option favored by the highest discriminating cue on ALL of pairs 1-10, with roughly constant accuracy independent of how many lower cues oppose it, and above-chance systematic choice on the tally-tie pairs. Tallying predicts the opposite direction on pairs 1-6 with strength graded by tally margin, and exact chance on pairs 7-10.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "I advocate TALLYING and design for a quantitative dissociation that the previous experiment (which emphasized direction reversals only) did not isolate: the *graded* dependence of choice probability on the tally margin versus TTB's flat, margin-independent confidence.\n\nSix experts with clearly ordered, well-spread validities (0.90, 0.82, 0.74, 0.66, 0.58, 0.53) make TTB's cue cascade unambiguous (cue order = feature order). Every pair appears in both A/B mirror forms (16 unique pairs, K=6 -> 96 trials) so side bias cancels.\n\n(1) GRADED-AGREEMENT trials (pairs 1-6). Both models favour the same option, but the tally margin is +1 (3 wins vs 2), +3 (4 vs 1), and +5 (5 vs 0). TTB always stops at expert 1 and therefore predicts EXACTLY the same accuracy, (1-eps)*sigmoid(beta)+eps/2, in all three conditions; its binary score never changes. Tallying predicts monotonically increasing accuracy, sigmoid(1*beta) < sigmoid(3*beta) < sigmoid(5*beta). Because these conditions also pin down beta and epsilon, a reliable accuracy gradient across margin 1 -> 3 -> 5 cannot be absorbed by any TTB parameter setting: TTB is forced to attribute all three to one sigmoid(beta). This is the core quantitative signature, and it is measured on trials where the two theories agree in DIRECTION, so it is not contaminated by any direction-reversal strategy shift.\n\n(2) CONFLICT trials (pairs 7-12). Expert 1 favours one option while the remaining experts favour the other, with tally margins of -1 (2 vs 3), -3 (1 vs 4) and -4 (1 vs 5). TTB predicts choice of the expert-1 option with the SAME probability in all three; Tallying predicts choice of the opposite option with strength graded by margin. Direction and gradient are both diagnostic here, and the -1 case is the crucial one: Tallying predicts a weak reversal (sigmoid(beta)) exactly matching in magnitude what it predicts for the +1 agreement pairs.\n\n(3) DESCENT trial (pairs 13-14): experts 1 and 2 both rate the two products 1, so TTB must descend to expert 3 and choose that option, while the tally is 1 vs 3 the other way. This tests whether people truly stop at the first discriminating cue or integrate across the three low-validity experts.\n\n(4) TALLY-TIE trial (pairs 15-16): A wins on experts 1-2, B wins on experts 3-4 (2-2 tie). Tallying is forced to exactly 50/50 for any parameters; TTB predicts a decisive expert-1-based preference. This is the competing theory's unique prediction and provides a hard falsification test of my advocated model; including it makes the design fair rather than stacked.\n\nPredictions if Tallying is right: accuracy rises steeply with |tally margin| in both the agreement and conflict blocks, choices reverse against expert 1 on pairs 7-12 and 13-14, and choices sit at chance on pairs 15-16. If TTB is right: flat accuracy across all margins, expert-1 option chosen throughout pairs 1-12, expert-3 option on 13-14, and clearly above-chance systematic choice on the tally-tie pairs.",
  "validities": [
    0.9,
    0.82,
    0.74,
    0.66,
    0.58,
    0.53
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
      0,
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
      0,
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
      0,
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
      1,
      1
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
      0,
      0,
      1,
      1,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
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
      1,
      1,
      0,
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
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 5758,
  "output_tokens": 6165
}
```
