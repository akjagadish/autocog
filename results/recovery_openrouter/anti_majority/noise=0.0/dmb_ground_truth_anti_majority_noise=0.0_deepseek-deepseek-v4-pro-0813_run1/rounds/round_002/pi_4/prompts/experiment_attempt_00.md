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
**Description:** People scan the displayed features from left to right and accumulate a signed preference signal, without ever committing to a hard lexicographic stop. Each discriminating feature contributes an increment whose weight is a continuous function of serial position, stated validity, and list length. For short lists the position weight is a steep exponential primacy gradient, so the first discriminator dominates probabilistically rather than absolutely. As list length exceeds working-memory capacity the gradient collapses toward a flat weighting function, making the rule behave like a tallying heuristic on long lists. The accumulated score is passed through a softmax with inverse temperature beta, and an independent lapse probability permits occasional intention-independent errors.

**Parameters:**
- lambda_5: [0.692, 0.694]
- capacity_power: [14.0, 18.0]
- validity_power: [0.049, 0.051]
- beta: [4.51, 4.53]
- epsilon: [0.095, 0.105]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # Accept either a dict with option ratings or a (2, n_features) array.
    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Capacity-sensitive evidence accumulation expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Stated validities weakly modulate feature weight.
    if 'validities' in parameters and parameters['validities'] is not None:
        validities = np.asarray(parameters['validities'], dtype=float)
        if validities.shape[0] != n_features:
            validities = np.ones(n_features, dtype=float)
    else:
        validities = np.ones(n_features, dtype=float)

    lambda_5 = float(parameters['lambda_5'])
    capacity_power = float(parameters['capacity_power'])
    validity_power = float(parameters['validity_power'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    # Capacity-sensitive flattening. For n_features = 5 this factor is 1,
    # preserving a strong exponential primacy gradient. For longer lists it
    # shrinks rapidly, approximating equal tallying weights.
    capacity_factor = (5.0 / n_features) ** capacity_power

    lam = lambda_5 * capacity_factor
    nu = validity_power * capacity_factor

    positions = np.arange(n_features, dtype=float)
    weights = np.exp(-lam * positions) * (validities ** nu)

    diff = a - b
    score_a = float(np.sum(weights[diff > 0]))
    score_b = float(np.sum(weights[diff < 0]))
    relative_score = score_a - score_b

    # Stable softmax over the accumulated relative evidence.
    logits = np.array([beta * relative_score, 0.0], dtype=float)
    logits = logits - np.max(logits)
    exponentials = np.exp(logits)
    core_probs = exponentials / np.sum(exponentials)

    probs = (1.0 - epsilon) * core_probs + epsilon * np.full(2, 0.5, dtype=float)
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People execute a display-order Take The Best rule: they scan the physical feature vector left-to-right and stop at the first feature that discriminates between the two options. That first displayed discriminator determines the choice, and later features are ignored. If no feature discriminates, the person guesses. Stated validities do not reorder the scan, but they weakly modulate salience: a first discriminator with lower stated validity is used less decisively than one with higher stated validity. Response noise enters through softmax with inverse temperature beta plus a small lapse probability epsilon.

**Parameters:**
- beta: [0.75, 0.85]
- lambda: [6.0, 8.0]
- epsilon: [0.0, 0.03]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # Accept either a dict with option ratings or a (2, n_features) array.
    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Display-order TTB expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    n_features = stim.shape[1]
    a = stim[0]
    b = stim[1]

    # Optional validities; they affect decisiveness, not cue order.
    if 'validities' in parameters and parameters['validities'] is not None:
        val = np.asarray(parameters['validities'], dtype=float)
        if val.shape[0] != n_features:
            raise ValueError(
                f'validities length {val.shape[0]} != n_features {n_features}.'
            )
    else:
        val = np.ones(n_features, dtype=float)

    # Scan features in physical display order: 0, 1, 2, ..., n-1.
    winner = None
    discriminator = None
    for j in range(n_features):
        if a[j] > b[j]:
            winner = 0
            discriminator = j
            break
        if b[j] > a[j]:
            winner = 1
            discriminator = j
            break

    # Complete tie: no discriminator, so guess.
    if winner is None:
        return np.ones(2, dtype=float) / 2.0

    beta = float(parameters['beta'])
    lam = float(parameters['lambda'])
    epsilon = float(parameters['epsilon'])

    # The first displayed discriminator is the only cue used for the choice.
    # Its stated validity weakly scales the effective inverse temperature.
    v_j = float(val[discriminator])
    salience = 1.0 + lam * max(0.0, v_j - 0.5)
    beta_eff = beta * salience

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax.
    z = beta_eff * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts, dtype=float) / n_opts)
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
[0] Five experts are used with the high-validity cue at display position 3 (list index 2, validity 0.91) so visual display order is not confounded with TTB cue order; TTB re-sorts it to first. Every trial has that top cue discriminating. The lower four cues are arranged to produce tallying margins d = wins(A) - wins(B) of -3, -2, -1, 0 in trials 1-4 and +3, +2, +1, 0 in the mirrored trials 5-8. Under TTB, P(A) should be flat within a top-cue direction: high for all top-cue favors A trials and low for all top-cue favors B trials, regardless of d. Under Tallying, P(A) should increase monotonically with d and stay at chance when d = 0. This creates opposite qualitative predictions at d = -3, -2, -1, +1, +2, +3, and a chance-versus-directional difference at d = 0. Eight unique pairs repeated twelve times give 96 trials per subject.

[1] Five experts are used. The highest-validity expert (91%) is placed at display index 2, so visual order is not confounded with Take The Best's validity order (which re-sorts to: index 2, then 0, then 4, then 1, then 3). Every trial is constructed so that the net feature tally is exactly balanced: A wins on exactly as many features as B wins, so Tallying predicts p(A)=0.5 on every trial. By contrast, Take The Best never tallies; it searches down the validity-ordered cue cascade and uses the first discriminating cue. In trials 1-2 the first discriminator is the top expert and favors A; trials 3-4 favor B; trials 5-6 tie the top expert and use the second expert (favoring A/B); trials 7-8 tie the top two cues and use the third expert (favoring A/B). Thus TTB predicts a high cluster for A-predicting cues and a low cluster for B-predicting cues, while Tallying predicts a flat line at chance. This isolates the core dissociation: balanced evidence produces guessing under tallying but one-reason directional choice under Take The Best, and it does so without repeating the already-explored monotone tally-margin design.

[2] This design dissociates display-order Take The Best (TTB) from Tallying by making the first left-to-right discriminator favor one option while the full feature tally favors the other option, or is exactly balanced. The five validities are non-monotonic across display positions ([0.55, 0.90, 0.80, 0.65, 0.60]), so the first discriminator's validity varies across trials. In pairs 1-6 and 9-10, the first discriminating feature favors one option while the tally margin has the opposite sign, producing opposite directional predictions. In pairs 7, 8, 11, and 12, the tally is exactly balanced (d = 0), so Tallying must predict chance regardless of beta or epsilon, whereas TTB predicts a directional choice based on the first discriminator. Pairs 5 vs 9 and 6 vs 10 hold the tally margin constant (±1) but move the first discriminator from a high-validity feature (0.80) to a low-validity feature (0.55), so TTB predicts different choice probabilities while Tallying predicts identical probabilities. This yields both qualitative directional dissociations and quantitative validity-modulation dissociations. The 12 unique pairs are repeated 8 times for 96 trials.

[3] Eight experts with non-monotonic validities [0.53, 0.91, 0.82, 0.66, 0.59, 0.76, 0.71, 0.87]. The first three display positions form a first-discriminator validity staircase (0.53, 0.91, 0.82). Leading ties place the first discriminator at position 0, 1, or 2. The design is a 3 (first-discriminator validity) x 3 (tally margin d = +4/+2/0 for B-first; -4/-2/0 for A-first) x 2 (first-discriminator direction) factorial. Under Tallying, P(A) depends only on d: high for d=+4, above chance for d=+2, exactly chance for d=0, below chance for d=-2, and low for d=-4; it is identical across the three validity levels for a given d. Under display-order TTB, P(A) depends only on the first displayed discriminator's direction and validity: low for B-first (lowest for v=0.91, highest for v=0.53) and high for A-first, and it is flat across d. Thus all 18 pairs are dissociative: d!=0 trials give opposite directional predictions, and d=0 trials give chance (Tallying) versus directional (TTB). The factorial also provides quantitative tests: Tallying predicts a d effect with no validity effect, while TTB predicts a validity effect with no d effect. The 18 unique pairs are repeated 5 times for 90 trials, close to the 96-trial target.

[4] Five experts with validities [0.55, 0.85, 0.85, 0.85, 0.85]. The low-validity expert occupies display position 0; positions 1-3 are high-validity; position 4 is a high-validity filler that never discriminates. Every unique trial has exactly one discriminating feature, placed at display position p=0,1,2,3 by leading ties, with all later features tied. Thus the tally margin is always +/-1, so any flat-weight tallying process predicts identical P(A) across positions. The advocated display-order TTB predicts a flat P(A) across p=1-3 (same first-discriminator validity 0.85; about 0.93 for A-first and 0.07 for B-first) and a less decisive P(A) at p=0 (validity 0.55; about 0.74 for A-first and 0.26 for B-first). The competing primacy-gradient accumulator predicts a steep position decline: A-first P(A) about 0.94 at p=0, 0.86 at p=1, 0.73 at p=2, and 0.62 at p=3, with mirrored B-first values. This produces a crossover at p=0 versus p=1 and a flat-versus-declining dissociation across p=1-3. Eight unique pairs repeated 12 times give 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design uses a long 12-feature list to push the advocated capacity-sensitive accumulator into its flat-weight regime. With n_features=12, the advocated capacity_factor is (5/12)^14 \u2248 5e-6, so the exponential position gradient and the validity_power modulation are both effectively zero: the model becomes a flat tally of discriminating features with beta\u22484.5 and epsilon\u22480.1. The competing display-order TTB, by contrast, always stops at the first displayed discriminator and ignores all later evidence, with validity weakly modulating decisiveness. The 12 unique pairs form a 2 (first-discriminator validity: 0.55 at position 0 vs 0.90 at position 1) \u00d7 3 (full tally margin) \u00d7 2 (first-discriminator direction) factorial. Crucially, for A-first trials the full tally is negative or zero, and for B-first trials the full tally is positive or zero, so the two theories never make the same directional prediction. The advocated model predicts P(A) \u2248 0.05, 0.06, 0.50, 0.94, 0.95 for d = -2, -1, 0, +1, +2, with no validity effect. Display-order TTB predicts P(A) \u2248 0.72-0.77 for low-validity A-first, \u2248 0.93-0.97 for high-validity A-first, and mirrored values for B-first, with no tally-margin effect. This yields both qualitative directional dissociations and quantitative validity-modulation versus tally-margin dissociations. The 12 unique pairs are repeated 8 times for 96 trials.",
  "validities": [
    0.55,
    0.9,
    0.65,
    0.8,
    0.5,
    0.95,
    0.7,
    0.85,
    0.6,
    0.75,
    0.92,
    0.58
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
      0
    ],
    [
      0,
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
      0
    ],
    [
      0,
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
      0
    ],
    [
      0,
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
      0
    ],
    [
      0,
      0,
      1,
      1,
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
      1,
      1,
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
      1,
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
      0,
      1,
      1,
      1,
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
      1,
      1,
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
      1,
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
      0
    ],
    [
      0,
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
      0
    ],
    [
      0,
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 4901,
  "output_tokens": 17265,
  "reasoning_tokens": 11906,
  "provider": "CoreWeave"
}
```
