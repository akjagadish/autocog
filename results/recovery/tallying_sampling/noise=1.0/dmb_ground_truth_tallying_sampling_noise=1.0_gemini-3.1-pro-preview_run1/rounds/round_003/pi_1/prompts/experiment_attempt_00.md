# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** In complex, multi-attribute binary choice tasks without trial-by-trial feedback, subjects largely fail to integrate cue validities or even simple feature counts. Instead, their decisions are dominated by high response noise, effectively reducing their behavior to random guessing with a slight baseline bias for one option over the other. This explains the ~0.5 choice rates, ~0.5 TTB match rates, and ~0.0 differences across experimental conditions.

**Parameters:**
- bias_a: [0.3, 0.7]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we constructed trials with a compensatory structure. In many trials, one option possesses the single most valid discriminating cue (triggering TTB to choose it), while the other option possesses a larger number of slightly less valid cues such that their sum outweighs the single best cue (triggering WADD to choose the opposite). By distributing the validities linearly from 0.95 to 0.55, we ensure robust differences between the weighted sums and the single-highest-validity cue.

[1] This design quantitatively dissociates WADD from TTB by exploiting the models' different sensitivity to choice margins. Across a sequence of trials, the highest-validity discriminating cue is held constant (meaning TTB always chooses the same option with the exact same confidence), but the number of supporting lower-validity cues systematically varies. TTB predicts a flat choice probability across these trials because it ignores all cues beyond the first discriminator. In contrast, WADD integrates all cues, predicting a graded choice probability that scales with the net difference in weighted features. We repeat this logic across different top-discriminating cues (e.g., by tying the most valid cues) to ensure the flat-vs-graded prediction holds robustly throughout the cue hierarchy.

[2] This design quantitatively dissociates Take The Best (TTB) from Tallying (Equal Weight). TTB stops at the highest-validity discriminating cue, ignoring all other features, while Tallying sums the total number of positive features, completely ignoring their validities. The trials are constructed to pit these two mechanisms against each other. In several trials, one option possesses the single highest-validity discriminating cue (triggering TTB to choose it), but the other option has a strictly greater total number of positive cues (triggering Tallying to choose the opposite). In other trials, the two options have an equal number of positive cues (causing Tallying to guess uniformly), but they differ on high-validity cues (causing TTB to make a deterministic choice). Because TTB only guesses when all features tie (in which case Tallying also guesses), we maximize the divergence between the models by maximizing the occurrences of opposing choices and Tallying-guesses.

[3] To quantitatively dissociate Tallying (Equal Weight) from Take The Best (TTB), this design holds the TTB prediction constant while systematically varying the prediction of Tallying. Across all trials, Option A possesses the highest-validity cue (Cue 1) while Option B does not, meaning TTB will deterministically choose Option A every single time and predict a flat choice probability across the experiment. However, the total number of positive features for each option is systematically varied. In some trials, Option B has up to 3 more positive features than Option A (predicting a strong Tallying preference for B), while in others, Option A has up to 3 more positive features than Option B (predicting a strong Tallying preference for A). Tallying predicts a graded, parametric shift in choice probability for A as the difference in feature counts increases from -3 to +3, whereas TTB predicts a constant choice for A.

[4] This design tests a unique, quantitative prediction of Take The Best (TTB) that sharply contrasts with the Rank-Weighted Additive (WADD) model. TTB predicts that choice confidence is driven exclusively by the single discriminating cue, completely independent of its absolute validity or rank. As long as a cue is the first to discriminate, TTB's binary score difference is always 1 vs 0, predicting a perfectly flat choice probability across trials where the top discriminating cue moves down the hierarchy. In contrast, WADD weights features by an inverse power of their rank. Therefore, WADD predicts a sharply graded, decaying choice probability as the discriminating cue shifts from the 1st rank to the 5th rank. By systematically tying the top N cues and letting the (N+1)th cue discriminate, we can observe whether confidence remains constant (TTB) or decays (WADD). We also include trials where the single discriminating cue is opposed by all remaining lower-ranked cues to further separate the models.

[5] This design quantitatively dissociates the Rank-Weighted Additive (WADD) model from Take The Best (TTB) by isolating the effect of opposing, lower-ranked cues. Across all trials, Option A possesses a single discriminating cue of higher validity than any cue possessed by Option B. TTB predicts that choice probability for Option A will be perfectly flat across all these trials, because it stops at the first discriminating cue and completely ignores the rank, number, and distribution of any opposing lower-validity cues. In contrast, WADD integrates all cues weighted by an inverse power function of their rank. Therefore, WADD predicts a graded choice probability: as the opposing cue(s) held by Option B drop in rank (e.g., from rank 2 to rank 6), their subjective weight decreases, leading to a parametrically increasing preference for Option A. Furthermore, WADD predicts cumulative penalization when Option B holds multiple opposing cues, whereas TTB remains entirely blind to them.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "TTB predicts that choices will be entirely determined by the highest-validity discriminating cue, resulting in large shifts in choice probability depending on whether Option A or B possesses this cue. The competing theory assumes subjects ignore all features and simply guess with a fixed bias, predicting a constant choice rate for Option A across all trials. By systematically varying which option has the top discriminating cue (and tying higher-validity cues to shift the discriminator down the hierarchy), we can observe whether P(A) swings dramatically between near 0 and near 1 (supporting TTB) or remains flat (supporting the random guessing model).",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55
  ],
  "trial_a_ratings": [
    [
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
      1
    ],
    [
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
      1
    ],
    [
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1
    ],
    [
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
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
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
      0
    ],
    [
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
      0
    ],
    [
      0,
      0,
      0,
      1,
      1
    ],
    [
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
      1
    ],
    [
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
  "prompt_token_count": 3361,
  "candidates_token_count": 477,
  "total_token_count": 4837
}
```
