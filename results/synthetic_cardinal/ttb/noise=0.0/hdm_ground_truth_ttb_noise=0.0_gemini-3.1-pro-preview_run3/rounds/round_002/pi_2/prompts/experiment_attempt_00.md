# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Heuristic Decision Making domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Choose `rating_max` (>= 1) to set the upper bound of each rating value: rating_max=1 gives binary ratings (but note that Tallying ≡ Equal-Weight on binary features), while rating_max >= 2 gives cardinal ratings that dissociate magnitude-based heuristics (EW, WADD) from sign-based ones (Tallying). Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, rating_max]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities, rating_max, and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see a set of ratings for each product. Each rating comes from an expert and is an integer between 0 and an upper bound fixed for this experiment (higher = more positive). The experts differ in how well they typically predict product quality, and their accuracies are communicated before the experiment begins; they are always shown in the same order. On every trial, decide which product you believe is of higher quality and press F for product A or J for product B. There is no time limit, but please respond as accurately as you can. There is no feedback during the task: try to make the best inference you can from the ratings and the experts' accuracies. Press SPACE when you are ready to begin.

## ADVOCATED THEORY
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    w = np.asarray(parameters["validities"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"validities length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ w

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    return int(np.argmax(probabilities))
```


## COMPETING THEORY
**Description:** Human decision-making in multi-attribute choice is not governed by a single universal heuristic. Instead, individuals differ in their strategies or switch between them, such that the population's choices reflect a mixture of Tallying (which counts strict feature-wise wins and ignores magnitudes and validities) and Weighted Additive (WADD, which integrates both magnitudes and validities). A mixture weight parameter 'alpha' dictates the probability of using Tallying versus WADD on any given trial. Response noise enters through a softmax over the scores of the chosen heuristic, plus an independent lapse rate.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- validities: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    beta = float(parameters["beta"])
    
    # Tallying heuristic
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # WADD heuristic
    w = np.asarray(parameters["validities"], dtype=float)
    wadd_scores = stim @ w
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of heuristics
    alpha = float(parameters["alpha"])
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    return p_mixed
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    return int(np.argmax(probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design dissociates Tallying (advocated) from Weighted Additive (WADD; competing) by exploiting Tallying's insensitivity to both feature validities and cardinal rating magnitudes. We use a 4-feature design with a high rating_max (10) and a spread of validities. In the critical trials, one option wins on a majority of features by a very small margin (which drives Tallying to strongly prefer it), while the other option wins on fewer, higher-validity features by a massive margin (which drives WADD to strongly prefer the latter).

[1] This design dissociates the advocated Weighted Additive (WADD) theory from the competing Tallying heuristic. WADD integrates both the cardinal magnitudes of features and their validities, while Tallying relies solely on the sign of the difference (counting strict feature-wise wins) and ignores both magnitude and validity. By using a cardinal rating scale (rating_max = 10) and a spread of validities, we construct trials where an option wins on a majority of features by a small margin (favored by Tallying) but loses on fewer features by a massive margin or on higher-validity features (favored by WADD). We also include trials where Tallying predicts a tie (equal number of feature wins) but WADD strongly prefers one option due to validity and magnitude differences.

[2] To quantitatively dissociate the advocated mixture model (Tallying + WADD) from the competing pure WADD model, this design systematically pits the two heuristics against each other using cardinal ratings. We include trials where Tallying and WADD strongly disagree (e.g., one option wins by a small margin on many low-validity features, while the other wins by a massive margin on a single high-validity feature). Because the competing model relies entirely on WADD, it will predict extreme choice probabilities for the option with the higher weighted sum. In contrast, the advocated model includes a mixture of Tallying, which will pull the predicted choice probabilities toward the option with more feature-wise wins (or toward 0.5 in case of a Tallying tie). By including trials where the heuristics agree, disagree, and where Tallying predicts a tie but WADD does not, we can isolate the unique variance explained by the Tallying component.

[3] To quantitatively dissociate the advocated pure WADD theory from the competing Mixture (Tallying + WADD) theory, we exploit trials where the two models make qualitatively different predictions about indifference. Specifically, we construct 'WADD-tie' trials where the weighted sums of the two options are exactly identical, meaning pure WADD predicts a 50/50 guess. However, on these same trials, one option strictly wins on a majority of features, meaning Tallying strongly prefers it. The competing mixture model predicts a significant deviation from 50/50 towards the Tallying winner, whereas the advocated pure WADD model predicts strict indifference. We also include 'Tallying-tie' trials where options win on an equal number of features (Tallying predicts 50/50) but the weighted sums differ, allowing us to observe if choice probabilities are dampened (as the mixture model predicts) or extreme (as pure WADD predicts).

[4] To quantitatively dissociate the advocated Mixture theory (Tallying + WADD) from the competing pure WADD theory, this design systematically varies the agreement between Tallying and WADD while carefully controlling the WADD score differences. The critical trials include 'WADD-ties' where the weighted sums are identical (pure WADD predicts 50/50 guessing) but Tallying strongly favors one option (Mixture predicts a preference for the Tallying winner). We also include trials where WADD and Tallying strongly conflict, and trials where Tallying predicts a tie but WADD does not. This allows us to estimate the mixture parameter 'alpha' and identify the presence of Tallying, which pure WADD cannot account for.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- rating_max: Upper bound (inclusive) of each rating value. Ratings are integers in [0, rating_max]. Choose rating_max=1 for binary ratings (note: Tallying and Equal-Weight are behaviorally identical on binary features — if you want to dissociate magnitude-based from sign-based heuristics, choose rating_max >= 2 for cardinal ratings).
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is an integer in [0, rating_max]. `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the advocated pure WADD theory from the competing Mixture (Tallying + WADD) theory, we exploit trials where the two models make qualitatively different predictions about indifference. Specifically, we construct 'WADD-tie' trials where the weighted sums of the two options are exactly identical, meaning pure WADD predicts a 50/50 guess. However, on these same trials, one option strictly wins on a majority of features, meaning Tallying strongly prefers it. The competing mixture model predicts a significant deviation from 50/50 towards the Tallying winner, whereas the advocated pure WADD model predicts strict indifference. We also include 'Tallying-tie' trials where options win on an equal number of features (Tallying predicts 50/50) but the weighted sums differ, allowing us to observe if choice probabilities are extreme (as pure WADD predicts) or dampened towards 0.5 (as the mixture model predicts).",
  "validities": [
    1.0,
    0.5,
    0.5,
    0.5
  ],
  "rating_max": 10,
  "trial_a_ratings": [
    [
      0,
      10,
      10,
      10
    ],
    [
      10,
      3,
      3,
      4
    ],
    [
      0,
      10,
      10,
      10
    ],
    [
      8,
      3,
      3,
      4
    ],
    [
      10,
      0,
      0,
      0
    ],
    [
      0,
      6,
      7,
      7
    ],
    [
      10,
      5,
      0,
      0
    ],
    [
      0,
      0,
      10,
      10
    ],
    [
      10,
      10,
      0,
      0
    ],
    [
      0,
      0,
      10,
      10
    ]
  ],
  "trial_b_ratings": [
    [
      10,
      3,
      3,
      4
    ],
    [
      0,
      10,
      10,
      10
    ],
    [
      8,
      3,
      3,
      4
    ],
    [
      0,
      10,
      10,
      10
    ],
    [
      0,
      6,
      7,
      7
    ],
    [
      10,
      0,
      0,
      0
    ],
    [
      0,
      0,
      10,
      10
    ],
    [
      10,
      5,
      0,
      0
    ],
    [
      0,
      0,
      10,
      10
    ],
    [
      10,
      10,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3331,
  "candidates_token_count": 591,
  "total_token_count": 6644
}
```
