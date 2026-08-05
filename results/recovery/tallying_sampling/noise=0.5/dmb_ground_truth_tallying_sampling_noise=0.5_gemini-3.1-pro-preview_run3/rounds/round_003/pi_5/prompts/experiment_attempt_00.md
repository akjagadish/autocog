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
**Description:** Mixture of Subjective WADD and Tallying: Decision-makers probabilistically mix between a simple equal-weight heuristic (Tallying) and a weighted additive rule that uses subjective, free-varying feature weights rather than objective validities. The subjective weights allow the WADD component to capture non-compensatory, Take-The-Best-like behavior (by assigning heavily skewed weights to features), while the Tallying component accounts for the strong equal-weighting pull observed when individuals fall back on simply counting positive features. Choice probabilities are a mixture of the softmax probabilities derived from each strategy, further blended with a uniform lapse rate to account for response errors. Increased choice noise bounds allow the model to better match human sub-optimal choice frequencies.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 1.0]
- w_tally: [0.0, 1.0]
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Subjective WADD ---
    # Uses free subjective weights instead of objective validities
    scores_wadd = stim @ w
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

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

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

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
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD), this design employs a 'compensatory' structure across four features with varying validities. TTB decides solely on the highest-validity discriminating cue, ignoring all other cues. WADD, conversely, integrates all cues weighted by their validities. The trials are constructed so that one option always wins on the single most valid discriminating cue (favored by TTB), while the other option has a larger sum of validities across multiple less valid cues (favored by WADD). Baseline trials where both heuristics agree are also included.

[1] To quantitatively dissociate WADD from TTB, we use five features with a linear spread of validities. The trials are designed as 'compensatory' choices where the single best discriminating cue (which strictly dictates the TTB choice) is pitted against a coalition of multiple lower-validity cues (which dictate the WADD choice). For instance, in several pairs, one option has a '1' on the highest available discriminating validity but '0's everywhere else, while the other option has '0' on that highest cue but '1's on multiple remaining cues, making the weighted sum larger for the latter. This creates a strong divergence in predicted choice probabilities between the two models.

[2] To quantitatively dissociate Tallying from WADD, this design leverages five features with a skewed distribution of validities (two high, three low). Tallying ignores validities and simply counts the number of positive features, while WADD computes a validity-weighted sum. The trial set includes 'conflict' pairs where one option has fewer positive features but those features have much higher validities (favored by WADD), while the other option has more positive features of lower validities (favored by Tallying). We also include 'tie' pairs where both options have the same number of positive features (Tallying predicts guessing) but differ in validity (WADD strongly prefers one), as well as 'agreement' pairs where both heuristics favor the same option. This design maximizes the divergence in predicted choice probabilities.

[3] To quantitatively dissociate WADD from Tallying, this design uses 5 features with a split in validities (two high, three low). Tallying ignores validities and simply counts the number of positive features, while WADD computes a validity-weighted sum. The trials are constructed to create critical dissociations: (1) Tallying predicts a tie (equal number of positive features) while WADD predicts a strong preference for the option with high-validity features, and (2) Tallying predicts a preference for one option (more positive features) while WADD predicts a preference for the other (fewer but higher-validity features). We also include agreement trials to establish baseline choice behavior.

[4] To quantitatively dissociate the Mixture of Heuristics (Tallying + WADD) from a pure WADD model with subjective weights, we use a 5-feature design with a clear split between high-validity and low-validity features. The trials are constructed to include 'conflict' cases where WADD and Tallying strongly disagree (e.g., fewer high-validity features vs. more low-validity features), 'tie' cases where Tallying predicts guessing but WADD has a strict preference, and 'agreement' cases. Because the Mixture model computes a convex combination of two distinct choice probability distributions (one based on unweighted sums, one on weighted sums), it generates a non-linear profile of choice probabilities across these trial types. A pure WADD model, which uses a single softmax over a single set of linearly weighted sums, cannot simultaneously capture the flat 'tie' behavior of Tallying and the sharp 'conflict' preferences without its subjective weights being pulled in contradictory directions.

[5] To quantitatively dissociate a pure Weighted Additive (WADD) model with subjective weights from a Mixture of Heuristics (Tallying + objective WADD), this design uses 5 features with a linear spread of validities. The trials comprehensively pit individual higher-validity features against combinations of lower-validity features. The Mixture model is restricted to a convex combination of equal weights (Tallying) and strictly objective validities (WADD). In contrast, the advocated pure WADD model allows for subjective weighting, meaning it can capture non-linear transformations of the validities (e.g., extreme overweighting of the most valid cue, resembling Take-The-Best, or idiosyncratic trade-offs between middle-tier cues) that cannot be expressed as any mixture of objective WADD and Tallying. By spanning diverse trade-offs across the validity hierarchy, we can identify choice probability profiles that fall outside the rigid manifold of the Mixture model but are perfectly captured by WADD with free subjective weights.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Mixture of Subjective WADD and Tallying (Advocated) from a pure WADD model with subjective weights and objective validities (Competing). The Advocated theory generates choice probabilities by mixing two distinct softmax distributions: one based on unweighted feature sums (Tallying) and one based on subjectively weighted sums (Subjective WADD). The Competing theory uses a single softmax over sums weighted by the product of objective validities and subjective weights. By including a spectrum of trials\u2014where Tallying and WADD strongly disagree, where one ties while the other has a strict preference, and where they agree\u2014the mixture model predicts a non-linear blending of choice probabilities across conditions. A single softmax model cannot replicate this pattern because it must map all trade-offs onto a single linear scale, preventing it from simultaneously fitting the 'flat' probabilities on Tallying ties and the sharp preferences on conflict trials without contradictory weight assignments.",
  "validities": [
    1.0,
    0.9,
    0.7,
    0.6,
    0.5
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
      1,
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
      0
    ],
    [
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
      1,
      0
    ],
    [
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
      0,
      1
    ],
    [
      0,
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
      0
    ],
    [
      1,
      1,
      0,
      1,
      0
    ],
    [
      0,
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
      0
    ]
  ],
  "trial_b_ratings": [
    [
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
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
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
      0
    ],
    [
      1,
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
      0
    ],
    [
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
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1
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
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3555,
  "candidates_token_count": 669,
  "total_token_count": 7063
}
```
