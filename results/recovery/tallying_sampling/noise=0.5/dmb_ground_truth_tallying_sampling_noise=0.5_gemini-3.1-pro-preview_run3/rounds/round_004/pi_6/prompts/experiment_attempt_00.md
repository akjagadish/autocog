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
**Description:** Mixture of Take-The-Best (TTB) and Tallying: Decision-makers probabilistically mix between two cognitively simple heuristics. Take-The-Best (TTB) is a strict lexicographic rule that inspects cues in order of their validity and chooses based entirely on the first discriminating cue. Tallying is an equal-weight rule that simply counts the number of positive features for each option. The mixture is biased towards Tallying to account for the strong equal-weighting pull observed in human data, while still allowing TTB to explain some non-compensatory choices. A small uniform lapse rate accounts for general response errors.

**Parameters:**
- beta: [0.01, 10.0]
- epsilon: [0.0, 0.3]
- w_tally: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) ---
    # Order cues by validity (descending)
    cue_order = np.argsort(val)[::-1]
    diff = stim[0] - stim[1]
    
    p_ttb = np.array([0.5, 0.5])
    for cue in cue_order:
        if diff[cue] > 0:
            p_ttb = np.array([1.0, 0.0])
            break
        elif diff[cue] < 0:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD), this design employs a 'compensatory' structure across four features with varying validities. TTB decides solely on the highest-validity discriminating cue, ignoring all other cues. WADD, conversely, integrates all cues weighted by their validities. The trials are constructed so that one option always wins on the single most valid discriminating cue (favored by TTB), while the other option has a larger sum of validities across multiple less valid cues (favored by WADD). Baseline trials where both heuristics agree are also included.

[1] To quantitatively dissociate WADD from TTB, we use five features with a linear spread of validities. The trials are designed as 'compensatory' choices where the single best discriminating cue (which strictly dictates the TTB choice) is pitted against a coalition of multiple lower-validity cues (which dictate the WADD choice). For instance, in several pairs, one option has a '1' on the highest available discriminating validity but '0's everywhere else, while the other option has '0' on that highest cue but '1's on multiple remaining cues, making the weighted sum larger for the latter. This creates a strong divergence in predicted choice probabilities between the two models.

[2] To quantitatively dissociate Tallying from WADD, this design leverages five features with a skewed distribution of validities (two high, three low). Tallying ignores validities and simply counts the number of positive features, while WADD computes a validity-weighted sum. The trial set includes 'conflict' pairs where one option has fewer positive features but those features have much higher validities (favored by WADD), while the other option has more positive features of lower validities (favored by Tallying). We also include 'tie' pairs where both options have the same number of positive features (Tallying predicts guessing) but differ in validity (WADD strongly prefers one), as well as 'agreement' pairs where both heuristics favor the same option. This design maximizes the divergence in predicted choice probabilities.

[3] To quantitatively dissociate WADD from Tallying, this design uses 5 features with a split in validities (two high, three low). Tallying ignores validities and simply counts the number of positive features, while WADD computes a validity-weighted sum. The trials are constructed to create critical dissociations: (1) Tallying predicts a tie (equal number of positive features) while WADD predicts a strong preference for the option with high-validity features, and (2) Tallying predicts a preference for one option (more positive features) while WADD predicts a preference for the other (fewer but higher-validity features). We also include agreement trials to establish baseline choice behavior.

[4] To quantitatively dissociate the Mixture of Heuristics (Tallying + WADD) from a pure WADD model with subjective weights, we use a 5-feature design with a clear split between high-validity and low-validity features. The trials are constructed to include 'conflict' cases where WADD and Tallying strongly disagree (e.g., fewer high-validity features vs. more low-validity features), 'tie' cases where Tallying predicts guessing but WADD has a strict preference, and 'agreement' cases. Because the Mixture model computes a convex combination of two distinct choice probability distributions (one based on unweighted sums, one on weighted sums), it generates a non-linear profile of choice probabilities across these trial types. A pure WADD model, which uses a single softmax over a single set of linearly weighted sums, cannot simultaneously capture the flat 'tie' behavior of Tallying and the sharp 'conflict' preferences without its subjective weights being pulled in contradictory directions.

[5] To quantitatively dissociate a pure Weighted Additive (WADD) model with subjective weights from a Mixture of Heuristics (Tallying + objective WADD), this design uses 5 features with a linear spread of validities. The trials comprehensively pit individual higher-validity features against combinations of lower-validity features. The Mixture model is restricted to a convex combination of equal weights (Tallying) and strictly objective validities (WADD). In contrast, the advocated pure WADD model allows for subjective weighting, meaning it can capture non-linear transformations of the validities (e.g., extreme overweighting of the most valid cue, resembling Take-The-Best, or idiosyncratic trade-offs between middle-tier cues) that cannot be expressed as any mixture of objective WADD and Tallying. By spanning diverse trade-offs across the validity hierarchy, we can identify choice probability profiles that fall outside the rigid manifold of the Mixture model but are perfectly captured by WADD with free subjective weights.

[6] To quantitatively dissociate the Mixture of Subjective WADD and Tallying (Advocated) from a pure WADD model with subjective weights and objective validities (Competing). The Advocated theory generates choice probabilities by mixing two distinct softmax distributions: one based on unweighted feature sums (Tallying) and one based on subjectively weighted sums (Subjective WADD). The Competing theory uses a single softmax over sums weighted by the product of objective validities and subjective weights. By including a spectrum of trials—where Tallying and WADD strongly disagree, where one ties while the other has a strict preference, and where they agree—the mixture model predicts a non-linear blending of choice probabilities across conditions. A single softmax model cannot replicate this pattern because it must map all trade-offs onto a single linear scale, preventing it from simultaneously fitting the 'flat' probabilities on Tallying ties and the sharp preferences on conflict trials without contradictory weight assignments.

[7] The Advocated theory (WADD) and Competing theory (Mixture of Subjective WADD and Tallying) structurally overlap, but they are subject to different parameter bounds that enforce distinct behavioral predictions. Crucially, the Advocated model allows for much higher inverse temperature (beta up to 20) compared to the Competing model (beta up to 5). By employing a fine-grained hierarchy of 6 features with linearly decreasing validities, we can test choices between adjacent features (e.g., Feature 1 vs. 2, Feature 2 vs. 3, etc.). The Advocated model can predict consistently strong, deterministic preferences (>85%) across all adjacent feature comparisons because its high beta can magnify small differences in objective validities. To mimic this consistent determinism, the Competing model would need its subjective weights to drop dramatically at each step in the hierarchy. However, because its beta is capped at 5, maintaining an 85% choice probability requires a subjective weight difference of at least 0.4 between adjacent features. Across 6 features, this would require a total weight drop of 2.0, which is impossible since subjective weights are bounded in [0, 1]. Thus, the Competing model is forced to predict highly noisy choices for fine-grained trade-offs, while the Advocated model uniquely predicts sharp, consistent preferences.

[8] To quantitatively dissociate the Mixture of Subjective WADD and Tallying (Advocated) from the Mixture of TTB and Tallying (Competing), this design uses 5 features with a linear spread of validities. The Competing theory restricts non-compensatory choices to a strict lexicographic rule (Take-The-Best) governed by objective validities, mixed with equal-weighting (Tallying). In contrast, the Advocated theory's WADD component uses free subjective weights, allowing it to capture compensatory choices or non-standard lexicographic orders that deviate from the objective validities. By including trials where Tallying predicts a tie but TTB rigidly prefers one option, and trials where Tallying favors one option while TTB strictly favors the other, we create a space of choices where human data often shows graded, compensatory trade-offs. The Competing model is forced to predict either strict TTB choices, Tallying choices, or a rigid linear mixture of the two, while the Advocated model can flexibly accommodate idiosyncratic weighting profiles through its subjective WADD component.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates the Advocated theory (TTB + Tallying) from the Competing theory (Subjective WADD + Tallying) by exploiting the step-function nature of Take-The-Best versus the continuous linear nature of WADD. In pairs 1-4, Tallying is tied (1 positive feature each), and TTB favors Option A based on adjacent cues (Cue 1 vs 2, Cue 2 vs 3, etc.). Because TTB makes a discrete 100% decision as long as it finds a discriminating cue, the Advocated theory predicts the exact same choice probability for Option A across all these pairs. For the Competing theory to mimic this constant probability, its subjective weights must form a strict arithmetic progression (w1 - w2 = w2 - w3 = ...). However, if the Competing theory's weights are linear, it must predict exactly 50% (complete guessing) on symmetrically balanced pairs (like Pair 5 and 6, where the sum of indices is equal). The Advocated theory, meanwhile, continues to predict the same strong preference for Option A on these balanced pairs because TTB strictly breaks the tie using the highest validity cue. This creates a geometric impossibility for the Competing theory to simultaneously fit the adjacent-cue trials and the balanced-cue trials.",
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
    ],
    [
      0,
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
      0,
      1,
      0,
      0,
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
      0,
      0
    ],
    [
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
    ],
    [
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
      1,
      1
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
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4113,
  "candidates_token_count": 698,
  "total_token_count": 10005
}
```
