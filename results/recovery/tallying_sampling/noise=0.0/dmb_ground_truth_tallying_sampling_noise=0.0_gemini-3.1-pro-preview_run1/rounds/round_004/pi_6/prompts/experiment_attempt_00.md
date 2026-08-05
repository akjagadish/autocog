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
**Description:** Weighted Additive (WADD) Model. Decision-makers compute a weighted sum of the positive features for each option, using the provided cue validities as weights. They then choose the option with the higher weighted sum. This serves as a strong compensatory, rational baseline to rigorously test whether participants are partially or fully integrating validities, as opposed to ignoring them (Tallying) or using a non-compensatory rule (Take The Best).

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # WADD: sum of validities for positive features
    scores = stim @ validities
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People make decisions using a 'Tallying' or 'Equal Weight' heuristic. Instead of weighting features by their validity or importance, decision-makers simply count the total number of positive features for each option. They choose the option with the highest tally, effectively ignoring the provided cue validities. This explains why participants consistently choose options with more positive cues even when a single higher-validity cue points in the opposite direction.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: sum the positive features for each option, ignoring validities.
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from the Weighted Additive rule (WADD), we design a 4-feature environment where the highest-validity cue frequently contradicts the aggregated sum of the lower-validity cues. TTB makes decisions based solely on the single most valid discriminating cue, ignoring the rest. WADD integrates all features, allowing multiple weaker cues to compensate for a deficit on the most valid cue. By setting validities to [0.95, 0.65, 0.60, 0.55], we create strong compensatory trials (e.g., A has the best cue but B has the remaining three) where TTB and WADD yield diametrically opposed predictions.

[1] To quantitatively dissociate the Weighted Additive (WADD) rule from Take The Best (TTB), we use a 5-feature environment where the compensatory nature of WADD can be systematically pitted against the non-compensatory stopping rule of TTB at different levels of the cue hierarchy. By setting validities such that the most valid cue is outweighed by the sum of the remaining cues, we create trials where TTB stops early and chooses the option with the highest validity cue, while WADD integrates all features and chooses the opposite. We include trials where the conflict happens at the first, second, and third most valid cues to ensure the dissociation is robust across the cue cascade.

[2] To quantitatively dissociate Tallying (Equal Weight) from the Weighted Additive (WADD) rule, we use a 5-feature environment with a steep drop-off in cue validities: two highly valid cues (0.95, 0.90) and three lower-validity cues (0.60, 0.55, 0.50). Tallying ignores validities and simply counts the number of positive features. WADD weights each feature by its validity. By pitting an option with fewer, but highly valid cues (e.g., the top two cues) against an option with more, but lower-validity cues (e.g., the bottom three cues), we create strong dissociable trials. For example, an option with the top 2 cues (Tally = 2, WADD = 1.85) will be chosen by WADD over an option with the bottom 3 cues (Tally = 3, WADD = 1.65), whereas Tallying predicts the exact opposite. We include a mix of such dissociation trials, agreement trials (where the higher tally also has the higher weighted sum), and tally-tie trials (where Tallying is indifferent but WADD has a strict preference) to thoroughly map the strategy space.

[3] To quantitatively dissociate the Weighted Additive (WADD) rule from Tallying (Equal Weight) without repeating prior 5-feature designs, we introduce a 7-feature environment with a gradual linear decline in validities. Tallying simply counts the number of positive features, completely ignoring their validities, whereas WADD computes a validity-weighted sum. We construct trials where an option with fewer but higher-validity features is pitted against an option with more but lower-validity features. By tuning the number of features on each side, we create combinations where WADD strongly prefers the option with fewer features, trials where WADD is indifferent but Tallying has a strict preference, and agreement trials where both models align. This 7-feature structure provides a wider dynamic range of sums to distinguish between the models.

[4] Since Tallying and Majority of Confirming Dimensions (MCD) are mathematically identical in their choice probability predictions for binary features (the difference in total sums always equals the difference in pairwise wins), a strict choice-probability dissociation is impossible in this paradigm. However, to conceptually probe the difference between absolute independent evaluation (Tallying) and relative pairwise comparison (MCD), we design a 6-feature environment. We systematically manipulate the number of shared positive features (where both A and B equal 1) and shared negative features (where both A and B equal 0) while keeping the difference in sums constant. If subjects use Tallying, cognitive load or response times might scale with the total number of '1's. If they use MCD, load might scale with the number of discriminating features (where A != B), ignoring shared features. The validities are set to a linear spread to maintain the cover story, though both target heuristics ignore them.

[5] Because Tallying and Majority of Confirming Dimensions (MCD) produce mathematically identical choice probabilities for binary features (the difference in total sums exactly equals the difference in pairwise wins), a pure choice-probability dissociation is impossible. To distinguish them, we rely on process signatures such as response times or cognitive load. Building on prior findings, we use a 10-feature environment to maximize the variance in shared features. We systematically manipulate the number of shared positive cues (where both A and B equal 1) and shared negative cues (where both A and B equal 0) while keeping the net difference in scores constant. If decision-makers use Tallying, processing effort should scale with the total number of '1's present across both options. If they use MCD, processing effort should scale only with the number of discriminating features (where A != B), rendering shared features irrelevant.

[6] To quantitatively dissociate Tallying from Take-The-Best (TTB), we pit a single highly valid cue against a larger quantity of less valid cues. Tallying predicts participants will choose the option with the greater total number of positive features, entirely disregarding validities. Conversely, TTB predicts participants will stop searching at the first discriminating cue (ordered by validity) and choose the option favored by that cue, ignoring the sheer quantity of lower-validity cues. We include trials where the highest validity cue directly contradicts the sum of the remaining cues, trials where the conflict occurs lower in the cue hierarchy, and baseline agreement trials.

[7] To quantitatively dissociate Take-The-Best (TTB) from Tallying, we design a 5-feature environment where validities are mixed in order to verify that TTB correctly sorts by validity while Tallying ignores validities entirely. The highest validity cue is placed at index 1 (0.90). We construct trials where the single most valid cue favors one option, but the sheer quantity of lower-validity cues favors the other. TTB predicts a choice based on the first discriminating cue in descending order of validity, whereas Tallying predicts a choice based on the total sum of positive features. We include trials where the conflict occurs at the top cue, trials where the conflict occurs lower in the cue hierarchy (after ties on the most valid cues), trials where Tallying is completely indifferent but TTB has a strict preference, and agreement trials to serve as a baseline.

[8] To quantitatively dissociate Tallying (Equal Weight) from the Weighted Additive (WADD) rule, we design a 6-feature environment with a carefully constructed validity distribution: [0.95, 0.85, 0.80, 0.60, 0.55, 0.50]. This specific set of validities allows us to create a unique 'WADD tie' condition where the sum of validities for two features (0.85 + 0.80 = 1.65) exactly equals the sum of validities for three features (0.60 + 0.55 + 0.50 = 1.65). In this trial, WADD is completely indifferent, whereas Tallying shows a strict preference for the option with three features. We also include strong dissociation trials where an option with 4 lower-validity features is pitted against an option with 3 higher-validity features, causing diametrically opposed predictions between the two models. These conditions, alongside Tally-tie trials where WADD has a strict preference, provide a comprehensive map of the strategy space without relying on the structures of previously explored 5-feature or 7-feature designs.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Tallying (Equal Weight) from the Weighted Additive (WADD) model, we introduce an 8-feature environment with a sharp step-down in cue validities: three highly valid cues (0.99, 0.95, 0.90) and five low-validity cues (all 0.51). This structure provides enough features to create stark contrasts between the quantity of positive cues and their weighted sum. We construct strong dissociation trials where WADD chooses an option with fewer but highly valid cues (e.g., 3 cues summing to 2.84), while Tallying chooses the alternative with more but less valid cues (e.g., 5 cues summing to 2.55). We also include 'Tally tie' trials where both options have the same number of positive features but different validity sums, forcing WADD to express a strict preference while Tallying is indifferent.",
  "validities": [
    0.99,
    0.95,
    0.9,
    0.51,
    0.51,
    0.51,
    0.51,
    0.51
  ],
  "trial_a_ratings": [
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
      0
    ],
    [
      1,
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
      1,
      1,
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
      0
    ],
    [
      0,
      1,
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
      0,
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
  "prompt_token_count": 3653,
  "candidates_token_count": 715,
  "total_token_count": 7329
}
```
