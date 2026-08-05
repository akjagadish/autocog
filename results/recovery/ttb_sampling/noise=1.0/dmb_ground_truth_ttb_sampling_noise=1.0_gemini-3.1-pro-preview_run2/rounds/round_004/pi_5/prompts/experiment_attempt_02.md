# experiment_attempt_02

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
**Description:** Random Choice / Minimal Effort: In the absence of trial-by-trial feedback and when faced with complex multi-attribute binary arrays, participants largely abandon systematic cue-based strategies. Instead, they resort to minimal effort processing, which manifests as random guessing or behavior dominated by an extremely high lapse rate. Any residual systematicity is extremely weak, resulting in choice probabilities that are consistently very close to 0.5 across all experimental conditions.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.9, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Minimal effort evaluation (e.g., simple tallying of 1s)
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Dominated by a extremely high lapse rate (epsilon near 1.0)
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
**Description:** Weak Tallying with Position Bias: In the absence of trial-by-trial feedback, participants abandon complex, validity-based compensatory or non-compensatory strategies. Instead, their choices are heavily dominated by random guessing (a very high lapse rate). The tiny fraction of systematic variance that remains is driven by minimal-effort heuristics: a weak preference for the option with a higher total number of positive features (tallying), combined with a slight positional bias that might favor or disfavor features based on reading order.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- position_bias: [-1.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    n_features = stim.shape[1]
    position_bias = float(parameters["position_bias"])
    
    # Features on the left (index 0) get higher weights if position_bias > 0
    # Weights range from 1.0 (rightmost) to 1.0 + position_bias * (n_features - 1) (leftmost)
    w = 1.0 + position_bias * np.arange(n_features - 1, -1, -1)
    
    # Calculate score for each option
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Dominated by a extremely high lapse rate (epsilon near 1.0)
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
[0] This design aims to dissociate Tallying (which counts the number of features favoring each option, ignoring validities and magnitudes) from Weighted Additive (WADD) (which computes the weighted sum of feature values). By using five features with a steep validity drop-off, we can create pairs where a few high-validity features outweigh several lower-validity features (WADD preference) versus pairs where the sheer number of lower-validity features dictates the choice (Tallying preference). For instance, in Trial 1, Option A has the two best features (WADD wins) while Option B has the three worst features (Tallying wins).

[1] This design quantitatively dissociates Weighted Additive (WADD) from Tallying using 5 features. The validities are chosen such that the sum of the top two validities (0.9 + 0.8 = 1.7) is slightly greater than the sum of the bottom three validities (0.6 + 0.55 + 0.5 = 1.65). This enables critical trials where Option A has the top two features and Option B has the bottom three: WADD prefers Option A (higher weighted sum), whereas Tallying prefers Option B (higher count of winning features, 3 vs 2). Additional trials include cases where Tallying predicts a tie (equal number of winning features) but WADD strongly prefers one option due to validity differences, as well as baseline trials where both models agree.

[2] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) while distinguishing from previously explored designs, this experiment uses a 6-feature environment with a distinct cluster of high validities and a cluster of low validities. By making the top three validities very close (0.90, 0.88, 0.86), TTB's subjective cue hierarchy (which includes noise) will frequently swap their order, whereas WADD integrates them stably. We design trials where the highest-validity feature points to one option, but a combination of the slightly-lower high-validity features and multiple low-validity features points to the other. This setup not only tests compensatory vs. non-compensatory choice but also probes the stochastic nature of TTB's cue hierarchy.

[3] This design quantitatively dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic by pitting a single highly valid feature against multiple moderately valid features. We use five features where the highest validity is 0.95 and the others range from 0.80 to 0.65. In critical trials, Option B excels on the single most predictive feature, while Option A is supported by a combination of the remaining features. Because the sum of the lower validities far exceeds the highest validity, WADD strongly prefers Option A. In contrast, TTB (even with validity noise) will primarily rely on the highest-validity feature and thus prefer Option B. By varying which subset of lower-validity features Option A possesses, we can map out the precise compensatory threshold predicted by WADD but absent in TTB.

[4] To quantitatively dissociate Take The Best (TTB) from Take The First (Left-to-Right Lexicographic), we arrange the feature validities in ascending order from left to right. The leftmost feature has the lowest validity, while the rightmost feature has the highest validity. Take The First strictly evaluates features from left to right, thus heavily relying on the lowest-validity features if they discriminate. In contrast, Take The Best reorders features by their subjective validities (with noise) and will predominantly evaluate the rightmost, highest-validity feature first. By pitting the leftmost features against the rightmost features, the two models make completely opposite predictions.

[5] To quantitatively dissociate Take The First (Left-to-Right Lexicographic) from Take The Best (TTB), we place the highest validity features in the middle of the array, with lower validities on the left and right edges. Take The First strictly evaluates features from left to right, meaning it will base its decisions on the leftmost features (indices 0 and 1) whenever they discriminate, regardless of their low validity. Take The Best, however, sorts features by their subjective validities and will jump straight to the middle features (indices 2 and 3). By designing trials where the leftmost features favor one option and the middle features favor the other, we can cleanly separate the two models. This avoids the confound of a strictly ascending validity array where left-to-right could be mistaken for reverse-validity search.

[6] To dissociate Take The Best (TTB) from the Random Choice / Minimal Effort model, we employ a 5-feature design with a steep validity hierarchy (0.95 to 0.55). The critical trials pit the single most valid feature against all remaining features. TTB predicts a strong preference for the option with the highest-validity feature, as it sequentially evaluates and stops at the first discriminating cue. In contrast, the Random Choice model predicts choices that are either perfectly random (due to a near 1.0 lapse rate) or slightly leaning toward the option with more positive features (due to minimal effort tallying). The stark difference in predicted choice probabilities (strong systematic preference vs. ~0.5) will clearly adjudicate between the models.

[7] To cleanly dissociate the Random Choice / Minimal Effort theory from Take The Best (TTB), we use a 5-feature design with a clear validity hierarchy. TTB predicts strong, systematic preferences based on the most valid discriminating cue, regardless of the sheer number of cues favoring the other option. In contrast, the Random Choice theory predicts that participants will exhibit a massive lapse rate and rely on minimal effort tallying, leading to choice probabilities near 0.5 across all trials. By pitting one or two high-validity cues against multiple lower-validity cues, and including trials where the number of positive cues is equal but validities differ, we can expose the lack of systematic cue-based processing if the Random Choice theory holds true.

[8] This design quantitatively dissociates Weak Tallying with Position Bias from the Random Choice / Minimal Effort theory by manipulating the spatial position of positive features while controlling for their sum. The competing theory relies purely on simple tallying, predicting perfect 50/50 ties whenever both options have the same number of positive features, regardless of where they appear. In contrast, the advocated theory incorporates a position bias (e.g., favoring left-most features due to reading order), predicting systematic deviations from 50/50 in these 'tied' tallying trials. Furthermore, we include a trial where one option has fewer positive features but located on the extreme left, testing if position bias can override a simple tallying advantage.

[9] To quantitatively dissociate the Random Choice / Minimal Effort theory (advocated) from the Weak Tallying with Position Bias theory (competing), this design manipulates the spatial position of positive features while controlling for their sum. The advocated theory posits that any residual systematicity is purely based on simple tallying, predicting perfect 50/50 ties whenever both options have the same number of positive features, regardless of where they appear. In contrast, the competing theory incorporates a position bias (e.g., favoring left-most features), predicting systematic deviations from 50/50 in these 'tied' tallying trials. Furthermore, we include trials where one option has fewer positive features but located on the extreme left, testing if position bias can override a simple tallying advantage.

[10] To quantitatively dissociate the Random Choice / Minimal Effort theory (advocated) from the Weak Tallying with Position Bias theory (competing), this design focuses on spatial feature placement while controlling feature sums. The advocated theory assumes decisions are driven purely by simple tallying (with a very high lapse rate), predicting exactly 50/50 choices whenever both options have the same number of positive features, regardless of their left-to-right position. The competing theory, however, posits a position bias where features on the left are weighted differently than features on the right. By including trials where the tally is tied but the positions of the positive features are completely separated (e.g., all left vs. all right), the competing theory predicts systematic deviations from 50/50, whereas the advocated theory predicts perfect indifference. We also include trials pitting a tally advantage against a positional advantage to see if position bias can overcome a simple sum difference.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Random Choice / Minimal Effort theory (advocated) from the Weak Tallying with Position Bias theory (competing), this design utilizes a 6-feature array to amplify potential positional effects. The advocated theory asserts that any residual systematicity is purely based on simple tallying, predicting perfect 50/50 choices whenever both options have the same number of positive features, entirely independent of their left-to-right position. The competing theory, however, posits a position bias where features on the left are weighted more (or less) than features on the right. By including trials where the tally is tied but the positions of the positive features are completely separated (e.g., extreme left vs. extreme right), the competing theory predicts systematic deviations from 50/50, whereas the advocated theory predicts perfect indifference. We also pit a tally advantage against a positional advantage to test if position bias can overcome a simple sum difference.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
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
      0
    ]
  ],
  "trial_b_ratings": [
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
      0,
      0,
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
      0,
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
  "prompt_token_count": 3908,
  "candidates_token_count": 440,
  "total_token_count": 5759
}
```
