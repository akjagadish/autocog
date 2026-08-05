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
**Description:** Zero-Intelligence / Random Choice Theory: In this specific experimental paradigm, participants do not systematically integrate feature validities or use non-compensatory heuristics. Due to task disengagement, cognitive overload, or the lack of trial-by-trial feedback, participants do not learn or apply any complex decision rules. Instead, they simply guess randomly between the two options on every trial, resulting in a 50/50 probability for any pair of options regardless of their features.

**Parameters:**
(none)

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The Zero-Intelligence model completely ignores the state and history,
    # assuming participants guess uniformly at random on every trial.
    return np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Position Bias / Option A Default Theory: Participants do not systematically integrate feature validities or use complex heuristics. Instead, due to reading order (e.g., left-to-right or top-to-bottom) or motor ease, they exhibit a slight systematic bias toward choosing the first option presented (Option A). This structural bias is mixed with random guessing, leading to a state-independent choice probability where Option A is chosen slightly more often than Option B, regardless of the features of the options.

**Parameters:**
- p_A: [0.5, 0.6]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the state and history, relying solely on a structural bias toward Option A.
    p_A = float(parameters["p_A"])
    return np.array([p_A, 1.0 - p_A])
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
[0] This design aims to strongly dissociate Take The Best (TTB) from Tallying by pitting the highest-validity discriminating cue against the sheer number of winning cues. In most trials, one option wins on a single high-validity cue, while the other option wins on multiple lower-validity cues. TTB will consistently choose the option endorsed by the single best cue, whereas Tallying will choose the option with more total winning features.

[1] This design strictly dissociates Tallying from Take The Best (TTB) by pitting the single most valid discriminating cue against the sheer number of winning cues. In each trial, one option is endorsed by the highest-validity cue that discriminates the pair, meaning TTB will invariably choose it. However, the other option is endorsed by a larger number of lower-validity cues, meaning Tallying will consistently prefer the alternative. By varying whether the highest validity cue is the 1st or 2nd cue overall, and varying the number of opposing cues, we can rule out alternative heuristics like Equal-Weight or WADD if they happen to correlate with either in specific subsets.

[2] This design strongly dissociates Take The Best (TTB) from the heavily stochastic Weighted Additive (WADD) model. In every trial, Option A is endorsed by the single most valid discriminating cue, meaning TTB will consistently and deterministically (due to its high allowed beta) choose Option A. Conversely, Option B is supported by a larger number of lower-validity cues such that its weighted sum is strictly greater than Option A's. The competing WADD theory is constrained to a very low inverse temperature (beta <= 0.5), meaning it predicts either a weak preference for Option B or near-random guessing, but it can never predict a systematic preference for Option A. Observing a strong preference for Option A will therefore uniquely support TTB.

[3] This design aims to demonstrate that choices are driven by the weighted sum of all features (WADD) rather than a non-compensatory, one-reason rule (TTB), and that decision behavior is inherently noisy rather than deterministic. Across all trials, Option A is endorsed by the single most valid discriminating cue, meaning TTB will consistently predict Option A (and can do so with near certainty given its unconstrained beta). However, the number and validity of cues supporting Option B vary parametrically. In some trials, Option B's lower-validity cues sum to a much higher total than Option A's single cue, leading WADD to predict a preference for Option B. In other trials, Option A's weighted sum is higher. Because WADD is constrained to low beta values in this formulation, it predicts that choices will be highly stochastic but will linearly track the difference in weighted sums, occasionally flipping to favor Option B. If subjects show noisy choices that shift toward Option B when it has many lower-validity cues, it will strongly support WADD over TTB.

[4] This design tests the Zero-Intelligence (Random Choice) theory against the heavily stochastic Weighted Additive (WADD) theory. By parametrically varying the difference in the weighted sums of features between Option A and Option B from extreme to near-zero, WADD predicts a graded, sigmoid-like shift in choice probabilities (even with its highly constrained, low beta). In contrast, the Zero-Intelligence theory predicts a completely flat choice probability of 50% across all trials, regardless of how overwhelmingly one option dominates the other in feature value. A lack of sensitivity to these massive feature differences would strongly dissociate Zero-Intelligence from WADD.

[5] This design tests the heavily stochastic Weighted Additive (WADD) theory against the Zero-Intelligence (Random Choice) theory by systematically varying the difference in weighted sums between Option A and Option B across a wide range. While WADD is constrained to low beta values (implying high stochasticity), it still predicts that choice probabilities will monotonically track the difference in weighted feature sums. In contrast, the Zero-Intelligence theory predicts a completely flat 50/50 choice probability across all pairs regardless of how heavily one option dominates the other. By including pairs with extreme differences as well as very subtle differences, we can observe whether choices exhibit the graded, sigmoid-like sensitivity predicted by WADD or the complete insensitivity predicted by Random Choice.

[6] This design tests the Zero-Intelligence (Random Choice) theory against the Single-Cue Heuristic with Very High Lapse. By creating trials where one option clearly dominates on the single most valid cue while losing on all other cues, the Single-Cue theory (even with up to 90% lapse rate) predicts a small but systematic bias (up to 55% choice probability) toward the option endorsed by the best cue. In contrast, the Zero-Intelligence theory predicts exactly 50% choice probability across all trials, completely insensitive to the best cue.

[7] This design isolates the Single-Cue Heuristic with Very High Lapse from Zero-Intelligence by systematically varying the single most valid cue against all other cues. While Zero-Intelligence predicts exactly 50% choice probability across all trials, the Single-Cue theory predicts a small but systematic deviation (up to 55%) strictly aligned with the single most valid cue, completely ignoring the remaining cues. By including trials where the best cue ties but all other cues strongly favor one option, we can further demonstrate that the residual bias is exclusively driven by the highest-validity cue and not by a weak compensatory integration.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design aims to dissociate the Zero-Intelligence theory from the Position Bias theory. Because both theories assume participants ignore the feature values entirely, the specific feature patterns do not change their predictions. However, by including trials where Option B overwhelmingly dominates Option A (and vice versa), we can verify that participants are indeed ignoring the features. The Zero-Intelligence theory predicts exactly 50/50 guessing across all trials, whereas the Position Bias theory predicts a systematic, state-independent preference for Option A (up to 60%). If participants choose Option A exactly 50% of the time even when Option B is objectively superior on all features, it will support Zero-Intelligence and rule out Position Bias.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1
    ],
    [
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
  "prompt_token_count": 2730,
  "candidates_token_count": 435,
  "total_token_count": 4082
}
```
