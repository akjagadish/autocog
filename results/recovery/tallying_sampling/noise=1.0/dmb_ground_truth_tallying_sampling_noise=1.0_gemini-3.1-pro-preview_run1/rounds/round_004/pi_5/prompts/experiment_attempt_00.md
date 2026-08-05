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


## COMPETING THEORY
**Description:** Biased Logistic Tallying: Subjects make decisions by counting the total number of positive features for each option (Tallying), ignoring cue validities. However, their choices are highly noisy and subject to a baseline spatial/option bias. The decision process is modeled as a logistic function over the difference in feature tallies between the two options, parameterized by a sensitivity weight and a baseline bias. The sensitivity to tally differences is weak, reflecting empirical data where feature counts only slightly sway noisy guessing.

**Parameters:**
- beta: [-0.2, 0.2]
- bias: [-2.0, 2.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters['beta'])
    bias = float(parameters['bias'])
    
    # Logistic choice rule based on tally difference and baseline bias
    diff = scores[0] - scores[1]
    logit = beta * diff + bias
    
    # Numerically stable sigmoid
    if logit >= 0:
        p_a = 1.0 / (1.0 + np.exp(-logit))
    else:
        exp_logit = np.exp(logit)
        p_a = exp_logit / (1.0 + exp_logit)
        
    return np.array([p_a, 1.0 - p_a])
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

[6] TTB predicts that choices will be entirely determined by the highest-validity discriminating cue, resulting in large shifts in choice probability depending on whether Option A or B possesses this cue. The competing theory assumes subjects ignore all features and simply guess with a fixed bias, predicting a constant choice rate for Option A across all trials. By systematically varying which option has the top discriminating cue (and tying higher-validity cues to shift the discriminator down the hierarchy), we can observe whether P(A) swings dramatically between near 0 and near 1 (supporting TTB) or remains flat (supporting the random guessing model).

[7] To quantitatively dissociate the Random Guessing (bias) model from Take The Best (TTB), this design uses exact mirror-image stimulus pairs (swapping the features of Option A and Option B). TTB predicts that the choice probability for Option A will perfectly invert when the features are swapped (e.g., shifting from near 1.0 to near 0.0), as choices are driven entirely by the cues. In contrast, the Random Guessing model posits that subjects ignore the cues and rely on a fixed positional or baseline bias for Option A. Therefore, it predicts that the choice probability for Option A will remain constant across the mirror-image pairs, completely invariant to which option possesses the higher-validity cues.

[8] This design quantitatively dissociates Biased Logistic Tallying from Random Guessing by systematically varying the difference in the total number of positive features (tallies) between Option A and Option B from -8 to +8. The Biased Logistic Tallying model predicts a graded, parametric shift in the probability of choosing Option A as a function of this tally difference. In contrast, the Random Guessing model posits that subjects completely ignore the features and rely on a fixed baseline bias, predicting a perfectly flat choice probability across all trials regardless of how many features support either option.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Random Guessing from Biased Logistic Tallying, this design orthogonally manipulates the Tally difference (from -5 to +5) and the prediction of Take The Best (by flipping the most valid cue). Biased Logistic Tallying predicts a graded choice probability driven by the tally difference, while TTB predicts a step function based on the top cue. The advocated Random Guessing theory predicts that subjects ignore both feature counts and validities, resulting in a completely flat choice probability across all trials regardless of the strong compensatory or lexicographic signals.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.6,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
    [
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
      1,
      0,
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
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
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
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
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
      1,
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
      0
    ],
    [
      0,
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
      1,
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
      0,
      0
    ],
    [
      1,
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3258,
  "candidates_token_count": 758,
  "total_token_count": 8318
}
```
