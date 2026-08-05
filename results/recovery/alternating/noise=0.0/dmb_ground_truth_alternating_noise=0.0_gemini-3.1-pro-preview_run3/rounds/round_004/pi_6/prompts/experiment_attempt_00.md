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
**Description:** Strict Alternation: Subjects in these experimental paradigms ignore all feature information and simply alternate their choices between Option A and Option B deterministically on every trial. This response heuristic reflects a complete lack of engagement with the stimulus validities, substituting a simple motor-alternation pattern.

**Parameters:**
- dummy: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # Reference the dummy parameter to satisfy the parameter usage constraint
    _ = float(parameters["dummy"])
    
    # If first trial, guess randomly to start the sequence
    if len(history["response"]) == 0:
        return np.array([0.5, 0.5])
        
    # Otherwise, strictly alternate from the last response
    last_response = history["response"][-1]
    if last_response == 0:
        return np.array([0.0, 1.0])
    else:
        return np.array([1.0, 0.0])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Response Heuristic / Spatial Bias: Subjects in these paradigms often ignore feature information entirely and instead rely on presentation-based response heuristics. Decision-making is driven by a persistent spatial bias (e.g., a preference for Option A over Option B) combined with a sequential dependence, such as a tendency to repeat the previous choice or alternate between options. This produces behavior that appears random with respect to cue validities but contains structured autocorrelation and spatial preference.

**Parameters:**
- beta_bias: [-3.0, 3.0]
- beta_repeat: [-3.0, 3.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    beta_bias = float(parameters["beta_bias"])
    beta_repeat = float(parameters["beta_repeat"])
    
    # Determine the indicator for the previous choice
    # 1.0 if Option A was chosen, -1.0 if Option B was chosen, 0.0 if first trial
    if len(history["response"]) == 0:
        last_a = 0.0
    else:
        last_a = 1.0 if history["response"][-1] == 0 else -1.0
        
    # Logit for choosing Option A
    z = beta_bias + beta_repeat * last_a
    
    # Convert to probability using sigmoid
    p_a = 1.0 / (1.0 + np.exp(-z))
    
    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design specifically pits Take The Best (TTB) against Tallying. TTB relies solely on the highest-validity cue that discriminates between the two options, completely ignoring the sheer number of cues favoring each option. In contrast, Tallying counts the total number of features favoring each option, ignoring cue validities. The trials are constructed so that one option always wins on the highest-validity discriminating cue (favored by TTB), while the other option wins on a greater number of lower-validity cues (favored by Tallying). This provides a strong quantitative dissociation between the two models.

[1] This design quantitatively dissociates Tallying from Take The Best (TTB) by exploiting how each model generates choice probabilities. TTB is a 'one-reason' decision rule: once the highest-validity cue discriminates between options, it stops searching. Consequently, TTB's predicted choice probability depends entirely on a fixed binary score difference (1 vs 0) and remains constant across all trials where the first cue discriminates. In contrast, Tallying counts the total number of features favoring each option, and its choice probability scales with the *magnitude* of the difference in tallies. By holding the highest-validity cue constant (always discriminating in favor of one option) while systematically varying the tally difference on the remaining cues (from strongly opposing the first cue to strongly supporting it), Tallying predicts a graded, monotonic change in choice probabilities across trials, whereas TTB predicts a completely flat probability profile.

[2] This design pits Take The Best (TTB) against the Weighted Additive (WADD) model by systematically varying the compensatory evidence from lower-validity cues while keeping the highest-discriminating cue constant. TTB is a non-compensatory, 'one-reason' heuristic: it decides solely based on the highest-validity cue that discriminates between the options, completely ignoring all other cues. Consequently, TTB predicts identical choice probabilities across all trials where the same cue determines the choice. In contrast, WADD computes a weighted sum of all features. By presenting trials where the highest-validity cue favors one option, but an increasing number of lower-validity cues oppose it, WADD predicts a graded change in choice probabilities (even flipping the choice if the gamma parameter allows for compensatory weighting). TTB predicts a completely flat probability profile across these same trials.

[3] This design quantitatively dissociates the Weighted Additive (WADD) model with non-linear validity scaling from Take The Best (TTB) by systematically varying the amount of compensatory evidence on lower-validity cues while holding the highest-validity discriminating cue constant. TTB is a non-compensatory 'one-reason' heuristic: its choice and predicted choice probability are determined entirely by the first discriminating cue, meaning it predicts a completely flat choice probability profile across any set of trials where the same top cue discriminates. In contrast, WADD integrates all features. By creating a gradient of trials where the highest-validity cue favors Option A but the number of lower-validity cues favoring Option B systematically decreases, WADD predicts a graded, monotonic change in choice probabilities, whereas TTB predicts identical probabilities across the gradient. We repeat this logic for subsets of trials where the first cue is tied and the second cue becomes the highest discriminating cue.

[4] This design quantitatively dissociates Random Guessing from the Weighted Additive (WADD) model by presenting a systematic gradient of evidence. By varying the difference in weighted feature values from Option A dominating completely to Option B dominating completely, WADD predicts a monotonic change in choice probabilities tracking the evidence difference. In stark contrast, Random Guessing predicts a completely flat choice probability profile at exactly 0.5 across all trials, regardless of how overwhelming the evidence is for one option over the other.

[5] This design quantitatively dissociates the Weighted Additive (WADD) model from Random Guessing by contrasting trials with overwhelming evidence against trials with complex trade-offs. Random Guessing predicts a completely flat 50/50 choice probability across all trials, asserting total insensitivity to feature information. In contrast, WADD predicts that choice probabilities will track the weighted evidence difference. By including 'dominance' trials (where all features favor one option) alongside various trade-off trials (where high-validity cues are pitted against multiple lower-validity cues), we not only test the core divergence between evidence-sensitivity (WADD) and complete insensitivity (Guessing), but also provide the necessary variance to identify WADD's non-linear scaling parameter (gamma) if subjects are indeed evaluating the features.

[6] Both the Random Guessing (advocated) and Spatial Bias (competing) theories assume that subjects ignore the actual feature values. To quantitatively dissociate them, we must isolate response-based patterns from stimulus-driven effects. By using a perfectly symmetric and counterbalanced set of trial pairs (where every rating vector for Option A appears equally often for Option B), we ensure that the aggregate feature evidence for A and B is exactly identical. Random Guessing predicts exactly a 50/50 choice distribution overall and zero autocorrelation in the sequential responses. The Spatial Bias model predicts a deviation from 50% (due to spatial preference for A or B) and significant autocorrelation (tendency to repeat or alternate responses). The symmetric stimulus design ensures that any observed bias or autocorrelation is purely a response heuristic and not an artifact of the stimuli.

[7] To quantitatively dissociate the Spatial Bias / Response Heuristic model from Random Guessing, we must isolate response-based patterns from stimulus-driven effects. Because both theories assume subjects ignore feature values, the stimulus design must ensure that any observed spatial bias or autocorrelation cannot be attributed to the stimuli themselves. We use a perfectly symmetric set of trial pairs where Option A's rating vectors are exactly mirrored by Option B's vectors across the experiment. This guarantees that any feature-based strategy would result in a 50/50 overall choice rate and zero sequential dependence. Consequently, any significant deviation from 50% (spatial bias) or significant autocorrelation (tendency to repeat/alternate) will unambiguously identify the advocated theory's parameters, whereas Random Guessing strictly predicts zero for both.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "Both the Strict Alternation (advocated) and Spatial Bias (competing) theories assume subjects completely ignore stimulus features. The key difference is that Strict Alternation predicts a deterministic sequence with exactly zero spatial bias, whereas the Spatial Bias model allows for a persistent preference for one option over the other alongside probabilistic sequential dependence. To ensure that any observed sequence effects or biases are purely response-driven and not artifacts of the stimuli, we use a perfectly symmetric and counterbalanced set of trial pairs. Every feature configuration presented for Option A is also presented identically for Option B across the experiment. This guarantees that any stimulus-driven model would predict a 50/50 overall choice rate and zero sequential dependence, allowing us to unambiguously isolate the deterministic alternation pattern from a probabilistic spatial bias.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      1,
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
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
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
      0,
      0,
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
      1,
      0,
      0,
      1,
      0
    ],
    [
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
      1
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
      0,
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
  "prompt_token_count": 3094,
  "candidates_token_count": 571,
  "total_token_count": 4333
}
```
