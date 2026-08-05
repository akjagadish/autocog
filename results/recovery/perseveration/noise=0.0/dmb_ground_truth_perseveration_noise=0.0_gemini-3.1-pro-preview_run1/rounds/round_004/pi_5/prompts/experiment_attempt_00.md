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
**Description:** Task Disengagement / Position Bias Theory: Subjects find the fictitious product choice task unengaging or too cognitively demanding to evaluate systematically. Instead of using the expert ratings, they adopt a completely deterministic, zero-effort heuristic of exclusively selecting the same option position (e.g., always choosing Option A or always Option B) on every single trial.

**Parameters:**
- preferred_option: {0, 1}

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    pref = int(parameters["preferred_option"])
    n_opts = np.asarray(state).shape[0]
    
    probs = np.zeros(n_opts)
    if pref < n_opts:
        probs[pref] = 1.0
    else:
        probs[0] = 1.0
        
    return probs
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
**Description:** First Impression Anchoring: Subjects find the task repetitive and cognitively demanding. To minimize effort, they genuinely evaluate the options on the very first trial using a simple tallying heuristic (choosing the option with the most positive features, or guessing if tied). They then use this initial choice as an anchor, deterministically repeating the same position choice (e.g., always choosing A or always B) for the remainder of the experiment. This results in a completely zero within-subject variance while maintaining an even 50/50 split across the population.

**Parameters:**
- dummy: {1.0}

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # Dummy parameter to satisfy the contract
    _ = float(parameters["dummy"])
    
    stim = np.asarray(state, dtype=float)
    n_opts = stim.shape[0]
    
    # On the first trial, evaluate based on simple feature tallying
    if len(history.get("response", [])) == 0:
        sums = np.sum(stim, axis=1)
        max_sum = np.max(sums)
        is_max = (sums == max_sum).astype(float)
        return is_max / np.sum(is_max)
    else:
        # On all subsequent trials, repeat the first trial's choice
        first_resp = int(history["response"][0])
        probs = np.zeros(n_opts)
        if first_resp < n_opts:
            probs[first_resp] = 1.0
        else:
            probs[0] = 1.0
        return probs
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
[0] This design perfectly dissociates Take The Best (TTB) from Tallying using a 5-feature environment with widely spread validities. In each trial, the option favored by the most valid discriminating cue (which TTB follows) is opposed by the option that has a higher total number of winning cues (which Tallying follows). By varying which cue is the highest valid discriminating cue (1st, 2nd, or 3rd), we ensure that the dissociation is robust across different stopping points of the TTB cue cascade.

[1] To cleanly dissociate Tallying from Take The Best (TTB), this design employs a 5-feature environment with strictly decreasing validities. In the critical trials, one option is engineered to win on the single highest-validity discriminating feature (which TTB relies on exclusively), while the competing option wins on a greater number of lower-validity features (which Tallying favors due to its equal-weight counting). By varying which cue is the highest valid discriminating cue (1st, 2nd, or 3rd) and creating symmetric pairs, we ensure robust quantitative divergence across the entire cue cascade without confounding from overall option quality.

[2] To quantitatively dissociate the Weighted Additive (WADD) model from Tallying, we use a 5-feature environment with one highly predictive feature and several weakly predictive features. Tallying ignores cue validities and simply counts the number of features favoring each option, treating all features equally. WADD, conversely, weights features by the log-odds of their (smoothed) validities, meaning the single highly predictive feature will dominate the compensatory sum. By designing trials where one option is favored by the single strong feature and the other option is favored by a larger number of weak features, Tallying and WADD yield strongly opposing predictions.

[3] This design perfectly dissociates the Tallying heuristic from the Weighted Additive (WADD) model by contrasting a single highly predictive feature against multiple weakly predictive features. Tallying ignores cue validities entirely, making choices based purely on the number of features favoring each option. WADD, conversely, scales feature contributions by their log-odds validity (even after Laplace smoothing), meaning a single extremely valid cue will dominate several weak cues. By pairing an option that is favored by the single strong cue against an option favored by 3 or 4 weak cues, Tallying and WADD yield diametrically opposed predictions on every trial.

[4] To cleanly dissociate the Weighted Additive (WADD) Model from Overload/Indifference Theory, this design pits a single highly predictive cue against varying numbers of moderately or weakly predictive cues. Overload/Indifference Theory posits that such adversarial conflicts cause cognitive overload, resulting in pure random guessing (a 50/50 choice probability) across all these trials. Conversely, WADD leverages Laplace smoothing to regularize the extreme validities and then computes a compensatory log-odds sum. Under WADD, these conflicts do not yield perfect ties; instead, the model predicts systematic, non-random preferences depending on the exact balance of the smoothed log-odds weights. By varying the number and strength of the opposing weak cues, we generate trials where WADD predicts distinct, directional choices (away from 0.5), sharply contrasting with the universal 0.5 prediction of the Overload theory.

[5] To conclusively demonstrate Overload/Indifference Theory over the Weighted Additive (WADD) model, we must prevent WADD from using a global 'guessing' parameter (like epsilon=1.0 or beta=0.0) to fit the data. We achieve this by mixing 'strict dominance' trials (where one option is clearly superior and subjects should rarely guess) with 'adversarial conflict' trials. In the conflict trials, a single highly predictive cue is pitted against varying numbers of weakly predictive cues (1 to 4 weak cues). WADD computes a compensatory sum based on smoothed log-odds; because the number of weak cues varies, WADD mathematically must predict a shifting preference across these conflict trials. Overload/Indifference Theory, however, predicts a flat 50/50 guessing rate across all adversarial conflicts due to cognitive overload, regardless of the exact number of weak cues. If subjects show strong preferences on dominance trials but flat 50/50 choices on all conflict trials, WADD will fail to fit the data.

[6] To conclusively dissociate the Weighted Additive (WADD) model from the Task Disengagement / Position Bias theory, this design employs strong stimulus-driven reversals. Position Bias theory posits that subjects ignore the stimuli and deterministically choose the same position (e.g., always Option A or always Option B) across all trials. WADD, however, integrates the feature values weighted by their smoothed validities. By including trials where Option A strongly dominates Option B, and symmetrically matched trials where Option B strongly dominates Option A, WADD predicts clear switching behavior corresponding to the superior option. Position Bias cannot account for any variance in choices driven by the changing feature values, thus failing completely on trials where the non-preferred position is objectively superior.

[7] To conclusively dissociate the Task Disengagement / Position Bias theory from the Weighted Additive (WADD) model, this design features pairs with extreme, unambiguous stimulus-driven advantages for Option A, and symmetrically matched trials with extreme advantages for Option B. The Position Bias theory predicts that subjects completely ignore the ratings and deterministically choose the same position (e.g., always Option A) across all trials. WADD, as a compensatory model driven by feature values, must predict that choices shift to track the superior option. WADD cannot mathematically predict a 100% preference for Option A on trials where Option B has a strictly dominant feature profile. Thus, a subject exhibiting pure position bias will drastically penalize the WADD model.

[8] To dissociate First Impression Anchoring from Task Disengagement (Position Bias), we design trials with strong, unambiguous variance in the Tallying heuristic's favored option. Under the Advocated Theory, subjects evaluate the very first trial using Tallying and then deterministically anchor on that position for the rest of the experiment. By ensuring that Tallying strongly favors Option A on half the trials and Option B on the other half, we create a scenario where the population's anchored position depends entirely on the randomized presentation order (specifically, the first trial), rather than an intrinsic, stimulus-independent position bias.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To conclusively dissociate the Advocated Theory (Task Disengagement / Position Bias) from the Competing Theory (First Impression Anchoring), this design ensures that the simple Tallying heuristic strictly favors Option B on every single trial. Under the Competing Theory, subjects use Tallying on the very first trial and then deterministically anchor on that choice for the rest of the experiment. Because Tallying always favors Option B in this design, the Competing Theory predicts that 100% of subjects will anchor on Option B, completely failing to account for any subjects who exhibit a position bias for Option A. The Advocated Theory, possessing a free parameter for the preferred position, can perfectly fit subjects who always choose Option A.",
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
      0,
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
  "prompt_token_count": 3125,
  "candidates_token_count": 427,
  "total_token_count": 5491
}
```
