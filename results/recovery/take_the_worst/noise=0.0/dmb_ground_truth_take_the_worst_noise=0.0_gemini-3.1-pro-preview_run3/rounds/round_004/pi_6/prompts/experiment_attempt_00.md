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
**Description:** Misinterpreted Expert Ratings (Mixture of Standard Tallying and Reverse WADD): Decision-makers consist of two sub-populations. One group understands the instructions correctly and uses a standard Tallying heuristic, choosing the option with more positive features. A second group misinterprets the binary expert ratings as 'defects' or 'costs' (1 = bad, 0 = good) and evaluates options using a Reverse WADD strategy, minimizing the validity-weighted sum of these perceived defects. In scenarios where Tallying and Reverse WADD agree, choice consistency is high. When they disagree, they pull the aggregate preference in opposite directions, weakening the apparent Tallying effect. Crucially, when options tie under Tallying, the standard group guesses uniformly, allowing the Reverse WADD group to entirely drive the aggregate behavior, resulting in a strong 'Negative WADD' effect.

**Parameters:**
- beta_tally: [0.1, 20.0]
- beta_rev: [0.1, 100.0]
- w_tally: [0.0, 1.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    # Strategy 1: Standard Tallying
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * np.array([a_wins, b_wins])
    z_tally -= np.max(z_tally)
    p_tally = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # Strategy 2: Reverse WADD (interpreting 1s as costs/defects)
    val = np.asarray(parameters["validities"], dtype=float)
    wadd_a = np.dot(a, val)
    wadd_b = np.dot(b, val)
    
    beta_rev = float(parameters["beta_rev"])
    # Negative WADD scores to penalize options with higher validity-weighted 1s
    z_rev = beta_rev * np.array([-wadd_a, -wadd_b])
    z_rev -= np.max(z_rev)
    p_rev = np.exp(z_rev) / np.sum(np.exp(z_rev))
    
    # Mixture of the two strategies
    w_tally = float(parameters["w_tally"])
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_rev
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Tallying with a Sub-optimal Tie-Breaker (Negative Cue-Weighting in Ties): Decision-makers primarily compare options using a frugal Tallying heuristic, choosing the option with the greater number of winning cues. However, when options are tied on the number of winning features, individuals do not simply guess or use a rational compensatory strategy. Instead, they exhibit a sub-optimal tie-breaking mechanism where they systematically favor the option with lower-ranked or lower-validity cues (effectively a Negative WADD score). This explains both the strong adherence to Tallying when cue counts differ, and the counter-intuitive preference for lower-validity options when cue counts are tied.

**Parameters:**
- beta: [0.1, 10.0]
- tie_beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    beta = float(parameters["beta"])
    tie_beta = float(parameters["tie_beta"])
    epsilon = float(parameters["epsilon"])
    
    if a_wins != b_wins:
        # Primary strategy: Tallying
        scores = np.array([a_wins, b_wins])
        z = beta * scores
    else:
        # Tie-breaker: Negative WADD (favoring lower validity cues)
        val = np.asarray(parameters["validities"], dtype=float)
        wadd_a = np.dot(a, val)
        wadd_b = np.dot(b, val)
        scores = np.array([-wadd_a, -wadd_b])
        z = tie_beta * scores
        
    z = z - np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design leverages the core difference between Take The Best (TTB) and Tallying: TTB relies exclusively on the single most valid discriminating cue, ignoring all other cues, while Tallying counts the number of winning cues across all features, ignoring their validities. By using five features with strictly descending validities, we can construct trials where the option that wins on the highest-validity cue (favored by TTB) loses on the majority of the lower-validity cues (favored by Tallying). We also include trials where Tallying predicts a tie (guessing) while TTB has a strict preference, providing a clear quantitative dissociation.

[1] To strongly dissociate Tallying from Take The Best (TTB), we use five features with strictly descending validities. TTB decides solely based on the highest-validity cue that discriminates between the two options. Tallying, however, counts the total number of winning cues for each option, completely ignoring their validities. The design includes 'opposition' trials where the option with the highest valid discriminating cue (favored by TTB) is outnumbered by the total number of lower-validity cues favoring the other option (favored by Tallying). We also include 'tie' trials for Tallying where both options win on an equal number of cues (leading to guessing under Tallying) while TTB makes a deterministic prediction based on the single most valid cue. This mixture isolates the feature-summing behavior of Tallying from the one-reason stopping rule of TTB.

[2] To quantitatively dissociate Weighted Additive (WADD) from Tallying, this design uses five features with a large spread in validities. Tallying ignores validities entirely, counting only the number of features on which one option strictly beats the other. WADD, by contrast, weights each feature by its validity. The trials include 'opposition' cases where one option wins on a few high-validity features while the other wins on a larger number of low-validity features (WADD prefers the former, Tallying prefers the latter). We also include 'tie' trials for Tallying where both options win on an equal number of features (leading to guessing under Tallying), but WADD strongly prefers the option that wins on the higher-validity features. Option bias in WADD will further shift preferences asymmetrically, a pattern Tallying cannot capture.

[3] To quantitatively dissociate Tallying from Weighted Additive (WADD) with Option Bias, we use five features with a large spread in validities. Tallying ignores validities entirely, counting only the number of features on which one option strictly beats the other. WADD, by contrast, weights each feature by its validity. The trials include 'opposition' cases where one option wins on a few high-validity features while the other wins on a larger number of low-validity features (WADD prefers the former, Tallying prefers the latter). We also include 'tie' trials for Tallying where both options win on an equal number of features (leading to guessing under Tallying), but WADD strongly prefers the option that wins on the higher-validity features. This mixture isolates the feature-summing behavior of Tallying from the validity-weighted integration of WADD.

[4] This design tests a stark quantitative dissociation between the Advocated Theory (Mixture of Deterministic Tallying + Probabilistic WADD) and the Competing Theory (Probabilistic Tallying). By setting validities to [1.0, 0.5, 0.5, 0.6, 0.6], we create paired trials where the WADD score difference between options is held exactly constant, but the Tallying cue-count difference varies (e.g., a difference of 1 vs. 3). Under the Competing Theory, choice probability scales continuously with the cue-count difference, predicting a much stronger preference in the 3-difference trial than the 1-difference trial. Under the Advocated Theory, Tallying is deterministic (outputting 1.0 for the winner as long as the difference is > 0), and the WADD component is identical across both trials; thus, it predicts the exact same choice probability for both trials. We also include Tally-tie trials where WADD breaks the tie, which Probabilistic Tallying cannot capture.

[5] To quantitatively dissociate the Advocated Theory (Probabilistic Tallying) from the Competing Theory (Mixture of Deterministic Tallying + Probabilistic WADD), this design holds the Tallying score difference constant across multiple trials while varying the WADD score difference. Under the Advocated Theory, choice probabilities are entirely determined by the difference in the number of winning cues (the Tally difference). Thus, any two trials where Option A wins on 2 cues and Option B wins on 1 cue will yield the exact same choice probability. Under the Competing Theory, Tallying acts as a deterministic baseline, but the final choice probability is heavily modulated by the WADD score difference. By constructing pairs that all have a Tally difference of +1 for Option A but widely varying WADD differences (e.g., +1.0 vs +0.2), the Advocated Theory predicts a flat consistency curve across these pairs, whereas the Competing Theory predicts varying choice probabilities. We also include 'Tally tie' trials where the Advocated Theory predicts exactly 50% guessing, but the Competing Theory predicts a preference driven by WADD.

[6] To quantitatively dissociate the Advocated Theory (Tallying with Negative WADD tie-breaker) from the Competing Theory (standard Tallying), we focus on trials where both options win on an equal number of cues (Tallying ties). Under the Competing Theory, a Tally tie results in uniformly random guessing (50/50) regardless of the cue validities. Under the Advocated Theory, decision-makers resolve these ties sub-optimally by systematically preferring the option with lower-validity cues (Negative WADD). The design includes several 'tie' trials with stark differences in validity weights between the winning cues of A and B, allowing us to detect this counter-intuitive preference for lower-validity options. We also include 'non-tie' trials where both theories predict the same Tallying-based choice, anchoring the baseline choice behavior.

[7] To quantitatively dissociate the Advocated Theory (Standard Tallying) from the Competing Theory (Tallying with Negative WADD tie-breaker), we focus on trials where both options win on an equal number of cues (Tallying ties). Under the Advocated Theory, a Tally tie results in uniformly random guessing (50/50) regardless of the cue validities. Under the Competing Theory, decision-makers resolve these ties sub-optimally by systematically preferring the option with lower-validity cues (Negative WADD). The design includes several 'tie' trials with stark differences in validity weights between the winning cues of A and B, allowing us to detect whether a counter-intuitive preference for lower-validity options exists. We also include 'non-tie' trials where both theories predict the same Tallying-based choice, anchoring the baseline choice behavior.

[8] This design quantitatively dissociates the Advocated Theory (Tallying with Negative WADD tie-breaker) from the Competing Theory (Mixture of Standard Tallying and Reverse WADD) by focusing on non-tie trials. Under the Advocated Theory, when options do not tie on the number of winning cues, the decision is driven exclusively by Tallying; the magnitude of the WADD or Negative WADD difference is completely ignored. Thus, across multiple trials where the Tallying difference is held constant (e.g., Option A always wins exactly one more cue than Option B), the Advocated Theory predicts an identical choice probability for Option A. In contrast, the Competing Theory posits a mixture of Tallying and Reverse WADD across all trials. Because Reverse WADD penalizes options with a higher validity-weighted sum of 1s (interpreting them as defects), it is often in tension with Tallying. By holding the Tallying difference constant at +1 for Option A but systematically varying the Reverse WADD difference from strongly favoring B to favoring A, the Competing Theory predicts a steeply varying choice probability across these trials, while the Advocated Theory predicts a perfectly flat line.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates the Advocated Theory (Mixture of Standard Tallying and Reverse WADD) from the Competing Theory (Tallying with Negative WADD tie-breaker) by focusing on non-tie trials. Under the Competing Theory, when options do not tie on the number of winning cues, the decision is driven exclusively by Tallying; the magnitude of the WADD or Reverse WADD difference is completely ignored. Thus, across multiple trials where the Tallying difference is held constant (e.g., Option A always wins exactly one more cue than Option B), the Competing Theory predicts an identical choice probability for Option A. In contrast, the Advocated Theory posits a mixture of Tallying and Reverse WADD across all trials. By holding the Tallying difference constant at +1 for Option A but systematically varying the Reverse WADD difference from strongly favoring B to actually favoring A, the Advocated Theory predicts a steeply varying choice probability across these trials, while the Competing Theory predicts a perfectly flat line. Tie trials are also included to anchor the Negative/Reverse WADD effect.",
  "validities": [
    0.95,
    0.9,
    0.6,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
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
      0,
      1,
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
      1,
      1,
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
      0,
      1,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
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
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      1
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4367,
  "candidates_token_count": 542,
  "total_token_count": 7835
}
```
