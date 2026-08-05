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
**Description:** Moderate-Cue Biased Tallying: People primarily compare options by counting the number of features where one option strictly dominates the other (Tallying). However, rather than giving a bonus to the most valid cue, subjects exhibit an aversion to options that rely too heavily on a single extreme cue at the expense of moderate cues. When tallies are close or tied, this manifests as a penalty to the option endorsed by the most valid cue, reflecting a preference for a broader cluster of supporting evidence over a single strong piece of evidence. This preserves the dominance of Tallying for clear differences but explains the systematic anti-top-cue preference observed in tie-breaking scenarios.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Moderate-Cue Biased Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Penalty for relying on the most valid cue
    val = np.asarray(parameters["validities"], dtype=float)
    best_cue = np.argmax(val)
    
    w = float(parameters["w"])
    a_best_win = float(a[best_cue] > b[best_cue])
    b_best_win = float(b[best_cue] > a[best_cue])
    
    # Subtract the penalty weight from the tally score
    score_a = a_wins - w * a_best_win
    score_b = b_wins - w * b_best_win
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

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
[0] These trials are designed to maximally dissociate Take The Best (TTB) from Tallying. TTB decides solely based on the highest-validity discriminating cue, whereas Tallying counts the total number of winning features while ignoring cue validities. In the critical trials, one option is superior on the single most valid cue, but the other option is superior on a larger number of less valid cues. TTB will deterministically choose the former, while Tallying will deterministically choose the latter. We also include trials where Tallying predicts a tie while TTB makes a decisive prediction.

[1] This design strictly dissociates Tallying (which counts the total number of winning features regardless of their validities) from Take The Best (which relies exclusively on the highest-validity discriminating feature). The trials are constructed such that the option favored by the most valid discriminating cue is consistently outvoted by a larger number of less valid cues favoring the other option. This forces TTB and Tallying to make opposite predictions. We also include trials where Tallying predicts a tie (equal number of winning features) while TTB makes a deterministic prediction, further isolating the specific mechanisms of the two heuristics.

[2] This design quantitatively dissociates WADD from Tallying and TTB. By employing a specific set of feature validities, we create trials where WADD (which integrates all cues proportionally to their validities) conflicts with Tallying (which merely counts the number of winning features, ignoring validities). For instance, in trials where one option possesses the top two validities while the other has three lower-validity cues, WADD will favor the former while Tallying will favor the latter. Additionally, we include trials where Tallying predicts a tie (equal number of winning features) but WADD decisively prefers one option due to the validity weights. Finally, we include trials that dissociate WADD from Take The Best (TTB) by having multiple moderately valid cues outweigh the single most valid cue, forcing TTB and WADD to make opposite predictions.

[3] This design quantitatively dissociates Tallying (which simply counts the number of winning features) from WADD (which integrates all cues weighted by their validities). By using a specific set of feature validities, we construct trials where one option possesses the two most valid cues (thus strongly favored by WADD) while the other option possesses three less valid cues (thus favored by Tallying). We also include trials where both heuristics agree, and trials where Tallying predicts a tie (equal number of winning features) but WADD makes a decisive prediction based on the differing validities of those features.

[4] This design specifically dissociates standard Tallying from Validity-Biased Tallying. Standard Tallying counts feature wins and predicts exactly 50/50 when both options win on the same number of features, treating all features equally. Validity-Biased Tallying behaves similarly but adds a fractional bonus to the option that wins on the single most valid cue. We include trials where options tie in the tally but differ on the most valid cue, forcing standard Tallying to guess while VBT predicts a preference. We also include trials where one option wins the tally by 1, but the most valid cue is won by either the tally-winner or the tally-loser. Standard Tallying predicts identical choice certainty across these latter trials, whereas VBT predicts higher certainty when the most valid cue aligns with the tally winner and lower certainty when it opposes.

[5] To conclusively dissociate standard Tallying from Validity-Biased Tallying (VBT), this design systematically varies the tally difference (0, 1, or 2) and the ownership of the single most valid cue (Option A, Option B, or tied). The advocated theory (Tallying) predicts that choice probabilities depend purely on the tally difference, yielding exactly 50/50 guessing on all tally ties and uniform confidence across trials with the same tally difference. The competing theory (VBT) predicts a specific probability shift whenever the single most valid cue is not tied, breaking tally ties and modulating confidence on tally wins. Crucially, we include trials where the most valid cue is tied but other validities differ; VBT predicts these will behave identically to standard Tallying, allowing us to isolate the unique mechanism of VBT's top-cue bonus.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design dissociates Moderate-Cue Biased Tallying from standard Tallying by manipulating the tally difference and the ownership of the most valid cue. While standard Tallying predicts choice probabilities based solely on the tally difference, Moderate-Cue Biased Tallying posits a penalty for the option winning the most valid cue (reflecting an aversion to relying on a single extreme cue over moderate ones). We include trials where the tally is tied but one option wins the most valid cue (forcing standard Tallying to guess, while the advocated model prefers the option winning on moderate cues), and trials where the tally winner either wins or loses the most valid cue (which modulates the advocated model's confidence but not standard Tallying's). A control trial with a tied most valid cue and tied tally ensures baseline agreement.",
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
      1,
      0,
      0
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
      1,
      0,
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
      0,
      0,
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
      0,
      1,
      0
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
      1,
      0,
      0,
      0,
      1
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
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3362,
  "candidates_token_count": 484,
  "total_token_count": 5307
}
```
