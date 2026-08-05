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
**Description:** Tally-then-TTB (Tally with Validity Tie-Breaker): Decision-makers first compare options by tallying the total number of winning features for each option, ignoring cue validities. If one option has strictly more wins, it is chosen. This captures the compensatory nature of decision-making when there is a clear majority of supporting cues. However, if the tally results in a tie, the decision-maker falls back to a non-compensatory 'Take-The-Best' strategy to break the tie, choosing the option that wins on the single most valid cue. This hybrid model preserves robust Tallying performance on unequal-tally trials while capturing deterministic validity-based tie-breaking on ambiguous trials.

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
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying phase: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    score_a = a_wins
    score_b = b_wins
    
    # Tie-breaking phase: if tallies are equal, use TTB
    if a_wins == b_wins:
        cue_order = np.argsort(-val, kind="stable")
        for idx in cue_order:
            if a[idx] > b[idx]:
                score_a += 1.0
                break
            elif b[idx] > a[idx]:
                score_b += 1.0
                break
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
[0] To dissociate Take The Best (TTB) and Tallying, we use a 4-feature design where the validities are strictly ordered. The trials are constructed such that the single highest-validity discriminating cue points to one option, while the simple sum of winning features (the tally) points to the other option or results in a tie. TTB will reliably choose the option favored by the top discriminating cue, ignoring the sheer number of lower-validity cues, whereas Tallying will choose the option with more total feature wins, ignoring the validities entirely.

[1] To strongly dissociate Tallying (advocated) from Take The Best (competing), we use a 5-feature design with strictly ordered validities. The trials are constructed so that the highest-validity discriminating cue points to one option, while the simple sum of winning features (the tally) points to the other option or results in a tie. TTB will reliably choose the option favored by the top discriminating cue, ignoring the sheer number of lower-validity cues, whereas Tallying will choose the option with more total feature wins, ignoring the validities entirely.

[2] To dissociate Weighted Additive (WADD) from Tallying, we exploit the fact that WADD integrates cue validities into a weighted sum, whereas Tallying simply counts the number of features favoring each option, treating all validities equally. We use a 5-feature design with a clear split between two high-validity cues and three low-validity cues. In key trials, one option wins on the two high-validity cues (resulting in a higher WADD score), while the other option wins on the three low-validity cues (resulting in a higher Tallying score). Additionally, we include trials where Tallying predicts a tie (equal number of winning features), but WADD strongly prefers one option due to the validity differences. This quantitative and qualitative divergence ensures the models yield distinct, distinguishable choice patterns.

[3] This design strictly dissociates Tallying from Weighted Additive (WADD) by contrasting the quantity of winning cues with their cumulative validities. We use a 5-feature design where two cues have very high validities and three have low validities. In the key dissociation trials, one option is favored by the two high-validity cues (winning the WADD evaluation), while the other option is favored by the three low-validity cues (winning the Tallying evaluation). We also include trials where Tallying predicts a perfect tie (each option wins on an equal number of cues), but WADD strongly prefers the option supported by higher-validity cues. This ensures a robust quantitative and qualitative divergence in the choice probabilities generated by the two models.

[4] To dissociate the Take-K heuristic from Tallying, we exploit their differential treatment of the lowest-validity cues. Take-K sorts cues by validity and ignores the bottom fraction (depending on its k_prop parameter), whereas Tallying counts wins across all cues regardless of validity. We use a 6-feature design where the 6th cue has a substantially lower validity than the rest. The trials are constructed so that the top 5 cues result in a tie (which Take-K will guess on) but the 6th cue breaks the tie (which Tallying will follow), or conversely, the top 5 cues favor one option but the 6th cue balances the overall tally (leading Tallying to guess, while Take-K makes a deterministic choice).

[5] To dissociate Tallying (the advocated theory) from the Take-K heuristic, we exploit both the structural difference in cue usage and the bounds on their noise parameters. Take-K sorts cues by validity and may discard the lowest-validity cues (depending on its k_prop parameter), whereas Tallying counts wins across all cues equally. Furthermore, Take-K is constrained to have high determinism (beta >= 2.0, epsilon <= 0.1), while Tallying allows for much softer, noisier choices (beta down to 0.1, epsilon up to 0.5). We use a 7-feature design where the 7th cue has the lowest validity. In key trials, the top 6 cues result in a perfect tie (3 wins each), but the 7th cue breaks the tie. Take-K (if k_prop < 1.0) will ignore the 7th cue and predict a 50/50 guess, while Tallying will use the 7th cue to predict a preference. Even if Take-K sets k_prop = 1.0 to mimic Tallying, its high-beta constraint forces it to predict very sharp probabilities (>= 88% for a 1-win difference), whereas Tallying can correctly capture a much softer empirical preference (e.g., 60%) driven by the small 1-cue tally difference.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To cleanly dissociate Tally-then-TTB from pure Tallying, we use a 5-feature design with distinct validities. The core difference between the two models arises when the simple tally of features results in a tie. Pure Tallying has no mechanism to break ties and will strictly predict a 50/50 guess regardless of its noise parameters. In contrast, Tally-then-TTB will fall back to the highest-validity discriminating cue to break the tie, resulting in a deterministic preference. We include trials where the tallies are tied but the highest-validity cue points to either A or B, as well as trials where the highest-validity cue is tied so the second-highest must be used. We also include unequal-tally trials where both models agree, to ensure baseline tallying behavior is captured.",
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
      0,
      1,
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
      1,
      0,
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
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3490,
  "candidates_token_count": 527,
  "total_token_count": 5833
}
```
