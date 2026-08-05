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


## COMPETING THEORY
**Description:** Weighted Additive (WADD) Theory with Normalized Linear Interpolation: Decision-makers evaluate options by computing a compensatory utility for each option. This utility is the dot product of the option's feature vector and a subjectively transformed set of cue validities. To decouple subjective weighting from the choice temperature, the validities and uniform weights (Tallying) are both normalized to sum to 1 before being linearly mixed by an individual-specific parameter (alpha). Choices are then made probabilistically using a softmax function over these bounded utilities.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- alpha: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])
    
    # Normalize validities and uniform weights to sum to 1
    val_norm = val / np.sum(val)
    uniform_weight = 1.0 / len(val)
    
    # Linear interpolation between normalized uniform weighting and normalized validity weighting
    w = (1.0 - alpha) * uniform_weight + alpha * val_norm
    
    # Compute compensatory utility as the dot product of features and subjective validities
    u_a = np.dot(a, w)
    u_b = np.dot(b, w)
    
    scores = np.array([u_a, u_b])
    
    # Softmax choice rule with max-subtraction for numerical stability
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design systematically pits Take The Best (TTB) against Tallying. By using five features with clearly separated validities, we can construct trials where the highest-validity discriminating cue strongly favors one option, while the sheer number of winning cues favors the other. TTB predicts choices based exclusively on the first discriminating cue in descending validity order, completely ignoring the count of lower-validity cues. In contrast, Tallying treats all features equally and chooses the option with the most winning cues, ignoring their validities. These opposing predictions will yield a stark quantitative dissociation between the two theories.

[1] By utilizing a 4-feature design, we construct scenarios where the option favored by the single highest-validity discriminating cue (Take The Best) is consistently opposed by the option with the greater total number of winning cues (Tallying). We also include trials where Tallying predicts a tie (guessing) because both options win on an equal number of cues, whereas TTB has a strict preference based on the validities of the discriminating cues. This mix of opposing predictions and differential tie-breaking guarantees a strong quantitative dissociation.

[2] This experiment is designed to strongly dissociate the Weighted Additive (WADD) model from the Tallying heuristic using a 5-feature binary choice task. WADD integrates all features by weighting them according to their validities, allowing a smaller number of highly valid cues to compensate for a larger number of less valid cues. Tallying, on the other hand, strictly counts the number of winning features for each option and ignores their validities. By setting a steep distribution of validities (e.g., two very high and three lower validities), we can create critical trials where one option is favored by WADD due to possessing the high-validity features, while the alternative is favored by Tallying because it possesses a greater total number of winning features. We also include trials where Tallying predicts a tie (equal number of winning features) but WADD predicts a distinct preference, as well as baseline trials where both models agree.

[3] This design specifically pits Tallying against the Weighted Additive (WADD) model by using five features with a steep distribution of validities (two high, three low). Tallying simply counts the number of features favoring each option, treating all cues equally regardless of validity. WADD, conversely, integrates validities to form a compensatory sum. The trial set includes pairs where a smaller number of high-validity cues competes against a larger number of low-validity cues (producing opposing choices), as well as trials where both options are supported by an equal number of cues (producing a Tallying tie) but WADD strongly favors the option with the higher-validity cues. This ensures a robust quantitative dissociation across different choice profiles.

[4] To dissociate Hybrid Tallying from Pure Tallying, we must exploit the only condition where their predictions diverge: when the total number of winning features is tied. Pure Tallying predicts a 50/50 guess on any trial where both options win on an equal number of cues. Hybrid Tallying, however, applies a 'premium' strictly as a tie-breaker for the single most valid cue. Thus, on tally-tied trials where one option wins the highest-validity cue, Hybrid Tallying will predict a distinct preference for that option. We also include control trials where one option has a strict tally advantage but loses the most valid cue; because the premium is strictly less than 1, Hybrid Tallying and Pure Tallying will agree on these, confirming that the premium acts only as a soft modulator rather than a lexicographic rule.

[5] To quantitatively dissociate Pure Tallying from Hybrid Tallying, we focus on the specific conditions where their predictions diverge. Pure Tallying predicts a strict 50/50 guess whenever the number of winning features is equal, completely ignoring cue validities. Hybrid Tallying, however, applies a 'premium' to the single most valid cue, predicting a preference for the option that wins this top cue even when the overall tally is tied. Furthermore, when one option wins the tally by exactly one cue but loses the top cue, Pure Tallying predicts a standard preference strength based on a score difference of 1, whereas Hybrid Tallying predicts a significantly attenuated preference (score difference of 1 - premium). This design includes tally ties with top-cue wins, narrow tally advantages opposing the top cue, and control trials where the top cue is tied, providing a comprehensive test of whether the top cue exerts any special influence over choices.

[6] To pit Take-The-Best (TTB) against Tallying, we use a 5-feature design with distinct validities. TTB evaluates cues strictly by validity, choosing the option favored by the single most valid discriminating cue. Tallying ignores validity entirely and simply counts the number of winning cues for each option. We include 'conflict' trials where the highest valid discriminating cue favors one option while a larger number of lower-validity cues favors the other. We also include 'tally tie' trials where both options win on an equal number of cues (predicting a 50/50 guess in Tallying) but TTB makes a deterministic choice based on the validity hierarchy.

[7] This design systematically contrasts Tallying against Take-The-Best (TTB) using five features with distinct validities. TTB relies solely on the highest-validity discriminating cue, ignoring all others. Tallying, however, counts the total number of winning features for each option, ignoring their validities entirely. The trial set includes pairs where the single best cue favors one option while a larger quantity of lower-validity cues favors the alternative, leading to opposing predictions. We also include 'tally tie' trials where both options win on an equal number of features (leading Tallying to predict a 50/50 guess) but TTB makes a deterministic prediction based on the highest-validity cue.

[8] To quantitatively dissociate the Weighted Additive (WADD) Model from the Tallying heuristic, we use a 5-feature design with a steep validity distribution (two highly valid experts, three low-validity experts). Tallying simply counts the number of winning features for each option, ignoring their validities. WADD integrates features by weighting them according to their normalized validities. The design includes 'conflict' trials where an option is favored by WADD due to winning the high-validity features, while the alternative is favored by Tallying because it wins on a greater number of low-validity features. We also include 'tally tie' trials where both options win on an equal number of features (leading Tallying to predict a 50/50 guess), but WADD makes a deterministic prediction based on the validity weights.

[9] This design systematically pits Tallying against the Normalized Weighted Additive (WADD) Model. We use five features with a steep validity distribution (two highly valid cues, three low-validity cues). Tallying evaluates options by counting the number of strictly winning features, completely ignoring their validities. WADD integrates all features by weighting them according to their normalized validities. The design includes 'conflict' trials where WADD prefers the option with fewer, higher-validity features, while Tallying prefers the option with a greater total number of winning features. It also includes 'tally tie' trials where both options win on an equal number of features (leading Tallying to predict a 50/50 guess), but WADD makes a deterministic prediction based on the validity weights. These divergent predictions ensure a strong quantitative dissociation.

[10] To dissociate the Mixture Model (TTB + Tallying) from Pure Tallying, we need trials where the two heuristic components make divergent predictions. Pure Tallying only counts the number of winning features, completely ignoring validities, and predicts a 50/50 guess when the tally is tied. The Mixture Model, however, includes a Take-The-Best (TTB) component that breaks ties lexicographically based on validities, and pulls probabilities towards the TTB prediction even when the tally favors the other option. By including 'tally tie' trials where TTB has a strict preference, and 'conflict' trials where Tallying favors one option but the highest validity cue favors the other, we can observe whether choices are strictly governed by feature counts or if there is a lexicographic validity influence.

[11] To dissociate Pure Tallying from a Mixture Model of TTB and Tallying, we exploit trials where the two models make distinct predictions. Pure Tallying relies solely on the count of winning features, ignoring validities, and predicts a strict 50/50 guess when the tally is tied. The Mixture Model incorporates a Take-The-Best (TTB) component that breaks ties based on the highest-validity discriminating cue and attenuates the tallying preference when the highest-validity cue points in the opposite direction. The design includes 'tally tie' trials where TTB has a strict preference, and 'conflict' trials where Tallying favors one option but TTB favors the other.

[12] To quantitatively dissociate the Frugal Threshold Heuristic from Pure Tallying, we utilize a 6-feature design. Frugal Threshold evaluates cues in descending order of validity and stops as soon as the net evidence (difference in winning cues) reaches a threshold 'theta'. In contrast, Tallying evaluates all cues exhaustively and ignores validities entirely, choosing the option with the most winning cues. We include 'conflict' trials where the top valid cues strongly favor one option (triggering early stopping in Frugal Threshold) while a larger quantity of lower-validity cues favors the other option (dictating the Tallying choice). We also include 'tally tie' trials where both options win on an equal number of cues overall (predicting a 50/50 guess in Tallying), but Frugal Threshold makes a deterministic choice due to early stopping on the highest-validity cues.

[13] To quantitatively dissociate the Tallying heuristic from the Frugal Threshold Heuristic, we use a 6-feature design. Tallying evaluates all cues exhaustively and ignores validities entirely, predicting choices based strictly on the total number of winning cues (and predicting a 50/50 guess when the tally is tied). The Frugal Threshold Heuristic evaluates cues in descending order of validity and stops searching as soon as the net evidence reaches a cognitive threshold 'theta'. We include 'conflict' trials where the highest-validity cues strongly favor one option (triggering early stopping in Frugal Threshold) while a larger quantity of lower-validity cues favors the other option (dictating the Tallying choice). We also include 'tally tie' trials where both options win on an equal number of cues overall (predicting a 50/50 guess in Tallying), but Frugal Threshold makes a deterministic choice due to early stopping on the highest-validity cues.

[14] To dissociate the Advocated Theory (Tallying with Salience-Biased Tie-Breaking) from Pure Tallying, we exploit trials where the two models diverge in their treatment of validities. Pure Tallying ignores validities entirely, predicting a strict 50/50 guess whenever the total number of winning features is tied, and predicting choices based solely on the count difference otherwise. The Advocated Theory integrates a non-linear validity-weighted component that can either heavily weight high-validity cues or paradoxically weight low-validity cues (via a negative gamma). We include 'tally tie' trials where options win on disjoint sets of features with varying validities (e.g., highest vs. lowest validities); Pure Tallying predicts 50/50, while the Advocated Theory predicts a systematic preference. We also include 'conflict' trials where an option wins the tally but loses on highly salient features, causing the Advocated Theory to attenuate or reverse the pure tallying prediction.

[15] To strongly dissociate Pure Tallying from the Competing Theory (Tallying with Salience-Biased Tie-Breaking), this design exploits the core difference in how they handle cue validities. Pure Tallying evaluates options strictly by counting the number of winning features, ignoring validities completely; it predicts a 50/50 guess whenever the tally is tied. The Competing Theory integrates a non-linear validity-weighted component that breaks tally ties and modulates preference strength based on validities. We include 'tally tie' trials where options win on disjoint sets of features with varying validities (e.g., highest vs. lowest); Pure Tallying predicts exactly 50/50, while the Competing Theory predicts a systematic preference. We also include 'conflict' trials where one option wins the tally but loses the highest-validity features, causing the Competing Theory to attenuate or reverse the pure tallying prediction.

[16] To quantitatively dissociate the Heuristic Mixture Theory (TTB + Tallying) from Pure Tallying, we use a 5-feature design and construct trials where the two models make divergent predictions. Pure Tallying relies solely on the count of winning features, ignoring validities, and predicts a strict 50/50 guess when the tally is tied. The Mixture Theory incorporates a Take-The-Best (TTB) component that bases choices on the highest-validity discriminating cue. We include 'tally tie' trials where Tallying predicts guessing but TTB has a strict preference, and 'conflict' trials where Tallying favors one option (due to a higher quantity of winning cues) while TTB favors the other (due to the highest-validity cue). These opposing predictions will yield a strong quantitative dissociation.

[17] To quantitatively dissociate Pure Tallying from the Heuristic Mixture Theory (TTB + Tallying), we use a 5-feature design and exploit trials where the two models make divergent predictions. Pure Tallying relies solely on the count of winning features, ignoring validities, and predicts a strict 50/50 guess when the tally is tied. The Mixture Theory incorporates a Take-The-Best (TTB) component that bases choices on the highest-validity discriminating cue. We include 'tally tie' trials where Tallying predicts guessing but TTB has a strict preference, and 'conflict' trials where Tallying favors one option (due to a higher quantity of winning cues) while TTB favors the other (due to the highest-validity cue). These opposing predictions will yield a strong quantitative dissociation.

[18] This design quantitatively dissociates Pure Tallying from the Weighted Additive (WADD) Model with Dampened Log-Odds. Pure Tallying evaluates options strictly by counting the number of winning features, ignoring validities completely, and predicts a 50/50 guess whenever the tally is tied. The WADD model integrates validities via a dampened log-odds weighting scheme, allowing high-validity cues to compensate for a smaller quantity of winning features without producing extreme, non-compensatory lexicographic dominance. The trial set includes 'tally tie' trials where WADD predicts a strong preference due to high-validity features, and 'conflict' trials where Tallying favors an option with a greater quantity of low-validity features while WADD favors the alternative with fewer, but higher-validity features.

[19] This design systematically dissociates Pure Tallying from the Weighted Additive (WADD) Model with Dampened Log-Odds. Pure Tallying simply counts the number of winning features for each option, completely ignoring cue validities, and predicts a strict 50/50 guess when the tally is tied. In contrast, the WADD model integrates validities via a dampened log-odds weighting scheme, allowing high-validity cues to compensate for a smaller quantity of winning features. The trial set includes 'tally tie' trials where WADD predicts a strong preference due to high-validity features, and 'conflict' trials where Tallying favors an option with a greater quantity of low-validity features while WADD favors the alternative with fewer, but higher-validity features.

[20] To quantitatively dissociate the Mixture Model (TTB + Tallying) from Pure Tallying, this design leverages trials where the two heuristic components make divergent predictions. Pure Tallying evaluates options strictly by counting the number of winning features, ignoring validities completely; it predicts a 50/50 guess whenever the tally is tied. The Mixture Model, however, incorporates a Take-The-Best (TTB) component that breaks ties based on the highest-validity discriminating cue, and pulls probabilities towards the TTB prediction even when the tally favors the other option. By including 'tally tie' trials where TTB has a strict preference, and 'conflict' trials where Tallying favors one option but the highest validity cue favors the other, we can observe whether choices are strictly governed by feature counts or if there is a lexicographic validity influence.

[21] To quantitatively dissociate Pure Tallying from a Mixture Model of Take-The-Best (TTB) and Tallying, we use a 5-feature design with clearly separated validities. Pure Tallying predicts choices based strictly on the count of winning features, ignoring validities, and predicts a 50/50 guess when the tally is tied. The Mixture Model blends Tallying with TTB, which lexicographically breaks ties based on the highest-validity discriminating cue and pulls probabilities towards the TTB prediction even when the tally favors the other option. We include 'tally tie' trials where Pure Tallying predicts guessing but TTB has a strict preference, and 'conflict' trials where Tallying favors one option (due to a higher quantity of winning cues) while TTB favors the other (due to winning the highest-validity cue).

[22] To quantitatively dissociate the Advocated Theory (Normalized Linear Interpolation WADD) from the Competing Theory (Pure Tallying), we use a 5-feature design with a steep validity distribution. Pure Tallying strictly counts the number of winning features, ignoring validities, and predicts a 50/50 guess whenever the tally is tied. The Advocated Theory computes a compensatory utility by interpolating between uniform weights and normalized validities, allowing high-validity features to compensate for a smaller quantity of winning features. The trial set includes 'tally tie' trials where both options win on an equal number of features (Tallying predicts guessing) but the Advocated Theory predicts a strong preference based on validities. It also includes 'conflict' trials where Tallying favors one option due to a greater quantity of low-validity features, while the Advocated Theory favors the alternative due to fewer but higher-validity features.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design aims to quantitatively dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (Normalized Linear Interpolation WADD). Pure Tallying simply counts the number of winning features for each option, completely ignoring cue validities, and predicts a strict 50/50 guess when the tally is tied. In contrast, the WADD model integrates validities, allowing high-validity cues to compensate for a smaller quantity of winning features. The trial set includes 'tally tie' trials where WADD predicts a strong preference due to high-validity features, and 'conflict' trials where Tallying favors an option with a greater quantity of low-validity features while WADD favors the alternative with fewer, but higher-validity features.",
  "validities": [
    0.95,
    0.9,
    0.6,
    0.55,
    0.5,
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
      0,
      0
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
      0
    ],
    [
      0,
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
      0,
      1,
      1
    ],
    [
      0,
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
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
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
  "prompt_token_count": 6376,
  "candidates_token_count": 560,
  "total_token_count": 9505
}
```
