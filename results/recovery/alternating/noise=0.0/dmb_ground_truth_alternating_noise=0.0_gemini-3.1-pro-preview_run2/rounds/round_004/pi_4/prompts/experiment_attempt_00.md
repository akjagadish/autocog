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
**Description:** Take-The-Best (TTB) with High Noise: Decision makers employ a lexicographic heuristic, searching through cues in order of descending validity. They stop at the first cue that discriminates between the two options and choose the option with the higher value on that cue. However, to accommodate the empirical observation that agreement with any deterministic strategy hovers around 50%, the model incorporates a very high lapse rate (epsilon) and a low softmax inverse temperature (beta). This restricts the model to primarily exhibit random guessing, with only a weak TTB signal, matching the high degree of noise in the observed data.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity; stable sort handles ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    
    # Lexicographic search
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No cue discriminates, guess uniformly
        return np.array([0.5, 0.5])
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over binary TTB scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Contrarian WADD: Decision makers evaluate options by computing a weighted sum of their features, but they may distrust the provided expert ratings or view them as added complexity. Thus, they apply a scaling factor to the validities that can be negative, leading to an 'Anti-Tallying' or contrarian preference for options with lower scores. This weak contrarian signal is obscured by a very high rate of random guessing (lapse rate).

**Parameters:**
- alpha: [-2.0, 1.0]
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Contrarian WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Compute scores with the alpha scaling factor (which can be negative)
    score_a = np.dot(stim[0], val) * alpha
    score_b = np.dot(stim[1], val) * alpha
    scores = np.array([score_a, score_b])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design dissociates Take The Best (TTB) from Tallying by creating direct conflicts between the single most valid discriminating cue and the total count of winning features. In several trials, one option wins on the highest validity cue (which determines the TTB choice) while the other option wins on a larger number of lower-validity cues (which determines the Tallying choice). Additionally, cases where Tallying predicts a tie (equal number of winning features) but TTB makes a deterministic prediction based on validity order are included.

[1] This design robustly dissociates Tallying from Take The Best (TTB) by pitting the single most valid discriminating cue against the sheer count of winning features. The validities are strictly ordered (0.95, 0.85, 0.75, 0.65). In many trials, one option possesses the highest-validity discriminating feature (and thus wins under TTB), while the opposing option wins on a larger number of lower-validity features (and thus wins under Tallying). Additionally, we include trials where the total feature-wise wins are perfectly tied (causing Tallying to guess uniformly at random), yet TTB makes a deterministic prediction because one option has a higher value on the top discriminating cue.

[2] To quantitatively dissociate Non-linear WADD from Tallying, we must decouple the sheer count of winning features from the cumulative validity of those features. Tallying strictly predicts choices based on the number of feature-wise wins, collapsing to random guessing (50/50) whenever the number of winning features is equal. In contrast, Non-linear WADD integrates the validities of the winning features, scaled by a non-linear parameter (gamma). This design includes trials where Tallying predicts a tie but WADD predicts a strong preference based on validity (e.g., winning on 2 high-validity vs 2 low-validity features). It also includes trials where Tallying favors one option (more wins) but WADD favors the other (fewer wins, but on features with much higher validities), allowing us to estimate the gamma parameter and show that subjects are sensitive to validity magnitudes rather than just counts.

[3] This design strictly dissociates Tallying from Non-linear WADD by systematically varying the count of winning features independently of their validities. Tallying relies solely on the number of strict feature-wise wins, predicting a uniform 50/50 guess whenever the number of wins is tied, and a robust preference whenever one option has more wins, completely ignoring validity magnitudes. WADD, by contrast, weights wins by their validities (scaled by gamma). The trial set includes 'Tallying Ties' where options win on an equal number of features but differ in validity (WADD strongly prefers the high-validity option), and 'Reversal' trials where one option wins on more features (favored by Tallying) but the other wins on fewer, higher-validity features (favored by WADD).

[4] The competing theory (TTB with High Noise) is heavily constrained to predict near-random guessing across all trials due to its high lapse rate (epsilon >= 0.8) and low inverse temperature (beta <= 0.5). To robustly dissociate this from the advocated theory (Non-linear WADD, which spans the full parameter range), we must include trials where WADD predicts strong, systematic deviations from 50/50 guessing. Furthermore, to prove that any structured choice behavior is specifically driven by WADD rather than a deterministic TTB heuristic, we include 'compensatory' trials where the single best validity cue (favored by TTB) is pitted against a coalition of lower-validity cues (favored by WADD). If subjects show strong, consistent preferences for the coalition over the single best cue, it falsifies both the 'high noise' assumption of the competing theory and the 'lexicographic' assumption of standard TTB, while perfectly aligning with the advocated WADD model.

[5] This design leverages a unique, strong quantitative dissociation between TTB with High Noise and Non-linear WADD. Because TTB operates lexicographically and maps choices to binary scores (1.0 for the winner, 0.0 for the loser) before applying softmax and lapse, it predicts the *exact same* weak preference (e.g., 52%) for the TTB-favored option across all trials where a single cue discriminates, regardless of whether that cue is the most valid or the least valid. In contrast, WADD computes a weighted sum. To predict a weak preference rather than 50/50 guessing, WADD must use a non-zero beta. However, any non-zero beta in WADD will cause the predicted choice probabilities to vary depending on the magnitude of the score difference. By including trials where the TTB decision rests on cues of varying validities (e.g., Cue 1 vs Cues 2-5, or merely Cue 5 breaking a tie), TTB with High Noise predicts a constant flat line of weak preference across all trial types, whereas WADD necessarily predicts fluctuating probabilities depending on the specific cues involved.

[6] TTB with High Noise is highly constrained: because it applies a massive lapse rate to a lexicographic rule, it predicts a constant, weak preference (e.g., ~55%) for the option favored by the highest-validity discriminating cue across all trials. It cannot predict strong preferences, nor can it account for the magnitude of the evidence difference. In contrast, Non-linear WADD computes a weighted sum, predicting choice probabilities that scale dynamically with the cumulative score difference. By including a spectrum of trials—ranging from complete dominance (where WADD predicts a strong preference but TTB is forced to predict a weak one) to various compensatory conflicts (where a coalition of lower-validity cues outweighs the single best cue, causing WADD to predict strong preference reversals while TTB rigidly predicts a weak preference for the top cue)—we can strictly dissociate the two models based on the variance and direction of choice probabilities across trials.

[7] To uniquely validate 'TTB with High Noise' over 'Non-linear WADD', we must exploit the parameter bounds of WADD (specifically, gamma <= 5.0). By pitting the single highest-validity cue against a coalition of multiple slightly lower-validity cues, we create trials where WADD mathematically cannot favor the top cue, even at its maximum non-linear scaling (gamma = 5.0). If subjects consistently show a weak preference for the option with the single best cue, it perfectly matches the lexicographic-but-noisy predictions of TTB with High Noise, while strictly falsifying the WADD model which is forced to predict a preference for the coalition.

[8] To quantitatively dissociate Contrarian WADD from TTB with High Noise, we exploit the 'alpha' parameter in Contrarian WADD, which can be negative. A negative alpha implies an 'Anti-Tallying' or contrarian preference for the option with the lower weighted sum of features. TTB, regardless of its high noise, will always consistently predict a weak preference for the option that wins on the highest-validity discriminating cue. By designing trials where the TTB-favored option has a higher weighted sum, we create a strict divergence: TTB predicts the higher-sum option, while Contrarian WADD predicts the lower-sum option. This includes dominance trials and compensatory trials where the TTB winner is heavy with positive features.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To uniquely validate TTB with High Noise over Contrarian WADD, we exploit the fact that TTB is strictly lexicographic, whereas Contrarian WADD computes a weighted sum (with a potentially negative alpha). By keeping the highest-validity cue fixed in favor of one option while drastically varying the remaining cues, we create trials where the WADD score difference radically changes sign. If subjects consistently show a weak preference for the option favored by the highest-validity cue (e.g., Option A in Trial 1 and Trial 2), Contrarian WADD cannot fit this behavior: a positive alpha would fail on Trial 1 (where B has a much higher weighted sum), and a negative alpha would fail on Trial 2 (where A has a much higher weighted sum). Thus, constant weak preference for the top cue strictly falsifies Contrarian WADD while perfectly matching TTB with High Noise.",
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
      0,
      0,
      0
    ],
    [
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
      1
    ],
    [
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
      0
    ],
    [
      1,
      0,
      1,
      1,
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
      1,
      1,
      1,
      1
    ],
    [
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
      0
    ],
    [
      1,
      1,
      1,
      1,
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
      0
    ],
    [
      1,
      0,
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
  "prompt_token_count": 3822,
  "candidates_token_count": 535,
  "total_token_count": 6209
}
```
