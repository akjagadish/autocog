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
**Description:** Extreme Cognitive Noise / Single-Cue Focus: In complex multi-cue decision environments, subjects attempt to rely on the provided cue validities but are quickly overwhelmed by cognitive friction. Instead of integrating multiple cues (like Weighted Additive) or falling back to simple counting (Tallying), they occasionally fixate solely on the single most valid cue to make their decision. However, this fragile single-cue heuristic is heavily masked by an overwhelmingly high baseline guessing rate (lapse), meaning that on the vast majority of trials, subjects simply guess randomly. This explains why choice behavior hovers very close to 0.5 across various conflict and agreement metrics, while allowing for slight, systematic deviations driven by the top validity cue rather than overall cue counts.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.9, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the single highest-validity cue
    best_cue = np.argmax(validities)
    
    # Evaluate options based only on this single cue
    scores = stim[:, best_cue]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the single-cue scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Overwhelmingly high uniform lapse blended in
    n_opts = len(p_core)
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
**Description:** Weak Weighted Additive with Extreme Noise: In complex multi-cue decision environments, subjects attempt to integrate all available information by weighting each cue according to its provided validity (a compensatory Weighted Additive strategy). However, the cognitive demands of multiplying and summing multiple cues lead to severe cognitive overload and disengagement. Consequently, subjects exhibit an overwhelmingly high lapse rate, effectively guessing on the vast majority of trials. The underlying Weighted Additive evaluation only weakly shines through the noise, explaining why choice behavior hovers very close to random guessing (0.5) across conflict metrics while maintaining a slight systematic preference for higher-validity cue combinations.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.9, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters['validities'], dtype=float)
    
    # Weighted Additive (WADD): sum of cues weighted by their validities
    scores = stim @ validities
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax over the WADD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Overwhelmingly high uniform lapse blended in
    n_opts = len(p_core)
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
[0] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, this design exploits compensatory vs. non-compensatory feature weighting. By using four features with validities [0.9, 0.8, 0.7, 0.6], we can create 'compensatory' trials where the highest validity cue favors one option, but the sum of the remaining lower-validity cues favors the other option. TTB relies strictly on the first discriminating cue (the highest validity one) and ignores the rest, whereas WADD integrates all features and will choose the option with more lower-validity cues if their summed weight exceeds the top cue's weight. The trial pairs systematically reverse which option is favored by the top cue vs. the sum of the remaining cues, ensuring that overall choice rates and trial-by-trial responses will clearly separate the two models.

[1] To quantitatively dissociate the Weighted Additive (WADD) model from Take The Best (TTB), we use an environment with 5 features where the validities are spread but relatively high overall (e.g., [0.90, 0.80, 0.75, 0.70, 0.65]). This allows the creation of 'compensatory' trials where the highest-validity discriminating cue strongly favors one option, but the sum of the remaining lower-validity cues favors the other. TTB will invariably choose the option favored by the single best discriminating cue, completely ignoring the sheer number of lower-validity cues pointing the other way. By contrast, WADD integrates all features weighted by their validities, meaning that a coalition of multiple lower-validity cues will systematically override the single highest-validity cue. The trial pairs pit the top cue against various combinations of the remaining cues to clearly separate the predictions of the two models.

[2] To quantitatively dissociate the Strategy Mixture model (TTB + WADD) from the pure Weighted Additive (WADD) model, we need to examine choice probabilities on 'compensatory' trials where the highest-validity cue points to one option, but the sum of the remaining lower-validity cues strongly points to the other. A pure WADD model consistently and confidently chooses the option favored by the weighted sum. In contrast, the Strategy Mixture model assumes that subjects sometimes use the non-compensatory Take The Best (TTB) heuristic. Consequently, the mixture model predicts a 'pull' toward the option favored by the single best cue, resulting in more moderate choice probabilities (or even a reversal, depending on the mixture weight) compared to the pure WADD model's predictions. By systematically varying the magnitude of the WADD difference while keeping the TTB prediction constant across trials, we can clearly separate the two models.

[3] This design quantitatively dissociates pure WADD from a WADD/TTB mixture by testing the 'squashed asymptote' prediction of the mixture model. We create a gradient of trials where the Weighted Additive difference varies from strongly favoring Option A to strongly favoring Option B. Crucially, on a subset of these trials, the highest-validity cue (which drives TTB) points in the opposite direction of the WADD sum. Pure WADD predicts that choice probability will follow a standard sigmoid function of the WADD score difference, reaching high confidence when the WADD difference is large, regardless of what the top cue says. The mixture model, however, predicts that whenever WADD and TTB conflict, the choice probability for the WADD-favored option will hit a hard ceiling determined by the mixture parameter (wadd_prob), never reaching full confidence. By comparing choice probabilities on strong-conflict vs. strong-agreement trials, we can isolate this ceiling effect predicted uniquely by the competing theory.

[4] To dissociate the Strategy Mixture model (Advocated) from the Equal Weights with High Lapse model (Competing), we exploit the fact that the Competing theory relies exclusively on 'Tallying' (counting positive cues) heavily masked by a high guessing rate (lapse >= 0.8). By designing trials where Tallying predicts a dead heat (equal number of positive cues for both options) but the validities strongly favor one option, the Competing theory is forced to predict exactly a 50/50 choice probability. In contrast, the Advocated theory will predict a strong preference for the option favored by the high-validity cues, driven by both Take-The-Best (TTB) and Weighted Additive (WADD) components. Additionally, we include trials where Tallying strongly opposes the highest-validity cue to further contrast the high-lapse equal-weighting predictions against the structured TTB/WADD mixture.

[5] To quantitatively dissociate the Advocated theory (Equal Weights with High Lapse) from the Competing theory (Strategy Mixture of TTB and WADD), this design creates 'anti-validity' trials where the sheer number of positive cues opposes the highest-validity cues. For example, in Trial 1, Option A is supported by only two cues, but they are the most valid ones (0.95 and 0.90), whereas Option B is supported by three low-validity cues (0.55, 0.55, 0.50). The Advocated theory, which relies on tallying (Equal Weights) heavily masked by guessing, predicts a weak preference for Option B (since 3 > 2). Conversely, the Competing theory predicts a strong preference for Option A, because both TTB (relying on the 0.95 cue) and WADD (1.85 vs 1.60) strongly favor A. Additional trials systematically vary whether EQW predicts a tie while TTB/WADD predict a clear winner, providing a comprehensive test of whether subjects are entirely validity-insensitive as the Advocated theory claims.

[6] To dissociate the Single-Cue Focus model (Advocated) from the Equal Weights model (Competing), both of which assume a high lapse rate, we create trials that pit the single highest-validity cue against the sheer number of positive cues. In some trials, Option A has only the highest-validity cue while Option B has multiple lower-validity cues. The Advocated theory predicts a preference for Option A (driven by the single best cue), whereas the Competing theory predicts a preference for Option B (driven by tallying). We also include trials where the total number of cues is equal, forcing the Competing theory to predict exactly a 50/50 split, while the Advocated theory still predicts a clear preference based on the single best cue.

[7] To quantitatively dissociate the Advocated theory (Equal Weights with High Lapse) from the Competing theory (Extreme Cognitive Noise / Single-Cue Focus), we create trials that independently manipulate the option favored by the single highest-validity cue and the option favored by the raw tally of positive cues. In some trials, the single best cue strongly opposes the sheer number of cues (e.g., Option A has only the best cue, while Option B has 3 or 4 lower-validity cues). The Advocated theory predicts a preference for the option with more positive cues, completely ignoring validities, whereas the Competing theory predicts a preference for the option with the single best cue, completely ignoring the count. We also include tie-breaker trials: trials where the total number of cues is equal but the best cue favors one option (Advocated predicts a tie, Competing predicts a clear winner), and trials where both options share the best cue but one has more total cues (Advocated predicts a clear winner, Competing predicts a tie).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Single-Cue Focus model (Advocated) from the Weak Weighted Additive model (Competing), we exploit the fact that the Advocated theory relies exclusively on the single highest-validity cue, completely ignoring all other features, while the Competing theory integrates all features via a weighted sum. We design 'compensatory' trials where the highest-validity cue points to Option A, but the sum of the remaining lower-validity cues strongly favors Option B. The Advocated theory predicts a preference for Option A, whereas the Competing theory predicts a preference for Option B. Crucially, we also include 'tie-breaker' trials where both options share the exact same value for the highest-validity cue. On these trials, the Advocated theory is completely blind to any differences and predicts an exact 50/50 random guess, whereas the Competing theory still predicts a clear preference driven by the remaining cues.",
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
      0,
      0,
      0
    ],
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
      0,
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
      0,
      1,
      1
    ],
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
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3625,
  "candidates_token_count": 442,
  "total_token_count": 5503
}
```
