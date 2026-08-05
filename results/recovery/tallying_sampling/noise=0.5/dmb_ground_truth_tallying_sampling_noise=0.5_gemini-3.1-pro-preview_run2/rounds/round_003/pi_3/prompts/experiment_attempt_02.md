# experiment_attempt_02

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
**Description:** People make decisions between options using a Tallying (Equal Weight) heuristic. Instead of weighting features by their validities or relying on a single discriminating cue, decision-makers simply count the number of positive features for each option. They choose the option with the higher total count, treating all cues as equally important. When counts are tied, they guess. Response noise is modeled via a softmax over the tally scores and a uniform lapse rate. The choice is relatively noisy, preventing the strategy from becoming perfectly deterministic even when one option has a clear tally advantage.

**Parameters:**
- beta: [0.1, 1.5]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: compute the sum of features for each option (equal weighting)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
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


## COMPETING THEORY
**Description:** Tally-Gated Validity Bias: Decision-makers primarily rely on a Tallying heuristic, simply counting the number of positive features for each option. If the tally results in a tie, the decision process abruptly concludes and they guess randomly, without falling back on cue validities. However, if there is a difference in tally scores, the strength of their preference is modulated by the explicit cue validities. This means validities act as a secondary confidence-adjuster rather than a tie-breaker, explaining why validity bias appears in overall choices but is absent when options have an equal number of positive features.

**Parameters:**
- beta: [0.1, 2.0]
- epsilon: [0.0, 0.5]
- w_val: [0.0, 0.6]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    # Primary strategy: Tallying
    tally_scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_val = float(parameters["w_val"])
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # If tallying results in a tie, guess randomly (no validity tie-breaking)
    if tally_scores[0] == tally_scores[1]:
        p_core = np.ones(2) / 2.0
    else:
        # If there is a tally difference, validities modulate the response strength
        val_scores = stim @ validities
        scores = (1.0 - w_val) * tally_scores + w_val * val_scores
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
[0] To cleanly dissociate Take The Best (TTB) from the Weighted Additive rule (WADD), this design employs a highly skewed validity distribution where the most valid cue strongly discriminates, but the sum of the remaining cues outweighs it. By presenting options where one alternative dominates on the highest available validity cue while the other alternative dominates on all lower validity cues, TTB will deterministically pick the former (ignoring the compensatory evidence) while WADD will pick the latter (integrating all features). The trials include variations where the highest overall cue is tied, forcing the heuristics to evaluate the remaining cues in the same compensatory vs. non-compensatory conflict.

[1] To quantitatively dissociate WADD from TTB, this design varies the compensatory evidence on lower-validity cues while keeping the highest-validity discriminating cue constant. TTB makes its decision based solely on the first discriminating cue; thus, its predicted choice probability (and direction) remains completely flat across these trials. In contrast, WADD integrates all features, so its choice probabilities will scale continuously with the net difference in weighted sums, shifting from a strong preference for A, to indifference, to a strong preference for B. This tests WADD's core prediction that choice confidence scales with the cardinal magnitude of the weighted evidence difference, rather than just the ordinal rank of the best cue.

[2] To quantitatively dissociate Tallying (Equal Weight) from the Weighted Additive (WADD) rule, we use a 5-feature design with a skewed set of validities. In this design, some trials feature an option with fewer, but highly valid, cues pitted against an option with more, but less valid, cues. Tallying simply counts the number of positive features and will predict that the option with more features is chosen. WADD, conversely, weights each feature by its validity and will predict that the option with higher total validity is chosen. By including trials where Tallying predicts a strong preference for one option while WADD predicts the opposite, as well as trials where one model predicts a tie while the other predicts a clear winner, we can distinctly identify which strategy subjects are employing.

[3] To quantitatively dissociate WADD from Tallying (Equal Weight), this design pairs options such that the two models produce divergent predictions. We use a 5-feature setup with validities strategically chosen so that a small number of highly valid cues can perfectly balance a larger number of less valid cues. The trials include cases where WADD predicts a tie but Tallying predicts a strong preference (because one option has more positive features), cases where Tallying predicts a tie (equal number of positive features) but WADD predicts a strong preference (due to higher validity weights), and cases where the two models predict completely opposite choices. This multi-pronged dissociation ensures that the models are distinguishable not just by overall accuracy, but by their trial-by-trial choice directions and confidence.

[4] To quantitatively dissociate pure Tallying from Tallying with Validity Bias, this design orthogonally manipulates the difference in feature counts (Tally Difference) and the difference in validity-weighted sums (Validity Difference). Pure Tallying predicts that choice probabilities are determined entirely by the Tally Difference, resulting in flat, step-like choice functions where variations in Validity Difference have zero effect (e.g., all Tally Ties yield 50/50 choices). In contrast, Tallying with Validity Bias predicts a graded response: within any given Tally Difference (including ties), the choice probability will shift systematically according to the Validity Difference. By including trials where Tally and Validity conflict, align, or tie, we can isolate the precise contribution of the validity bias.

[5] To cleanly dissociate pure Tallying from Tallying with Validity Bias, this design focuses on trials where the two models predict qualitatively different behavior. Specifically, we include 'Tally Tie' trials where both options have the same number of positive features, but one option's features have much higher validities. Pure Tallying predicts exactly 50/50 choice probabilities on these trials, whereas Tallying with Validity Bias predicts a systematic preference for the option with higher validities. Furthermore, we include 'Conflict' trials where one option has more positive features (favored by Tallying) but the other option has fewer, highly valid features (favored by the Validity component). Comparing these to 'Alignment' trials reveals the graded influence of validity weights that the pure Tallying model completely ignores.

[6] To quantitatively dissociate Pure Tallying from Tally-Gated Validity Bias, we must focus on trials where the tally scores are NOT tied. Both theories predict exactly 50/50 guessing when the options have an equal number of positive features (Tally Ties), making them indistinguishable on those trials. However, when there is a tally difference (e.g., Option A has 3 features and Option B has 2), Pure Tallying predicts a constant choice probability determined solely by the +1 tally advantage. In contrast, Tally-Gated Validity Bias predicts that the choice probability will be modulated by the validities of those features. By fixing the tally difference at +1 (or -1) but varying the validity difference across trials—ranging from strong alignment (A has more features AND higher validities) to opposition (A has more features but B has the higher validities)—we can distinctly test if validities modulate choice only in the presence of a tally difference.

[7] To quantitatively dissociate Pure Tallying from Tally-Gated Validity Bias, we must focus on trials where the tally scores are NOT tied, because both theories predict exactly 50/50 guessing on tally ties. When there is a tally difference (e.g., Option A has 3 features and Option B has 2), Pure Tallying predicts a constant choice probability determined solely by the +1 tally advantage, completely ignoring validities. In contrast, Tally-Gated Validity Bias predicts that the choice probability will be modulated by the validities of those features. By fixing the tally difference at +1 but varying the validity difference across trials—ranging from strong alignment (A has more features AND higher validities) to opposition (A has more features but B has the higher validities)—we can distinctly test if validities modulate choice only in the presence of a tally difference. Pure Tallying will show a flat line across these trials, while the Tally-Gated model will show a graded slope.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To cleanly dissociate Pure Tallying from Tally-Gated Validity Bias, we focus on trials where the tally scores are strictly unequal (e.g., Option A has 3 features, Option B has 2 features). By utilizing a highly skewed validity distribution, we can create trial pairs where the +1 tally advantage for Option A is pitted against varying validity differences. Pure Tallying predicts a constant, positive choice probability for Option A across all these trials because the tally difference remains fixed at +1. In contrast, the Tally-Gated Validity Bias model predicts a graded response: as Option B's features become more valid compared to Option A's, the preference for A should diminish, and can even reverse if the validity difference opposes the tally difference strongly enough. Including tally-tie trials serves as a baseline check, as both models predict exactly 50/50 guessing when the number of positive features is equal.",
  "validities": [
    1.0,
    0.9,
    0.6,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
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
      0,
      1,
      1,
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
      0
    ],
    [
      1,
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
      0,
      0,
      0,
      0
    ],
    [
      0,
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3469,
  "candidates_token_count": 571,
  "total_token_count": 5959
}
```
