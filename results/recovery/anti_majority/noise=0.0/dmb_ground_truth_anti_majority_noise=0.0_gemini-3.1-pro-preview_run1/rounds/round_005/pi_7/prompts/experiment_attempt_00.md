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
**Description:** Anchor and Adjust: Decision-makers evaluate options by first identifying the highest-validity positive feature, which serves as an anchor. Subsequent positive features adjust this anchored evaluation downwards if their validities are lower than the anchor. This explains the 'less is more' dilution effect, as adding low-validity features drags the overall evaluation down, while naturally predicting that an option with only high-validity features will be preferred over one that adds additional low-validity features (strict subset preference).

**Parameters:**
- gamma: [0.1, 10.0]
- alpha: [0.0, 50.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # The highest validity positive feature is the anchor
        anchor = np.max(active_w)
        idx = np.argmax(active_w)
        
        # Remaining features adjust the evaluation
        other_w = np.delete(active_w, idx)
        if len(other_w) == 0:
            return anchor
            
        # Weighted average of anchor and adjusting features
        return (anchor + alpha * np.sum(other_w)) / (1.0 + alpha * len(other_w))
        
    score_a = get_score(a)
    score_b = get_score(b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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
**Description:** Averaging Model: Decision-makers evaluate options not by summing the evidence (additive) or relying on a single best cue (lexicographic), but by averaging the subjective validities of all positive features present in an option. This mechanism naturally accounts for the 'dilution effect', where adding a positive but low-validity feature to an option can actually decrease its overall subjective value and choice probability.

**Parameters:**
- gamma: [1.0, 10.0]
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities to subjective weights
    w = val ** gamma
    
    # Count positive features
    k_a = np.sum(a)
    k_b = np.sum(b)
    
    # Calculate average validity of positive features (0 if no positive features)
    score_a = np.sum(w * a) / k_a if k_a > 0 else 0.0
    score_b = np.sum(w * b) / k_b if k_b > 0 else 0.0
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from Tallying, we use a 5-feature design where the options are constructed such that the single highest-validity discriminating cue favors one option (the choice predicted by TTB), while a larger number of lower-validity cues favor the other option (the choice predicted by Tallying). By varying which cue is the highest-validity discriminator and the margin of the tallying difference, we can robustly separate the one-reason decision making of TTB from the unweighted evidence accumulation of Tallying.

[1] To quantitatively dissociate Tallying from Take The Best (TTB), this design uses 5 features with descending validities. We construct trials where the single highest-validity discriminating cue strongly favors one option (the choice predicted by TTB), while a larger number of lower-validity cues favor the other option (the choice predicted by Tallying). We also include trials where the options tie under Tallying (equal number of winning features) but have a strict preference under TTB, as well as trials where the top cues are tied, forcing TTB to rely on secondary cues while Tallying continues to count all feature wins equally.

[2] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we use a 5-feature design with a linear spread of validities. The trials are constructed to pit the single highest-validity discriminating cue against a coalition of multiple lower-validity cues. TTB, being a non-compensatory 'one-reason' heuristic, will consistently choose the option favored by the single highest-validity discriminating cue, ignoring all others. WADD, being compensatory, integrates all cues and (unless the exponent gamma is extremely high) will often favor the option supported by the combination of multiple lower-validity cues. By varying the position of the discriminating cue and the number of opposing cues, we can map out the precise compensatory threshold and robustly distinguish TTB's lexicographic choice pattern from WADD's integration.

[3] To quantitatively dissociate the Weighted Additive (WADD) model from Take The Best (TTB), this design uses 5 features with linearly decreasing validities. The trials are constructed to systematically pit the single highest-validity discriminating cue against a coalition of multiple lower-validity cues. TTB, being non-compensatory, will consistently choose the option favored by the single highest-validity discriminating cue, ignoring the rest. WADD, being compensatory, integrates all cues and will often favor the option supported by the combination of multiple lower-validity cues (depending on the value of gamma). By varying the position of the discriminating cue (e.g., cue 1 vs. cues 2-5, or cue 2 vs. cues 3-5) and the number of opposing cues, we can robustly distinguish TTB's lexicographic choice pattern from WADD's integration.

[4] To quantitatively dissociate the Strategy Repertoire (Mixture of TTB and WADD) from the parameterized WADD model (which uses an exponent 'gamma' to interpolate between Tallying and TTB), we use a 4-feature design with linearly spaced validities. The trials are designed to create varying degrees of conflict between the highest-validity cue and combinations of lower-validity cues. The competing WADD model applies a non-linear transformation (gamma exponent) to the weights before passing them through a single softmax function. In contrast, the Strategy Repertoire model computes a linear combination of two separate softmax probabilities (one from pure TTB, one from linear WADD). Because the softmax function is non-linear, a mixture of two distinct softmax distributions produces a probability profile across these diverse trade-off trials that cannot be perfectly mimicked by a single softmax over gamma-exponentiated weights. By analyzing the choice probabilities across trials where TTB and linear WADD strongly disagree versus where they weakly disagree, we can distinguish the mixture of strategies from a single unified compensatory strategy with non-linear weights.

[5] To quantitatively dissociate the parameterized WADD model (which uses an exponent 'gamma' to non-linearly scale validities before a single softmax) from the Strategy Repertoire model (which mixes probabilities from independent TTB and linear WADD processes), we use a 5-feature design. The trials systematically vary the margin of victory for the linear WADD component while keeping the TTB component constant. Because the Strategy Repertoire is a linear mixture of two softmax distributions, its choice probabilities will show a distinctive additive baseline from the pure TTB component, making it less sensitive to small changes in WADD scores compared to the WADD-gamma model, which integrates all evidence into a single non-linear softmax calculation.

[6] To quantitatively dissociate the Averaging model from the Weighted Additive (WADD) model, we exploit the 'dilution effect' unique to the Averaging model. In the Averaging model, the subjective value of an option is the average validity of its positive features. This leads to a counterintuitive prediction: adding a positive feature with a low validity to an option that already has high-validity features will *decrease* its overall score. In contrast, WADD computes a sum, meaning that adding any positive feature strictly increases the option's score. We design trials where Option A has a small number of high-validity features, and Option B has the exact same high-validity features PLUS several additional low-validity features. WADD will universally prefer Option B (more evidence), while the Averaging model will prefer Option A (higher average evidence quality).

[7] To quantitatively dissociate the Weighted Additive (WADD) model from the Averaging model, this design systematically pits the sum of evidence against the average quality of evidence across options with different numbers of positive features. In each trial, Option A possesses fewer but higher-validity features, whereas Option B possesses a larger number of lower-validity features. The WADD model, which integrates evidence additively, consistently predicts a preference for Option B because the total sum of subjective weights is greater. Conversely, the Averaging model, which computes the mean validity of positive features, consistently predicts a strong preference for Option A because the inclusion of lower-validity features in Option B dilutes its average value. By avoiding the simple 'A is a subset of B' dilution effect explored previously and instead contrasting entirely distinct feature sets, we provide a robust test of additive integration versus averaging.

[8] To quantitatively dissociate the pure Averaging model (Advocated) from the Lexicographic with Fallback model (Competing), we exploit their differing reliance on the single best discriminating cue. The Competing model uses a threshold ('theta') on the highest-validity discriminating cue to probabilistically trigger a Take-The-Best (TTB) choice; if triggered, it ignores all other features. The pure Averaging model, however, always evaluates options based on the mean validity of their positive features. We design trials where Option A possesses the highest-validity discriminating cue (strongly triggering TTB in the Competing model) but is paired with low-validity features that heavily 'dilute' its overall average. Option B lacks the top cue but possesses one or more moderately high-validity features, resulting in a higher average validity. The Competing model will frequently choose the option with the top cue due to its TTB component, while the pure Averaging model will consistently prefer the option with the higher mean validity.

[9] To quantitatively dissociate the Advocated Theory (Lexicographic with Fallback) from the Competing Theory (Averaging Model), we design trials where the highest-validity discriminating cue strongly favors Option A, but Option A also possesses several low-validity features. Option B lacks the highest-validity cue but consists solely of moderately high-validity features. The Advocated Theory, due to its TTB component triggered by the highly valid cue, will frequently choose Option A. In contrast, the Competing Theory, which evaluates options purely by the average validity of their positive features, will consistently prefer Option B because Option A's average is heavily diluted by its low-validity features.

[10] To quantitatively dissociate the Averaging Model from the Anchor and Adjust Model, we exploit their different weighting of features. While both models predict a 'dilution effect' (where adding lower-validity features reduces an option's overall value), they differ fundamentally in how they integrate multiple features. The Anchor and Adjust model places disproportionate weight on the single highest-validity positive feature (the anchor), with all other features merely adjusting this anchor. The Averaging model weighs all positive features equally when computing the mean. We design trials where Option A has a very high-validity anchor but is dragged down by multiple low-validity features, while Option B lacks the top anchor but consists of tightly clustered, moderately high-validity features. The Anchor and Adjust model will often prefer Option A due to the strong anchor dominating the evaluation, whereas the Averaging model will consistently prefer Option B due to its higher overall mean validity.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Anchor and Adjust model from the Averaging model, we exploit cases where both options have the exact same average validity but differ in their maximum validity (the anchor). The Averaging model evaluates options strictly by the mean of their positive features, predicting complete indifference (50/50 choice) whenever the averages are equal. In contrast, the Anchor and Adjust model evaluates options by anchoring on the single highest-validity feature and adjusting based on the others. By pairing options with identical mean validities but different anchors (e.g., one option has a very high and very low validity feature, while the other has moderately high validity features or a single feature equal to the mean), we create a scenario where Averaging predicts no preference, but Anchor and Adjust predicts systematic preferences driven by the anchor disparity.",
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
      0,
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
      0,
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
      0,
      1,
      0,
      1,
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
      1,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4273,
  "candidates_token_count": 516,
  "total_token_count": 6708
}
```
