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
**Description:** Weakest Link (Worst-Case Focus): Decision-makers evaluate an option primarily by averaging the subjective validities of its present features, but they apply a disproportionate penalty based on the option's 'weakest link' (the lowest validity active feature). Specifically, the penalty is proportional to the gap between the option's best feature and its weakest feature. This cognitive mechanism reflects a worst-case focus where attention is drawn to the most inferior attribute, naturally predicting both the dilution effect (adding weak features reduces the mean and increases the weakest-link penalty) and variance aversion (options with tighter feature clusters have smaller weakest-link penalties).

**Parameters:**
- gamma: [0.1, 10.0]
- lambda_weak: [0.0, 10.0]
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
    lambda_weak = float(parameters["lambda_weak"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        mean_w = np.mean(active_w)
        min_w = np.min(active_w)
        max_w = np.max(active_w)
        
        # Averaging baseline with a penalty based on the weakest link's distance from the best feature
        return mean_w - lambda_weak * (max_w - min_w)
        
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
**Description:** Relative Threshold Heuristic with Bounded Regret: Decision-makers evaluate options by directly comparing them on a cue-by-cue basis, focusing only on distinguishing features (unique advantages and disadvantages). To capture the dilution effect without relying on a holistic average, decision-makers apply a threshold (`theta`) to their unique advantages. Low-validity unique advantages that fall below this threshold actually penalize the option. Furthermore, the regret penalty for unique disadvantages is also thresholded (`theta_regret`), ensuring that decision-makers only penalize the absence of truly critical (high-validity) features. This prevents options with multiple medium-validity advantages from being overly punished for missing a single high-validity feature.

**Parameters:**
- gamma: [0.01, 5.0]
- delta: [0.01, 5.0]
- lambda_regret: [0.0, 2.0]
- theta: [0.0, 1.0]
- theta_regret: [0.0, 1.0]
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
    delta = float(parameters["delta"])
    lambda_regret = float(parameters["lambda_regret"])
    theta = float(parameters["theta"])
    theta_regret = float(parameters["theta_regret"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights for presence and absence
    w_pos = val ** gamma
    w_neg = val ** delta
    
    def get_score(x, other_x):
        unique_adv = (x > 0) & (other_x == 0)
        unique_disadv = (x == 0) & (other_x > 0)
        
        # Sum of unique advantages, penalized by a threshold (theta)
        # This allows low-validity features to have a net negative impact
        adv_score = np.sum(w_pos[unique_adv] - theta) if np.any(unique_adv) else 0.0
        
        # Regret penalty for unique disadvantages, also thresholded
        disadv_score = lambda_regret * np.sum(np.maximum(0.0, w_neg[unique_disadv] - theta_regret)) if np.any(unique_disadv) else 0.0
        
        return adv_score - disadv_score
        
    score_a = get_score(a, b)
    score_b = get_score(b, a)
    
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

[11] To quantitatively dissociate the Anchor and Adjust model from the Averaging model, we exploit cases where both options have the exact same average validity but differ in their maximum validity (the anchor). The Averaging model evaluates options strictly by the mean of their positive features, predicting complete indifference (50/50 choice) whenever the averages are equal. In contrast, the Anchor and Adjust model evaluates options by anchoring on the single highest-validity feature and adjusting based on the others. By pairing options with identical mean validities but different anchors (e.g., one option has a very high and very low validity feature, while the other has moderately high validity features or a single feature equal to the mean), we create a scenario where Averaging predicts no preference, but Anchor and Adjust predicts systematic preferences driven by the anchor disparity.

[12] This design quantitatively dissociates the Variance Aversion (Configural Weighting) model from the Anchor and Adjust model by exploiting their fundamentally different treatments of feature dispersion. The Anchor and Adjust model evaluates an option by anchoring on its highest-validity positive feature and adjusting based on the sum of the remaining features. In contrast, the Variance Aversion model computes the mean of the positive features and applies a penalty proportional to their variance. We construct trials that pit an option with a very high anchor but high variance (e.g., one very high and one very low validity feature) against an option with a lower anchor but tightly clustered, moderately high features (low variance). By varying the difference in means and the severity of the variance disparity across trials, we can distinguish the variance penalty (which strongly suppresses the high-dispersion option) from the anchoring effect (which elevates the high-dispersion option due to its superior maximum validity).

[13] To quantitatively dissociate the Anchor and Adjust model from the Variance Aversion model, we systematically manipulate the single highest-validity feature (the anchor) while keeping the remaining features identical across the two options. The Anchor and Adjust model evaluates an option by taking its maximum validity feature and adjusting it based on the others; thus, if Option A and Option B share the exact same set of lower-validity features but Option A has a higher anchor, Anchor and Adjust will universally prefer Option A, as improving the anchor strictly increases the evaluation. In contrast, the Variance Aversion model computes the mean of the validities and applies a penalty proportional to their variance. By increasing the highest validity feature while holding the lower ones constant, the variance of the feature set increases. If the variance penalty parameter is sufficiently high, the Variance Aversion model will paradoxically prefer Option B (the option with the lower maximum validity) because its features are more tightly clustered, leading to a smaller configural penalty. This provides a sharp qualitative divergence.

[14] To quantitatively dissociate the Variance Aversion (Configural Weighting) model from the Weakest Link (Worst-Case Focus) model, we exploit their differing penalties for feature dispersion. Both models compute the mean validity of active features but apply different penalties: Variance Aversion penalizes by the variance of the active validities, whereas Weakest Link penalizes by the range (max - min). We design pairs where Option A has a smaller range but higher variance than Option B, while maintaining the same mean validity. In such cases, the Weakest Link model strictly prefers Option A (due to its smaller range penalty), whereas the Variance Aversion model strictly prefers Option B (due to its smaller variance penalty). We also include trials where the models predict indifference versus a strict preference, providing a clear map of their divergent predictions.

[15] This design quantitatively dissociates the Weakest Link (Worst-Case Focus) model from the Variance Aversion (Configural Weighting) model. Both models evaluate options based on the mean validity of their active features but apply different penalties: Weakest Link penalizes by the range of active validities (max - min), whereas Variance Aversion penalizes by their variance. By creating a set of features where an option can have a large range but a small variance (e.g., extreme max and min values but many features clustered tightly at the mean) compared to an option with a smaller range but larger variance (fewer features, moderately far from the mean), we can create trials where the models predict opposite preferences.

[16] To quantitatively dissociate the Advocated Theory (Relative Threshold Heuristic) from the Competing Theory (Weakest Link), we exploit their fundamentally different treatment of shared features. The Advocated Theory compares options cue-by-cue, meaning any feature possessed by both options is immediately canceled out and has zero effect on the choice. In contrast, the Competing Theory evaluates each option holistically by computing the mean and range (weakest link penalty) of its active features; thus, adding a shared feature alters both the mean and the range penalty of both options, often asymmetrically. We design base trials comparing unique advantages, and then create derivative trials by adding shared high-validity or shared low-validity features. The Advocated Theory predicts identical choice probabilities across these variations, whereas the Competing Theory predicts massive shifts in preference due to the changing weakest-link penalties.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates the Weakest Link (Worst-Case Focus) model from the Relative Threshold Heuristic by contrasting holistic feature integration with cue-by-cue advantage accumulation. The Relative Threshold model evaluates options by summing their unique advantages (minus a threshold) and subtracting a regret penalty for unique disadvantages. It generally favors options with a greater number of positive features, provided their validities exceed the threshold. In contrast, the Weakest Link model evaluates options by the mean of their active validities minus a penalty for the range (max - min) of those validities. We design trials where Option A has multiple features (yielding a high sum of unique advantages but a low mean and large range) while Option B has fewer, tightly clustered or single high-validity features (yielding a lower sum but a higher mean and smaller range). The Relative Threshold model will prefer Option A due to the accumulation of advantages, whereas the Weakest Link model will strongly prefer Option B due to its higher mean and avoidance of the weakest-link penalty.",
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
      1,
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      1,
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
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 5664,
  "candidates_token_count": 458,
  "total_token_count": 8314
}
```
