# experiment_attempt_01

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
**Description:** Feature Coherence and Anchoring (Unique Features Only): Decision-makers simplify choices by first cancelling out shared features, then evaluating the remaining unique features. They expect high-quality options to be coherent. A gap penalty is applied if an option's best unique feature (the anchor) is disconnected from its next best unique feature. However, to avoid unfairly penalizing naturally sparse fallback options, this gap penalty is only applied if the option has a dense profile (>2 unique features) OR if it boasts the absolute highest-validity feature in the environment (index 0) but fails to back it up. A spread penalty is also applied to unique features to penalize internal conflict.

**Parameters:**
- gamma: [0.1, 10.0]
- rho: [0.0, 1.0]
- lambda_iso: [0.0, 5.0]
- lambda_penalty: [0.0, 10.0]
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
    rho = float(parameters["rho"])
    lambda_iso = float(parameters["lambda_iso"])
    lambda_penalty = float(parameters["lambda_penalty"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities
    w = (val ** gamma) - rho
    
    # Identify shared and unique features
    shared = (a > 0) & (b > 0)
    unique_a = np.where((a > 0) & ~shared)[0]
    unique_b = np.where((b > 0) & ~shared)[0]
    
    def get_score(unique_idx):
        if len(unique_idx) == 0:
            return 0.0
            
        base_score = np.sum(w[unique_idx])
        
        # Isolation / Gap penalty for the anchor calculated strictly on unique features.
        # Applied if the profile is dense (>2 unique features) OR 
        # if it's a 2-feature profile that claims the absolute best feature (index 0).
        gap_penalty = 0.0
        if len(unique_idx) > 2 or (len(unique_idx) == 2 and unique_idx[0] == 0):
            anchor = unique_idx[0]
            next_best = unique_idx[1]
            gap = next_best - anchor - 1
            if gap > 0:
                gap_penalty = lambda_iso * gap
                
        # Spread penalty on unique features
        conflict = 0.0
        if len(unique_idx) > 1:
            conflict = lambda_penalty * (np.max(w[unique_idx]) - np.min(w[unique_idx]))
            
        return base_score - gap_penalty - conflict
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
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
**Description:** Thresholded Unique Features with Spread Penalty: Decision-makers simplify choices by cancelling out shared features, then evaluate the unique features relative to a subjective validity threshold. Features above the threshold provide positive evidence, while those below act as penalties. These values are integrated additively, but options with multiple unique features suffer a conflict penalty proportional to the spread (max - min) of their thresholded validities. This penalizes options with a wide variance in their unique features while strictly preserving shared-feature cancellation.

**Parameters:**
- gamma: [0.1, 10.0]
- rho: [0.0, 1.0]
- lambda_penalty: [0.0, 10.0]
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
    rho = float(parameters["rho"])
    lambda_penalty = float(parameters["lambda_penalty"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Transform validities and apply subjective threshold
    v_trans = val ** gamma
    w = v_trans - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # Additive integration of thresholded evidence
        base_score = np.sum(active_w)
        
        # Spread penalty applied if there are multiple unique features
        if len(active_w) > 1:
            conflict_penalty = lambda_penalty * (np.max(active_w) - np.min(active_w))
            return base_score - conflict_penalty
            
        return base_score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
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

[17] This design quantitatively dissociates the Weakest Link (Worst-Case Focus) model from the Relative Threshold Heuristic by contrasting holistic feature integration with cue-by-cue advantage accumulation. The Relative Threshold model evaluates options by summing their unique advantages (minus a threshold) and subtracting a regret penalty for unique disadvantages. It generally favors options with a greater number of positive features, provided their validities exceed the threshold. In contrast, the Weakest Link model evaluates options by the mean of their active validities minus a penalty for the range (max - min) of those validities. We design trials where Option A has multiple features (yielding a high sum of unique advantages but a low mean and large range) while Option B has fewer, tightly clustered or single high-validity features (yielding a lower sum but a higher mean and smaller range). The Relative Threshold model will prefer Option A due to the accumulation of advantages, whereas the Weakest Link model will strongly prefer Option B due to its higher mean and avoidance of the weakest-link penalty.

[18] To quantitatively dissociate the Advocated Theory (Unique Features Weakest-Link) from the Competing Theory (Weakest Link), we exploit their fundamentally different treatment of shared features. The Advocated Theory hypothesizes that decision-makers first cancel out any features shared by both options, evaluating each option solely on its unique features (computing their mean and weakest-link penalty). The Competing Theory evaluates options holistically, computing the mean and weakest-link penalty across all active features, regardless of whether they are shared. We design base trials where Option A has a superior unique feature and is preferred by both models. We then create derivative trials by adding a shared low-validity feature to both options. Because the Advocated Theory cancels this shared feature, its preference for Option A remains unchanged. However, for the Competing Theory, adding the shared low-validity feature asymmetrically increases the weakest-link penalty (the range between the best and worst feature) for Option A more than for Option B, causing a preference reversal where Option B is now favored.

[19] To quantitatively dissociate the Advocated Theory (Weakest Link) from the Competing Theory (Unique Features Weakest-Link), we exploit their fundamentally different treatment of shared features. The Competing Theory cancels out features shared by both options, evaluating each option solely on its unique features. Therefore, adding a shared feature to any pair of options will not change the Competing Theory's predicted choice probabilities. In contrast, the Advocated Theory evaluates options holistically across all active features. Adding a shared low-validity feature alters both the mean validity and the range (weakest link penalty) of the options. Because the range penalty is asymmetrical depending on the option's best unique feature, adding a shared feature can cause a complete preference reversal in the Advocated Theory while leaving the Competing Theory's predictions perfectly constant. The trials are grouped into base pairs with unique advantages, followed by derivative pairs that add various shared features to systematically map this divergence.

[20] To quantitatively dissociate the Advocated Theory (Unique Features Weakest-Link) from the Competing Theory (Strict Averaging of Active Features), we exploit their fundamentally different treatments of shared features and feature dispersion. The Advocated Theory cancels out shared features and applies a weakest-link penalty (max - min) to the remaining unique features. The Competing Theory computes a simple average of all active features, regardless of whether they are shared. We design base trials where Option A has two unique features with a large range (but the same mean as Option B) and Option B has a single unique feature (zero range). The Competing Theory predicts indifference, while the Advocated Theory strongly prefers Option B due to the weakest-link penalty on A. We then create derivative trials by adding a shared low-validity feature. Because the Competing Theory averages all features, the shared low-validity feature dilutes Option B's average more than Option A's (since B has fewer features), causing the Competing Theory to prefer Option A. Meanwhile, the Advocated Theory cancels the shared feature and continues to strictly prefer Option B.

[21] To quantitatively dissociate the Advocated Theory (Strict Averaging of Active Features) from the Competing Theory (Unique Features Weakest-Link), we exploit their fundamentally different treatments of shared features and feature dispersion. The Competing Theory cancels out any shared features and applies a 'weakest-link' penalty based on the range of the remaining unique features. The Advocated Theory evaluates options holistically by computing the simple mean of all active features, regardless of whether they are shared. We design base trials where Option A has a single high-validity unique feature, and Option B has two unique features with the same mean but a larger range. The Competing Theory strongly prefers Option A because Option B suffers a weakest-link penalty. We then create derivative trials by adding shared low-validity features to both options. Because the Competing Theory cancels shared features, its preference for Option A remains unchanged. However, for the Advocated Theory, adding shared low-validity features dilutes Option A's average much more severely (since it has fewer features) than Option B's average, causing a clear preference reversal where Option B is now favored.

[22] To quantitatively dissociate the Advocated Theory (Unique Features Weakest-Link) from the Competing Theory (Unique Features Diminishing Utility with Reference Point), we exploit two fundamental differences in how they evaluate unique features after cancellation. First, the Competing Theory integrates features additively, meaning a coalition of moderately high-validity features can overcome a single higher-validity feature; the Advocated Theory averages unique features and applies a weakest-link penalty, strictly preferring the single higher-validity feature. Second, the Competing Theory evaluates features relative to a reference point (rho), where low-validity features carry negative utility, whereas the Advocated Theory assigns positive value to any feature regardless of validity. By contrasting trials where one option accumulates multiple good features against an option with one superior feature, and trials where one option possesses a unique low-validity feature while the other has no unique features, we generate sharp, bidirectional preference reversals between the two models.

[23] To quantitatively dissociate the Advocated Theory (Unique Features Diminishing Utility with Reference Point) from the Competing Theory (Unique Features Weakest-Link), we exploit their fundamentally different evaluations of unique features after cancellation. First, the Advocated Theory evaluates validities relative to a reference point (rho), meaning low-validity features can carry negative utility. Thus, an option with a single low-validity unique feature can be evaluated as worse than an option with no unique features at all. The Competing Theory, however, always assigns positive value to any feature, strictly preferring an option with a low-validity feature over one with none. Second, the Advocated Theory integrates unique features additively (subject to diminishing returns), allowing a coalition of lower-validity features to overcome a single higher-validity feature. The Competing Theory averages unique features and applies a weakest-link penalty based on their range, strongly penalizing coalitions and typically favoring the single higher-validity feature. By contrasting pairs where one option has only a single low-validity feature against an empty option, and pairs pitting a single high-validity feature against a coalition of multiple lower-validity features, we can cleanly separate the reference-point and additive mechanisms of the Advocated Theory from the averaging and range-penalty mechanisms of the Competing Theory.

[24] To quantitatively dissociate the Advocated Theory (Unique Features Weakest-Link) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their fundamentally different integration rules for unique features after shared features are canceled. The Advocated Theory averages unique features and applies a weakest-link penalty, strictly preferring a single high-validity feature over a coalition of multiple moderately high-validity features because the coalition yields a lower mean and a higher range penalty. The Competing Theory sums unique features relative to a subjective threshold (rho). This additive integration allows a coalition of moderate features to easily overcome a single high-validity feature, provided they exceed the threshold. Furthermore, the Competing Theory's threshold implies that low-validity features can have negative utility, causing it to potentially prefer an option with no unique features over one with a single low-validity unique feature. The Advocated Theory, however, always assigns positive utility to any feature and will strictly prefer the option with the low-validity feature. By contrasting single high-validity features against coalitions of lower-validity features, and single low-validity features against empty feature sets, we can robustly separate the averaging/weakest-link mechanism from the thresholded additive mechanism.

[25] To quantitatively dissociate the Advocated Theory (Thresholded Unique Features with Spread Penalty) from the Competing Theory (Unique Features Weakest-Link), we exploit their fundamentally different treatments of unique features after cancellation. The Advocated Theory integrates features additively but subjects them to a validity threshold (rho), meaning low-validity features can act as penalties (negative utility). In contrast, the Competing Theory evaluates options by the average validity of their unique features and always treats any feature as positive evidence. We construct trials that contrast an option with no unique features against an option with a single low-validity unique feature; the Advocated Theory will often prefer the empty option (due to the low-validity feature acting as a penalty), whereas the Competing Theory will strictly prefer the option with the single feature. Furthermore, we pit a single high-validity feature against a coalition of multiple moderately high-validity features. The additive nature of the Advocated Theory allows the coalition to overcome the single best feature, while the averaging nature and weakest-link penalty of the Competing Theory will strongly favor the single high-validity feature.

[26] To quantitatively dissociate the Advocated Theory (Strict Diminishing Returns on Absolute Subjective Utility) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their divergent integration mechanisms for unique features. Both models cancel shared features and evaluate the remaining unique features relative to a subjective threshold. However, the Advocated Theory ranks these features by their absolute utility and heavily discounts subsequent features (diminishing returns). The Competing Theory integrates features additively but applies a penalty proportional to the spread (max - min) of their validities. We design trials pitting a single high-validity feature against a coalition of multiple moderately high-validity features with zero or small spread. The Competing Theory, being additive, will strongly prefer the coalition. The Advocated Theory, due to strict diminishing returns, will heavily discount the coalition and prefer the single high-validity feature. We also include trials with large spreads to trigger the Competing Theory's spread penalty, causing it to penalize options that the Advocated Theory evaluates more favorably.

[27] To quantitatively dissociate the Advocated Theory (Thresholded Unique Features with Spread Penalty) from the Competing Theory (Strict Diminishing Returns on Absolute Subjective Utility), we exploit their divergent integration mechanisms for unique features. Both models cancel shared features and evaluate the remaining unique features relative to a subjective threshold. However, the Advocated Theory integrates features additively but applies a penalty proportional to the spread (max - min) of their validities. The Competing Theory ranks these features by their absolute utility and heavily discounts subsequent features (diminishing returns). We design trials pitting a coalition of tightly clustered, moderately high-validity features against a single higher-validity feature. The Advocated Theory, being additive with a small spread penalty, will strongly prefer the coalition. The Competing Theory, due to strict diminishing returns, will heavily discount the coalition and prefer the single high-validity feature. Conversely, we pit an option with a very large spread (one very high and one very low validity feature) against an option with tightly clustered moderate features or a single moderate feature. The Advocated Theory's spread penalty will heavily penalize the large-spread option, while the Competing Theory will prefer it simply because its highest-validity feature dominates the evaluation.

[28] To quantitatively dissociate the Advocated Theory (Variance Aversion) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their differing penalties for feature dispersion. Both models compute the sum of thresholded validities for unique features but apply different penalties: the Competing Theory penalizes by the range (max - min) of the active validities, whereas the Advocated Theory penalizes by the standard deviation scaled by the number of active features. We design pairs where Option A and Option B have similar sums of validities and the exact same range, but different standard deviations (e.g., Option A has features clustered at the mean, while Option B has features clustered at the extremes). The Competing Theory applies the identical spread penalty to both options, whereas the Advocated Theory assigns a larger variance penalty to Option B (the extreme cluster) and thus strongly prefers Option A.

[29] To quantitatively dissociate the Advocated Theory (Thresholded Unique Features with Spread Penalty) from the Competing Theory (Variance Aversion and Feature Consistency), we exploit their differing penalties for feature dispersion. Both models compute the sum of thresholded validities for unique features but apply different penalties: the Advocated Theory penalizes by the range (max - min) of the active validities, whereas the Competing Theory penalizes by the standard deviation scaled by the number of active features. We design pairs where Option A and Option B have the exact same sum of validities and the exact same range, but different standard deviations (e.g., Option A has validities clustered at the mean, while Option B has validities clustered at the extremes). We also design full reversal trials where Option A has a larger range but a smaller standard deviation than Option B. The Advocated Theory will strictly prefer the option with the smaller range, while the Competing Theory will strictly prefer the option with the smaller standard deviation.

[30] To quantitatively dissociate the Advocated Theory (Diminishing Marginal Utility with Baseline Cost and Loss Aversion) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their fundamentally different treatments of shared features and feature dispersion. The Competing Theory strictly cancels shared features and evaluates the remaining unique features additively, applying a penalty based on their spread (max - min). The Advocated Theory evaluates all features (no cancellation) and applies exponential discounting to features based on their rank, while also amplifying negative features via a loss aversion multiplier. We design base trials pitting a single high-validity feature against a coalition of moderately high-validity features. The Competing Theory will often prefer the coalition (additive, low spread), while the Advocated Theory heavily discounts the coalition. We then introduce shared features (both high and low validity). For the Competing Theory, shared features cancel out, leaving predictions completely unchanged. For the Advocated Theory, adding a shared high-validity feature alters the rank of the unique features (increasing the discount on them), and adding a shared low-validity feature introduces a potential negative utility that is amplified by loss aversion, leading to strong preference reversals.

[31] To quantitatively dissociate the Advocated Theory (Thresholded Unique Features with Spread Penalty) from the Competing Theory (Diminishing Marginal Utility with Baseline Cost and Loss Aversion), we exploit their fundamentally different treatments of shared features and feature dispersion. The Advocated Theory strictly cancels shared features and evaluates the remaining unique features additively, applying a penalty based on their spread (max - min). The Competing Theory evaluates all features (no cancellation) and applies exponential discounting to features based on their rank, while also amplifying negative features via a loss aversion multiplier. We design base trials pitting options with different spreads and single features against coalitions. We then introduce shared features (both high and low validity). For the Advocated Theory, shared features cancel out, leaving choice probabilities completely unchanged. For the Competing Theory, adding a shared high-validity feature alters the rank of the unique features (increasing the discount on them), and adding a shared low-validity feature introduces a potential negative utility that is amplified by loss aversion, leading to strong preference shifts.

[32] To quantitatively dissociate the Advocated Theory (Sequential Unique Feature Comparison with Reference Point) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their fundamentally different integration mechanisms for unique features. The Advocated Theory evaluates features sequentially from highest to lowest validity and stops as soon as the evidence difference exceeds a threshold (theta). This means it can completely ignore lower-validity unique features once a decision is reached. The Competing Theory evaluates all unique features additively but applies a penalty proportional to their spread (max - min). We design trials pitting an option with a single high-validity feature against an option with one very high and one very low validity feature. The Advocated Theory, due to its sequential stopping rule, will often evaluate only the highest validity feature and ignore the lowest one. The Competing Theory will evaluate both, and because of the large spread, heavily penalize the option with two features, leading to a strong preference for the single-feature option. We also include trials pitting a single high-validity feature against a coalition of moderately high-validity features to contrast the sequential stopping (favoring the single feature) with additive integration (favoring the coalition).

[33] To quantitatively dissociate the Advocated Theory (Thresholded Unique Features with Spread Penalty) from the Competing Theory (Sequential Unique Feature Comparison with Reference Point), we exploit their fundamentally different integration mechanisms for unique features. The Competing Theory evaluates features sequentially from highest to lowest validity and stops as soon as the evidence difference exceeds a threshold (theta). This means it can completely ignore lower-validity unique features once a decision is reached. The Advocated Theory evaluates all unique features additively but applies a penalty proportional to their spread (max - min). We design trials pitting an option with a single high-validity feature against an option with one very high and one very low validity feature. The Competing Theory, due to its sequential stopping rule, will often evaluate only the highest validity feature and ignore the lowest one, choosing the option with the highest validity feature. The Advocated Theory will evaluate both, and because of the large spread, heavily penalize the option with two features, leading to a strong preference for the single-feature option. We also include trials pitting a single high-validity feature against a coalition of moderately high-validity features to contrast the sequential stopping (favoring the single feature) with additive integration (favoring the coalition).

[34] To quantitatively dissociate the Advocated Theory (Feature Coherence and Anchoring) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit their differing penalties for feature dispersion. While both models apply a spread penalty (max - min) to unique features, the Advocated Theory applies an additional 'gap penalty' based on the index distance between the best unique feature (the anchor) and the next best unique feature. Crucially, to avoid penalizing naturally sparse fallback options, the Advocated Theory only applies this gap penalty if the option has more than 2 unique features OR if it has exactly 2 unique features but claims the absolute best feature in the environment (index 0). We design trials that contrast 2-feature options starting at index 0 (large gap penalty) versus 2-feature options starting at index 1 (zero gap penalty), and 3-feature options with varying gaps. The Competing Theory, relying solely on additive values and spread penalties, will evaluate these shifts smoothly, whereas the Advocated Theory will show sharp preference reversals due to the conditional activation of the gap penalty.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Advocated Theory (Feature Coherence and Anchoring) from the Competing Theory (Thresholded Unique Features with Spread Penalty), we exploit the Advocated Theory's unique conditional 'gap penalty'. The Advocated Theory penalizes an option based on the index distance between its best unique feature and its next best unique feature, but for 2-feature options, this penalty is ONLY applied if the best feature is the absolute highest-validity feature in the environment (index 0). The Competing Theory has no index-based gap penalty and relies entirely on validities and spread (max - min). We design trials contrasting 2-feature options that claim the index 0 feature (triggering the gap penalty in the Advocated Theory) against 2-feature options that start at index 1 or 2 (triggering NO gap penalty in the Advocated Theory). The Competing Theory will systematically prefer the options with the index 0 feature due to higher validities, whereas the Advocated Theory will show sharp preference reversals, penalizing the index 0 options and favoring the fallback options.",
  "validities": [
    0.95,
    0.9,
    0.85,
    0.8,
    0.75
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
      1,
      0
    ],
    [
      1,
      0,
      0,
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
      1,
      0,
      1,
      0,
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
      0,
      1,
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
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
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
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 9962,
  "candidates_token_count": 541,
  "total_token_count": 12110
}
```
