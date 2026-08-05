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
**Description:** Evidence Dilution and Non-linear Weighting Theory (Validity-based Dilution with Amplified Penalty): Decision-makers evaluate options by integrating the validities of present features. However, instead of purely adding evidence, they partially average it. The presence of many low-validity features can paradoxically dilute the overall subjective value of an option (Evidence Dilution). This dilution is proportional to the sum of the validities of the present cues, and subjects apply a non-linear scaling to feature validities, amplifying the impact of the most valid cues. A potentially strong dilution penalty allows for severe subjective devaluation of options burdened with numerous weak features.

**Parameters:**
- lambda_val: [1.0, 20.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting to capture TTB-like reliance on top cues
    w = val ** lambda_val
    
    # Dilute by the sum of validities of the present cues
    sum_val_a = np.sum(val * a)
    sum_val_b = np.sum(val * b)
    
    # Calculate subjective values with a dilution factor (gamma)
    v_a = np.sum(w * a) / (sum_val_a ** gamma) if sum_val_a > 0 else 0.0
    v_b = np.sum(w * b) / (sum_val_b ** gamma) if sum_val_b > 0 else 0.0
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Lexicographic Thresholding with Additive Penalties Theory: Decision-makers first attempt a lexicographic (Take-The-Best) evaluation. If the top cues differ by more than a threshold, the decision is based solely on them. Otherwise, they fall back to an additive tallying mechanism where cues with a validity above a threshold 'theta' add their scaled validity to the option's value, while low-validity cues (below 'theta') subtract a penalty 'gamma'. This heavily penalizes options bloated with weak, uninformative features without discounting strong secondary cues.

**Parameters:**
- tau: [0.0, 0.5]
- gamma: [0.0, 5.0]
- lambda_val: [0.1, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- theta: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    tau = float(parameters["tau"])
    gamma = float(parameters["gamma"])
    lambda_val = float(parameters["lambda_val"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    theta = float(parameters["theta"])
    
    top_a = np.max(val[a == 1]) if np.sum(a) > 0 else 0.0
    top_b = np.max(val[b == 1]) if np.sum(b) > 0 else 0.0
    
    if abs(top_a - top_b) > tau:
        v_a = top_a
        v_b = top_b
    else:
        # Additive tallying with penalty for low validity
        w = np.where(val >= theta, val ** lambda_val, -gamma)
        v_a = np.sum(w * a)
        v_b = np.sum(w * b)
        
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
[0] To dissociate Take The Best (TTB) from Tallying, we use a 4-feature design where the most valid cue often points to one option, while the sheer number of winning cues points to the other. TTB will choose the option favored by the single highest-validity discriminating cue, whereas Tallying will choose the option favored by the majority of discriminating cues, regardless of their validities. We also include trials where Tallying predicts a tie but TTB strongly prefers one option.

[1] To robustly dissociate Tallying from Take The Best (TTB), we use 5 features with descending validities. TTB will always choose the option favored by the first discriminating feature (the one with the highest validity). Tallying, however, simply counts the number of features favoring each option, regardless of validity. By pitting a single high-validity feature against multiple lower-validity features, we create clear adversarial trials: TTB strongly prefers the option with the single best feature, while Tallying prefers the option with the majority of winning features. We also include trials where Tallying predicts a tie (equal number of winning features) but TTB makes a deterministic choice, further differentiating the two models.

[2] To dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we use a 4-feature design with a steep drop-off in validities after the first cue, but closely spaced validities for the remaining cues (e.g., 0.95, 0.75, 0.65, 0.55). TTB relies exclusively on the single highest-validity discriminating cue, ignoring all others. WADD, on the other hand, integrates all cues. If WADD's gamma parameter is low, it will behave compensatorily and favor options with multiple lower-validity cues over an option with a single high-validity cue. We construct trials that pit the highest-validity discriminating cue against a coalition of multiple lower-validity cues. For instance, an option favored by only the top cue will be chosen by TTB, whereas WADD (with low-to-moderate gamma) will choose the alternative favored by the three remaining cues. We also include trials where the highest cue is tied, shifting the TTB decision to the second cue, which is then pitted against the third and fourth cues to provide a robust test across different levels of the cue hierarchy.

[3] To robustly dissociate the Weighted Additive (WADD) theory from Take The Best (TTB), we use a 5-feature design with one highly valid cue and several moderately valid cues. TTB relies strictly on the highest-validity discriminating cue, ignoring all others. WADD, however, integrates all cues, allowing a coalition of lower-validity cues to compensate for the lack of the highest-validity cue. We construct a series of adversarial trials where the top discriminating cue points to Option A, but the majority of remaining cues point to Option B. Across these trials, TTB consistently chooses Option A, while WADD (assuming a moderate gamma parameter) will consistently choose Option B due to the overwhelming additive evidence of the lesser cues. By varying which cues tie and which discriminate, we test the compensatory nature of WADD against the non-compensatory lexicographic stopping rule of TTB at different levels of the cue hierarchy.

[4] To dissociate Strategy Mixture Theory from the Generalized WADD theory, we exploit how each model produces intermediate choice probabilities. Strategy Mixture Theory assumes a probabilistic mixture of pure TTB and pure WADD, predicting relatively constant intermediate choice proportions (e.g., matching the mixing parameter alpha) across any trial where TTB and WADD strongly conflict, regardless of the exact margin of the WADD sum. In contrast, Generalized WADD achieves intermediate probabilities either via high noise (low beta) or by precisely tuning the gamma parameter to make the exponentiated sums of the two options nearly equal. By presenting multiple distinct conflict trials with varying differences in raw feature sums, Generalized WADD cannot simultaneously equate the sums for all trials with a single gamma parameter. Consequently, Generalized WADD will predict extreme probabilities on some conflict trials and intermediate on others, whereas Strategy Mixture Theory predicts consistent intermediate probabilities across all such conflict trials.

[5] To quantitatively dissociate Generalized WADD from Strategy Mixture Theory, we manipulate the margin by which the compensatory WADD strategy opposes the non-compensatory TTB strategy. Strategy Mixture Theory assumes choices are a stable probabilistic coin-flip between pure TTB and pure WADD. Therefore, on any trial where TTB strongly favors Option A and WADD strongly favors Option B, Strategy Mixture predicts a relatively constant intermediate choice probability driven by the mixing parameter 'alpha'. In contrast, Generalized WADD integrates all features non-linearly. A single 'gamma' parameter cannot flatten the varying evidence margins across different trials. By presenting a spectrum of conflicts—ranging from the top cue being opposed by all four remaining cues, to being opposed by only two—Generalized WADD is forced to predict a graded shift in probabilities, whereas Strategy Mixture predicts a step-function or constant mixture across these conflicts.

[6] This design quantitatively dissociates Sequential Evidence Accumulation (SEA) from the Weighted Additive (WADD) theory by exploiting their fundamentally different mechanisms for scaling evidence. WADD scales validities via exponentiation (gamma), which relies on the ratio between validities. Because the ratio between adjacent validities increases as validities get smaller (e.g., 0.75/0.65 > 0.85/0.75), WADD naturally becomes *more* non-compensatory (TTB-like) at lower levels of the cue hierarchy. In contrast, SEA uses an absolute difference threshold (theta) on unscaled validities. Because the absolute values of the validities decrease down the hierarchy, SEA naturally becomes *less* non-compensatory at lower levels, as single cues are no longer large enough to cross the fixed threshold. By pitting a single discriminating cue against the remaining lower cues at different levels of the hierarchy (e.g., Cue 2 vs Cues 3-5, and Cue 3 vs Cues 4-5), SEA predicts non-compensatory choices at the top of the hierarchy and compensatory choices at the bottom. WADD predicts the exact opposite pattern, making them highly distinguishable.

[7] This design quantitatively dissociates Sequential Evidence Accumulation (SEA) from the Weighted Additive (WADD) theory by exploiting the bounded nature of SEA's stopping threshold (theta max = 1.25). In Trials 1 and 2, one option is favored by the top two most valid cues, while the other is favored by the remaining four. Because SEA accumulates evidence sequentially, the first two cues generate a running difference (0.85 + 0.80 = 1.65) that strictly exceeds the maximum possible threshold. Consequently, SEA is forced to stop and choose the option with the top two cues across 100% of valid parameter combinations, completely ignoring the remaining four cues. In contrast, WADD integrates all available information. Because the linear sum of the bottom four cues (2.70) is vastly greater than the top two (1.65), WADD strongly favors the option with the bottom four cues when its non-linear scaling parameter (gamma) is low, and smoothly shifts to favoring the top two cues as gamma increases. Trials 3 and 4 serve as controls where the running difference fluctuates, allowing SEA to also predict a shift in choices across its parameter range.

[8] This design quantitatively dissociates Dual-Process Strategy Selection Theory from Generalized WADD by holding the top discriminating cue constant across a sequence of trials while systematically increasing the number of opposing lower-validity cues. Dual-Process assumes the probability of employing the non-compensatory Take-The-Best (TTB) strategy depends solely on the absolute validity of the top discriminating cue. Because the top cue is identical across the first four trials, the probability of executing TTB remains fixed. As the compensatory WADD fallback strategy increasingly favors the opposing option, the overall choice probability for the top-cue option will drop but then plateau at the exact probability of TTB selection (e.g., an intermediate asymptote like 60%). In contrast, Generalized WADD integrates all features; adding more opposing cues continuously increases the evidence sum for the alternative option, driving the probability of choosing the top-cue option progressively toward zero. A single gamma scaling parameter in WADD cannot produce a stable intermediate plateau across these escalating conflicts.

[9] This design quantitatively dissociates Generalized WADD from Dual-Process Strategy Selection by pitting constant absolute WADD evidence differences against decreasing top-cue validities. Across trials, the top discriminating cue shifts down the hierarchy (from Cue 1 to Cue 2 to Cue 3), while the linear WADD sum difference favoring the opposing option remains strictly constant. Dual-Process Strategy Selection relies on the absolute validity of the top discriminating cue to probabilistically trigger the non-compensatory Take-The-Best (TTB) strategy; therefore, it predicts choices will become less TTB-like (more compensatory) as the top cue moves down the hierarchy and its validity drops. In contrast, Generalized WADD scales validities via exponentiation, which depends on the ratio between validities. Because the ratio between adjacent validities increases down the hierarchy (e.g., 0.75/0.65 > 0.95/0.85), Generalized WADD predicts decisions will inherently become MORE TTB-like at lower levels of the hierarchy. These diametrically opposed predictions provide a strong quantitative dissociation.

[10] This design quantitatively dissociates Sequential Evidence Accumulation (SEA) from the Weighted Additive (WADD) theory by exploiting the 'Shared Cue Muting' effect. WADD computes a linear sum of exponentiated validities. Because its softmax choice rule depends only on the difference between the two options' scores, adding a shared positive feature to both options perfectly cancels out, leaving the predicted choice probabilities mathematically identical. In contrast, SEA accumulates evidence dynamically. When a highly valid feature is shared by both options, it is sampled frequently, simultaneously adding evidence to both accumulators. This shared excitation drives both options rapidly toward the decision threshold, causing early and often simultaneous threshold crossings. This premature stopping prevents the lower-validity distinguishing features from reliably separating the options, shifting SEA's choice probabilities significantly toward 50% compared to when the shared feature is absent. By pairing trials with and without a shared top cue, we create a strict test: WADD predicts identical probabilities across pair variations, whereas SEA predicts a strong regression to chance when the top cue is shared.

[11] This design aims to dissociate the Weighted Additive (WADD) theory from Sequential Evidence Accumulation (SEA) by systematically varying the distribution of evidence across the cue hierarchy, pitting single highly-valid cues against coalitions of lower-validity cues. By presenting both extreme conflicts (e.g., Cue 1 vs Cues 2-5) and shifted conflicts (e.g., Cue 2 vs Cues 3-5), we can observe how each model scales evidence. WADD's exponentiation of validities naturally makes it more non-compensatory at lower levels of the hierarchy, whereas SEA's fixed absolute threshold and sequential sampling make it more compensatory at lower levels because individual lower-validity cues struggle to cross the threshold alone.

[12] This design quantitatively dissociates the Evidence Dilution Theory from the Weighted Additive (WADD) theory by testing for a paradoxical 'less-is-more' effect. Under WADD, adding any valid feature to an option strictly increases its weighted sum, meaning that supplementing a strong feature with several weak features will always increase the option's choice probability. In contrast, the Evidence Dilution Theory posits that adding multiple low-validity features can severely inflate the dilution denominator while adding very little to the non-linearly scaled numerator. By comparing a baseline trial (Top vs Second cue) to trials where weak cues are added to either the favored or unfavored option, WADD predicts the supplemented option will gain preference, whereas Dilution Theory predicts a massive penalty, potentially causing a preference reversal where the option with more features is chosen less often.

[13] This design quantitatively dissociates the Weighted Additive (WADD) theory from Evidence Dilution Theory by explicitly testing for a 'less-is-more' effect. Under WADD, the addition of any cue with a validity > 0.5 strictly increases the total weighted sum of the option, meaning that supplementing a strong feature with several weak features will consistently increase the probability of choosing that option. Conversely, Evidence Dilution Theory posits that adding multiple low-validity features can severely inflate the dilution denominator while adding very little to the non-linearly scaled numerator, effectively decreasing the subjective value of the option. By comparing a baseline trial (Trial 1: single top cue vs single second cue) to a dilution trial (Trial 2: top cue supplemented by three weak cues vs single second cue), WADD predicts a stronger preference for the supplemented option, whereas Dilution Theory predicts a massive penalty, potentially causing a preference reversal.

[14] This design quantitatively dissociates Evidence Dilution Theory from Attention-Gated Integration by manipulating the distance between the top cue and the diluting weak cues. Evidence Dilution Theory predicts that adding weak cues will dilute the option's value regardless of the top cue's validity, as the denominator simply sums all present validities. In contrast, Attention-Gated Integration employs a threshold (theta) relative to the maximum present validity. When a strong top cue (0.95) is paired with very weak cues (0.50-0.60), the weak cues are likely to fall outside the attention gate and be completely ignored, resulting in NO dilution. However, when a moderate top cue (0.60) is paired with those same weak cues, they pass the attention gate and cause dilution. By comparing dilution effects at the top vs. middle of the cue hierarchy, the two theories make strongly divergent predictions.

[15] This design isolates the 'attention gate' mechanism of Attention-Gated Integration (AGI) from the global dilution mechanism of Evidence Dilution Theory (ED). Under AGI, weak cues only cause dilution if their validity falls within the attention gate (theta) of the maximum validity present in that option. In Trial 1, Option A contains only weak cues, which pass the gate, causing dilution and making A less preferred than B. In Trials 2-4, a perfectly valid top cue (1.0) is added to Option A. AGI predicts that this top cue 'rescues' Option A from dilution by gating out the weak cues entirely, equating A and B in Trial 2, and making A strongly preferred in Trials 3 and 4. Conversely, ED integrates the validities of all present cues into its dilution denominator. Thus, ED predicts the weak cues will continue to heavily penalize Option A even when the top cue is present, leading to diametrically opposed predictions on whether the weak cues hurt or are ignored.

[16] This design quantitatively dissociates Evidence Dilution Theory (Advocated) from Absolute Evidence with Own-Cue Dilution Theory (Competing) by contrasting the proportional growth of the dilution denominators. The Competing theory dilutes by cue count, meaning adding any cue (even a weak one) increases the denominator by a full integer step (e.g., from 2 to 3), heavily penalizing options with many weak cues. The Advocated theory dilutes by the sum of validities, so adding a weak cue (e.g., 0.55) increases the denominator by only 0.55, resulting in a proportionally milder penalty. By comparing trials where the ratio of cue counts diverges significantly from the ratio of validity sums (e.g., 3:1 count ratio vs ~2:1 validity sum ratio), the models predict distinctly different magnitudes of preference shifts.

[17] This design quantitatively dissociates 'Absolute Evidence with Own-Cue Dilution' (Advocated) from 'Evidence Dilution and Non-linear Weighting' (Competing) by orthogonally manipulating the raw count of present cues and the sum of their validities. The Advocated theory dilutes evidence purely by the raw count of present cues (e.g., 2 cues vs 1 cue). The Competing theory dilutes evidence by the sum of the validities of those cues. By including trials where cue counts differ but validity sums are identical (e.g., one 1.0 validity cue vs two 0.5 validity cues), the Competing theory applies identical dilution to both options, while the Advocated theory heavily penalizes the two-cue option. Conversely, on trials where cue counts are equal but validity sums differ, the Advocated theory applies equal dilution, while the Competing theory penalizes the option with the higher validity sum.

[18] This design quantitatively dissociates 'Evidence Dilution and Non-linear Weighting Theory' from 'Heuristic Switching Theory with Rank-based Tallying' by testing the effect of adding multiple low-validity features to a strong feature. Under the Dilution theory, adding weak features inflates the validity sum in the denominator while adding very little to the non-linearly scaled numerator, causing a massive dilution penalty that reduces the option's subjective value ('less-is-more' effect). Under the Heuristic Switching Theory, adding weak features never hurts: Take-The-Best ignores them, while Rank-based Tallying simply adds their rank-based weights (1/rank) without any division, increasing the option's value ('more-is-more' effect). By comparing a baseline trial (top cue vs second cue) to trials where weak cues are added to either the favored or unfavored option, the two theories predict opposite directions of preference shifts.

[19] This design quantitatively dissociates 'Heuristic Switching Theory with Rank-based Tallying' (Advocated) from 'Evidence Dilution and Non-linear Weighting Theory' (Competing) by exploiting their diametrically opposed predictions regarding the addition of weak features. Under the Advocated theory, Rank-based Tallying sums the inverse ranks of active features without any division. Thus, adding weak features strictly increases an option's score ('more-is-more'). Under the Competing theory, evidence is diluted by the sum of the validities of all present features. Adding multiple low-validity features heavily inflates this denominator while adding little to the non-linearly scaled numerator, severely penalizing the option ('less-is-more'). By comparing a baseline trial (Trial 1) to trials where weak cues are appended to either the favored or unfavored option (Trials 2 and 3), the theories predict preference shifts in completely opposite directions. Trial 4 further tests this by equating the top cues and testing if the presence of weak cues hurts (Competing) or helps (Advocated).

[20] This design quantitatively dissociates 'Evidence Dilution and Non-linear Weighting Theory' (Advocated) from 'Threshold-Gated Dilution Theory' (Competing) by incrementally adding weak cues to a strong cue. The Advocated theory posits that dilution is proportional to the sum of the validities of all present cues, predicting a continuous, graded decline in subjective value as more weak cues are added (a smooth 'less-is-more' effect). In contrast, the Competing theory posits a threshold mechanism (tau): cues are integrated purely additively until the number of active cues strictly exceeds tau, at which point dilution abruptly kicks in. By comparing trials where Option A has 1, 2, 3, or 4 cues against a constant 1-cue Option B, the Advocated theory predicts a smooth drop in preference for A, whereas the Competing theory predicts A's preference will increase additively until the threshold is crossed, followed by a sudden, discontinuous drop.

[21] This design quantitatively dissociates 'Threshold-Gated Dilution Theory' (Advocated) from 'Evidence Dilution and Non-linear Weighting Theory' (Competing) by equating the sum of validities while varying the raw count of active cues. The Competing theory dilutes evidence based on the sum of validities; therefore, when two options have the same validity sum, they suffer identical dilution, and the choice is driven purely by the non-linearly scaled numerators. In contrast, the Advocated theory dilutes based on the raw count of active cues, but only after a threshold (tau) is crossed. By pairing options with equal validity sums but different cue counts (e.g., n=2 vs n=3, or n=3 vs n=4), we force the options to potentially straddle the threshold in the Advocated theory. This results in one option being evaluated additively while the other is heavily diluted, predicting massive preference shifts that the Competing theory cannot replicate.

[22] This design quantitatively dissociates 'Evidence Dilution and Non-linear Weighting Theory' (Advocated) from 'Sequential Search with Relative Evidence Thresholding' (Competing) by testing the impact of adding low-validity cues. Under the Advocated theory, adding weak cues to an option inflates the dilution denominator (sum of validities) while adding very little to the numerator (due to non-linear scaling), causing a paradoxical 'less-is-more' effect where adding cues hurts the option's choice probability. In contrast, the Competing theory evaluates cues sequentially and additively; weak cues either add to the accumulated evidence (increasing the option's preference) or are ignored due to early stopping, but they never penalize the option. Trial 1 establishes a baseline between the top two cues. Trial 2 adds weak cues to Option A, which the Advocated theory predicts will penalize A, while the Competing theory predicts will help or have no effect on A. Trial 4 equates the top two cues, leaving only weak cues favoring B; the Advocated theory predicts A is preferred because B is heavily diluted, whereas the Competing theory predicts B is chosen because it accumulates more evidence.

[23] This design quantitatively dissociates 'Sequential Search with Relative Evidence Thresholding' (Advocated) from 'Evidence Dilution and Non-linear Weighting Theory' (Competing) by exploiting the mathematical invariance of shared cues. Under the Advocated theory, cues are evaluated sequentially, and the stopping rule depends on the difference between accumulators relative to the maximum possible remaining evidence. Adding a shared cue to both options adds identical evidence to both accumulators, leaving their difference perfectly unchanged. Furthermore, because the softmax choice rule relies only on the relative difference between scores, the Advocated theory predicts that adding a shared cue will have absolutely zero effect on choice probabilities. In contrast, the Competing theory integrates evidence and dilutes it by the sum of the validities of all present cues. Adding a shared cue increases both the non-linear numerator and the dilution denominator, fundamentally altering the subjective values and predicting massive shifts in choice probabilities. By comparing baseline trials to trials where a highly valid shared cue is added, we test this strict invariance.

[24] This design quantitatively dissociates 'Evidence Dilution and Non-linear Weighting Theory' (Advocated) from 'Rank-Weighted Capacity-Bounded Integration' (Competing) by exploiting how they penalize options with multiple features. The Competing theory penalizes options based strictly on the *count* of excess features beyond a capacity limit K, meaning two options with the same number of features and the same top feature(s) will suffer the exact same penalty, leaving the option with the stronger secondary features preferred or tied. In contrast, the Advocated theory dilutes evidence based on the *sum of the validities* of all present features. By pairing options that share a top cue and have the same total count of active features, but differ in the validities of their secondary features (e.g., strong secondary cues vs. weak secondary cues), the Competing theory predicts the option with strong secondary cues will always be preferred or tied. The Advocated theory, however, predicts that the strong secondary cues will heavily inflate the dilution denominator, severely penalizing the option and causing a paradoxical preference for the option with the weaker secondary cues.

[25] This design quantitatively dissociates 'Rank-Weighted Capacity-Bounded Integration' (Advocated) from 'Evidence Dilution and Non-linear Weighting Theory' (Competing) by exploiting how they penalize options with multiple features. The Advocated theory penalizes options based strictly on the count of excess features beyond a capacity limit K. Therefore, if two options have the same number of active features and share the same top feature(s), they suffer the exact same penalty, leaving the option with the stronger secondary features preferred or tied. In contrast, the Competing theory dilutes evidence based on the sum of the validities of all present features. By pairing options that share a top cue and have the same total count of active features, but differ in the validities of their secondary features (e.g., strong secondary cues vs. weak secondary cues), the Advocated theory predicts the option with strong secondary cues will always be preferred or tied. The Competing theory, however, predicts that the strong secondary cues will heavily inflate the dilution denominator, severely penalizing the option and causing a paradoxical preference for the option with the weaker secondary cues.

[26] This design quantitatively dissociates 'Evidence Dilution and Non-linear Weighting Theory' (Advocated) from 'Cue-Redundancy Discounting Theory' (Competing) by explicitly testing for a paradoxical 'less-is-more' effect when weak features are added to a strong feature. Under the Advocated theory, dilution is driven by the sum of validities of all present cues in the denominator. Adding multiple low-validity features heavily inflates this denominator while adding very little to the non-linearly scaled numerator, severely penalizing the option and potentially causing preference reversals. In contrast, the Competing theory employs a rank-based redundancy discount where each additional feature, no matter how weak, adds a strictly positive marginal value to the option's total score. By comparing a baseline trial (Trial 1: single top cue vs single second cue) to trials where weak cues are appended to either the favored or unfavored option (Trials 2 and 3), the Advocated theory predicts that adding weak cues will hurt the option's choice probability, while the Competing theory predicts it will strictly help.

[27] This design quantitatively dissociates 'Cue-Redundancy Discounting Theory' (Advocated) from 'Evidence Dilution and Non-linear Weighting Theory' (Competing) by exploiting their opposing predictions regarding the addition of weak features. Under the Advocated theory, adding any feature contributes a strictly positive marginal value (albeit discounted by its rank), meaning adding weak features strictly increases an option's choice probability ('more-is-more'). In contrast, the Competing theory dilutes evidence by the sum of all present validities in the denominator. Adding multiple low-validity features severely inflates this denominator while adding very little to the non-linearly scaled numerator, causing a paradoxical 'less-is-more' effect where adding weak features penalizes the option. By comparing a baseline trial (Trial 1) to trials where weak cues are appended to either the favored or unfavored option (Trials 2, 3, and 4), the two theories predict preference shifts in completely opposite directions.

[28] This design quantitatively dissociates 'Evidence Dilution and Non-linear Weighting Theory' (Advocated) from 'Relative Information Overload with Categorical Salience' (Competing) by exploiting the 'Top-Cue Protection' mechanism of the Competing theory. Under the Competing theory, if an option possesses the absolute most valid cue in the environment, it is completely immune to any information overload penalties, meaning adding weak cues to it will never decrease its value. In contrast, the Advocated theory applies dilution universally based on the sum of validities of all present cues. By comparing a baseline trial (Trial 1: Top cue vs Second cue) to a critical trial (Trial 2: Top cue + three weak cues vs Second cue), the Advocated theory predicts a massive dilution penalty for the first option, potentially causing a preference reversal. The Competing theory, however, predicts the first option remains protected and strongly preferred. Trial 3 and Trial 4 provide further structural variations to test dilution and overload symmetry.

[29] This design quantitatively dissociates 'Relative Information Overload with Categorical Salience' (Advocated) from 'Evidence Dilution and Non-linear Weighting Theory' (Competing) by exploiting their divergent predictions on 'Matched Weak Cues' and 'Top-Cue Protection'. Under the Advocated theory, the overload penalty is based on the *relative difference* in the count of non-top-K cues. If the same number of weak cues is added to both options, the relative difference is zero, and the Advocated theory predicts choice probabilities will remain mathematically identical to baseline. In contrast, the Competing theory dilutes evidence by the sum of validities; adding matched weak cues inflates both denominators, shrinking both subjective values and driving choice probabilities strongly toward 50%. Furthermore, Trial 3 tests 'Top-Cue Protection': the Advocated theory posits that possessing the absolute highest validity cue grants immunity to overload penalties, keeping the option strongly preferred despite many weak cues. The Competing theory lacks this protection and predicts massive dilution for the top-cue option when burdened with weak cues.

[30] The Advocated theory (Evidence Dilution and Non-linear Weighting) dilutes an option's value based on the sum of the validities of all present cues, regardless of their relative differences. Thus, it predicts that adding equally valid cues can paradoxically reduce an option's subjective value if the dilution exponent (gamma) is greater than 1 (a 'less-is-more' effect for identical validities). In contrast, the Competing theory (Relative Validity Dilution) only applies a subtractive penalty based on the discrepancy between the top cue and other active cues. For options composed of equally valid cues, the discrepancy is exactly zero, resulting in strictly additive integration with absolutely no penalty. By pitting options with identical top validities but different counts of equally valid cues (e.g., three 0.75 cues vs one 0.75 cue), the Advocated theory predicts a potential preference for the option with fewer cues, whereas the Competing theory strictly predicts 'more-is-more'.

[31] This design quantitatively dissociates 'Relative Validity Dilution Theory' (Advocated) from 'Evidence Dilution and Non-linear Weighting Theory' (Competing) by exploiting the 'reference point' shift in the Advocated theory's discrepancy penalty. Under the Advocated theory, the penalty is based on the difference between the active top cue and all other active cues. In Trial 1, Option A's top cue is 0.9 and B's is 0.8; because 0.9 is further from the weak 0.5 cues, A suffers a larger penalty, allowing B to be preferred if the penalty weight (gamma) is high. In Trial 2, a shared 1.0 cue is added to both options. Now, 1.0 is the top cue for both, shifting the reference point. Option A's 0.9 cue is closer to 1.0 than B's 0.8 cue is, meaning A now has a SMALLER total discrepancy penalty than B. This guarantees a preference reversal (A > B) in the Advocated theory. In contrast, the Competing theory applies dilution based on the sum of active validities. Adding the shared 1.0 cue inflates both denominators equally, preserving the relative structural disadvantage of Option A's higher validity sum, thus predicting no preference reversal.

[32] This design quantitatively dissociates 'Evidence Dilution and Non-linear Weighting Theory' (Advocated) from 'Rank-Dependent Averaging Theory' (Competing) by exploiting their diametrically opposed predictions when a highly valid cue is added to an option that already has a weak cue. Under the Competing theory, evidence is integrated via a rank-dependent weighted average. Because the newly added strong cue is more valid than the existing weak cue, it pulls the average UP, strictly increasing the option's subjective value. In contrast, the Advocated theory dilutes evidence by the sum of all present validities. Adding a strong cue (e.g., 0.95) to a weak cue (e.g., 0.50) massively inflates the dilution denominator (from 0.50 to 1.45). If the dilution exponent (gamma) is sufficiently high, this denominator inflation overwhelmingly outpaces the added numerator value, causing the option's overall subjective value to plummet. Thus, the Competing theory predicts that adding a strong cue helps ('more-is-more'), while the Advocated theory predicts it can paradoxically severely penalize the option.

[33] This design quantitatively dissociates 'Rank-Dependent Averaging Theory' (Advocated) from 'Evidence Dilution and Non-linear Weighting Theory' (Competing) by testing the evaluation of multiple equally valid cues. Under the Advocated theory, subjective value is a weighted average of active cue validities. Consequently, an option with multiple identical cues has the exact same subjective value as an option with just one of those cues, predicting a strict 50% choice probability when they are pitted against each other (Trials 1 and 2), and identical preference margins when pitted against a constant alternative (Trials 3 and 4). In contrast, the Competing theory integrates evidence additively but dilutes it by the sum of validities. Because the numerator scales linearly with the number of identical cues while the denominator scales exponentially, the Competing theory predicts that options with different counts of identical cues will have vastly different subjective values, leading to strong preferences rather than indifference.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates 'Evidence Dilution and Non-linear Weighting Theory' (Advocated) from 'Lexicographic Thresholding with Additive Penalties Theory' (Competing) by exploiting the mathematical invariance of additive penalties versus non-linear divisive dilution. Under the Competing theory, when the top cues of two options are tied, it defaults to an additive tallying rule where weak cues add a constant penalty or reward. Because this integration is strictly additive, adding a shared highly valid cue to both options (Trial 2 vs Trial 1) increases both options' scores by the exact same amount. Since the softmax choice rule depends only on the difference between scores, the Competing theory predicts mathematically identical choice probabilities across these two trials. In contrast, the Advocated theory dilutes evidence by dividing by the sum of validities. Adding a shared highly valid cue increases both denominators, non-linearly altering the subjective values and predicting a significant shift in choice probabilities. Trials 3 and 4 further test the Lexicographic threshold of the Competing theory, which can completely ignore weak cues if the top cues differ, whereas the Advocated theory applies dilution universally.",
  "validities": [
    0.95,
    0.9,
    0.55,
    0.55,
    0.55
  ],
  "trial_a_ratings": [
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
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
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
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 9555,
  "candidates_token_count": 453,
  "total_token_count": 12910
}
```
