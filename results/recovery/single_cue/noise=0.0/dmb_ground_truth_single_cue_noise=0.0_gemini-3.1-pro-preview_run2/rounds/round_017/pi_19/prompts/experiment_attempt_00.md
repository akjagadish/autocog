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
**Description:** Primacy-Dominant Anchoring: Decision-makers use the first cue as a powerful anchor that overwhelmingly dominates the evaluation process. While the final cue may receive a secondary recency boost due to short-term memory, the primacy weight is structurally much larger than both the recency weight and the middle cue validities. This explicitly enforces a hierarchy where primacy is the primary driver of choice.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- primacy_weight: [10.0, 30.0]
- recency_weight: [0.0, 9.0]
- gamma: [0.0, 5.0]
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
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    primacy_weight = float(parameters["primacy_weight"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    
    # Enforce Primacy-Dominant Anchoring hierarchy
    w[0] = primacy_weight
    if len(w) > 1:
        w[-1] = recency_weight
    
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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
**Description:** Serial Position Dual-Overweighting: Decision-makers evaluate options by integrating features, but due to memory and attention constraints at the sequence boundaries, both the first (primacy) and the last (recency) cues are assigned independent, disproportionately large weights. Unlike models that normalize attention or weights, these boundary weights are unnormalized, allowing them to independently dominate choice when necessary. Middle cues are weighted by their stated validities, scaled non-linearly. This mechanism captures both extreme primacy and extreme recency effects without the dampening effect of weight normalization.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 5.0]
- primacy_weight: [0.0, 10.0]
- recency_weight: [0.0, 10.0]
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
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    primacy_weight = float(parameters["primacy_weight"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    
    # Overweight the first and final features due to primacy and recency
    if len(w) > 1:
        w[0] = primacy_weight
        w[-1] = recency_weight
    elif len(w) == 1:
        w[0] = primacy_weight + recency_weight
        
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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

[23] This design aims to quantitatively dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (Normalized Linear Interpolation WADD). Pure Tallying simply counts the number of winning features for each option, completely ignoring cue validities, and predicts a strict 50/50 guess when the tally is tied. In contrast, the WADD model integrates validities, allowing high-validity cues to compensate for a smaller quantity of winning features. The trial set includes 'tally tie' trials where WADD predicts a strong preference due to high-validity features, and 'conflict' trials where Tallying favors an option with a greater quantity of low-validity features while WADD favors the alternative with fewer, but higher-validity features.

[24] To quantitatively dissociate the Advocated Theory (Strict Tallying with Probabilistic TTB Tie-Breaker) from the Competing Theory (Pure Tallying), we exploit the specific condition where their predictions diverge: tally ties. When the tally difference is non-zero, both theories predict identical choices based strictly on the tally difference with softmax noise. However, when the tally is tied, Pure Tallying predicts a strict 50/50 guess, while the Advocated Theory applies a Take-The-Best (TTB) tie-breaker in probability space. By including 'tally tie' trials where the highest-validity discriminating cue strongly favors one option, we can observe whether choices remain strictly at chance or shift systematically toward the TTB prediction. We also include 'tally win' trials (some where TTB disagrees with the tally) to anchor the shared beta and epsilon parameters.

[25] To quantitatively dissociate Pure Tallying (Advocated Theory) from Strict Tallying with Probabilistic TTB Tie-Breaker (Competing Theory), we exploit the specific condition where their predictions diverge: tally ties. When the tally difference is non-zero, both theories predict identical choices based strictly on the tally difference with softmax noise. However, when the tally is tied, Pure Tallying predicts a strict 50/50 guess, while the Competing Theory applies a Take-The-Best (TTB) tie-breaker in probability space. By including 'tally tie' trials where the highest-validity discriminating cue strongly favors one option, we can observe whether choices remain strictly at chance or shift systematically toward the TTB prediction. We also include 'tally win' trials to anchor the shared beta and epsilon parameters.

[26] To quantitatively dissociate the Advocated Theory (Tallying with Probabilistic Least-Valid Cue Dropping) from Pure Tallying, we exploit their differing sensitivity to the single least valid feature in a 5-feature design. Pure Tallying treats all features equally, predicting choices based strictly on the overall tally difference, and predicting a 50/50 guess whenever the tally is tied. The Advocated Theory assumes decision-makers occasionally drop the least valid cue and tally the remainder. We include 'tally tie' trials where the least valid cue favors one option; here, the Advocated Theory predicts a systematic preference for the opposing option (since dropping the least valid cue breaks the tie in its favor), whereas Pure Tallying predicts a strict guess. We also include 'tally win' trials where the least valid cue either supports or opposes the tally winner, leading the Advocated Theory to respectively attenuate or amplify the choice probability compared to Pure Tallying. Control trials where the least valid cue is tied ensure both models make identical baseline predictions.

[27] To quantitatively dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (Tallying with Probabilistic Least-Valid Cue Dropping), we exploit their differing sensitivity to the single least valid feature in a 5-feature design. Pure Tallying treats all features equally, predicting choices based strictly on the overall tally difference, and predicting a 50/50 guess whenever the tally is tied. The Competing Theory assumes decision-makers occasionally drop the least valid cue and tally the remainder. We include 'tally tie' trials where the least valid cue favors one option; here, the Competing Theory predicts a systematic preference for the opposing option (since dropping the least valid cue breaks the tie in its favor), whereas Pure Tallying predicts a strict guess. We also include 'tally win' trials where the least valid cue either supports or opposes the tally winner, leading the Competing Theory to respectively attenuate or amplify the choice probability compared to Pure Tallying. Control trials where the least valid cue is tied ensure both models make identical baseline predictions.

[28] To quantitatively dissociate Recency-Biased Cue Overweighting (Advocated Theory) from Pure Tallying (Competing Theory), we exploit their differing treatments of the final feature in the sequence. Pure Tallying treats all features equally and bases choices strictly on the overall tally of winning features, predicting a 50/50 guess whenever the tally is tied. The Advocated Theory posits that the final feature in the sequence receives a disproportionately large weight due to recency effects, potentially overriding the rest of the cues. We use a 5-feature design where the final feature is nominally the least valid. The trial set includes 'tally tie' trials where the final feature favors one option (Tallying predicts guessing; Advocated Theory predicts a strong preference for that option), 'conflict' trials where Tallying favors one option based on a greater quantity of winning cues but the final feature favors the alternative, and control trials where the final feature is tied. This design ensures a robust quantitative dissociation driven specifically by the final feature's influence.

[29] To quantitatively dissociate Pure Tallying (Advocated Theory) from Recency-Biased Cue Overweighting (Competing Theory), we exploit their differing treatments of the final feature in the sequence. Pure Tallying treats all features equally and bases choices strictly on the overall tally of winning features, predicting a 50/50 guess whenever the tally is tied. The Competing Theory posits that the final feature in the sequence receives a disproportionately large weight due to recency effects, potentially overriding the rest of the cues. We use a 5-feature design where the final feature is nominally the least valid. The trial set includes 'tally tie' trials where the final feature favors one option (Tallying predicts guessing; Competing Theory predicts a strong preference for that option), 'conflict' trials where Tallying favors one option based on a greater quantity of winning cues but the final feature favors the alternative, and control trials where the final feature is tied. This design ensures a robust quantitative dissociation driven specifically by the final feature's influence.

[30] To quantitatively dissociate Recency-Biased Cue Overweighting (Advocated Theory) from Primacy-Recency Cue Integration (Competing Theory), we exploit two structural differences: the treatment of the first cue (primacy) and the normalization of weights. The Competing Theory includes an explicit, independent primacy weight and normalizes all weights to sum to 1, bounding the maximum score difference. The Advocated Theory strictly ties the first cue's weight to its validity (via gamma) while allowing the final cue's weight to be an independent, unnormalized parameter, which can lead to extreme, unscaled score differences and deterministic choices even at moderate softmax temperatures. The trial set includes direct conflicts between the first and last cues, conflicts between the middle cues and the extremes, and trials where the extremes are tied, forcing the models to reveal whether the first cue has an independent primacy boost and whether the choice probabilities exhibit the compression characteristic of normalized weights.

[31] To quantitatively dissociate the Advocated Theory (Primacy-Recency Cue Integration with Normalized Weights) from the Competing Theory (Recency-Biased Cue Overweighting), we exploit two key structural differences: the treatment of the first cue (primacy) and the normalization of weights. The Advocated Theory includes an explicit primacy weight added to the first cue and normalizes all weights to sum to 1, bounding maximum score differences. The Competing Theory has no primacy mechanism (treating the first cue purely by its validity) and overwrites the last cue's weight with an unnormalized parameter, allowing for extreme score differences. By setting the first and last cues to have lower baseline validities than the middle cues, we can construct trials where the first cue opposes a higher-validity middle cue. The Advocated Theory can predict a preference for the first cue due to primacy, whereas the Competing Theory will deterministically favor the higher-validity middle cue. We also pit the first and last cues together against the middle cues to test the bounding effect of normalization versus unnormalized recency domination.

[32] To quantitatively dissociate Recency-Biased Cue Overweighting (Advocated Theory) from Serial Position Dual-Overweighting (Competing Theory), we exploit the critical difference in their treatment of the first cue. The Advocated Theory treats the first cue exactly like any middle cue, weighting it strictly according to its stated validity (scaled by gamma). The Competing Theory, however, posits an independent, unnormalized 'primacy_weight' for the first cue, allowing it to dominate choices regardless of its stated validity. We use a 5-feature design where the first and last cues have low validities, while the middle cues have high validities. The trial set includes 'primacy conflicts' where the first cue favors one option while the high-validity middle cues favor the alternative (Advocated predicts middle cues win; Competing allows the first cue to win). We also include direct conflicts between the first and last cues to contrast the independent primacy and recency weights in the Competing Theory against the isolated recency weight in the Advocated Theory.

[33] To quantitatively dissociate Serial Position Dual-Overweighting (Advocated Theory) from Recency-Biased Cue Overweighting (Competing Theory), we exploit their critical difference in handling the first cue. The Advocated Theory assigns an independent, unnormalized 'primacy_weight' to the first cue, allowing it to dominate choices regardless of its stated validity. The Competing Theory treats the first cue like any middle cue, weighting it strictly by its stated validity (scaled by gamma). We use a 5-feature design where the first and last cues have low stated validities, while the middle cues have high validities. The trial set includes 'primacy conflicts' where the first cue favors one option while the high-validity middle cues favor the alternative (the Advocated Theory predicts the first cue can win; the Competing Theory predicts the middle cues will win). We also include direct conflicts between the first and last cues to contrast the dual independent weights in the Advocated Theory against the single recency weight in the Competing Theory.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Primacy-Dominant Anchoring (Advocated Theory) from Serial Position Dual-Overweighting (Competing Theory), we exploit the structural constraints on their primacy and recency weights. The Advocated Theory strictly enforces that the primacy weight (10-30) is significantly larger than the recency weight (0-9) and middle validities. The Competing Theory allows both primacy and recency weights to vary in the same range (0-10), meaning recency can potentially override primacy. We use a 5-feature design where the first and last cues are in direct conflict, as well as conflicts between the boundary cues and the high-validity middle cues. Trials where the first cue favors one option and the last cue favors the other will forcefully separate the models: the Advocated Theory dictates the first cue must win, whereas the Competing Theory allows the last cue to win.",
  "validities": [
    0.6,
    0.9,
    0.85,
    0.9,
    0.6
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
      0,
      0,
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
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
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
      0,
      1,
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
      1,
      0,
      1,
      1
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
      1,
      1,
      1
    ],
    [
      1,
      1,
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
  "prompt_token_count": 8715,
  "candidates_token_count": 535,
  "total_token_count": 10268
}
```
