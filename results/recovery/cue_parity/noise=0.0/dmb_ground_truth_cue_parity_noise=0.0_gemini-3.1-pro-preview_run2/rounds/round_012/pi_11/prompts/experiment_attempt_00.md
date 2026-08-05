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
**Description:** Weighted Additive (WADD) Integration with Zero-Anchored Soft Validity Transformation: Decision-makers compute a subjective value for each option by summing its features, weighted by a zero-anchored exponential transformation of their validities. This transformation (exp(gamma * val) - 1) ensures that non-predictive cues receive no weight, preventing the artificial inflation of tallies by low-validity cues while allowing the highest validity cues to exponentially dominate when necessary. This naturally bridges compensatory and non-compensatory decision-making without heuristic switching.

**Parameters:**
- gamma: [0.0, 10.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Subjective transformation of validities
    # Subtracting 1.0 ensures that a zero-validity cue would receive exactly 0 weight,
    # preventing artificial inflation of low-validity cues and allowing the highest
    # validity cues to dominate when necessary.
    w = np.exp(gamma * val) - 1.0
    
    # Compute subjective values (Weighted Additive sum)
    v_a = np.sum(w * a)
    v_b = np.sum(w * b)
    
    scores = np.array([v_a, v_b])
    
    # Softmax choice rule
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
**Description:** Sequential Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity. Each cue provides evidence proportional to a non-linear transformation of its validity above chance. Evidence is accumulated as a running difference between the two options. If the absolute accumulated evidence exceeds a threshold, search is terminated and a choice is made based on the current evidence. If all cues are evaluated without crossing the threshold, a decision is made based on the final accumulated evidence. This allows for fast, non-compensatory decisions when top cues are highly valid, while gracefully falling back to compensatory integration when early cues are less decisive.

**Parameters:**
- theta: [0.0, 10.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale weights by transforming validity above chance, allowing better separation
    weights = np.maximum(val - 0.5, 0.001) ** gamma
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if absolute accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from Tallying, we use a 4-feature design with descending validities. TTB decides solely based on the highest-validity cue that discriminates between the options. Tallying ignores validities and simply counts the number of features favoring each option. The trials are constructed so that one option always wins on the highest-validity discriminating cue (favored by TTB), while the other option wins on a greater number of lower-validity cues (favored by Tallying), or the total tally results in a tie while TTB makes a deterministic choice.

[1] To strongly dissociate Tallying (which gives equal weight to all features and chooses based on the simple sum of winning features) from Take The Best (which relies exclusively on the single highest-validity discriminating cue), we use a 5-feature design with descending validities. We construct trial pairs where one option wins on the highest-validity discriminating feature (so TTB strongly prefers it) but loses on a larger number of lower-validity features (so Tallying prefers the other option). We also include trials where Tallying results in a tie (no preference) while TTB makes a deterministic choice based on the first discriminating cue.

[2] This design quantitatively dissociates Take The Best (TTB) from Weighted Additive (WADD) theory by exploiting TTB's strict one-reason decision making. Under TTB, the choice probability depends only on the first discriminating cue and is entirely insensitive to the number or weight of any lower-validity cues. Thus, TTB predicts identical choice probabilities across trials where the top discriminating cue favors an option, regardless of whether the other option is favored by one, none, or all remaining lower-validity cues. WADD, by contrast, integrates all features. Because it scales validities but does not discard them, WADD predicts that the choice probability will vary systematically with the margin of integrated scores (e.g., an option will be chosen less frequently if it is opposed by many lower-validity cues compared to when it is unopposed). By holding the TTB choice constant while varying the compensatory evidence, we can cleanly separate the two models.

[3] This design quantitatively dissociates WADD from Take The Best (TTB) by exploiting TTB's invariance to the rank of the highest discriminating cue. Under TTB, the choice probability is determined solely by the first cue that discriminates. Because TTB assigns a fixed binary score (1 for winner, 0 for loser) before applying response noise, it predicts exactly the same choice probability across trials regardless of whether the decision is triggered by the 1st, 2nd, 3rd, or 4th most valid cue. WADD, conversely, computes a compensatory score based on scaled validities. It predicts that the choice probability will systematically decrease as the validity of the highest discriminating cue decreases. By pitting a shifting highest discriminating cue (from most to least valid) against a constant lowest-validity opposing cue, we can cleanly test whether choice confidence is invariant to the cue's rank (TTB) or proportional to its scaled validity (WADD).

[4] This design quantitatively dissociates the Strategy Mixture model from WADD by exploiting a fundamental difference in how they process cue validities. The Strategy Mixture model combines Take The Best (TTB) and Tallying. Crucially, its TTB component is completely invariant to the absolute validity of the highest discriminating cue (it assigns a fixed 1/0 score to the winner), and its Tallying component is invariant to validities entirely. Therefore, the Mixture model predicts that choice probabilities will remain identical across trials as long as the TTB winner is unchanged and the Tally difference is held constant. WADD, by contrast, scales validities and integrates them; it predicts that choice probabilities will systematically decline if the highest discriminating cue shifts to a lower-validity feature, even if the Tally difference is constant. We present a 3x3 grid of trials varying the rank of the highest discriminating cue (1st, 2nd, or 3rd) and the Tally difference (-1, 0, +1). Strategy Mixture predicts three perfectly flat lines across cue ranks, whereas WADD predicts three downward-sloping lines.

[5] This design quantitatively dissociates the Weighted Additive (WADD) theory from the Strategy Mixture (TTB + Tallying) model by exploiting how secondary cues are integrated. Across a set of trials, we hold two things strictly constant: the highest discriminating cue (which fixes the TTB choice) and the total number of cues favoring each option (which fixes the Tallying difference at zero). Because the Strategy Mixture model's TTB component only cares about the top cue and its Tallying component only counts the sheer number of wins, it predicts identical choice probabilities across all these trials. WADD, however, integrates the scaled validities of all cues. By systematically swapping which secondary cues belong to Option A versus Option B, WADD predicts a steep, systematic change in choice probabilities as the secondary support for the TTB winner increases or decreases in validity.

[6] This design quantitatively dissociates Sequential Cue Accumulation (SCA / Take-The-Best with Verification) from Weighted Additive (WADD) theory by probing for a 'dead zone' of guessing. SCA posits a threshold of evidence; if the accumulated evidence across all cues fails to reach this threshold (e.g., due to highly contradictory cues), the decision-maker defaults to exact 50/50 guessing. WADD, conversely, always computes a graded choice probability based on the integrated score difference, no matter how small. By presenting a spectrum of trials with highly contradictory cues that result in very small but non-zero net validity differences, SCA predicts a flat 50% choice rate for all trials falling under the threshold, whereas WADD predicts a smooth, graded curve of probabilities that strictly tracks the small variations in net evidence.

[7] This design quantitatively dissociates Weighted Additive (WADD) theory from Sequential Cue Accumulation (SCA / Take-The-Best with Verification) by probing for 'early stopping' behavior. SCA assumes decision-makers accumulate evidence sequentially by validity and stop searching as soon as a threshold is reached. If the first few high-validity cues strongly favor one option and exceed this threshold, SCA completely ignores all remaining lower-validity cues, predicting identical choice probabilities across trials regardless of how the remaining cues are distributed. WADD, by contrast, integrates all information. It predicts that choice probabilities will systematically shift as the distribution of lower-validity cues changes, even when the top cues provide overwhelming evidence. By fixing the top two highly valid cues to favor Option A and systematically varying whether the remaining four cues favor A or B, we can cleanly test whether lower-validity cues are ignored (SCA) or integrated (WADD).

[8] This design quantitatively dissociates 'Tallying with Validity Tie-Breaking' from Weighted Additive (WADD) theory by contrasting trials where Tallying and WADD conflict, and by probing the nature of tie-breaking. Under the advocated theory, decision-makers strictly choose the option with the most positive cues (Tallying) regardless of validities; validities are only consulted via Take-The-Best (TTB) when the tallies are tied. Therefore, the advocated model predicts that choice probabilities will be identical across all tally-tie trials as long as TTB selects a winner, regardless of the validity of the tie-breaking cue. WADD, by contrast, integrates all scaled validities. It predicts graded choice probabilities that systematically decrease as the highest discriminating cue shifts to lower validities. Furthermore, in trials where one option has more total cues but the other has fewer, higher-validity cues, the advocated model predicts a categorical preference for the high-tally option, whereas WADD predicts a preference for the high-validity option.

[9] This design quantitatively dissociates the advocated Weighted Additive (WADD) theory from the competing 'Tallying with Validity Tie-Breaking' model by contrasting scenarios where the two models fundamentally disagree on the integration of evidence. The competing theory posits a strict lexical process: it first compares the sheer number of positive cues (Tallying), and only if these are equal does it consult validities via Take-The-Best (TTB). Thus, it predicts categorical, invariant choice probabilities across all trials where the tally difference is the same (e.g., all trials where A has more cues than B, or all tally-tie trials where TTB favors A). WADD, in contrast, integrates all exponentially scaled validities. It predicts graded choice probabilities that systematically shift as the validities of the specific cues favoring each option change, regardless of the raw tally. By presenting pairs where Option A wins the tally but loses on high-validity cues, and tally-tie pairs where the TTB-determining cue shifts down the validity hierarchy, we can cleanly test whether choices are categorically driven by tallies/ties or continuously driven by integrated validities.

[10] This design quantitatively dissociates the advocated Dual-Process Strategy Selection theory from the competing Weighted Additive (WADD) theory by probing for a discrete threshold in evidence integration. The advocated theory posits that if the raw tally difference between two options meets or exceeds a certain threshold, the decision-maker entirely ignores validities and relies purely on the tally difference. If the tally difference is below the threshold, they fall back to a validity-sensitive strategy (TTB or WADD). WADD, in contrast, continuously integrates scaled validities across all trials regardless of the tally difference. By presenting pairs of trials where the tally difference is held constant (at 0, 1, 2, 3, or 4) but the validities of the discriminating features vary wildly (e.g., Option B holds the highest validity cue in one trial, but the lowest in the other), we can cleanly test for this threshold. The advocated theory predicts that once the tally difference is large enough, choice probabilities will suddenly become identical across matched-tally pairs, completely insensitive to the validities. WADD predicts continuous sensitivity to validities (and thus different choice probabilities) across all pairs.

[11] To quantitatively dissociate the advocated Weighted Additive (WADD) theory from the competing Dual-Process Strategy Selection theory, we probe for the existence of a discrete tally-difference threshold. The competing theory posits that if the raw tally difference between two options meets or exceeds a certain threshold (from 1 to 5), the decision-maker entirely ignores cue validities and relies purely on the unweighted tally difference. WADD, in contrast, continuously integrates scaled validities across all trials regardless of the tally difference. By presenting pairs of trials where the tally difference is held constant (at 1, 2, 3, and 4) but the validities of the discriminating features vary drastically (e.g., Option B holds the highest validity cue in one trial, but the lowest in the other), we can cleanly test for this threshold. The competing theory predicts that once the tally difference reaches the threshold, choice probabilities will suddenly become identical across matched-tally pairs, completely insensitive to the validities. WADD predicts continuous sensitivity to validities (and thus divergent choice probabilities) across all pairs.

[12] This design quantitatively dissociates the advocated Dual-Process Strategy Selection theory from the competing Strict Tallying theory by exploiting their divergent handling of small tally differences. The competing theory posits that any non-zero tally difference strictly dictates the choice, ignoring validities entirely. Thus, across trials where the tally difference is exactly 1 (e.g., Option A has 2 cues, Option B has 3 cues), the competing theory predicts invariant choice probabilities favoring the higher-tally option. The advocated theory, however, includes a variable threshold. If the tally difference is below this threshold (e.g., a threshold of 2), the decision-maker falls back to a validity-sensitive strategy (TTB or WADD). By presenting multiple trials with a constant tally difference of 1 but drastically shifting which option holds the highest-validity cues, the advocated theory predicts robust, validity-driven shifts in choice probabilities, whereas the competing theory predicts completely flat, tally-driven responses.

[13] To quantitatively dissociate the advocated 'Strict Tallying with Validity Tie-Breaker' theory from the competing 'Dual-Process Strategy Selection' theory, we probe the core difference in their threshold mechanisms. The advocated theory asserts that ANY non-zero tally difference strictly determines the choice, with validities only used to break perfect ties. The competing theory posits a variable threshold (1 to 5), predicting that for small tally differences (e.g., 1, 2, or 3), decision-makers might fall back to a validity-sensitive strategy (TTB or WADD). We present a series of trials where Option A has a lower tally but possesses the single highest-validity cue, while Option B has a higher tally built from lower-validity cues. By systematically varying the tally difference (1, 2, 3) against the validities, the advocated theory rigidly predicts the higher-tally option will always be chosen, regardless of the difference magnitude. The competing theory predicts a reversal in choice (favoring the high-validity, low-tally option) whenever the tally difference falls below the subject's threshold.

[14] To quantitatively dissociate the advocated 'Configural Log-Odds Evidence Accumulation' theory from the competing 'Strict Tallying with Validity Tie-Breaker' theory, we systematically vary the raw tally difference against the log-odds validity of the cues, and manipulate the distribution of missing cues. The competing theory posits that any non-zero tally difference strictly dictates the choice, ignoring validities entirely, and only uses validities (via TTB or WADD) when tallies are exactly tied. The advocated theory, however, evaluates options using non-linearly scaled log-odds validities and applies a squared configural penalty for missing cues. By presenting pairs where one option has a higher tally but the other holds the highest-validity cue, the competing theory predicts a strict preference for the higher-tally option, whereas the advocated theory predicts graded responses that can favor the lower-tally option. Furthermore, in tally-tie trials, the advocated theory's configural penalty heavily punishes options missing the most critical cues, leading to divergent predictions from simple linear WADD or TTB tie-breaking.

[15] To quantitatively dissociate the advocated 'Strict Tallying with Validity Tie-Breaker' theory from the competing 'Configural Log-Odds Evidence Accumulation' theory, we construct trials that pit raw tally differences against extreme log-odds validities and configural penalties. The advocated theory strictly chooses the option with the higher tally, completely ignoring validities unless there is a perfect tie. The competing theory uses log-odds scaling and a squared configural penalty for missing cues, meaning an option with a lower tally but possessing the most valid cues will be preferred, as the higher-tally option will suffer a massive penalty for missing the critical high-validity cues. By varying the tally difference (e.g., A has 2 cues but the highest validities, B has 3 or 4 cues with lower validities) and including tally-tie trials where the squared penalty diverges from linear tie-breaking, we can cleanly separate the two models.

[16] This design quantitatively dissociates the advocated 'Threshold-Gated Adaptive Strategy' from the competing 'Strict Tallying with Validity Tie-Breaker' theory by probing decisions where the tally difference is exactly 1. The competing theory posits that ANY non-zero tally difference strictly triggers a tallying choice, meaning the higher-tally option is always preferred. The advocated theory, however, incorporates a variable threshold (0, 1, or 2). If the subject's threshold is 2, a tally difference of 1 falls below the threshold, causing the decision-maker to fall back to the validity-driven Take-The-Best (TTB) strategy. By presenting trials where Option A has a tally advantage of 1 but Option B possesses the highest-validity cue, the competing theory predicts a strict preference for Option A, whereas the advocated theory predicts a preference for Option B. Additionally, tally-tie trials are included where TTB and Weighted Additive (WADD) conflict, ensuring we can also identify the specific tie-breaking mechanism used by the competing theory.

[17] To quantitatively dissociate the advocated 'Strict Tallying with Validity Tie-Breaker' theory from the competing 'Threshold-Gated Adaptive Strategy', we focus on trials with a tally difference of exactly 1, where the lower-tally option is favored by the highest-validity cue. The advocated theory asserts that any non-zero tally difference strictly dictates the choice, meaning the higher-tally option is always preferred regardless of validities. The competing theory, however, posits a variable threshold (0, 1, or 2). If a subject's threshold is 2, a tally difference of 1 falls below the threshold, prompting a fallback to Take-The-Best (TTB). This causes the competing theory to predict a preference for the lower-tally, high-validity option. We also include tally-tie trials where TTB and Weighted Additive (WADD) disagree, to identify the specific tie-breaking mechanism used by the advocated theory, and large tally-difference trials where both models converge on tallying.

[18] To quantitatively dissociate the Threshold-Gated Adaptive Strategy from WADD with Zero-Anchored Soft Validity Transformation, we exploit their divergent responses to varying tally differences. The advocated theory posits a discrete threshold (0, 1, or 2) for the raw tally difference; if the difference meets or exceeds the threshold, the decision relies purely on Tallying (ignoring validities), otherwise it falls back to Take-The-Best (TTB). WADD, on the other hand, continuously integrates all cues weighted by an exponential function of their validities, regardless of the raw tally difference. By presenting a series of trials where the tally difference systematically increases (from 0 to 3) while the lower-tally option always holds the single highest-validity cue, we can pinpoint the threshold. The advocated theory predicts a sharp, categorical reversal in choice from the high-validity option to the high-tally option once the tally difference crosses the threshold. WADD predicts a continuous, graded shift in choice probabilities as the compensatory weight of the multiple lower-validity cues gradually overcomes the single high-validity cue.

[19] This design quantitatively dissociates WADD from the Threshold-Gated Adaptive Strategy by manipulating the tally difference and the distribution of secondary cues. The competing theory posits a discrete tally threshold (0, 1, or 2) above which decision-makers strictly tally, and below which they use TTB. WADD, conversely, continuously integrates all cues with an exponential weighting. By varying the tally difference (from 0 to 3) while keeping the highest validity cue on the lower-tally option, the competing theory predicts abrupt, categorical shifts from TTB to Tallying based on the threshold. Furthermore, in tally-tied trials, the competing theory relies purely on TTB (ignoring secondary cues), whereas WADD integrates secondary cues, predicting that choice probabilities will shift when secondary cues are rearranged.

[20] This design quantitatively and qualitatively dissociates the advocated 'Noisy Take-The-Best with Validity Differences' theory from the competing 'Weighted Additive (WADD) with Zero-Anchored Soft Validity Transformation' theory by exploiting a critical inversion in their predictions. The advocated theory posits that the probability of successfully using a cue depends strictly on the difference in validity between that cue and the next one in the hierarchy. By setting validities such that a lower-ranked cue has a much larger validity difference to its successor than the highest-ranked cue has to its successor (e.g., v0=0.95, v1=0.90, v2=0.70; diffs are 0.05 and 0.20), the advocated theory paradoxically predicts higher choice certainty when the decision rests solely on the second cue compared to the first cue. WADD, conversely, integrates cues based on their absolute validities using a monotonic exponential transformation, strictly predicting higher choice certainty for the highest-validity cue. Furthermore, trials pitting a single high-validity cue against multiple lower-validity cues test WADD's compensatory integration against the advocated model's non-compensatory, sequential skipping behavior.

[21] This design quantitatively and qualitatively dissociates the advocated Weighted Additive (WADD) theory from the competing Noisy Take-The-Best (TTB) with Validity Differences theory by creating a massive inversion in cue utilization. The competing theory posits that the probability of using a cue depends strictly on the difference between its validity and the next highest validity. By setting the top two validities extremely close (e.g., 0.98 and 0.95) but creating a huge gap between the second and third (e.g., 0.95 and 0.55), the competing theory paradoxically predicts that the highest validity cue will be frequently skipped due to its small difference from the next cue, while the second cue will heavily dictate choices. WADD, conversely, scales validities absolutely via a zero-anchored exponential transformation, ensuring the highest validity cue always carries the most weight. Trials are constructed where the first and second cues conflict; WADD strictly predicts choosing the option favored by the first cue, whereas the competing theory predicts choosing the option favored by the second cue.

[22] To quantitatively dissociate the advocated Two-Stage Tally-Threshold Theory from the competing WADD with Zero-Anchored Soft Validity Transformation theory, we systematically vary the tally difference while pitting a single highly valid cue against multiple lower-validity cues. The advocated theory proposes a discrete threshold for the tally difference; if the difference meets or exceeds this threshold, the decision-maker strictly follows the tally (ignoring validities), leading to a sharp, categorical reversal in choice. If the difference is below the threshold, they fall back to a validity-sensitive strategy. The competing theory, however, applies an exponential transformation to the validities, meaning the highest validity cue can exponentially dominate multiple lower-validity cues regardless of the raw tally difference. By presenting trials where Option A has the single highest validity cue but Option B has a progressively larger tally advantage (from 0 to 4), the advocated theory predicts a step-function reversal to Option B once the threshold is crossed, whereas the competing theory predicts a continuous, graded shift where Option A may remain preferred even at large tally differences due to the exponential weight of its top cue.

[23] To quantitatively dissociate the advocated WADD theory (with Zero-Anchored Soft Validity Transformation) from the competing Two-Stage Tally-Threshold Theory, we exploit their divergent responses to varying tally differences when pitted against a single highly valid cue. The competing theory proposes a discrete threshold (ranging from 1 to 4) for the tally difference; if the difference meets or exceeds this threshold, the decision-maker strictly follows the tally (ignoring validities), leading to a sharp, categorical reversal in choice. The advocated WADD theory applies an exponential transformation to the validities, meaning the highest validity cue can exponentially dominate multiple lower-validity cues regardless of the raw tally difference. By presenting trials where Option A possesses the single highest-validity cue but Option B has a progressively larger tally advantage (from 1 to 4), the competing theory predicts a step-function reversal to Option B once the threshold is crossed. WADD, in contrast, predicts a continuous, graded shift where Option A may remain strongly preferred even at large tally differences due to the exponential weight of its top cue. We also include tally-tied trials to dissociate the competing theory's fallback strategies (TTB vs. unweighted WADD) from the advocated theory's exponentially weighted WADD.

[24] This design quantitatively and qualitatively dissociates the advocated Sequential Evidence Accumulation (SCA) theory from the competing Weighted Additive (WADD) theory by exploiting their fundamentally divergent treatment of chance-level (0.5) validities. SCA correctly centers evidence at chance, assigning near-zero weight to 0.5-validity cues via its transformation `max(val - 0.5, 0.001)^gamma`. The competing WADD model, however, uses a zero-anchored transformation `exp(gamma * val) - 1.0` that fails to discount chance-level cues, assigning them substantial positive weight. By pitting a single moderately valid cue (e.g., 0.60) against multiple chance-level cues (0.50), SCA predicts a strict preference for the option with the valid cue, as the 0.50 cues provide no accumulated evidence. WADD, conversely, integrates the chance-level cues, predicting that multiple 0.50 cues can easily outweigh a 0.60 validity cue. Furthermore, in trials where options differ solely on 0.50 cues, SCA predicts pure guessing, while WADD predicts a strong deterministic preference.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the advocated Weighted Additive (WADD) theory from the competing Sequential Evidence Accumulation (SCA) theory, we exploit their fundamentally different approaches to stopping rules and low-validity cues. SCA accumulates evidence sequentially and stops when a threshold is reached; furthermore, it centers validities at chance (0.5), assigning near-zero weight to low-validity cues. The advocated WADD theory uses a zero-anchored exponential transformation, meaning it continuously integrates all cues and assigns substantial positive weight even to chance-level (0.5) validities. By fixing the top two high-validity cues to favor Option A while systematically varying the distribution of the remaining low-validity cues, SCA predicts identical choice probabilities across these trials (due to early stopping or near-zero weighting of the bottom cues). WADD, in contrast, predicts systematic, graded shifts in choice probabilities as the lower-validity cues are integrated.",
  "validities": [
    1.0,
    0.95,
    0.6,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
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
      1,
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
    ],
    [
      1,
      0,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
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
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 7546,
  "candidates_token_count": 439,
  "total_token_count": 9588
}
```
