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
**Description:** Leaky Competing Accumulator with Non-linear Configural Weighting and Evidence Transduction: Decision-makers evaluate cues simultaneously, with evidence for each option dynamically inhibiting the other in a leaky competing accumulator (LCA). Cues have a configural impact, scaled non-linearly by the total number of supporting cues. Additionally, the integrated evidence for each option is passed through a non-linear transducer (alpha) before entering the accumulation process, allowing the model to amplify the differences driven by high-validity cues and capture strong non-compensatory reversals.

**Parameters:**
- gamma: [-10.0, 10.0]
- leak: [0.1, 2.0]
- inhibition: [0.0, 5.0]
- theta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- alpha: [0.1, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    # Configural weighting: effective validity of a cue is non-linearly modulated by the total number of cues
    # Using max(1e-6, sum) to avoid 0^negative_gamma undefined errors
    sum_a_safe = max(1e-6, sum_a)
    sum_b_safe = max(1e-6, sum_b)
    
    w_a = val * (sum_a_safe ** gamma)
    w_b = val * (sum_b_safe ** gamma)
    
    # Make sure inputs are non-negative and apply non-linear transducer alpha
    I_A = max(0.0, np.sum(w_a * a)) ** alpha
    I_B = max(0.0, np.sum(w_b * b)) ** alpha
    
    # Leaky Competing Accumulator (LCA) simulation
    x_a, x_b = 0.0, 0.0
    dt = 0.1
    steps = 100
    
    for _ in range(steps):
        dx_a = (I_A - leak * x_a - inhibition * x_b) * dt
        dx_b = (I_B - leak * x_b - inhibition * x_a) * dt
        
        x_a = max(0.0, x_a + dx_a)
        x_b = max(0.0, x_b + dx_b)
        
    # Softmax choice based on final activations
    z = theta * np.array([x_a, x_b])
    z = z - np.max(z)
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
**Description:** Sequential Cue Sampling with Conflict-Modulated Threshold: Decision-makers sample cues sequentially in order of their validities. Instead of integrating all cues simultaneously, they accumulate evidence until it reaches a decision threshold. Crucially, the level of conflict between the options (inversely related to the absolute tally difference) dynamically modulates this threshold. Under low conflict, the threshold is low, leading to fast, non-compensatory decisions (like Take-The-Best). Under high conflict, the threshold increases, enforcing extensive sampling and yielding compensatory behavior. This provides a natural speed-accuracy tradeoff without mathematical artifacts.

**Parameters:**
- theta_0: [0.0, 3.0]
- theta_1: [0.0, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta_0 = float(parameters["theta_0"])
    theta_1 = float(parameters["theta_1"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate conflict based on the absolute tally difference of discriminating cues
    N_a = np.sum((a > b).astype(int))
    N_b = np.sum((b > a).astype(int))
    tally_diff = abs(N_a - N_b)
    
    # High tally difference means low conflict; zero tally difference means maximal conflict
    conflict = 1.0 / (1.0 + tally_diff)
    
    # Decision threshold is modulated by the level of conflict
    theta = theta_0 + theta_1 * conflict
    
    # Sequential sampling of cues prioritized by validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for j in cue_order:
        E += (a[j] - b[j]) * val[j]
        if abs(E) >= theta:
            break
            
    # Softmax choice probability based on accumulated evidence
    z = beta * np.array([E, -E])
    z = z - np.max(z)
    p = np.exp(z) / np.sum(np.exp(z))
    
    # Incorporate lapse rate
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

[24] This design quantitatively and qualitatively dissociates the advocated Sequential Evidence Accumulation theory from the competing WADD theory by exploiting a mathematical artifact in the competing theory's validity transformation. The competing theory claims its transformation `exp(gamma * val) - 1` ensures non-predictive cues (validity = 0.5) receive no weight. However, `exp(gamma * 0.5) - 1` is strictly positive for any gamma > 0, meaning WADD will incorrectly accumulate substantial evidence from completely non-predictive cues. The advocated theory correctly transforms validities via `(val - 0.5)^gamma`, ensuring 0.5 validity cues provide near-zero evidence. By presenting trials where the options are tied on all high-validity cues but differ exclusively on 0.5 validity cues, the advocated theory predicts an exact 50/50 guess rate, whereas WADD predicts a strong, systematic preference for the option with more 0.5 validity cues.

[25] This design quantitatively and qualitatively dissociates the advocated Weighted Additive (WADD) theory from the competing Sequential Evidence Accumulation (SEA) theory by probing for a counter-intuitive 'Less is More' anomaly caused by early stopping. SEA posits that evidence is accumulated sequentially and search terminates as soon as a threshold is crossed. Consequently, a single highly valid cue can trigger premature stopping, yielding a final evidence score just above the threshold. If that top cue is removed (made a tie), the model is forced to continue accumulating multiple weaker cues, which can collectively overshoot the threshold by a much larger margin. This paradoxically predicts higher choice confidence when the best supportive cue is missing. WADD, as an exhaustive compensatory model, strictly predicts that removing supportive cues must monotonically decrease choice confidence. By systematically tying the top cues while leaving multiple lower-validity cues in favor of Option A, we cleanly test for this non-monotonic inversion.

[26] This design quantitatively dissociates the advocated Sequential Evidence Accumulation (SEA) theory from the competing Environment-Contingent Strategy Selection theory by exploiting the difference between true sequential early-stopping and a fixed mixture of heuristics. The competing theory proposes a constant probabilistic mixture of Take-The-Best (TTB) and Tallying for any given environment (since dispersion is fixed per experiment). Consequently, it predicts that choice probabilities will always shift when the raw tally difference changes, because the Tallying component is always active to some degree. The advocated SEA theory, however, accumulates evidence sequentially and halts as soon as a threshold is met. If the first cue provides enough evidence to cross the threshold, SEA completely ignores all subsequent cues, predicting absolutely identical choice probabilities across trials with different tally differences as long as the top cue is unchanged. By presenting trials where the top cue strongly favors Option A but the remaining cues systematically shift the tally advantage toward Option B, we can cleanly test whether lower-validity cues are partially integrated (competing theory's Tallying component) or completely ignored due to early stopping (advocated theory).

[27] This design quantitatively dissociates the advocated 'Environment-Contingent Strategy Selection' theory from the competing 'Sequential Evidence Accumulation' (SEA) theory by probing for validity-insensitive compensatory integration. The advocated theory posits a probability mixture of Take-The-Best (TTB) and unweighted Tallying. Because TTB only evaluates the top discriminating cue and Tallying only counts the sheer number of cues, the advocated model is completely blind to the specific validities of secondary cues. It strongly predicts that as long as the TTB winner is unchanged and the raw tally difference is constant, choice probabilities will be strictly identical (flat lines within groups), but will step up or down categorically when the tally difference changes (steps between groups). SEA, conversely, evaluates cues sequentially and accumulates evidence weighted by a non-linear transformation of validity. If SEA's threshold is low, it stops at the first cue and predicts flat lines everywhere (failing to predict the steps between tally groups). If its threshold is high, it integrates secondary cues, predicting systematic slopes WITHIN each tally group as the validities of the secondary cues shift. SEA mathematically cannot produce stepped flat lines.

[28] The competing theory relies on a mixture of Take-The-Best (TTB) and Tallying. Because TTB is only sensitive to the highest validity discriminating cue and Tallying is only sensitive to the unweighted sum of cues, the competing theory predicts identical choice probabilities across any trials where the TTB winner and the Tally difference are both held constant. The advocated theory, however, evaluates options using a Leaky Competing Accumulator with configural weighting and non-linear transduction of all integrated validities. By fixing the top discriminating cue and the total number of cues for each option, but systematically varying the validities of the secondary cues, the advocated theory predicts graded, systematic shifts in choice probabilities. The competing theory strictly predicts flat, invariant responses across these matched trials.

[29] The advocated theory mixes Take-The-Best (TTB) and Tallying based on a fixed environment property (dispersion of validities). Because TTB is only sensitive to the highest validity discriminating cue and Tallying is only sensitive to the unweighted sum of cues, the advocated theory predicts identical choice probabilities across any trials where both the TTB winner and the Tally difference are held constant. The competing theory, however, evaluates options using a Leaky Competing Accumulator with configural weighting and non-linear transduction of all integrated validities. By fixing the top discriminating cue and the total number of cues for each option, but systematically varying the validities of the secondary cues, the competing theory predicts graded, systematic shifts in choice probabilities. The advocated theory strictly predicts flat, invariant responses across these matched trials.

[30] This design quantitatively dissociates the advocated LCA model with Configural Weighting from the competing Sequential Evidence Accumulation (SPRT) theory by exploiting their divergent treatment of tied cues. The competing theory processes cues sequentially based on validity and calculates log-odds evidence differences; crucially, it completely ignores any cue that is tied (where both options have a 1), meaning its choice predictions remain exactly identical regardless of how many tied cues are added. The advocated theory, however, utilizes a configural weighting mechanism where the effective weight of every cue is non-linearly scaled by the total number of cues favoring that option. By presenting base trials with a fixed set of discriminating cues and systematically adding tied cues (1s for both options) to the remaining features, the competing theory predicts perfectly flat, invariant choice probabilities across each set. The advocated theory predicts systematic, graded shifts in choice probabilities as the increasing total cue counts modulate the configural weights of the discriminating cues.

[31] This design quantitatively dissociates the advocated Sequential Evidence Accumulation (SPRT) theory from the competing Leaky Competing Accumulator (LCA) theory by neutralizing LCA's configural weighting and exploiting the extreme non-linearity of SPRT's log-odds update. The LCA model configurally scales the weight of each cue by the total number of cues favoring that option (sum ** gamma). By strictly matching the total number of positive cues for both options on every critical trial (sum_a == sum_b), we completely neutralize this configural effect, reducing LCA to a linear integrator of raw validities (passed through a transducer). SPRT, however, accumulates evidence using log-odds (log(v / (1 - v))), which becomes extremely non-linear for high validities. By pitting a single very high validity cue (e.g., 0.95, which has a massive log-odds weight) accompanied by weak cues against multiple moderate validity cues (e.g., 0.80, 0.75), LCA predicts that the sum of the moderate raw validities will easily outweigh the top cue. SPRT predicts the exact opposite: the massive log-odds weight of the top cue will dominate the accumulation, leading to a strict preference for the option with the single best cue, regardless of the raw validity sum.

[32] This design quantitatively dissociates the advocated LCA model with Configural Weighting from the competing Dual-Process Dynamic Attention Model by exploiting their divergent handling of tied cues when the overall tally is matched. The competing theory posits that when the tally difference is zero, the model relies 100% on a linear validity-weighted sum (WADD); thus, adding cues that are tied (present for both options) adds equal validity to both sides, keeping the validity difference constant and predicting perfectly invariant choice probabilities across trials. The advocated theory, however, applies a non-linear configural weight to all cues based on the total number of cues favoring the option (sum ** gamma). By starting with a single discriminating cue pair and systematically adding tied cues to both options, the total sum increases, which non-linearly scales the effective evidence in the LCA. The advocated theory predicts a systematic shift in choice probabilities as the background tally increases, while the competing theory strictly predicts a flat line. Additional trials with large tally differences are included to exercise the competing theory's unweighted tallying component.

[33] This design quantitatively dissociates the advocated Dual-Process Dynamic Attention Model from the competing LCA theory by exploiting their divergent sensitivity to absolute versus relative validity sums. The advocated theory posits a linear integration of validities (WADD) mixed with a linear tally difference, where the mixture weight depends solely on the absolute tally difference. Consequently, it predicts perfectly invariant choice probabilities across any set of trials where the tally difference and the linear validity difference (V_A - V_B) are both held constant, regardless of the absolute magnitude of the validities. LCA, conversely, applies a non-linear transducer (alpha) and configural weighting (gamma) to the absolute sum of validities for each option. It predicts that adding a constant to the validities of both options (shifting V_A and V_B up while keeping V_A - V_B constant) will non-linearly scale the accumulated evidence, systematically changing the choice probabilities. By presenting sets of trials with constant tally differences and constant validity differences, but increasing absolute validities, the advocated theory strictly predicts flat, invariant lines, whereas LCA predicts systematic curves.

[34] This design quantitatively dissociates the advocated Leaky Competing Accumulator (LCA) with Configural Weighting from the competing Rank-Weighted Sequential Sampling theory by exploiting their fundamentally divergent treatment of tied cues. The competing theory evaluates cues sequentially and accumulates evidence based on the difference between options (a[idx] - b[idx]). For any cue where both options have a '1' (a tied cue), the evidence difference is exactly zero, and the stopping probability (which depends only on accumulated evidence and maximum remaining possible evidence) is unchanged. Thus, the competing theory predicts absolutely identical choice probabilities across sets of trials where tied cues are added. The advocated LCA theory, however, applies a configural weight to all validities based on the total number of cues favoring each option (sum ** gamma). Adding tied cues increases the total sum for both options, which non-linearly scales the effective weights of the discriminating cues. Because this scaling is non-linear, it disproportionately boosts the option with fewer initial cues, predicting a systematic, graded shift in choice probabilities as tied cues are added. By presenting base trials with fixed discriminating cues and systematically adding 1 or 2 tied low-validity cues, we can cleanly test for configural tied-cue integration (LCA) versus tied-cue invariance (Sequential Sampling).

[35] This design quantitatively and qualitatively dissociates the advocated Rank-Weighted Sequential Sampling theory from the competing Leaky Competing Accumulator (LCA) theory by exploiting LCA's perfect symmetry when linear validity sums and cue counts are strictly matched. The competing LCA theory applies a configural weight to all validities based on the total number of cues favoring each option. By constructing critical trials where both options have the exact same number of positive cues (neutralizing configural differences) and the exact same linear sum of validities (e.g., Option A has validities 0.9 and 0.6 [sum=1.5], Option B has 0.8 and 0.7 [sum=1.5]), LCA's integrated evidence for both options becomes mathematically identical. Consequently, LCA strictly predicts an exact 50/50 guess rate for these trials, regardless of any of its parameters. The advocated theory, however, evaluates cues sequentially and features a forward-looking early stopping mechanism. Because Option A possesses the single highest-validity cue (0.9), the model will accumulate strong early evidence for A and frequently halt search before the lower-validity cues can compensatorily balance the evidence. Thus, the advocated theory predicts a strong, systematic preference for the option with the highest validity cue, cleanly breaking the 50/50 symmetry predicted by LCA.

[36] This design quantitatively and qualitatively dissociates the advocated LCA model with Configural Weighting from the competing Tally-Difference Modulated Feature Differencing theory by exploiting a paradoxical non-monotonicity in the competing theory. The competing theory posits that a large tally difference triggers a shift toward exponential validity weighting, while a small tally difference triggers equal-weighting (tallying). By pitting a single extremely high-validity cue for Option A against an increasing number of lower-validity cues for Option B, we systematically increase the tally difference in favor of B. The competing theory predicts a bizarre reversal: when B has slightly more cues (e.g., 2 vs 1), the small tally difference triggers tallying, causing B to be strongly preferred. But as B gains even MORE cues (e.g., 5 vs 1), the large tally difference triggers validity weighting, and the exponential scaling of A's top cue causes the model to suddenly reverse its preference back to Option A. The advocated LCA model, which integrates evidence using configural weighting and a non-linear transducer, predicts that adding more supportive cues to Option B will monotonically increase its accumulated evidence, strictly predicting no such reversal back to Option A.

[37] This design quantitatively and qualitatively dissociates the advocated Tally-Difference Modulated Feature Differencing theory from the competing Leaky Competing Accumulator (LCA) theory by exploiting the strict convexity of the advocated theory's exponential validity scaling. By constructing critical trials where both options have the exact same number of positive cues (neutralizing LCA's configural weighting) and the exact same linear sum of validities, LCA's integrated evidence for both options becomes mathematically identical, leading it to strictly predict an exact 50/50 guess rate. The advocated theory, however, applies an exponential transformation to the validities. Because the exponential function is strictly convex, an option with more extreme validities (e.g., one very high and one very low) will yield a strictly greater sum of transformed validities than an option with moderate validities, even when their linear sums are identical. Thus, the advocated theory predicts a systematic preference for the option with the higher validity variance, cleanly breaking the 50/50 symmetry predicted by LCA.

[38] This design quantitatively dissociates the advocated LCA model with Configural Weighting from the competing Conflict-Modulated Threshold Sequential Sampling theory by exploiting their fundamentally divergent treatment of tied cues. The competing theory calculates conflict (and thus its decision threshold) based solely on the tally difference of DISCRIMINATING cues, and it accumulates evidence only from these discriminating cues. Consequently, if we add cues that are present for both options (tied cues), the competing theory's conflict metric, threshold, and accumulated evidence remain mathematically identical, leading it to predict perfectly invariant choice probabilities. The advocated LCA theory, however, applies a configural weight to all validities scaled by the total number of cues favoring each option (sum ** gamma). Adding tied cues increases the total sum for both options, which non-linearly alters the effective weights of the discriminating cues. By presenting sets of trials with constant discriminating cues but an increasing number of tied cues, the competing theory predicts perfectly flat lines across each set, whereas the advocated theory predicts systematic, graded shifts in choice probabilities.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively and qualitatively dissociates the advocated Leaky Competing Accumulator (LCA) theory from the competing Conflict-Modulated Threshold Sequential Sampling theory by exploiting a paradoxical non-monotonicity in the competing theory's threshold mechanism. The competing theory assumes that the decision threshold is modulated by 'conflict', which it defines as the inverse of the absolute tally difference between the options. Consequently, a large tally difference in any direction yields low conflict and a low threshold, triggering early stopping. By pitting a single extremely high-validity cue for Option A against a systematically increasing number of lower-validity cues for Option B, we increase the tally difference in favor of B. Paradoxically, the competing theory predicts that as B gains MORE supportive cues, the threshold drops, causing the model to stop searching immediately after sampling A's top cue and completely ignore B's evidence\u2014resulting in a stronger preference for A. The advocated LCA model, which evaluates all cues simultaneously and configurally scales them, correctly predicts that adding supportive cues to B will monotonically increase B's accumulated evidence and decrease the preference for A.",
  "validities": [
    0.95,
    0.85,
    0.8,
    0.75,
    0.7,
    0.65
  ],
  "trial_a_ratings": [
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
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
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
      1,
      1
    ],
    [
      0,
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
  "prompt_token_count": 11090,
  "candidates_token_count": 551,
  "total_token_count": 15778
}
```
