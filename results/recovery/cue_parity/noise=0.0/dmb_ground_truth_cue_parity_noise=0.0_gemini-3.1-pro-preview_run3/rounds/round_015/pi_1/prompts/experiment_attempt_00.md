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
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Validity-Scaled Take The Best: Decision-makers process cues sequentially in order of validity and stop at the first discriminating cue, completely ignoring all subsequent cues. However, the confidence in the choice is not absolute; it decays based on the rank (or validity) of the discriminating cue. When a tie forces the decision down to lower-validity cues, the resulting choice becomes noisier and closer to a guess.

**Parameters:**
- decay_rate: [0.0, 1.0]
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
    
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    winner = None
    disc_rank = None
    
    # Find the first discriminating cue
    for rank, j in enumerate(cue_order):
        if a[j] > b[j]:
            winner = 0
            disc_rank = rank
            break
        elif b[j] > a[j]:
            winner = 1
            disc_rank = rank
            break
            
    if winner is None:
        return np.array([0.5, 0.5])
        
    decay_rate = float(parameters["decay_rate"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Confidence decays exponentially with the rank of the discriminating cue
    confidence = decay_rate ** disc_rank
    
    scores = np.zeros(2)
    scores[winner] = confidence
    
    # Convert scores to probabilities via softmax
    z = beta * scores
    z -= np.max(z)  # Numerical stability
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design systematically pits the one-reason decision rule (Take The Best) against the unweighted feature-counting rule (Tallying). By employing five features with strictly descending validities, we can construct trials where the option favored by the single highest-validity discriminating cue (TTB's choice) is opposed by a larger number of lower-validity cues (Tallying's choice). We also include trials where Tallying predicts a tie (guessing) but TTB makes a deterministic prediction, ensuring a strong quantitative dissociation across the parameter space.

[1] This design systematically dissociates Tallying from Take The Best (TTB) by contrasting multi-feature majorities against the single highest-validity discriminating cue. By using 5 features with descending validities, we construct trials where an option is favored by a single high-validity cue but opposed by a larger number of lower-validity cues (where Tallying and TTB make opposite predictions). We also include trials where Tallying predicts a tie (equal number of wins for both options) while TTB makes a deterministic choice based on the top cue, ensuring a strong quantitative dissociation.

[2] To strongly dissociate Take The Best (TTB) from the Weighted Additive (WADD) strategy, we employ a 4-feature design with linearly spaced validities. TTB relies exclusively on the single highest-validity discriminating cue, ignoring all others. WADD, depending on its gamma parameter, integrates all cues. We include 'compensatory' trials where the highest-validity cue favors option A, but all remaining cues favor option B. For TTB, option A is always the deterministic choice. For WADD, the accumulation of lower-validity cues will easily override the top cue unless gamma is extremely high. We also include a trial where the sum of validities for A and B are perfectly balanced (e.g., cues 1 and 4 vs. cues 2 and 3), leading WADD to predict a tie (when gamma=1) or a slight preference, whereas TTB maintains a strong preference for the option with cue 1.

[3] To effectively dissociate the Weighted Additive (WADD) strategy from Take The Best (TTB), we employ a 5-feature design with closely spaced validities. TTB evaluates options solely based on the highest-validity discriminating cue, ignoring the number or weight of all lower-validity cues. In contrast, WADD integrates all cues weighted by their validities (transformed by gamma). The trials are designed as 'compensatory' choices where the single best discriminating cue favors one option (TTB's choice), but the sum of the remaining validities strongly favors the other option (WADD's choice, for low-to-moderate gamma values). We also include trials where WADD predicts a near-tie but TTB has a deterministic preference, maximizing the quantitative divergence between the two models.

[4] To distinguish the pure Take The Best (TTB) model from the TTB-Tallying mixture model, this design includes trials where the single highest-validity cue strongly conflicts with the simple sum of features (Tallying). In the pure TTB model, the number of lower-validity cues opposing the choice has zero effect on the choice probabilities; only the top discriminating cue and the global noise parameters matter. In the mixture model, a strong opposition from Tallying (e.g., 1 feature vs. 4 features) will systematically drag the choice probability toward the Tallying favorite or toward 50/50, depending on the mixture weight. We also include a trial where Tallying predicts a tie while TTB makes a deterministic choice, isolating the mixture model's tendency to dilute TTB's confidence with Tallying's uncertainty.

[5] To quantitatively dissociate the pure Take The Best (TTB) model from the TTB-Tallying mixture model, we employ a 6-feature design and parametrically vary the support for Tallying while keeping TTB's prediction constant. Across five trials, the highest-validity cue always favors Option A (so pure TTB predicts a constant probability of choosing A, subject only to global noise). However, the number of lower-validity cues favoring Option B systematically decreases from 5 to 1. The mixture model predicts a monotonic increase in the probability of choosing Option A across these trials, as Tallying's preference shifts from strongly opposing TTB, to tying with it, to strongly agreeing with it. Pure TTB predicts a flat line across these same trials.

[6] To quantitatively dissociate Take The Best (TTB) from Probabilistic Cue Integration (WADD), we use a 6-feature design with specifically tuned validities. The validities are chosen such that the log-odds (evidence weight) of any single feature is strictly less than the sum of the log-odds of all features with lower validities. Across the first four trials, we systematically shift the highest discriminating cue from the 1st feature down to the 4th feature. In each of these trials, Option A is favored by the single highest discriminating cue (TTB's choice), while Option B is favored by all remaining lower-validity cues. For a broad range of its gamma parameter, WADD will integrate the lower-validity cues and choose Option B, whereas TTB will deterministically choose Option A. We also include trials where both models agree, ensuring we can distinguish the strategies from pure noise.

[7] To quantitatively dissociate Probabilistic Cue Integration (WADD) from Take The Best (TTB), we employ a 5-feature design. TTB uses only the single highest-validity discriminating cue, strictly ignoring the rest. WADD integrates all cues via a log-odds transformation. We construct trials where the single highest-validity discriminating cue favors one option (TTB's choice), but the sum of the evidence weights of lower-validity cues strongly favors the other option (WADD's choice). By varying the number and weight of the opposing cues across trials, WADD predicts a graded shift in choice probability, while TTB predicts deterministic choices solely based on the top cue.

[8] To dissociate pure Take The Best (TTB) from the TTB-WADD mixture model, we use a 5-feature design and parametrically vary the support from lower-validity cues while holding the highest-validity discriminating cue constant. Across the first three trials, the highest-validity cue always favors Option A, so pure TTB predicts a constant probability of choosing A. However, the WADD score systematically shifts from strongly favoring Option B to favoring Option A. The mixture model, which incorporates WADD, predicts a monotonic increase in the probability of choosing Option A across these trials, while pure TTB predicts a flat line. Additional trials apply the same logic when the first cue is tied, providing further points of divergence.

[9] This design leverages a key quantitative divergence in how the two models handle choice confidence across different discriminating cues. In the pure Take The Best (TTB) model, choice probability is determined solely by the fact that *some* cue discriminated, yielding a binary score (1 for winner, 0 for loser); it completely ignores the validity of the discriminating cue, predicting a flat, constant confidence across trials as long as the top discriminating cue favors the same option. In contrast, the Advocated Theory's mixture model incorporates WADD, which integrates the validities of all cues. By systematically shifting which cue is the highest discriminating cue (e.g., from the 1st to the 5th) while keeping the lower-validity cues opposed, the pure TTB model predicts identical choice probabilities across all these trials. The mixture model, however, predicts a monotonic shift in choice probability because the WADD component's score difference changes drastically as more high-validity cues become tied.

[10] To quantitatively dissociate the pure Take The Best (TTB) model from the dynamic sequential evidence accumulation model (Competing Theory), we systematically vary both the validity of the top discriminating cue and the support from lower-validity cues. Pure TTB relies strictly on the first cue that discriminates between the options. Crucially, TTB ignores the magnitude of the cue's validity and all subsequent cues, predicting identical choice probabilities across any trials where the top discriminating cue favors the same option. In contrast, the Competing Theory accumulates evidence: the validity of the top cue determines the initial evidence magnitude (and thus the probability of stopping early), and lower-validity cues continue to shape the choice probability if search continues. By comparing trials where the first cue favors Option A but lower cues strongly oppose it, are tied, or strongly support it, and by shifting which cue is the top discriminator, we test TTB's prediction of flat, invariant choice probabilities against the Competing Theory's prediction of graded, evidence-dependent probabilities.

[11] To quantitatively dissociate the dynamic sequential evidence accumulation model (Advocated Theory) from the pure Take The Best (TTB) model (Competing Theory), we systematically vary the support from lower-validity cues and the identity of the highest discriminating cue. In pure TTB, choice probability depends strictly on the first cue that discriminates between the options, ignoring both the magnitude of that cue's validity and any subsequent cues. Thus, TTB predicts identical choice probabilities across trials as long as the first discriminating cue favors the same option. In contrast, the Advocated Theory accumulates evidence weighted by validities; the initial evidence magnitude determines stopping probability, and if search continues, subsequent cues modify the choice probability. By comparing trials where the first discriminating cue favors Option A but lower cues strongly oppose it, or where the first discriminating cue shifts to lower validities, we test TTB's prediction of flat, invariant choice probabilities against the Advocated Theory's graded, evidence-dependent probabilities.

[12] To quantitatively dissociate the pure Take The Best (TTB) model (Advocated Theory) from the TTB-Tallying mixture model (Competing Theory), we systematically vary the support from Tallying while holding the highest-validity discriminating cue constant. Across all five trials, the highest-validity cue (Feature 1) favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring the remaining features, and thus predicts an identical, flat probability of choosing Option A across all trials. In contrast, the Competing Theory incorporates Tallying (which counts the total number of discriminating cues). As we parametrically increase the number of lower-validity cues favoring Option A (from 0 to 4), the Tallying component shifts from strongly favoring Option B to strongly favoring Option A. Consequently, the mixture model predicts a monotonic increase in the probability of choosing Option A across these trials, directly contradicting the flat line predicted by pure TTB.

[13] To quantitatively dissociate the TTB-Tallying mixture model (Advocated Theory) from the pure Take The Best model (Competing Theory), we systematically vary the support from the Tallying strategy while holding the pure TTB prediction constant. Across all trials, the highest-validity discriminating cue always favors Option A. In pure TTB, choice probability depends strictly on the first cue that discriminates between the options, ignoring both the magnitude of that cue's validity and any subsequent cues. Thus, pure TTB predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Advocated Theory incorporates Tallying, which counts the total number of discriminating cues. By parametrically shifting the Tallying score from strongly favoring Option B to favoring Option A (even while top cues are tied), the mixture model predicts a graded shift in choice probabilities, directly contradicting the flat line predicted by pure TTB.

[14] To quantitatively dissociate the pure Take The Best (TTB) model (Advocated Theory) from the exponential decay weighting model (Competing Theory), we systematically vary the opposition from lower-validity cues while keeping the highest-validity discriminating cue constant. Across the first four trials, the highest-validity cue (Feature 1) favors Option A. Pure TTB bases its decision exclusively on this single cue, completely ignoring all others, and thus predicts a perfectly flat probability of choosing Option A across these trials. In contrast, the Competing Theory integrates all cues using a rank-based exponential decay. As we parametrically decrease the number of lower-validity cues favoring Option B (from 4 down to 0), the weighted sum shifts increasingly in favor of Option A. Consequently, the Competing Theory predicts a monotonic increase in the probability of choosing Option A, directly contradicting the invariant flat line predicted by pure TTB. The final two trials replicate this logic when the first cue is tied, testing the same principle on the second-highest validity cue.

[15] To quantitatively dissociate the exponential decay weighting model (Advocated Theory) from pure Take The Best (Competing Theory), we systematically vary the support from lower-rank cues while holding the highest discriminating cue constant. Pure TTB strictly stops at the first discriminating cue, completely ignoring all subsequent cues. Therefore, as long as the top discriminating cue favors Option A, TTB predicts an identical, flat probability of choosing A across trials. In contrast, the Advocated Theory integrates all cues using an exponentially decaying weight based on validity rank. By parametrically shifting the lower-rank cues from strongly favoring Option B to strongly favoring Option A, the Advocated Theory predicts a monotonic shift in choice probability, directly contradicting the flat line predicted by pure TTB. We test this across different top discriminating cues to ensure robustness.

[16] To quantitatively dissociate the pure Take The Best (TTB) model (Advocated Theory) from the TTB-Tallying mixture model (Competing Theory), we systematically vary the support from Tallying while holding the highest-validity discriminating cue constant. Across the first four trials, the highest-validity cue (Feature 1) favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring the remaining features, and thus predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Competing Theory incorporates Tallying (which counts the total number of positive features). As we parametrically increase the number of lower-validity cues favoring Option A (from 0 to 3), the Tallying component shifts from strongly favoring Option B to strongly favoring Option A. Consequently, the mixture model predicts a monotonic increase in the probability of choosing Option A across these trials, directly contradicting the flat line predicted by pure TTB. Trials 5-7 repeat this logic with Feature 1 tied, testing the same principle on the second-highest validity cue.

[17] To quantitatively dissociate the TTB-Tallying mixture model (Advocated Theory) from the pure Take The Best model (Competing Theory), we systematically vary the support from the Tallying strategy while holding the pure TTB prediction constant. Across the first four trials, the highest-validity discriminating cue (Feature 1) always favors Option A. In pure TTB, choice probability depends strictly on the first cue that discriminates between the options, ignoring both the magnitude of that cue's validity and any subsequent cues. Thus, pure TTB predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Advocated Theory incorporates Tallying, which counts the total number of positive features. By parametrically shifting the Tallying score from strongly favoring Option B to strongly favoring Option A, the mixture model predicts a graded, monotonic shift in choice probabilities, directly contradicting the flat line predicted by pure TTB. The final two trials test this same principle when the highest-validity cue is tied.

[18] To quantitatively dissociate the pure Take The Best (TTB) model (Advocated Theory) from the Sequential Evidence Accumulation model (Competing Theory), we systematically vary the support from lower-validity cues while holding the highest-validity discriminating cue constant. Across the first five trials, the highest-validity cue (Feature 1) always favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring the remaining features, and thus predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Competing Theory accumulates evidence sequentially and only stops probabilistically. As we parametrically increase the number of lower-validity cues favoring Option A (from 0 to 4), the accumulated evidence shifts from strongly favoring Option B to strongly favoring Option A. Consequently, the Competing Theory predicts a monotonic increase in the probability of choosing Option A across these trials, directly contradicting the flat line predicted by pure TTB. Trials 6 and 7 test this same principle when the first cue is tied, isolating the effect on the second-highest validity cue.

[19] To quantitatively dissociate the Sequential Evidence Accumulation model (Advocated Theory) from pure Take The Best (Competing Theory), we systematically vary the support from lower-validity cues while holding the highest-validity discriminating cue constant. Across all trials, the highest-validity cue (Feature 1) favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring the remaining features, and thus predicts an identical, flat probability of choosing Option A across all trials. In contrast, the Advocated Theory accumulates evidence sequentially and stops probabilistically. As we parametrically decrease the number of lower-validity cues favoring Option B and eventually add cues favoring Option A, the accumulated evidence shifts from strongly favoring Option B to strongly favoring Option A. Consequently, the Advocated Theory predicts a monotonic increase in the probability of choosing Option A across these trials, directly contradicting the flat line predicted by pure TTB.

[20] To cleanly dissociate the pure Take The Best (TTB) model (Advocated Theory) from the Strategy Mixture Model (Competing Theory), we employ a 6-feature design that varies the magnitude of the compensatory Weighted Additive (WADD) score while holding the TTB prediction constant. Across the first four trials, the highest-validity cue always favors Option A, leading pure TTB to predict an invariant, flat probability of choosing A, completely ignoring the lower-validity cues. The Competing Theory, however, mixes TTB with WADD. By systematically shifting the lower-validity cues from strongly favoring Option B to strongly favoring Option A, the WADD component's preference reverses, causing the mixture model to predict a monotonic increase in the probability of choosing A. We also include trials where the highest-validity cues are tied, forcing TTB to rely on the second or third cue while WADD continues to integrate all cues, further exposing the quantitative divergence between the one-reason stopping rule and the compensatory mixture.

[21] To quantitatively dissociate the Strategy Mixture Model (Advocated Theory) from the pure Take The Best model (Competing Theory), we systematically vary the support from lower-validity cues while holding the highest-validity discriminating cue constant. Across the first five trials, the highest-validity cue (Feature 1) always favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring the remaining features, and thus predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Advocated Theory mixes TTB with a compensatory Weighted Additive (WADD) strategy. As we parametrically shift the lower-validity cues from strongly favoring Option B to strongly favoring Option A, the WADD component's preference reverses. Consequently, the mixture model predicts a monotonic increase in the probability of choosing Option A across these trials, directly contradicting the flat line predicted by pure TTB. Trials 6-8 replicate this logic when the first cue is tied, isolating the effect on the second-highest validity cue.

[22] To quantitatively dissociate the pure Take The Best (TTB) model (Advocated Theory) from the TTB with Confirmatory Search model (Competing Theory), we systematically vary the number of lower-validity cues that oppose the first discriminating cue. Across the first seven trials, the highest-validity cue (Feature 1) always favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring the remaining features, and thus predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Competing Theory performs a confirmatory check on the remaining cues. As we parametrically increase the 'net contradiction' from 0 to 7 opposing cues, the Competing Theory predicts a sharp drop in the probability of choosing Option A once the number of opposing cues reaches a specific threshold. This step-function drop directly contradicts the flat line predicted by pure TTB. The final trial tests the same principle when the highest-validity cue is tied.

[23] To quantitatively dissociate the TTB with Confirmatory Search model (Advocated Theory) from the pure Take The Best (TTB) model (Competing Theory), we systematically vary the 'net contradiction' from lower-validity cues. Pure TTB strictly stops at the first discriminating cue, completely ignoring all subsequent cues, and thus predicts an identical, flat probability of choosing Option A across all trials where the top cue favors A. In contrast, the Advocated Theory posits a confirmatory check: if the net number of opposing lower-validity cues (opposing minus supporting) exceeds a specific threshold, confidence drops. By parametrically increasing the net contradiction from 0 to 7 across trials, and by including trials where supporting cues cancel out opposing cues to yield the same net contradiction, we test the Advocated Theory's prediction of a step-function drop in choice probability against pure TTB's prediction of a perfectly flat line.

[24] To strongly dissociate the Advocated Theory (Take The Best) from the Competing Theory (Tallying), we use a 5-feature design with descending validities. TTB decides solely based on the highest-validity discriminating cue, whereas Tallying merely counts the total number of positive cues, ignoring validities. We construct trials where these strategies make opposite predictions (e.g., TTB strongly favors Option A based on the top cue, while Tallying strongly favors Option B based on a larger number of lower-validity cues), as well as trials where Tallying predicts a tie but TTB makes a deterministic choice. This ensures a sharp quantitative divergence across the parameter spaces of both models.

[25] To quantitatively dissociate the Advocated Theory (Tallying) from the Competing Theory (Take The Best, TTB), we design trials where the two models make sharply contradictory predictions. Tallying ignores validities and simply counts positive cues, whereas TTB ignores the number of cues and relies exclusively on the highest-validity discriminating cue. We include trials where TTB's chosen option has fewer total positive cues than the alternative (TTB vs. Tallying opposition), and trials where the options have an equal number of positive cues but differ on high-validity cues (Tallying predicts a tie, TTB predicts a deterministic choice).

[26] To quantitatively dissociate the pure Take The Best (TTB) model (Advocated Theory) from the Strategy Mixture Model (Competing Theory), we systematically vary the support from lower-validity cues while holding the highest-validity discriminating cue constant. Across the first five trials, the highest-validity cue (Feature 1) always favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring the remaining features, and thus predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Competing Theory mixes TTB with a compensatory Weighted Additive (WADD) strategy. As we parametrically shift the lower-validity cues from strongly favoring Option B to strongly favoring Option A, the WADD component's preference reverses. Consequently, the mixture model predicts a monotonic increase in the probability of choosing Option A across these trials, directly contradicting the flat line predicted by pure TTB. Trials 6 and 7 replicate this logic when the first cue is tied, isolating the effect on the second-highest validity cue.

[27] To quantitatively dissociate the Strategy Mixture Model (Advocated Theory) from the pure Take The Best model (Competing Theory), we systematically vary the support from lower-validity cues while holding the highest-validity discriminating cue constant. Across the first five trials, the highest-validity cue (Feature 1) always favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring the remaining features, and thus predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Advocated Theory mixes TTB with a compensatory Weighted Additive (WADD) strategy. As we parametrically shift the lower-validity cues from strongly favoring Option B to strongly favoring Option A, the WADD component's preference reverses. Consequently, the mixture model predicts a monotonic increase in the probability of choosing Option A across these trials, directly contradicting the flat line predicted by pure TTB. The final two trials replicate this logic when the first cue is tied, isolating the effect on the second-highest validity cue.

[28] To quantitatively dissociate pure Take The Best (TTB, the Advocated Theory) from the Non-linear Rank-Weighted Additive Integration model (the Competing Theory), we systematically vary the support from lower-ranked cues while holding the highest-validity discriminating cue constant. Across the first four trials, the highest-validity cue (Feature 1) favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring all subsequent features, and thus predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Competing Theory integrates all features using a rank-based exponential decay weight. As we parametrically shift the lower-validity cues from strongly favoring Option B to strongly favoring Option A, the Competing Theory predicts a monotonic increase in the probability of choosing Option A, directly contradicting the flat line predicted by pure TTB. We also include trials where the highest-validity cue is tied, isolating the same effect on the second- and third-highest validity cues.

[29] To quantitatively dissociate the Non-linear Rank-Weighted Additive Integration model (Advocated Theory) from pure Take The Best (Competing Theory), we systematically vary the support from lower-ranked cues while holding the highest-validity discriminating cue constant. Across the first four trials, the highest-validity cue (Feature 1) favors Option A. Pure TTB bases its decision solely on this first discriminating cue, completely ignoring all subsequent features, and thus predicts an identical, flat probability of choosing Option A across all these trials. In contrast, the Advocated Theory integrates all features using a rank-based exponential decay weight. As we parametrically shift the lower-validity cues from strongly favoring Option B to strongly favoring Option A, the Advocated Theory predicts a monotonic increase in the probability of choosing Option A, directly contradicting the flat line predicted by pure TTB. We also include trials where the highest-validity cue is tied to test the same effect on the second-highest validity cue.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the pure Take The Best (TTB) model (Advocated Theory) from the Validity-Scaled Take The Best model (Competing Theory), we systematically vary the rank of the first discriminating cue. Both models process cues sequentially and stop at the first discriminating cue, completely ignoring all subsequent cues. However, pure TTB predicts that the choice probability is invariant to the rank of the discriminating cue (yielding a flat confidence across trials as long as some cue discriminates). In contrast, the Competing Theory posits that choice confidence decays exponentially with the rank of the discriminating cue. By constructing a series of trials where the first discriminating cue shifts from the highest validity cue down to the lowest validity cue, pure TTB predicts a constant, flat probability of choosing the favored option, whereas the Competing Theory predicts a monotonic decay in choice probability toward 50/50.",
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
      0,
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
      0,
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
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1
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
      1,
      0,
      1,
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
  "prompt_token_count": 8344,
  "candidates_token_count": 528,
  "total_token_count": 10216
}
```
