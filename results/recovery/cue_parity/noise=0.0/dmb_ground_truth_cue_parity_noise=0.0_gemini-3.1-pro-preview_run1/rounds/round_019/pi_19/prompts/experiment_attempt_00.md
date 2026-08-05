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
**Description:** Probabilistic Strategy Mixture (WADD and Tallying): Decision-makers consist of a population that probabilistically mixes between two distinct compensatory strategies: pure Weighted Additive (WADD) and pure Tallying. On any given decision, a subject acts as a WADD user with probability p_wadd (accumulating evidence proportional to validities centered at chance) or as a Tallying user with probability 1 - p_wadd (weighing all features equally). This directly produces the intermediate tie-breaking proportions observed in experiments without suffering from the extreme numerical distortions of exponentiated validity weights.

**Parameters:**
- p_wadd: [0.0, 1.0]
- beta: [0.1, 25.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    p_wadd = float(parameters["p_wadd"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # WADD component (normative weights centered at 0.5)
    w_wadd = val - 0.5
    diff_wadd = np.sum(a * w_wadd) - np.sum(b * w_wadd)
    scores_wadd = np.array([diff_wadd, -diff_wadd])
    
    z_wadd = beta * scores_wadd
    z_wadd -= np.max(z_wadd)
    e_wadd = np.exp(z_wadd)
    p_wadd_core = e_wadd / np.sum(e_wadd)
    
    # Tallying component (equal weights)
    diff_tally = np.sum(a) - np.sum(b)
    scores_tally = np.array([diff_tally, -diff_tally])
    
    z_tally = beta * scores_tally
    z_tally -= np.max(z_tally)
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # Mixture of strategies
    p_core = p_wadd * p_wadd_core + (1.0 - p_wadd) * p_tally_core
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Lexicographic Semi-Order Evidence Accumulation with Tallying Fallback: Decision-makers evaluate features sequentially in order of their validity, accumulating validity-weighted evidence. If the accumulated evidence difference between the options exceeds an internal confidence threshold at any step, they stop and make a choice. If they exhaust all features without reaching the threshold (e.g., due to conflicting top features), they abandon the weighted sequential process and resort to a simple tally of all features to break the tie.

**Parameters:**
- threshold: [0.0, 5.0]
- tally_weight: [0.0, 1.5]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    tally_weight = float(parameters["tally_weight"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = a - b
    evidence = 0.0
    stopped = False
    
    for idx in order:
        evidence += diff[idx] * val[idx]
        if abs(evidence) >= threshold and abs(evidence) > 0:
            stopped = True
            break
            
    if not stopped:
        # Resort to a simple tally of all features if threshold not met
        evidence = np.sum(diff) * tally_weight
        
    scores = np.array([evidence, -evidence])
    
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] This design systematically pits Take The Best (TTB) against Tallying. By using five features with distinct, descending validities, we can construct trials where the option favored by the single most valid discriminating cue (TTB's choice) is opposed by the option that wins on a larger number of less valid cues (Tallying's choice). This creates a sharp quantitative dissociation: TTB predicts strong preference for the single-best-cue winner, while Tallying predicts preference for the option with more overall feature wins.

[1] This design systematically pits Tallying against Take The Best (TTB) using five features with distinct, descending validities. We construct trials where the option favored by the single most valid discriminating cue (TTB's choice) is opposed by the option that wins on a larger number of less valid cues (Tallying's choice). We also include trials where Tallying predicts a tie (equal number of feature wins) but TTB predicts a clear winner, providing a sharp quantitative dissociation.

[2] This design systematically pits the Weighted Additive Model (WADD) against Tallying by contrasting the number of feature wins with the validities of those features. Tallying strictly counts the number of features where one option dominates, completely ignoring the validities. In contrast, WADD scales the features by a non-linear transformation of their validities centered at chance. We constructed a set of 5 features with a steep drop-off in validity. The trial pairs include cases where one option wins on a single highly valid feature while the other wins on multiple less valid features (causing opposing predictions between WADD and Tallying), as well as cases where both options win on an equal number of features (causing Tallying to predict a tie while WADD predicts a strict preference).

[3] This design pits Tallying against the Weighted Additive Model (WADD). Tallying simply counts the number of features where an option is superior, entirely ignoring the validities of those features. WADD, on the other hand, scales validities non-linearly and computes a weighted sum. By using a set of validities with a steep drop-off (e.g., 0.95, 0.80, 0.60, 0.55, 0.55), we can construct trials where one option wins on a single highly valid feature while the other wins on multiple low-validity features. Tallying will consistently choose the option with more feature wins, whereas WADD will prefer the option with the higher-validity features. Furthermore, we include trials where both options win on an equal number of features (e.g., 2 vs 2), causing Tallying to predict a tie (guessing), while WADD predicts a deterministic preference based on feature validities.

[4] To dissociate the Advocated Theory (Tallying with a validity-weighted tie-breaker) from the Competing Theory (Pure Tallying), the design relies heavily on trials where both options win on an equal number of features. According to Pure Tallying, any tie in the number of feature wins results in a 50/50 guess, regardless of the validities of the features involved. In contrast, the Advocated Theory predicts that when tallies are tied, subjects will evaluate the options using a secondary compensatory mechanism that weights features by their validities (centered at chance). By configuring trials where A and B have the same number of feature wins but differ significantly in the cumulative validity of those winning features, we create a sharp quantitative dissociation: Pure Tallying predicts uniform probability (guessing), while the Advocated Theory predicts a deterministic preference for the option with higher-validity features. Filler trials where one option strictly wins the tally are also included to ensure subjects do not realize that tallying always results in a tie.

[5] To dissociate Pure Tallying (Advocated Theory) from Tallying with a validity-weighted tie-breaker (Competing Theory), we focus on trials where both options win on an equal number of features. Pure Tallying strictly counts feature wins and predicts a 50/50 guess whenever these tallies are tied, regardless of the validities of the winning features. The Competing Theory, however, uses a validity-weighted tie-breaker when tallies are tied, predicting a strong preference for the option whose winning features have higher validities. By pairing high-validity feature wins against low-validity feature wins in equal numbers, we create a sharp quantitative dissociation. Filler trials where one option wins on more features are included to prevent subjects from noticing that tallies are often tied and to ensure both models have trials where they agree.

[6] To effectively dissociate Salience-Weighted Tallying with a Near-Tie Threshold from Pure Tallying, we must test 'near-ties' (tally differences of 1 or 2) rather than just exact ties. Pure Tallying invariably favors the option with more feature wins, regardless of the margin or the validities of the features involved. The Advocated theory, however, posits that if the tally difference falls within a specific threshold, a secondary compensatory tie-breaker evaluates the options based on feature salience. By pitting an option that wins on fewer but highly valid (salient) features against an option that wins on more but lower-validity features, we create trials where Pure Tallying strongly predicts a win for the tally-leader, but the Advocated theory predicts a reversal in preference due to the near-tie threshold activating the salience-weighted tie-breaker.

[7] To dissociate Pure Tallying (Advocated Theory) from Salience-Weighted Tallying with a Near-Tie Threshold (Competing Theory), we focus on 'near-tie' trials where the tally difference is exactly 1 or 2. Pure Tallying strictly follows the tally count, predicting a consistent preference for the option with more feature wins, regardless of the validities of those features. The Competing Theory posits that small tally differences fall under a threshold, triggering a salience-weighted compensatory tie-breaker. By pitting an option with fewer but highly valid feature wins against an option with more but lower-validity feature wins, Pure Tallying predicts the tally-leader will win, whereas the Competing Theory predicts a preference reversal towards the option with fewer, more salient features. We also include exact ties to contrast Pure Tallying's prediction of 50/50 guessing against the Competing Theory's deterministic choice.

[8] To cleanly dissociate the Advocated Theory (Additive Utility with power-law validity weighting) from the Competing Theory (Pure Tallying), this design exploits the Advocated Theory's ability to smoothly scale the influence of highly valid features via the 'gamma' parameter. We use five features with a steep drop-off in validity. Pure Tallying strictly counts feature-wise wins, meaning a single highly valid feature is easily outweighed by multiple low-validity features, and any tie in the number of wins results in a 50/50 guess. In contrast, the Advocated Theory predicts that subjects will assign disproportionate weight to the most valid features (especially for gamma >= 1). We construct trials where an option with fewer but highly valid wins competes against an option with more but lower-validity wins, as well as trials where the tally is perfectly tied but the validities heavily favor one option. This provides a clear quantitative dissociation across the spectrum of possible compensatory and non-compensatory behaviors.

[9] To cleanly dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (Additive Utility with power-law weighting), this design utilizes five features with a linear spread of validities. Pure Tallying strictly counts feature-wise wins and ignores the magnitudes of validities, predicting a 50/50 guess whenever the number of wins is tied, and consistently favoring the option with more wins. The Competing Theory scales features by chance-centered validities and can mimic Take-The-Best behavior at high gamma values. We construct trials where an option with fewer but highly valid wins competes against an option with more but lower-validity wins (separating Tallying from WADD), trials where the tally is tied but validities strongly favor one option (Tallying predicts guessing, WADD predicts a deterministic choice), and trials where an option wins on many low-to-medium validities against a single highest validity (separating WADD with low gamma from WADD with high gamma/TTB).

[10] To quantitatively dissociate the Advocated Theory (Additive Utility with a power-law transformation of chance-centered validities) from the Competing Theory (a mixture of Tallying and WADD with Bayesian log-odds weights), we exploit the divergent scaling properties of their weighting functions. The log-odds transformation in the Competing Theory's WADD component grows extremely fast as validity approaches 1.0, while the Advocated Theory's power-law function allows for a broader, smoother interpolation between equal weighting (Tallying) and Take-The-Best (TTB) via the gamma parameter. Furthermore, the Competing Theory explicitly mixes in a pure Tallying component. By pitting options with single high-validity features against options with multiple lower-validity features, we can create trials where the log-odds WADD and pure Tallying strongly conflict, forcing the Competing Theory to predict a diluted, mixture-driven choice probability. In contrast, the Advocated Theory can predict stark, deterministic choices depending on the gamma parameter, without the structural constraint of mixing two distinct choice distributions.

[11] To quantitatively dissociate the Advocated Theory (a mixture of Tallying and WADD with Bayesian log-odds weights) from the Competing Theory (a single-stage Additive Utility with power-law transformation of chance-centered validities), we exploit the divergent scaling properties of their weighting functions and the structural difference of a mixture model. The log-odds transformation in the Advocated Theory's WADD component grows extremely fast as validity approaches 1.0. Furthermore, the Advocated Theory explicitly mixes in a pure Tallying component. By pitting options with single high-validity features against options with multiple lower-validity features, we create trials where log-odds WADD and pure Tallying strongly conflict. The Advocated Theory will predict a mixture-driven choice probability (e.g., pulling probabilities toward the center depending on the mixture weight 'alpha'), whereas the Competing Theory, lacking a pure Tallying mixture and relying on a single power-law scaling, predicts a single deterministic choice direction depending solely on its 'gamma' parameter. We include trials with Tallying ties (where the Advocated Theory's Tallying component predicts exactly 50/50, leaving the decision entirely to the log-odds WADD component) and strong Tallying conflicts to fully map the dissociation.

[12] To quantitatively dissociate the Advocated Theory (Additive Utility with chance-centered validity weighting) from the Competing Theory (Rank-Based Weighting), we exploit the difference between magnitude-based and ordinal-based feature evaluation. The Advocated Theory scales weights based on the absolute magnitude of the validities, meaning that features with very similar validities will receive almost identical weights. In contrast, the Competing Theory relies strictly on ordinal ranks, enforcing a fixed drop-off in weight between rank 1, 2, and 3, regardless of how close their actual validities are. By using a set of validities with a tight cluster at the top (e.g., 0.95, 0.93, 0.91) and a drop-off to lower validities (0.60, 0.55), we construct trials where the sum of the 2nd and 3rd best features easily outweighs the 1st best feature in the Advocated Theory (because 0.93 and 0.91 are almost as good as 0.95), but fails to do so in the Competing Theory (because rank 2 and 3 are heavily penalized compared to rank 1). Conversely, we create trials where multiple low-rank features can outweigh a single higher-rank feature in the Competing Theory, but fail to do so in the Advocated Theory due to the massive gap in objective validity.

[13] This design quantitatively dissociates the Advocated Theory (Rank-Based Weighting) from the Competing Theory (Magnitude-Based Additive Utility) by exploiting the difference between ordinal ranks and absolute validities. The validities are chosen to have a tight cluster of high values (0.95, 0.90, 0.85) and a drop-off to lower values (0.60, 0.55). Because the Advocated Theory relies strictly on ordinal rank, it enforces a fixed, steep drop-off between Rank 1, 2, and 3 regardless of their numerical proximity. Consequently, it predicts that Rank 1 will robustly outweigh Ranks 2 and 3 combined. Conversely, the Competing Theory scales weights by the magnitude of chance-centered validities, meaning Ranks 1, 2, and 3 will receive very similar weights. Thus, the Competing Theory predicts that Ranks 2 and 3 combined will easily overpower Rank 1. By systematically pitting the single best feature against multiple slightly-less-valid features, we create a sharp double dissociation between rank-based and magnitude-based weighting.

[14] This design quantitatively dissociates the Advocated Theory (Additive Utility with power-law validity weighting) from the Competing Theory (Sequential Evidence Accumulation with a stopping threshold) by exploiting how they handle conflicting evidence across multiple features. The Competing Theory accumulates raw binary differences (+1 or -1) and either stops early (acting like pure Take-The-Best if threshold <= 1) or proceeds to the end, falling back on a pure feature tally. In contrast, the Advocated Theory continuously scales feature differences by a power-law transformation of their validities. By constructing trials where the running tally drops back to zero before the final feature, we create cases where the Competing Theory is forced to guess if its threshold is greater than 1, while the Advocated Theory predicts a deterministic preference due to the asymmetric validities of the features involved. Pitting these against trials where a high threshold is required to choose the option with more feature wins creates a logical contradiction for the Competing Theory, as it cannot simultaneously hold a low and high threshold to match the smoothly interpolated compensatory choices predicted by the Advocated Theory.

[15] To quantitatively dissociate the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (Additive Utility with power-law validities), we exploit how the Advocated Theory accumulates *unweighted* evidence differences. Because the Advocated Theory merely counts +1 or -1 in validity order, it can never differentiate between the magnitudes of features once the order is set, except by stopping early. By constructing trials where the running tally of differences fluctuates but never reaches a high threshold, we force the Advocated Theory to either guess (if it reaches the end with a tie) or choose the Take-The-Best option (if the threshold is low). In contrast, the Competing Theory weights features by their chance-centered validities, allowing combinations of slightly less valid features to overpower the single best feature, predicting a strong preference for an option that the Advocated Theory can literally never choose under any parameterization.

[16] To quantitatively dissociate the Advocated Theory (a Strategy Selection/Mixture model of TTB, Tallying, and WADD) from the Competing Theory (Sequential Evidence Accumulation with a stopping threshold), we exploit the fact that Sequential Accumulation only maintains a running sum of unweighted differences (+1 or -1) in validity order. If the threshold is 1, it acts exactly like Take-The-Best (TTB). If the threshold is high, it falls back to Pure Tallying. Consequently, if TTB and Tallying both agree on an option, Sequential Accumulation MUST choose that option, regardless of its threshold parameter. The Advocated Theory, however, includes WADD as a sub-model, which weights features by their log-odds validities. By selecting a specific set of validities (e.g., 0.95, 0.90, 0.85, 0.60, 0.55), we can construct trials where the single best feature (F1) and two weak features (F4, F5) support Option A, while two strong intermediate features (F2, F3) support Option B. In this scenario, TTB chooses A (wins F1) and Tallying chooses A (3 wins vs 2). Thus, the Competing Theory strictly predicts A. However, the log-odds sum for B (F2+F3) outweighs A (F1+F4+F5), meaning WADD chooses B. The Advocated Theory can therefore predict a substantial probability of choosing B, achieving a sharp qualitative and quantitative dissociation.

[17] To quantitatively dissociate the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (a Strategy Selection mixture of TTB, Tallying, and WADD), we exploit the Advocated Theory's unweighted accumulation and early stopping mechanism. By setting a specific sequence of validities, we can construct trials where the accumulated difference temporarily reaches a threshold (e.g., +2) favoring Option A, causing the Advocated Theory to stop early and choose A. However, by ensuring that Option B wins on the single most valid feature (so TTB chooses B), wins on a greater total number of features (so Tallying chooses B), and wins on the log-odds weighted sum of features (so WADD chooses B), the Competing Theory is structurally forced to predict a strict preference for B across all of its sub-models. This creates a stark qualitative and quantitative dissociation where the Advocated Theory can predict a choice that is literally impossible under the Competing Theory's framework.

[18] To quantitatively dissociate the Advocated Theory (Validity-Weighted Sequential Accumulation) from the Competing Theory (Unweighted Sequential Evidence Accumulation), we exploit the difference between accumulating continuous weighted evidence versus discrete unweighted (+1/-1) steps. By using a steep set of validities, we can construct trials where the unweighted running tally fluctuates between -1 and +1, ending on -1 (so the Competing Theory predicts a strict preference for B regardless of its threshold, as early stopping at 1 also favors B). In contrast, the Advocated Theory weights features by a non-linear transformation of their chance-centered validities. Thus, a combination of the 2nd and 3rd best features can overpower the 1st, 4th, and 5th best features, resulting in a positive final weighted tally. This creates a stark dissociation where the Competing Theory structurally must choose B, while the Advocated Theory can robustly choose A.

[19] To quantitatively dissociate the Advocated Theory (Unweighted Sequential Evidence Accumulation) from the Competing Theory (Validity-Weighted Sequential Accumulation), we exploit the difference between unweighted (+1/-1) and weighted evidence accumulation. By setting validities with a tight high cluster (0.95, 0.94, 0.93) and a drop-off (0.60, 0.55), we can construct trials where the unweighted running tally fluctuates but structurally forces the Advocated Theory to predict one option regardless of its threshold, whereas the Competing Theory's weighted sum overcomes the early deficit and predicts the opposite option for any non-minimal threshold.

[20] To quantitatively dissociate the Advocated Theory (Two-Phase Heuristic: TTB with Tallying Fallback) from the Competing Theory (Sequential Evidence Accumulation), we exploit their divergent behavior when the most valid feature is tied. The Advocated Theory posits that if the best feature is tied, decision-makers abandon sequential search entirely and tally all remaining features. The Competing Theory, however, continues sequential accumulation and can still stop early on the second or third feature if its threshold is met. By constructing trials where the most valid feature is tied, the second most valid feature favors Option A, but the remaining features favor Option B, we create a sharp dissociation. The Advocated Theory will tally the remaining features and predict a robust preference for Option B. In contrast, the Competing Theory (with a low threshold) will stop early on the second feature and strongly predict Option A.

[21] To quantitatively dissociate the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (Two-Phase Heuristic: TTB with Tallying Fallback), we exploit their divergent behavior depending on whether the most valid feature is tied. The Competing Theory strictly uses the best feature if it discriminates; if tied, it abandons sequential search and tallies the rest. The Advocated Theory strictly accumulates differences (+1/-1) sequentially and stops if a threshold is reached. By constructing trials where the first feature is tied but the second feature strongly favors Option A while the remaining features favor Option B, the Competing Theory tallies and chooses B, whereas the Advocated Theory (with a low threshold) stops early on the second feature and chooses A. Conversely, when the first feature is NOT tied but all other features favor the opposite option, the Competing Theory chooses the TTB option, while the Advocated Theory (with a high threshold) will tally and choose the opposite.

[22] To quantitatively dissociate the Advocated Theory (Weighted Additive Strategy, WADD) from the Competing Theory (Sequential Evidence Accumulation), we exploit the difference between magnitude-based weighting and unweighted sequential tallying. By using a tight cluster of high validities (0.95, 0.94, 0.93) and a drop-off to lower validities (0.60, 0.55), we can construct trials where the unweighted running tally of the Competing Theory fluctuates but structurally forces it to predict one option regardless of its threshold, whereas the Advocated Theory's weighted sum overcomes the early deficit and predicts the opposite option. For example, in a trial where Option B wins on the 1st, 4th, and 5th features, and Option A wins on the 2nd and 3rd, the Competing Theory's unweighted tally starts at -1 (favoring B), goes to 0, +1, 0, and ends at -1, meaning it will choose B whether it stops early (threshold <= 1) or tallies to the end. In contrast, the Advocated Theory heavily weights the 2nd and 3rd features, allowing their combined magnitude to robustly overpower the 1st feature, predicting a strict preference for A across a wide range of its gamma parameter.

[23] To quantitatively dissociate the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (Weighted Additive Strategy), we exploit the difference between unweighted sequential accumulation and magnitude-weighted compensatory evaluation. By using a tight cluster of high validities (0.95, 0.94, 0.93) and a drop-off (0.60, 0.55), we can construct trials where the unweighted running tally of the Advocated Theory fluctuates but structurally forces it to predict one option regardless of its threshold, whereas the Competing Theory's weighted sum overcomes the early deficit and predicts the opposite option. For example, when Option B wins on the 1st, 4th, and 5th features, and Option A wins on the 2nd and 3rd, the Advocated Theory's unweighted tally starts at -1 (favoring B), goes to 0, +1, 0, and ends at -1, meaning it will choose B whether it stops early or tallies to the end. In contrast, the Competing Theory heavily weights the 2nd and 3rd features, allowing their combined magnitude to robustly overpower the 1st feature, predicting a strict preference for A across a wide range of its scaling parameter.

[24] To quantitatively dissociate the Advocated Theory (Strategy Mixture Theory blending TTB and a WADD/Tallying compensatory process) from the Competing Theory (Sequential Evidence Accumulation with a stopping threshold), we exploit the structural limitations of unweighted sequential accumulation. The Competing Theory evaluates features in validity order, maintaining a running unweighted tally (+1/-1) and stopping if a threshold is reached, or defaulting to the final tally if it isn't. Consequently, if an option is favored by the first discriminating feature (TTB) AND has a higher overall number of feature wins (Tallying), and the running tally never strongly favors the other option, the Competing Theory MUST predict that option regardless of its threshold parameter. The Advocated Theory, however, incorporates a Weighted Additive (WADD) component that scales features by their chance-centered validities. By using validities with a tight cluster at the top (0.95, 0.90, 0.85) and a drop-off (0.60, 0.55), we construct 'trap' trials where Option A wins the 1st, 4th, and 5th features (winning TTB and Tallying), meaning the Competing Theory structurally must choose A. However, Option B wins the 2nd and 3rd features, which combined carry more compensatory weight in the Advocated Theory than A's features. This creates a sharp dissociation where the Advocated Theory can robustly predict B, a choice that is mathematically impossible under the Competing Theory.

[25] This design quantitatively dissociates the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (Strategy Mixture: TTB + Compensatory WADD/Tallying). The dissociation exploits the Advocated Theory's unweighted sequential tally and early stopping mechanism. By using 7 features with a specific validity profile, we construct 'trap' trials where Option A wins features 2, 3, and 4, while Option B wins feature 1 and features 5, 6, and 7. For the Advocated Theory, the running unweighted tally reaches +2 in favor of A at feature 4, triggering an early stop (if threshold = 2) and a robust choice for A. However, for the Competing Theory, Option B is structurally forced to win across all its sub-models: TTB chooses B (wins feature 1), Tallying chooses B (4 wins vs 3), and WADD chooses B (the massive weight of feature 1 plus the tail outscales features 2-4 for all gamma >= 0). Thus, the Competing Theory mathematically must predict B, while the Advocated Theory uniquely predicts A.

[26] This design quantitatively dissociates the Advocated Theory (Top-K Majority Heuristic) from the Competing Theory (Sequential Evidence Accumulation) by exploiting the structural differences between fixed-set majority voting and sequential running tallies. We use 5 features with descending validities. In 'Trap 1' trials, Option A wins the 1st, 4th, and 5th features, while Option B wins the 2nd and 3rd. The Competing Theory's running tally starts at +1 (favoring A), drops to 0, then -1, back to 0, and ends at +1. Because it either stops early at the 1st feature (choosing A) or tallies to the end (choosing A), the Competing Theory structurally MUST predict Option A regardless of its threshold parameter. However, the Advocated Theory evaluates fixed sets (Top K). If K=3, Option B wins the majority (features 2 and 3 vs feature 1), allowing the Advocated Theory to robustly predict Option B. We include mirror 'Trap 2' trials and standard dissociation trials (e.g., A wins 1st and 2nd, B wins 3rd, 4th, 5th) to fully map the parameter space for both models, highlighting choices that are mathematically impossible under the Competing Theory but predicted by the Advocated Theory.

[27] This design quantitatively dissociates the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (Top-K Majority Heuristic) by exploiting the structural constraints of unweighted running tallies versus fixed-set majority voting. In 'Trap' trials (e.g., Option A wins features 1, 4, 5; Option B wins features 2, 3), the running tally of differences fluctuates (+1, 0, -1, 0, +1). Because the absolute tally never exceeds 1 and ends at +1, the Advocated Theory structurally MUST predict Option A regardless of its stopping threshold parameter (it either stops at feature 1 and chooses A, or tallies to the end and chooses A). However, the Competing Theory evaluates fixed subsets (e.g., Top K=3). Within the top 3 features, Option B holds a strict majority (winning features 2 and 3 versus A's feature 1), allowing the Competing Theory to robustly predict Option B. By varying the winning features across trials, we create a sharp double dissociation where each model can be forced into a prediction that is mathematically impossible for the other.

[28] To quantitatively dissociate the Advocated Theory (Top-K Majority Heuristic) from the Competing Theory (Rank-Weighted Additive with Baseline Feature Decay), we exploit the structural differences between unweighted majority voting within a fixed subset and a compensatory weighted sum over all features. By presenting pairs where one option holds a majority among the top 2 or 3 features, while the other option wins the highest validity feature and accumulates lower-tier features, we create diametrically opposed predictions. The Top-K model enforces a strict cutoff and evaluates unweighted tallies within that window, whereas the Rank-Weighted Additive model applies exponentially decaying weights and a baseline, allowing combinations of the 1st and lower-ranked features to outweigh the 2nd and 3rd features, or vice versa depending on the decay parameter.

[29] This design quantitatively dissociates the Advocated Theory (Rank-Weighted Additive with Baseline Feature Decay) from the Competing Theory (Top-K Majority Heuristic with Proportional Confidence). The Top-K model imposes a hard cutoff, only evaluating a subset of the most valid features (e.g., the top 3) and performing an unweighted tally within that set. In contrast, the Rank-Weighted Additive model evaluates all features using exponentially decaying weights plus a baseline, allowing a sufficient number of lower-ranked features to compensate for higher-ranked ones, especially when the baseline weight is high. By constructing trials where one option wins a majority within the top 2 or 3 features, but the other option wins the highest validity feature along with several lower-tier features, we create diametrically opposed predictions. The Top-K model will strictly favor the option with the majority in the consideration set, whereas the Rank-Weighted Additive model can smoothly shift its preference depending on the decay and baseline parameters.

[30] This design quantitatively dissociates the Advocated Theory (Top-K Majority Heuristic) from the Competing Theory (Pure Tallying with Validity-Weighted Tie-Breaking) by exploiting the difference between evaluating a fixed subset of top features versus tallying all features. We construct trials where one option wins the overall unweighted tally across all 5 features (so the Competing Theory predicts it will win), but the other option holds a majority within the top 3 most valid features (so the Advocated Theory predicts it will win). Furthermore, we include 'tie' trials where both options win on an equal number of features overall. In these cases, the Competing Theory falls back on a validity-weighted sum across all features, while the Advocated Theory strictly tallies within the Top-K subset, leading to diametrically opposed predictions.

[31] To quantitatively dissociate the Advocated Theory (Pure Tallying with a Validity-Weighted Tie-Breaker) from the Competing Theory (Top-K Majority Heuristic), we exploit the difference between global unweighted tallying and subset-restricted majority voting. The Advocated Theory primarily counts all feature wins regardless of validity, only falling back to validities if the tallies are exactly tied. The Competing Theory only tallies within the top K most valid features. We construct 'conflict' trials where one option wins the global tally (favored by the Advocated Theory) but loses within the top 1, 2, or 3 features (favored by the Competing Theory). We also include 'tie' trials where the global tally is equal, forcing the Advocated Theory to use its validity-weighted tie-breaker, while the Competing Theory's prediction depends strictly on the Top-K subset.

[32] To quantitatively dissociate the Advocated Theory (Tallying with Lexicographic/TTB Tie-Breaking) from the Competing Theory (Pure Tallying with Validity-Weighted Tie-Breaking), we must focus on trials where the unweighted tallies of positive features are exactly tied. Both models share the primary tallying stage and will make identical predictions when one option has more feature wins. However, when tallies are tied, the Advocated Theory uses a non-compensatory lexicographic rule, choosing the option favored by the single most valid discriminating feature. In contrast, the Competing Theory uses a compensatory validity-weighted sum, allowing multiple slightly less valid features to outweigh the single best feature. By constructing tied-tally trials where one option wins on the highest validity feature but the other wins on multiple high-to-intermediate validity features, we create a sharp double dissociation. Filler trials where one option clearly wins the tally, as well as tie trials where both tie-breakers agree, are included to provide baseline measurements for the shared tallying mechanism.

[33] To quantitatively dissociate the Advocated Theory (Pure Tallying with Validity-Weighted Tie-Breaking) from the Competing Theory (Tallying with Lexicographic Tie-Breaking), we must focus on trials where the unweighted tallies of positive features are exactly tied. Both models share the primary tallying stage and will make identical predictions when one option has more feature wins. However, when tallies are tied, the Competing Theory uses a non-compensatory lexicographic rule, choosing the option favored by the single most valid discriminating feature. In contrast, the Advocated Theory uses a compensatory validity-weighted sum, allowing multiple slightly less valid features to outweigh the single best feature. By constructing tied-tally trials where one option wins on the highest validity feature but the other wins on multiple high-to-intermediate validity features, we create a sharp double dissociation. Filler trials where one option clearly wins the tally, as well as tie trials where both tie-breakers agree, are included to provide baseline measurements for the shared tallying mechanism.

[34] To dissociate the Advocated Theory (Tallying with Lexicographic Tie-Breaking) from the Competing Theory (Probabilistic Mixture of WADD and Tallying), we focus on trials where the unweighted tallies of features are exactly tied. In these cases, the Advocated Theory relies on a lexicographic (TTB) tie-breaker, choosing the option favored by the single most valid discriminating feature. In contrast, the Competing Theory's Tallying component predicts guessing, leaving the decision entirely to its WADD component, which evaluates the compensatory sum of chance-centered validities. By constructing tied-tally trials where one option wins on the highest validity feature but the other wins on multiple highly valid features (e.g., Rank 1 vs Ranks 2, 3, and 4), we create a sharp double dissociation. We also include trials with unequal tallies where WADD and Tallying conflict to further identify the mixture proportion in the Competing Theory.

[35] To cleanly dissociate the Advocated Theory (a probabilistic mixture of WADD and Tallying) from the Competing Theory (Tallying with Lexicographic/TTB tie-breaking), we focus on trials where the unweighted tallies of features are exactly tied. In these tied-tally cases, the Competing Theory relies on a lexicographic tie-breaker, decisively choosing the option favored by the single most valid discriminating feature. In contrast, the Advocated Theory's Tallying component predicts a 50/50 guess, leaving the decision entirely to its WADD component, which evaluates the compensatory sum of chance-centered validities. By constructing tied-tally trials where one option wins on the highest validity feature but the other wins on multiple highly valid features (e.g., Rank 1 vs Ranks 2 and 3), we create a sharp double dissociation. We also include trials with unequal tallies where WADD and Tallying conflict to further identify the mixture proportion in the Advocated Theory, as the Competing Theory will deterministically follow the tally.

[36] To quantitatively dissociate the Advocated Theory (Rank-Weighted Additive Model) from the Competing Theory (Probabilistic Mixture of WADD and Tallying), we exploit the divergent predictions between rank-based exponential weighting and magnitude-based chance-centered weighting. By selecting validities with a tight cluster at the top (0.95, 0.90, 0.85) and a drop-off to lower validities (0.60, 0.55), we can construct trials where both the WADD and Tallying components of the Competing Theory agree on a choice, structurally forcing the Competing Theory to predict that option. However, because the Advocated Theory relies on exponentially decaying rank weights, it can easily predict the opposite option for low-to-moderate values of its gamma parameter (acting more like Take-The-Best). Conversely, we create trials where the Advocated Theory mathematically must predict one option across all values of gamma, but the Competing Theory's WADD and Tallying components conflict, allowing it to predict the opposite option depending on its mixture proportion.

[37] This design quantitatively dissociates the Advocated Theory (Probabilistic Mixture of WADD and Tallying) from the Competing Theory (Rank-Weighted Additive Model) by exploiting the difference between magnitude-based weighting and rank-based exponential weighting. By using validities with a tight cluster at the top (0.95, 0.90, 0.85) and a drop-off (0.60, 0.55), we can construct trials where both the WADD and Tallying components of the Advocated Theory agree on a choice, structurally forcing the Advocated Theory to predict that option. However, because the Competing Theory relies on exponentially decaying rank weights, it can easily predict the opposite option for low-to-moderate values of its gamma parameter (acting more like Take-The-Best). Conversely, we create trials where the Competing Theory mathematically must predict one option across all values of gamma (because the sum of specific rank weights always dominates), but the Advocated Theory's WADD and Tallying components conflict, allowing it to predict the opposite option depending on its mixture proportion.

[38] To quantitatively dissociate the Advocated Theory (Lexicographic Semi-Order Evidence Accumulation with Tallying Fallback) from the Competing Theory (Probabilistic Mixture of WADD and Tallying), we exploit the Advocated Theory's early-stopping mechanism and its uncentered validity accumulation. By constructing 'trap' trials where Option B wins the first feature but Option A wins all remaining features (e.g., A=[0,1,1,1,1] vs B=[1,0,0,0,0]), the Advocated Theory can stop early and robustly choose B if its threshold is low. However, the Competing Theory's WADD component (using chance-centered validities) and Tallying component both strongly favor A, forcing the Competing Theory to predict A regardless of its mixture parameter. Conversely, we include trials where Option A wins features 2, 3, and 4 while Option B wins features 1 and 5. Here, both WADD and Tallying agree on A, structurally forcing the Competing Theory to choose A. Yet, the Advocated Theory can still stop early on feature 1 and choose B. These opposing structural constraints provide a sharp double dissociation.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Advocated Theory (Probabilistic Mixture of WADD and Tallying) from the Competing Theory (Lexicographic Semi-Order Evidence Accumulation with Tallying Fallback), we exploit the Competing Theory's sequential raw validity accumulation and its fallback to pure tallying. We design 'trap' trials where both the WADD component (using chance-centered validities) and the Tallying component of the Advocated Theory agree on an option, structurally forcing the Advocated Theory to predict that option. However, the Competing Theory can stop early on the first feature and predict the opposite. Conversely, we create trials where the Competing Theory is structurally forced to predict one option (because it either stops early on the first feature or falls back to a tally that favors the same option), while the Advocated Theory's WADD component strongly favors the opposite option, allowing it to predict a choice that is mathematically impossible under the Competing Theory.",
  "validities": [
    0.95,
    0.9,
    0.85,
    0.6,
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
      1,
      1,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
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
      1
    ],
    [
      1,
      0,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 11021,
  "candidates_token_count": 481,
  "total_token_count": 13904
}
```
