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
**Description:** Decision-makers use a Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity, maintaining a running sum of the differences between the options. If the absolute accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally. This allows the model to smoothly transition between Take-The-Best (low threshold) and Tallying (high threshold) behaviors.

**Parameters:**
- threshold: [0.0, 3.0]
- beta: [0.1, 15.0]
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
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
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
**Description:** Heuristic Toolbox / Strategy Selection Theory with Graded Sub-models: Decision-makers possess a repertoire of distinct decision heuristics (Take-The-Best, Tallying, and Weighted Additive). Instead of assuming pure determinism within each heuristic, each strategy produces graded choice probabilities via a softmax rule applied to its internal evidence (e.g., validity of the discriminating cue for TTB, cue sum for Tallying, log-odds sum for WADD) using a shared choice temperature. Individuals probabilistically select one of these softened heuristics on any given trial.

**Parameters:**
- w_ttb: [0.0, 50.0]
- w_tally: [0.0, 50.0]
- w_wadd: [0.0, 50.0]
- beta_choice: [0.1, 10.0]
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
    
    w_ttb = float(parameters["w_ttb"])
    w_tally = float(parameters["w_tally"])
    w_wadd = float(parameters["w_wadd"])
    beta_choice = float(parameters["beta_choice"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize weights to get mixture probabilities for the strategies
    total_w = w_ttb + w_tally + w_wadd + 1e-9
    p_ttb = w_ttb / total_w
    p_tally = w_tally / total_w
    p_wadd = w_wadd / total_w
    
    # Strategy 1: Take-The-Best (TTB) Evidence
    order = np.argsort(val)[::-1]
    ttb_score_a, ttb_score_b = 0.0, 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_score_a, ttb_score_b = val[idx], 0.0
            break
        elif b[idx] > a[idx]:
            ttb_score_a, ttb_score_b = 0.0, val[idx]
            break
            
    z_ttb = beta_choice * np.array([ttb_score_a, ttb_score_b])
    z_ttb -= np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    ttb_probs = e_ttb / np.sum(e_ttb)
    ttb_a, ttb_b = ttb_probs[0], ttb_probs[1]
            
    # Strategy 2: Tallying (Equal Weights) Evidence
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    z_tally = beta_choice * np.array([sum_a, sum_b])
    z_tally -= np.max(z_tally)
    e_tally = np.exp(z_tally)
    tally_probs = e_tally / np.sum(e_tally)
    tally_a, tally_b = tally_probs[0], tally_probs[1]
        
    # Strategy 3: Weighted Additive (WADD) Evidence
    val_clipped = np.clip(val, 0.501, 0.999)
    w = np.log(val_clipped / (1.0 - val_clipped))
    score_a = np.sum(a * w)
    score_b = np.sum(b * w)
    
    z_wadd = beta_choice * np.array([score_a, score_b])
    z_wadd -= np.max(z_wadd)
    e_wadd = np.exp(z_wadd)
    wadd_probs = e_wadd / np.sum(e_wadd)
    wadd_a, wadd_b = wadd_probs[0], wadd_probs[1]
        
    # Probabilistic mixture of discrete strategy predictions
    p_A_core = p_ttb * ttb_a + p_tally * tally_a + p_wadd * wadd_a
    p_B_core = p_ttb * ttb_b + p_tally * tally_b + p_wadd * wadd_b
    p_core = np.array([p_A_core, p_B_core])
    
    # Apply lapse rate
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (a Strategy Selection mixture of TTB, Tallying, and WADD), we exploit the Advocated Theory's unweighted accumulation and early stopping mechanism. By setting a specific sequence of validities, we can construct trials where the accumulated difference temporarily reaches a threshold (e.g., +2) favoring Option A, causing the Advocated Theory to stop early and choose A. However, by ensuring that Option B wins on the single most valid feature (so TTB chooses B), wins on a greater total number of features (so Tallying chooses B), and wins on the log-odds weighted sum of features (so WADD chooses B), the Competing Theory is structurally forced to predict a strict preference for B across all of its sub-models. This creates a stark qualitative and quantitative dissociation where the Advocated Theory can predict a choice that is literally impossible under the Competing Theory's framework.",
  "validities": [
    0.95,
    0.9,
    0.85,
    0.8,
    0.75,
    0.7,
    0.65
  ],
  "trial_a_ratings": [
    [
      0,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
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
      1,
      0
    ],
    [
      1,
      1,
      0,
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
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
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
      0,
      0,
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
      1,
      1,
      1
    ],
    [
      0,
      1,
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
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
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
  "prompt_token_count": 6532,
  "candidates_token_count": 665,
  "total_token_count": 11157
}
```
