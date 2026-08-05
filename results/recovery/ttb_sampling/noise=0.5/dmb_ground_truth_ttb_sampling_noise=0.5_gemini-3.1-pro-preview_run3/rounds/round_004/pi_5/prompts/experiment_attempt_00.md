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
**Description:** Probabilistic Cue Search (Sequential Evidence Accumulation): Decision-makers evaluate features sequentially in descending order of validity. However, instead of strictly stopping at the first discriminating cue (as in pure Take-The-Best) or exhaustively accumulating all cues (as in Weighted Additive), they exhibit a probabilistic stopping rule. After evaluating each discriminating cue and updating their internal evidence, they stop searching and make a choice with probability `theta`. If they do not stop, they continue to the next discriminating cue, accumulating its evidence. This naturally produces a graded interpolation between non-compensatory and compensatory decision-making without relying on a discrete mixture of distinct strategies.

**Parameters:**
- theta: [0.0, 1.0]
- beta: [0.1, 20.0]
- gamma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Subjective weights normalized so the sum of weights is 1.0
    w = validities ** gamma
    if np.sum(w) > 0:
        w = w / np.sum(w)
    
    # Sort all features by validity descending
    order = np.argsort(validities)[::-1]
    
    # Find discriminating cues in order of validity
    discrim_indices = [i for i in order if a[i] != b[i]]
    
    if len(discrim_indices) == 0:
        p_core = np.array([0.5, 0.5])
    else:
        p_A_total = 0.0
        p_B_total = 0.0
        p_continue = 1.0
        
        score_A = 0.0
        score_B = 0.0
        
        for j, i in enumerate(discrim_indices):
            # Accumulate evidence from the current discriminating cue
            score_A += w[i] * a[i]
            score_B += w[i] * b[i]
            
            # Determine stopping probability
            if j < len(discrim_indices) - 1:
                p_stop = theta
            else:
                p_stop = 1.0  # Must stop at the last discriminating cue
                
            # Softmax over accumulated scores so far
            z_A = beta * score_A
            z_B = beta * score_B
            max_z = max(z_A, z_B)
            e_A = np.exp(z_A - max_z)
            e_B = np.exp(z_B - max_z)
            p_A_given_stop = e_A / (e_A + e_B)
            p_B_given_stop = e_B / (e_A + e_B)
            
            # Marginalize over the stopping probability
            p_A_total += p_continue * p_stop * p_A_given_stop
            p_B_total += p_continue * p_stop * p_B_given_stop
            
            # Update the probability of continuing to the next cue
            p_continue *= (1.0 - p_stop)
            
        p_core = np.array([p_A_total, p_B_total])
        
    # Apply uniform lapse
    n_opts = 2
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Strategy Mixture Theory: Decision-makers do not uniformly apply a single choice rule. Instead, they possess a repertoire of strategies and flexibly draw from them. On any given trial, a subject acts as a mixture model, choosing to apply a non-compensatory heuristic (Take-The-Best) with probability alpha, and a compensatory rule (Weighted Additive / Tallying) with probability 1 - alpha. The compensatory rule weights features by its subjective validities, naturally subsuming Tallying and WADD. Crucially, the compensatory scores are normalized to the [0, 1] scale to perfectly match the scale of the heuristic's discrete scores, allowing a single temperature parameter to symmetrically control the determinism of both strategies without numerical compromise.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- gamma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture model expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) Prediction ---
    order = np.argsort(validities)[::-1]
    a, b = stim[0], stim[1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    z_ttb = beta * (ttb_scores - ttb_scores.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- Compensatory (WADD/Tallying) Prediction ---
    # Subjective validities: gamma=0 yields Tallying, gamma=1 yields strict WADD
    subjective_weights = validities ** gamma
    wadd_scores = stim @ subjective_weights
    
    # Normalize WADD scores to [0, 1] scale to match TTB scores
    wadd_scores = wadd_scores / np.sum(subjective_weights)
    
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Strategy Mixture ---
    p_core = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # --- Uniform Lapse ---
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design systematically dissociates Tallying from the Weighted Additive (WADD) rule by pitting a larger number of low-validity features against a smaller number of high-validity features. Tallying simply counts the number of features on which an option is superior, ignoring the validities entirely. WADD weights each feature by its subjective validity. By selecting a steep drop-off in validities (e.g., 0.95 and 0.90 vs. 0.60, 0.55, 0.50), we can create trials where one option wins on more features (favored by Tallying) but the other option has a higher validity-weighted sum (favored by WADD). We also include trials where Tallying predicts a tie (equal number of winning features) while WADD strongly prefers one option, as well as control trials where both models agree.

[1] This design systematically dissociates the Weighted Additive (WADD) theory from the Tallying heuristic by exploiting the difference between validity-weighted sums and unweighted feature-win counts. We use a set of 5 features with a steep drop-off in validities. In some trials, one option wins on more features (favored by Tallying) but the other option has a higher validity-weighted sum (favored by WADD). We also include trials where Tallying predicts a tie (equal number of winning features) while WADD strongly prefers one option, as well as trials where the models agree to ensure a balanced design. This allows for a robust quantitative dissociation between the two choice mechanisms.

[2] This design uses a 6-feature environment with a pronounced drop-off in validities to robustly separate Tallying from WADD. Tallying counts the number of features favoring each option, ignoring validities. WADD computes a weighted sum. By pitting a small number of high-validity features against a larger number of low-validity features, we create critical trials where Tallying chooses the option with more winning features while WADD chooses the option that wins on the most important ones. We also include trials where Tallying predicts a tie but WADD strongly prefers one option, as well as trials where both models agree. This diversity of trial types ensures a clear quantitative dissociation.

[3] To robustly dissociate the Weighted Additive (WADD) rule from the Tallying heuristic, this design employs a 5-feature environment with a pronounced bimodal distribution of validities (two highly valid features and three low-validity features). Tallying counts the number of strict feature-wise wins, treating all features equally regardless of validity. In contrast, WADD scales the feature values by their subjective validities. By pairing options where one wins on a few high-validity features while the other wins on a greater number of low-validity features, Tallying predicts a choice for the latter, whereas WADD predicts a choice for the former. We also include trials where Tallying predicts a tie (equal number of winning features) but WADD strongly prefers the option with higher-validity features, as well as trials where the models agree, providing a comprehensive quantitative dissociation.

[4] This design robustly dissociates the non-compensatory Take-The-Best (TTB) heuristic from the compensatory Weighted Additive (WADD) rule. By using a set of features with a steep drop-off in validities (e.g., one highly predictive feature and several moderately predictive ones), we can construct trials where the most valid feature strongly favors one option, but the combined weight of the remaining features favors the other. TTB will strictly choose the option endorsed by the single highest-validity discriminating feature and ignore the rest. WADD, by contrast, integrates all features, allowing a deficit on the highest-validity cue to be compensated by a sufficient advantage on the lower-validity cues. We also include trials where both models agree, as well as trials where the critical dissociation occurs on the second most valid feature (when the first feature is tied).

[5] This design robustly dissociates the Weighted Additive (WADD) rule from the non-compensatory Take-The-Best (TTB) heuristic. By using a set of five features with linearly decreasing validities, we can create critical 'compensatory' trials where the single highest-validity discriminating feature favors one option (which TTB will strictly choose), but the sum of the remaining lower-validity features favors the other option (which WADD will choose). We include variations where the highest-validity feature is tied, shifting the TTB decision to the second or third feature, while WADD still integrates all available information. Control trials where both models agree are also included.

[6] To quantitatively dissociate pure Take-The-Best (TTB) from the Strategy Mixture Theory (which posits a mixture of TTB and compensatory strategies like WADD), we need trials where TTB and WADD make strongly opposing predictions. By using an environment where the most valid cue is outweighed by the sum of the remaining cues, we create 'compensatory' trials. Pure TTB will consistently and confidently choose the option favored by the highest-validity discriminating cue. In contrast, the Strategy Mixture Theory will predict a softer, mixed probability, or even a preference for the opposite option (depending on the mixture weight alpha), because the WADD component strongly pulls the prediction in the opposite direction. Including trials where the first cue is tied extends this dissociation to lower-validity cues.

[7] To quantitatively dissociate pure Take-The-Best (TTB) from Strategy Mixture Theory, we construct a series of 'compensatory' trials where the single highest-validity discriminating cue strongly favors one option, but the sum of the remaining cues strongly favors the other option. Pure TTB strictly follows the highest-validity cue and ignores the rest, predicting a deterministic choice for the TTB-favored option. In contrast, the Strategy Mixture Theory blends TTB with a compensatory rule (WADD/Tallying). Because the compensatory component strongly favors the opposite option, the mixture model predicts a softer probability distribution, systematically deviating from pure TTB. By including trials where the first cue is tied, we extend this dissociation to secondary and tertiary cues.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design aims to quantitatively dissociate Probabilistic Cue Search (PCS) from Strategy Mixture Theory (SMT) by exploiting how they handle intermediate evidence. SMT posits a discrete mixture between a single-cue heuristic (TTB) and an exhaustive compensatory rule (WADD). In contrast, PCS accumulates evidence sequentially across discriminating cues, with a probability of stopping at each step. By designing trials where the first cue favors Option A, but the subsequent cues sequentially build a strong case for Option B before a final cue swings the balance back or maintains it, we can expose PCS's unique intermediate stopping states. For example, trials where the accumulated evidence briefly favors B in the middle of the search will pull PCS's predicted choice probabilities in ways that SMT (which only sees the first cue or the total sum) cannot mimic. We include a mix of trials where TTB and WADD agree, where they conflict strongly, and where intermediate cues create 'traps' for sequential accumulators.",
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
      0,
      0,
      1,
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
      1,
      0,
      1,
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
      1,
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
      0,
      1,
      1,
      0,
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
      1,
      1
    ],
    [
      1,
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
  "prompt_token_count": 4332,
  "candidates_token_count": 482,
  "total_token_count": 8216
}
```
