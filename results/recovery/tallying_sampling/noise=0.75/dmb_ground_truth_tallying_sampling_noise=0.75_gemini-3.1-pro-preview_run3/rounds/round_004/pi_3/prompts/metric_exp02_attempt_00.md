# metric_exp02_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=7):** [0.9, 0.8, 0.7, 0.6, 0.55, 0.55, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 2: A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  trial 4: A=[1, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1]

**Rationale:** To robustly dissociate a pure Tallying model from a Tallying/Take-The-Best (TTB) mixture model, we use a 7-feature design. This allows for a wider range of feature sum differences (Δ). The pure Tallying model's choice probabilities depend exclusively on Δ. In contrast, the mixture model's probabilities are systematically perturbed by the TTB component. We include 'Tie' trials (Δ = 0) where pure Tallying predicts exactly 50/50 guessing, but the mixture model predicts a bias toward the TTB-favored option. We also include trials with varying Δ where TTB either aligns with or opposes Tallying, testing whether the choice probabilities exhibit the TTB-induced shifts predicted by the mixture model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers use a 'Tallying' (Equal-Weight) heuristic, a compensatory strategy that ignores cue validities. They simply count the number of positive features (or advantages) each option has and choose the option with the highest total count. If the counts are equal, they guess. Because pure Tallying makes choices that strongly oppose Take The Best on compensatory trials, high levels of choice stochasticity (noise) are needed to pull the predicted consistency up toward the observed ~0.40-0.42 range, reflecting uncertainty or lapses in applying the heuristic.

**Parameters:**
- beta: [0.01, 1.0]
- epsilon: [0.3, 0.8]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: sum the unweighted feature values for each option.
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Decision-makers probabilistically sample between a compensatory Equal-Weight (Tallying) heuristic and a non-compensatory Lexicographic (Take-The-Best) heuristic on a trial-by-trial basis, but they heavily favor Tallying. Furthermore, when they do use Take-The-Best, the application of the rule is subject to decision noise (modeled via a softmax temperature), reflecting uncertainty or stochasticity in identifying the most valid cue rather than a perfectly deterministic lexicographic choice.

**Parameters:**
- beta: [0.01, 5.0]
- beta_ttb: [0.01, 5.0]
- w_tally: [0.5, 1.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying: sum the unweighted feature values for each option
    scores_tally = stim.sum(axis=1)
    beta = float(parameters["beta"])
    z = beta * (scores_tally - scores_tally.max())
    e = np.exp(z)
    p_tally = e / e.sum()
    
    # Take-The-Best (TTB): find the first discriminating cue ordered by validity
    order = np.argsort(validities)[::-1]
    diff = stim[0, order] - stim[1, order]
    non_zero = np.where(diff != 0)[0]
    
    p_ttb = np.array([0.5, 0.5])
    if len(non_zero) > 0:
        first_diff = diff[non_zero[0]]
        score_ttb = np.array([1.0, 0.0]) if first_diff > 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * score_ttb
        e_ttb = np.exp(z_ttb - np.max(z_ttb))
        p_ttb = e_ttb / e_ttb.sum()
            
    # Mixture of Tallying and TTB
    w_tally = float(parameters["w_tally"])
    p_mix = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    # Global lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_mix)
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
[0] rationale: By isolating trials where the Tallying heuristic predicts a strict tie (i.e., both options have an equal number of positive features), we can clearly separate the two theories. The pure Tallying model will predict a 50/50 guess on these trials, leading to a TTB-alignment proportion of ~0.5. In contrast, the Mixture model (which incorporates Take-The-Best) will systematically break these ties using the highest-validity discriminating feature, resulting in a TTB-alignment proportion significantly greater than 0.5. This metric computes the exact proportion of TTB-aligned choices exclusively on these tie trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Extract ratings as 2D arrays
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    # Identify tie trials (where Tallying predicts a strict tie)
    sum_a = A.sum(axis=1)
    sum_b = B.sum(axis=1)
    tie_mask = (sum_a == sum_b)
    
    if not np.any(tie_mask):
        return 0.5
        
    A_tie = A[tie_mask]
    B_tie = B[tie_mask]
    resp_tie = data['response'].values[tie_mask]
    
    # Find TTB choice for tie trials
    diff = A_tie - B_tie
    
    ttb_choices = []
    for d in diff:
        non_zero = np.where(d != 0)[0]
        if len(non_zero) > 0:
            first_diff = d[non_zero[0]]
            ttb_choices.append(0 if first_diff > 0 else 1)
        else:
            ttb_choices.append(-1)
            
    ttb_choices = np.array(ttb_choices)
    
    # Calculate proportion of TTB-aligned choices on tie trials
    valid_mask = (ttb_choices != -1)
    if not np.any(valid_mask):
        return 0.5
        
    aligned = (resp_tie[valid_mask] == ttb_choices[valid_mask])
    return float(np.mean(aligned))
outcome: self_sim=0.5125 (var=0.0073) adversary_sim=0.5206 (var=0.0119) welch_t=-0.293 p=0.7705 (N=25, alpha=0.01) -> reject

[1] rationale: By restricting the analysis to trials where the Tallying heuristic predicts an identical choice probability (score difference of exactly 1), we can elegantly dissociate the Pure Tallying model from the Mixture model. For Pure Tallying, the probability of choosing the Tally-favored option depends solely on the score difference; thus, the difference in choice rates between Agreement trials (where TTB also favors the Tally choice) and Disagreement trials (where TTB opposes it) should be exactly zero. Conversely, the Mixture model will exhibit a significantly positive difference, as the intermittent use of TTB will boost the Tally-aligned choice rate on Agreement trials and suppress it on Disagreement trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    A_sum = A.sum(axis=1)
    B_sum = B.sum(axis=1)
    
    # Tally choice is 0 if A_sum > B_sum else 1
    tally_choice = np.where(A_sum > B_sum, 0, 1)
    
    diff = A - B
    ttb_choice = np.zeros(len(A), dtype=int)
    for i in range(len(A)):
        non_zero = np.where(diff[i] != 0)[0]
        if len(non_zero) > 0:
            ttb_choice[i] = 0 if diff[i][non_zero[0]] > 0 else 1
        else:
            ttb_choice[i] = -1
            
    # Filter to trials where the Tallying score difference is exactly 1
    diff_1_mask = np.abs(A_sum - B_sum) == 1
    
    agree_mask = diff_1_mask & (tally_choice == ttb_choice)
    disagree_mask = diff_1_mask & (tally_choice != ttb_choice)
    
    if np.sum(agree_mask) == 0 or np.sum(disagree_mask) == 0:
        return 0.0
        
    p_agree = np.mean(resp[agree_mask] == tally_choice[agree_mask])
    p_disagree = np.mean(resp[disagree_mask] == tally_choice[disagree_mask])
    
    return float(p_agree - p_disagree)
outcome: self_sim=-0.0331 (var=0.0253) adversary_sim=0.0537 (var=0.0265) welch_t=-1.909 p=0.06226 (N=25, alpha=0.01) -> reject

[2] rationale: This metric leverages the opposing predictions of the two models across different levels of Tallying evidence. Pure Tallying predicts that the probability of choosing the Tally-favored option should be strictly higher when the score difference is 3 compared to when it is 1. Thus, P(Tally | diff=3) - P(Tally | diff=1, Agree) should be significantly positive. In contrast, the Mixture model will have a depressed P(Tally | diff=3) because TTB strictly opposes Tallying on these trials, while P(Tally | diff=1, Agree) is boosted by TTB agreement. Therefore, the Mixture model predicts this difference to be near zero or negative, creating a strong and stable contrast between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    A_sum = A.sum(axis=1)
    B_sum = B.sum(axis=1)
    
    tally_choice = np.where(A_sum > B_sum, 0, 1)
    
    diff = A - B
    ttb_choice = np.zeros(len(A), dtype=int)
    for i in range(len(A)):
        non_zero = np.where(diff[i] != 0)[0]
        if len(non_zero) > 0:
            ttb_choice[i] = 0 if diff[i][non_zero[0]] > 0 else 1
        else:
            ttb_choice[i] = -1
            
    diff_3_mask = np.abs(A_sum - B_sum) == 3
    diff_1_agree_mask = (np.abs(A_sum - B_sum) == 1) & (tally_choice == ttb_choice)
    
    if np.sum(diff_3_mask) == 0 or np.sum(diff_1_agree_mask) == 0:
        return 0.0
        
    p_diff_3 = np.mean(resp[diff_3_mask] == tally_choice[diff_3_mask])
    p_diff_1_agree = np.mean(resp[diff_1_agree_mask] == tally_choice[diff_1_agree_mask])
    
    return float(p_diff_3 - p_diff_1_agree)
outcome: self_sim=0.0906 (var=0.0273) adversary_sim=-0.0150 (var=0.0300) welch_t=+2.207 p=0.03212 (N=25, alpha=0.01) -> reject

[3] rationale: This metric combines two distinct choice contrasts to strongly separate the theories. First, it measures the difference in Tally-aligned choice rates between Agreement and Disagreement trials when the Tally score difference is 1 (p_agree_1 - p_dis_1). Second, it measures the difference between Agreement trials at diff=1 and Disagreement trials at diff=3 (p_agree_1 - p_dis_3). For the Advocated theory (Pure Tallying), the first contrast is exactly zero, and the second contrast is distinctly negative (since a score difference of 3 provides stronger evidence than a difference of 1). Thus, the overall metric will be negative. For the Competing theory (Mixture), intermittent use of Take-The-Best strongly boosts p_agree_1 and depresses both p_dis_1 and p_dis_3, driving both contrasts to be positive. This additive combination maximizes the mean separation between the two theories while maintaining stable variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    A_sum = A.sum(axis=1)
    B_sum = B.sum(axis=1)
    
    # Tally choice is 0 if A has more positive features, else 1
    tally_choice = np.where(A_sum > B_sum, 0, 1)
    
    # TTB choice based on the first discriminating feature
    diff = A - B
    ttb_choice = np.zeros(len(A), dtype=int)
    for i in range(len(A)):
        non_zero = np.where(diff[i] != 0)[0]
        if len(non_zero) > 0:
            ttb_choice[i] = 0 if diff[i][non_zero[0]] > 0 else 1
        else:
            ttb_choice[i] = -1
            
    # Identify trial types based on score differences and TTB agreement
    diff_1_mask = np.abs(A_sum - B_sum) == 1
    diff_3_mask = np.abs(A_sum - B_sum) == 3
    
    agree_1_mask = diff_1_mask & (tally_choice == ttb_choice)
    disagree_1_mask = diff_1_mask & (tally_choice != ttb_choice)
    disagree_3_mask = diff_3_mask & (tally_choice != ttb_choice)
    
    # Ensure we have data for all required trial types
    if np.sum(agree_1_mask) == 0 or np.sum(disagree_1_mask) == 0 or np.sum(disagree_3_mask) == 0:
        return 0.0
        
    # Calculate proportion of Tally-aligned choices for each trial type
    p_agree_1 = np.mean(resp[agree_1_mask] == tally_choice[agree_1_mask])
    p_dis_1 = np.mean(resp[disagree_1_mask] == tally_choice[disagree_1_mask])
    p_dis_3 = np.mean(resp[disagree_3_mask] == tally_choice[disagree_3_mask])
    
    # The metric combines two contrasts: (p_agree_1 - p_dis_1) + (p_agree_1 - p_dis_3)
    return float(2 * p_agree_1 - p_dis_1 - p_dis_3)

outcome: self_sim=-0.1238 (var=0.0732) adversary_sim=0.0687 (var=0.0854) welch_t=-2.417 p=0.01951 (N=25, alpha=0.01) -> reject

[4] rationale: Previous metrics failed because computing differences between multiple trial types inflated the within-subject binomial variance, burying the theoretical gap. To minimize variance while maximizing the contrast, this metric isolates Trial 5—the single 'Agreement' trial where Tallying and Take-The-Best (TTB) favor the same option, and the Tally score difference is minimal (diff=1). For the Advocated theory (Pure Tallying), the choice probability is strictly constrained by high stochasticity (epsilon) and the weak score difference, yielding an expected alignment of ~0.55. In contrast, the Competing theory (Mixture model) strongly boosts this probability (~0.71) because TTB synergistically agrees with Tallying, and the model permits much lower stochasticity. By evaluating this raw proportion without subtracting a baseline, we preserve a large mean gap while restricting noise to the sampling variance of a single trial type.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    A_sum = A.sum(axis=1)
    B_sum = B.sum(axis=1)
    
    # Trial 5 is the unique Agreement trial, identifiable by sum scores of 2 and 1
    mask_5_normal = (A_sum == 2) & (B_sum == 1)
    mask_5_flipped = (A_sum == 1) & (B_sum == 2)
    mask = mask_5_normal | mask_5_flipped
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    
    # Determine which option is favored by Tallying (and inherently TTB on this trial)
    tally_favored = np.where(A_sum[mask] > B_sum[mask], 0, 1)
    
    # Return the proportion of choices aligning with both heuristics
    return float(np.mean(responses == tally_favored))
outcome: self_sim=0.5575 (var=0.0159) adversary_sim=0.6462 (var=0.0221) welch_t=-2.275 p=0.02752 (N=25, alpha=0.01) -> reject

[5] rationale: To maximize the discriminative signal while suppressing binomial noise, this metric isolates trials where the TTB component of the Mixture model provides a pure positive boost without being perfectly offset by a strong Tallying disagreement. Specifically, it pools the Tie trials (T1, T2, T6) and the single Agreement trial (T5). On these trials, Pure Tallying predicts a baseline choice rate (exactly 50% for Ties, and ~55% for T5 due to high stochasticity), yielding an expected pooled TTB-alignment of ~51%. In contrast, the Mixture model's TTB component systematically boosts the TTB-aligned choice on all these trials, pulling the expected pooled alignment up to ~59%. By excluding Disagreement trials (T3, T4)—where the Mixture model's parameter distributions allow it to perfectly mimic Pure Tallying—and aggregating over 64 trials per subject, we create a large mean gap with tightly constrained variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    A_sum = A.sum(axis=1)
    B_sum = B.sum(axis=1)
    
    # Isolate Tie trials (T1, T2, T6; total features = 6) 
    # and the Agreement trial (T5; total features = 3)
    # We exclude Disagreement trials (T3, T4; total features = 5) where the Mixture model mimics Pure Tallying
    mask = (A_sum + B_sum == 6) | (A_sum + B_sum == 3)
    
    if not np.any(mask):
        return 0.5
        
    A_filtered = A[mask]
    B_filtered = B[mask]
    resp_filtered = resp[mask]
    
    # Find the TTB choice for the filtered trials
    diff = A_filtered - B_filtered
    ttb_choice = np.zeros(len(A_filtered), dtype=int)
    for i in range(len(A_filtered)):
        non_zero = np.where(diff[i] != 0)[0]
        if len(non_zero) > 0:
            ttb_choice[i] = 0 if diff[i][non_zero[0]] > 0 else 1
        else:
            ttb_choice[i] = -1
            
    valid = ttb_choice != -1
    if not np.any(valid):
        return 0.5
        
    aligned = (resp_filtered[valid] == ttb_choice[valid])
    return float(np.mean(aligned))
outcome: self_sim=0.5141 (var=0.0032) adversary_sim=0.5591 (var=0.0071) welch_t=-2.215 p=0.03226 (N=25, alpha=0.01) -> reject

[6] rationale: This metric isolates the two trial types that provide the starkest opposing predictions between the theories: T5 (Agreement, Tally diff=1) and T3 (Disagreement, Tally diff=3). For the Advocated theory (Pure Tallying), the choice probability is strictly determined by the sum difference. A difference of 3 provides much stronger evidence than a difference of 1, so the proportion of Tally-aligned choices MUST be higher on T3 than on T5, making (T5 - T3) distinctly negative. Conversely, for the Competing theory (Mixture model), the intermittent use of the Take-The-Best heuristic strongly boosts the Tally-aligned choice on T5 (where they agree) and heavily depresses it on T3 (where they disagree). This synergistic effect completely overrides the underlying Tally score difference, causing the Mixture model to predict a higher Tally-alignment on T5 than on T3, making (T5 - T3) distinctly positive. By calculating this paired within-subject difference, we also perfectly cancel out baseline subject-level variations in choice stochasticity, ensuring a massive mean gap with exceptionally tight variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    A_sum = A.sum(axis=1)
    B_sum = B.sum(axis=1)
    
    # Identify Tally choice (0 for A, 1 for B)
    tally_choice = np.where(A_sum > B_sum, 0, 1)
    
    # Identify TTB choice based on the first discriminating feature
    diff = A - B
    ttb_choice = np.zeros(len(A), dtype=int)
    for i in range(len(A)):
        non_zero = np.where(diff[i] != 0)[0]
        if len(non_zero) > 0:
            ttb_choice[i] = 0 if diff[i][non_zero[0]] > 0 else 1
        else:
            ttb_choice[i] = -1
            
    # T5: Agreement trial (Tally diff = 1, TTB agrees with Tally)
    # T3: Disagreement trial (Tally diff = 3, TTB disagrees with Tally)
    mask_T5 = (np.abs(A_sum - B_sum) == 1) & (tally_choice == ttb_choice)
    mask_T3 = (np.abs(A_sum - B_sum) == 3) & (tally_choice != ttb_choice)
    
    if np.sum(mask_T5) == 0 or np.sum(mask_T3) == 0:
        return 0.0
        
    p_tally_T5 = np.mean(resp[mask_T5] == tally_choice[mask_T5])
    p_tally_T3 = np.mean(resp[mask_T3] == tally_choice[mask_T3])
    
    # The metric contrasts the Tally-alignment on T5 vs T3.
    # Pure Tallying predicts T3 > T5 (since score diff 3 > score diff 1).
    # Mixture model predicts T5 > T3 (since TTB boosts T5 and depresses T3).
    return float(p_tally_T5 - p_tally_T3)

outcome: self_sim=-0.0637 (var=0.0357) adversary_sim=0.0487 (var=0.0344) welch_t=-2.124 p=0.03883 (N=25, alpha=0.01) -> reject

[7] rationale: By filtering the dataset to a subset of trials that is perfectly symmetric with respect to the Pure Tallying model (D_TTB ∈ {-1, 0, +1}), we completely eliminate the influence of the Tallying parameters (beta and epsilon). For Pure Tallying, the expected proportion of TTB-aligned choices on the D_TTB=0 trials is exactly 0.5. Furthermore, because the D_TTB=+1 and D_TTB=-1 trials occur with equal frequency, their Tallying choice probabilities exactly average to 0.5. Thus, the Advocated theory (Pure Tallying) predicts a metric value of EXACTLY 0.5, with between-subject variance restricted solely to binomial sampling noise. In stark contrast, the Competing theory (Mixture model) includes a Take-The-Best component that systematically biases choices toward the TTB-favored option on ALL of these filtered trials, driving the expected value strictly and significantly above 0.5. This creates a robust mean gap combined with the lowest possible theoretical variance for the null model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    A_sum = A.sum(axis=1)
    B_sum = B.sum(axis=1)
    
    # The first cue (index 0) has the highest validity (0.95) and perfectly discriminates 
    # between A and B on every single trial in this design. 
    # Therefore, Take-The-Best (TTB) will ALWAYS favor the option where cue 0 == 1.
    ttb_favors_A = A[:, 0] == 1
    
    # Calculate the Tallying score difference from the perspective of the TTB-favored option.
    sum_ttb_favored = np.where(ttb_favors_A, A_sum, B_sum)
    sum_ttb_disfavored = np.where(ttb_favors_A, B_sum, A_sum)
    d_ttb = sum_ttb_favored - sum_ttb_disfavored
    
    # We isolate trials where the Tallying score difference (d_ttb) is perfectly symmetric around 0.
    # This includes:
    # - T1, T2, T6: Tie trials (d_ttb = 0)
    # - T5: Agreement trial (d_ttb = +1)
    # - T4: Disagreement trial (d_ttb = -1)
    # We explicitly exclude T3 (d_ttb = -3) because there is no +3 trial to balance it.
    mask = (d_ttb == 0) | (d_ttb == 1) | (d_ttb == -1)
    
    if not np.any(mask):
        return 0.5
        
    resp_filtered = resp[mask]
    ttb_favors_A_filtered = ttb_favors_A[mask]
    
    # Determine if the subject chose the TTB-favored option (0 means A, 1 means B)
    chose_A = (resp_filtered == 0)
    chose_ttb = (chose_A == ttb_favors_A_filtered)
    
    return float(np.mean(chose_ttb))

outcome: self_sim=0.4958 (var=0.0028) adversary_sim=0.5305 (var=0.0054) welch_t=-1.922 p=0.06115 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    A = np.stack(data['option_a_ratings'].values)\n    B = np.stack(data['option_b_ratings'].values)\n    resp = data['response'].values\n    \n    A_sum = A.sum(axis=1)\n    B_sum = B.sum(axis=1)\n    A_0 = A[:, 0]\n    \n    # Identify specific trials by their Tallying sums and Cue 0 values\n    # T3: A sum=4, B sum=2, TTB favors B (A_0=0)\n    mask_T3 = (A_sum == 4) & (B_sum == 2) & (A_0 == 0)\n    # T4: A sum=4, B sum=2, TTB favors A (A_0=1)\n    mask_T4 = (A_sum == 4) & (B_sum == 2) & (A_0 == 1)\n    # T5: A sum=2, B sum=3, TTB favors A (A_0=1)\n    mask_T5 = (A_sum == 2) & (B_sum == 3) & (A_0 == 1)\n    # T6: A sum=2, B sum=3, TTB favors B (A_0=0)\n    mask_T6 = (A_sum == 2) & (B_sum == 3) & (A_0 == 0)\n    \n    # Compute the proportion of choices aligning with the Tallying heuristic\n    p_A_T3 = np.mean(resp[mask_T3] == 0) if np.sum(mask_T3) > 0 else 0.5\n    p_A_T4 = np.mean(resp[mask_T4] == 0) if np.sum(mask_T4) > 0 else 0.5\n    p_B_T5 = np.mean(resp[mask_T5] == 1) if np.sum(mask_T5) > 0 else 0.5\n    p_B_T6 = np.mean(resp[mask_T6] == 1) if np.sum(mask_T6) > 0 else 0.5\n    \n    # Metric: Difference in Tally-alignment between Agreement and Disagreement trials\n    # For Tallying, this expectation is exactly 0.\n    # For Mixture, TTB consistently boosts agreement trials (T4, T6) and depresses disagreement (T3, T5), making it strictly positive.\n    return float((p_A_T4 - p_A_T3) + (p_B_T6 - p_B_T5))\n",
  "rationale": "This metric perfectly isolates the effect of Take-The-Best (TTB) while controlling for the Pure Tallying model's predictions. By comparing pairs of trials that have identical Tallying score differences but opposing TTB directions, we completely eliminate the influence of the Tallying parameters (beta and epsilon). For the Advocated theory (Pure Tallying), the expected difference between these matched trials is exactly 0. In contrast, the Competing theory (Mixture model) will systematically exhibit a positive difference, as the intermittent use of TTB boosts Tally-aligned choices on Agreement trials (T4, T6) and suppresses them on Disagreement trials (T3, T5). Combining both diff=2 and diff=1 pairs maximizes the mean gap while maintaining a theoretical expected value of 0 for the Pure Tallying model."
}
```

## Usage

```json
{
  "prompt_token_count": 9471,
  "candidates_token_count": 860,
  "total_token_count": 14105
}
```
