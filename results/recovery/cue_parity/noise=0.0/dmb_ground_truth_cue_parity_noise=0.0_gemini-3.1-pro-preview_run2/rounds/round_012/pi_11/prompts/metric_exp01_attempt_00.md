# metric_exp01_attempt_00

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
**Validities (n_features=8):** [0.98, 0.92, 0.86, 0.8, 0.74, 0.68, 0.62, 0.56]

**Trial pairs (n=15):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 6: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  trial 8: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 9: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  trial 10: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  trial 11: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  trial 12: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 1]
  trial 13: A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 14: A=[1, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]

**Rationale:** This design quantitatively and qualitatively dissociates the advocated Weighted Additive (WADD) theory from the competing Sequential Evidence Accumulation (SEA) theory by probing for a counter-intuitive 'Less is More' anomaly caused by early stopping. SEA posits that evidence is accumulated sequentially and search terminates as soon as a threshold is crossed. Consequently, a single highly valid cue can trigger premature stopping, yielding a final evidence score just above the threshold. If that top cue is removed (made a tie), the model is forced to continue accumulating multiple weaker cues, which can collectively overshoot the threshold by a much larger margin. This paradoxically predicts higher choice confidence when the best supportive cue is missing. WADD, as an exhaustive compensatory model, strictly predicts that removing supportive cues must monotonically decrease choice confidence. By systematically tying the top cues while leaving multiple lower-validity cues in favor of Option A, we cleanly test for this non-monotonic inversion.

**Computed schedule:** 15 unique pairs × 6 reps = 90 trials per subject.



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
[0] rationale: This metric calculates the probability of a subject choosing the 'compensatory' option (the one featuring multiple intermediate-validity cues like 1, 2, and 3) over the 'non-compensatory' option (the one featuring the single highest-validity cue 0). WADD, which always integrates all cues via a strictly convex transformation, is likely to favor the compensatory option more often than Sequential Evidence Accumulation (SCA), which features early stopping thresholds that strongly bias it towards simply choosing the option with the highest-validity cue.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def is_compensatory(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        # Check if option A has the best cue (0) but B has the next three (1, 2, 3)
        if a[0] == 1 and b[0] == 0 and b[1] == 1 and b[2] == 1 and b[3] == 1:
            return 1.0 if row['response'] == 1 else 0.0
        # Check if option B has the best cue (0) but A has the next three (1, 2, 3)
        elif b[0] == 1 and a[0] == 0 and a[1] == 1 and a[2] == 1 and a[3] == 1:
            return 1.0 if row['response'] == 0 else 0.0
        return float('nan')
        
    vals = data.apply(is_compensatory, axis=1).dropna()
    return float(vals.mean()) if len(vals) > 0 else 0.5
outcome: self_sim=0.6170 (var=0.1295) adversary_sim=0.5652 (var=0.0431) welch_t=+0.624 p=0.5363 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in preference for the 'extreme' option between Trial 2 (cues 0,4 vs 1,3) and Trial 3 (cues 1,4 vs 2,3). Because WADD uses a strictly convex exponential validity transformation, the advantage of the extreme option is dramatically magnified in Trial 2 relative to Trial 3. By contrast, Sequential Evidence Accumulation either stops early (producing similarly high choices for the extreme option in both trials) or integrates using a linear/concave function, which mutes the difference in preference between these two trials. Therefore, WADD predicts a significantly larger positive gap (P(Extreme|T2) - P(Extreme|T3)) than SCA.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 2: Extreme (0,4) vs Intermediate (1,3)
    is_t2_A = (a_tuples == (1, 0, 0, 0, 1)) & (b_tuples == (0, 1, 0, 1, 0))
    is_t2_B = (b_tuples == (1, 0, 0, 0, 1)) & (a_tuples == (0, 1, 0, 1, 0))
    
    # Trial 3: Extreme (1,4) vs Intermediate (2,3)
    is_t3_A = (a_tuples == (0, 1, 0, 0, 1)) & (b_tuples == (0, 0, 1, 1, 0))
    is_t3_B = (b_tuples == (0, 1, 0, 0, 1)) & (a_tuples == (0, 0, 1, 1, 0))
    
    t2_extreme_chosen = []
    t3_extreme_chosen = []
    
    for idx, row in data.iterrows():
        if is_t2_A[idx]:
            t2_extreme_chosen.append(1.0 if row['response'] == 0 else 0.0)
        elif is_t2_B[idx]:
            t2_extreme_chosen.append(1.0 if row['response'] == 1 else 0.0)
            
        if is_t3_A[idx]:
            t3_extreme_chosen.append(1.0 if row['response'] == 0 else 0.0)
        elif is_t3_B[idx]:
            t3_extreme_chosen.append(1.0 if row['response'] == 1 else 0.0)
            
    t2_rate = np.mean(t2_extreme_chosen) if t2_extreme_chosen else 0.5
    t3_rate = np.mean(t3_extreme_chosen) if t3_extreme_chosen else 0.5
    
    return float(t2_rate - t3_rate)
outcome: self_sim=-0.0022 (var=0.0378) adversary_sim=0.0022 (var=0.0368) welch_t=-0.081 p=0.9355 (N=25, alpha=0.01) -> reject

[2] rationale: This metric calculates the difference in preference for the option with cues {0,3} when pitted against {1,2} (Trial 1) versus when pitted against {1,2,4} (Trial 5). Trial 5 is identical to Trial 1 except the alternative option gains the lowest-validity cue (cue 4). WADD integrates all cues, so the addition of cue 4 strictly increases the value of the alternative, reliably dropping the probability of choosing {0,3}. Sequential Evidence Accumulation (SCA), however, evaluates cues in descending order of validity and utilizes early stopping. Cue 4 is evaluated last and carries minimal weight (val-0.5 is near zero); thus, SCA will almost always terminate search before reaching cue 4, or its impact will be negligible. Consequently, WADD predicts a significantly larger positive difference (P({0,3} in T1) - P({0,3} in T5)) than SCA.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: {0,3} vs {1,2} 
    is_t1_A = (a_tuples == (1, 0, 0, 1, 0)) & (b_tuples == (0, 1, 1, 0, 0))
    is_t1_B = (b_tuples == (1, 0, 0, 1, 0)) & (a_tuples == (0, 1, 1, 0, 0))
    
    # Trial 5: {1,2,4} vs {0,3}
    is_t5_A = (a_tuples == (0, 1, 1, 0, 1)) & (b_tuples == (1, 0, 0, 1, 0))
    is_t5_B = (b_tuples == (0, 1, 1, 0, 1)) & (a_tuples == (1, 0, 0, 1, 0))
    
    t1_03_chosen = []
    t5_03_chosen = []
    
    for idx, row in data.iterrows():
        if is_t1_A[idx]:
            t1_03_chosen.append(1.0 if row['response'] == 0 else 0.0)
        elif is_t1_B[idx]:
            t1_03_chosen.append(1.0 if row['response'] == 1 else 0.0)
            
        if is_t5_A[idx]:
            t5_03_chosen.append(1.0 if row['response'] == 1 else 0.0)
        elif is_t5_B[idx]:
            t5_03_chosen.append(1.0 if row['response'] == 0 else 0.0)
            
    p_t1 = sum(t1_03_chosen) / len(t1_03_chosen) if t1_03_chosen else 0.5
    p_t5 = sum(t5_03_chosen) / len(t5_03_chosen) if t5_03_chosen else 0.5
    
    return float(p_t1 - p_t5)
outcome: self_sim=0.2711 (var=0.2001) adversary_sim=0.0667 (var=0.0509) welch_t=+2.041 p=0.04879 (N=25, alpha=0.01) -> reject

[3] rationale: This metric calculates the difference in choice probability for the compensatory option in Trial 7 ({1,2,3,4} vs {0}) compared to Trials 4 and 8 ({1,2,3} vs {0,4}). The only difference between these choice sets is the allocation of the lowest-validity cue (cue 4). In WADD, cue 4's weight is strictly bounded away from zero (exp(0.55 * gamma) - 1), so shifting cue 4 from the non-compensatory option to the compensatory option substantially increases the probability of choosing the compensatory option. In Sequential Evidence Accumulation (SCA), however, cue 4 is evaluated last and has a near-zero subjective weight ((0.55 - 0.5)^gamma = 0.05^gamma). Furthermore, SCA's early stopping probability at cue 0 is identical across both choice sets. Consequently, SCA predicts virtually no difference in choice probabilities between these trials, whereas WADD predicts a robust positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 7: {0} vs {1,2,3,4} -> Compensatory option is {1,2,3,4}
    is_t7_A = (a_tuples == (1, 0, 0, 0, 0)) & (b_tuples == (0, 1, 1, 1, 1))
    is_t7_B = (b_tuples == (1, 0, 0, 0, 0)) & (a_tuples == (0, 1, 1, 1, 1))
    
    # Trial 4: {1,2,3} vs {0,4} -> Compensatory option is {1,2,3}
    is_t4_A = (a_tuples == (0, 1, 1, 1, 0)) & (b_tuples == (1, 0, 0, 0, 1))
    is_t4_B = (b_tuples == (0, 1, 1, 1, 0)) & (a_tuples == (1, 0, 0, 0, 1))
    
    # Trial 8: {0,4} vs {1,2,3} -> Compensatory option is {1,2,3}
    is_t8_A = (a_tuples == (1, 0, 0, 0, 1)) & (b_tuples == (0, 1, 1, 1, 0))
    is_t8_B = (b_tuples == (1, 0, 0, 0, 1)) & (a_tuples == (0, 1, 1, 1, 0))
    
    comp_t7 = []
    comp_t4_8 = []
    
    for idx, row in data.iterrows():
        if is_t7_A[idx]:
            comp_t7.append(1.0 if row['response'] == 1 else 0.0)
        elif is_t7_B[idx]:
            comp_t7.append(1.0 if row['response'] == 0 else 0.0)
            
        if is_t4_A[idx]:
            comp_t4_8.append(1.0 if row['response'] == 0 else 0.0)
        elif is_t4_B[idx]:
            comp_t4_8.append(1.0 if row['response'] == 1 else 0.0)
            
        if is_t8_A[idx]:
            comp_t4_8.append(1.0 if row['response'] == 1 else 0.0)
        elif is_t8_B[idx]:
            comp_t4_8.append(1.0 if row['response'] == 0 else 0.0)
            
    p_comp_t7 = sum(comp_t7) / len(comp_t7) if comp_t7 else 0.5
    p_comp_t4_8 = sum(comp_t4_8) / len(comp_t4_8) if comp_t4_8 else 0.5
    
    return float(p_comp_t7 - p_comp_t4_8)

outcome: self_sim=0.0311 (var=0.0306) adversary_sim=-0.0078 (var=0.0496) welch_t=+0.686 p=0.496 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_tuples = data['option_a_ratings'].apply(tuple)\n    b_tuples = data['option_b_ratings'].apply(tuple)\n    \n    # Trial 1: Top cue is untied\n    is_t1 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (b_tuples == (0, 0, 0, 0, 0, 0, 0, 0))\n    is_t1_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (a_tuples == (0, 0, 0, 0, 0, 0, 0, 0))\n    \n    # Trials 5 and 6: Top cues are tied, forcing accumulation of weaker cues\n    is_t5 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (b_tuples == (1, 1, 1, 1, 0, 0, 0, 0))\n    is_t5_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (a_tuples == (1, 1, 1, 1, 0, 0, 0, 0))\n    is_t6 = (a_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (b_tuples == (1, 1, 1, 1, 1, 0, 0, 0))\n    is_t6_rev = (b_tuples == (1, 1, 1, 1, 1, 1, 1, 1)) & (a_tuples == (1, 1, 1, 1, 1, 0, 0, 0))\n    \n    p_A_t1 = []\n    p_A_late = []\n    \n    for idx, row in data.iterrows():\n        if is_t1[idx]:\n            p_A_t1.append(1.0 if row['response'] == 0 else 0.0)\n        elif is_t1_rev[idx]:\n            p_A_t1.append(1.0 if row['response'] == 1 else 0.0)\n            \n        if is_t5[idx] or is_t6[idx]:\n            p_A_late.append(1.0 if row['response'] == 0 else 0.0)\n        elif is_t5_rev[idx] or is_t6_rev[idx]:\n            p_A_late.append(1.0 if row['response'] == 1 else 0.0)\n            \n    m_t1 = sum(p_A_t1) / len(p_A_t1) if p_A_t1 else 0.5\n    m_late = sum(p_A_late) / len(p_A_late) if p_A_late else 0.5\n    \n    return float(m_late - m_t1)",
  "rationale": "This metric directly tests for the 'Less is More' anomaly predicted by Sequential Evidence Accumulation (SEA). We compare the probability of choosing the dominating option (all 1s) when the alternative has all 0s (Trial 1) versus when the alternative has 1s on the top 4 or 5 cues (Trials 5 and 6). WADD, being exhaustively compensatory, strictly predicts that giving the alternative more 1s will monotonically decrease the probability of choosing the dominating option (metric < 0). SEA, however, predicts that in Trial 1, search stops immediately at the first cue, yielding a moderate accumulated evidence. In Trials 5 and 6, tying the top cues forces SEA to accumulate evidence from multiple weaker cues, which together can sum to a much larger final tally than the single top cue, paradoxically increasing the probability of choosing the dominating option (metric > 0)."
}
```

## Usage

```json
{
  "prompt_token_count": 7164,
  "candidates_token_count": 1005,
  "total_token_count": 11806
}
```
