# metric_exp00_attempt_02

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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Rationale:** This design quantitatively dissociates the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (Top-K Majority Heuristic) by exploiting the structural constraints of unweighted running tallies versus fixed-set majority voting. In 'Trap' trials (e.g., Option A wins features 1, 4, 5; Option B wins features 2, 3), the running tally of differences fluctuates (+1, 0, -1, 0, +1). Because the absolute tally never exceeds 1 and ends at +1, the Advocated Theory structurally MUST predict Option A regardless of its stopping threshold parameter (it either stops at feature 1 and chooses A, or tallies to the end and chooses A). However, the Competing Theory evaluates fixed subsets (e.g., Top K=3). Within the top 3 features, Option B holds a strict majority (winning features 2 and 3 versus A's feature 1), allowing the Competing Theory to robustly predict Option B. By varying the winning features across trials, we create a sharp double dissociation where each model can be forced into a prediction that is mathematically impossible for the other.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



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
**Description:** Top-K Majority Heuristic with Proportional Confidence: Decision-makers evaluate options by considering a subset of the most valid features. They identify the top K most valid features and perform an unweighted tally within this set. If tied, they may expand the set. Their confidence in the choice scales with the proportional majority margin (vote difference divided by K), meaning a given vote margin yields higher confidence in smaller consideration sets.

**Parameters:**
- k_initial: [1.0, 10.0]
- expand_on_tie: [0.0, 1.0]
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
    
    # Determine initial consideration set size K
    k_init = int(round(float(parameters["k_initial"]))) # e.g., 3 or 5
    k = min(k_init, len(val))
    k = max(1, k)
    
    expand_on_tie = float(parameters["expand_on_tie"]) > 0.5
    
    while True:
        top_k_idx = order[:k]
        a_wins = np.sum(a[top_k_idx] > b[top_k_idx])
        b_wins = np.sum(b[top_k_idx] > a[top_k_idx])
        
        # Stop if there's a strict majority winner within top K
        if a_wins != b_wins:
            break
            
        # If tied, either expand K or accept the tie (and guess)
        if expand_on_tie and k < len(val):
            k += 1
        else:
            break
            
    diff = float(a_wins - b_wins) / k
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the vote difference
    scores = np.array([diff, -diff])
    z = beta * scores
    z -= np.max(z)  # numerical stability
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
[0] rationale: This metric calculates the difference in the rate of choosing the option that wins on the most valid feature (feature 1) between 'Control' trials (trials 5 & 6) and 'Trap' trials (trials 1 & 2). In Control trials, both Sequential Accumulation and Top-K Majority generally agree on the feature 1 winner. In Trap trials, Sequential Accumulation is mathematically forced to choose the feature 1 winner (the running tally never exceeds 1 and ends at +1), whereas Top-K Majority with K=3 or K=4 will reliably choose the opposite option because the opposing features outnumber the most valid feature in that consideration set. Thus, Sequential Accumulation predicts a difference near zero, while Top-K Majority predicts a significantly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = a_str == '10011'
    t2_mask = a_str == '01100'
    t5_mask = a_str == '10101'
    t6_mask = a_str == '01010'
    
    trap_f1_wins = ((t1_mask & (data['response'] == 0)) | (t2_mask & (data['response'] == 1))).sum()
    trap_total = (t1_mask | t2_mask).sum()
    
    control_f1_wins = ((t5_mask & (data['response'] == 0)) | (t6_mask & (data['response'] == 1))).sum()
    control_total = (t5_mask | t6_mask).sum()
    
    if trap_total == 0 or control_total == 0:
        return 0.0
        
    return float((control_f1_wins / control_total) - (trap_f1_wins / trap_total))
outcome: self_sim=0.0094 (var=0.0065) adversary_sim=0.0694 (var=0.0395) welch_t=-1.399 p=0.1716 (N=25, alpha=0.01) -> reject

[1] rationale: Measures the difference in the rate of choosing the option that wins Feature 1 between Trials 3/4 and Trials 1/2. For Top-K Majority (e.g., K=3), the model strongly prefers the Feature 1 winner in Trials 3/4 (where it wins 2 out of the top 3 features) but opposes it in Trials 1/2 (where it only wins 1 out of the top 3), yielding a positive difference. Conversely, Sequential Evidence Accumulation ALWAYS chooses the Feature 1 winner in Trials 1/2 (because the tally never swings negative before the end, where it is positive), but can choose the Feature 1 loser in Trials 3/4 if the threshold is high (since the final tally is negative). Thus, Sequential Accumulation predicts a zero or negative difference, creating a sharp quantitative contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = a_str == '10011'
    t2_mask = a_str == '01100'
    t3_mask = a_str == '11000'
    t4_mask = a_str == '00111'
    
    t12_f1_wins = ((t1_mask & (data['response'] == 0)) | (t2_mask & (data['response'] == 1))).sum()
    t12_total = (t1_mask | t2_mask).sum()
    
    t34_f1_wins = ((t3_mask & (data['response'] == 0)) | (t4_mask & (data['response'] == 1))).sum()
    t34_total = (t3_mask | t4_mask).sum()
    
    if t12_total == 0 or t34_total == 0:
        return 0.0
        
    return float((t34_f1_wins / t34_total) - (t12_f1_wins / t12_total))
outcome: self_sim=-0.2681 (var=0.1597) adversary_sim=-0.3881 (var=0.2132) welch_t=+0.983 p=0.3309 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    t1_mask = a_str == '10011'\n    t2_mask = a_str == '01100'\n    t3_mask = a_str == '11000'\n    t4_mask = a_str == '00111'\n    \n    # Option A wins Feature 1 in T1 and T3; Option B wins Feature 1 in T2 and T4\n    f1_a_wins = ((t1_mask | t3_mask) & (data['response'] == 0)).sum()\n    f1_b_wins = ((t2_mask | t4_mask) & (data['response'] == 1)).sum()\n    \n    total = (t1_mask | t2_mask | t3_mask | t4_mask).sum()\n    if total == 0:\n        return 0.0\n        \n    return float((f1_a_wins + f1_b_wins) / total)",
  "rationale": "The metric computes the overall rate of choosing the option that is superior on the most valid feature (Feature 1) across Trials 1, 2, 3, and 4. For Sequential Evidence Accumulation, the model always chooses the Feature 1 winner in Trials 1/2, and does so in Trials 3/4 if the stopping threshold is low to moderate, leading to a high overall average rate (~83%). For the Top-K Majority Heuristic, the model frequently chooses the Feature 1 loser: in Trials 1/2 when K=3, and in Trials 3/4 when K>=4. Because K>=4 is highly probable given the uniform prior on k_initial, Top-K frequently opposes the Feature 1 winner, leading to a much lower overall rate (~55%). Pooling these trials significantly reduces between-subject variance compared to taking differences, yielding a highly stable and statistically significant contrast."
}
```

## Usage

```json
{
  "prompt_token_count": 4552,
  "candidates_token_count": 491,
  "total_token_count": 12105
}
```
